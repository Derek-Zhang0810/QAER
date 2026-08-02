import argparse
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import ExperimentConfig, load_config, save_config
from data_loader import TemporalReasoningDataset, collate_temporal_batch
from model import SingleStepTemporalReasoner
from score_calibration import fit_score_calibration, save_score_calibration
from semantic_encoder import build_text_features
from utils import aggregate_ranking_metrics, ensure_dir, filtered_rank, resolve_device, save_json, set_seed


def _amp_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def _amp_autocast(enabled: bool, dtype: torch.dtype):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled, dtype=dtype)
    return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _enable_cuda_fast_math(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _loader_kwargs(config: ExperimentConfig, device: torch.device, shuffle: bool, batch_size: int) -> Dict:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": config.train.num_workers,
        "collate_fn": collate_temporal_batch,
        "pin_memory": config.train.pin_memory and device.type == "cuda",
    }
    if config.train.num_workers > 0:
        kwargs["persistent_workers"] = config.train.persistent_workers
        kwargs["prefetch_factor"] = config.train.prefetch_factor
    return kwargs


def evaluate(model, data_loader, filter_map, device, use_amp: bool = False):
    model.eval()
    original_eval_path_topk = getattr(model, "eval_path_topk", 0)
    model.eval_path_topk = 0
    ranks = []
    try:
        with torch.inference_mode():
            for batch in tqdm(data_loader, desc="validate", leave=False):
                with _amp_autocast(enabled=use_amp, dtype=_amp_dtype()):
                    output = model(batch, return_explanations=False)
                scores = output["scores"]
                targets = batch["targets"]
                subjects = batch["subjects"]
                relations = batch["relations"]
                times = batch["times"]
                for row in range(scores.size(0)):
                    key = (int(subjects[row]), int(relations[row]), int(times[row]))
                    rank = filtered_rank(scores[row], int(targets[row]), filter_map.get(key, set()))
                    ranks.append(rank)
    finally:
        model.eval_path_topk = original_eval_path_topk
    return aggregate_ranking_metrics(ranks)


def _init_ema_state(model) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and torch.is_floating_point(parameter)
    }


def _update_ema_state(model, ema_state: Dict[str, torch.Tensor], decay: float) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in ema_state:
                continue
            ema_state[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)


def _apply_ema_state(model, ema_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    backup = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in ema_state:
                continue
            backup[name] = parameter.detach().clone()
            parameter.copy_(ema_state[name].to(device=parameter.device, dtype=parameter.dtype))
    return backup


def _restore_parameter_state(model, backup: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in backup:
                parameter.copy_(backup[name].to(device=parameter.device, dtype=parameter.dtype))


def _warmup_factor(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, epoch / warmup_epochs)))


def _apply_training_schedule(
        config: ExperimentConfig,
        model: SingleStepTemporalReasoner,
        train_dataset: TemporalReasoningDataset,
        epoch: int,
        base_schedule: Dict[str, float],
) -> Dict[str, float]:
    train_dataset.set_epoch(epoch)
    heavy_factor = _warmup_factor(epoch, config.train.heavy_evidence_warmup_epochs)
    heavy_start = float(config.train.heavy_evidence_start_rate)
    heavy_end = float(base_schedule["heavy_evidence_rate"])
    heavy_rate = heavy_start + (heavy_end - heavy_start) * heavy_factor
    train_dataset.data_config.train_heavy_evidence_rate = max(0.0, min(1.0, heavy_rate))

    context_factor = _warmup_factor(epoch, config.train.context_warmup_epochs)
    context_start = int(config.train.context_start_limit)
    context_end = int(base_schedule["train_context_batch_limit"])
    model.train_context_batch_limit = max(
        0,
        int(round(context_start + (context_end - context_start) * context_factor)),
    )

    evidence_factor = _warmup_factor(epoch, config.train.evidence_warmup_epochs)
    route_factor = _warmup_factor(epoch, config.train.route_warmup_epochs)
    model.evidence_loss_weight = float(base_schedule["evidence_loss_weight"]) * evidence_factor
    model.route_supervision_weight = float(base_schedule["route_supervision_weight"]) * route_factor

    return {
        "heavy_evidence_rate": float(train_dataset.data_config.train_heavy_evidence_rate),
        "train_context_batch_limit": float(model.train_context_batch_limit),
        "evidence_loss_weight": float(model.evidence_loss_weight),
        "route_supervision_weight": float(model.route_supervision_weight),
    }


def train_model(config: ExperimentConfig) -> Dict[str, str]:
    output_dir = ensure_dir(config.output_dir)
    save_config(config, str(output_dir / "resolved_config.json"))
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)
    _enable_cuda_fast_math(device)
    print(
        {
            "stage": "train_start",
            "output_dir": str(output_dir),
            "dataset_dir": config.data.dataset_dir,
            "device": str(device),
            "epochs": int(config.train.epochs),
        },
        flush=True,
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    print({"stage": "dataset_build_start", "split": "train"}, flush=True)
    train_dataset = TemporalReasoningDataset(deepcopy(config.data), split="train")
    print({"stage": "dataset_build_start", "split": "valid"}, flush=True)
    valid_dataset = TemporalReasoningDataset(deepcopy(config.data), split="valid")
    print(
        {
            "stage": "dataset_build_done",
            "train_samples": len(train_dataset),
            "valid_samples": len(valid_dataset),
        },
        flush=True,
    )
    cache_start = time.perf_counter()
    should_warm_prior_cache = bool(getattr(config.train, "warm_prior_cache", True))
    print(
        {
            "stage": "prior_cache_warmup_config",
            "enabled": should_warm_prior_cache,
            "num_workers": int(config.train.num_workers),
        },
        flush=True,
    )
    train_prior_cache_size = (
        train_dataset.warm_prior_cache()
        if should_warm_prior_cache and config.train.num_workers == 0 else 0
    )
    valid_prior_cache_size = (
        valid_dataset.warm_prior_cache()
        if should_warm_prior_cache and config.train.num_workers == 0 else 0
    )
    if train_prior_cache_size or valid_prior_cache_size:
        print(
            {
                "stage": "prior_cache_warmup",
                "train_times": train_prior_cache_size,
                "valid_times": valid_prior_cache_size,
                "seconds": time.perf_counter() - cache_start,
            },
            flush=True,
        )
    elif not should_warm_prior_cache:
        print({"stage": "prior_cache_warmup_disabled", "seconds": time.perf_counter() - cache_start}, flush=True)
    print(
        {
            "stage": "train_setup",
            "method": config.method_name,
            "dataset_dir": config.data.dataset_dir,
            "device": str(device),
            "train_samples": len(train_dataset),
            "train_base_facts": len(train_dataset.data),
            "inverse_train": config.data.use_inverse_train,
            "inverse_train_mode": config.data.inverse_train_mode,
            "valid_samples": len(valid_dataset),
            "num_entities": train_dataset.num_entities,
            "num_relations": train_dataset.num_relations,
            "semantic_backend": config.semantic.backend,
            "semantic_model": config.semantic.transformer_name,
            "history_window": config.data.history_window,
            "train_heavy_evidence_rate": config.data.train_heavy_evidence_rate,
            "dynamic_heavy_evidence": config.data.dynamic_heavy_evidence,
            "strict_eval_history": config.data.strict_eval_history,
            "local_graph_cache_size": config.data.local_graph_cache_size,
            "prior_window": config.data.prior_window,
            "long_prior_window": config.data.long_prior_window,
            "long_prior_mix": config.data.long_prior_mix,
            "path_prior_topk": config.data.path_prior_topk,
            "path_prior_weight": config.model.path_prior_weight,
            "distmult_score_weight": config.model.distmult_score_weight,
            "transe_score_weight": config.model.transe_score_weight,
            "conv_score_weight": config.model.conv_score_weight,
            "relation_transfer_topk": config.data.relation_transfer_topk,
            "relation_transfer_min_sim": config.data.relation_transfer_min_sim,
            "relation_transfer_prior_weight": config.model.relation_transfer_prior_weight,
            "relation_transfer_copy_weight": config.model.relation_transfer_copy_weight,
            "relation_transfer_copy_temperature": config.model.relation_transfer_copy_temperature,
            "relation_transfer_copy_history_len": config.model.relation_transfer_copy_history_len,
            "concurrent_subject_topk": config.data.concurrent_subject_topk,
            "concurrent_time_topk": config.data.concurrent_time_topk,
            "use_concurrent_query_context": config.model.use_concurrent_query_context,
            "concurrent_query_weight": config.model.concurrent_query_weight,
            "concurrent_query_temperature": config.model.concurrent_query_temperature,
            "evidence_candidate_gating": config.model.evidence_candidate_gating,
            "evidence_gate_sources": config.model.evidence_gate_sources,
            "evidence_gate_mode": config.model.evidence_gate_mode,
            "evidence_gate_penalty": config.model.evidence_gate_penalty,
            "evidence_gate_boost": config.model.evidence_gate_boost,
            "evidence_gate_band_width": config.model.evidence_gate_band_width,
            "evidence_gate_min_sources": config.model.evidence_gate_min_sources,
            "evidence_gate_support_threshold": config.model.evidence_gate_support_threshold,
            "evidence_gate_source_weights": config.model.evidence_gate_source_weights,
            "evidence_gate_neural_topk": config.model.evidence_gate_neural_topk,
            "evidence_rank_fusion_weight": config.model.evidence_rank_fusion_weight,
            "evidence_rank_fusion_k": config.model.evidence_rank_fusion_k,
            "learnable_evidence_scales": True,
            "coverage_route_weight": config.model.coverage_route_weight,
            "relation_evidence_weight": config.model.relation_evidence_weight,
            "evidence_consensus_weight": config.model.evidence_consensus_weight,
            "evidence_consensus_min_sources": config.model.evidence_consensus_min_sources,
            "use_evidence_feature_reranker": config.model.use_evidence_feature_reranker,
            "evidence_feature_rerank_weight": config.model.evidence_feature_rerank_weight,
            "evidence_feature_rerank_topk": config.model.evidence_feature_rerank_topk,
            "evidence_residual_scale": config.model.evidence_residual_scale,
            "evidence_neural_agreement_weight": config.model.evidence_neural_agreement_weight,
            "recent_train_weight": config.data.recent_train_weight,
            "use_ema": config.train.use_ema,
            "ema_decay": config.train.ema_decay,
            "train_context_batch_limit": config.model.train_context_batch_limit,
            "train_path_topk": config.model.train_path_topk,
            "train_path_batch_limit": config.model.train_path_batch_limit,
            "route_supervision_weight": config.model.route_supervision_weight,
            "entity_text_example": train_dataset.entity_texts[:3],
            "relation_text_example": train_dataset.relation_texts[:3],
        },
        flush=True,
    )

    semantic_cache_dir = (
        Path(config.semantic.cache_dir)
        if config.semantic.cache_dir
        else Path(config.data.dataset_dir) / ".semantic_cache" / config.semantic.transformer_name.replace("/", "__")
    )
    print(
        {
            "stage": "semantic_features_start",
            "cache_dir": str(semantic_cache_dir),
            "backend": config.semantic.backend,
            "model": config.semantic.transformer_name,
        },
        flush=True,
    )
    print({"stage": "semantic_entity_features_start", "num_entities": train_dataset.num_entities}, flush=True)
    entity_semantic = build_text_features(
        texts=train_dataset.entity_texts,
        backend=config.semantic.backend,
        feature_dim=config.semantic.feature_dim,
        transformer_name=config.semantic.transformer_name,
        transformer_batch_size=config.semantic.transformer_batch_size,
        device=str(device),
        max_length=config.semantic.max_length,
        pooling=config.semantic.pooling,
        normalize=config.semantic.normalize,
        precision=config.semantic.precision,
        trust_remote_code=config.semantic.trust_remote_code,
        local_files_only=config.semantic.local_files_only,
        cache_dir=str(semantic_cache_dir),
        cache_name="entity",
        cache_features=config.semantic.cache_features,
        force_rebuild_cache=config.semantic.force_rebuild_cache,
    )
    print({"stage": "semantic_entity_features_done", "shape": list(entity_semantic.shape)}, flush=True)
    print({"stage": "semantic_relation_features_start", "num_relations": train_dataset.num_relations}, flush=True)
    relation_semantic = build_text_features(
        texts=train_dataset.relation_texts,
        backend=config.semantic.backend,
        feature_dim=config.semantic.feature_dim,
        transformer_name=config.semantic.transformer_name,
        transformer_batch_size=config.semantic.transformer_batch_size,
        device=str(device),
        max_length=config.semantic.max_length,
        pooling=config.semantic.pooling,
        normalize=config.semantic.normalize,
        precision=config.semantic.precision,
        trust_remote_code=config.semantic.trust_remote_code,
        local_files_only=config.semantic.local_files_only,
        cache_dir=str(semantic_cache_dir),
        cache_name="relation",
        cache_features=config.semantic.cache_features,
        force_rebuild_cache=config.semantic.force_rebuild_cache,
    )
    print({"stage": "semantic_relation_features_done", "shape": list(relation_semantic.shape)}, flush=True)

    print({"stage": "model_init_start"}, flush=True)
    model = SingleStepTemporalReasoner(
        num_entities=train_dataset.num_entities,
        num_relations=train_dataset.num_relations,
        entity_semantic_features=entity_semantic,
        relation_semantic_features=relation_semantic,
        hidden_dim=config.model.hidden_dim,
        num_rgcn_layers=config.model.num_rgcn_layers,
        dropout=config.model.dropout,
        shortlist_size=config.model.shortlist_size,
        train_context_batch_limit=config.model.train_context_batch_limit,
        train_path_topk=config.model.train_path_topk,
        train_path_batch_limit=config.model.train_path_batch_limit,
        eval_path_topk=config.model.eval_path_topk,
        max_path_len=config.data.max_path_len,
        max_paths_per_candidate=config.data.max_paths_per_candidate,
        time_decay=config.model.time_decay,
        logic_temperature=config.model.logic_temperature,
        alignment_weight=config.model.alignment_weight,
        ranking_weight=config.model.ranking_weight,
        ranking_margin=config.model.ranking_margin,
        ranking_topk=config.model.ranking_topk,
        label_smoothing=config.model.label_smoothing,
        context_score_weight=config.model.context_score_weight,
        triple_score_weight=config.model.triple_score_weight,
        distmult_score_weight=config.model.distmult_score_weight,
        transe_score_weight=config.model.transe_score_weight,
        complex_score_weight=config.model.complex_score_weight,
        conv_score_weight=config.model.conv_score_weight,
        conv_num_channels=config.model.conv_num_channels,
        conv_kernel_size=config.model.conv_kernel_size,
        conv_dropout=config.model.conv_dropout,
        normalize_neural_scores=config.model.normalize_neural_scores,
        neural_score_scale=config.model.neural_score_scale,
        use_local_graph_context=config.model.use_local_graph_context,
        use_subject_history_context=config.model.use_subject_history_context,
        use_history_transformer=config.model.use_history_transformer,
        history_transformer_layers=config.model.history_transformer_layers,
        history_transformer_heads=config.model.history_transformer_heads,
        use_concurrent_query_context=config.model.use_concurrent_query_context,
        concurrent_query_weight=config.model.concurrent_query_weight,
        concurrent_query_temperature=config.model.concurrent_query_temperature,
        history_attention_temperature=config.model.history_attention_temperature,
        history_copy_weight=config.model.history_copy_weight,
        history_copy_decay=config.model.history_copy_decay,
        relation_transfer_prior_weight=config.model.relation_transfer_prior_weight,
        relation_transfer_copy_weight=config.model.relation_transfer_copy_weight,
        relation_transfer_copy_temperature=config.model.relation_transfer_copy_temperature,
        relation_transfer_copy_history_len=config.model.relation_transfer_copy_history_len,
        relation_prior_weight=config.model.relation_prior_weight,
        local_prior_weight=config.model.local_prior_weight,
        subject_prior_weight=config.model.subject_prior_weight,
        global_prior_weight=config.model.global_prior_weight,
        path_prior_weight=config.model.path_prior_weight,
        learned_prior_weight=config.model.learned_prior_weight,
        evidence_candidate_gating=config.model.evidence_candidate_gating,
        evidence_gate_sources=config.model.evidence_gate_sources,
        evidence_gate_threshold=config.model.evidence_gate_threshold,
        evidence_gate_mode=config.model.evidence_gate_mode,
        evidence_gate_penalty=config.model.evidence_gate_penalty,
        evidence_gate_boost=config.model.evidence_gate_boost,
        evidence_gate_band_width=config.model.evidence_gate_band_width,
        evidence_gate_min_sources=config.model.evidence_gate_min_sources,
        evidence_gate_support_threshold=config.model.evidence_gate_support_threshold,
        evidence_gate_source_weights=config.model.evidence_gate_source_weights,
        evidence_gate_neural_topk=config.model.evidence_gate_neural_topk,
        evidence_gate_value=config.model.evidence_gate_value,
        evidence_gate_include_target_train=config.model.evidence_gate_include_target_train,
        evidence_rank_fusion_weight=config.model.evidence_rank_fusion_weight,
        evidence_rank_fusion_k=config.model.evidence_rank_fusion_k,
        evidence_router_temperature=config.model.evidence_router_temperature,
        use_evidence_router=config.model.use_evidence_router,
        evidence_route_floor=config.model.evidence_route_floor,
        coverage_route_weight=config.model.coverage_route_weight,
        relation_evidence_weight=config.model.relation_evidence_weight,
        evidence_consensus_weight=config.model.evidence_consensus_weight,
        evidence_consensus_min_sources=config.model.evidence_consensus_min_sources,
        use_evidence_feature_reranker=config.model.use_evidence_feature_reranker,
        evidence_feature_rerank_weight=config.model.evidence_feature_rerank_weight,
        evidence_feature_rerank_topk=config.model.evidence_feature_rerank_topk,
        evidence_residual_scale=config.model.evidence_residual_scale,
        evidence_neural_agreement_weight=config.model.evidence_neural_agreement_weight,
        evidence_neural_agreement_floor=config.model.evidence_neural_agreement_floor,
        evidence_neural_agreement_temperature=config.model.evidence_neural_agreement_temperature,
        evidence_neural_agreement_threshold=config.model.evidence_neural_agreement_threshold,
        evidence_loss_weight=config.model.evidence_loss_weight,
        evidence_margin=config.model.evidence_margin,
        route_supervision_weight=config.model.route_supervision_weight,
        route_supervision_temperature=config.model.route_supervision_temperature,
        route_entropy_weight=config.model.route_entropy_weight,
        path_weight=config.model.path_weight,
        bidirectional_paths=config.model.bidirectional_paths,
    ).to(device)
    print(
        {
            "stage": "model_init_done",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        flush=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.train.epochs),
        eta_min=config.train.min_learning_rate,
    )

    print({"stage": "dataloader_build_start"}, flush=True)
    train_loader = DataLoader(
        train_dataset,
        **_loader_kwargs(config, device=device, shuffle=True, batch_size=config.train.batch_size),
    )
    valid_loader = DataLoader(
        valid_dataset,
        **_loader_kwargs(config, device=device, shuffle=False, batch_size=config.train.eval_batch_size),
    )
    print(
        {
            "stage": "dataloader_build_done",
            "train_batches": len(train_loader),
            "valid_batches": len(valid_loader),
        },
        flush=True,
    )

    best_mrr = -1.0
    best_checkpoint = ""
    training_log = []
    use_amp = config.train.use_amp and device.type == "cuda"
    scaler = _make_grad_scaler(enabled=use_amp)
    epochs_without_improvement = 0
    ema_state = _init_ema_state(model) if config.train.use_ema else {}
    ema_decay = float(config.train.ema_decay)
    base_schedule = {
        "heavy_evidence_rate": float(config.data.train_heavy_evidence_rate),
        "train_context_batch_limit": float(config.model.train_context_batch_limit),
        "evidence_loss_weight": float(config.model.evidence_loss_weight),
        "route_supervision_weight": float(config.model.route_supervision_weight),
    }

    for epoch in range(1, config.train.epochs + 1):
        epoch_start = time.perf_counter()
        print({"stage": "epoch_start", "epoch": epoch, "epochs": config.train.epochs}, flush=True)
        model.train()
        schedule_state = _apply_training_schedule(
            config=config,
            model=model,
            train_dataset=train_dataset,
            epoch=epoch,
            base_schedule=base_schedule,
        )
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            optimizer.zero_grad(set_to_none=True)
            with _amp_autocast(enabled=use_amp, dtype=_amp_dtype()):
                output = model(batch, return_explanations=False)
            loss = output["loss"]
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.gradient_clip)
                optimizer.step()
            if ema_state:
                _update_ema_state(model, ema_state, decay=ema_decay)
            epoch_loss += float(loss.item())
            if step == 1 or step % 10 == 0 or step == len(train_loader):
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    cls=f"{output['classification_loss'].item():.4f}",
                    rank=f"{output['ranking_loss'].item():.4f}",
                    evid=f"{output['evidence_loss'].item():.4f}",
                    route=f"{output['route_loss'].item():.4f}",
                    align=f"{output['alignment_loss'].item():.4f}",
                )

        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(1, len(train_loader)),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
            "seconds_per_step": epoch_seconds / max(1, len(train_loader)),
            "schedule": schedule_state,
            "evidence_scales": {
                name: float(value)
                for name, value in zip(
                    model.evidence_source_names,
                    model.evidence_base_scales().detach().float().cpu().tolist(),
                )
            },
        }
        should_eval = epoch % max(1, config.train.eval_every) == 0
        if should_eval:
            print({"stage": "valid_start", "epoch": epoch}, flush=True)
            ema_backup = _apply_ema_state(model, ema_state) if ema_state else {}
            try:
                metrics = evaluate(model, valid_loader, valid_dataset.filter_map, device, use_amp=use_amp)
                record.update({f"valid_{key}": value for key, value in metrics.items()})
                print({"stage": "valid_done", "epoch": epoch, **metrics}, flush=True)
                improved = metrics["MRR"] > best_mrr + config.train.early_stopping_min_delta
                if improved:
                    best_mrr = metrics["MRR"]
                    epochs_without_improvement = 0
                    checkpoint_path = output_dir / "best_model.pt"
                    best_checkpoint = str(checkpoint_path)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "config": asdict(config),
                            "num_entities": train_dataset.num_entities,
                            "num_relations": train_dataset.num_relations,
                            "entity_semantic": entity_semantic.cpu(),
                            "relation_semantic": relation_semantic.cpu(),
                            "ema_checkpoint": bool(ema_state),
                        },
                        checkpoint_path,
                    )
                    record["best_checkpoint"] = best_checkpoint
                    print(
                        {
                            "stage": "checkpoint_saved",
                            "epoch": epoch,
                            "checkpoint": best_checkpoint,
                            "best_valid_mrr": best_mrr,
                        },
                        flush=True,
                    )
                else:
                    epochs_without_improvement += 1
            finally:
                if ema_backup:
                    _restore_parameter_state(model, ema_backup)
        training_log.append(record)
        print(record, flush=True)
        save_json({"log": training_log}, str(output_dir / "training_log.json"))
        if (
                config.train.early_stopping_patience > 0
                and epoch >= config.train.min_epochs
                and epochs_without_improvement >= config.train.early_stopping_patience
        ):
            print(
                {
                    "stage": "early_stop",
                    "epoch": epoch,
                    "best_valid_mrr": best_mrr,
                    "patience": config.train.early_stopping_patience,
                },
                flush=True,
            )
            break

    print(f"Best validation MRR: {best_mrr:.6f}", flush=True)
    if not best_checkpoint:
        checkpoint_path = output_dir / "best_model.pt"
        best_checkpoint = str(checkpoint_path)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "num_entities": train_dataset.num_entities,
                "num_relations": train_dataset.num_relations,
                "entity_semantic": entity_semantic.cpu(),
                "relation_semantic": relation_semantic.cpu(),
                "ema_checkpoint": bool(ema_state),
            },
            checkpoint_path,
        )
        print(
            {
                "stage": "checkpoint_saved",
                "reason": "final_epoch_fallback",
                "checkpoint": best_checkpoint,
            },
            flush=True,
        )
    if best_checkpoint and getattr(config.model, "use_validation_calibration", False):
        try:
            from eval import load_model

            print({"stage": "score_calibration_start", "checkpoint": best_checkpoint}, flush=True)
            calibration_model, _ = load_model(best_checkpoint, device)
            calibration_model.eval_path_topk = 0
            calibration_model.path_weight = 0.0
            calibration = fit_score_calibration(
                model=calibration_model,
                config=config,
                valid_loader=valid_loader,
                filter_map=valid_dataset.filter_map,
                device=device,
                use_amp=use_amp,
            )
            save_score_calibration(calibration, str(output_dir / "score_calibration.json"))
            print({"stage": "score_calibration_done", "checkpoint": best_checkpoint}, flush=True)
        except Exception as exc:
            print(
                {
                    "stage": "score_calibration_failed",
                    "error": str(exc),
                    "note": "Training checkpoint is still usable; evaluation will continue without validation calibration.",
                },
                flush=True,
            )
    return {
        "best_checkpoint": best_checkpoint,
        "output_dir": str(output_dir),
        "best_valid_mrr": f"{best_mrr:.6f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the QAER single-step reasoner")
    parser.add_argument("--config", type=str, default="", help="Optional config JSON")
    parser.add_argument("--dataset-dir", type=str, default="", help="Override dataset directory")
    parser.add_argument("--output-dir", type=str, default="", help="Override output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dataset_dir:
        config.data.dataset_dir = args.dataset_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    train_model(config)


if __name__ == "__main__":
    main()
