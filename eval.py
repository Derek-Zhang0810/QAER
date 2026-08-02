import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import load_config
from data_loader import TemporalReasoningDataset, collate_temporal_batch
from model import SingleStepTemporalReasoner
from score_calibration import (
    apply_score_calibration,
    fit_score_calibration,
    load_score_calibration,
    save_score_calibration,
)
from utils import aggregate_ranking_metrics, filtered_rank, resolve_device, save_json


CALIBRATION_CONFIG_FIELDS = [
    "use_validation_calibration",
    "calibration_epochs",
    "calibration_learning_rate",
    "calibration_l2",
    "use_relation_calibration",
    "calibration_relation_l2",
    "use_prior_feature_calibration",
    "calibration_feature_l2",
    "use_relation_feature_calibration",
    "calibration_relation_feature_l2",
    "calibration_batch_limit",
    "calibration_pairwise_weight",
    "calibration_pairwise_margin",
    "calibration_pairwise_topk",
    "calibration_select_by_mrr",
    "calibration_selection_metric",
    "calibration_selection_mrr_weight",
    "calibration_selection_hits1_weight",
    "calibration_nonnegative_source_deltas",
    "calibration_decompose_evidence_scores",
    "calibration_evidence_scale_l2",
    "use_relation_evidence_calibration",
    "calibration_relation_evidence_l2",
    "use_entity_bias_calibration",
    "calibration_entity_bias_l2",
]

MODEL_EVAL_CONFIG_FIELDS = [
    "normalize_neural_scores",
    "neural_score_scale",
    "distmult_score_weight",
    "transe_score_weight",
    "complex_score_weight",
    "conv_score_weight",
    "history_copy_weight",
    "history_copy_decay",
    "relation_transfer_prior_weight",
    "relation_transfer_copy_weight",
    "relation_transfer_copy_temperature",
    "relation_transfer_copy_history_len",
    "relation_prior_weight",
    "local_prior_weight",
    "subject_prior_weight",
    "global_prior_weight",
    "path_prior_weight",
    "learned_prior_weight",
    "use_evidence_router",
    "evidence_router_temperature",
    "evidence_route_floor",
    "coverage_route_weight",
    "relation_evidence_weight",
    "evidence_consensus_weight",
    "evidence_consensus_min_sources",
    "use_evidence_feature_reranker",
    "evidence_feature_rerank_weight",
    "evidence_feature_rerank_topk",
    "evidence_residual_scale",
    "evidence_neural_agreement_weight",
    "evidence_neural_agreement_floor",
    "evidence_neural_agreement_temperature",
    "evidence_neural_agreement_threshold",
    "evidence_candidate_gating",
    "evidence_gate_sources",
    "evidence_gate_threshold",
    "evidence_gate_mode",
    "evidence_gate_penalty",
    "evidence_gate_boost",
    "evidence_gate_band_width",
    "evidence_gate_min_sources",
    "evidence_gate_support_threshold",
    "evidence_gate_source_weights",
    "evidence_gate_neural_topk",
    "evidence_gate_value",
    "evidence_gate_include_target_train",
    "evidence_rank_fusion_weight",
    "evidence_rank_fusion_k",
]

DATA_EVAL_CONFIG_FIELDS = [
    "valid_heavy_evidence",
    "eval_heavy_evidence",
    "path_prior_topk",
    "path_prior_cache_size",
]


def _amp_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def _amp_autocast(enabled: bool, dtype: torch.dtype):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled, dtype=dtype)
    return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)


def _enable_cuda_fast_math(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def load_model(checkpoint_path: str, device: torch.device, config_path: str = "") -> tuple:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = load_config("")
    if "config" in checkpoint:
        config_payload = checkpoint["config"]
        from configs import _deep_update_dataclass

        _deep_update_dataclass(config, config_payload)
    if config_path:
        override = load_config(config_path)
        for field_name in MODEL_EVAL_CONFIG_FIELDS:
            if hasattr(config.model, field_name) and hasattr(override.model, field_name):
                setattr(config.model, field_name, getattr(override.model, field_name))
        for field_name in DATA_EVAL_CONFIG_FIELDS:
            if hasattr(config.data, field_name) and hasattr(override.data, field_name):
                setattr(config.data, field_name, getattr(override.data, field_name))
    model = SingleStepTemporalReasoner(
        num_entities=checkpoint["num_entities"],
        num_relations=checkpoint["num_relations"],
        entity_semantic_features=checkpoint["entity_semantic"],
        relation_semantic_features=checkpoint["relation_semantic"],
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
    checkpoint_state = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    compatible_state = {}
    skipped_keys = []
    for key, value in checkpoint_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped_keys.append(key)
    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
    if missing_keys or unexpected_keys or skipped_keys:
        incompatible_keys = list(missing_keys) + list(skipped_keys)
        disabled_modules = []
        if any(key.startswith("triple_query_proj") for key in incompatible_keys):
            model.triple_score_weight = 0.0
            disabled_modules.append("triple_query_proj")
        if any(key.startswith("prior_feature_mlp") for key in incompatible_keys):
            model.learned_prior_weight = 0.0
            disabled_modules.append("prior_feature_mlp")
        if any(key.startswith("history_transformer") for key in incompatible_keys):
            model.use_history_transformer = False
            disabled_modules.append("history_transformer")
        if any(key.startswith("concurrent_query_proj") for key in incompatible_keys):
            with torch.no_grad():
                for parameter in model.concurrent_query_proj.parameters():
                    parameter.zero_()
            disabled_modules.append("concurrent_query_proj_zeroed")
        if any(key.startswith("query_mlp.0") for key in incompatible_keys):
            disabled_modules.append("query_mlp_shape_mismatch")
        if any(key.startswith("evidence_feature_reranker") for key in incompatible_keys):
            model.use_evidence_feature_reranker = False
            disabled_modules.append("evidence_feature_reranker")
        if any(key.startswith("conv_decoder") for key in incompatible_keys):
            model.conv_score_weight = 0.0
            disabled_modules.append("conv_decoder")
        if any(key.startswith("relation_evidence_deltas") for key in incompatible_keys):
            model.relation_evidence_weight = 0.0
            disabled_modules.append("relation_evidence_deltas")
        print(
            {
                "stage": "checkpoint_compatibility",
                "missing_keys": missing_keys[:8],
                "unexpected_keys": unexpected_keys[:8],
                "skipped_shape_keys": skipped_keys[:8],
                "disabled_modules": sorted(set(disabled_modules)),
                "note": "Missing compatible parameters use config initialization; shape-mismatched modules may be disabled. Retrain for final results.",
            },
            flush=True,
        )
    model.eval()
    return model, config


def _label(items: List[str], idx: int) -> str:
    if 0 <= idx < len(items):
        return items[idx]
    return str(idx)


def _relation_label(items: List[str], relation: int, num_relations: int) -> str:
    base = relation % num_relations
    label = _label(items, base)
    return f"{label} (inverse)" if relation >= num_relations else label


def _path_to_natural_text(path_payload: Dict, entity_texts: List[str], relation_texts: List[str]) -> str:
    nodes = path_payload.get("nodes", [])
    relations = path_payload.get("relations", [])
    timestamps = path_payload.get("timestamps", [])
    if not nodes:
        return ""
    chunks = [_label(entity_texts, int(nodes[0]))]
    for rel, node, tim in zip(relations, nodes[1:], timestamps):
        chunks.append(f"--[{_relation_label(relation_texts, int(rel), len(relation_texts))} @ {int(tim)}]-->")
        chunks.append(_label(entity_texts, int(node)))
    return " ".join(chunks)


def _find_weight(entity: int, entities: List[int], weights: List[float]) -> float:
    for candidate, weight in zip(entities, weights):
        if int(candidate) == int(entity):
            return float(weight)
    return 0.0


def _score_breakdown(
        entity: int,
        row: int,
        batch: Dict,
        model: SingleStepTemporalReasoner,
        output: Dict,
        calibrated_scores: torch.Tensor,
        gate_mask: torch.Tensor = None,
        gate_support: torch.Tensor = None,
) -> Dict:
    relation = int(batch["relations"][row])
    entity_id = int(entity)
    neural_scores = output.get("neural_scores")
    pre_calibration_scores = output["scores"]
    route_weights = output.get("evidence_route_weights")
    route_names = output.get("evidence_source_names", model.evidence_source_names)
    if route_weights is None:
        route_vector = torch.full(
            (len(route_names),),
            1.0 / max(1, len(route_names)),
            dtype=torch.float32,
        )
    else:
        route_vector = route_weights[row].detach().float().cpu()
    base_scales = model.evidence_base_scales(relation=relation).detach().float().cpu()
    source_fields = [
        ("copy", "copy_entities", "copy_weights"),
        ("local", "local_prior_entities", "local_prior_weights"),
        ("relation", "relation_prior_entities", "relation_prior_weights"),
        ("subject", "subject_prior_entities", "subject_prior_weights"),
        ("global", "global_prior_entities", "global_prior_weights"),
        ("path", "path_prior_entities", "path_prior_weights"),
    ]
    source_count = float(len(source_fields))
    evidence_route_floor = float(getattr(model, "evidence_route_floor", 0.0))
    source_breakdown = {}
    routed_sum = 0.0
    for source_idx, (name, entity_key, weight_key) in enumerate(source_fields):
        raw_support = _find_weight(
            entity_id,
            batch.get(entity_key, [[]])[row],
            batch.get(weight_key, [[]])[row],
        )
        route_weight = float(route_vector[source_idx].item()) if source_idx < route_vector.numel() else 0.0
        base_scale = float(base_scales[source_idx].item()) if source_idx < base_scales.numel() else 0.0
        routed_scale = route_weight * source_count + evidence_route_floor
        contribution = raw_support * routed_scale * base_scale
        routed_sum += contribution
        source_breakdown[name] = {
            "raw_support": raw_support,
            "route_weight": route_weight,
            "base_scale": base_scale,
            "routed_contribution": contribution,
        }
    relation_transfer_support = _find_weight(
        entity_id,
        batch.get("relation_transfer_entities", [[]])[row],
        batch.get("relation_transfer_weights", [[]])[row],
    )
    relation_transfer_contribution = (
        relation_transfer_support * float(getattr(model, "relation_transfer_prior_weight", 0.0))
    )
    gate_info = {"enabled": bool(getattr(model, "evidence_candidate_gating", False))}
    if gate_info["enabled"]:
        if gate_mask is None or gate_support is None:
            gate_mask, gate_support = model.evidence_candidate_support(
                batch=batch,
                row=row,
                score_template=pre_calibration_scores[row],
                target=None,
                route_weights=route_weights[row] if route_weights is not None else None,
                neural_scores=neural_scores[row] if neural_scores is not None else None,
            )
        gate_info.update(
            {
                "mode": str(getattr(model, "evidence_gate_mode", "")),
                "sources": sorted(getattr(model, "evidence_gate_sources", [])),
                "supported": bool(gate_mask[entity_id].detach().cpu().item()),
                "support": float(gate_support[entity_id].detach().float().cpu().item()),
                "penalty": float(getattr(model, "evidence_gate_penalty", 0.0)),
                "boost": float(getattr(model, "evidence_gate_boost", 0.0)),
                "band_width": float(getattr(model, "evidence_gate_band_width", 0.0)),
                "min_sources": int(getattr(model, "evidence_gate_min_sources", 1)),
                "support_threshold": float(getattr(model, "evidence_gate_support_threshold", 0.0)),
                "source_weights": {
                    name: float(weight)
                    for name, weight in zip(
                        getattr(model, "evidence_source_names", []),
                        getattr(model, "evidence_gate_source_weight_values", []),
                    )
                },
                "neural_topk": int(getattr(model, "evidence_gate_neural_topk", 0)),
            }
        )
    neural_score = (
        float(neural_scores[row, entity_id].detach().float().cpu().item())
        if neural_scores is not None else None
    )
    pre_score = float(pre_calibration_scores[row, entity_id].detach().float().cpu().item())
    final_score = float(calibrated_scores[row, entity_id].detach().float().cpu().item())
    other_adjustment = (
        pre_score - (neural_score or 0.0) - routed_sum - relation_transfer_contribution
        if neural_score is not None else None
    )
    return {
        "entity": entity_id,
        "neural_score": neural_score,
        "qaer_pre_calibration_score": pre_score,
        "calibrated_final_score": final_score,
        "routed_evidence_sum": routed_sum,
        "relation_transfer_prior_support": relation_transfer_support,
        "relation_transfer_prior_contribution": relation_transfer_contribution,
        "other_qaer_adjustment": other_adjustment,
        "gate": gate_info,
        "sources": source_breakdown,
    }


def _masked_filtered_scores(scores: torch.Tensor, target: int, filtered_tails: set) -> torch.Tensor:
    masked = scores.detach().clone()
    for tail in filtered_tails:
        if int(tail) != int(target):
            masked[int(tail)] = -1e9
    return masked


def _top_predictions(
        scores: torch.Tensor,
        entity_texts: List[str],
        k: int,
) -> tuple:
    top_values, top_indices = torch.topk(scores, k=min(k, scores.size(0)), dim=0)
    predictions = [
        {"entity": int(entity), "score": float(score)}
        for entity, score in zip(top_indices.tolist(), top_values.tolist())
        if float(score) > -1e8
    ]
    prediction_text = "; ".join(
        f"{rank}. {_label(entity_texts, item['entity'])}({item['entity']}):{item['score']:.4f}"
        for rank, item in enumerate(predictions, start=1)
    )
    return predictions, prediction_text


def _build_evidence_text(
        predicted: int,
        explanations: Dict,
        batch: Dict,
        row: int,
        dataset: TemporalReasoningDataset,
        route_weights: List[float] = None,
        route_names: List[str] = None,
) -> str:
    pieces = []
    if route_weights is not None and route_names is not None:
        routed = sorted(
            zip(route_names, route_weights),
            key=lambda item: item[1],
            reverse=True,
        )
        pieces.append(
            "Query-adaptive evidence routing: "
            + ", ".join(f"{name}={weight:.3f}" for name, weight in routed)
            + "."
        )
    path_payloads = explanations.get(int(predicted), []) or explanations.get(str(int(predicted)), [])
    for payload in path_payloads[:3]:
        text = _path_to_natural_text(payload, dataset.entity_texts, dataset.relation_texts)
        if text:
            pieces.append(f"Path evidence: {text}")

    copy_weight = _find_weight(predicted, batch.get("copy_entities", [[]])[row], batch.get("copy_weights", [[]])[row])
    if copy_weight > 0:
        pieces.append(
            f"Subject-history evidence: the query subject recently reached {_label(dataset.entity_texts, predicted)} "
            f"under the same relation; normalized weight={copy_weight:.3f}."
        )

    local_weight = _find_weight(
        predicted,
        batch.get("local_prior_entities", [[]])[row],
        batch.get("local_prior_weights", [[]])[row],
    )
    if local_weight > 0:
        pieces.append(
            f"Local temporal evidence: within the recent time window, this subject-relation pattern pointed to "
            f"{_label(dataset.entity_texts, predicted)}; normalized weight={local_weight:.3f}."
        )

    relation_weight = _find_weight(
        predicted,
        batch.get("relation_prior_entities", [[]])[row],
        batch.get("relation_prior_weights", [[]])[row],
    )
    if relation_weight > 0:
        pieces.append(
            f"Relation-time evidence: {_label(dataset.entity_texts, predicted)} is a frequent recent candidate for "
            f"this relation before the query time; normalized weight={relation_weight:.3f}."
        )

    subject_weight = _find_weight(
        predicted,
        batch.get("subject_prior_entities", [[]])[row],
        batch.get("subject_prior_weights", [[]])[row],
    )
    if subject_weight > 0:
        pieces.append(
            f"Subject temporal evidence: the query subject recently interacted with "
            f"{_label(dataset.entity_texts, predicted)} before the query time; normalized weight={subject_weight:.3f}."
        )

    global_weight = _find_weight(
        predicted,
        batch.get("global_prior_entities", [[]])[row],
        batch.get("global_prior_weights", [[]])[row],
    )
    if global_weight > 0:
        pieces.append(
            f"Global temporal evidence: {_label(dataset.entity_texts, predicted)} is active in the recent temporal window; "
            f"normalized weight={global_weight:.3f}."
        )

    path_weight = _find_weight(
        predicted,
        batch.get("path_prior_entities", [[]])[row],
        batch.get("path_prior_weights", [[]])[row],
    )
    if path_weight > 0:
        pieces.append(
            f"Training-aligned path evidence: {_label(dataset.entity_texts, predicted)} is reachable from the query subject "
            f"through recent temporal paths before the query time; normalized path support={path_weight:.3f}."
        )

    if not pieces:
        pieces.append("Neural structural-semantic evidence: ranked by the fused entity, relation, local graph, and history representations; no explicit path was found in the retained evidence set.")
    return " | ".join(pieces)


def save_evidence_csv(rows: List[Dict], output_csv: str) -> None:
    if not output_csv:
        return
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_subject",
        "query_relation",
        "query_time",
        "target",
        "target_rank",
        "raw_target_rank",
        "predicted_entity",
        "prediction_score",
        "top_predictions",
        "raw_predicted_entity",
        "raw_prediction_score",
        "raw_top_predictions",
        "score_breakdown",
        "structured_evidence_chain",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(
        checkpoint_path: str,
        dataset_dir: str = "",
        split: str = "test",
        output_json: str = "",
        output_csv: str = "",
        config_path: str = "",
        max_eval_batches: int = 0,
) -> \
        Dict[str, float]:
    device = resolve_device("cuda")
    model, config = load_model(checkpoint_path, device, config_path=config_path)
    _enable_cuda_fast_math(device)
    if dataset_dir:
        config.data.dataset_dir = dataset_dir
    model.eval_path_topk = 0
    model.path_weight = 0.0
    calibration = load_score_calibration(checkpoint_path)

    dataset = TemporalReasoningDataset(config.data, split=split)
    loader_kwargs = {
        "batch_size": config.train.eval_batch_size,
        "shuffle": False,
        "collate_fn": collate_temporal_batch,
        "num_workers": config.train.num_workers,
        "pin_memory": config.train.pin_memory and device.type == "cuda",
    }
    if config.train.num_workers > 0:
        loader_kwargs["persistent_workers"] = config.train.persistent_workers
        loader_kwargs["prefetch_factor"] = config.train.prefetch_factor
    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )

    ranks = []
    evidence_records = []
    evidence_csv_rows = []
    rank_audit = {
        "evaluated_examples": 0,
        "target_present_in_saved_top_predictions": 0,
        "rank_le_1_but_target_absent_from_saved_top_predictions": 0,
        "rank_le_3_but_target_absent_from_saved_top_predictions": 0,
        "rank_le_10_but_target_absent_from_saved_top_predictions": 0,
        "saved_top_predictions_total": 0,
    }
    use_amp = config.train.use_amp and device.type == "cuda"
    print(
        {
            "stage": "eval_setup",
            "metric_path_rerank": False,
            "path_prior_scoring": True,
            "validation_calibration": bool(calibration.get("enabled", False)),
            "note": "Final ranking now matches validation: path prior can support scores, but untrained explicit path reranking is disabled.",
        },
        flush=True,
    )
    eval_iter = loader
    total_batches = len(loader)
    if max_eval_batches:
        total_batches = min(total_batches, int(max_eval_batches))
        eval_iter = itertools.islice(loader, int(max_eval_batches))
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(eval_iter, desc=f"evaluate-{split}", total=total_batches)):
            with _amp_autocast(enabled=use_amp, dtype=_amp_dtype()):
                output = model(batch, return_explanations=False)
            scores = apply_score_calibration(
                output["scores"],
                batch=batch,
                calibration=calibration,
                route_weights=output.get("evidence_route_weights"),
                neural_scores=output.get("neural_scores"),
            )
            for row in range(scores.size(0)):
                subject = int(batch["subjects"][row])
                relation = int(batch["relations"][row])
                target = int(batch["targets"][row])
                query_time = int(batch["times"][row])
                filtered_tails = dataset.filter_map.get((subject, relation, query_time), set())
                rank = filtered_rank(
                    scores[row],
                    target=target,
                    filtered_tails=filtered_tails,
                )
                raw_rank = filtered_rank(scores[row], target=target, filtered_tails=set())
                ranks.append(rank)
                filtered_scores = _masked_filtered_scores(
                    scores=scores[row],
                    target=target,
                    filtered_tails=filtered_tails,
                )
                predictions, top_prediction_text = _top_predictions(
                    scores=filtered_scores,
                    entity_texts=dataset.entity_texts,
                    k=config.explain.top_k_predictions,
                )
                raw_predictions, raw_top_prediction_text = _top_predictions(
                    scores=scores[row],
                    entity_texts=dataset.entity_texts,
                    k=config.explain.top_k_predictions,
                )
                predicted_entity = int(predictions[0]["entity"]) if predictions else -1
                prediction_score = float(predictions[0]["score"]) if predictions else float("-inf")
                raw_predicted_entity = int(raw_predictions[0]["entity"]) if raw_predictions else -1
                raw_prediction_score = float(raw_predictions[0]["score"]) if raw_predictions else float("-inf")
                gate_mask = None
                gate_support = None
                if bool(getattr(model, "evidence_candidate_gating", False)):
                    gate_mask, gate_support = model.evidence_candidate_support(
                        batch=batch,
                        row=row,
                        score_template=output["scores"][row],
                        target=None,
                        route_weights=output.get("evidence_route_weights", None)[row]
                        if output.get("evidence_route_weights", None) is not None else None,
                        neural_scores=output.get("neural_scores", None)[row]
                        if output.get("neural_scores", None) is not None else None,
                    )
                score_breakdown = (
                    _score_breakdown(
                        entity=predicted_entity,
                        row=row,
                        batch=batch,
                        model=model,
                        output=output,
                        calibrated_scores=scores,
                        gate_mask=gate_mask,
                        gate_support=gate_support,
                    )
                    if predicted_entity >= 0 else {}
                )
                if predictions:
                    enriched_predictions = []
                    for prediction in predictions:
                        prediction_entity = int(prediction["entity"])
                        enriched_prediction = dict(prediction)
                        enriched_prediction["score_breakdown"] = _score_breakdown(
                            entity=prediction_entity,
                            row=row,
                            batch=batch,
                            model=model,
                            output=output,
                            calibrated_scores=scores,
                            gate_mask=gate_mask,
                            gate_support=gate_support,
                        )
                        enriched_predictions.append(enriched_prediction)
                    predictions = enriched_predictions
                    score_breakdown = predictions[0]["score_breakdown"]
                saved_prediction_ids = {int(item["entity"]) for item in predictions}
                target_visible = int(target) in saved_prediction_ids
                rank_audit["evaluated_examples"] += 1
                rank_audit["saved_top_predictions_total"] += len(predictions)
                rank_audit["target_present_in_saved_top_predictions"] += int(target_visible)
                if not target_visible:
                    rank_audit["rank_le_1_but_target_absent_from_saved_top_predictions"] += int(rank <= 1)
                    rank_audit["rank_le_3_but_target_absent_from_saved_top_predictions"] += int(rank <= 3)
                    rank_audit["rank_le_10_but_target_absent_from_saved_top_predictions"] += int(rank <= 10)
                explanations = output["explanations"][row]
                evidence_target = predicted_entity if predicted_entity >= 0 else raw_predicted_entity
                raw_predicted_label = (
                    f"{_label(dataset.entity_texts, raw_predicted_entity)} ({raw_predicted_entity})"
                    if raw_predicted_entity >= 0
                    else ""
                )
                predicted_label = (
                    f"{_label(dataset.entity_texts, predicted_entity)} ({predicted_entity})"
                    if predicted_entity >= 0
                    else ""
                )
                evidence_text = _build_evidence_text(
                    predicted=evidence_target,
                    explanations=explanations,
                    batch=batch,
                    row=row,
                    dataset=dataset,
                    route_weights=output.get("evidence_route_weights", torch.empty(0))[row].float().cpu().tolist()
                    if "evidence_route_weights" in output else None,
                    route_names=output.get("evidence_source_names"),
                )
                evidence_csv_rows.append(
                    {
                        "query_subject": f"{_label(dataset.entity_texts, subject)} ({subject})",
                        "query_relation": f"{_relation_label(dataset.relation_texts, relation, dataset.num_relations)} ({relation})",
                        "query_time": query_time,
                        "target": f"{_label(dataset.entity_texts, target)} ({target})",
                        "target_rank": rank,
                        "raw_target_rank": raw_rank,
                        "predicted_entity": predicted_label,
                        "prediction_score": f"{prediction_score:.6f}",
                        "top_predictions": top_prediction_text,
                        "raw_predicted_entity": raw_predicted_label,
                        "raw_prediction_score": f"{raw_prediction_score:.6f}",
                        "raw_top_predictions": raw_top_prediction_text,
                        "score_breakdown": json.dumps(score_breakdown, ensure_ascii=False),
                        "structured_evidence_chain": evidence_text,
                    }
                )
                evidence_records.append(
                    {
                        "query": {
                            "subject": subject,
                            "relation": relation,
                            "target": target,
                            "time": query_time,
                        },
                        "rank": rank,
                        "raw_rank": raw_rank,
                        "top_predictions": predictions,
                        "raw_top_predictions": raw_predictions,
                        "score_breakdown": score_breakdown,
                        "candidate_explanations": explanations,
                    }
                )

    metrics = aggregate_ranking_metrics(ranks)
    print(metrics, flush=True)
    if rank_audit["evaluated_examples"] > 0:
        rank_audit["avg_saved_top_predictions"] = (
            rank_audit["saved_top_predictions_total"] / rank_audit["evaluated_examples"]
        )
        print({"stage": "rank_audit", **rank_audit}, flush=True)
    if output_json:
        save_json(
            {
                "split": split,
                "metrics": metrics,
                "rank_audit": rank_audit,
                "records": evidence_records,
            },
            output_json,
        )
        print(f"Saved evidence JSON to {output_json}", flush=True)
    if output_csv:
        save_evidence_csv(evidence_csv_rows, output_csv)
        print(f"Saved evidence CSV to {output_csv}", flush=True)
    return metrics


def _apply_calibration_config_override(config, config_path: str) -> None:
    if not config_path:
        return
    override = load_config(config_path)
    for field_name in CALIBRATION_CONFIG_FIELDS:
        if hasattr(config.model, field_name) and hasattr(override.model, field_name):
            setattr(config.model, field_name, getattr(override.model, field_name))
    config.train.eval_batch_size = override.train.eval_batch_size
    config.train.use_amp = override.train.use_amp
    print(
        {
            "stage": "calibration_config_override",
            "config": config_path,
            "pairwise_weight": config.model.calibration_pairwise_weight,
            "pairwise_topk": config.model.calibration_pairwise_topk,
            "select_by_mrr": config.model.calibration_select_by_mrr,
            "decompose_scores": config.model.calibration_decompose_evidence_scores,
            "entity_bias": config.model.use_entity_bias_calibration,
        },
        flush=True,
    )


def fit_checkpoint_calibration(checkpoint_path: str, dataset_dir: str = "", config_path: str = "") -> Dict:
    device = resolve_device("cuda")
    model, config = load_model(checkpoint_path, device, config_path=config_path)
    _apply_calibration_config_override(config, config_path)
    if dataset_dir:
        config.data.dataset_dir = dataset_dir
    dataset = TemporalReasoningDataset(config.data, split="valid")
    loader_kwargs = {
        "batch_size": config.train.eval_batch_size,
        "shuffle": False,
        "collate_fn": collate_temporal_batch,
        "num_workers": config.train.num_workers,
        "pin_memory": config.train.pin_memory and device.type == "cuda",
    }
    if config.train.num_workers > 0:
        loader_kwargs["persistent_workers"] = config.train.persistent_workers
        loader_kwargs["prefetch_factor"] = config.train.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    use_amp = config.train.use_amp and device.type == "cuda"
    calibration = fit_score_calibration(
        model=model,
        config=config,
        valid_loader=loader,
        filter_map=dataset.filter_map,
        device=device,
        use_amp=use_amp,
    )
    save_score_calibration(calibration, str(Path(checkpoint_path).with_name("score_calibration.json")))
    return calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the QAER single-step reasoner")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional config whose calibration fields override the checkpoint config when fitting calibration.",
    )
    parser.add_argument("--dataset-dir", type=str, default="", help="Override dataset directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument(
        "--fit-calibration",
        action="store_true",
        help="Fit validation-only score calibration next to the checkpoint before evaluation.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Debug helper: evaluate only the first N batches when N > 0.",
    )
    args = parser.parse_args()

    if args.fit_calibration:
        fit_checkpoint_calibration(
            checkpoint_path=args.checkpoint,
            dataset_dir=args.dataset_dir,
            config_path=args.config,
        )

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        split=args.split,
        output_json=args.output_json,
        output_csv=args.output_csv,
        config_path=args.config,
        max_eval_batches=args.max_eval_batches,
    )


if __name__ == "__main__":
    main()
