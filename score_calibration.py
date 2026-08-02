import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import aggregate_ranking_metrics, filtered_rank


SOURCE_FIELDS = [
    ("copy", "copy_entities", "copy_weights"),
    ("local", "local_prior_entities", "local_prior_weights"),
    ("relation", "relation_prior_entities", "relation_prior_weights"),
    ("subject", "subject_prior_entities", "subject_prior_weights"),
    ("global", "global_prior_entities", "global_prior_weights"),
    ("path", "path_prior_entities", "path_prior_weights"),
]

PRIOR_FEATURE_DIM = 48


def _amp_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def _amp_autocast(enabled: bool, dtype: torch.dtype):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled, dtype=dtype)
    return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)


def _inverse_softplus(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float32)
    return float(torch.log(torch.expm1(tensor)).item())


def _add_sparse_source_bias(
        scores: torch.Tensor,
        batch: Dict,
        source_values: torch.Tensor,
) -> torch.Tensor:
    calibrated = torch.zeros_like(scores, dtype=torch.float32)
    for source_idx, (_, entity_key, weight_key) in enumerate(SOURCE_FIELDS):
        entity_rows: List = batch.get(entity_key, [[] for _ in range(scores.size(0))])
        weight_rows: List = batch.get(weight_key, [[] for _ in range(scores.size(0))])
        for row, (entities, weights) in enumerate(zip(entity_rows, weight_rows)):
            if not entities:
                continue
            scale = source_values[source_idx] if source_values.dim() == 1 else source_values[row, source_idx]
            ids = torch.as_tensor(entities, dtype=torch.long, device=scores.device)
            values = torch.as_tensor(weights, dtype=torch.float32, device=scores.device)
            calibrated[row].index_add_(0, ids, values * scale)
    return calibrated


def _candidate_prior_features(
        batch: Dict,
        row: int,
        route_weights: torch.Tensor = None,
        neural_scores: torch.Tensor = None,
        model_scores: torch.Tensor = None,
        feature_dim: int = PRIOR_FEATURE_DIM,
) -> tuple:
    feature_map: Dict[int, List[float]] = {}
    source_rank_map: Dict[int, List[float]] = {}

    def collect(entity_key: str, weight_key: str, column: int) -> None:
        for rank, (entity, weight) in enumerate(
                zip(batch.get(entity_key, [[]])[row], batch.get(weight_key, [[]])[row]),
                start=1,
        ):
            entity_id = int(entity)
            feature_map.setdefault(entity_id, [0.0] * len(SOURCE_FIELDS))[column] += float(weight)
            source_rank_map.setdefault(entity_id, [0.0] * len(SOURCE_FIELDS))[column] = max(
                source_rank_map.setdefault(entity_id, [0.0] * len(SOURCE_FIELDS))[column],
                1.0 / float(rank),
            )

    for column, (_, entity_key, weight_key) in enumerate(SOURCE_FIELDS):
        collect(entity_key, weight_key, column)
    if not feature_map:
        return [], []

    if route_weights is None:
        route_vector = [1.0 / float(len(SOURCE_FIELDS))] * len(SOURCE_FIELDS)
    elif torch.is_tensor(route_weights):
        route_vector = [float(value) for value in route_weights.detach().float().clamp_min(1e-8).tolist()]
    else:
        route_vector = [float(value) for value in route_weights]
    route_entropy = -sum(value * torch.log(torch.tensor(max(value, 1e-8))).item() for value in route_vector)
    max_route = max(route_vector)

    candidate_ids = sorted(feature_map)
    neural_z_lookup = {}
    neural_rank_lookup = {}
    if neural_scores is not None:
        row_neural = neural_scores.detach().float()
        neural_mean = float(row_neural.mean().item())
        neural_std = float(row_neural.std().clamp_min(1e-4).item())
        order = torch.argsort(row_neural, descending=True)
        ranks = torch.empty_like(order, dtype=torch.long)
        ranks[order] = torch.arange(order.numel(), dtype=torch.long, device=order.device)
        for entity_id in candidate_ids:
            if 0 <= int(entity_id) < row_neural.numel():
                score_value = float(row_neural[int(entity_id)].item())
                rank_value = int(ranks[int(entity_id)].item()) + 1
                neural_z_lookup[entity_id] = max(-5.0, min(5.0, (score_value - neural_mean) / neural_std))
                neural_rank_lookup[entity_id] = 1.0 / float(rank_value)

    model_z_lookup = {}
    model_rank_lookup = {}
    model_gap_lookup = {}
    if model_scores is not None:
        row_scores = model_scores.detach().float()
        score_mean = float(row_scores.mean().item())
        score_std = float(row_scores.std().clamp_min(1e-4).item())
        top_score = float(row_scores.max().item())
        order = torch.argsort(row_scores, descending=True)
        ranks = torch.empty_like(order, dtype=torch.long)
        ranks[order] = torch.arange(order.numel(), dtype=torch.long, device=order.device)
        for entity_id in candidate_ids:
            if 0 <= int(entity_id) < row_scores.numel():
                score_value = float(row_scores[int(entity_id)].item())
                rank_value = int(ranks[int(entity_id)].item()) + 1
                model_z_lookup[entity_id] = max(-5.0, min(5.0, (score_value - score_mean) / score_std))
                model_rank_lookup[entity_id] = 1.0 / float(rank_value)
                model_gap_lookup[entity_id] = max(-20.0, min(0.0, score_value - top_score))

    features = []
    for entity_id in candidate_ids:
        copy_weight, local_weight, relation_weight, subject_weight, global_weight, path_weight = feature_map[entity_id]
        source_rank_rr = source_rank_map.get(entity_id, [0.0] * len(SOURCE_FIELDS))
        raw_values = [
            copy_weight,
            local_weight,
            relation_weight,
            subject_weight,
            global_weight,
            path_weight,
        ]
        routed_values = [value * route for value, route in zip(raw_values, route_vector)]
        source_sum = sum(raw_values)
        source_max = max(raw_values)
        source_count = float(sum(1 for value in raw_values if value > 0.0))
        routed_sum = sum(routed_values)
        routed_max = max(routed_values)
        routed_count = float(sum(1 for value in routed_values if value > 0.0))
        indicators = [float(value > 0.0) for value in raw_values]
        neural_z = float(neural_z_lookup.get(entity_id, 0.0))
        neural_rr = float(neural_rank_lookup.get(entity_id, 0.0))
        model_z = float(model_z_lookup.get(entity_id, 0.0))
        model_rr = float(model_rank_lookup.get(entity_id, 0.0))
        model_gap = float(model_gap_lookup.get(entity_id, 0.0))
        full_features = [
            copy_weight,
            local_weight,
            relation_weight,
            subject_weight,
            global_weight,
            path_weight,
            source_sum,
            source_max,
            source_count,
            float(copy_weight > 0.0 and local_weight > 0.0),
            float(subject_weight > 0.0 and relation_weight > 0.0),
            float(path_weight > 0.0 and local_weight > 0.0),
            float(path_weight > 0.0 and relation_weight > 0.0),
            *routed_values,
            routed_sum,
            routed_max,
            routed_count,
            route_entropy,
            max_route,
            *indicators,
            neural_z,
            neural_rr,
            source_sum * neural_rr,
            source_max * neural_rr,
            source_count * neural_rr,
            float(copy_weight > 0.0) * neural_rr,
            model_z,
            model_rr,
            model_gap,
            source_sum * model_rr,
            source_max * model_rr,
            source_count * model_rr,
            model_z - neural_z,
            *source_rank_rr,
            max(source_rank_rr) if source_rank_rr else 0.0,
        ]
        if feature_dim < len(full_features):
            full_features = full_features[:feature_dim]
        elif feature_dim > len(full_features):
            full_features = full_features + [0.0] * (feature_dim - len(full_features))
        features.append(full_features)
    return candidate_ids, features


def _add_prior_feature_bias(
        scores: torch.Tensor,
        batch: Dict,
        feature_weights: torch.Tensor,
        feature_bias: torch.Tensor,
        relation_feature_deltas: torch.Tensor = None,
        route_weights: torch.Tensor = None,
        neural_scores: torch.Tensor = None,
        model_scores: torch.Tensor = None,
) -> torch.Tensor:
    calibrated = torch.zeros_like(scores, dtype=torch.float32)
    relation_ids = batch["relations"].to(scores.device).long()
    feature_dim = int(feature_weights.numel())
    for row in range(scores.size(0)):
        row_route = route_weights[row] if route_weights is not None else None
        row_neural = neural_scores[row] if neural_scores is not None else None
        candidate_ids, features = _candidate_prior_features(
            batch=batch,
            row=row,
            route_weights=row_route,
            neural_scores=row_neural,
            model_scores=model_scores[row] if model_scores is not None else None,
            feature_dim=feature_dim,
        )
        if not candidate_ids:
            continue
        ids = torch.as_tensor(candidate_ids, dtype=torch.long, device=scores.device)
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=scores.device)
        weights = feature_weights
        if relation_feature_deltas is not None and relation_feature_deltas.numel() > 0:
            relation_id = int(relation_ids[row].item()) % max(1, relation_feature_deltas.size(0))
            relation_delta = relation_feature_deltas[relation_id]
            common_dim = min(feature_tensor.size(1), weights.numel(), relation_delta.numel())
            feature_tensor = feature_tensor[:, :common_dim]
            weights = weights[:common_dim] + relation_delta[:common_dim]
        values = torch.matmul(feature_tensor, weights) + feature_bias
        calibrated[row].index_add_(0, ids, values)
    return calibrated


def apply_score_calibration(
        scores: torch.Tensor,
        batch: Dict,
        calibration: Dict = None,
        route_weights: torch.Tensor = None,
        neural_scores: torch.Tensor = None,
) -> torch.Tensor:
    if not calibration or not calibration.get("enabled", False):
        return scores
    source_deltas = calibration.get("source_deltas", {})
    values = torch.tensor(
        [float(source_deltas.get(name, 0.0)) for name, _, _ in SOURCE_FIELDS],
        dtype=torch.float32,
        device=scores.device,
    )
    relation_deltas = calibration.get("relation_source_deltas", [])
    if relation_deltas:
        relation_tensor = torch.tensor(relation_deltas, dtype=torch.float32, device=scores.device)
        relation_ids = batch["relations"].to(scores.device).long() % max(1, relation_tensor.size(0))
        values = values.unsqueeze(0) + relation_tensor.index_select(0, relation_ids)
    if calibration.get("decompose_evidence_scores", False) and neural_scores is not None:
        neural = neural_scores.float()
        evidence_residual = scores.float() - neural
        neural_scale = float(calibration.get("neural_scale", 1.0))
        evidence_scale = float(calibration.get("evidence_scale", 1.0))
        relation_evidence_deltas = calibration.get("relation_evidence_scale_deltas", [])
        if relation_evidence_deltas:
            relation_tensor = torch.tensor(relation_evidence_deltas, dtype=torch.float32, device=scores.device)
            relation_ids = batch["relations"].to(scores.device).long() % max(1, relation_tensor.size(0))
            raw_evidence_scale = torch.tensor(
                _inverse_softplus(evidence_scale),
                dtype=torch.float32,
                device=scores.device,
            )
            evidence_scale_tensor = F.softplus(
                raw_evidence_scale + relation_tensor.index_select(0, relation_ids)
            ).unsqueeze(-1)
            calibrated = neural * neural_scale + evidence_residual * evidence_scale_tensor
        else:
            calibrated = neural * neural_scale + evidence_residual * evidence_scale
    else:
        base_scale = float(calibration.get("base_scale", 1.0))
        calibrated = scores.float() * base_scale
    calibrated = calibrated + _add_sparse_source_bias(scores=scores.float(), batch=batch, source_values=values)
    feature_weights = calibration.get("prior_feature_weights", [])
    if feature_weights:
        feature_weight_tensor = torch.tensor(feature_weights, dtype=torch.float32, device=scores.device)
        feature_bias = torch.tensor(float(calibration.get("prior_feature_bias", 0.0)), dtype=torch.float32, device=scores.device)
        relation_feature_deltas = calibration.get("relation_prior_feature_deltas", [])
        relation_feature_tensor = (
            torch.tensor(relation_feature_deltas, dtype=torch.float32, device=scores.device)
            if relation_feature_deltas else None
        )
        calibrated = calibrated + _add_prior_feature_bias(
            scores=scores.float(),
            batch=batch,
            feature_weights=feature_weight_tensor,
            feature_bias=feature_bias,
            relation_feature_deltas=relation_feature_tensor,
            route_weights=route_weights,
            neural_scores=neural_scores.float() if neural_scores is not None else None,
            model_scores=scores.float(),
        )
    entity_bias = calibration.get("entity_bias", [])
    if entity_bias:
        entity_bias_tensor = torch.tensor(entity_bias, dtype=torch.float32, device=scores.device)
        if entity_bias_tensor.numel() == scores.size(1):
            calibrated = calibrated + entity_bias_tensor.unsqueeze(0)
    return calibrated


def _filtered_cross_entropy(scores: torch.Tensor, batch: Dict, filter_map: Dict) -> torch.Tensor:
    targets = batch["targets"].to(scores.device)
    subjects = batch["subjects"]
    relations = batch["relations"]
    times = batch["times"]
    masked = scores.clone()
    for row in range(scores.size(0)):
        target = int(targets[row].item())
        key = (int(subjects[row]), int(relations[row]), int(times[row]))
        for tail in filter_map.get(key, set()):
            tail = int(tail)
            if tail != target:
                masked[row, tail] = -1e9
    return F.cross_entropy(masked, targets)


def _filtered_pairwise_loss(
        scores: torch.Tensor,
        batch: Dict,
        filter_map: Dict,
        topk: int,
        margin: float,
) -> torch.Tensor:
    targets = batch["targets"].to(scores.device)
    subjects = batch["subjects"]
    relations = batch["relations"]
    times = batch["times"]
    masked = scores.clone()
    for row in range(scores.size(0)):
        target = int(targets[row].item())
        key = (int(subjects[row]), int(relations[row]), int(times[row]))
        for tail in filter_map.get(key, set()):
            tail = int(tail)
            if tail != target:
                masked[row, tail] = -torch.inf
        masked[row, target] = -torch.inf
    hard_k = min(max(1, int(topk)), max(1, scores.size(1) - 1))
    hard_negatives = torch.topk(masked, k=hard_k, dim=1).values
    positives = scores.gather(1, targets.view(-1, 1))
    return F.softplus(hard_negatives - positives + float(margin)).mean()


def _build_calibrated_scores(
        base_scores: torch.Tensor,
        batch: Dict,
        model,
        raw_base_scale: torch.nn.Parameter,
        raw_neural_scale: torch.nn.Parameter,
        raw_evidence_scale: torch.nn.Parameter,
        relation_evidence_scale_deltas: torch.nn.Parameter,
        source_deltas: torch.nn.Parameter,
        relation_source_deltas: torch.nn.Parameter,
        prior_feature_weights: torch.nn.Parameter,
        prior_feature_bias: torch.nn.Parameter,
        relation_prior_feature_deltas: torch.nn.Parameter,
        entity_bias: torch.nn.Parameter,
        use_relation_calibration: bool,
        use_prior_feature_calibration: bool,
        use_relation_feature_calibration: bool,
        use_score_decomposition: bool,
        use_relation_evidence_calibration: bool,
        use_entity_bias_calibration: bool,
        route_weights: torch.Tensor = None,
        neural_scores: torch.Tensor = None,
) -> torch.Tensor:
    relation_ids = batch["relations"].to(base_scores.device).long() % max(1, model.num_relations)
    if use_score_decomposition and neural_scores is not None:
        neural = neural_scores.to(device=base_scores.device, dtype=torch.float32)
        evidence_residual = base_scores.float() - neural
        neural_scale = F.softplus(raw_neural_scale)
        evidence_scale = F.softplus(raw_evidence_scale)
        if use_relation_evidence_calibration:
            evidence_scale = F.softplus(
                raw_evidence_scale + relation_evidence_scale_deltas.index_select(0, relation_ids)
            ).unsqueeze(-1)
        calibrated_scores = neural * neural_scale + evidence_residual * evidence_scale
    else:
        base_scale = F.softplus(raw_base_scale)
        calibrated_scores = base_scores * base_scale
    row_source_values = source_deltas
    if use_relation_calibration:
        row_source_values = source_deltas.unsqueeze(0) + relation_source_deltas.index_select(0, relation_ids)
    calibrated_scores = calibrated_scores + _add_sparse_source_bias(
        scores=base_scores,
        batch=batch,
        source_values=row_source_values,
    )
    if use_prior_feature_calibration:
        relation_feature_values = relation_prior_feature_deltas if use_relation_feature_calibration else None
        calibrated_scores = calibrated_scores + _add_prior_feature_bias(
            scores=base_scores,
            batch=batch,
            feature_weights=prior_feature_weights,
            feature_bias=prior_feature_bias,
            relation_feature_deltas=relation_feature_values,
            route_weights=route_weights,
            neural_scores=neural_scores,
            model_scores=base_scores.float(),
        )
    if use_entity_bias_calibration:
        calibrated_scores = calibrated_scores + entity_bias.unsqueeze(0)
    return calibrated_scores


def _calibration_state(parameters: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _restore_calibration_state(parameters: List[torch.nn.Parameter], state: List[torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter, value in zip(parameters, state):
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _selection_score(metrics: Dict[str, float], config) -> float:
    metric = str(getattr(config.model, "calibration_selection_metric", "MRR")).strip().lower()
    if metric in {"hits@1", "h@1", "hit@1", "hits1"}:
        return float(metrics.get("Hits@1", 0.0))
    if metric in {"mrr_hits1", "mrr+h1", "mrr_h1", "composite"}:
        mrr_weight = float(getattr(config.model, "calibration_selection_mrr_weight", 1.0))
        h1_weight = float(getattr(config.model, "calibration_selection_hits1_weight", 0.0))
        return mrr_weight * float(metrics.get("MRR", 0.0)) + h1_weight * float(metrics.get("Hits@1", 0.0))
    return float(metrics.get("MRR", 0.0))


def _cache_batch_for_calibration(batch: Dict) -> Dict:
    required_fields = {"subjects", "relations", "targets", "times"}
    for _, entity_key, weight_key in SOURCE_FIELDS:
        required_fields.add(entity_key)
        required_fields.add(weight_key)

    cached = {}
    for key in required_fields:
        if key not in batch:
            continue
        value = batch[key]
        if torch.is_tensor(value):
            cached[key] = value.detach().cpu()
        else:
            cached[key] = value
    return cached


def _cache_calibration_batches(
        model,
        valid_loader,
        use_amp: bool,
        max_batches: int = 0,
) -> List[Dict]:
    cached_batches = []
    # Cached model scores are constants during calibration, but they must remain
    # ordinary tensors because trainable calibration parameters use them in autograd.
    with torch.no_grad():
        cache_iter = valid_loader
        total_batches = len(valid_loader)
        if max_batches > 0:
            import itertools

            total_batches = min(total_batches, int(max_batches))
            cache_iter = itertools.islice(valid_loader, int(max_batches))
        for batch in tqdm(cache_iter, desc="cache-calibration-scores", leave=False, total=total_batches):
            with _amp_autocast(enabled=use_amp, dtype=_amp_dtype()):
                output = model(batch, return_explanations=False)
            route_weights = output.get("evidence_route_weights")
            cached_batches.append(
                {
                    "batch": _cache_batch_for_calibration(batch),
                    "base_scores": output["scores"].detach().float().cpu(),
                    "neural_scores": output.get("neural_scores").detach().float().cpu()
                    if output.get("neural_scores") is not None else None,
                    "route_weights": route_weights.detach().float().cpu() if route_weights is not None else None,
                }
            )
    return cached_batches


def _evaluate_calibration_metrics_cached(
        cached_batches: List[Dict],
        filter_map: Dict,
        device: torch.device,
        num_relations: int,
        raw_base_scale: torch.nn.Parameter,
        raw_neural_scale: torch.nn.Parameter,
        raw_evidence_scale: torch.nn.Parameter,
        relation_evidence_scale_deltas: torch.nn.Parameter,
        source_deltas: torch.nn.Parameter,
        relation_source_deltas: torch.nn.Parameter,
        prior_feature_weights: torch.nn.Parameter,
        prior_feature_bias: torch.nn.Parameter,
        relation_prior_feature_deltas: torch.nn.Parameter,
        entity_bias: torch.nn.Parameter,
        use_relation_calibration: bool,
        use_prior_feature_calibration: bool,
        use_relation_feature_calibration: bool,
        use_score_decomposition: bool,
        use_relation_evidence_calibration: bool,
        use_entity_bias_calibration: bool,
) -> Dict[str, float]:
    ranks = []
    with torch.inference_mode():
        for cached in cached_batches:
            batch = cached["batch"]
            base_scores = cached["base_scores"].to(device=device, dtype=torch.float32)
            neural_scores = cached["neural_scores"]
            neural_scores = neural_scores.to(device=device, dtype=torch.float32) if neural_scores is not None else None
            route_weights = cached["route_weights"]
            route_weights = route_weights.to(device=device, dtype=torch.float32) if route_weights is not None else None
            calibrated_scores = _build_calibrated_scores(
                base_scores=base_scores,
                batch=batch,
                model=type("CalibrationModelShape", (), {"num_relations": num_relations})(),
                raw_base_scale=raw_base_scale,
                raw_neural_scale=raw_neural_scale,
                raw_evidence_scale=raw_evidence_scale,
                relation_evidence_scale_deltas=relation_evidence_scale_deltas,
                source_deltas=source_deltas,
                relation_source_deltas=relation_source_deltas,
                prior_feature_weights=prior_feature_weights,
                prior_feature_bias=prior_feature_bias,
                relation_prior_feature_deltas=relation_prior_feature_deltas,
                entity_bias=entity_bias,
                use_relation_calibration=use_relation_calibration,
                use_prior_feature_calibration=use_prior_feature_calibration,
                use_relation_feature_calibration=use_relation_feature_calibration,
                use_score_decomposition=use_score_decomposition,
                use_relation_evidence_calibration=use_relation_evidence_calibration,
                use_entity_bias_calibration=use_entity_bias_calibration,
                route_weights=route_weights,
                neural_scores=neural_scores,
            )
            targets = batch["targets"]
            subjects = batch["subjects"]
            relations = batch["relations"]
            times = batch["times"]
            for row in range(calibrated_scores.size(0)):
                key = (int(subjects[row]), int(relations[row]), int(times[row]))
                rank = filtered_rank(
                    calibrated_scores[row],
                    int(targets[row]),
                    filter_map.get(key, set()),
                )
                ranks.append(rank)
    return aggregate_ranking_metrics(ranks)


def fit_score_calibration(
        model,
        config,
        valid_loader,
        filter_map: Dict,
        device: torch.device,
        use_amp: bool,
) -> Dict:
    if not getattr(config.model, "use_validation_calibration", False):
        return {"enabled": False}

    model.eval()
    raw_base_scale = torch.nn.Parameter(
        torch.tensor(_inverse_softplus(1.0), dtype=torch.float32, device=device)
    )
    raw_neural_scale = torch.nn.Parameter(
        torch.tensor(_inverse_softplus(1.0), dtype=torch.float32, device=device)
    )
    raw_evidence_scale = torch.nn.Parameter(
        torch.tensor(_inverse_softplus(1.0), dtype=torch.float32, device=device)
    )
    source_deltas = torch.nn.Parameter(torch.zeros(len(SOURCE_FIELDS), dtype=torch.float32, device=device))
    use_prior_feature_calibration = bool(getattr(config.model, "use_prior_feature_calibration", True))
    prior_feature_weights = torch.nn.Parameter(
        torch.zeros(PRIOR_FEATURE_DIM, dtype=torch.float32, device=device),
        requires_grad=use_prior_feature_calibration,
    )
    prior_feature_bias = torch.nn.Parameter(
        torch.zeros((), dtype=torch.float32, device=device),
        requires_grad=use_prior_feature_calibration,
    )
    use_relation_calibration = bool(getattr(config.model, "use_relation_calibration", True))
    relation_source_deltas = torch.nn.Parameter(
        torch.zeros(model.num_relations, len(SOURCE_FIELDS), dtype=torch.float32, device=device),
        requires_grad=use_relation_calibration,
    )
    use_relation_feature_calibration = bool(getattr(config.model, "use_relation_feature_calibration", True))
    relation_prior_feature_deltas = torch.nn.Parameter(
        torch.zeros(model.num_relations, PRIOR_FEATURE_DIM, dtype=torch.float32, device=device),
        requires_grad=use_prior_feature_calibration and use_relation_feature_calibration,
    )
    use_entity_bias_calibration = bool(getattr(config.model, "use_entity_bias_calibration", False))
    entity_bias = torch.nn.Parameter(
        torch.zeros(model.num_entities, dtype=torch.float32, device=device),
        requires_grad=use_entity_bias_calibration,
    )
    use_score_decomposition = bool(getattr(config.model, "calibration_decompose_evidence_scores", False))
    raw_base_scale.requires_grad_(not use_score_decomposition)
    raw_neural_scale.requires_grad_(use_score_decomposition)
    raw_evidence_scale.requires_grad_(use_score_decomposition)
    use_relation_evidence_calibration = bool(getattr(config.model, "use_relation_evidence_calibration", False))
    relation_evidence_scale_deltas = torch.nn.Parameter(
        torch.zeros(model.num_relations, dtype=torch.float32, device=device),
        requires_grad=use_score_decomposition and use_relation_evidence_calibration,
    )
    optimizer = torch.optim.AdamW(
        [
            parameter for parameter in [
                raw_base_scale,
                raw_neural_scale,
                raw_evidence_scale,
                relation_evidence_scale_deltas,
                source_deltas,
                relation_source_deltas,
                prior_feature_weights,
                prior_feature_bias,
                relation_prior_feature_deltas,
                entity_bias,
            ]
            if parameter.requires_grad
        ],
        lr=float(config.model.calibration_learning_rate),
        weight_decay=0.0,
    )
    max_batches = int(getattr(config.model, "calibration_batch_limit", 0))
    l2_weight = float(getattr(config.model, "calibration_l2", 0.0))
    relation_l2_weight = float(getattr(config.model, "calibration_relation_l2", 0.0))
    feature_l2_weight = float(getattr(config.model, "calibration_feature_l2", 0.0))
    relation_feature_l2_weight = float(getattr(config.model, "calibration_relation_feature_l2", 0.0))
    pairwise_weight = float(getattr(config.model, "calibration_pairwise_weight", 0.0))
    pairwise_margin = float(getattr(config.model, "calibration_pairwise_margin", 0.1))
    pairwise_topk = int(getattr(config.model, "calibration_pairwise_topk", 8))
    select_by_mrr = bool(getattr(config.model, "calibration_select_by_mrr", True))
    selection_metric = str(getattr(config.model, "calibration_selection_metric", "MRR"))
    nonnegative_source_deltas = bool(getattr(config.model, "calibration_nonnegative_source_deltas", False))
    evidence_scale_l2_weight = float(getattr(config.model, "calibration_evidence_scale_l2", 0.0))
    relation_evidence_l2_weight = float(getattr(config.model, "calibration_relation_evidence_l2", 0.0))
    entity_bias_l2_weight = float(getattr(config.model, "calibration_entity_bias_l2", 0.0))
    epochs = max(1, int(getattr(config.model, "calibration_epochs", 1)))
    original_eval_path_topk = getattr(model, "eval_path_topk", 0)
    original_path_weight = getattr(model, "path_weight", 0.0)
    model.eval_path_topk = 0
    model.path_weight = 0.0

    history = []
    calibration_parameters = [
        raw_base_scale,
        raw_neural_scale,
        raw_evidence_scale,
        relation_evidence_scale_deltas,
        source_deltas,
        relation_source_deltas,
        prior_feature_weights,
        prior_feature_bias,
        relation_prior_feature_deltas,
        entity_bias,
    ]
    best_state = _calibration_state(calibration_parameters)
    best_metric = -1.0
    best_metrics = {}
    try:
        cached_batches = _cache_calibration_batches(
            model=model,
            valid_loader=valid_loader,
            use_amp=use_amp,
            max_batches=max_batches,
        )
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_pairwise_loss = 0.0
            steps = 0
            for batch_idx, cached in enumerate(tqdm(cached_batches, desc=f"calibrate {epoch}", leave=False), start=1):
                if max_batches > 0 and batch_idx > max_batches:
                    break
                batch = cached["batch"]
                base_scores = cached["base_scores"].to(device=device, dtype=torch.float32)
                neural_scores = cached["neural_scores"]
                neural_scores = neural_scores.to(device=device, dtype=torch.float32) if neural_scores is not None else None
                route_weights = cached["route_weights"]
                route_weights = route_weights.to(device=device, dtype=torch.float32) if route_weights is not None else None
                calibrated_scores = _build_calibrated_scores(
                    base_scores=base_scores,
                    batch=batch,
                    model=model,
                    raw_base_scale=raw_base_scale,
                    raw_neural_scale=raw_neural_scale,
                    raw_evidence_scale=raw_evidence_scale,
                    relation_evidence_scale_deltas=relation_evidence_scale_deltas,
                    source_deltas=source_deltas,
                    relation_source_deltas=relation_source_deltas,
                    prior_feature_weights=prior_feature_weights,
                    prior_feature_bias=prior_feature_bias,
                    relation_prior_feature_deltas=relation_prior_feature_deltas,
                    entity_bias=entity_bias,
                    use_relation_calibration=use_relation_calibration,
                    use_prior_feature_calibration=use_prior_feature_calibration,
                    use_relation_feature_calibration=use_relation_feature_calibration,
                    use_score_decomposition=use_score_decomposition,
                    use_relation_evidence_calibration=use_relation_evidence_calibration,
                    use_entity_bias_calibration=use_entity_bias_calibration,
                    route_weights=route_weights,
                    neural_scores=neural_scores,
                )
                ce_loss = _filtered_cross_entropy(calibrated_scores, batch=batch, filter_map=filter_map)
                pairwise_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                if pairwise_weight > 0:
                    pairwise_loss = _filtered_pairwise_loss(
                        calibrated_scores,
                        batch=batch,
                        filter_map=filter_map,
                        topk=pairwise_topk,
                        margin=pairwise_margin,
                    )
                loss = ce_loss + pairwise_weight * pairwise_loss
                if l2_weight > 0:
                    loss = loss + l2_weight * (
                        torch.square(F.softplus(raw_base_scale) - 1.0) + torch.sum(torch.square(source_deltas))
                    )
                if use_score_decomposition and evidence_scale_l2_weight > 0:
                    loss = loss + evidence_scale_l2_weight * (
                        torch.square(F.softplus(raw_neural_scale) - 1.0)
                        + torch.square(F.softplus(raw_evidence_scale) - 1.0)
                    )
                if (
                        use_score_decomposition
                        and use_relation_evidence_calibration
                        and relation_evidence_l2_weight > 0
                ):
                    loss = loss + relation_evidence_l2_weight * torch.mean(
                        torch.square(relation_evidence_scale_deltas)
                    )
                if use_relation_calibration and relation_l2_weight > 0:
                    loss = loss + relation_l2_weight * torch.mean(torch.square(relation_source_deltas))
                if use_prior_feature_calibration and feature_l2_weight > 0:
                    loss = loss + feature_l2_weight * (
                        torch.mean(torch.square(prior_feature_weights)) + torch.square(prior_feature_bias)
                    )
                if (
                        use_prior_feature_calibration
                        and use_relation_feature_calibration
                        and relation_feature_l2_weight > 0
                ):
                    loss = loss + relation_feature_l2_weight * torch.mean(torch.square(relation_prior_feature_deltas))
                if use_entity_bias_calibration and entity_bias_l2_weight > 0:
                    centered_entity_bias = entity_bias - entity_bias.mean()
                    loss = loss + entity_bias_l2_weight * torch.mean(torch.square(centered_entity_bias))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if nonnegative_source_deltas:
                    with torch.no_grad():
                        source_deltas.clamp_(min=0.0)
                if use_entity_bias_calibration:
                    with torch.no_grad():
                        entity_bias.sub_(entity_bias.mean())
                epoch_loss += float(loss.item())
                epoch_ce_loss += float(ce_loss.item())
                epoch_pairwise_loss += float(pairwise_loss.item())
                steps += 1
            record = {
                "epoch": epoch,
                "loss": epoch_loss / max(1, steps),
                "ce_loss": epoch_ce_loss / max(1, steps),
                "pairwise_loss": epoch_pairwise_loss / max(1, steps),
                "steps": steps,
            }
            if select_by_mrr:
                metrics = _evaluate_calibration_metrics_cached(
                    cached_batches=cached_batches,
                    filter_map=filter_map,
                    device=device,
                    num_relations=model.num_relations,
                    raw_base_scale=raw_base_scale,
                    raw_neural_scale=raw_neural_scale,
                    raw_evidence_scale=raw_evidence_scale,
                    relation_evidence_scale_deltas=relation_evidence_scale_deltas,
                    source_deltas=source_deltas,
                    relation_source_deltas=relation_source_deltas,
                    prior_feature_weights=prior_feature_weights,
                    prior_feature_bias=prior_feature_bias,
                    relation_prior_feature_deltas=relation_prior_feature_deltas,
                    entity_bias=entity_bias,
                    use_relation_calibration=use_relation_calibration,
                    use_prior_feature_calibration=use_prior_feature_calibration,
                    use_relation_feature_calibration=use_relation_feature_calibration,
                    use_score_decomposition=use_score_decomposition,
                    use_relation_evidence_calibration=use_relation_evidence_calibration,
                    use_entity_bias_calibration=use_entity_bias_calibration,
                )
                record.update({f"valid_{key}": value for key, value in metrics.items()})
                selected_metric = _selection_score(metrics, config)
                record["valid_selection_score"] = selected_metric
                record["valid_selection_metric"] = selection_metric
                if selected_metric > best_metric:
                    best_metric = selected_metric
                    best_metrics = metrics
                    best_state = _calibration_state(calibration_parameters)
            history.append(record)
        if select_by_mrr:
            _restore_calibration_state(calibration_parameters, best_state)
    finally:
        model.eval_path_topk = original_eval_path_topk
        model.path_weight = original_path_weight

    calibration = {
        "enabled": True,
        "base_scale": float(F.softplus(raw_base_scale).detach().cpu().item()),
        "decompose_evidence_scores": use_score_decomposition,
        "neural_scale": float(F.softplus(raw_neural_scale).detach().cpu().item())
        if use_score_decomposition else 1.0,
        "evidence_scale": float(F.softplus(raw_evidence_scale).detach().cpu().item())
        if use_score_decomposition else 1.0,
        "relation_evidence_scale_deltas": relation_evidence_scale_deltas.detach().cpu().tolist()
        if use_score_decomposition and use_relation_evidence_calibration else [],
        "source_deltas": {
            name: float(value)
            for (name, _, _), value in zip(SOURCE_FIELDS, source_deltas.detach().cpu().tolist())
        },
        "relation_source_deltas": relation_source_deltas.detach().cpu().tolist()
        if use_relation_calibration else [],
        "prior_feature_schema": "v3_source_rank_and_model_rank",
        "prior_feature_dim": PRIOR_FEATURE_DIM if use_prior_feature_calibration else 0,
        "prior_feature_weights": prior_feature_weights.detach().cpu().tolist()
        if use_prior_feature_calibration else [],
        "prior_feature_bias": float(prior_feature_bias.detach().cpu().item())
        if use_prior_feature_calibration else 0.0,
        "relation_prior_feature_deltas": relation_prior_feature_deltas.detach().cpu().tolist()
        if use_prior_feature_calibration and use_relation_feature_calibration else [],
        "entity_bias": entity_bias.detach().cpu().tolist()
        if use_entity_bias_calibration else [],
        "history": history,
        "selection": {
            "select_by_mrr": select_by_mrr,
            "selection_metric": selection_metric,
            "best_selection_score": best_metric,
            "best_valid_metrics": best_metrics,
            "nonnegative_source_deltas": nonnegative_source_deltas,
            "pairwise_weight": pairwise_weight,
            "pairwise_margin": pairwise_margin,
            "pairwise_topk": pairwise_topk,
            "decompose_evidence_scores": use_score_decomposition,
            "neural_scale": float(F.softplus(raw_neural_scale).detach().cpu().item())
            if use_score_decomposition else 1.0,
            "evidence_scale": float(F.softplus(raw_evidence_scale).detach().cpu().item())
            if use_score_decomposition else 1.0,
            "use_relation_evidence_calibration": use_relation_evidence_calibration,
            "use_entity_bias_calibration": use_entity_bias_calibration,
            "entity_bias_l2": entity_bias_l2_weight,
        },
        "note": "Learned on the validation split only; no test labels are used.",
    }
    print(
        {
            "stage": "score_calibration",
            "enabled": True,
            "base_scale": calibration["base_scale"],
            "neural_scale": calibration["neural_scale"],
            "evidence_scale": calibration["evidence_scale"],
            "source_deltas": calibration["source_deltas"],
            "prior_feature_schema": calibration["prior_feature_schema"],
            "prior_feature_dim": calibration["prior_feature_dim"],
            "entity_bias_summary": {
                "enabled": use_entity_bias_calibration,
                "min": float(entity_bias.detach().min().cpu().item()) if use_entity_bias_calibration else 0.0,
                "max": float(entity_bias.detach().max().cpu().item()) if use_entity_bias_calibration else 0.0,
                "std": float(entity_bias.detach().std().cpu().item()) if use_entity_bias_calibration else 0.0,
            },
            "history": history,
            "selection": calibration["selection"],
            "note": calibration["note"],
        },
        flush=True,
    )
    return calibration


def save_score_calibration(calibration: Dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(calibration, fp, ensure_ascii=False, indent=2)


def load_score_calibration(checkpoint_path: str) -> Dict:
    path = Path(checkpoint_path).with_name("score_calibration.json")
    if not path.exists():
        return {"enabled": False}
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return payload if isinstance(payload, dict) else {"enabled": False}
