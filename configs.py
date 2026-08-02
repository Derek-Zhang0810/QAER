import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class DataConfig:
    dataset_dir: str = "./data/ICEWS18"
    preprocess_dir: str = ""
    history_len: int = 12
    history_window: int = 96
    max_hops: int = 2
    max_nodes: int = 24
    max_path_len: int = 3
    max_paths_per_candidate: int = 4
    path_prior_topk: int = 64
    path_prior_max_branches: int = 8
    path_prior_state_limit: int = 32
    path_prior_length_decay: float = 0.8
    path_prior_relation_bonus: float = 0.5
    path_prior_use_reverse: bool = True
    train_heavy_evidence_rate: float = 0.05
    dynamic_heavy_evidence: bool = True
    valid_heavy_evidence: bool = True
    eval_heavy_evidence: bool = True
    local_graph_cache_size: int = 50000
    path_prior_cache_size: int = 50000
    prior_table_cache_size: int = 0
    strict_eval_history: bool = False
    use_inverse_train: bool = True
    inverse_train_mode: str = "sampled"
    prior_window: int = 168
    long_prior_window: int = 720
    long_prior_time_decay: float = 0.006
    long_prior_mix: float = 0.35
    copy_topk: int = 96
    local_prior_topk: int = 96
    prior_topk: int = 128
    subject_prior_topk: int = 96
    global_prior_topk: int = 96
    relation_transfer_topk: int = 96
    relation_transfer_min_sim: float = 0.08
    relation_transfer_time_decay: float = 0.03
    concurrent_subject_topk: int = 16
    concurrent_time_topk: int = 32
    prior_time_decay: float = 0.03
    recent_train_weight: float = 0.6
    recent_train_power: float = 1.25


@dataclass
class SemanticConfig:
    backend: str = "transformer"
    feature_dim: int = 256
    transformer_name: str = "Qwen/Qwen3-Embedding-0.6B"
    transformer_batch_size: int = 16
    max_length: int = 128
    pooling: str = "last"
    normalize: bool = True
    precision: str = "auto"
    trust_remote_code: bool = True
    local_files_only: bool = False
    cache_features: bool = True
    force_rebuild_cache: bool = False
    cache_dir: str = ""


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_rgcn_layers: int = 1
    dropout: float = 0.16
    shortlist_size: int = 48
    train_context_batch_limit: int = 16
    train_path_topk: int = 0
    train_path_batch_limit: int = 0
    eval_path_topk: int = 0
    time_decay: float = 0.05
    logic_temperature: float = 0.8
    alignment_weight: float = 0.03
    ranking_weight: float = 0.30
    ranking_margin: float = 0.75
    ranking_topk: int = 8
    label_smoothing: float = 0.04
    context_score_weight: float = 1.0
    triple_score_weight: float = 0.45
    distmult_score_weight: float = 0.0
    transe_score_weight: float = 0.0
    complex_score_weight: float = 0.0
    conv_score_weight: float = 0.0
    conv_num_channels: int = 32
    conv_kernel_size: int = 3
    conv_dropout: float = 0.2
    normalize_neural_scores: bool = True
    neural_score_scale: float = 1.0
    use_local_graph_context: bool = True
    use_subject_history_context: bool = True
    use_history_transformer: bool = False
    history_transformer_layers: int = 1
    history_transformer_heads: int = 4
    use_concurrent_query_context: bool = False
    concurrent_query_weight: float = 0.35
    concurrent_query_temperature: float = 0.7
    history_attention_temperature: float = 0.7
    history_copy_weight: float = 3.0
    history_copy_decay: float = 0.03
    relation_transfer_prior_weight: float = 0.0
    relation_transfer_copy_weight: float = 0.0
    relation_transfer_copy_temperature: float = 0.35
    relation_transfer_copy_history_len: int = 8
    relation_prior_weight: float = 0.95
    local_prior_weight: float = 4.0
    subject_prior_weight: float = 1.65
    global_prior_weight: float = 0.5
    path_prior_weight: float = 2.45
    learned_prior_weight: float = 1.0
    evidence_candidate_gating: bool = False
    evidence_gate_sources: str = "copy,local,subject,path"
    evidence_gate_threshold: float = 0.0
    evidence_gate_mode: str = "hard"
    evidence_gate_penalty: float = -2.0
    evidence_gate_boost: float = 0.0
    evidence_gate_band_width: float = 0.0
    evidence_gate_min_sources: int = 1
    evidence_gate_support_threshold: float = 0.0
    evidence_gate_source_weights: str = ""
    evidence_gate_neural_topk: int = 0
    evidence_gate_value: float = -1000000000.0
    evidence_gate_include_target_train: bool = True
    evidence_rank_fusion_weight: float = 0.0
    evidence_rank_fusion_k: float = 20.0
    evidence_router_temperature: float = 0.7
    use_evidence_router: bool = True
    evidence_route_floor: float = 0.2
    use_validation_calibration: bool = True
    calibration_epochs: int = 1
    calibration_learning_rate: float = 0.05
    calibration_l2: float = 0.01
    use_relation_calibration: bool = True
    calibration_relation_l2: float = 0.05
    use_prior_feature_calibration: bool = True
    calibration_feature_l2: float = 0.02
    use_relation_feature_calibration: bool = True
    calibration_relation_feature_l2: float = 0.05
    calibration_batch_limit: int = 0
    calibration_pairwise_weight: float = 0.0
    calibration_pairwise_margin: float = 0.1
    calibration_pairwise_topk: int = 8
    calibration_select_by_mrr: bool = True
    calibration_selection_metric: str = "MRR"
    calibration_selection_mrr_weight: float = 1.0
    calibration_selection_hits1_weight: float = 0.0
    calibration_nonnegative_source_deltas: bool = False
    calibration_decompose_evidence_scores: bool = False
    calibration_evidence_scale_l2: float = 0.02
    use_relation_evidence_calibration: bool = False
    calibration_relation_evidence_l2: float = 0.05
    use_entity_bias_calibration: bool = False
    calibration_entity_bias_l2: float = 0.08
    coverage_route_weight: float = 1.0
    relation_evidence_weight: float = 0.35
    evidence_consensus_weight: float = 0.0
    evidence_consensus_min_sources: int = 2
    use_evidence_feature_reranker: bool = False
    evidence_feature_rerank_weight: float = 1.0
    evidence_feature_rerank_topk: int = 384
    evidence_residual_scale: float = 1.0
    evidence_neural_agreement_weight: float = 0.0
    evidence_neural_agreement_floor: float = 0.35
    evidence_neural_agreement_temperature: float = 1.0
    evidence_neural_agreement_threshold: float = -0.2
    evidence_loss_weight: float = 0.10
    evidence_margin: float = 0.25
    route_supervision_weight: float = 0.06
    route_supervision_temperature: float = 0.5
    route_entropy_weight: float = 0.002
    path_weight: float = 0.0
    bidirectional_paths: bool = True


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 20
    min_epochs: int = 2
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    batch_size: int = 512
    eval_batch_size: int = 512
    learning_rate: float = 7e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 5e-5
    gradient_clip: float = 1.0
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    warm_prior_cache: bool = True
    use_amp: bool = True
    eval_every: int = 1
    quick_eval_samples: int = 0
    full_eval_every: int = 1
    heavy_evidence_start_rate: float = 0.025
    heavy_evidence_warmup_epochs: int = 4
    context_start_limit: int = 8
    context_warmup_epochs: int = 3
    evidence_warmup_epochs: int = 3
    route_warmup_epochs: int = 3
    device: str = "cuda"
    use_ema: bool = True
    ema_decay: float = 0.995


@dataclass
class ExplainConfig:
    top_k_predictions: int = 10
    top_k_paths: int = 3


@dataclass
class ExperimentConfig:
    method_name: str = "QAER"
    data: DataConfig = field(default_factory=DataConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    output_dir: str = "./outputs/QAER"


def _deep_update_dataclass(instance: Any, update_dict: Dict[str, Any]) -> Any:
    for key, value in update_dict.items():
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _deep_update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(config_path: str = "") -> ExperimentConfig:
    config = ExperimentConfig()
    if config_path:
        with open(config_path, "r", encoding="utf-8-sig") as fp:
            payload = json.load(fp)
        _deep_update_dataclass(config, payload)
    return config


def save_config(config: ExperimentConfig, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(asdict(config), fp, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="QAER configuration helper")
    parser.add_argument("--load", type=str, default="", help="Optional config JSON path")
    parser.add_argument("--dump", type=str, default="", help="Write resolved config to JSON")
    args = parser.parse_args()

    config = load_config(args.load)
    if args.dump:
        save_config(config, args.dump)
        print(f"Saved config to {args.dump}")
    else:
        print(json.dumps(asdict(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
