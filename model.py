import argparse
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal_graph import LocalSubgraph, TemporalEdge, TemporalPath, enumerate_temporal_paths, temporal_path_to_text
from utils import exp_time_decay


class SinusoidalTimeEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, deltas: torch.Tensor) -> torch.Tensor:
        half_dim = self.hidden_dim // 2
        basis = torch.exp(
            torch.arange(half_dim, device=deltas.device, dtype=torch.float32)
            * -(math.log(10000.0) / max(1, half_dim - 1))
        )
        angles = deltas.float().unsqueeze(-1) * basis.unsqueeze(0)
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if encoded.size(-1) < self.hidden_dim:
            encoded = F.pad(encoded, (0, self.hidden_dim - encoded.size(-1)))
        return encoded


class RelationalGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int, dropout: float, apply_activation: bool):
        super().__init__()
        self.rel_embeddings = nn.Embedding(num_relations * 2, hidden_dim)
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.message_linear = nn.Linear(hidden_dim, hidden_dim)
        self.apply_activation = apply_activation
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)

    def forward(self, graph: LocalSubgraph, node_features: torch.Tensor) -> torch.Tensor:
        if graph.num_edges() == 0:
            out = self.self_linear(node_features)
            if self.apply_activation:
                out = F.relu(out)
            return self.dropout(out)

        src_nodes, dst_nodes = graph.edges()
        edge_types = graph.edata["type"]
        rel_bias = self.rel_embeddings(edge_types).to(dtype=node_features.dtype)
        messages = node_features[src_nodes] + rel_bias
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, dst_nodes, messages)
        aggregated = aggregated * graph.ndata["norm"].to(dtype=aggregated.dtype)
        out = self.self_linear(node_features) + self.message_linear(aggregated)
        if self.apply_activation:
            out = F.relu(out)
        return self.dropout(out)


class LocalRGCNEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RelationalGraphLayer(
                    hidden_dim=hidden_dim,
                    num_relations=num_relations,
                    dropout=dropout,
                    apply_activation=idx < num_layers - 1,
                )
                for idx in range(num_layers)
            ]
        )

    def forward(self, graph: LocalSubgraph, node_features: torch.Tensor) -> torch.Tensor:
        h = node_features
        for layer in self.layers:
            h = layer(graph, h)
        return h


class TemporalLogicPathScorer(nn.Module):
    def __init__(self, hidden_dim: int, time_decay: float, logic_temperature: float):
        super().__init__()
        self.time_decay = time_decay
        self.logic_temperature = logic_temperature
        self.path_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
            self,
            query_relation_embedding: torch.Tensor,
            candidate_embedding: torch.Tensor,
            source_embedding: torch.Tensor,
            relation_table: torch.Tensor,
            entity_table: torch.Tensor,
            query_time: int,
            paths: List[TemporalPath],
    ) -> Tuple[torch.Tensor, List[Dict]]:
        if not paths:
            return torch.tensor(0.0, device=query_relation_embedding.device), []

        raw_scores = []
        evidence_payload = []
        for path in paths:
            rel_ids = torch.tensor(path.relations, dtype=torch.long, device=query_relation_embedding.device)
            node_ids = torch.tensor(path.nodes[1:], dtype=torch.long, device=query_relation_embedding.device)
            rel_repr = relation_table[rel_ids].mean(dim=0)
            node_repr = entity_table[node_ids].mean(dim=0) if len(node_ids) > 0 else source_embedding
            temporal_decay = exp_time_decay(query_time - max(path.timestamps), self.time_decay)
            monotonicity = 1.0 if path.timestamps == sorted(path.timestamps) else 0.5
            logic_bias = math.log(max(temporal_decay * monotonicity, 1e-8))
            neural_input = torch.cat(
                [query_relation_embedding, candidate_embedding + node_repr, source_embedding + rel_repr], dim=-1)
            neural_score = self.path_mlp(neural_input).squeeze(-1)
            full_score = neural_score + logic_bias / self.logic_temperature
            raw_scores.append(full_score)
            evidence_payload.append({"path": path, "score_tensor": full_score})

        stacked = torch.stack(raw_scores)
        attention = F.softmax(stacked, dim=0)
        final_score = torch.sum(attention * stacked)
        ranked = sorted(
            [
                {
                    "score": float((attention[idx] * stacked[idx]).detach().cpu().item()),
                    "path": evidence_payload[idx]["path"],
                }
                for idx in range(len(evidence_payload))
            ],
            key=lambda item: item["score"],
            reverse=True,
        )
        return final_score, ranked


class ConvTransEDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = max(0, int(kernel_size) // 2)
        self.input_dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(2, int(num_channels), kernel_size=int(kernel_size), padding=padding)
        self.feature_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(int(num_channels) * hidden_dim, hidden_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
            self,
            subject_tensor: torch.Tensor,
            relation_tensor: torch.Tensor,
            entity_table: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.stack([subject_tensor, relation_tensor], dim=1)
        hidden = self.input_dropout(stacked)
        hidden = F.relu(self.conv(hidden))
        if hidden.size(-1) != subject_tensor.size(-1):
            hidden = hidden[..., :subject_tensor.size(-1)]
            if hidden.size(-1) < subject_tensor.size(-1):
                hidden = F.pad(hidden, (0, subject_tensor.size(-1) - hidden.size(-1)))
        hidden = self.feature_dropout(hidden)
        hidden = hidden.reshape(hidden.size(0), -1)
        query = self.output_dropout(F.relu(self.proj(hidden)))
        return torch.matmul(query, entity_table.t()) / math.sqrt(max(1, entity_table.size(-1)))


class SingleStepTemporalReasoner(nn.Module):
    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            entity_semantic_features: torch.Tensor,
            relation_semantic_features: torch.Tensor,
            hidden_dim: int = 256,
            num_rgcn_layers: int = 2,
            dropout: float = 0.2,
            shortlist_size: int = 32,
            train_context_batch_limit: int = 0,
            train_path_topk: int = 8,
            train_path_batch_limit: int = 0,
            eval_path_topk: int = 24,
            max_path_len: int = 3,
            max_paths_per_candidate: int = 8,
            time_decay: float = 0.08,
            logic_temperature: float = 0.7,
            alignment_weight: float = 0.05,
            ranking_weight: float = 0.0,
            ranking_margin: float = 1.0,
            ranking_topk: int = 1,
            label_smoothing: float = 0.0,
            context_score_weight: float = 1.0,
            triple_score_weight: float = 0.0,
            distmult_score_weight: float = 0.0,
            transe_score_weight: float = 0.0,
            complex_score_weight: float = 0.0,
            conv_score_weight: float = 0.0,
            conv_num_channels: int = 32,
            conv_kernel_size: int = 3,
            conv_dropout: float = 0.2,
            normalize_neural_scores: bool = False,
            neural_score_scale: float = 1.0,
            use_local_graph_context: bool = True,
            use_subject_history_context: bool = True,
            use_history_transformer: bool = False,
            history_transformer_layers: int = 1,
            history_transformer_heads: int = 4,
            use_concurrent_query_context: bool = False,
            concurrent_query_weight: float = 0.35,
            concurrent_query_temperature: float = 0.7,
            history_attention_temperature: float = 0.7,
            history_copy_weight: float = 0.0,
            history_copy_decay: float = 0.05,
            relation_transfer_prior_weight: float = 0.0,
            relation_transfer_copy_weight: float = 0.0,
            relation_transfer_copy_temperature: float = 0.35,
            relation_transfer_copy_history_len: int = 8,
            relation_prior_weight: float = 0.0,
            local_prior_weight: float = 0.0,
            subject_prior_weight: float = 0.0,
            global_prior_weight: float = 0.0,
            path_prior_weight: float = 0.0,
            learned_prior_weight: float = 0.0,
            evidence_candidate_gating: bool = False,
            evidence_gate_sources: str = "copy,local,subject,path",
            evidence_gate_threshold: float = 0.0,
            evidence_gate_mode: str = "hard",
            evidence_gate_penalty: float = -2.0,
            evidence_gate_boost: float = 0.0,
            evidence_gate_band_width: float = 0.0,
            evidence_gate_min_sources: int = 1,
            evidence_gate_support_threshold: float = 0.0,
            evidence_gate_source_weights: str = "",
            evidence_gate_neural_topk: int = 0,
            evidence_gate_value: float = -1000000000.0,
            evidence_gate_include_target_train: bool = True,
            evidence_rank_fusion_weight: float = 0.0,
            evidence_rank_fusion_k: float = 20.0,
            evidence_router_temperature: float = 0.7,
            use_evidence_router: bool = True,
            evidence_route_floor: float = 0.0,
            coverage_route_weight: float = 0.0,
            relation_evidence_weight: float = 0.0,
            evidence_consensus_weight: float = 0.0,
            evidence_consensus_min_sources: int = 2,
            use_evidence_feature_reranker: bool = False,
            evidence_feature_rerank_weight: float = 1.0,
            evidence_feature_rerank_topk: int = 384,
            evidence_residual_scale: float = 1.0,
            evidence_neural_agreement_weight: float = 0.0,
            evidence_neural_agreement_floor: float = 0.35,
            evidence_neural_agreement_temperature: float = 1.0,
            evidence_neural_agreement_threshold: float = -0.2,
            evidence_loss_weight: float = 0.0,
            evidence_margin: float = 0.25,
            route_supervision_weight: float = 0.0,
            route_supervision_temperature: float = 0.5,
            route_entropy_weight: float = 0.0,
            path_weight: float = 1.0,
            bidirectional_paths: bool = False,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.shortlist_size = shortlist_size
        self.train_context_batch_limit = train_context_batch_limit
        self.train_path_topk = train_path_topk
        self.train_path_batch_limit = train_path_batch_limit
        self.eval_path_topk = eval_path_topk
        self.max_path_len = max_path_len
        self.max_paths_per_candidate = max_paths_per_candidate
        self.alignment_weight = alignment_weight
        self.ranking_weight = ranking_weight
        self.ranking_margin = ranking_margin
        self.ranking_topk = max(1, int(ranking_topk))
        self.label_smoothing = max(0.0, float(label_smoothing))
        self.context_score_weight = context_score_weight
        self.triple_score_weight = triple_score_weight
        self.distmult_score_weight = float(distmult_score_weight)
        self.transe_score_weight = float(transe_score_weight)
        self.complex_score_weight = float(complex_score_weight)
        self.conv_score_weight = float(conv_score_weight)
        self.normalize_neural_scores = bool(normalize_neural_scores)
        self.neural_score_scale = float(neural_score_scale)
        self.use_local_graph_context = bool(use_local_graph_context)
        self.use_subject_history_context = bool(use_subject_history_context)
        self.use_history_transformer = bool(use_history_transformer)
        self.use_concurrent_query_context = bool(use_concurrent_query_context)
        self.concurrent_query_weight = float(concurrent_query_weight)
        self.concurrent_query_temperature = max(float(concurrent_query_temperature), 1e-3)
        self.history_attention_temperature = max(float(history_attention_temperature), 1e-3)
        self.history_copy_weight = history_copy_weight
        self.history_copy_decay = history_copy_decay
        self.relation_transfer_prior_weight = float(relation_transfer_prior_weight)
        self.relation_transfer_copy_weight = float(relation_transfer_copy_weight)
        self.relation_transfer_copy_temperature = max(float(relation_transfer_copy_temperature), 1e-3)
        self.relation_transfer_copy_history_len = max(0, int(relation_transfer_copy_history_len))
        self.relation_prior_weight = relation_prior_weight
        self.local_prior_weight = local_prior_weight
        self.subject_prior_weight = subject_prior_weight
        self.global_prior_weight = global_prior_weight
        self.path_prior_weight = path_prior_weight
        self.learned_prior_weight = learned_prior_weight
        self.evidence_candidate_gating = bool(evidence_candidate_gating)
        self.evidence_gate_sources = {
            source.strip().lower()
            for source in str(evidence_gate_sources).split(",")
            if source.strip()
        }
        self.evidence_gate_threshold = float(evidence_gate_threshold)
        self.evidence_gate_mode = str(evidence_gate_mode).strip().lower()
        self.evidence_gate_penalty = float(evidence_gate_penalty)
        self.evidence_gate_boost = float(evidence_gate_boost)
        self.evidence_gate_band_width = float(evidence_gate_band_width)
        self.evidence_gate_min_sources = max(1, int(evidence_gate_min_sources))
        self.evidence_gate_support_threshold = max(0.0, float(evidence_gate_support_threshold))
        self.evidence_gate_neural_topk = int(evidence_gate_neural_topk)
        self.evidence_gate_value = float(evidence_gate_value)
        self.evidence_gate_include_target_train = bool(evidence_gate_include_target_train)
        self.evidence_rank_fusion_weight = float(evidence_rank_fusion_weight)
        self.evidence_rank_fusion_k = max(1e-3, float(evidence_rank_fusion_k))
        self.evidence_router_temperature = max(float(evidence_router_temperature), 1e-3)
        self.use_evidence_router = bool(use_evidence_router)
        self.evidence_route_floor = max(0.0, float(evidence_route_floor))
        self.coverage_route_weight = coverage_route_weight
        self.relation_evidence_weight = relation_evidence_weight
        self.evidence_consensus_weight = float(evidence_consensus_weight)
        self.evidence_consensus_min_sources = max(1, int(evidence_consensus_min_sources))
        self.use_evidence_feature_reranker = bool(use_evidence_feature_reranker)
        self.evidence_feature_rerank_weight = float(evidence_feature_rerank_weight)
        self.evidence_feature_rerank_topk = max(0, int(evidence_feature_rerank_topk))
        self.evidence_residual_scale = float(evidence_residual_scale)
        self.evidence_neural_agreement_weight = float(max(0.0, min(1.0, evidence_neural_agreement_weight)))
        self.evidence_neural_agreement_floor = float(max(0.0, min(1.0, evidence_neural_agreement_floor)))
        self.evidence_neural_agreement_temperature = float(max(1e-4, evidence_neural_agreement_temperature))
        self.evidence_neural_agreement_threshold = float(evidence_neural_agreement_threshold)
        self.evidence_loss_weight = evidence_loss_weight
        self.evidence_margin = evidence_margin
        self.route_supervision_weight = route_supervision_weight
        self.route_supervision_temperature = max(float(route_supervision_temperature), 1e-3)
        self.route_entropy_weight = route_entropy_weight
        self.path_weight = path_weight
        self.bidirectional_paths = bidirectional_paths
        self.evidence_source_names = ["copy", "local", "relation", "subject", "global", "path"]
        self.evidence_gate_source_weight_values = self._parse_evidence_gate_source_weights(
            evidence_gate_source_weights
        )
        initial_evidence_scales = torch.tensor(
            [
                history_copy_weight,
                local_prior_weight,
                relation_prior_weight,
                subject_prior_weight,
                global_prior_weight,
                path_prior_weight,
            ],
            dtype=torch.float32,
        ).clamp_min(1e-4)
        self.evidence_base_log_scales = nn.Parameter(torch.log(torch.expm1(initial_evidence_scales)))
        self.relation_evidence_deltas = nn.Embedding(num_relations * 2, len(self.evidence_source_names))

        self.entity_embeddings = nn.Embedding(num_entities, hidden_dim)
        self.relation_embeddings = nn.Embedding(num_relations * 2, hidden_dim)
        self.entity_semantic_features = nn.Parameter(entity_semantic_features, requires_grad=False)
        self.relation_semantic_features = nn.Parameter(relation_semantic_features, requires_grad=False)
        self.entity_semantic_proj = nn.Linear(entity_semantic_features.size(1), hidden_dim)
        self.relation_semantic_proj = nn.Linear(relation_semantic_features.size(1), hidden_dim)
        self.entity_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relation_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.time_encoder = SinusoidalTimeEncoder(hidden_dim)
        self.history_proj = nn.Linear(hidden_dim, hidden_dim)
        history_heads = max(1, int(history_transformer_heads))
        if hidden_dim % history_heads != 0:
            history_heads = 1
        history_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=history_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        transformer_layers = max(1, int(history_transformer_layers))
        try:
            self.history_transformer = nn.TransformerEncoder(
                history_layer,
                num_layers=transformer_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            self.history_transformer = nn.TransformerEncoder(
                history_layer,
                num_layers=transformer_layers,
            )
        self.triple_query_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.evidence_router = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(self.evidence_source_names)),
        )
        prior_hidden_dim = max(32, hidden_dim // 4)
        self.prior_feature_mlp = nn.Sequential(
            nn.Linear(24, prior_hidden_dim),
            nn.ReLU(),
            nn.Linear(prior_hidden_dim, 1),
        )
        self.evidence_feature_reranker = nn.Sequential(
            nn.Linear(24, prior_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prior_hidden_dim, 1),
        )
        self.concurrent_query_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        query_input_dim = hidden_dim * (5 if self.use_concurrent_query_context else 4)
        self.query_mlp = nn.Sequential(
            nn.Linear(query_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.graph_encoder = LocalRGCNEncoder(hidden_dim, num_relations, num_rgcn_layers, dropout)
        self.path_scorer = TemporalLogicPathScorer(hidden_dim, time_decay, logic_temperature)
        self.conv_decoder = ConvTransEDecoder(
            hidden_dim=hidden_dim,
            num_channels=conv_num_channels,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        nn.init.zeros_(self.relation_evidence_deltas.weight)
        nn.init.zeros_(self.evidence_router[-1].weight)
        nn.init.zeros_(self.evidence_router[-1].bias)

    def _parse_evidence_gate_source_weights(self, payload: str) -> List[float]:
        weights = {name: 1.0 for name in self.evidence_source_names}
        if payload:
            for piece in str(payload).split(","):
                if not piece.strip() or ":" not in piece:
                    continue
                name, value = piece.split(":", 1)
                name = name.strip().lower()
                if name in weights:
                    try:
                        weights[name] = max(0.0, float(value))
                    except ValueError:
                        continue
        return [weights[name] for name in self.evidence_source_names]

    def evidence_gate_source_weight_tensor(
            self,
            device: torch.device,
            dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.as_tensor(self.evidence_gate_source_weight_values, dtype=dtype, device=device)

    def evidence_base_scales(self, relation: int = None) -> torch.Tensor:
        base_scales = F.softplus(self.evidence_base_log_scales)
        if relation is None or self.relation_evidence_weight <= 0:
            return base_scales
        if torch.is_tensor(relation):
            relation_id = relation.to(
                device=self.relation_evidence_deltas.weight.device,
                dtype=torch.long,
                non_blocking=True,
            ).clamp(min=0, max=self.num_relations * 2 - 1)
        else:
            relation_id = torch.as_tensor(
                int(relation),
                dtype=torch.long,
                device=self.relation_evidence_deltas.weight.device,
            ).clamp(min=0, max=self.num_relations * 2 - 1)
        relation_delta = self.relation_evidence_deltas(relation_id).squeeze(0)
        if relation_delta.dim() == 1:
            return F.softplus(self.evidence_base_log_scales + self.relation_evidence_weight * relation_delta)
        return F.softplus(base_scales.new_zeros(relation_delta.shape) + self.evidence_base_log_scales + self.relation_evidence_weight * relation_delta)

    @staticmethod
    def weighted_mean(values: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        if weights is None:
            return values.mean()
        sample_weights = weights.to(device=values.device, dtype=values.dtype)
        while sample_weights.dim() < values.dim():
            sample_weights = sample_weights.unsqueeze(-1)
        return (values * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)

    def encode_concurrent_query_context(
            self,
            subject_relations: List[int],
            time_relations: List[int],
            relation_table: torch.Tensor,
            query_relation_embedding: torch.Tensor,
            query_relation: int,
    ) -> torch.Tensor:
        device = query_relation_embedding.device
        if not self.use_concurrent_query_context:
            return torch.zeros_like(query_relation_embedding)

        def relation_pool(relation_ids: List[int]) -> torch.Tensor:
            if not relation_ids:
                return torch.zeros_like(query_relation_embedding)
            query_base = int(query_relation) % self.num_relations
            cleaned = []
            for rel in relation_ids:
                rel_id = int(rel)
                if rel_id % self.num_relations == query_base:
                    continue
                cleaned.append(max(0, min(rel_id, self.num_relations * 2 - 1)))
            if not cleaned:
                return torch.zeros_like(query_relation_embedding)
            ids = torch.as_tensor(cleaned, dtype=torch.long, device=device)
            rel_repr = relation_table[ids]
            logits = F.cosine_similarity(
                rel_repr.float(),
                query_relation_embedding.float().unsqueeze(0),
                dim=-1,
            ) / self.concurrent_query_temperature
            weights = F.softmax(logits, dim=0).to(dtype=query_relation_embedding.dtype)
            return torch.sum(rel_repr * weights.unsqueeze(-1), dim=0)

        subject_context = relation_pool(subject_relations)
        time_context = relation_pool(time_relations)
        context = self.concurrent_query_proj(
            torch.cat([query_relation_embedding, subject_context, time_context], dim=-1)
        )
        return self.concurrent_query_weight * context

    def batch_source_coverage(self, batch: Dict, device: torch.device) -> torch.Tensor:
        weight_keys = [
            "copy_weights",
            "local_prior_weights",
            "relation_prior_weights",
            "subject_prior_weights",
            "global_prior_weights",
            "path_prior_weights",
        ]
        rows = []
        batch_size = len(batch.get("copy_weights", []))
        for row in range(batch_size):
            coverage = []
            for weight_key in weight_keys:
                weights = batch.get(weight_key, [[]])[row]
                if not weights:
                    coverage.append(0.0)
                else:
                    coverage.append(float(max(weights)))
            rows.append(coverage)
        return torch.as_tensor(rows, dtype=torch.float32, device=device)

    def fused_entity_table(self, projected_semantic: torch.Tensor = None) -> torch.Tensor:
        struct = self.entity_embeddings.weight
        semantic = projected_semantic
        if semantic is None:
            semantic = self.entity_semantic_proj(self.entity_semantic_features.to(struct.device, non_blocking=True))
        gate = torch.sigmoid(self.entity_gate(torch.cat([struct, semantic], dim=-1)))
        return gate * struct + (1.0 - gate) * semantic

    def fused_relation_table(self, projected_semantic: torch.Tensor = None) -> torch.Tensor:
        base_struct = self.relation_embeddings.weight
        semantic = projected_semantic
        if semantic is None:
            semantic = self.relation_semantic_proj(self.relation_semantic_features.to(base_struct.device, non_blocking=True))
        if semantic.size(0) == self.num_relations:
            semantic = torch.cat([semantic, semantic], dim=0)
        gate = torch.sigmoid(self.relation_gate(torch.cat([base_struct, semantic], dim=-1)))
        return gate * base_struct + (1.0 - gate) * semantic

    def sparse_prior_bias(
            self,
            score_template: torch.Tensor,
            entities: List = None,
            weights: List = None,
            scale: float = 0.0,
    ) -> torch.Tensor:
        bias = torch.zeros_like(score_template)
        if isinstance(scale, (float, int)) and scale <= 0:
            return bias
        if not entities:
            return bias
        nbr_ids = torch.as_tensor(entities, dtype=torch.long, device=score_template.device)
        nbr_ids = nbr_ids.clamp(min=0, max=self.num_entities - 1)
        values = torch.as_tensor(weights, dtype=score_template.dtype, device=score_template.device)
        bias.index_add_(0, nbr_ids, values)
        if torch.is_tensor(scale):
            return bias * scale.to(device=score_template.device, dtype=score_template.dtype)
        return scale * bias

    def routed_evidence_bias(
            self,
            score_template: torch.Tensor,
            batch: Dict,
            row: int,
            route_weights: torch.Tensor,
            relation: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_scales = [
            scale for scale in self.evidence_base_scales(relation=relation).to(
                device=score_template.device,
                dtype=score_template.dtype,
            )
        ]
        entity_keys = [
            "copy_entities",
            "local_prior_entities",
            "relation_prior_entities",
            "subject_prior_entities",
            "global_prior_entities",
            "path_prior_entities",
        ]
        weight_keys = [
            "copy_weights",
            "local_prior_weights",
            "relation_prior_weights",
            "subject_prior_weights",
            "global_prior_weights",
            "path_prior_weights",
        ]
        routed_bias = torch.zeros_like(score_template)
        evidence_support = torch.zeros_like(score_template)
        source_count = float(len(base_scales))
        for source_idx, (base_scale, entity_key, weight_key) in enumerate(zip(base_scales, entity_keys, weight_keys)):
            entities = batch.get(entity_key, [[]])[row]
            weights = batch.get(weight_key, [[]])[row]
            if not entities:
                continue
            adaptive_scale = route_weights[source_idx] * source_count + self.evidence_route_floor
            entity_ids = torch.as_tensor(entities, dtype=torch.long, device=score_template.device)
            entity_ids = entity_ids.clamp(min=0, max=self.num_entities - 1)
            values = torch.as_tensor(weights, dtype=score_template.dtype, device=score_template.device)
            source_support = values * adaptive_scale.to(device=score_template.device, dtype=score_template.dtype)
            routed_bias.index_add_(0, entity_ids, source_support * base_scale)
            evidence_support.index_add_(0, entity_ids, source_support.detach())
        return routed_bias, evidence_support

    def batched_routed_evidence_bias(
            self,
            score_template: torch.Tensor,
            batch: Dict,
            route_weights: torch.Tensor,
            relations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        entity_tensor = batch.get("evidence_entities_tensor")
        weight_tensor = batch.get("evidence_weights_tensor")
        if entity_tensor is None or weight_tensor is None:
            return None, None
        batch_size, _, num_entities = score_template.shape[0], len(self.evidence_source_names), self.num_entities
        if entity_tensor.numel() == 0 or entity_tensor.size(-1) == 0:
            zeros = torch.zeros_like(score_template)
            return zeros, zeros

        device = score_template.device
        dtype = score_template.dtype
        entities = entity_tensor.to(device=device, non_blocking=True)
        weights = weight_tensor.to(device=device, dtype=dtype, non_blocking=True)
        source_count = float(entities.size(1))
        base_scales = self.evidence_base_scales(relation=relations).to(device=device, dtype=dtype)
        if base_scales.dim() == 1:
            base_scales = base_scales.unsqueeze(0).expand(batch_size, -1)
        adaptive_scale = route_weights.to(device=device, dtype=dtype) * source_count + self.evidence_route_floor
        support_values = weights * adaptive_scale.unsqueeze(-1)
        bias_values = support_values * base_scales.unsqueeze(-1)

        valid = (entities >= 0) & (entities < num_entities) & (weights != 0)
        if not bool(valid.any()):
            zeros = torch.zeros_like(score_template)
            return zeros, zeros
        row_offsets = torch.arange(batch_size, device=device, dtype=torch.long).view(batch_size, 1, 1) * num_entities
        flat_indices = (entities.clamp(min=0, max=num_entities - 1) + row_offsets)[valid]
        flat_size = batch_size * num_entities
        routed_bias = score_template.new_zeros(flat_size)
        evidence_support = score_template.new_zeros(flat_size)
        routed_bias.index_add_(0, flat_indices, bias_values[valid])
        evidence_support.index_add_(0, flat_indices, support_values.detach()[valid])
        return routed_bias.view(batch_size, num_entities), evidence_support.view(batch_size, num_entities)

    def evidence_consensus_bias(
            self,
            score_template: torch.Tensor,
            batch: Dict,
            row: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bias = torch.zeros_like(score_template)
        support = torch.zeros_like(score_template)
        if self.evidence_consensus_weight <= 0:
            return bias, support
        entity_keys = [
            "copy_entities",
            "local_prior_entities",
            "relation_prior_entities",
            "subject_prior_entities",
            "global_prior_entities",
            "path_prior_entities",
        ]
        weight_keys = [
            "copy_weights",
            "local_prior_weights",
            "relation_prior_weights",
            "subject_prior_weights",
            "global_prior_weights",
            "path_prior_weights",
        ]
        source_count = torch.zeros_like(score_template)
        strength_sum = torch.zeros_like(score_template)
        for entity_key, weight_key in zip(entity_keys, weight_keys):
            entities = batch.get(entity_key, [[]])[row]
            weights = batch.get(weight_key, [[]])[row]
            if not entities:
                continue
            entity_ids = torch.as_tensor(entities, dtype=torch.long, device=score_template.device)
            entity_ids = entity_ids.clamp(min=0, max=self.num_entities - 1)
            values = torch.as_tensor(weights, dtype=score_template.dtype, device=score_template.device).clamp_min(0)
            present = torch.ones_like(values)
            source_count.index_add_(0, entity_ids, present)
            strength_sum.index_add_(0, entity_ids, values)
        mask = source_count >= float(self.evidence_consensus_min_sources)
        if not bool(mask.any()):
            return bias, support
        mean_strength = strength_sum / source_count.clamp_min(1.0)
        consensus = torch.log1p(source_count) * mean_strength
        consensus = torch.where(mask, consensus, torch.zeros_like(consensus))
        bias = self.evidence_consensus_weight * consensus
        support = consensus.detach()
        return bias, support

    def relation_transfer_copy_bias(
            self,
            score_template: torch.Tensor,
            subject_histories: List,
            subject_history_times: List,
            relation_table: torch.Tensor,
            query_relation_embedding: torch.Tensor,
            query_time: int,
            route_weights: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bias = torch.zeros_like(score_template)
        support = torch.zeros_like(score_template)
        if self.relation_transfer_copy_weight <= 0:
            return bias, support
        if not subject_histories:
            return bias, support

        histories = subject_histories
        history_times = subject_history_times
        if self.relation_transfer_copy_history_len > 0:
            histories = histories[-self.relation_transfer_copy_history_len:]
            history_times = history_times[-self.relation_transfer_copy_history_len:]

        device = score_template.device
        dtype = score_template.dtype
        query_relation = query_relation_embedding.float().unsqueeze(0)
        route_scale = torch.as_tensor(1.0, dtype=dtype, device=device)
        if route_weights is not None:
            copy_route = route_weights[0].to(device=device, dtype=dtype)
            subject_route = route_weights[3].to(device=device, dtype=dtype)
            route_scale = (copy_route + subject_route).clamp_min(0.0) * 3.0 + self.evidence_route_floor

        for block, tim in zip(histories, history_times):
            if block is None or len(block) == 0:
                continue
            block_tensor = torch.as_tensor(block, dtype=torch.long, device=device)
            rel_ids = block_tensor[:, 0].clamp(min=0, max=relation_table.size(0) - 1)
            nbr_ids = block_tensor[:, 1].clamp(min=0, max=self.num_entities - 1)
            rel_repr = relation_table[rel_ids].float()
            relation_match = F.cosine_similarity(rel_repr, query_relation, dim=-1)
            match_weight = torch.sigmoid(relation_match / self.relation_transfer_copy_temperature)
            time_weight = math.exp(-self.history_copy_decay * max(0, int(query_time) - int(tim)))
            values = (match_weight * float(time_weight)).to(dtype=dtype)
            support.index_add_(0, nbr_ids, values.detach())
            bias.index_add_(0, nbr_ids, values)
        return self.relation_transfer_copy_weight * route_scale * bias, support

    def evidence_rank_fusion_bias(
            self,
            score_template: torch.Tensor,
            batch: Dict,
            row: int,
            route_weights: torch.Tensor,
            relation: int,
    ) -> torch.Tensor:
        bias = torch.zeros_like(score_template)
        if self.evidence_rank_fusion_weight <= 0:
            return bias
        base_scales = self.evidence_base_scales(relation=relation).to(
            device=score_template.device,
            dtype=score_template.dtype,
        )
        entity_keys = [
            "copy_entities",
            "local_prior_entities",
            "relation_prior_entities",
            "subject_prior_entities",
            "global_prior_entities",
            "path_prior_entities",
        ]
        source_count = float(len(entity_keys))
        k_value = torch.as_tensor(self.evidence_rank_fusion_k, dtype=score_template.dtype, device=score_template.device)
        for source_idx, entity_key in enumerate(entity_keys):
            entities = batch.get(entity_key, [[]])[row]
            if not entities:
                continue
            adaptive_scale = route_weights[source_idx] * source_count + self.evidence_route_floor
            entity_ids = torch.as_tensor(entities, dtype=torch.long, device=score_template.device).clamp(
                min=0,
                max=self.num_entities - 1,
            )
            ranks = torch.arange(
                1,
                len(entities) + 1,
                dtype=score_template.dtype,
                device=score_template.device,
            )
            values = (
                self.evidence_rank_fusion_weight
                * base_scales[source_idx]
                * adaptive_scale.to(dtype=score_template.dtype)
                / (k_value + ranks)
            )
            bias.index_add_(0, entity_ids, values)
        return bias

    def batched_evidence_rank_fusion_bias(
            self,
            score_template: torch.Tensor,
            batch: Dict,
            route_weights: torch.Tensor,
            relations: torch.Tensor,
    ) -> torch.Tensor:
        if self.evidence_rank_fusion_weight <= 0:
            return torch.zeros_like(score_template)
        entity_tensor = batch.get("evidence_entities_tensor")
        if entity_tensor is None or entity_tensor.numel() == 0 or entity_tensor.size(-1) == 0:
            return torch.zeros_like(score_template)
        device = score_template.device
        dtype = score_template.dtype
        batch_size = score_template.size(0)
        entities = entity_tensor.to(device=device, non_blocking=True)
        valid = (entities >= 0) & (entities < self.num_entities)
        if not bool(valid.any()):
            return torch.zeros_like(score_template)
        base_scales = self.evidence_base_scales(relation=relations).to(device=device, dtype=dtype)
        if base_scales.dim() == 1:
            base_scales = base_scales.unsqueeze(0).expand(batch_size, -1)
        adaptive_scale = route_weights.to(device=device, dtype=dtype) * float(entities.size(1)) + self.evidence_route_floor
        ranks = torch.arange(
            1,
            entities.size(-1) + 1,
            dtype=dtype,
            device=device,
        ).view(1, 1, -1)
        values = (
            self.evidence_rank_fusion_weight
            * base_scales.unsqueeze(-1)
            * adaptive_scale.unsqueeze(-1)
            / (torch.as_tensor(self.evidence_rank_fusion_k, dtype=dtype, device=device) + ranks)
        )
        row_offsets = torch.arange(batch_size, device=device, dtype=torch.long).view(batch_size, 1, 1) * self.num_entities
        flat_indices = (entities.clamp(min=0, max=self.num_entities - 1) + row_offsets)[valid]
        bias = score_template.new_zeros(batch_size * self.num_entities)
        bias.index_add_(0, flat_indices, values.expand_as(entities.to(dtype=dtype))[valid])
        return bias.view(batch_size, self.num_entities)

    @staticmethod
    def _list_weight(entity: int, entities: List = None, weights: List = None) -> float:
        if not entities:
            return 0.0
        for candidate, weight in zip(entities, weights or []):
            if int(candidate) == int(entity):
                return float(weight)
        return 0.0

    def target_source_supports(
            self,
            batch: Dict,
            row: int,
            target: int,
            device: torch.device,
    ) -> torch.Tensor:
        source_keys = [
            ("copy_entities", "copy_weights"),
            ("local_prior_entities", "local_prior_weights"),
            ("relation_prior_entities", "relation_prior_weights"),
            ("subject_prior_entities", "subject_prior_weights"),
            ("global_prior_entities", "global_prior_weights"),
            ("path_prior_entities", "path_prior_weights"),
        ]
        values = [
            self._list_weight(
                entity=target,
                entities=batch.get(entity_key, [[]])[row],
                weights=batch.get(weight_key, [[]])[row],
            )
            for entity_key, weight_key in source_keys
        ]
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    def batch_target_source_supports(
            self,
            batch: Dict,
            targets: torch.Tensor,
            device: torch.device,
    ) -> torch.Tensor:
        entity_tensor = batch.get("evidence_entities_tensor")
        weight_tensor = batch.get("evidence_weights_tensor")
        if entity_tensor is None or weight_tensor is None or entity_tensor.numel() == 0:
            return torch.stack(
                [
                    self.target_source_supports(batch=batch, row=row, target=int(targets[row].item()), device=device)
                    for row in range(targets.size(0))
                ],
                dim=0,
            )
        entities = entity_tensor.to(device=device, non_blocking=True)
        weights = weight_tensor.to(device=device, dtype=torch.float32, non_blocking=True)
        if entities.size(-1) == 0:
            return torch.zeros(targets.size(0), len(self.evidence_source_names), dtype=torch.float32, device=device)
        target_view = targets.to(device=device, dtype=torch.long, non_blocking=True).view(-1, 1, 1)
        matches = entities == target_view
        matched_weights = torch.where(matches, weights, torch.zeros_like(weights))
        return matched_weights.amax(dim=-1)

    def route_supervision_loss(
            self,
            route_weights: torch.Tensor,
            source_target_supports: torch.Tensor,
            sample_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.route_supervision_weight <= 0:
            return torch.tensor(0.0, device=route_weights.device)
        support_sum = source_target_supports.sum(dim=-1)
        mask = support_sum > 0
        if torch.count_nonzero(mask) == 0:
            return torch.tensor(0.0, device=route_weights.device)
        supports = source_target_supports[mask].float()
        target_distribution = F.softmax(
            torch.log1p(supports) / self.route_supervision_temperature,
            dim=-1,
        )
        log_routes = torch.log(route_weights[mask].float().clamp_min(1e-8))
        losses = -(target_distribution * log_routes).sum(dim=-1)
        selected_weights = sample_weights[mask] if sample_weights is not None else None
        return self.weighted_mean(losses, selected_weights)

    def route_entropy_loss(self, route_weights: torch.Tensor) -> torch.Tensor:
        if self.route_entropy_weight <= 0:
            return torch.tensor(0.0, device=route_weights.device)
        routes = route_weights.float().clamp_min(1e-8)
        return -(routes * torch.log(routes)).sum(dim=-1).mean()

    def learned_sparse_prior_bias(
            self,
            score_template: torch.Tensor,
            copy_entities: List = None,
            copy_weights: List = None,
            local_entities: List = None,
            local_weights: List = None,
            relation_entities: List = None,
            relation_weights: List = None,
            subject_entities: List = None,
            subject_weights: List = None,
            global_entities: List = None,
            global_weights: List = None,
            path_entities: List = None,
            path_weights: List = None,
            route_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        bias = torch.zeros_like(score_template)
        if self.learned_prior_weight <= 0:
            return bias

        feature_map: Dict[int, List[float]] = {}

        def collect(entities: List, weights: List, column: int) -> None:
            if not entities:
                return
            for entity, weight in zip(entities, weights):
                entity_id = int(entity)
                if entity_id < 0 or entity_id >= self.num_entities:
                    continue
                feature_map.setdefault(entity_id, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])[column] += float(weight)

        collect(copy_entities or [], copy_weights or [], 0)
        collect(local_entities or [], local_weights or [], 1)
        collect(relation_entities or [], relation_weights or [], 2)
        collect(subject_entities or [], subject_weights or [], 3)
        collect(global_entities or [], global_weights or [], 4)
        collect(path_entities or [], path_weights or [], 5)
        if not feature_map:
            return bias

        candidate_ids = sorted(feature_map)
        raw_features = []
        route_vector = None
        route_entropy = 0.0
        max_route = 0.0
        if route_weights is not None:
            if torch.is_tensor(route_weights):
                route_vector = [float(value) for value in route_weights.detach().float().clamp_min(1e-8).tolist()]
            else:
                route_vector = [float(value) for value in route_weights]
            route_entropy = -sum(value * math.log(max(value, 1e-8)) for value in route_vector)
            max_route = max(route_vector)
        else:
            route_vector = [1.0 / 6.0] * 6
        for entity_id in candidate_ids:
            copy_weight, local_weight, relation_weight, subject_weight, global_weight, path_weight = feature_map[entity_id]
            raw_values = [
                copy_weight,
                local_weight,
                relation_weight,
                subject_weight,
                global_weight,
                path_weight,
            ]
            routed_values = [value * route for value, route in zip(raw_values, route_vector)]
            temporal_sum = (
                copy_weight
                + local_weight
                + relation_weight
                + subject_weight
                + global_weight
                + path_weight
            )
            routed_sum = sum(routed_values)
            raw_features.append(
                [
                    copy_weight,
                    local_weight,
                    relation_weight,
                    subject_weight,
                    global_weight,
                    path_weight,
                    temporal_sum,
                    max(raw_values),
                    float(sum(1 for value in raw_values if value > 0.0)),
                    float(copy_weight > 0.0 and local_weight > 0.0),
                    float(subject_weight > 0.0 and relation_weight > 0.0),
                    float(path_weight > 0.0 and local_weight > 0.0),
                    float(path_weight > 0.0 and relation_weight > 0.0),
                    *routed_values,
                    routed_sum,
                    max(routed_values),
                    float(sum(1 for value in routed_values if value > 0.0)),
                    route_entropy,
                    max_route,
                ]
            )
        features = torch.as_tensor(raw_features, dtype=score_template.dtype, device=score_template.device)
        learned_bias = self.prior_feature_mlp(features).squeeze(-1)
        ids = torch.as_tensor(candidate_ids, dtype=torch.long, device=score_template.device)
        bias.index_copy_(0, ids, learned_bias.to(dtype=score_template.dtype))
        return self.learned_prior_weight * bias

    def evidence_feature_rerank_bias(
            self,
            score_template: torch.Tensor,
            batch: Dict,
            row: int,
            route_weights: torch.Tensor,
            neural_scores: torch.Tensor,
            target: int = None,
    ) -> torch.Tensor:
        bias = torch.zeros_like(score_template)
        if not self.use_evidence_feature_reranker or self.evidence_feature_rerank_weight <= 0:
            return bias
        source_keys = [
            ("copy_entities", "copy_weights"),
            ("local_prior_entities", "local_prior_weights"),
            ("relation_prior_entities", "relation_prior_weights"),
            ("subject_prior_entities", "subject_prior_weights"),
            ("global_prior_entities", "global_prior_weights"),
            ("path_prior_entities", "path_prior_weights"),
        ]
        feature_map: Dict[int, List[float]] = {}

        def ensure(entity_id: int) -> List[float]:
            return feature_map.setdefault(int(entity_id), [0.0] * 6)

        for column, (entity_key, weight_key) in enumerate(source_keys):
            entities = batch.get(entity_key, [[]])[row]
            weights = batch.get(weight_key, [[]])[row]
            if not entities:
                continue
            for entity, weight in zip(entities, weights):
                entity_id = int(entity)
                if 0 <= entity_id < self.num_entities:
                    ensure(entity_id)[column] += float(weight)

        if self.evidence_feature_rerank_topk > 0 and neural_scores is not None:
            topk = min(int(self.evidence_feature_rerank_topk), int(neural_scores.numel()))
            if topk > 0:
                for entity_id in torch.topk(neural_scores.detach(), k=topk, dim=0).indices.tolist():
                    ensure(int(entity_id))
        if self.training and target is not None:
            ensure(int(target))
            for entity_id in batch.get("positive_entities", [[]])[row]:
                ensure(int(entity_id))
        if not feature_map:
            return bias

        ids = sorted(feature_map)
        raw_sources = [feature_map[entity_id] for entity_id in ids]
        source_tensor = torch.as_tensor(raw_sources, dtype=score_template.dtype, device=score_template.device)
        indicators = (source_tensor > 0).to(dtype=score_template.dtype)
        log_sources = torch.log1p(source_tensor.clamp_min(0.0))
        source_sum = source_tensor.sum(dim=-1, keepdim=True)
        source_max = source_tensor.max(dim=-1, keepdim=True).values
        source_count = indicators.sum(dim=-1, keepdim=True)
        if route_weights is None:
            route_vector = torch.full(
                (6,),
                1.0 / 6.0,
                dtype=score_template.dtype,
                device=score_template.device,
            )
        else:
            route_vector = route_weights.to(device=score_template.device, dtype=score_template.dtype)
        routed_sum = (source_tensor * route_vector.unsqueeze(0)).sum(dim=-1, keepdim=True)
        candidate_ids = torch.as_tensor(ids, dtype=torch.long, device=score_template.device)
        candidate_scores = neural_scores[candidate_ids] if neural_scores is not None else score_template[candidate_ids]
        neural_mean = neural_scores.mean() if neural_scores is not None else score_template.mean()
        neural_std = (
            neural_scores.float().std().to(dtype=score_template.dtype)
            if neural_scores is not None
            else score_template.float().std().to(dtype=score_template.dtype)
        ).clamp_min(1e-4)
        neural_z = ((candidate_scores - neural_mean) / neural_std).unsqueeze(-1)
        neural_rank_feature = torch.zeros_like(neural_z)
        if neural_scores is not None and self.evidence_feature_rerank_topk > 0:
            topk = min(int(self.evidence_feature_rerank_topk), int(neural_scores.numel()))
            top_ids = torch.topk(neural_scores.detach(), k=topk, dim=0).indices.tolist()
            rank_lookup = {int(entity_id): rank for rank, entity_id in enumerate(top_ids, start=1)}
            neural_rank_feature = torch.as_tensor(
                [1.0 / float(rank_lookup.get(int(entity_id), topk + 1)) for entity_id in ids],
                dtype=score_template.dtype,
                device=score_template.device,
            ).unsqueeze(-1)
        features = torch.cat(
            [
                source_tensor,
                log_sources,
                indicators,
                source_sum,
                source_max,
                source_count,
                routed_sum,
                neural_z.clamp(min=-5.0, max=5.0),
                neural_rank_feature,
            ],
            dim=-1,
        )
        learned_bias = self.evidence_feature_reranker(features).squeeze(-1)
        bias.index_copy_(0, candidate_ids, learned_bias.to(dtype=score_template.dtype))
        return self.evidence_feature_rerank_weight * bias

    def evidence_candidate_support(
            self,
            batch: Dict,
            row: int,
            score_template: torch.Tensor,
            target: int = None,
            route_weights: torch.Tensor = None,
            neural_scores: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.zeros_like(score_template, dtype=torch.bool)
        support = torch.zeros_like(score_template)
        if not self.evidence_candidate_gating:
            mask[:] = True
            return mask, support
        source_to_keys = {
            "copy": ("copy_entities", "copy_weights"),
            "local": ("local_prior_entities", "local_prior_weights"),
            "relation": ("relation_prior_entities", "relation_prior_weights"),
            "subject": ("subject_prior_entities", "subject_prior_weights"),
            "global": ("global_prior_entities", "global_prior_weights"),
            "path": ("path_prior_entities", "path_prior_weights"),
        }
        source_to_index = {name: idx for idx, name in enumerate(self.evidence_source_names)}
        threshold = float(self.evidence_gate_threshold)
        source_count = torch.zeros_like(score_template)
        gate_source_weights = self.evidence_gate_source_weight_tensor(
            device=score_template.device,
            dtype=score_template.dtype,
        )
        for source_name in self.evidence_gate_sources:
            if source_name not in source_to_keys:
                continue
            entity_key, weight_key = source_to_keys[source_name]
            entities = batch.get(entity_key, [[]])[row]
            weights = batch.get(weight_key, [[]])[row]
            if not entities:
                continue
            source_scale = 1.0
            source_weight = 1.0
            if source_name in source_to_index:
                source_weight = float(gate_source_weights[source_to_index[source_name]].detach().float().item())
            if source_weight <= 0:
                continue
            if route_weights is not None and source_name in source_to_index:
                source_scale = float(route_weights[source_to_index[source_name]].detach().float().item())
            for entity, weight in zip(entities, weights):
                if float(weight) >= threshold:
                    entity_id = int(entity)
                    if 0 <= entity_id < self.num_entities:
                        support[entity_id] += float(weight) * source_scale * source_weight
                        source_count[entity_id] += 1.0
        if self.evidence_gate_support_threshold > 0:
            evidence_mask = support >= float(self.evidence_gate_support_threshold)
        else:
            evidence_mask = source_count >= float(self.evidence_gate_min_sources)
        mask[evidence_mask] = True
        should_force_target = self.evidence_gate_include_target_train and self.evidence_gate_mode not in {
            "penalty",
            "soft",
            "soft_penalty",
            "adaptive",
            "support",
            "support_penalty",
            "tiered",
            "evidence_first",
            "band",
        }
        if self.training and should_force_target and target is not None:
            target_id = int(target)
            if 0 <= target_id < self.num_entities:
                mask[target_id] = True
            for entity in batch.get("positive_entities", [[]])[row]:
                entity_id = int(entity)
                if 0 <= entity_id < self.num_entities:
                    mask[entity_id] = True
        if self.evidence_gate_neural_topk > 0 and neural_scores is not None:
            topk = min(int(self.evidence_gate_neural_topk), int(neural_scores.numel()))
            if topk > 0:
                neural_ids = torch.topk(neural_scores.detach(), k=topk, dim=0).indices
                mask[neural_ids] = True
        if not bool(mask.any().item()):
            mask[:] = True
        return mask, support

    def batched_evidence_candidate_support(
            self,
            batch: Dict,
            score_template: torch.Tensor,
            route_weights: torch.Tensor = None,
            neural_scores: torch.Tensor = None,
            targets: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.evidence_candidate_gating:
            return torch.ones_like(score_template, dtype=torch.bool), torch.zeros_like(score_template)
        entity_tensor = batch.get("evidence_entities_tensor")
        weight_tensor = batch.get("evidence_weights_tensor")
        if entity_tensor is None or weight_tensor is None:
            return None, None

        device = score_template.device
        dtype = score_template.dtype
        batch_size = score_template.size(0)
        mask = torch.zeros_like(score_template, dtype=torch.bool)
        support = torch.zeros_like(score_template)
        if entity_tensor.numel() > 0 and entity_tensor.size(-1) > 0:
            entities = entity_tensor.to(device=device, non_blocking=True)
            weights = weight_tensor.to(device=device, dtype=dtype, non_blocking=True)
            selected_sources = torch.tensor(
                [source in self.evidence_gate_sources for source in self.evidence_source_names],
                dtype=torch.bool,
                device=device,
            ).view(1, -1, 1)
            gate_source_weights = self.evidence_gate_source_weight_tensor(device=device, dtype=dtype).view(1, -1, 1)
            valid = (
                selected_sources
                & (gate_source_weights > 0)
                & (entities >= 0)
                & (entities < self.num_entities)
                & (weights >= float(self.evidence_gate_threshold))
            )
            if bool(valid.any()):
                source_scale = torch.ones(
                    batch_size,
                    len(self.evidence_source_names),
                    1,
                    dtype=dtype,
                    device=device,
                )
                if route_weights is not None:
                    source_scale = route_weights.to(device=device, dtype=dtype).unsqueeze(-1)
                values = weights * source_scale * gate_source_weights
                row_offsets = torch.arange(batch_size, device=device, dtype=torch.long).view(batch_size, 1, 1)
                row_offsets = row_offsets * self.num_entities
                flat_indices = (entities.clamp(min=0, max=self.num_entities - 1) + row_offsets)[valid]
                flat_support = score_template.new_zeros(batch_size * self.num_entities)
                flat_count = score_template.new_zeros(batch_size * self.num_entities)
                flat_support.index_add_(0, flat_indices, values[valid])
                flat_count.index_add_(0, flat_indices, torch.ones_like(values[valid]))
                support = flat_support.view(batch_size, self.num_entities)
                source_count = flat_count.view(batch_size, self.num_entities)
                if self.evidence_gate_support_threshold > 0:
                    mask = support >= float(self.evidence_gate_support_threshold)
                else:
                    mask = source_count >= float(self.evidence_gate_min_sources)

        should_force_target = self.evidence_gate_include_target_train and self.evidence_gate_mode not in {
            "penalty",
            "soft",
            "soft_penalty",
            "adaptive",
            "support",
            "support_penalty",
            "tiered",
            "evidence_first",
            "band",
        }
        if self.training and should_force_target and targets is not None:
            row_ids = torch.arange(batch_size, device=device)
            safe_targets = targets.to(device=device, dtype=torch.long, non_blocking=True).clamp(
                min=0,
                max=self.num_entities - 1,
            )
            mask[row_ids, safe_targets] = True
            for row, positives in enumerate(batch.get("positive_entities", [[]])):
                for entity in positives:
                    entity_id = int(entity)
                    if 0 <= entity_id < self.num_entities:
                        mask[row, entity_id] = True
        if self.evidence_gate_neural_topk > 0 and neural_scores is not None:
            topk = min(int(self.evidence_gate_neural_topk), int(neural_scores.size(1)))
            if topk > 0:
                neural_ids = torch.topk(neural_scores.detach(), k=topk, dim=1).indices
                mask.scatter_(
                    1,
                    neural_ids,
                    torch.ones_like(neural_ids, dtype=torch.bool, device=device),
                )
        empty_rows = ~mask.any(dim=1)
        if bool(empty_rows.any()):
            mask[empty_rows] = True
        return mask, support

    def evidence_candidate_mask(
            self,
            batch: Dict,
            row: int,
            score_template: torch.Tensor,
            target: int = None,
            neural_scores: torch.Tensor = None,
    ) -> torch.Tensor:
        mask, _ = self.evidence_candidate_support(
            batch=batch,
            row=row,
            score_template=score_template,
            target=target,
            route_weights=None,
            neural_scores=neural_scores,
        )
        return mask

    def apply_evidence_candidate_gate(
            self,
            scores: torch.Tensor,
            batch: Dict,
            row: int,
            target: int = None,
            route_weights: torch.Tensor = None,
            neural_scores: torch.Tensor = None,
            precomputed_mask: torch.Tensor = None,
            precomputed_support: torch.Tensor = None,
    ) -> torch.Tensor:
        if not self.evidence_candidate_gating:
            return scores
        if precomputed_mask is not None and precomputed_support is not None:
            mask, support = precomputed_mask, precomputed_support
        else:
            mask, support = self.evidence_candidate_support(
                batch=batch,
                row=row,
                score_template=scores,
                target=target,
                route_weights=route_weights,
                neural_scores=neural_scores,
            )
        if self.evidence_gate_mode in {"adaptive", "support", "support_penalty"}:
            penalty = torch.as_tensor(self.evidence_gate_penalty, dtype=scores.dtype, device=scores.device)
            boost_scale = torch.as_tensor(self.evidence_gate_boost, dtype=scores.dtype, device=scores.device)
            support_boost = torch.log1p(support.clamp_min(0.0)) * boost_scale
            return scores + support_boost + (~mask).to(dtype=scores.dtype) * penalty
        if self.evidence_gate_mode in {"tiered", "evidence_first", "band"}:
            penalty = torch.as_tensor(self.evidence_gate_penalty, dtype=scores.dtype, device=scores.device)
            boost_scale = torch.as_tensor(self.evidence_gate_boost, dtype=scores.dtype, device=scores.device)
            band_width = torch.as_tensor(self.evidence_gate_band_width, dtype=scores.dtype, device=scores.device)
            evidence_mask = mask & (support > 0)
            fallback_mask = mask & (~evidence_mask)
            tier = 2.0 * evidence_mask.to(dtype=scores.dtype) + fallback_mask.to(dtype=scores.dtype)
            support_boost = torch.log1p(support.clamp_min(0.0)) * boost_scale
            return scores + band_width * tier + support_boost + (~mask).to(dtype=scores.dtype) * penalty
        if self.evidence_gate_mode in {"lexical", "lexicographic", "evidence_lexical"}:
            tier_gap = torch.as_tensor(
                max(abs(self.evidence_gate_band_width), 1.0),
                dtype=scores.dtype,
                device=scores.device,
            )
            penalty = torch.as_tensor(self.evidence_gate_penalty, dtype=scores.dtype, device=scores.device)
            boost_scale = torch.as_tensor(self.evidence_gate_boost, dtype=scores.dtype, device=scores.device)
            evidence_mask = mask & (support > 0)
            fallback_mask = mask & (~evidence_mask)
            score_mean = scores.mean()
            score_std = scores.float().std().to(dtype=scores.dtype).clamp_min(1e-4)
            neural_tiebreak = torch.tanh((scores - score_mean) / (2.0 * score_std))
            support_score = torch.log1p(support.clamp_min(0.0)) * boost_scale
            final_scores = torch.full_like(scores, penalty)
            final_scores = torch.where(
                fallback_mask,
                0.5 * tier_gap + 0.25 * neural_tiebreak,
                final_scores,
            )
            final_scores = torch.where(
                evidence_mask,
                1.5 * tier_gap + support_score + 0.25 * neural_tiebreak,
                final_scores,
            )
            return final_scores
        if self.evidence_gate_mode in {"penalty", "soft", "soft_penalty"}:
            penalty = torch.as_tensor(self.evidence_gate_penalty, dtype=scores.dtype, device=scores.device)
            return scores + (~mask).to(dtype=scores.dtype) * penalty
        gate_value = torch.as_tensor(self.evidence_gate_value, dtype=scores.dtype, device=scores.device)
        return torch.where(mask, scores, gate_value)

    def apply_evidence_residual_blend(
            self,
            neural_scores: torch.Tensor,
            scores: torch.Tensor,
    ) -> torch.Tensor:
        residual = scores - neural_scores
        scale = torch.full_like(scores, float(self.evidence_residual_scale))
        if self.evidence_neural_agreement_weight > 0:
            centered = neural_scores - neural_scores.mean()
            neural_std = neural_scores.float().std().to(dtype=scores.dtype).clamp_min(1e-4)
            neural_z = centered / neural_std
            agreement = torch.sigmoid(
                (neural_z - self.evidence_neural_agreement_threshold)
                / self.evidence_neural_agreement_temperature
            )
            floor = torch.as_tensor(self.evidence_neural_agreement_floor, dtype=scores.dtype, device=scores.device)
            agreement_scale = floor + (1.0 - floor) * agreement.to(dtype=scores.dtype)
            weight = torch.as_tensor(self.evidence_neural_agreement_weight, dtype=scores.dtype, device=scores.device)
            scale = scale * ((1.0 - weight) + weight * agreement_scale)
        return neural_scores + residual * scale

    def complex_scores(
            self,
            subject_tensor: torch.Tensor,
            relation_tensor: torch.Tensor,
            entity_table: torch.Tensor,
    ) -> torch.Tensor:
        if subject_tensor.size(-1) < 2:
            return torch.matmul(subject_tensor, entity_table.t())
        split = subject_tensor.size(-1) // 2
        s_re, s_im = subject_tensor[:, :split], subject_tensor[:, split:split * 2]
        r_re, r_im = relation_tensor[:, :split], relation_tensor[:, split:split * 2]
        e_re, e_im = entity_table[:, :split], entity_table[:, split:split * 2]
        real_query = s_re * r_re - s_im * r_im
        imag_query = s_re * r_im + s_im * r_re
        return (torch.matmul(real_query, e_re.t()) + torch.matmul(imag_query, e_im.t())) / math.sqrt(max(1, split))

    def distmult_scores(
            self,
            subject_tensor: torch.Tensor,
            relation_tensor: torch.Tensor,
            entity_table: torch.Tensor,
    ) -> torch.Tensor:
        query = subject_tensor * relation_tensor
        return torch.matmul(query, entity_table.t()) / math.sqrt(max(1, subject_tensor.size(-1)))

    def transe_scores(
            self,
            subject_tensor: torch.Tensor,
            relation_tensor: torch.Tensor,
            entity_table: torch.Tensor,
    ) -> torch.Tensor:
        query = subject_tensor + relation_tensor
        query_norm = query.pow(2).sum(dim=-1, keepdim=True)
        entity_norm = entity_table.pow(2).sum(dim=-1).unsqueeze(0)
        distance = query_norm + entity_norm - 2.0 * torch.matmul(query, entity_table.t())
        return -distance.clamp_min(0.0) / math.sqrt(max(1, subject_tensor.size(-1)))

    def pairwise_ranking_loss(
            self,
            scores: torch.Tensor,
            targets: torch.Tensor,
            positive_entities: List[List[int]] = None,
            sample_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.ranking_weight <= 0:
            return torch.tensor(0.0, device=scores.device)
        positive = scores.gather(1, targets.view(-1, 1))
        masked = scores.clone()
        if positive_entities is None:
            masked.scatter_(1, targets.view(-1, 1), -torch.inf)
        else:
            for row, positives in enumerate(positive_entities):
                if not positives:
                    masked[row, int(targets[row].item())] = -torch.inf
                    continue
                ids = torch.as_tensor(positives, dtype=torch.long, device=scores.device)
                ids = ids.clamp(min=0, max=scores.size(1) - 1)
                masked[row, ids] = -torch.inf
        topk = min(self.ranking_topk, max(1, scores.size(1) - 1))
        hard_negatives = torch.topk(masked, k=topk, dim=1).values
        losses = F.softplus(hard_negatives - positive + self.ranking_margin).mean(dim=1)
        return self.weighted_mean(losses, sample_weights)

    def multi_positive_cross_entropy(
            self,
            scores: torch.Tensor,
            targets: torch.Tensor,
            positive_entities: List[List[int]] = None,
            sample_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        if not positive_entities:
            losses = F.cross_entropy(
                scores,
                targets,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
            return self.weighted_mean(losses, sample_weights)

        losses = []
        for row, positives in enumerate(positive_entities):
            if not positives:
                positives = [int(targets[row].item())]
            if int(targets[row].item()) not in {int(entity) for entity in positives}:
                positives = list(positives) + [int(targets[row].item())]
            ids = torch.as_tensor(positives, dtype=torch.long, device=scores.device)
            ids = ids.clamp(min=0, max=scores.size(1) - 1)
            log_denominator = torch.logsumexp(scores[row], dim=0)
            log_positive = torch.logsumexp(scores[row].index_select(0, ids), dim=0)
            losses.append(-(log_positive - log_denominator))
        positive_loss = torch.stack(losses)
        if self.label_smoothing <= 0:
            return self.weighted_mean(positive_loss, sample_weights)
        smooth_loss = -F.log_softmax(scores, dim=1).mean(dim=1)
        mixed = (1.0 - self.label_smoothing) * positive_loss + self.label_smoothing * smooth_loss
        return self.weighted_mean(mixed, sample_weights)

    def evidence_consistency_loss(
            self,
            scores: torch.Tensor,
            evidence_support: torch.Tensor,
            targets: torch.Tensor,
            positive_entities: List[List[int]] = None,
            sample_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.evidence_loss_weight <= 0:
            return torch.tensor(0.0, device=scores.device)
        target_support = evidence_support.gather(1, targets.view(-1, 1))
        if torch.count_nonzero(target_support > 0) == 0:
            return torch.tensor(0.0, device=scores.device)
        masked_scores = scores.detach().clone()
        if positive_entities is None:
            masked_scores.scatter_(1, targets.view(-1, 1), -torch.inf)
        else:
            for row, positives in enumerate(positive_entities):
                if not positives:
                    masked_scores[row, int(targets[row].item())] = -torch.inf
                    continue
                ids = torch.as_tensor(positives, dtype=torch.long, device=scores.device)
                ids = ids.clamp(min=0, max=scores.size(1) - 1)
                masked_scores[row, ids] = -torch.inf
        hard_negative = masked_scores.max(dim=1, keepdim=True).indices
        target_scores = scores.gather(1, targets.view(-1, 1))
        negative_scores = scores.gather(1, hard_negative)
        negative_support = evidence_support.gather(1, hard_negative)
        losses = F.softplus(
            negative_scores - target_scores + negative_support - target_support + self.evidence_margin
        ).squeeze(-1)
        mask = target_support.squeeze(-1) > 0
        selected_weights = sample_weights[mask] if sample_weights is not None else None
        return self.weighted_mean(losses[mask], selected_weights)

    def encode_subject_history(
            self,
            subject_histories: List,
            subject_history_times: List,
            relation_table: torch.Tensor,
            entity_table: torch.Tensor,
            query_relation_embedding: torch.Tensor,
            query_time: int,
    ) -> torch.Tensor:
        device = entity_table.device
        event_reprs = []
        event_logits = []
        for block, tim in zip(subject_histories, subject_history_times):
            if block is None or len(block) == 0:
                continue
            block_tensor = torch.as_tensor(block, dtype=torch.long, device=device)
            rel_ids = block_tensor[:, 0]
            nbr_ids = block_tensor[:, 1]
            rel_repr = relation_table[rel_ids]
            nbr_repr = entity_table[nbr_ids]
            time_weight = math.exp(-self.history_copy_decay * max(0, query_time - int(tim)))
            event_repr = time_weight * 0.5 * (rel_repr + nbr_repr)
            relation_match = F.cosine_similarity(
                rel_repr.float(),
                query_relation_embedding.float().unsqueeze(0),
                dim=-1,
            ).to(dtype=event_repr.dtype)
            recency_log = math.log(max(time_weight, 1e-8))
            event_reprs.append(event_repr)
            event_logits.append(relation_match / self.history_attention_temperature + recency_log)
        if not event_reprs:
            return torch.zeros(self.hidden_dim, device=device)
        stacked_repr = torch.cat(event_reprs, dim=0)
        stacked_logits = torch.cat(event_logits, dim=0).float()
        attention = F.softmax(stacked_logits, dim=0).to(dtype=stacked_repr.dtype)
        attentive_context = torch.sum(attention.unsqueeze(-1) * stacked_repr, dim=0)
        if not self.use_history_transformer:
            return self.history_proj(attentive_context)
        query_token = query_relation_embedding.unsqueeze(0)
        tokens = torch.cat([query_token, stacked_repr], dim=0).unsqueeze(0)
        encoded = self.history_transformer(tokens.float()).to(dtype=stacked_repr.dtype).squeeze(0)
        return self.history_proj(attentive_context + encoded[0])

    def path_edges_for_reasoning(self, edges: List[TemporalEdge]) -> List[TemporalEdge]:
        if not self.bidirectional_paths:
            return edges
        reversed_edges = [
            TemporalEdge(edge.dst, edge.rel + self.num_relations, edge.src, edge.timestamp)
            for edge in edges
        ]
        return list(edges) + reversed_edges

    def encode_local_graph(
            self,
            graph: LocalSubgraph,
            subject: int,
            query_time: int,
            entity_table: torch.Tensor,
            relation_table: torch.Tensor,
    ) -> torch.Tensor:
        device = entity_table.device
        graph = graph.to(device)
        node_ids = graph.ndata["id"].view(-1)
        node_features = entity_table[node_ids]
        if graph.num_edges() > 0:
            edge_times = graph.edata["time"]
            edge_types = graph.edata["type"]
            delta = torch.full_like(edge_times, fill_value=query_time) - edge_times
            edge_context = relation_table[edge_types] + self.time_encoder(delta)
            node_updates = torch.zeros_like(node_features)
            dst_nodes = graph.edges()[1]
            node_updates.index_add_(0, dst_nodes, edge_context)
            node_features = node_features + node_updates
        encoded = self.graph_encoder(graph, node_features)
        local_subject = graph.ids.get(subject, 0)
        subject_repr = encoded[local_subject]
        graph_summary = encoded.mean(dim=0)
        return subject_repr + graph_summary

    def forward(
            self,
            batch: Dict,
            return_explanations: bool = False,
            route_weights_override: torch.Tensor = None,
    ) -> Dict:
        device = self.entity_embeddings.weight.device
        subjects = batch["subjects"].to(device, non_blocking=True)
        relations = batch["relations"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        query_times = batch["times"].to(device, non_blocking=True)
        sample_weights = batch.get("sample_weights")
        if sample_weights is not None:
            sample_weights = sample_weights.to(device, non_blocking=True).float()

        projected_entity_semantic = self.entity_semantic_proj(
            self.entity_semantic_features.to(device, non_blocking=True)
        )
        projected_relation_semantic = self.relation_semantic_proj(
            self.relation_semantic_features.to(device, non_blocking=True)
        )
        entity_table = self.fused_entity_table(projected_entity_semantic)
        relation_table = self.fused_relation_table(projected_relation_semantic)

        query_reprs = []
        triple_query_reprs = []
        sample_payloads = []
        per_sample_explanations: List[Dict] = []
        alignment_losses = []

        for idx in range(subjects.size(0)):
            subject = int(subjects[idx].item())
            relation = int(relations[idx].item())
            target = int(targets[idx].item())
            query_time = int(query_times[idx].item())

            relation_repr = relation_table[relation]
            subject_repr = entity_table[subject]
            heavy_evidence_flags = batch.get("use_heavy_evidence")
            has_heavy_evidence = True
            if heavy_evidence_flags is not None:
                has_heavy_evidence = bool(heavy_evidence_flags[idx].item())
            use_neural_context = has_heavy_evidence and not (
                self.training
                and self.train_context_batch_limit > 0
                and idx >= self.train_context_batch_limit
            )
            if use_neural_context and self.use_local_graph_context:
                graph_context = self.encode_local_graph(
                    graph=batch["local_graphs"][idx],
                    subject=subject,
                    query_time=query_time,
                    entity_table=entity_table,
                    relation_table=relation_table,
                )
            else:
                graph_context = torch.zeros_like(subject_repr)
            if use_neural_context and self.use_subject_history_context:
                history_context = self.encode_subject_history(
                    subject_histories=batch["subject_histories"][idx],
                    subject_history_times=batch["subject_history_times"][idx],
                    relation_table=relation_table,
                    entity_table=entity_table,
                    query_relation_embedding=relation_repr,
                    query_time=query_time,
                )
            else:
                history_context = torch.zeros_like(subject_repr)
            time_repr = self.time_encoder(query_times[idx:idx + 1])[0]
            query_features = [subject_repr, relation_repr, graph_context, history_context + time_repr]
            if self.use_concurrent_query_context:
                concurrent_context = self.encode_concurrent_query_context(
                    subject_relations=batch.get("concurrent_subject_relations", [[]])[idx],
                    time_relations=batch.get("concurrent_time_relations", [[]])[idx],
                    relation_table=relation_table,
                    query_relation_embedding=relation_repr,
                    query_relation=relation,
                )
                query_features.append(concurrent_context)
            query_repr = self.query_mlp(torch.cat(query_features, dim=-1))
            query_reprs.append(self.dropout(query_repr))
            triple_query_reprs.append(self.triple_query_proj(subject_repr * relation_repr))
            sample_payloads.append(
                {
                    "subject": subject,
                    "relation": relation,
                    "target": target,
                    "query_time": query_time,
                    "relation_repr": relation_repr,
                    "subject_repr": subject_repr,
                }
            )

            if use_neural_context and self.use_local_graph_context and batch["local_graphs"][idx].num_nodes() > 0:
                subgraph_nodes = batch["local_graphs"][idx].ndata["id"].view(-1).to(device)
                alignment_loss = 1.0 - F.cosine_similarity(
                    entity_table[subgraph_nodes],
                    projected_entity_semantic[subgraph_nodes],
                    dim=-1,
                ).mean()
                alignment_losses.append(alignment_loss)

        query_tensor = torch.stack(query_reprs, dim=0)
        if self.use_evidence_router:
            route_logits = self.evidence_router(query_tensor.float())
        else:
            route_logits = torch.zeros(
                query_tensor.size(0),
                len(self.evidence_source_names),
                dtype=torch.float32,
                device=device,
            )
        if self.use_evidence_router and self.coverage_route_weight > 0:
            coverage = self.batch_source_coverage(batch, device=device)
            route_logits = route_logits + self.coverage_route_weight * torch.log1p(coverage)
        route_logits = route_logits / self.evidence_router_temperature
        route_weights = F.softmax(route_logits, dim=-1).to(dtype=query_tensor.dtype)
        if route_weights_override is not None:
            route_weights = route_weights_override.to(device=device, dtype=query_tensor.dtype, non_blocking=True)
        route_weights_for_features = route_weights.detach().float().cpu().tolist()
        context_scores = torch.matmul(query_tensor, entity_table.t())
        if self.triple_score_weight != 0:
            triple_query_tensor = torch.stack(triple_query_reprs, dim=0)
            triple_scores = torch.matmul(triple_query_tensor, entity_table.t())
            base_scores_tensor = self.context_score_weight * context_scores + self.triple_score_weight * triple_scores
        else:
            base_scores_tensor = self.context_score_weight * context_scores
        subject_tensor = None
        relation_tensor = None
        if (
                self.distmult_score_weight != 0
                or self.transe_score_weight != 0
                or self.complex_score_weight != 0
                or self.conv_score_weight != 0
        ):
            subject_tensor = torch.stack([payload["subject_repr"] for payload in sample_payloads], dim=0)
            relation_tensor = torch.stack([payload["relation_repr"] for payload in sample_payloads], dim=0)
        if self.distmult_score_weight != 0:
            distmult_scores = self.distmult_scores(subject_tensor, relation_tensor, entity_table)
            base_scores_tensor = base_scores_tensor + self.distmult_score_weight * distmult_scores
        if self.transe_score_weight != 0:
            transe_scores = self.transe_scores(subject_tensor, relation_tensor, entity_table)
            base_scores_tensor = base_scores_tensor + self.transe_score_weight * transe_scores
        if self.complex_score_weight != 0:
            complex_scores = self.complex_scores(subject_tensor, relation_tensor, entity_table)
            base_scores_tensor = base_scores_tensor + self.complex_score_weight * complex_scores
        if self.conv_score_weight != 0:
            conv_scores = self.conv_decoder(subject_tensor, relation_tensor, entity_table)
            base_scores_tensor = base_scores_tensor + self.conv_score_weight * conv_scores
        if self.normalize_neural_scores:
            score_mean = base_scores_tensor.mean(dim=1, keepdim=True)
            score_std = base_scores_tensor.float().std(dim=1, keepdim=True).to(dtype=base_scores_tensor.dtype)
            base_scores_tensor = (base_scores_tensor - score_mean) / score_std.clamp_min(1e-4)
            base_scores_tensor = base_scores_tensor * self.neural_score_scale
        batched_evidence_bias, batched_evidence_support = self.batched_routed_evidence_bias(
            score_template=base_scores_tensor,
            batch=batch,
            route_weights=route_weights,
            relations=relations,
        )
        batched_rank_fusion_bias = self.batched_evidence_rank_fusion_bias(
            score_template=base_scores_tensor,
            batch=batch,
            route_weights=route_weights,
            relations=relations,
        )
        batched_gate_mask, batched_gate_support = self.batched_evidence_candidate_support(
            batch=batch,
            score_template=base_scores_tensor,
            route_weights=route_weights,
            neural_scores=base_scores_tensor,
            targets=targets,
        )
        batched_target_supports = self.batch_target_source_supports(batch=batch, targets=targets, device=device)
        logits = []
        pre_gate_logits = []
        evidence_supports = []
        source_target_supports = []
        path_topk = self.train_path_topk if self.training else self.eval_path_topk

        for idx, payload in enumerate(sample_payloads):
            subject = payload["subject"]
            relation = payload["relation"]
            relation_repr = payload["relation_repr"]
            subject_repr = payload["subject_repr"]
            target = payload["target"]
            query_time = payload["query_time"]
            base_scores = base_scores_tensor[idx]
            if batched_evidence_bias is None or batched_evidence_support is None:
                evidence_bias, evidence_support = self.routed_evidence_bias(
                    score_template=base_scores_tensor[idx],
                    batch=batch,
                    row=idx,
                    route_weights=route_weights[idx],
                    relation=relation,
                )
            else:
                evidence_bias = batched_evidence_bias[idx]
                evidence_support = batched_evidence_support[idx]
            base_scores = base_scores + evidence_bias
            consensus_bias, consensus_support = self.evidence_consensus_bias(
                score_template=base_scores_tensor[idx],
                batch=batch,
                row=idx,
            )
            base_scores = base_scores + consensus_bias
            evidence_support = evidence_support + consensus_support
            transfer_bias, transfer_support = self.relation_transfer_copy_bias(
                score_template=base_scores_tensor[idx],
                subject_histories=batch["subject_histories"][idx],
                subject_history_times=batch["subject_history_times"][idx],
                relation_table=relation_table,
                query_relation_embedding=relation_repr,
                query_time=query_time,
                route_weights=route_weights[idx],
            )
            base_scores = base_scores + transfer_bias
            evidence_support = evidence_support + transfer_support
            if self.relation_transfer_prior_weight > 0:
                prior_transfer_support = self.sparse_prior_bias(
                    score_template=base_scores_tensor[idx],
                    entities=batch.get("relation_transfer_entities", [[]])[idx],
                    weights=batch.get("relation_transfer_weights", [[]])[idx],
                    scale=1.0,
                )
                base_scores = base_scores + self.relation_transfer_prior_weight * prior_transfer_support
                evidence_support = evidence_support + prior_transfer_support.detach()
            if batched_rank_fusion_bias is None:
                base_scores = base_scores + self.evidence_rank_fusion_bias(
                    score_template=base_scores_tensor[idx],
                    batch=batch,
                    row=idx,
                    route_weights=route_weights[idx],
                    relation=relation,
                )
            else:
                base_scores = base_scores + batched_rank_fusion_bias[idx]
            base_scores = base_scores + self.learned_sparse_prior_bias(
                score_template=base_scores_tensor[idx],
                copy_entities=batch.get("copy_entities", [[]])[idx],
                copy_weights=batch.get("copy_weights", [[]])[idx],
                local_entities=batch.get("local_prior_entities", [[]])[idx],
                local_weights=batch.get("local_prior_weights", [[]])[idx],
                relation_entities=batch.get("relation_prior_entities", [[]])[idx],
                relation_weights=batch.get("relation_prior_weights", [[]])[idx],
                subject_entities=batch.get("subject_prior_entities", [[]])[idx],
                subject_weights=batch.get("subject_prior_weights", [[]])[idx],
                global_entities=batch.get("global_prior_entities", [[]])[idx],
                global_weights=batch.get("global_prior_weights", [[]])[idx],
                path_entities=batch.get("path_prior_entities", [[]])[idx],
                path_weights=batch.get("path_prior_weights", [[]])[idx],
                route_weights=route_weights_for_features[idx],
            )
            base_scores = base_scores + self.evidence_feature_rerank_bias(
                score_template=base_scores_tensor[idx],
                batch=batch,
                row=idx,
                route_weights=route_weights[idx],
                neural_scores=base_scores_tensor[idx],
                target=target,
            )
            base_scores = self.apply_evidence_residual_blend(
                neural_scores=base_scores_tensor[idx],
                scores=base_scores,
            )
            sample_path_topk = path_topk
            if self.training and self.train_path_batch_limit > 0 and idx >= self.train_path_batch_limit:
                sample_path_topk = 0
            if sample_path_topk <= 0:
                path_shortlist = []
            else:
                shortlist_size = min(self.shortlist_size, self.num_entities)
                shortlist = torch.topk(base_scores, k=shortlist_size).indices.tolist()
                if self.training and target not in shortlist:
                    shortlist[-1] = target
                path_shortlist = shortlist[: min(sample_path_topk, len(shortlist))]
                if self.training and target not in path_shortlist:
                    path_shortlist[-1:] = [target]
            path_indices = torch.as_tensor(path_shortlist, dtype=torch.long, device=device) if path_shortlist else None
            reranked_scores = base_scores[path_indices].clone() if path_indices is not None else None
            explanations_for_sample = {}

            if path_shortlist:
                path_edges = self.path_edges_for_reasoning(batch["local_edges"][idx])
                for local_idx, candidate in enumerate(path_shortlist):
                    paths = enumerate_temporal_paths(
                        path_edges,
                        source=subject,
                        target=int(candidate),
                        max_len=self.max_path_len,
                        max_paths=self.max_paths_per_candidate,
                        query_time=query_time,
                    )
                    path_bias, evidence = self.path_scorer(
                        query_relation_embedding=relation_repr,
                        candidate_embedding=entity_table[candidate],
                        source_embedding=subject_repr,
                        relation_table=relation_table,
                        entity_table=entity_table,
                        query_time=query_time,
                        paths=paths,
                    )
                    reranked_scores[local_idx] = reranked_scores[local_idx] + self.path_weight * path_bias
                    if return_explanations and evidence:
                        explanations_for_sample[int(candidate)] = [
                            {
                                "score": item["score"],
                                "nodes": item["path"].nodes,
                                "relations": item["path"].relations,
                                "timestamps": item["path"].timestamps,
                                "text": temporal_path_to_text(item["path"]),
                            }
                            for item in evidence[:3]
                        ]

            final_scores = base_scores.clone()
            if path_indices is not None:
                final_scores[path_indices] = reranked_scores
            pre_gate_logits.append(final_scores.clone())
            final_scores = self.apply_evidence_candidate_gate(
                scores=final_scores,
                batch=batch,
                row=idx,
                target=target,
                route_weights=route_weights[idx],
                neural_scores=base_scores_tensor[idx],
                precomputed_mask=batched_gate_mask[idx] if batched_gate_mask is not None else None,
                precomputed_support=batched_gate_support[idx] if batched_gate_support is not None else None,
            )
            logits.append(final_scores)
            evidence_supports.append(evidence_support)
            source_target_supports.append(batched_target_supports[idx])
            per_sample_explanations.append(explanations_for_sample)

        logits_tensor = torch.stack(logits, dim=0)
        pre_gate_logits_tensor = torch.stack(pre_gate_logits, dim=0)
        evidence_support_tensor = torch.stack(evidence_supports, dim=0)
        source_target_support_tensor = torch.stack(source_target_supports, dim=0)
        positive_entities = batch.get("positive_entities")
        classification_loss = self.multi_positive_cross_entropy(
            logits_tensor,
            targets,
            positive_entities=positive_entities,
            sample_weights=sample_weights,
        )
        ranking_loss = self.pairwise_ranking_loss(
            logits_tensor,
            targets,
            positive_entities=positive_entities,
            sample_weights=sample_weights,
        )
        evidence_loss = self.evidence_consistency_loss(
            logits_tensor,
            evidence_support_tensor,
            targets,
            positive_entities=positive_entities,
            sample_weights=sample_weights,
        )
        route_loss = self.route_supervision_loss(
            route_weights,
            source_target_support_tensor,
            sample_weights=sample_weights,
        )
        route_entropy = self.route_entropy_loss(route_weights)
        alignment_loss = torch.stack(alignment_losses).mean() if alignment_losses else torch.tensor(0.0, device=device)
        loss = (
            classification_loss
            + self.ranking_weight * ranking_loss
            + self.alignment_weight * alignment_loss
            + self.evidence_loss_weight * evidence_loss
            + self.route_supervision_weight * route_loss
            + self.route_entropy_weight * route_entropy
        )
        return {
            "loss": loss,
            "scores": logits_tensor,
            "pre_gate_scores": pre_gate_logits_tensor.detach(),
            "neural_scores": base_scores_tensor.detach(),
            "classification_loss": classification_loss.detach(),
            "ranking_loss": ranking_loss.detach(),
            "evidence_loss": evidence_loss.detach(),
            "route_loss": route_loss.detach(),
            "route_entropy": route_entropy.detach(),
            "alignment_loss": alignment_loss.detach(),
            "explanations": per_sample_explanations,
            "evidence_route_weights": route_weights.detach(),
            "evidence_source_names": self.evidence_source_names,
            "target_source_supports": source_target_support_tensor.detach(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Model self-test")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        num_entities, num_relations = 8, 4
        entity_sem = torch.randn(num_entities, 32)
        relation_sem = torch.randn(num_relations, 32)
        model = SingleStepTemporalReasoner(
            num_entities=num_entities,
            num_relations=num_relations,
            entity_semantic_features=entity_sem,
            relation_semantic_features=relation_sem,
            hidden_dim=32,
            shortlist_size=4,
        )
        from temporal_graph import TemporalEdge, build_dgl_subgraph

        edges = [TemporalEdge(0, 1, 2, 1), TemporalEdge(2, 0, 3, 2), TemporalEdge(0, 2, 4, 1)]
        graph = build_dgl_subgraph(edges, num_relations=num_relations)
        batch = {
            "subjects": torch.tensor([0]),
            "relations": torch.tensor([1]),
            "targets": torch.tensor([3]),
            "times": torch.tensor([4]),
            "subject_histories": [[[[1, 2], [2, 4]]]],
            "subject_history_times": [[1]],
            "object_histories": [[]],
            "object_history_times": [[]],
            "history_events": [[]],
            "local_edges": [edges],
            "local_graphs": [graph],
        }
        output = model(batch, return_explanations=True)
        print({"loss": float(output["loss"].item()), "top_score_shape": list(output["scores"].shape)})
    else:
        print("Model definition loaded. Use --demo for a toy forward pass.")


if __name__ == "__main__":
    main()
