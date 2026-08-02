import argparse
from bisect import bisect_left
import json
import math
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from configs import DataConfig
from history_preprocess import build_history_graphs, get_total_number, load_quadruples
from temporal_graph import TemporalEdge, build_dgl_subgraph, extract_local_temporal_graph
from utils import build_filter_map, load_id_text_mapping

def ensure_preprocessed(dataset_dir: Path, preprocess_dir: Path, history_len: int) -> None:
    print(
        {
            "stage": "preprocess_check",
            "dataset_dir": str(dataset_dir),
            "preprocess_dir": str(preprocess_dir),
            "history_len": int(history_len),
        },
        flush=True,
    )
    required = [
        "train_graphs.txt",
        "train_history_sub.txt",
        "train_history_ob.txt",
        "dev_history_sub.txt",
        "dev_history_ob.txt",
        "test_history_sub.txt",
        "test_history_ob.txt",
    ]
    meta_path = preprocess_dir / "history_meta.json"
    required_ready = all((preprocess_dir / file_name).exists() for file_name in required)
    if required_ready and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
        if int(meta.get("history_len", -1)) == int(history_len):
            print({"stage": "preprocess_ready", "reason": "history_meta_matches"}, flush=True)
            return
    if required_ready and not meta_path.exists() and int(history_len) == 10:
        print({"stage": "preprocess_ready", "reason": "legacy_history_files"}, flush=True)
        return
    print({"stage": "preprocess_build_start", "note": "building history graph files"}, flush=True)
    start = time.perf_counter()
    build_history_graphs(
        dataset_dir=str(dataset_dir),
        history_len=history_len,
        output_dir=str(preprocess_dir),
    )
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump({"history_len": int(history_len)}, fp, ensure_ascii=False, indent=2)
    print({"stage": "preprocess_build_done", "seconds": time.perf_counter() - start}, flush=True)


class TemporalReasoningDataset(Dataset):
    def __init__(self, data_config: DataConfig, split: str):
        init_start = time.perf_counter()
        self.data_config = data_config
        self.split = split
        self.dataset_dir = Path(data_config.dataset_dir).resolve()
        self.preprocess_dir = Path(data_config.preprocess_dir or self.dataset_dir).resolve()
        self.preprocess_dir.mkdir(parents=True, exist_ok=True)
        print(
            {
                "stage": "dataset_init_start",
                "split": split,
                "dataset_dir": str(self.dataset_dir),
            },
            flush=True,
        )
        ensure_preprocessed(self.dataset_dir, self.preprocess_dir, data_config.history_len)

        load_start = time.perf_counter()
        print({"stage": "dataset_load_quadruples_start", "split": split}, flush=True)
        self.train_data, _ = load_quadruples(str(self.dataset_dir), "train.txt")
        self.valid_data, _ = load_quadruples(str(self.dataset_dir), "valid.txt")
        self.test_data, _ = load_quadruples(str(self.dataset_dir), "test.txt")
        self.num_entities, self.num_relations = get_total_number(str(self.dataset_dir), "stat.txt")
        print(
            {
                "stage": "dataset_load_quadruples_done",
                "split": split,
                "train_facts": int(len(self.train_data)),
                "valid_facts": int(len(self.valid_data)),
                "test_facts": int(len(self.test_data)),
                "num_entities": int(self.num_entities),
                "num_relations": int(self.num_relations),
                "seconds": time.perf_counter() - load_start,
            },
            flush=True,
        )
        self.all_facts = np.concatenate([self.train_data, self.valid_data, self.test_data], axis=0)
        print({"stage": "dataset_filter_maps_start", "split": split}, flush=True)
        self.filter_map = build_filter_map(self.all_facts)
        self.train_positive_map = build_filter_map(self.train_data)
        print({"stage": "dataset_relation_transfer_start", "split": split}, flush=True)
        self.relation_transfer_weights = self._build_relation_transfer_weights(self.train_data)
        self.train_inverse_positive_map: Dict[Tuple[int, int, int], set] = {}
        for head, rel, tail, tim in self.train_data.tolist():
            key = (int(tail), int(rel) + self.num_relations, int(tim))
            self.train_inverse_positive_map.setdefault(key, set()).add(int(head))
        train_times = self.train_data[:, 3].astype(int)
        self.train_min_time = int(train_times.min()) if len(train_times) else 0
        self.train_max_time = int(train_times.max()) if len(train_times) else self.train_min_time

        self.entity_texts = load_id_text_mapping(
            file_path=str(self.dataset_dir / "entity2text.txt"),
            size=self.num_entities,
            fallback_prefix="Entity",
            fallback_file_path=str(self.dataset_dir / "entity2id.txt"),
        )
        self.relation_texts = load_id_text_mapping(
            file_path=str(self.dataset_dir / "relation2text.txt"),
            size=self.num_relations,
            fallback_prefix="Relation",
            fallback_file_path=str(self.dataset_dir / "relation2id.txt"),
        )

        self.use_strict_history = bool(getattr(data_config, "strict_eval_history", False)) and split in {"valid", "test"}

        if split == "train":
            self.data = self.train_data
            self.history_source = self.train_data
            history_sub_name = "train_history_sub.txt"
            history_ob_name = "train_history_ob.txt"
        elif split == "valid":
            self.data = self.valid_data
            self.history_source = self.train_data if self.use_strict_history else np.concatenate(
                [self.train_data, self.valid_data],
                axis=0,
            )
            history_sub_name = "dev_history_sub.txt"
            history_ob_name = "dev_history_ob.txt"
        elif split == "test":
            self.data = self.test_data
            self.history_source = (
                np.concatenate([self.train_data, self.valid_data], axis=0)
                if self.use_strict_history
                else np.concatenate([self.train_data, self.valid_data, self.test_data], axis=0)
            )
            history_sub_name = "test_history_sub.txt"
            history_ob_name = "test_history_ob.txt"
        else:
            raise ValueError(f"Unsupported split: {split}")

        print({"stage": "dataset_concurrent_index_start", "split": split}, flush=True)
        self.concurrent_subject_relations, self.concurrent_time_relations = self._build_concurrent_query_indexes(self.data)

        self.subject_history_index = {}
        self.object_history_index = {}
        self._subject_lookup_cache: Dict[Tuple[int, int], Tuple[List, List[int]]] = {}
        self._object_lookup_cache: Dict[Tuple[int, int], Tuple[List, List[int]]] = {}
        if self.use_strict_history:
            print({"stage": "dataset_strict_history_index_start", "split": split}, flush=True)
            self.subject_history_index, self.object_history_index = self._build_history_indexes(self.history_source)
            self.subject_history, self.subject_history_t = [], []
            self.object_history, self.object_history_t = [], []
        else:
            print(
                {
                    "stage": "dataset_history_pickle_load_start",
                    "split": split,
                    "subject_history": history_sub_name,
                    "object_history": history_ob_name,
                },
                flush=True,
            )
            with open(self.preprocess_dir / history_sub_name, "rb") as fp:
                self.subject_history, self.subject_history_t = pickle.load(fp)
            with open(self.preprocess_dir / history_ob_name, "rb") as fp:
                self.object_history, self.object_history_t = pickle.load(fp)

        print({"stage": "dataset_events_by_time_start", "split": split}, flush=True)
        self.events_by_time: Dict[int, List[List[int]]] = {}
        for head, rel, tail, tim in self.history_source.tolist():
            self.events_by_time.setdefault(int(tim), []).append([int(head), int(rel), int(tail), int(tim)])
        self.sorted_times = sorted(self.events_by_time)
        self._history_events_cache: Dict[int, List[List[int]]] = {}
        self._prior_events_cache: Dict[int, List[List[int]]] = {}
        self._long_prior_events_cache: Dict[int, List[List[int]]] = {}
        self._relation_prior_cache: Dict[tuple, Dict[str, List[float]]] = {}
        self._local_prior_cache: Dict[tuple, Dict[str, List[float]]] = {}
        self._subject_prior_cache: Dict[tuple, Dict[str, List[float]]] = {}
        self._global_prior_cache: Dict[tuple, Dict[str, List[float]]] = {}
        self._path_prior_cache: Dict[tuple, Dict[str, List[float]]] = {}
        self._local_edges_cache: Dict[tuple, List[TemporalEdge]] = {}
        self._prior_table_cache: Dict[int, Dict[str, Dict]] = {}
        self.current_epoch = 0
        print(
            {
                "stage": "dataset_init_done",
                "split": split,
                "samples": int(len(self)),
                "base_facts": int(len(self.data)),
                "history_times": int(len(self.sorted_times)),
                "seconds": time.perf_counter() - init_start,
            },
            flush=True,
        )

    def __len__(self) -> int:
        if (
                self.split == "train"
                and self.data_config.use_inverse_train
                and self.data_config.inverse_train_mode == "paired"
        ):
            return len(self.data) * 2
        return len(self.data)

    def _build_concurrent_query_indexes(
            self,
            events: np.ndarray,
    ) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[int, List[int]]]:
        subject_topk = max(0, int(getattr(self.data_config, "concurrent_subject_topk", 0)))
        time_topk = max(0, int(getattr(self.data_config, "concurrent_time_topk", 0)))
        if subject_topk == 0 and time_topk == 0:
            return {}, {}
        subject_buckets: Dict[Tuple[int, int], Dict[int, int]] = {}
        time_buckets: Dict[int, Dict[int, int]] = {}
        for head, rel, _, tim in events.tolist():
            key = (int(head), int(tim))
            relation = int(rel)
            subject_row = subject_buckets.setdefault(key, {})
            subject_row[relation] = subject_row.get(relation, 0) + 1
            time_row = time_buckets.setdefault(int(tim), {})
            time_row[relation] = time_row.get(relation, 0) + 1

        def top_relations(row: Dict[int, int], top_k: int) -> List[int]:
            if top_k <= 0 or not row:
                return []
            return [
                int(rel)
                for rel, _ in sorted(
                    row.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:top_k]
            ]

        subject_index = {
            key: top_relations(row, subject_topk)
            for key, row in subject_buckets.items()
        }
        time_index = {
            int(tim): top_relations(row, time_topk)
            for tim, row in time_buckets.items()
        }
        return subject_index, time_index

    def _concurrent_query_relations(
            self,
            subject: int,
            query_time: int,
    ) -> Tuple[List[int], List[int]]:
        subject_relations = list(self.concurrent_subject_relations.get((int(subject), int(query_time)), []))
        time_relations = list(self.concurrent_time_relations.get(int(query_time), []))
        return subject_relations, time_relations

    def _build_relation_transfer_weights(self, events: np.ndarray) -> Dict[int, Dict[int, float]]:
        pair_relations: Dict[Tuple[int, int], set] = {}
        relation_counts: Dict[int, int] = {}
        for head, rel, tail, _ in events.tolist():
            head = int(head)
            rel = int(rel)
            tail = int(tail)
            pair_relations.setdefault((head, tail), set()).add(rel)
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
        pair_counts: Dict[int, Dict[int, float]] = {}
        for relations in pair_relations.values():
            if len(relations) < 2:
                continue
            rel_list = sorted(int(rel) for rel in relations)
            for query_rel in rel_list:
                transfer = pair_counts.setdefault(query_rel, {})
                for history_rel in rel_list:
                    if history_rel == query_rel:
                        continue
                    transfer[history_rel] = transfer.get(history_rel, 0.0) + 1.0
        normalized: Dict[int, Dict[int, float]] = {}
        for query_rel, transfer in pair_counts.items():
            query_count = max(1.0, float(relation_counts.get(query_rel, 1)))
            row: Dict[int, float] = {}
            for history_rel, value in transfer.items():
                history_count = max(1.0, float(relation_counts.get(history_rel, 1)))
                row[int(history_rel)] = float(value / math.sqrt(query_count * history_count))
            if row:
                normalized[int(query_rel)] = row
        return normalized

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    @staticmethod
    def _build_history_indexes(events: np.ndarray) -> Tuple[Dict[int, Tuple[List[int], List[np.ndarray]]], Dict[int, Tuple[List[int], List[np.ndarray]]]]:
        subject_buckets: Dict[int, Dict[int, List[List[int]]]] = {}
        object_buckets: Dict[int, Dict[int, List[List[int]]]] = {}
        for head, rel, tail, tim in events.tolist():
            head = int(head)
            rel = int(rel)
            tail = int(tail)
            tim = int(tim)
            subject_buckets.setdefault(head, {}).setdefault(tim, []).append([rel, tail])
            object_buckets.setdefault(tail, {}).setdefault(tim, []).append([rel, head])

        def finalize(buckets: Dict[int, Dict[int, List[List[int]]]]) -> Dict[int, Tuple[List[int], List[np.ndarray]]]:
            index = {}
            for entity, by_time in buckets.items():
                times = sorted(by_time)
                blocks = [np.asarray(by_time[tim], dtype=np.int64) for tim in times]
                index[int(entity)] = (times, blocks)
            return index

        return finalize(subject_buckets), finalize(object_buckets)

    def _lookup_entity_history(
            self,
            history_index: Dict[int, Tuple[List[int], List[np.ndarray]]],
            cache: Dict[Tuple[int, int], Tuple[List, List[int]]],
            entity: int,
            query_time: int,
    ) -> Tuple[List, List[int]]:
        cache_key = (int(entity), int(query_time))
        if cache_key in cache:
            histories, times = cache[cache_key]
            return [block.copy() for block in histories], list(times)
        entry = history_index.get(int(entity))
        if entry is None:
            cache[cache_key] = ([], [])
            return [], []
        times, blocks = entry
        end = bisect_left(times, int(query_time))
        start = max(0, end - int(self.data_config.history_len))
        histories = [block.copy() for block in blocks[start:end]]
        history_times = list(times[start:end])
        cache[cache_key] = ([block.copy() for block in histories], list(history_times))
        return histories, history_times

    def _subject_history_for(self, subject: int, query_time: int) -> Tuple[List, List[int]]:
        return self._lookup_entity_history(
            self.subject_history_index,
            self._subject_lookup_cache,
            subject,
            query_time,
        )

    def _object_history_for(self, obj: int, query_time: int) -> Tuple[List, List[int]]:
        return self._lookup_entity_history(
            self.object_history_index,
            self._object_lookup_cache,
            obj,
            query_time,
        )

    def _history_events(self, query_time: int) -> List[List[int]]:
        if query_time in self._history_events_cache:
            return self._history_events_cache[query_time]
        lower = query_time - self.data_config.history_window
        collected: List[List[int]] = []
        for tim in self.sorted_times:
            if tim >= query_time:
                break
            if tim < lower:
                continue
            collected.extend(self.events_by_time[tim])
        self._history_events_cache[query_time] = collected
        return collected

    def _prior_events(self, query_time: int) -> List[List[int]]:
        if query_time in self._prior_events_cache:
            return self._prior_events_cache[query_time]
        lower = query_time - self.data_config.prior_window
        collected: List[List[int]] = []
        for tim in self.sorted_times:
            if tim >= query_time:
                break
            if tim < lower:
                continue
            collected.extend(self.events_by_time[tim])
        self._prior_events_cache[query_time] = collected
        return collected

    def _long_prior_events(self, query_time: int) -> List[List[int]]:
        if query_time in self._long_prior_events_cache:
            return self._long_prior_events_cache[query_time]
        lower = query_time - int(getattr(self.data_config, "long_prior_window", self.data_config.prior_window))
        collected: List[List[int]] = []
        for tim in self.sorted_times:
            if tim >= query_time:
                break
            if tim < lower:
                continue
            collected.extend(self.events_by_time[tim])
        self._long_prior_events_cache[query_time] = collected
        return collected

    @staticmethod
    def _normalize_signal(weights: Dict[int, float], top_k: int) -> Dict[str, List[float]]:
        if top_k <= 0 or not weights:
            return {"entities": [], "weights": []}
        ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:top_k]
        max_weight = max(weight for _, weight in ranked) or 1.0
        return {
            "entities": [entity for entity, _ in ranked],
            "weights": [float(np.log1p(weight) / np.log1p(max_weight)) for _, weight in ranked],
        }

    def _prior_tables(self, query_time: int) -> Dict[str, Dict]:
        if not hasattr(self, "_prior_table_cache"):
            self._prior_table_cache = {}
        if query_time in self._prior_table_cache:
            return self._prior_table_cache[query_time]

        def empty_tables() -> Dict[str, Dict]:
            return {
                "relation_forward": {},
                "relation_inverse": {},
                "local_forward": {},
                "local_inverse": {},
                "subject_forward": {},
                "subject_inverse": {},
                "global_forward": {},
                "global_inverse": {},
            }

        def accumulate(events: List[List[int]], target_tables: Dict[str, Dict], decay: float) -> None:
            for head, rel, tail, tim in events:
                head = int(head)
                rel = int(rel)
                tail = int(tail)
                time_weight = float(np.exp(-decay * max(0, query_time - int(tim))))

                rel_forward_weights = target_tables["relation_forward"].setdefault(rel, {})
                rel_forward_weights[tail] = rel_forward_weights.get(tail, 0.0) + time_weight
                rel_inverse_weights = target_tables["relation_inverse"].setdefault(rel, {})
                rel_inverse_weights[head] = rel_inverse_weights.get(head, 0.0) + time_weight
                local_forward_weights = target_tables["local_forward"].setdefault((head, rel), {})
                local_forward_weights[tail] = local_forward_weights.get(tail, 0.0) + time_weight
                local_inverse_weights = target_tables["local_inverse"].setdefault((tail, rel), {})
                local_inverse_weights[head] = local_inverse_weights.get(head, 0.0) + time_weight
                subject_forward_weights = target_tables["subject_forward"].setdefault(head, {})
                subject_forward_weights[tail] = subject_forward_weights.get(tail, 0.0) + time_weight
                subject_inverse_weights = target_tables["subject_inverse"].setdefault(tail, {})
                subject_inverse_weights[head] = subject_inverse_weights.get(head, 0.0) + time_weight
                target_tables["global_forward"][tail] = target_tables["global_forward"].get(tail, 0.0) + time_weight
                target_tables["global_inverse"][head] = target_tables["global_inverse"].get(head, 0.0) + time_weight

        short_tables = empty_tables()
        long_tables = empty_tables()
        accumulate(self._prior_events(query_time), short_tables, float(self.data_config.prior_time_decay))
        if float(getattr(self.data_config, "long_prior_mix", 0.0)) > 0:
            accumulate(
                self._long_prior_events(query_time),
                long_tables,
                float(getattr(self.data_config, "long_prior_time_decay", self.data_config.prior_time_decay)),
            )

        tables = dict(short_tables)
        for name, value in long_tables.items():
            tables[f"long_{name}"] = value
        cache_size = int(getattr(self.data_config, "prior_table_cache_size", 0))
        if cache_size > 0 and len(self._prior_table_cache) >= cache_size:
            self._prior_table_cache.clear()
        self._prior_table_cache[query_time] = tables
        return tables

    def _mix_long_weights(self, short_weights: Dict[int, float], long_weights: Dict[int, float]) -> Dict[int, float]:
        mix = float(getattr(self.data_config, "long_prior_mix", 0.0))
        if mix <= 0 or not long_weights:
            return dict(short_weights)
        weights = dict(short_weights)
        for entity, weight in long_weights.items():
            weights[int(entity)] = weights.get(int(entity), 0.0) + mix * float(weight)
        return weights

    def warm_prior_cache(self) -> int:
        if int(getattr(self.data_config, "prior_table_cache_size", 0)) > 0:
            print(
                {
                    "stage": "prior_cache_warmup_skip",
                    "split": self.split,
                    "reason": "bounded_prior_table_cache",
                    "prior_table_cache_size": int(getattr(self.data_config, "prior_table_cache_size", 0)),
                },
                flush=True,
            )
            return 0
        print(
            {
                "stage": "prior_cache_warmup_start",
                "split": self.split,
                "time_points": int(len(self.sorted_times)),
            },
            flush=True,
        )
        start = time.perf_counter()
        for query_time in tqdm(self.sorted_times, desc=f"warm-prior-{self.split}", mininterval=5.0):
            self._prior_tables(int(query_time))
        print(
            {
                "stage": "prior_cache_warmup_done",
                "split": self.split,
                "cached_time_points": int(len(self._prior_table_cache)),
                "seconds": time.perf_counter() - start,
            },
            flush=True,
        )
        return len(self._prior_table_cache)

    def _copy_signal(self, subject_history, subject_history_t, relation: int, query_time: int) -> Dict[str, List[float]]:
        base_relation = int(relation) % self.num_relations
        weights: Dict[int, float] = {}
        for block, tim in zip(subject_history, subject_history_t):
            if block is None or len(block) == 0:
                continue
            time_weight = float(np.exp(-self.data_config.prior_time_decay * max(0, query_time - int(tim))))
            for rel, nbr in block:
                if int(rel) == base_relation:
                    weights[int(nbr)] = weights.get(int(nbr), 0.0) + time_weight
        return self._normalize_signal(weights, top_k=int(getattr(self.data_config, "copy_topk", 64)))

    def _relation_transfer_signal(
            self,
            subject_history,
            subject_history_t,
            relation: int,
            query_time: int,
    ) -> Dict[str, List[float]]:
        top_k = int(getattr(self.data_config, "relation_transfer_topk", 0))
        if top_k <= 0:
            return {"entities": [], "weights": []}
        base_relation = int(relation) % self.num_relations
        transfer_row = self.relation_transfer_weights.get(base_relation, {})
        if not transfer_row:
            return {"entities": [], "weights": []}
        min_sim = float(getattr(self.data_config, "relation_transfer_min_sim", 0.0))
        decay = float(getattr(self.data_config, "relation_transfer_time_decay", self.data_config.prior_time_decay))
        weights: Dict[int, float] = {}
        for block, tim in zip(subject_history, subject_history_t):
            if block is None or len(block) == 0:
                continue
            time_weight = float(np.exp(-decay * max(0, query_time - int(tim))))
            for rel, nbr in block:
                history_relation = int(rel) % self.num_relations
                if history_relation == base_relation:
                    continue
                similarity = float(transfer_row.get(history_relation, 0.0))
                if similarity < min_sim:
                    continue
                weights[int(nbr)] = weights.get(int(nbr), 0.0) + time_weight * similarity
        return self._normalize_signal(weights, top_k=top_k)

    def _relation_prior_signal(self, query_time: int, relation: int) -> Dict[str, List[float]]:
        cache_key = (int(query_time), int(relation))
        if cache_key in self._relation_prior_cache:
            return self._relation_prior_cache[cache_key]
        base_relation = int(relation) % self.num_relations
        inverse = int(relation) >= self.num_relations
        table_name = "relation_inverse" if inverse else "relation_forward"
        tables = self._prior_tables(query_time)
        weights = self._mix_long_weights(
            tables[table_name].get(base_relation, {}),
            tables[f"long_{table_name}"].get(base_relation, {}),
        )
        signal = self._normalize_signal(weights, top_k=self.data_config.prior_topk)
        self._relation_prior_cache[cache_key] = signal
        return signal

    def _local_prior_signal(self, query_time: int, subject: int, relation: int) -> Dict[str, List[float]]:
        cache_key = (int(query_time), int(subject), int(relation))
        if cache_key in self._local_prior_cache:
            return self._local_prior_cache[cache_key]
        base_relation = int(relation) % self.num_relations
        inverse = int(relation) >= self.num_relations
        table_name = "local_inverse" if inverse else "local_forward"
        tables = self._prior_tables(query_time)
        weights = self._mix_long_weights(
            tables[table_name].get((int(subject), base_relation), {}),
            tables[f"long_{table_name}"].get((int(subject), base_relation), {}),
        )
        signal = self._normalize_signal(weights, top_k=int(getattr(self.data_config, "local_prior_topk", 64)))
        self._local_prior_cache[cache_key] = signal
        return signal

    def _subject_prior_signal(self, query_time: int, subject: int, relation: int) -> Dict[str, List[float]]:
        cache_key = (int(query_time), int(subject), int(relation))
        if cache_key in self._subject_prior_cache:
            return self._subject_prior_cache[cache_key]
        inverse = int(relation) >= self.num_relations
        table_name = "subject_inverse" if inverse else "subject_forward"
        tables = self._prior_tables(query_time)
        weights = self._mix_long_weights(
            tables[table_name].get(int(subject), {}),
            tables[f"long_{table_name}"].get(int(subject), {}),
        )
        signal = self._normalize_signal(weights, top_k=self.data_config.subject_prior_topk)
        self._subject_prior_cache[cache_key] = signal
        return signal

    def _global_prior_signal(self, query_time: int, relation: int) -> Dict[str, List[float]]:
        cache_key = (int(query_time), int(relation))
        if cache_key in self._global_prior_cache:
            return self._global_prior_cache[cache_key]
        inverse = int(relation) >= self.num_relations
        table_name = "global_inverse" if inverse else "global_forward"
        tables = self._prior_tables(query_time)
        weights = self._mix_long_weights(tables[table_name], tables[f"long_{table_name}"])
        signal = self._normalize_signal(weights, top_k=self.data_config.global_prior_topk)
        self._global_prior_cache[cache_key] = signal
        return signal

    def _use_heavy_evidence(self, raw_index: int, relation: int, query_time: int) -> bool:
        if self.split == "valid":
            return bool(getattr(self.data_config, "valid_heavy_evidence", True))
        if self.split != "train":
            return bool(getattr(self.data_config, "eval_heavy_evidence", True))
        rate = float(getattr(self.data_config, "train_heavy_evidence_rate", 1.0))
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        bucket = int(max(1, min(10000, round(rate * 10000))))
        key = (
            int(raw_index) * 1103515245
            + int(query_time) * 12345
            + int(relation) * 2654435761
        ) & 0x7FFFFFFF
        if bool(getattr(self.data_config, "dynamic_heavy_evidence", True)):
            key = (key + int(self.current_epoch) * 97531) & 0x7FFFFFFF
        return key % 10000 < bucket

    def _sample_weight(self, query_time: int) -> float:
        if self.split != "train":
            return 1.0
        weight = float(getattr(self.data_config, "recent_train_weight", 0.0))
        if weight <= 0:
            return 1.0
        span = max(1, self.train_max_time - self.train_min_time)
        normalized_time = (int(query_time) - self.train_min_time) / span
        normalized_time = float(np.clip(normalized_time, 0.0, 1.0))
        power = max(0.1, float(getattr(self.data_config, "recent_train_power", 1.0)))
        return float(1.0 + weight * (normalized_time ** power))

    def _path_proxy_signal(
            self,
            copy_signal: Dict[str, List[float]],
            local_prior: Dict[str, List[float]],
            relation_prior: Dict[str, List[float]],
            subject_prior: Dict[str, List[float]],
    ) -> Dict[str, List[float]]:
        weights: Dict[int, float] = {}

        def collect(signal: Dict[str, List[float]], scale: float) -> None:
            for entity, weight in zip(signal.get("entities", []), signal.get("weights", [])):
                entity_id = int(entity)
                weights[entity_id] = weights.get(entity_id, 0.0) + scale * float(weight)

        collect(local_prior, 1.00)
        collect(copy_signal, 0.75)
        collect(subject_prior, 0.55)
        collect(relation_prior, 0.35)
        return self._normalize_signal(weights, top_k=self.data_config.path_prior_topk)

    def _local_edges(
            self,
            subject: int,
            query_time: int,
            history_events: List[List[int]],
    ) -> List[TemporalEdge]:
        cache_size = int(getattr(self.data_config, "local_graph_cache_size", 0))
        cache_key = (
            int(subject),
            int(query_time),
            int(self.data_config.max_hops),
            int(self.data_config.max_nodes),
            int(self.data_config.history_window),
        )
        if cache_size > 0 and cache_key in self._local_edges_cache:
            return self._local_edges_cache[cache_key]
        local_edges = extract_local_temporal_graph(
            subject=subject,
            history_events=history_events,
            max_hops=self.data_config.max_hops,
            max_nodes=self.data_config.max_nodes,
        )
        if cache_size > 0:
            if len(self._local_edges_cache) >= cache_size:
                self._local_edges_cache.clear()
            self._local_edges_cache[cache_key] = local_edges
        return local_edges

    def _path_prior_signal(
            self,
            local_edges: List[TemporalEdge],
            subject: int,
            relation: int,
            query_time: int,
    ) -> Dict[str, List[float]]:
        if not hasattr(self, "_path_prior_cache"):
            self._path_prior_cache = {}
        cache_key = (int(subject), int(relation), int(query_time))
        if cache_key in self._path_prior_cache:
            return self._path_prior_cache[cache_key]
        if not local_edges or self.data_config.path_prior_topk <= 0:
            return {"entities": [], "weights": []}

        query_relation = int(relation)
        base_relation = query_relation % self.num_relations
        adjacency: Dict[int, List[TemporalEdge]] = {}
        for edge in local_edges:
            if int(edge.timestamp) >= int(query_time):
                continue
            adjacency.setdefault(int(edge.src), []).append(edge)
            if self.data_config.path_prior_use_reverse:
                reverse_edge = TemporalEdge(
                    src=int(edge.dst),
                    rel=int(edge.rel) + self.num_relations,
                    dst=int(edge.src),
                    timestamp=int(edge.timestamp),
                )
                adjacency.setdefault(reverse_edge.src, []).append(reverse_edge)

        max_branches = max(1, int(self.data_config.path_prior_max_branches))
        for node, edges in adjacency.items():
            adjacency[node] = sorted(edges, key=lambda item: item.timestamp, reverse=True)[:max_branches]

        weights: Dict[int, float] = {}
        max_len = max(1, int(self.data_config.max_path_len))
        length_decay = float(self.data_config.path_prior_length_decay)
        relation_bonus = float(self.data_config.path_prior_relation_bonus)
        state_limit = max(1, int(getattr(self.data_config, "path_prior_state_limit", 96)))
        states = [(int(subject), -10 ** 12, 1.0)]
        for depth in range(1, max_len + 1):
            next_state_scores: Dict[Tuple[int, int], float] = {}
            for node, previous_time, prefix_score in states:
                for edge in adjacency.get(int(node), []):
                    edge_time = int(edge.timestamp)
                    if edge_time < int(previous_time):
                        continue
                    dst = int(edge.dst)
                    if dst == int(subject):
                        continue
                    edge_exact = int(edge.rel) == query_relation
                    edge_base = int(edge.rel) % self.num_relations == base_relation
                    recency = float(np.exp(-self.data_config.prior_time_decay * max(0, int(query_time) - edge_time)))
                    relation_weight = 1.0
                    if edge_exact:
                        relation_weight += relation_bonus
                    elif edge_base:
                        relation_weight += 0.5 * relation_bonus
                    path_score = prefix_score * recency * (length_decay ** max(0, depth - 1)) * relation_weight
                    weights[dst] = weights.get(dst, 0.0) + path_score
                    state_key = (dst, edge_time)
                    next_state_scores[state_key] = max(next_state_scores.get(state_key, 0.0), path_score)
            if not next_state_scores:
                break
            states = [
                (node, edge_time, score)
                for (node, edge_time), score in sorted(
                    next_state_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:state_limit]
            ]
        signal = self._normalize_signal(weights, top_k=self.data_config.path_prior_topk)
        path_cache_size = int(getattr(self.data_config, "path_prior_cache_size", 50000))
        if path_cache_size > 0 and len(self._path_prior_cache) >= path_cache_size:
            self._path_prior_cache.clear()
        if path_cache_size != 0:
            self._path_prior_cache[cache_key] = signal
        return signal

    def __getitem__(self, index: int) -> Dict:
        raw_index = index
        inverse_sample = False
        if self.split == "train" and self.data_config.use_inverse_train:
            if self.data_config.inverse_train_mode == "paired":
                inverse_sample = index >= len(self.data)
            elif self.data_config.inverse_train_mode == "sampled":
                inverse_sample = bool(np.random.random() < 0.5)
            else:
                raise ValueError(f"Unsupported inverse_train_mode: {self.data_config.inverse_train_mode}")
        if inverse_sample and self.data_config.inverse_train_mode == "paired":
            index = index - len(self.data)
        subject, relation, target, query_time = [int(x) for x in self.data[index].tolist()]
        if self.use_strict_history:
            subject_history, subject_history_t = self._subject_history_for(subject, query_time)
            object_history, object_history_t = self._object_history_for(target, query_time)
        else:
            subject_history = self.subject_history[index]
            subject_history_t = self.subject_history_t[index]
            object_history = self.object_history[index]
            object_history_t = self.object_history_t[index]
        if inverse_sample:
            subject, target = target, subject
            relation = relation + self.num_relations
            subject_history, object_history = object_history, subject_history
            subject_history_t, object_history_t = object_history_t, subject_history_t
        if self.split == "train":
            positive_map = self.train_inverse_positive_map if inverse_sample else self.train_positive_map
            positive_entities = sorted(
                int(entity)
                for entity in positive_map.get((int(subject), int(relation), int(query_time)), {int(target)})
            )
            if int(target) not in positive_entities:
                positive_entities.append(int(target))
        else:
            positive_entities = [int(target)]
        concurrent_subject_relations, concurrent_time_relations = self._concurrent_query_relations(
            subject=subject,
            query_time=query_time,
        )
        copy_signal = self._copy_signal(
            subject_history=subject_history,
            subject_history_t=subject_history_t,
            relation=relation % self.num_relations,
            query_time=query_time,
        )
        relation_transfer = self._relation_transfer_signal(
            subject_history=subject_history,
            subject_history_t=subject_history_t,
            relation=relation,
            query_time=query_time,
        )
        relation_prior = self._relation_prior_signal(query_time=query_time, relation=relation)
        local_prior = self._local_prior_signal(query_time=query_time, subject=subject, relation=relation)
        subject_prior = self._subject_prior_signal(query_time=query_time, subject=subject, relation=relation)
        global_prior = self._global_prior_signal(query_time=query_time, relation=relation)
        use_heavy_evidence = self._use_heavy_evidence(
            raw_index=raw_index,
            relation=relation,
            query_time=query_time,
        )
        if use_heavy_evidence:
            history_events = self._history_events(query_time)
            local_edges = self._local_edges(
                subject=subject,
                history_events=history_events,
                query_time=query_time,
            )
            path_prior = self._path_prior_signal(
                local_edges=local_edges,
                subject=subject,
                relation=relation,
                query_time=query_time,
            )
        else:
            history_events = []
            local_edges = []
            path_prior = self._path_proxy_signal(
                copy_signal=copy_signal,
                local_prior=local_prior,
                relation_prior=relation_prior,
                subject_prior=subject_prior,
            )
        local_graph = build_dgl_subgraph(local_edges, num_relations=self.num_relations)
        return {
            "index": raw_index,
            "subject": subject,
            "relation": relation,
            "target": target,
            "positive_entities": positive_entities,
            "time": query_time,
            "subject_history": subject_history,
            "subject_history_t": subject_history_t,
            "object_history": object_history,
            "object_history_t": object_history_t,
            "history_events": history_events,
            "local_edges": local_edges,
            "local_graph": local_graph,
            "is_inverse": inverse_sample,
            "copy_entities": copy_signal["entities"],
            "copy_weights": copy_signal["weights"],
            "relation_transfer_entities": relation_transfer["entities"],
            "relation_transfer_weights": relation_transfer["weights"],
            "concurrent_subject_relations": concurrent_subject_relations,
            "concurrent_time_relations": concurrent_time_relations,
            "relation_prior_entities": relation_prior["entities"],
            "relation_prior_weights": relation_prior["weights"],
            "local_prior_entities": local_prior["entities"],
            "local_prior_weights": local_prior["weights"],
            "subject_prior_entities": subject_prior["entities"],
            "subject_prior_weights": subject_prior["weights"],
            "global_prior_entities": global_prior["entities"],
            "global_prior_weights": global_prior["weights"],
            "path_prior_entities": path_prior["entities"],
            "path_prior_weights": path_prior["weights"],
            "use_heavy_evidence": use_heavy_evidence,
            "sample_weight": self._sample_weight(query_time),
        }


EVIDENCE_SOURCE_KEYS = [
    ("copy_entities", "copy_weights"),
    ("local_prior_entities", "local_prior_weights"),
    ("relation_prior_entities", "relation_prior_weights"),
    ("subject_prior_entities", "subject_prior_weights"),
    ("global_prior_entities", "global_prior_weights"),
    ("path_prior_entities", "path_prior_weights"),
]


def _pack_evidence_tensors(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    max_items = 0
    for item in batch:
        for entity_key, _ in EVIDENCE_SOURCE_KEYS:
            max_items = max(max_items, len(item.get(entity_key, [])))
    entity_tensor = torch.full(
        (len(batch), len(EVIDENCE_SOURCE_KEYS), max_items),
        fill_value=-1,
        dtype=torch.long,
    )
    weight_tensor = torch.zeros(
        (len(batch), len(EVIDENCE_SOURCE_KEYS), max_items),
        dtype=torch.float32,
    )
    if max_items == 0:
        return entity_tensor, weight_tensor
    for row, item in enumerate(batch):
        for source_idx, (entity_key, weight_key) in enumerate(EVIDENCE_SOURCE_KEYS):
            entities = item.get(entity_key, [])
            weights = item.get(weight_key, [])
            count = min(len(entities), len(weights), max_items)
            if count <= 0:
                continue
            entity_tensor[row, source_idx, :count] = torch.tensor(entities[:count], dtype=torch.long)
            weight_tensor[row, source_idx, :count] = torch.tensor(weights[:count], dtype=torch.float32)
    return entity_tensor, weight_tensor


def collate_temporal_batch(batch: List[Dict]) -> Dict:
    batch = sorted(batch, key=lambda item: int(item.get("use_heavy_evidence", False)), reverse=True)
    evidence_entities, evidence_weights = _pack_evidence_tensors(batch)
    return {
        "indices": [item["index"] for item in batch],
        "subjects": torch.tensor([item["subject"] for item in batch], dtype=torch.long),
        "relations": torch.tensor([item["relation"] for item in batch], dtype=torch.long),
        "targets": torch.tensor([item["target"] for item in batch], dtype=torch.long),
        "positive_entities": [item.get("positive_entities", [item["target"]]) for item in batch],
        "times": torch.tensor([item["time"] for item in batch], dtype=torch.long),
        "subject_histories": [item["subject_history"] for item in batch],
        "subject_history_times": [item["subject_history_t"] for item in batch],
        "object_histories": [item["object_history"] for item in batch],
        "object_history_times": [item["object_history_t"] for item in batch],
        "history_events": [item["history_events"] for item in batch],
        "local_edges": [item["local_edges"] for item in batch],
        "local_graphs": [item["local_graph"] for item in batch],
        "is_inverse": torch.tensor([item["is_inverse"] for item in batch], dtype=torch.bool),
        "copy_entities": [item["copy_entities"] for item in batch],
        "copy_weights": [item["copy_weights"] for item in batch],
        "relation_transfer_entities": [item["relation_transfer_entities"] for item in batch],
        "relation_transfer_weights": [item["relation_transfer_weights"] for item in batch],
        "concurrent_subject_relations": [item["concurrent_subject_relations"] for item in batch],
        "concurrent_time_relations": [item["concurrent_time_relations"] for item in batch],
        "relation_prior_entities": [item["relation_prior_entities"] for item in batch],
        "relation_prior_weights": [item["relation_prior_weights"] for item in batch],
        "local_prior_entities": [item["local_prior_entities"] for item in batch],
        "local_prior_weights": [item["local_prior_weights"] for item in batch],
        "subject_prior_entities": [item["subject_prior_entities"] for item in batch],
        "subject_prior_weights": [item["subject_prior_weights"] for item in batch],
        "global_prior_entities": [item["global_prior_entities"] for item in batch],
        "global_prior_weights": [item["global_prior_weights"] for item in batch],
        "path_prior_entities": [item["path_prior_entities"] for item in batch],
        "path_prior_weights": [item["path_prior_weights"] for item in batch],
        "evidence_entities_tensor": evidence_entities,
        "evidence_weights_tensor": evidence_weights,
        "use_heavy_evidence": torch.tensor([item.get("use_heavy_evidence", False) for item in batch], dtype=torch.bool),
        "sample_weights": torch.tensor([item.get("sample_weight", 1.0) for item in batch], dtype=torch.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one dataset sample")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    data_config = DataConfig(dataset_dir=args.dataset_dir)
    dataset = TemporalReasoningDataset(data_config=data_config, split=args.split)
    sample = dataset[args.index]
    print(
        {
            "num_entities": dataset.num_entities,
            "num_relations": dataset.num_relations,
            "subject": sample["subject"],
            "relation": sample["relation"],
            "target": sample["target"],
            "time": sample["time"],
            "history_events": len(sample["history_events"]),
            "local_edges": len(sample["local_edges"]),
        }
    )


if __name__ == "__main__":
    main()
