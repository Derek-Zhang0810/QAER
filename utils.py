import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import torch


def ensure_dir(path: str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str) -> torch.device:
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred)


def _parse_id_text_line(line: str):
    if "\t" in line:
        parts = [part.strip() for part in line.split("\t") if part.strip()]
    else:
        parts = line.split()
    if len(parts) < 2:
        return None

    first, last = parts[0], parts[-1]
    if first.isdigit():
        return int(first), " ".join(parts[1:]).strip()
    if last.isdigit():
        return int(last), " ".join(parts[:-1]).strip()
    return None


def load_id_text_mapping(
        file_path: str,
        size: int,
        fallback_prefix: str,
        fallback_file_path: str = "",
) -> List[str]:
    mapping = {idx: f"{fallback_prefix} {idx}" for idx in range(size)}
    candidates = [Path(file_path)]
    if fallback_file_path:
        candidates.append(Path(fallback_file_path))

    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        return [mapping[idx] for idx in range(size)]

    with open(source, "r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            parsed = _parse_id_text_line(line)
            if parsed is None:
                continue
            idx, text = parsed
            if 0 <= idx < size and text:
                mapping[idx] = text
    return [mapping[idx] for idx in range(size)]


def save_json(payload: Dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def load_json(input_path: str) -> Dict:
    with open(input_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def build_filter_map(all_facts: np.ndarray) -> Dict[Tuple[int, int, int], Set[int]]:
    filters: Dict[Tuple[int, int, int], Set[int]] = {}
    for head, rel, tail, tim in all_facts.tolist():
        key = (int(head), int(rel), int(tim))
        filters.setdefault(key, set()).add(int(tail))
    return filters


def filtered_rank(scores: torch.Tensor, target: int, filtered_tails: Set[int]) -> int:
    masked = scores.detach().clone()
    target_score = masked[target].item()
    for tail in filtered_tails:
        if tail != target:
            masked[tail] = -1e9
    greater = masked > target_score
    tied = masked == target_score
    if 0 <= int(target) < tied.numel():
        tied[int(target)] = False
    for tail in filtered_tails:
        if tail != target and 0 <= int(tail) < tied.numel():
            tied[int(tail)] = False
    rank = int(greater.sum().item() + tied.sum().item()) + 1
    return rank


def aggregate_ranking_metrics(ranks: Sequence[int]) -> Dict[str, float]:
    if not ranks:
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
    reciprocal = [1.0 / rank for rank in ranks]
    return {
        "MRR": float(np.mean(reciprocal)),
        "Hits@1": float(np.mean([rank <= 1 for rank in ranks])),
        "Hits@3": float(np.mean([rank <= 3 for rank in ranks])),
        "Hits@10": float(np.mean([rank <= 10 for rank in ranks])),
    }


def maybe_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def exp_time_decay(delta_t: float, gamma: float) -> float:
    return math.exp(-gamma * max(0.0, float(delta_t)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared utility self-test")
    parser.add_argument("--demo", action="store_true", help="Run a toy metric demo")
    args = parser.parse_args()

    if args.demo:
        scores = torch.tensor([0.2, 0.9, 0.5, 0.1])
        rank = filtered_rank(scores, target=1, filtered_tails={1, 2})
        metrics = aggregate_ranking_metrics([rank, 2, 3, 1])
        print({"rank": rank, "metrics": metrics})
    else:
        print("Utilities loaded. Use --demo to run the toy check.")


if __name__ == "__main__":
    main()
