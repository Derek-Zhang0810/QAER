import argparse
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import torch


@dataclass
class TemporalEdge:
    src: int
    rel: int
    dst: int
    timestamp: int


@dataclass
class TemporalPath:
    nodes: List[int]
    relations: List[int]
    timestamps: List[int]
    score: float = 0.0


class LocalSubgraph:
    def __init__(
            self,
            node_ids: List[int],
            edge_src: List[int],
            edge_dst: List[int],
            edge_types: List[int],
            edge_times: List[int],
            ids: Dict[int, int],
    ):
        self.ids = ids
        self.ndata = {
            "id": torch.tensor(node_ids, dtype=torch.long).view(-1, 1),
            "norm": self._compute_norm(len(node_ids), edge_dst).view(-1, 1),
        }
        self.edata = {
            "type": torch.tensor(edge_types, dtype=torch.long),
            "time": torch.tensor(edge_times, dtype=torch.long),
        }
        self._edge_src = torch.tensor(edge_src, dtype=torch.long)
        self._edge_dst = torch.tensor(edge_dst, dtype=torch.long)

    @staticmethod
    def _compute_norm(num_nodes: int, edge_dst: List[int]) -> torch.Tensor:
        if num_nodes == 0:
            return torch.ones(1, dtype=torch.float32)
        in_deg = torch.zeros(num_nodes, dtype=torch.float32)
        if edge_dst:
            dst_tensor = torch.tensor(edge_dst, dtype=torch.long)
            in_deg.index_add_(0, dst_tensor, torch.ones_like(dst_tensor, dtype=torch.float32))
        in_deg[in_deg == 0] = 1.0
        return 1.0 / in_deg

    def num_nodes(self) -> int:
        return int(self.ndata["id"].shape[0])

    def num_edges(self) -> int:
        return int(self.edata["type"].shape[0])

    def edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._edge_src, self._edge_dst

    def to(self, device: torch.device):
        self.ndata = {key: value.to(device) for key, value in self.ndata.items()}
        self.edata = {key: value.to(device) for key, value in self.edata.items()}
        self._edge_src = self._edge_src.to(device)
        self._edge_dst = self._edge_dst.to(device)
        return self


def extract_local_temporal_graph(
        subject: int,
        history_events: Sequence[Sequence[int]],
        max_hops: int = 2,
        max_nodes: int = 128,
) -> List[TemporalEdge]:
    adjacency_undirected: Dict[int, List[TemporalEdge]] = {}
    for head, rel, tail, tim in history_events:
        edge = TemporalEdge(int(head), int(rel), int(tail), int(tim))
        adjacency_undirected.setdefault(edge.src, []).append(edge)
        adjacency_undirected.setdefault(edge.dst, []).append(edge)

    visited = {subject}
    frontier = {subject}
    selected: Dict[Tuple[int, int, int, int], TemporalEdge] = {}

    for _ in range(max_hops):
        next_frontier = set()
        for node in frontier:
            for edge in adjacency_undirected.get(node, []):
                key = (edge.src, edge.rel, edge.dst, edge.timestamp)
                selected[key] = edge
                next_frontier.add(edge.src)
                next_frontier.add(edge.dst)
                if len(visited | next_frontier) >= max_nodes:
                    break
            if len(visited | next_frontier) >= max_nodes:
                break
        frontier = next_frontier - visited
        visited |= next_frontier
        if not frontier or len(visited) >= max_nodes:
            break
    return sorted(selected.values(), key=lambda item: (item.timestamp, item.src, item.rel, item.dst))


def build_dgl_subgraph(edges: Sequence[TemporalEdge], num_relations: int) -> LocalSubgraph:
    if not edges:
        return LocalSubgraph(node_ids=[0], edge_src=[], edge_dst=[], edge_types=[], edge_times=[], ids={0: 0})

    node_ids = sorted({edge.src for edge in edges} | {edge.dst for edge in edges})
    mapping = {node_id: idx for idx, node_id in enumerate(node_ids)}
    src = [mapping[edge.src] for edge in edges]
    dst = [mapping[edge.dst] for edge in edges]
    rev_src = [mapping[edge.dst] for edge in edges]
    rev_dst = [mapping[edge.src] for edge in edges]
    edge_types = [edge.rel for edge in edges] + [edge.rel + num_relations for edge in edges]
    edge_times = [edge.timestamp for edge in edges] + [edge.timestamp for edge in edges]
    return LocalSubgraph(
        node_ids=node_ids,
        edge_src=src + rev_src,
        edge_dst=dst + rev_dst,
        edge_types=edge_types,
        edge_times=edge_times,
        ids=mapping,
    )


def enumerate_temporal_paths(
        edges: Sequence[TemporalEdge],
        source: int,
        target: int,
        max_len: int = 3,
        max_paths: int = 20,
        query_time: int = 10 ** 12,
) -> List[TemporalPath]:
    adjacency: Dict[int, List[TemporalEdge]] = {}
    for edge in edges:
        adjacency.setdefault(edge.src, []).append(edge)

    collected: List[TemporalPath] = []

    def dfs(node: int, visited: List[int], rels: List[int], times: List[int]) -> None:
        if len(collected) >= max_paths:
            return
        if node == target and rels:
            collected.append(TemporalPath(nodes=visited.copy(), relations=rels.copy(), timestamps=times.copy()))
            return
        if len(rels) >= max_len:
            return
        previous_time = times[-1] if times else -10 ** 12
        for edge in adjacency.get(node, []):
            if edge.timestamp >= query_time:
                continue
            if edge.timestamp < previous_time:
                continue
            if edge.dst in visited:
                continue
            visited.append(edge.dst)
            rels.append(edge.rel)
            times.append(edge.timestamp)
            dfs(edge.dst, visited, rels, times)
            visited.pop()
            rels.pop()
            times.pop()

    dfs(source, [source], [], [])
    return collected


def temporal_path_to_text(path: TemporalPath) -> str:
    chunks = [f"{path.nodes[0]}"]
    for rel, node, tim in zip(path.relations, path.nodes[1:], path.timestamps):
        chunks.append(f"-[{rel}@{tim}]-> {node}")
    return " ".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal graph self-test")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        history = [
            [0, 1, 2, 1],
            [2, 0, 3, 2],
            [0, 2, 4, 2],
            [4, 1, 3, 3],
        ]
        edges = extract_local_temporal_graph(subject=0, history_events=history, max_hops=2)
        graph = build_dgl_subgraph(edges, num_relations=3)
        paths = enumerate_temporal_paths(edges, source=0, target=3, max_len=3, query_time=5)
        print(
            {
                "num_edges": len(edges),
                "num_graph_nodes": graph.num_nodes(),
                "num_graph_edges": graph.num_edges(),
                "paths": [asdict(path) for path in paths],
            }
        )
    else:
        print("Temporal graph tools ready. Use --demo for a toy example.")


if __name__ == "__main__":
    main()
