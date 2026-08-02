import argparse
import os
import pickle
from pathlib import Path
from typing import Dict

import numpy as np


def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), "r", encoding="utf-8") as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), "r", encoding="utf-8") as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()
    return np.array(quadrupleList), np.asarray(times)


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), "r", encoding="utf-8") as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])
    raise RuntimeError(f"Cannot read stat file: {os.path.join(inPath, fileName)}")


def get_data_with_t(data, tim):
    x = data[np.where(data[:, 3] == tim)].copy()
    x = np.delete(x, 3, 1)
    return x


def get_big_graph(data, num_rels):
    src, rel, dst = data.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    src_all = np.concatenate((src, dst))
    dst_all = np.concatenate((dst, src))
    rel_all = np.concatenate((rel, rel + num_rels))
    return {
        "node_ids": uniq_v.tolist(),
        "src": src_all.tolist(),
        "dst": dst_all.tolist(),
        "rel": rel_all.tolist(),
    }


def _flush_history_buffers(
        num_e,
        history_len,
        s_his,
        o_his,
        s_his_t,
        o_his_t,
        s_his_cache,
        o_his_cache,
        s_his_cache_t,
        o_his_cache_t,
):
    for ee in range(num_e):
        if len(s_his_cache[ee]) != 0:
            if len(s_his[ee]) >= history_len:
                s_his[ee].pop(0)
                s_his_t[ee].pop(0)
            s_his[ee].append(s_his_cache[ee].copy())
            s_his_t[ee].append(s_his_cache_t[ee])
            s_his_cache[ee] = []
            s_his_cache_t[ee] = None
        if len(o_his_cache[ee]) != 0:
            if len(o_his[ee]) >= history_len:
                o_his[ee].pop(0)
                o_his_t[ee].pop(0)
            o_his[ee].append(o_his_cache[ee].copy())
            o_his_t[ee].append(o_his_cache_t[ee])
            o_his_cache[ee] = []
            o_his_cache_t[ee] = None


def _process_split(
        data,
        num_e,
        history_len,
        s_his,
        o_his,
        s_his_t,
        o_his_t,
        s_his_cache,
        o_his_cache,
        s_his_cache_t,
        o_his_cache_t,
        latest_t,
):
    s_history_data = [[] for _ in range(len(data))]
    o_history_data = [[] for _ in range(len(data))]
    s_history_data_t = [[] for _ in range(len(data))]
    o_history_data_t = [[] for _ in range(len(data))]

    for i, item in enumerate(data):
        if i % 10000 == 0:
            print(f"history-build {i}/{len(data)}")
        t = int(item[3])
        if latest_t != t:
            _flush_history_buffers(
                num_e,
                history_len,
                s_his,
                o_his,
                s_his_t,
                o_his_t,
                s_his_cache,
                o_his_cache,
                s_his_cache_t,
                o_his_cache_t,
            )
            latest_t = t

        s = int(item[0])
        r = int(item[1])
        o = int(item[2])
        s_history_data[i] = s_his[s].copy()
        o_history_data[i] = o_his[o].copy()
        s_history_data_t[i] = s_his_t[s].copy()
        o_history_data_t[i] = o_his_t[o].copy()

        if len(s_his_cache[s]) == 0:
            s_his_cache[s] = np.array([[r, o]])
        else:
            s_his_cache[s] = np.concatenate((s_his_cache[s], [[r, o]]), axis=0)
        s_his_cache_t[s] = t

        if len(o_his_cache[o]) == 0:
            o_his_cache[o] = np.array([[r, s]])
        else:
            o_his_cache[o] = np.concatenate((o_his_cache[o], [[r, s]]), axis=0)
        o_his_cache_t[o] = t

    return s_history_data, o_history_data, s_history_data_t, o_history_data_t, latest_t


def build_history_graphs(dataset_dir: str, history_len: int = 10, output_dir: str = "") -> Dict[str, Path]:
    dataset_dir = str(dataset_dir)
    output_root = Path(output_dir or dataset_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    graph_dict_train = {}
    train_data, train_times = load_quadruples(dataset_dir, "train.txt")
    test_data, _ = load_quadruples(dataset_dir, "test.txt")
    dev_data, _ = load_quadruples(dataset_dir, "valid.txt")
    num_e, num_r = get_total_number(dataset_dir, "stat.txt")

    s_his = [[] for _ in range(num_e)]
    o_his = [[] for _ in range(num_e)]
    s_his_t = [[] for _ in range(num_e)]
    o_his_t = [[] for _ in range(num_e)]
    latest_t = None
    s_his_cache = [[] for _ in range(num_e)]
    o_his_cache = [[] for _ in range(num_e)]
    s_his_cache_t = [None for _ in range(num_e)]
    o_his_cache_t = [None for _ in range(num_e)]

    for tim in train_times:
        data = get_data_with_t(train_data, tim)
        graph_dict_train[int(tim)] = get_big_graph(data, num_r)

    train_pack = _process_split(
        train_data,
        num_e,
        history_len,
        s_his,
        o_his,
        s_his_t,
        o_his_t,
        s_his_cache,
        o_his_cache,
        s_his_cache_t,
        o_his_cache_t,
        latest_t,
    )
    latest_t = train_pack[-1]
    dev_pack = _process_split(
        dev_data,
        num_e,
        history_len,
        s_his,
        o_his,
        s_his_t,
        o_his_t,
        s_his_cache,
        o_his_cache,
        s_his_cache_t,
        o_his_cache_t,
        latest_t,
    )
    latest_t = dev_pack[-1]
    test_pack = _process_split(
        test_data,
        num_e,
        history_len,
        s_his,
        o_his,
        s_his_t,
        o_his_t,
        s_his_cache,
        o_his_cache,
        s_his_cache_t,
        o_his_cache_t,
        latest_t,
    )

    outputs = {
        "train_graphs": output_root / "train_graphs.txt",
        "train_history_sub": output_root / "train_history_sub.txt",
        "train_history_ob": output_root / "train_history_ob.txt",
        "dev_history_sub": output_root / "dev_history_sub.txt",
        "dev_history_ob": output_root / "dev_history_ob.txt",
        "test_history_sub": output_root / "test_history_sub.txt",
        "test_history_ob": output_root / "test_history_ob.txt",
    }

    with open(outputs["train_graphs"], "wb") as fp:
        pickle.dump(graph_dict_train, fp)
    with open(outputs["train_history_sub"], "wb") as fp:
        pickle.dump([train_pack[0], train_pack[2]], fp)
    with open(outputs["train_history_ob"], "wb") as fp:
        pickle.dump([train_pack[1], train_pack[3]], fp)
    with open(outputs["dev_history_sub"], "wb") as fp:
        pickle.dump([dev_pack[0], dev_pack[2]], fp)
    with open(outputs["dev_history_ob"], "wb") as fp:
        pickle.dump([dev_pack[1], dev_pack[3]], fp)
    with open(outputs["test_history_sub"], "wb") as fp:
        pickle.dump([test_pack[0], test_pack[2]], fp)
    with open(outputs["test_history_ob"], "wb") as fp:
        pickle.dump([test_pack[1], test_pack[3]], fp)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build history graphs from raw TKG quadruples")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--history-len", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    outputs = build_history_graphs(
        dataset_dir=args.dataset_dir,
        history_len=args.history_len,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
