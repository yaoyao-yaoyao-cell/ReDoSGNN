"""Preprocess OutputAST regex dumps into RMGNN-ready meta graphs.

Pipeline:
1. Optionally convert raw ``dataset/OutputAST/<language>/*.txt`` dumps into
   TU-style HRG folders using :mod:`rmgnn.data.ast_converter`.
2. Build the RMGNN motif meta graph from the HRG graphs.
3. Attach graph-level structural, semantic, and motif-prior features.
4. Save a DGL graph pickle consumed by ``main.py``.
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import math
import multiprocessing as mp
import os
import pickle
import random
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

os.environ.setdefault("DGL_DISABLE_GRAPHBOLT", "1")

import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import EdgeWeightNorm
from tqdm import tqdm

from rmgnn.data.ast_converter import OUTPUTAST_DATASET_MAP, convert_output_ast_dataset

OUTPUTAST_HRG_DATASETS = ("Corpus_HRG", "Csharp_HRG", "Java_HRG", "Python_HRG")

CNT_OPS = {5, 6, 8}
UNI_OPS = {2, 4}
EXT_OPS = {11, 12, 13, 14, 15}
NUM_NODE_TYPES = 16
MOTIF_VOCAB_SIZE = 1000


@dataclass(frozen=True)
class MotifInstance:
    center: int
    token: int
    nodes: tuple[int, ...]
    u: int
    mtype: str


@dataclass
class RegexDataset:
    g_list: list[nx.Graph]
    node_labels: list[int]
    graph_labels: list[int]


class MotifMiner:
    """Implements the three RMGNN regex motif families.

    M1 links counting operators to nested/sibling counting operators, M2 links
    counting operators to child union/character-class nodes, and M3 links
    counting operators to extended regex constructs such as lookaround and
    backreferences.
    """

    def __init__(self, vocab_size: int = MOTIF_VOCAB_SIZE, max_depth: int = 6):
        self.vocab_size = int(vocab_size)
        self.max_depth = int(max_depth)

    def _decode_node_type(self, node_data: dict) -> int:
        feat = node_data.get("attr", [])
        if isinstance(feat, (list, tuple, np.ndarray)) and len(feat) >= NUM_NODE_TYPES:
            prefix = np.asarray(feat[:NUM_NODE_TYPES], dtype=np.float32)
            if float(prefix.sum()) > 0:
                return int(prefix.argmax())
        return int(node_data.get("node_type", 0))

    @staticmethod
    def _edge_type(g: nx.Graph, u: int, v: int) -> int:
        return int(g.edges[u, v].get("type", 1))

    def _hash(self, text: str) -> int:
        digest = hashlib.md5(text.encode("utf-8")).digest()
        return int.from_bytes(digest, byteorder="big") % self.vocab_size

    def _tree_children(self, g: nx.Graph):
        children = {n: [] for n in g.nodes()}
        parent_by_child = {}
        for u, v, data in g.edges(data=True):
            relations = set(data.get("relations", ()))
            if int(data.get("type", 1)) != 1 and "child" not in relations:
                continue
            parent, child = (u, v) if int(u) < int(v) else (v, u)
            children[parent].append(child)
            parent_by_child[child] = parent
        return children, parent_by_child

    def _descendants(self, children: dict, center: int):
        visited = {center}
        queue = collections.deque([(center, 0)])
        out = []
        while queue:
            node, depth = queue.popleft()
            if depth >= self.max_depth:
                continue
            for child in children.get(node, ()):
                if child in visited:
                    continue
                visited.add(child)
                out.append(child)
                queue.append((child, depth + 1))
        return out

    @staticmethod
    def _siblings(children: dict, parent_by_child: dict, center: int):
        parent = parent_by_child.get(center)
        if parent is None:
            return []
        return sorted(node for node in children[parent] if node != center)

    @staticmethod
    def _path_nodes(g: nx.Graph, center: int, target: int):
        if center == target:
            return [center]
        try:
            return list(dict.fromkeys(nx.shortest_path(g, source=center, target=target)))
        except nx.NetworkXNoPath:
            return [center, target]

    def canon(self, subg: nx.Graph, center: int, node_types: dict[int, int]) -> str:
        if center not in subg:
            center = min(subg.nodes())
        visited = {center}
        queue = collections.deque([center])
        seq = [f"C{node_types.get(center, 0)}"]
        while queue:
            node = queue.popleft()
            seq.append(f"V{node_types.get(node, 0)}")
            neighbors = [
                (self._edge_type(subg, node, nbr), node_types.get(nbr, 0), nbr)
                for nbr in subg.neighbors(node)
            ]
            neighbors.sort(key=lambda item: (item[0], item[1], item[2]))
            for edge_type, nbr_type, nbr in neighbors:
                seq.append(f"E{edge_type}T{nbr_type}")
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        return "|".join(seq)

    def extract_with_instances(self, g: nx.Graph, max_instances: int | None = None):
        node_types = {n: self._decode_node_type(data) for n, data in g.nodes(data=True)}
        counting_nodes = [node for node, kind in node_types.items() if kind in CNT_OPS]
        counting_set = set(counting_nodes)
        children, parent_by_child = self._tree_children(g)
        instances = []

        for center in counting_nodes:
            if max_instances is not None and len(instances) >= max_instances:
                break
            descendants = self._descendants(children, center)
            descendant_set = set(descendants)
            sibling_set = set(self._siblings(children, parent_by_child, center))

            m1 = [u for u in counting_set if u != center and (u in descendant_set or u in sibling_set)]
            m2 = [u for u in children.get(center, ()) if node_types.get(u, 0) in UNI_OPS]
            m3 = [u for u in descendants if node_types.get(u, 0) in EXT_OPS]

            for target in m1 + m2 + m3:
                if max_instances is not None and len(instances) >= max_instances:
                    break
                nodes = tuple(sorted(set(self._path_nodes(g, center, target))))
                subg = g.subgraph(nodes)
                token = self._hash(self.canon(subg, center, node_types))
                mtype = "M1" if target in m1 else ("M2" if target in m2 else "M3")
                instances.append(MotifInstance(center, token, nodes, target, mtype))
        return instances


def _extract_motif_instances(
    graph: nx.Graph,
    max_instances: int | None,
) -> list[MotifInstance]:
    """Process-pool entry point for deterministic motif extraction."""
    return MotifMiner().extract_with_instances(graph, max_instances)


class MetaGraphBuilder:
    def __init__(
        self,
        graphs: list[nx.Graph],
        max_instances_per_graph: int | None,
        num_workers: int = 0,
        worker_chunksize: int = 64,
    ):
        self.graphs = graphs
        self.max_instances_per_graph = max_instances_per_graph
        self.num_workers = max(0, int(num_workers))
        self.worker_chunksize = max(1, int(worker_chunksize))
        self.vocab = {}
        self.node_count = collections.Counter()
        self.edge_count = collections.Counter()
        self.whole_node_count = collections.Counter()
        self.instance_stats = []

    @staticmethod
    def _shift_right(value):
        if isinstance(value, int):
            return value
        value = list(value)
        return tuple([value[-1]] + value[:-1])

    def add_to_vocab(self, key):
        token_tuple, weight = key
        for _ in range(len(token_tuple)):
            candidate = (token_tuple, weight)
            if candidate in self.vocab:
                return self.vocab[candidate]
            token_tuple = self._shift_right(token_tuple)
            weight = self._shift_right(weight)
        self.vocab[(token_tuple, weight)] = len(self.vocab)
        return self.vocab[(token_tuple, weight)]

    def _instance_iterator(self, extraction_limit: int | None):
        if self.num_workers <= 1:
            miner = MotifMiner()
            return (
                miner.extract_with_instances(graph, extraction_limit)
                for graph in self.graphs
            )

        # ``imap`` preserves input order and bounds queued work, avoiding a
        # second in-memory copy of large regex datasets.
        context_name = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        context = mp.get_context(context_name)
        pool = context.Pool(processes=self.num_workers)
        extract = partial(_extract_motif_instances, max_instances=extraction_limit)
        return pool.imap(extract, self.graphs, chunksize=self.worker_chunksize), pool

    def build(
        self,
        teacher_tau: float | None = None,
        teacher_wl_h: int = 2,
        max_teacher_instances: int | None = None,
    ):
        meta = nx.Graph()
        meta.add_nodes_from(range(len(self.graphs)))
        teacher_dists = {}
        needs_teacher = teacher_tau is not None
        limits = [self.max_instances_per_graph]
        if needs_teacher:
            limits.append(max_teacher_instances)
        extraction_limit = None if any(limit is None for limit in limits) else max(limits)

        iterator_result = self._instance_iterator(extraction_limit)
        if isinstance(iterator_result, tuple):
            instance_iterator, pool = iterator_result
        else:
            instance_iterator, pool = iterator_result, None

        progress = tqdm(
            zip(self.graphs, instance_iterator),
            total=len(self.graphs),
            desc="Build motif meta graph",
            unit="graph",
        )
        try:
            for graph_id, (nx_g, extracted) in enumerate(progress):
                instances = extracted[: self.max_instances_per_graph]
                if needs_teacher:
                    teacher_instances = extracted[:max_teacher_instances]
                    subgraphs = [
                        nx_g.subgraph(inst.nodes)
                        for inst in teacher_instances
                        if len(inst.nodes) > 1
                    ]
                    teacher_dists[graph_id] = teacher_distribution_from_motifs(
                        subgraphs, teacher_tau, teacher_wl_h
                    )

                type_counts = collections.Counter(inst.mtype for inst in instances)
                self.instance_stats.append({
                    "graph_id": graph_id,
                    "num_instances": len(instances),
                    "num_motif_1": type_counts.get("M1", 0),
                    "num_motif_2": type_counts.get("M2", 0),
                    "num_motif_3": type_counts.get("M3", 0),
                })

                clique_ids = []
                for inst in instances:
                    clique_id = self.add_to_vocab(((inst.token,), inst.token))
                    clique_ids.append(clique_id)
                    self.whole_node_count[clique_id] += 1

                freqs = collections.Counter(clique_ids)
                for clique_id, freq in freqs.items():
                    self.node_count[clique_id] += 1
                    meta.add_edge(graph_id, clique_id + len(self.graphs),
                                  weight=float(freq) / max(1, len(instances)))

                node_to_instances = collections.defaultdict(list)
                for idx, inst in enumerate(instances):
                    for node in inst.nodes:
                        node_to_instances[node].append(idx)
                seen_pairs = set()
                for idxs in node_to_instances.values():
                    for i in range(len(idxs)):
                        for j in range(i + 1, len(idxs)):
                            pair = (idxs[i], idxs[j])
                            if pair in seen_pairs:
                                continue
                            seen_pairs.add(pair)
                            edge_key = tuple(sorted((clique_ids[pair[0]], clique_ids[pair[1]])))
                            self.edge_count[edge_key] += 1
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        offset = len(self.graphs)
        for (src, dst), count in self.edge_count.items():
            denom = math.sqrt(max(1, self.whole_node_count[src]) * max(1, self.whole_node_count[dst]))
            meta.add_edge(src + offset, dst + offset, weight=float(count) / denom)
        return meta, len(self.vocab), self.instance_stats, teacher_dists


def read_int_lines(path: Path):
    with path.open(encoding="utf-8") as source:
        return [int(line.strip()) for line in source if line.strip()]


def load_hrg_dataset(dataset: str, perturb: str | None = None, perturb_level: float = 0.1, seed: int = 14):
    base = Path("dataset") / dataset
    if not base.exists():
        raise FileNotFoundError(f"Dataset folder not found: {base}")

    indicators_raw = read_int_lines(base / "graph_indicator.txt")
    graph_base = min(indicators_raw)
    indicators = [value - graph_base for value in indicators_raw]

    labels_raw = read_int_lines(base / "graph_labels.txt")
    label_map = {label: idx for idx, label in enumerate(sorted(set(labels_raw)))}
    graph_labels = [label_map[label] for label in labels_raw]

    node_labels_path = base / "node_labels.txt"
    node_labels = read_int_lines(node_labels_path) if node_labels_path.exists() else [0] * len(indicators)

    attr_path = base / "node_attributes.txt"
    if attr_path.exists():
        with attr_path.open(encoding="utf-8") as source:
            node_attrs = [
                [float(value) for value in line.strip().split(",")]
                for line in source if line.strip()
            ]
    else:
        node_attrs = [[] for _ in indicators]
    if len(node_attrs) != len(indicators):
        raise ValueError(f"{attr_path} row count does not match graph_indicator.txt")

    if perturb and node_attrs:
        rng = np.random.default_rng(seed)
        attrs = np.asarray(node_attrs, dtype=np.float32)
        if perturb == "gaussian":
            attrs = attrs + rng.normal(0.0, attrs.std() * perturb_level, attrs.shape)
        elif perturb == "mask":
            mask = rng.random(attrs.shape) < perturb_level
            attrs[mask] = 0.0
        node_attrs = attrs.tolist()

    with (base / "A.txt").open(encoding="utf-8") as source:
        raw_edges = [
            tuple(map(int, line.replace(" ", "").strip().split(",")))
            for line in source if line.strip()
        ]
    node_base = min((min(edge) for edge in raw_edges), default=0)
    edges = [(u - node_base, v - node_base) for u, v in raw_edges]

    edge_labels_path = base / "edge_labels.txt"
    if edge_labels_path.exists():
        edge_types = read_int_lines(edge_labels_path)
        if len(edge_types) != len(edges):
            raise ValueError(f"{edge_labels_path} row count does not match A.txt")
    else:
        edge_types = [1] * len(edges)

    graph_nodes = collections.defaultdict(list)
    for global_id, graph_id in enumerate(indicators):
        graph_nodes[graph_id].append(global_id)

    graphs = []
    global_to_local = {}
    for graph_id in range(len(graph_labels)):
        graph = nx.Graph()
        for local_id, global_id in enumerate(graph_nodes.get(graph_id, ())):
            global_to_local[global_id] = (graph_id, local_id)
            attr = node_attrs[global_id]
            node_type = int(np.argmax(attr[:NUM_NODE_TYPES])) if attr[:NUM_NODE_TYPES] else node_labels[global_id]
            graph.add_node(local_id, attr=attr, node_type=node_type, global_id=global_id)
        graphs.append(graph)

    for (global_u, global_v), edge_type in zip(edges, edge_types):
        graph_u, local_u = global_to_local[global_u]
        graph_v, local_v = global_to_local[global_v]
        if graph_u != graph_v:
            raise ValueError(f"Cross-graph edge ({global_u}, {global_v}) in {dataset}")
        relation = "child" if edge_type == 1 else ("ref" if edge_type == 2 else f"rel_{edge_type}")
        graph = graphs[graph_u]
        if graph.has_edge(local_u, local_v):
            graph.edges[local_u, local_v].setdefault("relations", set()).add(relation)
        else:
            graph.add_edge(local_u, local_v, type=int(edge_type), weight=float(edge_type), relations={relation})

    return RegexDataset(graphs, node_labels, graph_labels)


def wl_feature_map(g: nx.Graph, h: int = 2):
    cur = {}
    for node, data in g.nodes(data=True):
        attr = data.get("attr", [])
        node_type = int(np.argmax(attr[:NUM_NODE_TYPES])) if len(attr) >= NUM_NODE_TYPES and sum(attr[:NUM_NODE_TYPES]) > 0 else 0
        cur[node] = f"T{node_type}"
    phi = collections.Counter(cur.values())
    for depth in range(1, h + 1):
        nxt = {}
        for node in g.nodes():
            neigh_sig = sorted(f"{g.edges[node, nbr].get('type', 1)}:{cur[nbr]}" for nbr in g.neighbors(node))
            label = "WL{}_{}".format(depth, hashlib.md5(("|".join([cur[node], *neigh_sig])).encode()).hexdigest())
            nxt[node] = label
            phi[label] += 1
        cur = nxt
    return phi


def teacher_distribution_from_motifs(motif_subgraphs: list[nx.Graph], tau: float, wl_h: int):
    if len(motif_subgraphs) <= 1:
        return None
    features = [wl_feature_map(g, h=wl_h) for g in motif_subgraphs]
    vocab = {key: idx for idx, key in enumerate(sorted({k for fm in features for k in fm.keys()}))}
    matrix = np.zeros((len(features), len(vocab)), dtype=np.float32)
    for row, fm in enumerate(features):
        for key, value in fm.items():
            matrix[row, vocab[key]] = float(value)
    kernel = torch.as_tensor(matrix @ matrix.T, dtype=torch.float32)
    return torch.softmax(kernel / float(tau), dim=1)


def graph_attribute_pool(graphs: list[nx.Graph], total_nodes: int, num_cliques: int):
    max_dim = max(
        (len(data.get("attr", [])) for graph in graphs for _, data in graph.nodes(data=True)),
        default=0,
    )
    if max_dim == 0:
        return None
    attrs = torch.zeros((total_nodes, 2 * max_dim), dtype=torch.float32)
    for graph_idx, graph in enumerate(graphs):
        node_count = graph.number_of_nodes()
        if node_count == 0:
            continue
        values = np.zeros((node_count, max_dim), dtype=np.float32)
        for row, (_, data) in enumerate(graph.nodes(data=True)):
            node_attr = data.get("attr", [])[:max_dim]
            values[row, :len(node_attr)] = node_attr
        pooled = np.concatenate((values.mean(axis=0), values.max(axis=0)))
        attrs[graph_idx] = torch.from_numpy(pooled)
    return attrs


def gen_features_labels(graph_labels: torch.Tensor, dgl_graph, num_cliques: int):
    labels = torch.cat((graph_labels, torch.zeros(num_cliques, dtype=torch.long)), dim=0)
    graph_nodes = dgl_graph.num_nodes() - num_cliques
    clique_features = torch.eye(num_cliques)
    graph_features = torch.zeros((graph_nodes, num_cliques), dtype=torch.float32)
    src, dst = dgl_graph.edges()
    mask = (src < graph_nodes) & (dst >= graph_nodes)
    if mask.any():
        graph_features[src[mask], dst[mask] - graph_nodes] = 1.0
    mask = (dst < graph_nodes) & (src >= graph_nodes)
    if mask.any():
        graph_features[dst[mask], src[mask] - graph_nodes] = 1.0
    return torch.cat((graph_features, clique_features), dim=0), labels


def motif_prior_tensor(stats, total_nodes: int):
    prior = torch.zeros((total_nodes, 4), dtype=torch.float32)
    for row in stats:
        graph_id = row["graph_id"]
        total = max(1, row["num_instances"])
        prior[graph_id] = torch.tensor([
            math.log1p(row["num_instances"]),
            row["num_motif_1"] / total,
            row["num_motif_2"] / total,
            row["num_motif_3"] / total,
        ])
    prior[:, 0] = prior[:, 0] / prior[:, 0].max().clamp_min(1.0)
    return prior


def preprocess_dataset(args, dataset: str):
    start_wall = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.convert:
        source_name = dataset.removesuffix("_HRG")
        convert_output_ast_dataset(
            source_name,
            output_name=dataset,
            sample_size=args.sample_size,
            balanced=args.balanced_sample,
            seed=args.seed,
            overwrite=args.overwrite,
        )

    t0 = time.perf_counter()
    data = load_hrg_dataset(
        dataset,
        perturb=args.perturb_type if args.perturb else None,
        perturb_level=args.perturb_level,
        seed=args.seed,
    )
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    builder = MetaGraphBuilder(
        data.g_list,
        args.max_instances_per_graph,
        num_workers=args.num_workers,
        worker_chunksize=args.worker_chunksize,
    )
    meta_graph, num_cliques, motif_stats, teacher_dists = builder.build(
        teacher_tau=None if args.skip_teacher else args.teacher_tau,
        teacher_wl_h=args.teacher_wl_h,
        max_teacher_instances=args.max_teacher_instances,
    )
    build_time = time.perf_counter() - t0
    print(f"{dataset}: motif nodes={num_cliques}, meta edges={meta_graph.number_of_edges()}")

    src, dst, weights = [], [], []
    for u, v, edge_data in meta_graph.edges(data=True):
        weight = float(edge_data.get("weight", 1.0))
        src.extend((u, v))
        dst.extend((v, u))
        weights.extend((weight, weight))

    dgl_graph = dgl.graph(
        (torch.as_tensor(src, dtype=torch.long), torch.as_tensor(dst, dtype=torch.long)),
        num_nodes=meta_graph.number_of_nodes(),
    )
    dgl_graph.edata["weight"] = torch.as_tensor(weights, dtype=torch.float32)

    graph_labels = torch.as_tensor(data.graph_labels, dtype=torch.long)
    features, labels = gen_features_labels(graph_labels, dgl_graph, num_cliques)
    attr_features = graph_attribute_pool(data.g_list, dgl_graph.num_nodes(), num_cliques)
    if attr_features is not None:
        features = torch.cat((features, attr_features), dim=1)

    prior = motif_prior_tensor(motif_stats, dgl_graph.num_nodes())
    dgl_graph.ndata["motif_prior"] = prior
    dgl_graph.ndata["feat"] = F.normalize(torch.cat((features, prior), dim=1), p=2, dim=1)
    dgl_graph.ndata["labels"] = labels

    norm = EdgeWeightNorm(norm="both")
    dgl_graph.edata["edge_weight"] = norm(dgl_graph, dgl_graph.edata["weight"])
    degs = dgl_graph.in_degrees().float()
    deg_norm = torch.pow(degs, -0.5)
    deg_norm[torch.isinf(deg_norm)] = 0
    dgl_graph.ndata["norm"] = deg_norm.unsqueeze(1)

    metrics = {
        "dataset": dataset,
        "seed": args.seed,
        "num_input_graphs": len(data.g_list),
        "num_meta_graph_nodes": int(dgl_graph.num_nodes()),
        "num_meta_graph_edges": int(dgl_graph.num_edges()),
        "num_cliques": int(num_cliques),
        "num_workers": args.num_workers,
        "load_time": load_time,
        "meta_graph_build_time": build_time,
        "wall_time": time.perf_counter() - start_wall,
    }

    Path("preprocessed_datasets").mkdir(exist_ok=True)
    with (Path("preprocessed_datasets") / dataset).open("wb") as target:
        pickle.dump({
            "g_dgl": dgl_graph,
            "teacher_dists": teacher_dists,
            "teacher_stats_per_graph": motif_stats,
            "preprocess_metrics": metrics,
        }, target)

    print(f"Saved preprocessed dataset to preprocessed_datasets/{dataset}")

    # Release large intermediate NetworkX objects between datasets.
    del data, meta_graph, dgl_graph
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess regex HRG datasets")
    parser.add_argument("-data", "--data", default="Corpus_HRG",
                        help="Dataset name, comma-separated names, or 'all'")
    parser.add_argument("--convert", action="store_true",
                        help="Convert dataset/OutputAST raw dumps before preprocessing")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Optional number of raw AST files to convert per dataset")
    parser.add_argument("--balanced-sample", action="store_true", default=True)
    parser.add_argument("--no-balanced-sample", dest="balanced_sample", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-instances-per-graph", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Parallel motif extraction workers; 0 runs sequentially")
    parser.add_argument("--worker-chunksize", type=int, default=64,
                        help="Graphs assigned to each multiprocessing work chunk")
    parser.add_argument("--skip-teacher", action="store_true",
                        help="Skip WL teacher distributions for faster preprocessing")
    parser.add_argument("--max-teacher-instances", type=int, default=50)
    parser.add_argument("--teacher-tau", type=float, default=0.5)
    parser.add_argument("--teacher-wl-h", type=int, default=2)
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--perturb-type", choices=("gaussian", "mask"), default="gaussian")
    parser.add_argument("--perturb-level", type=float, default=0.1)
    parser.add_argument("-seed", type=int, default=14)
    return parser.parse_args()


def expand_datasets(data_arg: str):
    if data_arg.lower() == "all":
        return list(OUTPUTAST_HRG_DATASETS)
    names = []
    for raw in data_arg.split(","):
        name = raw.strip()
        if not name:
            continue
        names.append(OUTPUTAST_DATASET_MAP.get(name, name))
    return names


def main():
    args = parse_args()
    for dataset in expand_datasets(args.data):
        preprocess_dataset(args, dataset)


if __name__ == "__main__":
    main()
