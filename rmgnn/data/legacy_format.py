"""Convert TU-style HRG files to the legacy graph-text representation.

New training code reads the TU files directly because that format preserves
edge relations. This utility remains for compatibility with older loaders.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import networkx as nx


def read_int_lines(path: Path):
    with path.open(encoding="utf-8") as source:
        return [int(line.strip()) for line in source if line.strip()]


def get_node_attrs(path: Path):
    with path.open(encoding="utf-8") as source:
        return [
            [float(value) for value in line.strip().split(",")]
            for line in source if line.strip()
        ]


def read_dataset(dataset_dir: Path):
    indicators_raw = read_int_lines(dataset_dir / "graph_indicator.txt")
    graph_base = min(indicators_raw)
    indicators = [value - graph_base for value in indicators_raw]
    labels = read_int_lines(dataset_dir / "graph_labels.txt")

    attrs_path = dataset_dir / "node_attributes.txt"
    attrs = get_node_attrs(attrs_path) if attrs_path.exists() else [[] for _ in indicators]
    if len(attrs) != len(indicators):
        raise ValueError("node_attributes.txt and graph_indicator.txt have different row counts")

    node_labels_path = dataset_dir / "node_labels.txt"
    node_labels = (
        read_int_lines(node_labels_path)
        if node_labels_path.exists()
        else [0] * len(indicators)
    )

    with (dataset_dir / "A.txt").open(encoding="utf-8") as source:
        raw_edges = [
            tuple(map(int, line.replace(" ", "").strip().split(",")))
            for line in source if line.strip()
        ]
    node_base = min((min(edge) for edge in raw_edges), default=0)
    edges = [(u - node_base, v - node_base) for u, v in raw_edges]
    return indicators, labels, node_labels, attrs, edges


def transform(dataset_dir: Path, output_file: Path):
    indicators, labels, node_labels, attrs, edges = read_dataset(dataset_dir)

    graph_nodes = defaultdict(list)
    for node_id, graph_id in enumerate(indicators):
        graph_nodes[graph_id].append(node_id)

    adjacency = [defaultdict(set) for _ in labels]
    for source, target in edges:
        source_graph = indicators[source]
        target_graph = indicators[target]
        if source_graph != target_graph:
            raise ValueError(f"Cross-graph edge ({source}, {target})")
        adjacency[source_graph][source].add(target)
        adjacency[source_graph][target].add(source)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as target:
        target.write(f"{len(labels)}\n")
        for graph_id, label in enumerate(labels):
            nodes = graph_nodes[graph_id]
            local = {node_id: index for index, node_id in enumerate(nodes)}
            target.write(f"{len(nodes)} {label}\n")

            # Iterate graph_indicator nodes, not adjacency keys, so isolated
            # AST/HRG nodes are never dropped.
            for node_id in nodes:
                neighbors = sorted(
                    local[neighbor]
                    for neighbor in adjacency[graph_id].get(node_id, ())
                    if neighbor in local
                )
                tag = node_labels[node_id] if node_id < len(node_labels) else 0
                attr_text = " ".join(map(str, attrs[node_id]))
                row = f"{tag} {len(neighbors)}"
                if neighbors:
                    row += " " + " ".join(map(str, neighbors))
                if attr_text:
                    row += " " + attr_text
                target.write(row + "\n")

    print(f"Wrote {len(labels)} graphs to {output_file}")


def load_data(graph_file: Path):
    """Small validator/generator for the legacy graph-text file."""
    with graph_file.open(encoding="utf-8") as source:
        graph_count = int(source.readline())
        for _ in range(graph_count):
            node_count, label = map(int, source.readline().split())
            graph = nx.Graph(label=label)
            for node_id in range(node_count):
                row = source.readline().split()
                tag, degree = int(row[0]), int(row[1])
                neighbors = [int(value) for value in row[2:2 + degree]]
                attributes = [float(value) for value in row[2 + degree:]]
                graph.add_node(node_id, tag=tag, attr=attributes)
                graph.add_edges_from((node_id, neighbor) for neighbor in neighbors)
            if graph.number_of_nodes() != node_count:
                raise ValueError("Malformed graph-text file")
            yield graph


def main():
    parser = argparse.ArgumentParser(description="Convert TU HRGs to legacy graph text")
    parser.add_argument("--dataset", default="Java_HRG")
    args = parser.parse_args()
    dataset_dir = Path("dataset") / args.dataset
    output_file = dataset_dir / f"{args.dataset}.txt"
    transform(dataset_dir, output_file)
    list(load_data(output_file))


if __name__ == "__main__":
    main()
