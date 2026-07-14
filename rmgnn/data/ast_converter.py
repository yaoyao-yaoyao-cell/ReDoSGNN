"""Convert regex AST dumps into the motif-enhanced HRG input format.

The generated files keep the repository's TU-style layout while implementing
the semantic augmentation described in RMGNN Sec. 4.2 and Appendix C:

* fixed-width, type-specific node attributes;
* AST Child edges;
* Capture-backreference (Ref) edges;
* Lookaround-scope (LookArd) edges;
* bounded counting operators represented by attributes, not tree expansion.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


TYPE_MAP = {
    "ROOT": 0,
    "EPSILON": 1,
    "CHARCLASS": 2,
    "CONCAT": 3,
    "UNION": 4,
    "STAR": 5,
    "PLUS": 6,
    "OPT": 7,
    "REPEAT": 8,
    "CAPTURELEFT": 9,
    "CAPTURERIGHT": 10,
    "PLOOKAHEAD": 11,
    "NLOOKAHEAD": 12,
    "PLOOKBEHIND": 13,
    "NLOOKBEHIND": 14,
    "BACKREFERENCE": 15,
}

NUM_TYPES = len(TYPE_MAP)
MAX_BOUND = 1000.0
MAX_GROUP_INDEX = 100.0
MAX_CHAR_CODE = 0x10FFFF

# Semantic attribute offsets after the 16-dimensional node-type one-hot.
SEMANTIC_DIMENSIONS = (
    "bounded",
    "lower_bound",
    "upper_bound",
    "unbounded",
    "greedy",
    "capture",
    "group_index",
    "group_index_raw",
    "backref_target",
    "backref_target_raw",
    "lookahead",
    "lookbehind",
    "positive",
    "char_digit",
    "char_word",
    "char_space",
    "char_custom",
    "char_cardinality",
    "char_center",
    "char_width",
)
ATTR_DIM = NUM_TYPES + len(SEMANTIC_DIMENSIONS)

EDGE_CHILD = 1
EDGE_REF = 2
EDGE_LOOKAROUND = 3

OUTPUTAST_DATASET_MAP = {
    "Corpus": "Corpus_HRG",
    "Csharp": "Csharp_HRG",
    "Java": "Java_HRG",
    "Python": "Python_HRG",
    "Corpus_HRG": "Corpus_HRG",
    "Csharp_HRG": "Csharp_HRG",
    "Java_HRG": "Java_HRG",
    "Python_HRG": "Python_HRG",
}


@dataclass
class ASTNode:
    node_id: int
    graph_id: int
    depth: int
    kind: str
    content: str
    parent: int | None
    children: list[int] = field(default_factory=list)
    capture_index: int = -1
    backref_index: int = -1


@dataclass
class HRGDataset:
    edges: list[tuple[int, int]] = field(default_factory=list)
    edge_labels: list[int] = field(default_factory=list)
    node_labels: list[int] = field(default_factory=list)
    graph_indicator: list[int] = field(default_factory=list)
    node_attributes: list[list[float]] = field(default_factory=list)
    graph_labels: list[int] = field(default_factory=list)

    def add_edge(self, source: int, target: int, relation: int):
        self.edges.append((source, target))
        self.edge_labels.append(relation)


def _depth_and_content(line: str) -> tuple[int, str]:
    """Return indentation depth and AST payload for lines using ``|   ``."""
    match = re.match(r"^(?P<prefix>(?:\|\s*)*)", line)
    prefix = match.group("prefix") if match else ""
    depth = prefix.count("|")
    return depth, line[len(prefix):].strip()


def _node_kind(content: str) -> str:
    token = content.split(":", 1)[0].split(None, 1)[0].strip()
    if token not in TYPE_MAP:
        raise ValueError(f"Unsupported AST node type {token!r}: {content!r}")
    return token


def _normalize(value: float, scale: float) -> float:
    return min(max(float(value), 0.0), scale) / scale


def _counting_bounds(kind: str, content: str) -> tuple[int, float, float, bool]:
    if kind == "STAR":
        return 0, 0.0, MAX_BOUND, True
    if kind == "PLUS":
        return 0, 1.0, MAX_BOUND, True
    if kind == "OPT":
        return 1, 0.0, 1.0, False
    if kind != "REPEAT":
        return 0, 0.0, 0.0, False

    match = re.search(r"\{\s*(\d+)\s*,\s*(\d+|∞|inf)?\s*\}", content, re.I)
    if not match:
        exact = re.search(r"\{\s*(\d+)\s*\}", content)
        if not exact:
            return 0, 0.0, 0.0, False
        value = float(exact.group(1))
        return 1, value, value, False

    lower = float(match.group(1))
    upper_text = match.group(2)
    unbounded = upper_text is None or upper_text.lower() == "inf" or upper_text == "∞"
    upper = MAX_BOUND if unbounded else float(upper_text)
    return int(not unbounded), lower, upper, unbounded


def _is_lazy(content: str) -> bool:
    lowered = content.lower()
    return "lazy" in lowered or content.rstrip().endswith("?")


def _char_category(low: int, high: int) -> tuple[float, float, float, float]:
    digit = int(low >= ord("0") and high <= ord("9"))
    space_ranges = ((9, 13), (32, 32))
    space = int(any(low >= start and high <= end for start, end in space_ranges))
    word_ranges = (
        (ord("0"), ord("9")),
        (ord("A"), ord("Z")),
        (ord("a"), ord("z")),
        (ord("_"), ord("_")),
    )
    word = int(any(low >= start and high <= end for start, end in word_ranges))
    custom = int(not (digit or space or word))
    return float(digit), float(word), float(space), float(custom)


def node_attributes(node: ASTNode) -> list[float]:
    """Build the fixed-width, masked semantic feature vector from Appendix C."""
    values = [0.0] * ATTR_DIM
    values[TYPE_MAP[node.kind]] = 1.0
    offset = NUM_TYPES

    if node.kind in {"STAR", "PLUS", "OPT", "REPEAT"}:
        bounded, lower, upper, unbounded = _counting_bounds(node.kind, node.content)
        values[offset + 0] = float(bounded)
        values[offset + 1] = _normalize(lower, MAX_BOUND)
        values[offset + 2] = _normalize(upper, MAX_BOUND)
        values[offset + 3] = float(unbounded)
        values[offset + 4] = float(not _is_lazy(node.content))

    if node.kind in {"CAPTURELEFT", "CAPTURERIGHT"}:
        values[offset + 5] = 1.0
        values[offset + 6] = _normalize(node.capture_index, MAX_GROUP_INDEX)
        values[offset + 7] = float(node.capture_index)

    if node.kind == "BACKREFERENCE":
        values[offset + 8] = _normalize(node.backref_index, MAX_GROUP_INDEX)
        values[offset + 9] = float(node.backref_index)

    if node.kind in {"PLOOKAHEAD", "NLOOKAHEAD", "PLOOKBEHIND", "NLOOKBEHIND"}:
        values[offset + 10] = float("LOOKAHEAD" in node.kind)
        values[offset + 11] = float("LOOKBEHIND" in node.kind)
        values[offset + 12] = float(node.kind.startswith("P"))

    if node.kind == "CHARCLASS":
        match = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", node.content)
        if match:
            low, high = int(match.group(1)), int(match.group(2))
            if low > high:
                low, high = high, low
            digit, word, space, custom = _char_category(low, high)
            cardinality = max(0, high - low + 1)
            values[offset + 13:offset + 17] = [digit, word, space, custom]
            values[offset + 17] = min(cardinality, 256) / 256.0
            values[offset + 18] = ((low + high) / 2.0) / MAX_CHAR_CODE
            values[offset + 19] = min(high - low, MAX_CHAR_CODE) / MAX_CHAR_CODE

    return values


def parse_ast_file(path: Path, graph_id: int, first_node_id: int):
    """Parse one indentation AST in linear time using a parent stack."""
    nodes: list[ASTNode] = []
    stack: list[ASTNode] = []

    root = ASTNode(
        node_id=first_node_id,
        graph_id=graph_id,
        depth=-1,
        kind="ROOT",
        content="ROOT",
        parent=None,
    )
    nodes.append(root)
    stack.append(root)

    next_node_id = first_node_id + 1
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            depth, content = _depth_and_content(raw_line.rstrip("\n"))
            try:
                kind = _node_kind(content)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error

            while len(stack) > depth + 1:
                stack.pop()
            parent = stack[-1]

            capture_match = re.search(r"Index:\s*(\d+)", content)
            ref_match = re.search(r"Refers to:\s*(\d+)", content)
            node = ASTNode(
                node_id=next_node_id,
                graph_id=graph_id,
                depth=depth,
                kind=kind,
                content=content,
                parent=parent.node_id,
                capture_index=int(capture_match.group(1)) if capture_match else -1,
                backref_index=int(ref_match.group(1)) if ref_match else -1,
            )
            parent.children.append(node.node_id)
            nodes.append(node)
            stack.append(node)
            next_node_id += 1

    return nodes, next_node_id


def _descendants(node_id: int, by_id: dict[int, ASTNode]):
    stack = list(reversed(by_id[node_id].children))
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(by_id[current].children))


def append_hrg(dataset: HRGDataset, nodes: list[ASTNode], graph_label: int):
    """Append AST nodes, then add all semantic HRG relations in a second pass."""
    by_id = {node.node_id: node for node in nodes}
    captures: dict[int, int] = {}

    for node in nodes:
        dataset.node_labels.append(TYPE_MAP[node.kind])
        dataset.graph_indicator.append(node.graph_id)
        dataset.node_attributes.append(node_attributes(node))
        if node.kind == "CAPTURELEFT" and node.capture_index >= 0:
            captures[node.capture_index] = node.node_id
        if node.parent is not None:
            dataset.add_edge(node.parent, node.node_id, EDGE_CHILD)
            dataset.add_edge(node.node_id, node.parent, EDGE_CHILD)

    # Resolve references after all capture groups have been seen. This handles
    # forward references without silently dropping the semantic dependency.
    for node in nodes:
        if node.kind == "BACKREFERENCE" and node.backref_index in captures:
            capture = captures[node.backref_index]
            dataset.add_edge(capture, node.node_id, EDGE_REF)
            dataset.add_edge(node.node_id, capture, EDGE_REF)

        if node.kind in {"PLOOKAHEAD", "NLOOKAHEAD", "PLOOKBEHIND", "NLOOKBEHIND"}:
            for scoped_node in _descendants(node.node_id, by_id):
                dataset.add_edge(node.node_id, scoped_node, EDGE_LOOKAROUND)
                dataset.add_edge(scoped_node, node.node_id, EDGE_LOOKAROUND)

    dataset.graph_labels.append(graph_label)


def label_from_filename(path: Path) -> int:
    match = re.search(r"_(0|1)$", path.stem)
    if not match:
        raise ValueError(f"Cannot infer binary label from filename {path.name!r}")
    return int(match.group(1))


def _file_sort_key(path: Path):
    stem_id = path.stem.split("_", 1)[0]
    try:
        return int(stem_id), path.name
    except ValueError:
        return math.inf, path.name


def _select_files(files: list[Path], sample_size: int | None, balanced: bool, seed: int):
    if sample_size is None or sample_size <= 0 or sample_size >= len(files):
        return files

    rng = random.Random(seed)
    if not balanced:
        selected = files[:]
        rng.shuffle(selected)
        return sorted(selected[:sample_size], key=_file_sort_key)

    by_label = {0: [], 1: []}
    for path in files:
        by_label[label_from_filename(path)].append(path)

    half = sample_size // 2
    counts = {
        0: min(len(by_label[0]), half),
        1: min(len(by_label[1]), sample_size - half),
    }
    remainder = sample_size - counts[0] - counts[1]
    for label in (0, 1):
        take = min(len(by_label[label]) - counts[label], remainder)
        counts[label] += take
        remainder -= take

    selected = []
    for label, count in counts.items():
        pool = by_label[label][:]
        rng.shuffle(pool)
        selected.extend(pool[:count])
    return sorted(selected, key=_file_sort_key)


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    sample_size: int | None = None,
    balanced: bool = True,
    seed: int = 14,
    overwrite: bool = False,
):
    dataset = HRGDataset()
    next_node_id = 0
    if output_dir.exists() and not overwrite and (output_dir / "graph_labels.txt").exists():
        print(f"Skip existing HRG dataset {output_dir}; pass --overwrite to rebuild it.")
        return

    files = sorted(input_dir.glob("*.txt"), key=_file_sort_key)
    if not files:
        raise ValueError(f"No AST .txt files found directly under {input_dir}")
    files = _select_files(files, sample_size=sample_size, balanced=balanced, seed=seed)

    started = time.perf_counter()

    for graph_id, path in enumerate(files):
        nodes, next_node_id = parse_ast_file(path, graph_id, next_node_id)
        append_hrg(dataset, nodes, label_from_filename(path))
        if (graph_id + 1) % 1000 == 0 or graph_id + 1 == len(files):
            elapsed = time.perf_counter() - started
            print(
                f"Parsed {graph_id + 1}/{len(files)} ASTs "
                f"({(graph_id + 1) / max(elapsed, 1e-9):.1f} graphs/s)"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "A.txt": (f"{u},{v}\n" for u, v in dataset.edges),
        "edge_labels.txt": (f"{value}\n" for value in dataset.edge_labels),
        "graph_indicator.txt": (f"{value}\n" for value in dataset.graph_indicator),
        "node_labels.txt": (f"{value}\n" for value in dataset.node_labels),
        "node_attributes.txt": (
            ",".join(f"{value:.10g}" for value in row) + "\n"
            for row in dataset.node_attributes
        ),
        "graph_labels.txt": (f"{value}\n" for value in dataset.graph_labels),
    }
    for filename, rows in outputs.items():
        with (output_dir / filename).open("w", encoding="utf-8") as target:
            target.writelines(rows)

    relation_counts = {
        relation: dataset.edge_labels.count(relation)
        for relation in (EDGE_CHILD, EDGE_REF, EDGE_LOOKAROUND)
    }
    print(
        f"HRG conversion complete: graphs={len(files)}, nodes={next_node_id}, "
        f"edges={len(dataset.edges)}, attr_dim={ATTR_DIM}, relations={relation_counts}"
    )


def convert_output_ast_dataset(
    name: str,
    output_name: str | None = None,
    sample_size: int | None = None,
    balanced: bool = True,
    seed: int = 14,
    overwrite: bool = False,
):
    """Convert one of the four OutputAST folders to a consistently named HRG dataset."""
    source_name = name.removesuffix("_HRG")
    if source_name not in OUTPUTAST_DATASET_MAP:
        raise ValueError(f"Unknown OutputAST dataset {name!r}")
    target_name = output_name or OUTPUTAST_DATASET_MAP[source_name]
    input_dir = Path("dataset") / "OutputAST" / source_name
    output_dir = Path("dataset") / target_name
    convert_directory(
        input_dir,
        output_dir,
        sample_size=sample_size,
        balanced=balanced,
        seed=seed,
        overwrite=overwrite,
    )


def main():
    parser = argparse.ArgumentParser(description="Convert regex AST dumps to RMGNN HRGs")
    parser.add_argument("--dataset", default=None,
                        help="Corpus, Csharp, Java, Python, or all")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--balanced-sample", action="store_true", default=True)
    parser.add_argument("--no-balanced-sample", dest="balanced_sample", action="store_false")
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dataset:
        names = OUTPUTAST_DATASET_MAP.keys() if args.dataset.lower() == "all" else [args.dataset]
        for name in names:
            if name.endswith("_HRG"):
                continue
            convert_output_ast_dataset(
                name,
                sample_size=args.sample_size,
                balanced=args.balanced_sample,
                seed=args.seed,
                overwrite=args.overwrite,
            )
        return

    input_dir = Path(args.input_dir or "dataset/OutputAST/Java")
    output_dir = Path(args.output_dir or "dataset/Java_HRG")
    convert_directory(
        input_dir,
        output_dir,
        sample_size=args.sample_size,
        balanced=args.balanced_sample,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
