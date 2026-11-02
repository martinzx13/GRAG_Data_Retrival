"""ExplaGraphs preprocessing: textualize the `graph` column of train_dev.tsv.

Replicates `GRAG/src/dataset/preprocess/expla_graphs.py:18-44` (step_one).
"""

import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

TRIPLET_RE = re.compile(r"\(([^()]*)\)")
NODE_COLUMNS = ["node_id", "node_attr"]
EDGE_COLUMNS = ["src", "edge_attr", "dst"]


def parse_triplets(graph_str: str) -> list[tuple[str, str, str]]:
    """Parse and validate `(src; relation; dst)` blocks.

    Raises ValueError on malformed blocks instead of silently dropping them.
    An empty or whitespace-only graph is a valid no-op and returns [].
    """
    triplets: list[tuple[str, str, str]] = []
    cursor = 0

    for match in TRIPLET_RE.finditer(graph_str):
        if graph_str[cursor:match.start()].strip():
            raise ValueError(
                f"unparsed text before triplet: {graph_str[cursor:match.start()]!r}"
            )

        parts = [part.strip().lower() for part in match.group(1).split(";")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"malformed triplet: {match.group(0)!r}")

        triplets.append((parts[0], parts[1], parts[2]))
        cursor = match.end()

    if graph_str[cursor:].strip():
        raise ValueError(f"unparsed text after last triplet: {graph_str[cursor:]!r}")

    return triplets


def textualize_graph(graph_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert one graph string into node and edge DataFrames.

    Node IDs follow first appearance (src before dst per triplet), matching
    upstream. All node and relation texts are stripped and lowercased.
    """
    node_to_id: dict[str, int] = {}
    edges_list = []

    for src_text, rel_text, dst_text in parse_triplets(graph_str):
        for node_text in (src_text, dst_text):
            if node_text not in node_to_id:
                node_to_id[node_text] = len(node_to_id)
        edges_list.append(
            {
                "src": node_to_id[src_text],
                "edge_attr": rel_text,
                "dst": node_to_id[dst_text],
            }
        )

    nodes_df = pd.DataFrame(
        [
            {"node_id": idx, "node_attr": name}
            for name, idx in sorted(node_to_id.items(), key=lambda item: item[1])
        ],
        columns=NODE_COLUMNS,
    )
    edges_df = pd.DataFrame(edges_list, columns=EDGE_COLUMNS)
    return nodes_df, edges_df


def step_one(
    tsv_path: Path, out_dir: Path, show_progress: bool = True
) -> tuple[int, int]:
    """Textualize every row into `nodes/{i}.csv` and `edges/{i}.csv`.

    Returns (number of graphs, total number of edges).
    """
    dataset = pd.read_csv(tsv_path, sep="\t")
    nodes_dir = out_dir / "nodes"
    edges_dir = out_dir / "edges"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    edges_dir.mkdir(parents=True, exist_ok=True)

    total_edges = 0
    rows = tqdm(dataset["graph"], total=len(dataset), desc="Textualizing graphs",
                disable=not show_progress)
    for i, graph_str in enumerate(rows):
        nodes_df, edges_df = textualize_graph(graph_str)
        nodes_df.to_csv(nodes_dir / f"{i}.csv", index=False, columns=NODE_COLUMNS)
        edges_df.to_csv(edges_dir / f"{i}.csv", index=False, columns=EDGE_COLUMNS)
        total_edges += len(edges_df)

    return len(dataset), total_edges


def _project_root() -> Path:
    """Locate the repo root by a dataset sentinel, independent of nesting depth."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "dataset" / "expla_graphs" / "train_dev.tsv").is_file():
            return parent
    raise FileNotFoundError(
        "could not locate dataset/expla_graphs/train_dev.tsv above this module"
    )


def main() -> None:
    root = _project_root()
    expla_dir = root / "dataset" / "expla_graphs"
    n_rows, n_edges = step_one(expla_dir / "train_dev.tsv", expla_dir)
    print(f"textualized {n_rows} graphs, {n_edges} edges -> {expla_dir}")


if __name__ == "__main__":
    main()
