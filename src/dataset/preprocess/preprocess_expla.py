"""ExplaGraphs preprocessing: textualize the `graph` column of train_dev.tsv.

Replicates `GRAG/src/dataset/preprocess/expla_graphs.py:18-44` (step_one).
"""
import os 
import torch 
from torch_geometric.data import Data 
from src.utils.lm_modeling import load_sbert, encode_texts

import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

TRIPLET_RE = re.compile(r"\(([^()]*)\)")
NODE_COLUMNS = ["node_id", "node_attr"]
EDGE_COLUMNS = ["src", "edge_attr", "dst"]


def build_questions(dataset:pd.DataFrame, mode: str="corrected") -> list[str]:
    """
    Builds the natural language question strings for all rows in the dataset.
    mode='corrected': uses row-specific arg1 and arg2 (Paper intent / bugfix).
    mode='literal': uses the whole Series stringification (reproduces upstream bug A10).
    """
    prompt = "Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of 'support' or 'counter'.\n\nAnswer:"
    questions = []

    if mode == "corrected":
        for _, row in dataset.iterrows():
            q = f"Argument 1: {row['arg1']}\nArgument 2: {row['arg2']}\n{prompt}"
            questions.append(q)
    elif mode == "literal":
        # upstream Bug : text.arg1 is the entire Series
        for _ in range(len(dataset)):
            q = f"Argument 1 : {dataset.arg1}\nArgument 2: {dataset.arg2}\n{prompt}"
            questions.append(q)
    else:
        raise ValueError(f"Unknown model : {mode}")

    return (questions)

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

def step_two(
        tsv_path: Path,
        out_dir: Path,
        mode: str = "corrected",
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        show_progress: bool = True,
        ) -> None:
        """Encodes questions and graphs into .pt tensors using SBERT.
        Generates:
          - out_dir / f"q_embs_{mode}.pt" (or q_embs.pt)
          - out_dir / "graphs/{i}.pt"
        """
        dataset = pd.read_csv(tsv_path, sep="\t")
        graphs_dir = out_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading SBERT model on {device}...")
        model, tokenizer = load_sbert(device=device)

        # 1. Encode all questions
        print(f"Encoding {len(dataset)} questions (mode='{mode}')...")
        questions = build_questions(dataset, mode=mode)
        q_embs = encode_texts(model, tokenizer, questions, batch_size=batch_size, device=device)

        q_embs_path = out_dir / f"q_embs_{mode}.pt"
        torch.save(q_embs, q_embs_path)
        # Also save as default q_embs.pt for seamless downstream compatibility
        torch.save(q_embs, out_dir / "q_embs.pt")
        print(f"Saved question embeddings -> {q_embs_path} ({q_embs.shape})")

        # 2. Encode all graphs (nodes + edges)
        nodes_dir = out_dir / "nodes"
        edges_dir = out_dir / "edges"

        rows = range(len(dataset))
        if show_progress:
            rows = tqdm(rows, desc="Encoding graph tensors", total=len(dataset))

        for i in rows:
            nodes_df = pd.read_csv(nodes_dir / f"{i}.csv")
            edges_df = pd.read_csv(edges_dir / f"{i}.csv")

            # Encode node attributes [num_nodes, 1024]
            x = encode_texts(model, tokenizer, nodes_df["node_attr"].tolist(), batch_size=batch_size, device=device)

            # Encode edge attributes [num_edges, 1024]
            edge_attr = encode_texts(model, tokenizer, edges_df["edge_attr"].tolist(), batch_size=batch_size, device=device)

            # Construct edge_index [2, num_edges]
            edge_index = torch.tensor(
                [edges_df["src"].tolist(), edges_df["dst"].tolist()],
                dtype=torch.long,
            )

            # Repeat question embeddings for node and edge alignment
            q_node = q_embs[i].unsqueeze(0).repeat(x.size(0), 1)
            q_edge = q_embs[i].unsqueeze(0).repeat(edge_attr.size(0), 1)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                question_node=q_node,
                question_edge=q_edge,
                num_nodes=len(nodes_df),
            )

            torch.save(data, graphs_dir / f"{i}.pt")
        print(f"Successfully generated {len(dataset)} graph tensors -> {graphs_dir}")

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
    tsv_path = expla_dir / "train_dev.tsv"
    step_two(tsv_path, expla_dir, mode="corrected", batch_size=64)


if __name__ == "__main__":
    main()
