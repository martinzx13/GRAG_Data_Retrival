"""Optional visualization for parsed ExplaGraphs graphs.

This module is outside the preprocessing pipeline; it is the only place that
depends on matplotlib. Requires `networkx` and `matplotlib`.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def visualize_parsed_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    output_path: str | Path = "figures/explagraph_visual.png",
) -> Path:
    """Render node/edge DataFrames as a directed graph and save a PNG."""
    graph = nx.DiGraph()

    for _, row in nodes_df.iterrows():
        graph.add_node(row["node_id"], label=row["node_attr"])

    for _, row in edges_df.iterrows():
        graph.add_edge(row["src"], row["dst"], label=row["edge_attr"])

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(graph, seed=42)

    node_labels = nx.get_node_attributes(graph, "label")
    edge_labels = nx.get_edge_attributes(graph, "label")

    nx.draw_networkx_nodes(graph, pos, node_color="lightblue", node_size=3000, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=20, edge_color="gray", width=2)
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=10, font_weight="bold")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=9, font_color="darkred")

    plt.title("ExplaGraphs Spatial Visualization", fontsize=14, pad=15)
    plt.axis("off")
    plt.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    return output


if __name__ == "__main__":
    from src.dataset.preprocess.preprocess_expla import _project_root, textualize_graph

    root = _project_root()
    dataset = pd.read_csv(root / "dataset" / "expla_graphs" / "train_dev.tsv", sep="\t")
    nodes_df, edges_df = textualize_graph(dataset.loc[0, "graph"])
    print(visualize_parsed_graph(nodes_df, edges_df))
