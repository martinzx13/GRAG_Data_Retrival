import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.dataset.preprocess.preprocess_expla import step_one, textualize_graph


class TestPreprocessExpla(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.tsv_path = self.project_root / "dataset" / "expla_graphs" / "train_dev.tsv"
        self.row_0_graph = (
            "(cannabis; synonym of; marijuana)"
            "(legal; causes; more available)"
            "(marijuana; capable of; good thing)"
            "(good thing; desires; legal)"
        )

    def test_t1_row_0_fixture(self):
        nodes, edges = textualize_graph(self.row_0_graph)

        expected_nodes = {
            0: "cannabis",
            1: "marijuana",
            2: "legal",
            3: "more available",
            4: "good thing",
        }
        self.assertEqual(dict(zip(nodes["node_id"], nodes["node_attr"])), expected_nodes)
        self.assertEqual(
            list(edges.itertuples(index=False, name=None)),
            [
                (0, "synonym of", 1),
                (2, "causes", 3),
                (1, "capable of", 4),
                (4, "desires", 2),
            ],
        )

    def test_t2_population_invariants(self):
        dataset = pd.read_csv(self.tsv_path, sep="\t")
        self.assertEqual(len(dataset), 2766, "Dataset should contain exactly 2766 rows.")

        total_edges = 0
        for graph_str in dataset["graph"]:
            nodes, edges = textualize_graph(graph_str)
            num_nodes = len(nodes)
            num_edges = len(edges)
            total_edges += num_edges

            self.assertTrue(
                4 <= num_nodes <= 9,
                f"Node count {num_nodes} out of bounds for graph: {graph_str}",
            )
            node_ids = set(nodes["node_id"])
            self.assertEqual(node_ids, set(range(num_nodes)), "Node IDs must be contiguous.")
            self.assertEqual(
                len(nodes["node_id"].unique()), num_nodes, "Duplicate node IDs detected."
            )
            self.assertTrue(
                set(edges["src"]) | set(edges["dst"]) == node_ids,
                "Every node must participate in at least one edge.",
            )
            self.assertTrue(edges["src"].max() < num_nodes, "Source ID exceeds node count.")
            self.assertTrue(edges["dst"].max() < num_nodes, "Destination ID exceeds node count.")

        self.assertEqual(
            total_edges, 11745, "Total population edges do not match expected 11745 count."
        )

    def test_t3_normalization(self):
        dirty_graph = (
            "( Cannabis ; SYNONYM OF ; Marijuana )"
            "( marijuana ; SYNONYM OF ; cannabis )"
        )
        nodes, edges = textualize_graph(dirty_graph)

        self.assertEqual(len(nodes), 2)
        self.assertEqual(list(nodes["node_attr"]), ["cannabis", "marijuana"])
        self.assertEqual(
            list(edges.itertuples(index=False, name=None)),
            [(0, "synonym of", 1), (1, "synonym of", 0)],
        )

    def test_t4_malformed_input(self):
        malformed = [
            "(a; b)",
            "(a; b; )",
            "(a;;c)",
            "(a; b; c; d)",
            "(a; b; c",
            "junk(a; b; c)",
            "(a; b; c)junk",
        ]
        for graph_str in malformed:
            with self.subTest(graph_str=graph_str):
                with self.assertRaises(ValueError):
                    textualize_graph(graph_str)

    def test_t5_determinism(self):
        nodes1, edges1 = textualize_graph(self.row_0_graph)
        nodes2, edges2 = textualize_graph(self.row_0_graph)

        pd.testing.assert_frame_equal(nodes1, nodes2)
        pd.testing.assert_frame_equal(edges1, edges2)

    def test_t6_empty_graph(self):
        for graph_str in ("", "   "):
            with self.subTest(graph_str=graph_str):
                nodes, edges = textualize_graph(graph_str)
                self.assertEqual(list(nodes.columns), ["node_id", "node_attr"])
                self.assertEqual(list(edges.columns), ["src", "edge_attr", "dst"])
                self.assertTrue(nodes.empty)
                self.assertTrue(edges.empty)

    def test_t7_step_one_writes_upstream_schema(self):
        dataset = pd.read_csv(self.tsv_path, sep="\t").head(3)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mini_tsv = tmp_path / "train_dev.tsv"
            dataset.to_csv(mini_tsv, sep="\t", index=False)

            out_dir = tmp_path / "expla_graphs"
            n_rows, n_edges = step_one(mini_tsv, out_dir, show_progress=False)

            self.assertEqual(n_rows, 3)
            self.assertEqual(n_edges, sum(len(textualize_graph(g)[1]) for g in dataset["graph"]))
            self.assertEqual(len(list((out_dir / "nodes").glob("*.csv"))), 3)
            self.assertEqual(len(list((out_dir / "edges").glob("*.csv"))), 3)

            nodes = pd.read_csv(out_dir / "nodes" / "0.csv")
            edges = pd.read_csv(out_dir / "edges" / "0.csv")
            self.assertEqual(list(nodes.columns), ["node_id", "node_attr"])
            self.assertEqual(list(edges.columns), ["src", "edge_attr", "dst"])


if __name__ == "__main__":
    unittest.main()
