from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.utils.collate import collate_fn

PATH = Path(__file__).resolve().parents[2] / "dataset" / "expla_graphs"

class ExplaGraphsDataset(Dataset):
    def __init__(self, data_dir: Path = PATH):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.text = pd.read_csv(self.data_dir / "train_dev.tsv", sep="\t")
        self.prompt = (
                "Question: Do argument 1 and argument 2 support or counter each other? "
                "Answer in one word in the form of 'support' or 'counter'.\n\nAnswer:"
            )
        self.graph = None
        self.graph_type = "Explanation Graph"

    def __len__(self):
        """Return the total number fo graphs in the dataset"""
        return (len(self.text))

    def __getitem__(self, index):
        text = self.text.iloc[index]
        graph = torch.load(self.data_dir / "graphs" / f"{index}.pt")
        question = f"Argument 1: {text['arg1']}\nArgument 2: {text['arg2']}\n{self.prompt}"
        nodes = pd.read_csv(self.data_dir / "nodes" / f"{index}.csv")
        edges = pd.read_csv(self.data_dir / "edges" / f"{index}.csv")
        desc = nodes.to_csv(index=False) + "\n" + edges.to_csv(index=False)
        return {
                "id": index,
                "label": text["label"],
                "desc": desc,
                "graph": graph,
                "question": question,
            }

    def get_idx_split(self):
        splits = {}
        for name in ("train", "val", "test"):
            split_file = self.data_dir / "split" / f"{name}_indices.txt"
            with open(split_file, "r") as f:
                splits[name] = [int(line.strip()) for line in f if line.strip()]
        return splits

if __name__ == "__main__":

    dataset = ExplaGraphsDataset()
    splits = dataset.get_idx_split()
    print(f"Total dataset len: {len(dataset)}")
    print(
        f"Splits -> train: {len(splits['train'])}, "
        f"val: {len(splits['val'])}, test: {len(splits['test'])}"
    )

    sample0, sample1 = dataset[0], dataset[1]
    batch = collate_fn([sample0, sample1])
    print("\n--- Collate Verification (B=2) ---")
    print(f"g0 nodes: {sample0['graph'].num_nodes}, g1 nodes: {sample1['graph'].num_nodes}")
    print(f"batch['graph'].x shape:          {batch['graph'].x.shape}")
    print(f"batch['graph'].edge_index shape: {batch['graph'].edge_index.shape}")
    print(f"batch['graph'].batch vector:     {batch['graph'].batch}")