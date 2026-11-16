import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def generate_split(num_samples: int, out_dir: Path, random_state: int = 42) -> tuple[int, int, int]:
    """Generates deterministic train/val/test index splits matching upstream GRAG"""
    indices = np.arange(num_samples)
    #split into 60 triain 40 temp
    train_indices , temp_indices = train_test_split(indices, test_size=0.4, random_state=random_state)

    #Split temp 50/50 into 20% val 20% test.
    val_indices , test_indices = train_test_split(temp_indices, test_size=0.5, random_state=random_state)

    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, idx_list in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
        with open(out_dir / f"{split_name}_indices.txt", "w") as f:
            f.write("\n".join(map(str, idx_list)))
    print(f"Split generated -> Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    return len(train_indices), len(val_indices), len(test_indices)

if __name__ == "__main__":
    from src.dataset.preprocess.preprocess_expla import _project_root
    root = _project_root()
    expla_dir = root / "dataset" / "expla_graphs"
    tsv_path = expla_dir / "train_dev.tsv"

    # Read dynamically without hardcoding 2766
    dataset = pd.read_csv(tsv_path, sep="\t")
    num_samples = len(dataset)

    split_dir = expla_dir / "split"
    generate_split(num_samples, split_dir, random_state=42)