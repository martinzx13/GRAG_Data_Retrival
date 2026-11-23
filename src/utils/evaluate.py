import json 
import re 
import pandas as pd 

def get_accuracy_expla_graphs(eval_output: list[dict], path:str) -> float:
    """Save batch predictions to JSONL and compute stance accuracy ('support' vs 'counter')."""
    df = pd.concat([pd.DataFrame(d) for d in eval_output], ignore_index=True)
    with open(path, "w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(dict(row)) + "\n")

    correct = 0
    for pred, label in zip(df["pred"], df["label"]):
        matches = re.findall(r"support|Support|Counter|counter", str(pred).strip())
        if len(matches) > 0 and matches[0].lower() == str(label).strip().lower():
            correct += 1

    return correct / len(df) if len(df) > 0 else 0.0


eval_funcs = {
    "expla_graphs": get_accuracy_expla_graphs,
}
