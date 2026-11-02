# GRAG_Data_Retrival — Clean-Room Replication of GRAG (Hu et al., Findings NAACL 2025)

Replication target for the ONTOGRAPH M.Sc. thesis. Upstream reference: [`../GRAG/`](../GRAG/) (read-only). Paper: [aclanthology.org/2025.findings-naacl.232](https://aclanthology.org/2025.findings-naacl.232.pdf).

## Current status — 2026-11-02

**Preprocessing parity achieved; model code not started.** `step_one` (textualized graphs) is verified byte-for-byte against upstream on all 2,766 ExplaGraphs rows. Theory gating is closed for Blocks A–D; the remaining pipeline work is `step_two` (embedding), `generate_split`, and the Dataset round-trip.

| Gate | Result |
|---|---|
| Unit tests (`unittest`, 7 cases) | **7/7 pass**, 3.0 s, 12 lines of output |
| Lockfile parses (`pip install --dry-run --no-deps --no-index`) | exit 0 |
| Oracle parity, `step_one` | 2,766 node + 2,766 edge CSVs per side; **5,532 manifest lines each; 0 hash diffs** |
| Row-0 CSVs | byte-identical (`nodes/0.csv` `ee39f1ce…`, `edges/0.csv` `90a7005a…`) |

**Question-encoder gate — decided (D8).** Upstream's `step_two` builds every question from the whole DataFrame, yielding **1 unique question** (measured), not 2,766. We produce **both** paths: `literal` reproduces the released code; `corrected` (`dataset.iloc[i]`) is primary and restores query-conditioned pruning. The two runs differ only in `z_q`, so their accuracy difference isolates the graph branch's contribution. See `../literature/repo_audit_GRAG.md` §A10 and deviation D8.

## Scope decision

| dataset | status | why |
|---|---|---|
| **ExplaGraphs** | **first target** | self-contained: `train_dev.tsv` ships with the paper, no Freebase, no cached assets, no 13B verbalizer |
| WebQSP | deferred | preprocessing chain (`nodes/`, `edges/`, `graphs/`, `q_embs.pt`, `cached_*`) is absent from upstream; needs RoG + Freebase reconstruction |

## Environment

```bash
python -m venv ~/venvs/grag_env
source ~/venvs/grag_env/bin/activate
pip install -r requirements.lock.txt
```

Verified stack: `torch 2.1.2+cu121` · `numpy 1.26.4` · `torch-geometric 2.8.0.post1` · PyG extensions `+pt21cu121` · `transformers 4.36.0` · `peft 0.7.1` · `bitsandbytes 0.43.3` on an RTX 3060 Laptop (6 GB).

> **Do not `pip install` anything unpinned in this venv.** On 2026-10-27 `torch` had drifted to 2.14.0+cu130 and silently broke 5 of 6 compiled extensions. See `../literature/repo_audit_GRAG.md` §0.

Smoke test before any run:

```python
import torch
from torch_scatter import scatter_add
from torch_geometric.nn import GATConv
x = torch.randn(5, 4, device='cuda')
print(scatter_add(x, torch.tensor([0,0,1,1,2], device='cuda'), dim=0).shape)   # [3, 4]
print(GATConv(4, 8, heads=4, concat=False).cuda()(x,
      torch.tensor([[0,1,2,3,4],[1,2,3,4,0]], device='cuda')).shape)           # [5, 8]
```

## Declared deviations from upstream

| # | deviation | reason |
|---|---|---|
| D1 | ExplaGraphs first | WebQSP pipeline absent |
| D2 | 7B in 4-bit NF4, not fp16 | 13 GB > 6 GB |
| D3 | ~1B proxy for plumbing | fast iteration; separates bugs from OOM |
| D4 | single-GPU `device_map` | upstream assumes 4×20 GiB |
| D5 | 1 soft graph token (kept) | faithful to upstream code |
| D6 | L=4 (kept) | paper App. A + `config.py:38` |
| D7 | `src/dataset/__init__.py` kept empty | upstream `import src.dataset` loads SBERT on CUDA (§A11); tests stay offline |
| D8 | question encoding: both paths, corrected primary — **decided 2026-11-02** | upstream builds all questions from the whole DataFrame (§A10); literal = code parity, corrected = paper intent |
| D9 | module named `preprocess_expla.py` | cosmetic; upstream name is `expla_graphs.py` |
| D10 | malformed triplets raise; empty graph is a no-op; explicit `step_one` output root | upstream silently drops malformed blocks and hardcodes the path; no effect on the released corpus (0 diffs) |

Full rationale: `../literature/repo_audit_GRAG.md` §D.

## How to run (ExplaGraphs)

```bash
cd 100_Projects/GraphRAG_Thesis/github_repos/GRAG_Data_Retrival
source ~/venvs/grag_env/bin/activate

# Textualize all 2,766 graphs -> dataset/expla_graphs/{nodes,edges}/
python -m src.dataset.preprocess.preprocess_expla

# Tests (full suite; per-test guide in tests/README.md)
python -m unittest discover -s tests -v
```

`step_one(tsv_path, out_dir, show_progress=True)` is importable and takes an explicit output root; the CLI resolves the repo root by the `dataset/expla_graphs/train_dev.tsv` sentinel.

Outputs (git-ignored): `dataset/expla_graphs/{nodes,edges}/{i}.csv`. `step_two` additionally produces `graphs/{i}.pt`, `q_embs.pt`, and `split/`.

## Directory layout

```
src/dataset/preprocess/preprocess_expla.py   parser, textualize_graph, step_one, CLI
src/utils/visualization.py                   optional PNG rendering (matplotlib, out of pipeline)
tests/test_preprocess_expla.py               7 unittest cases
tests/README.md                              per-test execution guide and troubleshooting
dataset/expla_graphs/train_dev.tsv           raw ExplaGraphs release (812 KB, committed)
lecture/                                     lecture 01, 02 (PyG, oversmoothing, soft prompting)
requirements.lock.txt                        pinned freeze of the working venv
../literature/repo_audit_GRAG.md             audit: 11 findings, divergence + deviation registers
```
