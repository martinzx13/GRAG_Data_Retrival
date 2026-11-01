# GRAG_Data_Retrival — Clean-Room Replication of GRAG (Hu et al., Findings NAACL 2025)

Replication target for the ONTOGRAPH M.Sc. thesis. Upstream reference: [`../GRAG/`](../GRAG/) (read-only). Paper: [aclanthology.org/2025.findings-naacl.232](https://aclanthology.org/2025.findings-naacl.232.pdf).

## Current status — 2026-10-27

**No model code written yet.** Theory gating is open (C2′, D1, D2′ outstanding). What exists: lecture notes under `lecture/`, and the upstream repo audit in
[`../literature/repo_audit_GRAG.md`](../literature/repo_audit_GRAG.md).

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

Full rationale: `../literature/repo_audit_GRAG.md` §D.

## How to run (ExplaGraphs) — target for Day 2

```bash
cd ../GRAG          # preprocessing scripts live upstream
python -m src.dataset.preprocess.expla_graphs   # step_one + step_two + generate_split
```

Produces `dataset/expla_graphs/{nodes,edges,graphs}/`, `q_embs.pt`, `split/`.

## Directory layout

```
lecture/                 lecture 01 (PyTorch/PyG), 02 (oversmoothing, soft prompting)
docs/                    (empty — populate or remove)
requirements.lock.txt    pinned freeze of the working venv
../literature/repo_audit_GRAG.md   the audit that defines what "replicate" means here
```
