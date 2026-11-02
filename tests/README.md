# Test guide — ExplaGraphs preprocessing

All commands assume the **repository root** as working directory:

```bash
cd 100_Projects/GraphRAG_Thesis/github_repos/GRAG_Data_Retrival
```

Use the project venv explicitly; do not rely on an activated shell:

```bash
PY=~/venvs/grag_env/bin/python
```

The suite is offline: it does **not** load SentenceBERT, touch CUDA, or write outside
`tempfile`. It reads `dataset/expla_graphs/train_dev.tsv` (committed, 812 KB).

## 1. Run everything

```bash
$PY -m unittest discover -s tests -v
```

Expected:

```
Ran 7 tests in ~3s
OK
```

Twelve lines of output, no debug prints.

## 2. Run one class

```bash
$PY -m unittest tests.test_preprocess_expla.TestPreprocessExpla -v
```

## 3. Run one test

```bash
$PY -m unittest tests.test_preprocess_expla.TestPreprocessExpla.test_t1_row_0_fixture -v
```

Expected for any single test:

```
test_t1_row_0_fixture (tests.test_preprocess_expla.TestPreprocessExpla) ... ok
----------------------------------------------------------------------
Ran 1 test in 0.012s
OK
```

## 4. Run every test one by one

Useful when a failure needs isolated output or timing:

```bash
for t in \
  test_t1_row_0_fixture \
  test_t2_population_invariants \
  test_t3_normalization \
  test_t4_malformed_input \
  test_t5_determinism \
  test_t6_empty_graph \
  test_t7_step_one_writes_upstream_schema
do
  echo "== $t =="
  $PY -m unittest "tests.test_preprocess_expla.TestPreprocessExpla.$t" -v || break
done
```

`-f` / `--failfast` stops the full suite at the first failure:

```bash
$PY -m unittest discover -s tests -v -f
```

## 5. What each test locks

| Test | Asserts | Typical runtime | A failure means |
|---|---|---|---|
| `test_t1_row_0_fixture` | Row 0 parses to nodes `{0: cannabis, 1: marijuana, 2: legal, 3: more available, 4: good thing}` and edges `(0,synonym of,1) (2,causes,3) (1,capable of,4) (4,desires,2)` | 0.01 s | Parser, normalization, node-ID order, or edge order changed |
| `test_t2_population_invariants` | 2,766 rows; 11,745 total edges; `4 ≤ N ≤ 9`; contiguous node IDs; every node participates in an edge; `src`/`dst` in range | ~2.9 s | Corpus file changed, or parser silently drops/adds triplets |
| `test_t3_normalization` | `( Cannabis ; SYNONYM OF ; Marijuana )` lowercases, strips, and deduplicates to two nodes | <0.01 s | Normalization or dictionary keying changed |
| `test_t4_malformed_input` | Seven malformed strings raise `ValueError`: missing field, empty field, four fields, unclosed paren, leading/trailing junk | <0.01 s | Validation was weakened; malformed blocks are being skipped again |
| `test_t5_determinism` | Parsing the same graph twice yields identical DataFrames (values and dtypes) | <0.01 s | Hidden state, set/dict iteration order, or timestamp leakage |
| `test_t6_empty_graph` | `""` and `"   "` return empty frames with columns `node_id,node_attr` and `src,edge_attr,dst` | <0.01 s | Empty-graph contract changed (it must be a no-op, not an error) |
| `test_t7_step_one_writes_upstream_schema` | A 3-row TSV produces 3 node + 3 edge CSVs; columns are exactly `node_id,node_attr` and `src,edge_attr,dst`; return counts match | ~0.1 s | `step_one` output path, filename, or CSV schema regressed |

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Wrong working directory | `cd` to the repo root (the directory containing `src/` and `tests/`) |
| `ModuleNotFoundError: No module named 'tests'` | `tests/__init__.py` missing, or wrong cwd | Restore the empty `tests/__init__.py`; run from the repo root |
| `FileNotFoundError: .../train_dev.tsv` | Raw data missing | `git checkout -- dataset/expla_graphs/train_dev.tsv` |
| `T2` takes much longer than ~3 s | Cold filesystem cache or CPU contention | Re-run once; investigate only if consistently >10 s |
| `ImportError` mentioning `matplotlib` or `networkx` | A test imported `src.utils.visualization` | Tests must not import the visualization module; check the import block |
| `pip`-style `Invalid requirement` | Lockfile problem, not a test failure | `$PY -m pip install --dry-run --no-deps --no-index -r requirements.lock.txt` |

## 7. What the suite does not cover

- `step_two` (SBERT embeddings, `graphs/{i}.pt`, `q_embs.pt`) — not implemented yet.
- `generate_split` — not implemented yet.
- `ExplaGraphsDataset` round-trip and GNN forward pass — not implemented yet.
- Oracle parity — a one-off 5,532-file hash comparison, not a unit test; see the session log
  `logs/2026-11-02_session8.md` and `README.md`.
