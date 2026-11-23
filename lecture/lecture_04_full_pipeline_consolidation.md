# Lecture 04: Master Consolidation — From Raw Triplets to End-to-End Graph-Conditioned LLM Training

**Date:** 2026-11-18  
**Scope:** Phases 1–5 Complete Consolidation (Data Ingestion $\to$ SBERT Geometry $\to$ PyG Disjoint Batching $\to$ Soft-Pruned GAT $\to$ Prefix-Injected Causal LLM)  
**Repository:** `100_Projects/GraphRAG_Thesis/github_repos/GRAG_Data_Retrival/`

---

## Module 1: The Big Picture — Why GRAG Exists & How the Dual-View Pipeline Works

### 1.1 Why Standard Text RAG Fails on Graphs
Standard Retrieval-Augmented Generation (RAG) retrieves flat text chunks and concatenates them into an LLM prompt. When reasoning requires **multi-hop relational structure** (e.g., $(A \xrightarrow{\text{causes}} B) \land (B \xrightarrow{\text{inhibits}} C)$), flat text loses explicit topological connectivity and floods the context window with irrelevant 1-hop and 2-hop distractors.

### 1.2 GRAG's Dual-View Solution
GRAG feeds the retrieved subgraph to the LLM through **two complementary channels simultaneously**:
1. **The Text View ($D_g$):** The node and edge CSV tables serialized as plain text (`desc`), giving the LLM exact lexical surface words.
2. **The Graph View ($\hat{G}$):** A 4-layer Graph Attention Network (`GAT`) that softly prunes irrelevant nodes/edges against the query, aggregates 4-hop neighborhood topology, pools the graph into a single vector, and projects it into **one continuous soft prefix token** $\mathbf{e}_{\text{prefix}} \in \mathbb{R}^{d_{\text{LLM}}}$ injected directly into the LLM's input embedding layer.

### 1.3 Master End-to-End Tensor Trace (`B = 2` Batch: `g0` = 5 nodes/4 edges, `g1` = 6 nodes/5 edges)

| Stage | File & Function | Input Tensor / Object | Output Tensor / Object | Coordinate Space |
|---|---|---|---|---|
| **Phase 1** | `preprocess_expla.py:step_one` | `train_dev.tsv` row string | `nodes/{i}.csv`, `edges/{i}.csv` | Raw English text |
| **Phase 2** | `lm_modeling.py:encode_texts` | Node, edge, & question strings | `graphs/{i}.pt` (`x [N_i, 1024]`, `edge_attr [E_i, 1024]`) | Unit-sphere SBERT ($\mathbb{R}^{1024}$, $\|\mathbf{z}\|_2=1$) |
| **Phase 3** | `collate.py:collate_fn` | `[dataset[0], dataset[1]]` | `batch['graph'].x` `[11, 1024]`, `edge_index` `[2, 9]`, `batch` `[11]` | Disjoint-union GNN space ($\mathbb{R}^{1024}$) |
| **Phase 4A** | `gnn.py:GAT.forward` (Pruner) | `x [11, 1024]`, `q_node [11, 1024]` | $\tilde{\mathbf{x}} = \mathbf{x} \odot \sigma(\text{MLP}_x(\|\mathbf{x}-\mathbf{q}\|))$ `[11, 1024]` | Soft-gated GNN space ($\mathbb{R}^{1024}$) |
| **Phase 4B** | `gnn.py:GAT.forward` (Convs) | $\tilde{\mathbf{x}}$ `[11, 1024]`, $\tilde{\mathbf{e}}$ `[9, 1024]` | `n_embeds` `[11, 1024]` (4-hop refined node vectors) | 4-hop GNN space ($\mathbb{R}^{1024}$) |
| **Phase 5A** | `graph_llm.py:encode_graphs` | `n_embeds [11, 1024]`, `batch [11]` | `g_embeds` `[2, 1024]` via `scatter(reduce='mean')` | Graph-level GNN space ($\mathbb{R}^{1024}$) |
| **Phase 5B** | `graph_llm.py:projector` | `g_embeds` `[2, 1024]` | `graph_embeds` `[2, d_LLM]` (`2048` on 1B, `4096` on 7B) | **LLM Token Embedding Space ($\mathbb{R}^{d_{\text{LLM}}}$)** |
| **Phase 5C** | `graph_llm.py:forward` | `[PAD.. | BOS | graph_embeds[i] | text]` | `inputs_embeds` `[2, L_max, d_LLM]` $\to$ Scalar `loss` | Causal Transformer Autograd DAG |

---

## Module 2: Phases 1 & 2 — Feature Extraction & Semantic Geometry

### 2.1 Mask-Aware Mean Pooling (`src/utils/lm_modeling.py`)
When encoding a batch of variable-length strings with SentenceBERT (`all-roberta-large-v1`), shorter strings are padded with `<pad>` tokens (`attention_mask = 0`).
- **Why `.mean(dim=1)` is wrong:** Averaging across all $L$ positions includes the `<pad>` vectors in the numerator and divides by $L$, diluting short 3-word node labels with 90% padding noise.
- **Mask-Aware Formula:**
  $$\mathbf{z}_{\text{raw}} = \frac{\sum_{t=1}^{L} \mathbf{h}_t \cdot m_t}{\max\!\left(\sum_{t=1}^{L} m_t, \; 10^{-9}\right)}, \qquad m_t \in \{0, 1\}$$

### 2.2 Unit-Sphere L2 Normalization ($\|\mathbf{z}\|_2 = 1$)
After mean pooling, we normalize every vector to unit length: $\mathbf{z} = \frac{\mathbf{z}_{\text{raw}}}{\|\mathbf{z}_{\text{raw}}\|_2}$.
- **Mathematical Payoff:** On the unit hypersphere ($\|\mathbf{z}_n\|_2^2 = \|\mathbf{z}_q\|_2^2 = 1$), squared Euclidean distance is strictly a monotonic function of Cosine Similarity:
  $$\|\mathbf{z}_n - \mathbf{z}_q\|_2^2 = \|\mathbf{z}_n\|_2^2 + \|\mathbf{z}_q\|_2^2 - 2\mathbf{z}_n^\top \mathbf{z}_q = 2\big(1 - \cos(\mathbf{z}_n, \mathbf{z}_q)\big)$$
  It also guarantees that every coordinate lies in $[-1, +1]$, preventing extreme distances from saturating the pruning MLP's `.sigmoid()` into flat zero-gradient tails.

### 2.3 Dynamic Per-Batch Padding vs. Static 512 Padding
Self-attention memory scales quadratically with sequence length: $\mathcal{O}(B \cdot L^2)$.
- Upstream padded short 4-word node labels globally to $L = 512$.
- Dynamic per-batch padding pads only to the longest string in the current mini-batch ($L_{\text{batch}} \approx 10$ for nodes), cutting attention memory by $(512 / 10)^2 \approx \mathbf{2,600\times}$.

### 2.4 Upstream Bug A10 (`literal` vs. `corrected` Question Encoding)
- Upstream wrote `f"Argument 1: {text.arg1}..."` where `text` was the entire DataFrame, producing **1 identical string** across all 2,766 graphs (`q_embs.pt` was constant $\implies$ pruning was completely question-blind).
- Indexing `row['arg1']` and `row['arg2']` per row (`mode='corrected'`) produces **2,766 unique query vectors** (`q_embs_corrected.pt`), restoring query-conditioned pruning.

---

## Module 3: Phase 3 — Disjoint-Union Graph Batching (`expla_graphs.py` & `collate.py`)

### 3.1 Why Graphs Cannot Use `torch.stack`
Graph 0 has $|V_0| = 5$ nodes (`[5, 1024]`) and Graph 1 has $|V_1| = 6$ nodes (`[6, 1024]`). `torch.stack` requires identical shapes along every axis and crashes immediately.

### 3.2 Block-Diagonal Disjoint Union & The Cumulative Node-Offset Law
`Batch.from_data_list([g0, g1, ..., g_{B-1}])` merges $B$ disconnected graphs into one giant supergraph:
1. **Node features (`x`):** Stacked vertically into $\big[\sum_{b=0}^{B-1} |V_b|, \; 1024\big]$.
2. **Edge indices (`edge_index`):** Concatenated horizontally into $\big[2, \; \sum_{b=0}^{B-1} |E_b|\big]$, where every node ID inside Graph $k$'s `edge_index` is shifted by the **cumulative number of nodes in all preceding graphs**:
   $$\text{Offset}(g_k) = \sum_{i=0}^{k-1} |V_i|$$
3. **Batch membership vector (`graphs.batch`):** A 1D integer tensor of length $\sum |V_b|$ recording which graph owns each row of `x`:
   $$\text{graphs.batch} = [\underbrace{0, 0, 0, 0, 0}_{|V_0|=5}, \; \underbrace{1, 1, 1, 1, 1, 1}_{|V_1|=6}]$$

---

## Module 4: Phase 4 — Soft Pruning, Multi-Head GAT, & Spectral Energy (`gnn.py`)

### 4.1 The Soft Pruner Equation & Differentiability
Before message passing, every node $\mathbf{x}_i$ and edge $\mathbf{e}_{uv}$ is gated against the query vector $\mathbf{q}$:
$$\mathbf{d}_i = |\mathbf{x}_i - \mathbf{q}|, \qquad z_i = \text{MLP}_{\phi_1}(\mathbf{d}_i), \qquad \alpha_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}} \in (0, 1), \qquad \tilde{\mathbf{x}}_i = \alpha_i \cdot \mathbf{x}_i$$

- **Why Soft Sigmoid Instead of Hard Deletion (`if z_i < 0: delete`):**
  A hard step function $H(z_i)$ has derivative $\frac{\partial H}{\partial z_i} = 0$ everywhere flat. By the chain rule:
  $$\frac{\partial \mathcal{L}}{\partial \phi_1} = \frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{x}}_i} \cdot \mathbf{x}_i \cdot \frac{\partial \alpha_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial \phi_1}$$
  If $\frac{\partial \alpha_i}{\partial z_i} = 0$, the gradient is zero and the MLP never updates. With Sigmoid at initialization ($z_i \approx 0$), $\sigma(0) = \mathbf{0.5}$ (gate half-open) and $\sigma'(0) = 0.5(1 - 0.5) = \mathbf{0.25}$ (maximum gradient slope).
- **Even vs. Odd Distance Operator (Audit Finding #2):**
  `"euclidean"` ($\sqrt{(\mathbf{x}-\mathbf{q})^2} = |\mathbf{x}-\mathbf{q}|$) is an **even function** ($|a - b| = |b - a|$), meaning it is blind to sign/direction. `"element"` ($\mathbf{x} - \mathbf{q}$) is an **odd function** that preserves directionality.

### 4.2 Multi-Head `GATConv` (`heads=4, concat=False`) & Audit Finding #12 (`edge_dim`)
- **`concat=False` (Head Averaging):** Each of the $K=4$ attention heads computes a `[N, 1024]` matrix. Instead of concatenating (`[N, 4096]`) or taking an unnormalized sum (which would multiply norms by $4^4 = 256\times$ across 4 layers), `concat=False` computes the **arithmetic mean**:
  $$\mathbf{h}'_i = \frac{1}{4}\sum_{k=1}^{4} \mathbf{h}_i^{(k)} \in \mathbb{R}^{1024}$$
- **Audit Finding #12 (`edge_dim=None` vs. `edge_dim=1024`):**
  - When `edge_dim=None` (upstream bug at `gnn.py:28-34`), PyG sets `conv.lin_edge = None` and **silently ignores `edge_attr`**, leaving `mlp_edge_attr.grad = None` (the edge pruner never trains!).
  - When `edge_dim=1024`, PyG allocates `lin_edge = Linear(1024, 4096, bias=False)` per layer ($1024 \times (4 \times 1024) = 4,194,304$ params/layer, or $16.77\text{M}$ params across 4 layers), incorporating edge semantics into attention scores $\alpha_{ij}$ and unblocking `mlp_edge_attr` gradients.

### 4.3 `BatchNorm1d`, Xavier Initialization, & Dirichlet Energy ($\mathcal{E}(\mathbf{H}^{(l)})$)
- **`BatchNorm1d(1024)`:** The `1d` means zero spatial axes (`[N, C]` 2D node table, vs. `2d` for `[B, C, H, W]` images). Normalizes each of the 1,024 channels to zero mean and unit variance across all nodes in the batch.
- **Xavier / Glorot Initialization:** Samples weights with variance $\text{Var}(W_{ij}) = \frac{2}{d_{\text{in}} + d_{\text{out}}}$ so forward activations and backward gradients neither explode nor vanish across layers.
- **Dirichlet Energy (Oversmoothing Thermometer):**
  $$\mathcal{E}(\mathbf{H}^{(l)}) = \frac{1}{|\mathcal{E}|} \sum_{(u, v) \in \mathcal{E}} \|\mathbf{h}_u^{(l)} - \mathbf{h}_v^{(l)}\|_2^2$$
  Measures average squared difference between connected neighbors. If $\mathcal{E}(\mathbf{H}^{(L)}) \to 0$, all connected nodes have collapsed into identical vectors (**oversmoothing**).

---

## Module 5: Phase 5 — Readout, Projection, Prefix Injection, & Gradient Starvation (`graph_llm.py`)

### 5.1 Two Separate Operations: `scatter` (Count) vs. `GraphProjector` (Space)
1. **Node-to-Graph Readout (`scatter`):**
   ```python
   g_embeds = scatter(n_embeds, graphs.batch, dim=0, dim_size=B, reduce="mean")
   ```
   Averages the node vectors within each graph ID ($0 \dots B-1$). Collapses `[11, 1024] -> [2, 1024]`. **Still in 1,024-d GNN space.**
2. **Coordinate Space Projection (`self.projector`):**
   $$\mathbf{e}_{\text{prefix}} = \mathbf{W}_2 \, \sigma(\mathbf{W}_1 \mathbf{g} + \mathbf{b}_1) + \mathbf{b}_2 \qquad (1024 \to 2048 \xrightarrow{\text{Sigmoid}} d_{\text{LLM}})$$
   Translates `[2, 1024]` from GNN space into `[2, d_LLM]` LLM embedding space (`d_LLM = self.model.config.hidden_size`: `2048` on TinyLlama-1.1B, `4096` on LLaMA-2-7B).

### 5.2 Embedding-Level Prefix Injection, Left-Padding, & `-100` Loss Masking
For each sample $i$ in the batch, the sequence of token embeddings is assembled at the input embedding layer:
$$\mathbf{E}_{\text{in}}^{(i)} = \big[\underbrace{\mathbf{E}_{\text{PAD}}}_{0\text{ attn}} \;\big|\; \underbrace{\mathbf{E}_{\text{BOS}}}_{\texttt{<s>[INST]}} \;\big|\; \underbrace{\mathbf{e}_{\text{prefix}}^{(i)}}_{[1, d_{\text{LLM}}]} \;\big|\; \underbrace{\mathbf{E}_{\text{desc + question + [/INST]}}}_{\text{Prompt}} \;\big|\; \underbrace{\mathbf{E}_{\text{label + </s>}}}_{\text{Target Answer}}\big]$$

- **Why Left-Padding (`padding_side = "left"`):** During autoregressive decoding (`model.generate`), the next token is predicted from the final column `logits[:, -1, :]`. Right-padding would place `[PAD]` tokens at the end of shorter prompts, corrupting generation. Left-padding aligns every prompt's final token (`Answer:`) at column `-1`.
- **Why `IGNORE_INDEX = -100` Masks Both Padding AND Prompt (`graph_llm.py:138, 145`):**
  PyTorch's `CrossEntropyLoss(ignore_index=-100)` ignores every position labeled `-100`. By setting `-100` on `[PAD]`, `[BOS]`, `e_prefix`, `desc`, and `question`, **100% of the loss and backward gradient is driven by predicting the target label (`'support'` or `'counter'` + `</s>`)**.

### 5.3 The Empirical Gradient Attenuation Cascade (Failure Mode F9)
In Session 13, your CUDA smoke test measured the exact norm of the backward gradient at each stage when the LLM backbone is frozen (`Frozen LLM grads: 0`):
- **`Projector` (`0.150338`):** Closest to the LLM attention layers.
- **`GAT mlp_x` (`0.000247`):** Attenuated $\sim 608\times$ after passing backward through the Projector Sigmoid ($\le 0.25$), `scatter(mean)` ($\frac{1}{|V_b|}$), 4 GAT layers, and `mlp_x` Sigmoid ($\le 0.25$).
- **`GAT mlp_edge_attr` ($\sim 5.8 \times 10^{-7}$):** Attenuated an additional $\sim 420\times$ because `edge_attr` enters `GATConv` only through the softmax attention score derivative ($\frac{\partial \alpha_{ij}}{\partial \mathbf{e}_{ij}}$) rather than the linear value path.

---

## Module 6: Self-Test Mastery Check (5 Study Questions)

Use these 5 questions to test your recall before we build the final training loop (`train.py`):
1. **Geometry:** Why does L2-normalizing SBERT embeddings ($\|\mathbf{z}\|_2 = 1$) make Euclidean distance equivalent to Cosine distance, and how does it protect `mlp_x`'s Sigmoid gate?
2. **Batching:** If a batch of $B=3$ graphs has $|V_0|=6$, $|V_1|=4$, and $|V_2|=9$ nodes, what is the shape of `batch.x`, what is the shape of `g_embeds` after `scatter(reduce='mean')`, and what integer offset is added to Graph 2's `edge_index`?
3. **GNN & Audit #12:** What happens to `edge_attr` and `mlp_edge_attr.grad` when `GATConv` is initialized with `edge_dim=None` vs. `edge_dim=1024`?
4. **LLM Interface:** What is the exact difference between the job of `scatter(reduce='mean')` and the job of `self.projector`?
5. **Loss & Backprop:** Why do we mask the prompt and graph prefix token with `IGNORE_INDEX = -100` in `label_input_ids`, and how does `mlp_x` still receive a gradient if the graph token position itself has label `-100`?
