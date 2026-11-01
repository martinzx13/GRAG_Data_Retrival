# Lecture 02: Oversmoothing in GNNs & Soft Prompt Tuning

**Date:** 2026-09-29  
**Session:** Theory Gating — Block C cleared, Block D + Repo Audit next  
**Reading:** Li et al. (2018) "Deeper Insights into Graph Convolutional Networks" (AAAI 2018) + Lester et al. (2021) "The Power of Scale for Parameter-Efficient Prompt Tuning" (EMNLP 2021)

---

## Part A: Oversmoothing — The Mathematical Limit of GNN Depth

### A.1 GCN as Laplacian Smoothing

The graph convolution in Kipf & Welling's GCN (Eq. 4) is:

$$\mathbf{H}^{(l+1)} = \sigma\left(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{H}^{(l)}\mathbf{\Theta}^{(l)}\right)$$

where $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ (self-loops) and $\tilde{\mathbf{D}}_{ii} = \sum_j \tilde{\mathbf{A}}_{ij}$.

**Li et al. (2018), Section 3.1 — Key Insight:** The graph convolution matrix $\hat{\mathbf{A}} = \tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}$ is **exactly symmetric Laplacian smoothing**.

*Proof:* The random-walk Laplacian is $\mathbf{L}_{\text{rw}} = \mathbf{I} - \tilde{\mathbf{D}}^{-1}\tilde{\mathbf{A}}$. Standard Laplacian smoothing with $\gamma=1$ gives $\hat{\mathbf{Y}} = \tilde{\mathbf{D}}^{-1}\tilde{\mathbf{A}}\mathbf{X}$. The symmetric version replaces $\tilde{\mathbf{D}}^{-1}\tilde{\mathbf{A}}$ with $\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}$ — this is $\hat{\mathbf{A}}$.

Therefore: **one GCN layer = one step of Laplacian smoothing** on the feature matrix.

### A.2 What Laplacian Smoothing Does

For node $i$, the smoothed feature is:

$$\mathbf{y}_i = \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{\tilde{a}_{ij}}{d_i} \mathbf{x}_j$$

This is a **weighted average** of the node's own feature and its neighbors' features. Since nodes in the same cluster are densely connected, smoothing makes their features similar — this is **why GCNs work** (classification becomes easier).

### A.3 The Oversmoothing Theorem (Li et al. 2018, Theorem 1)

Let $\mathcal{G}$ have $k$ connected components $\{\mathcal{C}_i\}_{i=1}^k$. Let $\mathbf{1}^{(i)} \in \mathbb{R}^n$ be the indicator vector for component $\mathcal{C}_i$.

**Theorem 1 (Li et al. 2018):** If the graph has no bipartite components, then for any $\mathbf{w} \in \mathbb{R}^n$ and $\alpha \in (0,1]$:

$$\lim_{m \to +\infty} (\mathbf{I} - \alpha \mathbf{L}_{\text{rw}})^m \mathbf{w} = [\mathbf{1}^{(1)}, \mathbf{1}^{(2)}, \dots, \mathbf{1}^{(k)}] \boldsymbol{\theta}_1$$

$$\lim_{m \to +\infty} (\mathbf{I} - \alpha \mathbf{L}_{\text{sym}})^m \mathbf{w} = \mathbf{D}^{-1/2} [\mathbf{1}^{(1)}, \mathbf{1}^{(2)}, \dots, \mathbf{1}^{(k)}] \boldsymbol{\theta}_2$$

where $\boldsymbol{\theta}_1, \boldsymbol{\theta}_2 \in \mathbb{R}^k$.

**In words:** Repeated Laplacian smoothing converges all node features to a **linear combination of component indicators**. Within each connected component, all nodes end up with **identical features** (up to degree scaling for symmetric Laplacian).

**Proof Sketch:**
1. $\mathbf{L}_{\text{rw}}$ and $\mathbf{L}_{\text{sym}}$ have eigenspaces for eigenvalue 0 spanned by $\{\mathbf{1}^{(i)}\}$ and $\{\mathbf{D}^{-1/2}\mathbf{1}^{(i)}\}$ respectively (Von Luxburg 2007).
2. For $\alpha \in (0,1]$, eigenvalues of $(\mathbf{I} - \alpha \mathbf{L})$ are in $(-1,1]$, with eigenspace for eigenvalue 1 = the 0-eigenspace of $\mathbf{L}$.
3. Repeated multiplication by $(\mathbf{I} - \alpha \mathbf{L})$ projects any vector onto the eigenspace of eigenvalue 1.

### A.3.1 The Consequence for Deep GNNs

Each GCN layer applies one step of smoothing. After $L$ layers:

$$\mathbf{H}^{(L)} = \sigma\left( \hat{\mathbf{A}}^L \mathbf{X} \mathbf{\Theta}^{(0)} \dots \mathbf{\Theta}^{(L-1)} \right)$$

As $L$ increases, $\hat{\mathbf{A}}^L$ drives node features toward the component-wise constant. **All nodes in the same connected component become indistinguishable.**

On the Zachary karate club (34 nodes, 2 classes), Li et al. Figure 2 shows:
- 1 layer: features not well separated
- 2 layers: classes separated well
- 3+ layers: features mix and collapse

### A.4 The $L \leq K$ Bound for GRAG

In GRAG, each ego-graph $G[\mathcal{N}_K(v)]$ has radius $K=2$.

**Message-passing radius = layer depth.** After $l$ layers, node $v$ has aggregated information from all nodes within $l$ hops.

- $L = 1$: center sees 1-hop neighbors
- $L = 2$: center sees entire ego-graph (radius 2)
- $L = 3$: center would pull from outside the ego-graph — but there IS no outside; it just re-mixes already-mixed signals

**Bound:** $L \leq K$. For GRAG with $K=2$: **maximum useful depth is $L=2$**. Deeper GNNs add no new neighborhood information but amplify oversmoothing.

---

## Part B: Soft Prompt Tuning — Conditioning Frozen LLMs

### B.1 The Problem: Adaptation Without Fine-tuning

Standard fine-tuning: update all $\theta$ (7B params for LLaMA-2-7B) — costly storage, serving, catastrophic forgetting.

Prompt design (GPT-3): hand-craft text prompts — limited by context window, no gradient signal.

**Prefix tuning (Li & Liang 2021):** prepend trainable activations at *every* transformer layer — still many params, complex.

### B.2 Prompt Tuning (Lester et al. 2021)

**Core idea:** Add $p$ **soft prompt tokens** with their own trainable embeddings $\mathbf{P}_e \in \mathbb{R}^{p \times e}$, where $e$ = embedding dimension (4096 for LLaMA-2-7B).

**Forward pass:**
1. Embed input tokens: $\mathbf{X}_e \in \mathbb{R}^{n \times e}$
2. Concatenate: $[\mathbf{P}_e; \mathbf{X}_e] \in \mathbb{R}^{(p+n) \times e}$
3. Pass through **frozen** transformer layers
4. Only $\mathbf{P}_e$ receives gradients via backprop

**Parameter count:** $p \times e$ task-specific params. For LLaMA-2-7B with $p=100$: $100 \times 4096 = 409,600$ params vs 7B full — **0.006%**.

### B.3 What Gets Trained vs. Frozen

| Component | Parameters | Frozen? | Gradients? |
|---|---|---|---|
| LLaMA-2 backbone $\theta$ | 7B | ✅ YES | ❌ NO |
| Soft prompt embeddings $\mathbf{P}_e$ | $p \times e$ | ❌ NO | ✅ YES |
| GRAG pruner MLPs $\phi_1, \phi_2$ | small | ❌ NO | ✅ YES |
| GRAG graph projector $\phi_3$ | small | ❌ NO | ✅ YES |
| GRAG GNN encoder $\Phi$ | ~few M | ❌ NO | ✅ YES |

### B.4 Table 2 Flags Decoded

GRAG row: **Prompt tuning ✓ / Fine-tuning ✗**

This means: the reported 0.7236 Hit@1 comes from a configuration where:
- LLaMA-2 weights $\theta$ are **never updated** (Fine-tuning ✗)
- Only soft prompt + $\phi_1, \phi_2, \phi_3, \Phi$ are trained (Prompt tuning ✓)

Adding LoRA on top (0.7236 → 0.7275) gives marginal gain because the soft tokens already provide the primary adaptation signal.

### B.5 Gradient Flow for GRAG

The loss is autoregressive cross-entropy:

$$\mathcal{L} = -\sum_{t=1}^T \log P(y_t \mid y_{<t}, \mathbf{h}_{\hat{g}}, D_g, q)$$

Gradient path to $\phi_1$ (pruner MLP for nodes):

$$\frac{\partial \mathcal{L}}{\partial \phi_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_{\hat{g}}} \frac{\partial \mathbf{h}_{\hat{g}}}{\partial \text{GNN}(\hat{g})} \frac{\partial \text{GNN}(\hat{g})}{\partial \alpha} \frac{\partial \alpha}{\partial \phi_1}$$

**Key point:** $\phi_1$ and $\phi_2$ (pruner MLPs) produce $\alpha$ which gates **both**:
1. Text view: token masking (Eq. 7–8) → affects $D_g$ → affects $\mathbf{h}_T$
2. Graph view: message passing gating (Eq. 9) → affects $\mathbf{h}_{\hat{g}}$

So $\phi_1/\phi_2$ receive learning signal **through generation quality** (answer loss), not through retrieval quality (cosine similarity). The retrieval step is not differentiable w.r.t. $\phi_1/\phi_2$ — the gradient comes entirely from the LLM's answer loss.

---

## Part C: Connection to GRAG Architecture

| GRAG Component | Oversmoothing Connection | Soft Prompt Connection |
|---|---|---|
| GNN encoder $\Phi$ (GAT, 4 layers, 4 heads, 1024 hidden) | **But GRAG uses $L=2$** (not 4!) because ego-graph radius $K=2$ → $L \leq K$ | Output projected by $\phi_3$ to $d_{\text{LLM}}=4096$ |
| Pruner MLPs $\phi_1, \phi_2$ | Not affected (shallow, no message passing) | Produce $\alpha$ that gates both views |
| Graph projector $\phi_3$ | Projects GNN output to match LLaMA-2 embedding dim | Enables $[\mathbf{h}_{\hat{g}}; \mathbf{h}_T]$ concatenation |
| Soft tokens $\mathbf{h}_{\hat{g}}$ | — | Embedding-level injection = prompt tuning; only $\mathbf{P}_e$ (here $\mathbf{h}_{\hat{g}}$) + $\phi$'s trained |

---

## Part D: Diagnostic Questions (P4–P5)

### P4: Oversmoothing Quantification
> **Setup:** Zachary's karate club network (34 nodes, 2 classes, 78 edges). GCN with 16-dim hidden layers, random weights, ReLU.
> 
> **Question:** After how many layers $L$ do the two-class node embeddings collapse to within $\epsilon = 0.01$ cosine similarity of each other? Derive the relationship between graph diameter, spectral gap of $\hat{\mathbf{A}}$, and the number of smoothing steps to reach $\epsilon$-oversmoothing. How does this bound change for a sparse graph (e.g., WebQSP ego-graph: ~1370 nodes, avg degree ~6) vs. a dense graph (e.g., citation graph)?

### P5: Soft Prompt Gradient Flow
> **Setup:** LLaMA-2-7B ($d=4096$, $L=32$ layers, $H=32$ heads). GRAG soft prefix length $p=10$. Frozen backbone $\theta$. Trainable: $\mathbf{P}_e \in \mathbb{R}^{10 \times 4096}$, $\phi_1, \phi_2, \phi_3, \Phi$.
> 
> **Question:** 
> 1. During backprop, at which layer does the gradient first reach the soft prompt embeddings? (Answer: after the embedding layer, before Layer 1 self-attention).
> 2. The gradient $\partial \mathcal{L} / \partial \mathbf{P}_e$ has shape $(10, 4096)$. If batch size $B=2$, sequence length $S=512$, how many gradient accumulation steps are needed to match the effective batch size of full fine-tuning with $B=8$? (Note: fine-tuning updates 7B params; prompt tuning updates 40,960 params — parameter-efficiency ratio is $7\text{B} / 41\text{K} \approx 170,000\times$).
> 3. If you mistakenly put `requires_grad=True` on the LLaMA-2 embedding layer weights instead of only $\mathbf{P}_e$, what would happen to the prompt tuning dynamics?

---

## References

1. **Li, Q., Han, Z., & Wu, X.-M. (2018).** Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning. *AAAI 2018*. arXiv:1801.07606
2. **Lester, B., Al-Rfou, R., & Constant, N. (2021).** The Power of Scale for Parameter-Efficient Prompt Tuning. *EMNLP 2021*. arXiv:2104.08691
3. **Kipf, T.N. & Welling, M. (2017).** Semi-Supervised Classification with Graph Convolutional Networks. *ICLR 2017*. arXiv:1609.02907
4. **Hu et al. (2025).** GRAG: Graph Retrieval-Augmented Generation. *Findings of NAACL 2025*.

---

## Summary for Next Session

- **Oversmoothing** is not a heuristic — it's a spectral convergence theorem. $L \leq K$ is forced by the message-passing radius matching the ego-graph radius.
- **Soft prompt tuning** is the minimal parameter-efficient adaptation: freeze $\theta$ (7B), train only $\mathbf{P}_e$ ($p \times e$) + task-specific heads ($\phi_1, \phi_2, \phi_3, \Phi$).
- **GRAG's architecture** respects both: GNN depth $L=2$ (not 4 as GAT paper suggests) because $K=2$; LLaMA-2 receives graph info via soft tokens at embedding level, not by fine-tuning.

**Next gate:** D1 (gradient flow trace) + D2 (prompt tuning config) + Repo Audit. Then Day 1 clears → Day 2 (Data Pipeline).