# Lecture 03: Gradient Starvation, Compounding Attenuation, and the Optimization Mechanics of Soft Pruning

**Date:** 2026-10-30  
**Context:** Block D Gating — Training Dynamics & Parameter Allocation in GRAG  
**Core Reference:** Hu et al. (Findings of NAACL 2025) vs. Upstream Codebase (`github_repos/GRAG/`)  
**Target Systems:** Graph Conditioning in Frozen Backbones, Multi-Hop Pruning, Backpropagation through Message-Passing Circuits

---

## Part 1: The Analytical Backward Path

In GRAG, the entire system is nominally trained end-to-end using the autoregressive generation loss $\mathcal{L}_{\text{gen}}$ over answer tokens (`graph_llm.py:171-176`). The pruner MLPs $\phi_1$ (for nodes) and $\phi_2$ (for edges) do not have direct access to supervised retrieval targets; they receive gradients solely through the generation quality of the frozen LLM.

To analyze whether $\phi_1$ can realistically learn, we factorize the full Jacobian across the intermediate computational tensors:

$$\frac{\partial \mathcal{L}_{\text{gen}}}{\partial \phi_1} = 
\underbrace{\frac{\partial \mathcal{L}_{\text{gen}}}{\partial \mathbf{e}_{\text{prefix}}}}_{\text{Step 1}} \cdot 
\underbrace{\frac{\partial \mathbf{e}_{\text{prefix}}}{\partial \mathbf{z}_g}}_{\text{Step 2}} \cdot 
\underbrace{\frac{\partial \mathbf{z}_g}{\partial \mathbf{H}^{(4)}}}_{\text{Step 3}} \cdot 
\underbrace{\frac{\partial \mathbf{H}^{(4)}}{\partial \mathbf{H}^{(0)}}}_{\text{Step 4}} \cdot 
\underbrace{\frac{\partial \mathbf{H}^{(0)}}{\partial \boldsymbol{\alpha}_n}}_{\text{Step 5}} \cdot 
\underbrace{\frac{\partial \boldsymbol{\alpha}_n}{\partial \phi_1}}_{\text{Step 6}}$$

```
  ∂ℒ_gen       ∂ℒ_gen        ∂𝐞_prefix      ∂𝐳_g        ∂𝐇^(4)      ∂𝐇^(0)        ∂𝛂_n
  ────── = ───────────── · ──────────── · ──────── ·  ──────── ·  ──────── ·  ─────────
   ∂φ₁     ∂𝐞_prefix          ∂𝐳_g         ∂𝐇^(4)      ∂𝐇^(0)       ∂𝛂_n         ∂φ₁
           ╰──────┬──────╯ ╰─────┬──────╯ ╰───┬───╯   ╰───┬────╯  ╰───┬────╯   ╰───┬─────╯
               Step 1          Step 2       Step 3      Step 4      Step 5       Step 6
             (Attention       (Projector   (Mean-Pool  (4-Layer    (Feature      (Pruner
              Dilution)        Sigmoid)     Collapse)     GAT)      Scaling)     Sigmoid)
```

---

## Part 2: Step-by-Step Jacobian Evaluation

### Step 1: Prefix Token in Frozen LLM (`graph_llm.py:150`)
The projected graph representation $\mathbf{e}_{\text{prefix}} \in \mathbb{R}^{4096}$ is concatenated at the embedding level:
```python
# graph_llm.py:150
inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
```
The graph embedding occupies exactly **one token slot** in a context sequence of length $T \approx 512$. In multi-head self-attention, each query token attends over the entire sequence. The gradient arriving at this single prefix slot from the autoregressive targets is diluted across the sequence length:
$$\left\| \frac{\partial \mathcal{L}_{\text{gen}}}{\partial \mathbf{e}_{\text{prefix}}} \right\| \sim \mathcal{O}\left(\frac{1}{T}\right) \approx \frac{1}{512} \approx 1.95 \times 10^{-3}$$

### Step 2: Projector Bottleneck $\phi_3$ (`graph_llm.py:88-92`)
The projector maps the graph summary $\mathbf{z}_g \in \mathbb{R}^{1024}$ to $\mathbf{e}_{\text{prefix}} \in \mathbb{R}^{4096}$:
```python
# graph_llm.py:88-92
self.projector = nn.Sequential(
    nn.Linear(gnn_hidden_dim, gnn_hidden_dim * 2),  # 1024 -> 2048
    nn.Sigmoid(),
    nn.Linear(gnn_hidden_dim * 2, llm_hidden_dim)   # 2048 -> 4096
)
```
Let $\mathbf{s} = \mathbf{W}_1 \mathbf{z}_g + \mathbf{b}_1$. The Jacobian contains the diagonal derivative matrix of the `Sigmoid` activation:
$$\mathbf{J}_{\sigma} = \text{diag}\left(\sigma(\mathbf{s}) \odot (1 - \sigma(\mathbf{s}))\right)$$
Because $\sigma'(x) \leq 0.25$ for all $x \in \mathbb{R}$, this introduces an unconditional contraction factor:
$$\sup_{s} |\sigma'(s)| = 0.25$$

### Step 3: Subgraph Mean-Pooling (`graph_llm.py:120`)
Before projection, node representations from the final GAT layer are aggregated via scatter mean:
```python
# graph_llm.py:120
graph_embeds = scatter(node_embeds, batch, dim=0, reduce='mean')
```
For ego-subgraph $g$ with node set $V_g$, the graph embedding is:
$$\mathbf{z}_g = \frac{1}{|V_g|} \sum_{v \in V_g} \mathbf{h}_v^{(4)}$$
Differentiating with respect to the representation of an individual node $v \in V_g$:
$$\frac{\partial \mathbf{z}_g}{\partial \mathbf{h}_v^{(4)}} = \frac{1}{|V_g|} \mathbf{I}_{1024}$$
For a typical WebQSP or ExplaGraphs retrieved subgraph with $|V_g| \approx 40$ nodes, the backward signal is divided by 40:
$$\frac{1}{|V_g|} \approx \frac{1}{40} = 0.025$$

### Step 4: 4-Layer GAT Backbone (`gnn.py:100-112`)
The representation passes backward through 4 layers of attention aggregation, batch normalization, and ReLU:
$$\mathbf{H}^{(4)} \leftarrow \text{GAT}^{(4)} \leftarrow \text{BN}^{(3)} \leftarrow \text{ReLU} \dots \leftarrow \text{GAT}^{(1)} \leftarrow \mathbf{H}^{(0)}$$
At each layer $l$, the gradient is multiplied by the attention Jacobian:
$$\frac{\partial \mathbf{h}_i^{(l)}}{\partial \mathbf{h}_j^{(l-1)}} = \alpha_{ij}^{(l)} \mathbf{W}^{(l)} + \sum_{k} \mathbf{W}^{(l)} \mathbf{h}_k^{(l-1)} \frac{\partial \alpha_{ik}^{(l)}}{\partial \mathbf{h}_j^{(l-1)}}$$
While GAT's dynamic attention weights do not collapse to zero by design, passing through 4 successive layers of affine projections and ReLU gates ($\mathbf{1}_{\{z > 0\}}$) introduces further selective gradient killing.

### Step 5: Input Feature Scaling (`gnn.py:75`)
Pruning weights $\boldsymbol{\alpha}_n$ scale the initial node features:
```python
# gnn.py:75
x *= x_weights
```
Expressed symbolically: $\mathbf{h}_v^{(0)} = \alpha_{n, v} \cdot \mathbf{x}_v$, yielding:
$$\frac{\partial \mathbf{h}_v^{(0)}}{\partial \alpha_{n, v}} = \mathbf{x}_v$$
This Jacobian scales linearly with the magnitude of SentenceBERT initial node embeddings $\|\mathbf{x}_v\|_2 \approx 1.0$.

### Step 6: Pruner MLP Activation (`gnn.py:74`)
The scalar pruning weight is generated via:
```python
# gnn.py:74
x_weights = self.mlp_x(x_dist).sigmoid()
```
Differentiating with respect to the pre-activation logit $u_v = \text{MLP}_{\phi_1}(\mathbf{d}_v)$:
$$\frac{\partial \alpha_{n, v}}{\partial u_v} = \sigma'(u_v) = \sigma(u_v)(1 - \sigma(u_v)) \leq 0.25$$
This introduces a second `Sigmoid` contraction bottleneck of at most $0.25$.

---

## Part 3: The Compounding Attenuation Factor ($\kappa$)

Multiplying the theoretical upper bounds of these contraction coefficients yields the aggregate attenuation multiplier $\kappa$:

$$\kappa = \left(\frac{1}{T}\right) \times \left(\sup \sigma'_{\text{proj}}\right) \times \left(\frac{1}{|V_g|}\right) \times \left(\sup \sigma'_{\text{prune}}\right)$$

Substituting the empirical parameters of the GRAG configuration:
- Attention dilution ($T \approx 512$): $\approx 1.95 \times 10^{-3}$
- Projector Sigmoid: $0.25$
- Subgraph mean-pool ($|V_g| \approx 40$): $0.025$
- Pruner Sigmoid: $0.25$

$$\kappa \approx 0.00195 \times 0.25 \times 0.025 \times 0.25 \approx 3.05 \times 10^{-6}$$

### Effective Parameter Update

In [`config.py:12`](file:///mnt/c/Users/martinxz13/Desktop/Juan/Knowlege_Center/knwolege_master/100_Projects/GraphRAG_Thesis/github_repos/GRAG/src/config.py#L12), the training learning rate is configured as:
$$\eta = 10^{-5}$$

Assuming a normalized loss gradient at the language output $\left\| \nabla_{\mathbf{e}} \mathcal{L} \right\| \approx \mathcal{O}(1)$, the effective parameter update on the pruner weights $\phi_1$ is:
$$\Delta \phi_1 = -\eta \cdot \frac{\partial \mathcal{L}}{\partial \phi_1} \approx 10^{-5} \times \left(3.05 \times 10^{-6}\right) \approx \mathbf{3.05 \times 10^{-11}}$$

> [!CAUTION]
> **Diagnostic Finding (Gradient Starvation):**
> An effective weight update of $\sim 10^{-11}$ per step means that across 3 epochs of training (~5,000 steps), the cumulative update to $\phi_1$ cannot exceed $\sim 1.5 \times 10^{-7}$. 
> **The pruner weights $\phi_1$ are mathematically frozen at initialization.** They do not undergo meaningful semantic optimization through backpropagation.

---

## Part 4: Absence of Auxiliary Supervision (`train.py`)

An audit of [`train.py:44-65`](file:///mnt/c/Users/martinxz13/Desktop/Juan/Knowlege_Center/knwolege_master/100_Projects/GraphRAG_Thesis/github_repos/GRAG/src/train.py#L44-L65) reveals the optimizer objective:
```python
# train.py:48-52
loss = model(batch)
loss.backward()
optimizer.step()
```
Where `model.forward()` executes:
```python
# graph_llm.py:171-176
loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
return loss
```

### Critical Findings:
1. **$\mathcal{L}_{\text{aux}} = 0$:** There is **no** $L_1$ sparsity penalty ($\lambda \|\boldsymbol{\alpha}\|_1$), no entropy minimization objective, and no ground-truth retrieval cross-entropy loss applied to $\boldsymbol{\alpha}$.
2. **Behavior of $\alpha$:** Because $\phi_1$ is initialized with Xavier uniform (`gnn.py:50-52`) and receives an attenuated gradient of $\mathcal{O}(10^{-11})$, $\text{MLP}_{\phi_1}(\mathbf{d}_v) \approx 0 \implies \alpha_{n,v} \approx \sigma(0) = 0.5$.
3. **The Table 4 Ablation Paradox:**  
   The paper reports that removing soft pruning (*w/o Soft Pruning*) drops WebQSP Hit@1 from 0.7236 to 0.5671. If $\alpha$ does not learn, why does removing it cause a 15-point drop?
   - **Hypothesis A (Magnitude Scaling):** $\alpha \approx 0.5$ acts as a constant temperature dampener ($\frac{1}{2}\mathbf{X}$), preventing internal GAT attention logits from saturating.
   - **Hypothesis B (Inductive Noise Regularization):** The pseudo-random initial variance of $\alpha$ prevents the frozen LLM from over-relying on spurious node features during prefix conditioning.

---

## Part 5: Frontier Literature & Thesis Implications

The broader GraphRAG research community has actively recognized these exact message-passing failure modes:

1. **Flow Diffusion (QAFD-RAG, ICLR 2026):**  
   *Query-Aware Flow Diffusion for Graph-Based RAG* eliminates deep GNN stacks entirely. Instead of propagating gradients through $L$ message-passing layers and nonlinear pooling bottlenecks, QAFD-RAG formulates edge reweighting as a continuous flow-diffusion process guided directly by the query embedding. This completely circumvents the $10^{-6}$ attenuation factor.

2. **Continuous Entangled Sparse-Dense Signals (LeSeGR):**  
   In the LeSeGR propagation method (the core reference for your thesis), lexical and semantic retrieval signals are mapped continuously without passing through dual unregularized `Sigmoid` gates.

3. **Architectural Directives for ONTOGRAPH (LASIGE Thesis):**
   - **Eliminate Double-Sigmoid Gating:** Replace nested sigmoids with scaled dot-product attention or gated linear units (GLUs) with identity gradient paths.
   - **Explicit Auxiliary Loss:** Formulate a direct supervision signal $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{gen}} + \lambda \mathcal{L}_{\text{prune}}$ to ensure $\frac{\partial \mathcal{L}_{\text{prune}}}{\partial \phi_1} = \mathcal{O}(1)$.
   - **Topology-Preserving Pooling:** Abandon unweighted mean-pooling $\frac{1}{|V_g|}$; use attentive pooling conditioned on query semantics.

---

## Diagnostic Check for Block D Gating

To certify mastery of this lecture and close **D1**:
- State clearly why the upstream paper's claim that $\phi_1$ *"learns to identify relevant subgraphs through end-to-end task feedback"* is an empirical illusion under the default optimizer setup.
- Identify what single code modification in `train.py` would instantly restore $\left\|\frac{\partial \mathcal{L}}{\partial \phi_1}\right\| \approx \mathcal{O}(1)$.
