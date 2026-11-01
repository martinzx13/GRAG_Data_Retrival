# Lecture 01: Foundations of PyTorch and PyTorch Geometric (PyG)
**Target Project:** ONTOGRAPH / GRAG Replication  
**Theoretical Focus:** Systems architecture, memory layouts, computational graphs, and sparse parallel reduction on GPU  

 Now place all sources in Row 0 and all targets in Row 1:

  Now place all sources in Row 0 and all targets in Row 1:



  Row 0 contains sources [0, 2, 3]. Row 1 contains targets [1, 4, 4].
---

## 1. PyTorch Core: The Systems Anatomy of a Tensor

A `torch.Tensor` is not a primitive multidimensional array. In PyTorch's underlying C++ engine (`ATen`), a tensor is decoupled into two architectural structures:

```
                  TensorImpl (Metadata Object)
               ┌──────────────────────────┐
               │ - sizes:   [2, 3]        │
               │ - strides: [3, 1]        │
               │ - offset:  0             │
               │ - dtype:   float32       │
               │ - storage_ptr ────────┐  │
               └───────────────────────┼──┘
                                       │
                                       ▼
                             Storage (Raw Buffer)
               ┌──────────────────────────────────────────────┐
               │ [ x0 | x1 | x2 | x3 | x4 | x5 ]              │
               │ contiguous void* on host or CUDA device      │
               └──────────────────────────────────────────────┘
```

### 1.1 Pointer Arithmetic via Strides
Given an $n$-dimensional index $(i_0, i_1, \dots, i_{n-1})$, the physical memory address offset in the flat storage buffer is computed as:

$$\text{Physical Offset} = \text{storage\_offset} + \sum_{k=0}^{n-1} i_k \cdot \text{stride}[k]$$

* **Zero-Copy Views:** Transposing (`tensor.t()`), reshaping (`view()`), or slicing creates a new `TensorImpl` referencing the identical underlying `Storage` buffer. Only the metadata (`sizes`, `strides`, `storage_offset`) is modified.
* **The Contiguity Invariant:** A tensor is contiguous in memory if and only if:
  $$\text{stride}[k] = \prod_{j=k+1}^{n-1} \text{size}[j]$$
  Transposing a 2D matrix flips strides from `[cols, 1]` to `[1, cols]`. Operations demanding contiguous memory (e.g., standard GEMM BLAS routines or specific CUDA kernels) require an explicit `.contiguous()`, triggering a fresh memory allocation and memory copy.

### 1.2 The Autograd Engine: Dynamic DAG of Vector-Jacobian Products (VJPs)
Standard reverse-mode automatic differentiation evaluates composite derivatives via the chain rule:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot J_f(g(x)) \cdot J_g(x)$$

PyTorch does not instantiate full Jacobian matrices $J \in \mathbb{R}^{m \times n}$ because their storage complexity $\mathcal{O}(m \cdot n)$ would exceed VRAM limits on deep networks. Instead, PyTorch executes the **Vector-Jacobian Product (VJP)**:

$$\mathbf{v}^T J$$

* During the forward pass, any tensor operation with `requires_grad=True` dynamically constructs a directed acyclic graph (DAG) of C++ `torch::autograd::Node` objects.
* Each node records pointers to inputs, saves intermediate activations via `ctx->save_for_backward()`, and defines the reverse gradient transformation.
* Calling `.backward()` initiates a reverse topological sort across this dynamic DAG, passing upstream gradient vectors through specialized CUDA kernels to populate the `.grad` buffers of leaf parameters.

---

## 2. Why Dense Tensors Fail on Graphs: The Sparse Necessity

In the WebQSP benchmark used by GRAG:
* Average nodes per graph: $|V| \approx 1370$
* Average edges per graph: $|E| \approx 4252$

### 2.1 The Dense Adjacency Catastrophe
A dense adjacency matrix representation requires:

$$\mathbf{A} \in \mathbb{R}^{|V| \times |V|} = 1370 \times 1370 \approx 1.88 \times 10^6 \text{ elements}$$

Graph density $\rho$ is calculated as:

$$\rho = \frac{|E|}{|V|(|V|-1)} \approx \frac{4252}{1370 \times 1369} \approx 0.00227 \quad (\mathbf{99.77\% \text{ zeros}})$$

Allocating dense matrices for a batch of subgraphs wastes over $99.7\%$ of GPU memory on zeros, and dense matrix multiplication $\mathbf{A}\mathbf{X}$ scales as $\mathcal{O}(|V|^2 \cdot d_{\text{feat}})$.

### 2.2 Sparse Coordinate (COO) Format
PyG bypasses dense representation by using Coordinate (COO) format:

$$\mathbf{E}_{\text{idx}} \in \mathbb{N}_0^{2 \times |E|}$$

Represented as a contiguous 2D integer tensor (`torch.int64`):
* `edge_index[0]`: Array of source indices $u \in V$.
* `edge_index[1]`: Array of target indices $v \in V$.

```
Source (u): [ 0,  1,  1,  2,  3,  ... ]
Target (v): [ 1,  0,  2,  3,  0,  ... ]
```

This enforces $\mathcal{O}(|E|)$ space complexity and aligns with edge-parallel CUDA thread scheduling.

---

## 3. PyG Data Representation & Disjoint Union Batching

A textual graph $G = (V, E, \{T_n\}, \{T_e\})$ maps to a `torch_geometric.data.Data` object:

```python
from torch_geometric.data import Data

data = Data(
    x=x,                      # Node feature tensor: [|V|, d_node]
    edge_index=edge_index,    # Graph topology:      [2, |E|] (COO int64)
    edge_attr=edge_attr,      # Edge feature tensor: [|E|, d_edge]
    y=y                       # Target labels
)
```

### 3.1 Disjoint Union Batching Mechanism
Standard deep learning frameworks pad variable sequences to $L_{\max}$, yielding $[B, L_{\max}, D]$. Zero-padding variable-sized graphs wastes significant VRAM and induces compute divergence on CUDA warps.

PyG instead leverages the principle that the **disjoint union of multiple graphs is itself a single disconnected graph**:


$$ \text{edge\_index}_{\text{batch}} = \begin{bmatrix} \underbrace{0}_{\text{Graph A}} & \underbrace{2}_{\text{Graph B}} & \underbrace{3}_{\text{Graph B}} \\ 1 & 4 & 4 \end{bmatrix} \in \mathbb{N}_0^{2 \times 3} $$

Given a batch of $B$ graphs $\{G_1, G_2, \dots, G_B\}$:
1. **Node features are concatenated vertically:**
   $$\mathbf{X}_{\text{batch}} = \begin{bmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \\ \vdots \\ \mathbf{X}_B \end{bmatrix} \in \mathbb{R}^{\left(\sum |V_i|\right) \times d}$$
2. **The adjacency matrix forms a block-diagonal structure:**
   $$\mathbf{A}_{\text{batch}} = \begin{bmatrix} \mathbf{A}_1 & \mathbf{0} & \dots & \mathbf{0} \\ \mathbf{0} & \mathbf{A}_2 & \dots & \mathbf{0} \\ \vdots & \vdots & \ddots & \vdots \\ \mathbf{0} & \mathbf{0} & \dots & \mathbf{A}_B \end{bmatrix}$$
3. **COO indices are offset cumulatively:**
   $$\text{offset}_k = \sum_{j=1}^{k-1} |V_j|$$
   $$\mathbf{E}_{\text{batch}} = \left[ \, \mathbf{E}_1, \; \mathbf{E}_2 + \text{offset}_2, \; \dots, \; \mathbf{E}_B + \text{offset}_B \, \right]$$
4. **An auxiliary vector `batch`:**
   $$\text{batch} \in \{0, \dots, B-1\}^{\sum |V_i|}$$
   maps every individual node back to its parent graph index for subsequent global pooling operations (`global_mean_pool`, `global_add_pool`).

---

## 4. Message Passing & Low-Level GPU Reduction Kernels

Message Passing Neural Networks (MPNN) proceed through three phases:

1. **Message Generation (Edge-Parallel):**
   $$\mathbf{m}_{uv}^{(l)} = \text{MSG}^{(l)}\left(\mathbf{h}_u^{(l-1)}, \, \mathbf{h}_v^{(l-1)}, \, \mathbf{e}_{uv}\right), \quad \forall (u, v) \in E$$
2. **Neighborhood Aggregation (Reduction):**
   $$\mathbf{m}_v^{(l)} = \bigoplus_{u \in \mathcal{N}(v)} \mathbf{m}_{uv}^{(l)}, \quad \bigoplus \in \{\sum, \text{mean}, \max\}$$
3. **Node Update (Node-Parallel):**
   $$\mathbf{h}_v^{(l)} = \text{UPDATE}^{(l)}\left(\mathbf{h}_v^{(l-1)}, \, \mathbf{m}_v^{(l)}\right)$$

### 4.1 CUDA Reduction & Write Hazards (`torch_scatter`)
In Phase 2 (Aggregation), multiple incoming edges terminate at the same target node $v$. If CUDA threads processing different edges write into the same output location `out[v]` concurrently, a **write collision (data race)** occurs.

Native Python loops over neighborhoods introduce catastrophic kernel launch overhead. `torch_scatter` resolves this by deploying optimized CUDA kernels executing parallel atomic instructions (`atomicAdd`) or radix-sorted segment reductions:

```cpp
// Conceptual atomic reduction kernel in CUDA
__global__ void scatter_add_kernel(const float* msg, const int64_t* target_nodes, float* out, int num_edges, int dim) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_idx < num_edges) {
        int v = target_nodes[edge_idx];
        for (int d = 0; d < dim; ++d) {
            atomicAdd(&out[v * dim + d], msg[edge_idx * dim + d]);
        }
    }
}
```

This forms the concrete computational foundation for GRAG's soft-pruned message passing:

$$\mathbf{m}_u^{(l)} = \text{MSG}^{(l)}\left(\alpha_u \cdot \mathbf{h}_u^{(l-1)}, \, \alpha_{uv} \cdot \mathbf{e}_{uv}\right)$$

---

## 5. Diagnostic Gating Examination Questions

Answer these three questions before advancing to code execution.

### Question P1: Memory Footprint Derivation
For a graph with $|V| = 2000$ nodes and an average node degree of $d = 4$ (directed, meaning $|E| = 8000$ edges):
1. Compute the exact memory in bytes required to store topology as a dense adjacency matrix using `float32`.
2. Compute the exact memory in bytes required to store topology in PyG's `edge_index` COO format using `int64`.
3. State the exact memory reduction factor.

### Question P2: Disjoint Union Index Offset
Given:
* Graph 1: 3 nodes ($\{0, 1, 2\}$) with edges `[[0, 1], [1, 2]]` (directed: $0 \to 1, 1 \to 2$).
* Graph 2: 4 nodes ($\{0, 1, 2, 3\}$) with edges `[[0, 1], [2, 3], [3, 0]]`.

Provide the exact numerical tensors for:
1. The batched `edge_index` tensor (shape $[2, 5]$).
2. The `batch` indicator tensor (shape $[7]$).

### Question P3: Hardware Hazard in GPU Scatter
During the aggregation step $\mathbf{m}_v = \sum_{u \in \mathcal{N}(v)} \mathbf{m}_{uv}$, why does naive parallel edge execution (`out[edge_index[1][i]] += msg[i]`) fail on a GPU Streaming Multiprocessor (SM) without atomic operations or sorting? Explain the read-modify-write hazard at the hardware register/cache level.
