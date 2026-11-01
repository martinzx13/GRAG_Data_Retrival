# Lecture 01 Deep-Dive: The Autograd Engine, Dynamic DAGs, and Vector-Jacobian Products (VJPs)

**Target Project:** ONTOGRAPH / GRAG Replication  
**Domain:** Deep Learning Systems, Automatic Differentiation, Mathematical Foundations  
**Reference Document:** `lecture_01_pytorch_pyg.md` (§1.2)

---

## 1. Problem Formalization: What is the Jacobian Matrix $J$?

Let a computational module or layer be formalized as a vector-valued function:

$$f: \mathbb{R}^n \to \mathbb{R}^m$$

taking an input tensor $\mathbf{x} \in \mathbb{R}^n$ and producing an output tensor $\mathbf{y} = f(\mathbf{x}) \in \mathbb{R}^m$.

The **Jacobian matrix** $J_f(\mathbf{x}) \in \mathbb{R}^{m \times n}$ is the complete matrix containing all first-order partial derivatives of each output component with respect to each input component:

$$J_f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \dots & \frac{\partial y_1}{\partial x_n} \\
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \dots & \frac{\partial y_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial x_1} & \frac{\partial y_m}{\partial x_2} & \dots & \frac{\partial y_m}{\partial x_n}
\end{bmatrix}, \quad \text{where } (J_f)_{ij} = \frac{\partial y_i}{\partial x_j}$$

### The Scalar Loss Invariant
In deep learning, neural networks terminate in an objective function that produces a single scalar loss:

$$\mathcal{L}: \mathbb{R}^m \to \mathbb{R}$$

By the multivariate chain rule, the gradient of the scalar loss $\mathcal{L}$ with respect to the intermediate input $\mathbf{x}$ is:

$$\underbrace{\frac{\partial \mathcal{L}}{\partial \mathbf{x}}}_{1 \times n} = \underbrace{\frac{\partial \mathcal{L}}{\partial \mathbf{y}}}_{1 \times m} \cdot \underbrace{J_f(\mathbf{x})}_{m \times n}$$

Let the incoming upstream gradient row-vector be denoted by $\mathbf{v}^T = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \in \mathbb{R}^{1 \times m}$. The target computation is strictly:

$$\mathbf{v}^T J_f(\mathbf{x})$$

---

## 2. The Systems Catastrophe: Why Explicit Jacobians Cause Hardware Failure

Attempting to evaluate $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ by first materializing $J_f(\mathbf{x}) \in \mathbb{R}^{m \times n}$ and subsequently performing a vector-matrix multiplication creates an immediate asymptotic memory catastrophe.

### Concrete Proof: Single Transformer Layer or Graph Feature Projection
Consider a standard hidden representation in a 7B parameter language model (such as LLaMA-2) or a graph node feature embedding tensor:
* Hidden feature dimension: $d = 4096$.
* Sequence length or graph node count: $N = 2048$.
* Total elements per activation tensor:
  $$k = N \times d = 2048 \times 4096 = 8,388,608 \text{ elements}$$

If a system materializes the full Jacobian $J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ between two such consecutive representations:
* Number of matrix elements in $J$:
  $$k \times k = (8.388 \times 10^6) \times (8.388 \times 10^6) \approx 7.037 \times 10^{13} \text{ elements}$$
* Assuming single-precision IEEE 754 floats (`float32`, 4 bytes per element):
  $$\text{Memory Required} = 7.037 \times 10^{13} \times 4 \text{ bytes} \approx 281.5 \times 10^{12} \text{ bytes} \approx \mathbf{281.5 \text{ Terabytes (TB)}}$$

Given a hardware budget of **6 GB (6,144 MiB)** on an NVIDIA RTX 3060 Laptop GPU, materializing the Jacobian for a single forward operation exceeds the physical VRAM envelope by a factor of over **46,000×**.

---

## 3. The Algorithmic Resolution: The Vector-Jacobian Product (VJP)

The mathematical resolution rests on a fundamental realization:
> **An optimization algorithm never requires the individual decoupled entries $\frac{\partial y_i}{\partial x_j}$ of the transformation. It only requires the accumulated directional derivative $\mathbf{v}^T J_f(\mathbf{x})$ projecting the scalar loss gradient backwards.**

Instead of computing $J$ and then computing $\mathbf{v}^T J$, reverse-mode automatic differentiation implements an analytical, closed-form function called the **Vector-Jacobian Product (VJP)**:

$$\text{VJP}_f(\mathbf{x}, \mathbf{v}): (\mathbb{R}^n, \mathbb{R}^m) \to \mathbb{R}^n$$

This function directly maps the incoming upstream gradient $\mathbf{v} \in \mathbb{R}^m$ back to the parameter or input space $\mathbb{R}^n$ without ever allocating the $m \times n$ matrix in memory.

### Mathematical Examples of Closed-Form VJPs

#### Example A: Element-wise Non-linear Activation (ReLU)
Let $\mathbf{y} = \text{ReLU}(\mathbf{x}) = \max(0, \mathbf{x})$.
* **Explicit Jacobian:** An $n \times n$ diagonal matrix:
  $$J_{ij} = \begin{cases} \mathbb{I}(x_i > 0) & \text{if } i = j \\ 0 & \text{if } i \ne j \end{cases}$$
  Allocating this matrix requires $\mathcal{O}(n^2)$ space.
* **VJP Implementation:**
  $$(\mathbf{v}^T J)_i = v_i \cdot \mathbb{I}(x_i > 0)$$
  Allocating space: $\mathcal{O}(n)$ (simply an element-wise Hadamard product between incoming gradient $\mathbf{v}$ and a boolean bitmask).

#### Example B: Linear Transformation (Matrix Multiplication)
Let $\mathbf{Y} = \mathbf{X}\mathbf{W}$, where $\mathbf{X} \in \mathbb{R}^{B \times d_{\text{in}}}$, $\mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$, and $\mathbf{Y} \in \mathbb{R}^{B \times d_{\text{out}}}$.
Let the incoming gradient of scalar loss $\mathcal{L}$ with respect to $\mathbf{Y}$ be denoted as:
$$\mathbf{V} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \in \mathbb{R}^{B \times d_{\text{out}}}$$

Through matrix calculus, the VJPs for the input activations $\mathbf{X}$ and the weight matrix $\mathbf{W}$ are:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \mathbf{V} \mathbf{W}^T \quad \left[\text{shape: } B \times d_{\text{in}}\right]$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^T \mathbf{V} \quad \left[\text{shape: } d_{\text{in}} \times d_{\text{out}}\right]$$

The memory complexity remains $\mathcal{O}(B \cdot d_{\text{in}} + d_{\text{in}} \cdot d_{\text{out}})$, maintaining absolute sub-quadratic spatial bounds.

---

## 4. Systems Execution: The Dynamic DAG in PyTorch C++ Engine

In PyTorch, the Vector-Jacobian Product machinery is coordinated through a dynamic Directed Acyclic Graph (DAG) managed by the `torch::autograd` engine.

```
================================================================================
FORWARD PASS (Dynamic DAG construction & activation caching):
================================================================================

 Tensor X (requires_grad=True)
      │
      ▼
 ┌──────────────────────────────────────────────┐
 │ Node: mm / AddmmBackward0                    │
 │ -------------------------------------------- │
 │ - ctx->saved_tensors: [X, W]                 │ <── Saved for backward
 │ - next_edges: [X.grad_fn, W.grad_fn]         │
 └──────────────────────────────────────────────┘
      │ produces Tensor Y
      ▼
 ┌──────────────────────────────────────────────┐
 │ Node: SoftPruningGatingBackward              │
 │ -------------------------------------------- │
 │ - ctx->saved_tensors: [α, h]                 │ <── Saved for backward
 │ - next_edges: [Y.grad_fn, α.grad_fn]         │
 └──────────────────────────────────────────────┘
      │ produces Tensor Z
      ▼
 ... ──► Scalar Loss ℒ


================================================================================
BACKWARD PASS (Reverse topological sweep via VJPs):
================================================================================

 ℒ.backward() seeds upstream gradient: v = [ 1.0 ]
      │
      ▼
 ┌──────────────────────────────────────────────┐
 │ Execute VJP Kernel: SoftPruningGating        │
 │ Computes: v_Y = VJP(v_Z, saved_α, saved_h)   │
 └──────────────────────────────────────────────┘
      │ sends v_Y
      ▼
 ┌──────────────────────────────────────────────┐
 │ Execute VJP Kernel: mmBackward               │
 │ Computes:                                    │
 │   v_X = v_Y · W^T                            │
 │   v_W = X^T · v_Y                            │ ──► Accumulates W.grad += v_W
 └──────────────────────────────────────────────┘
      │ sends v_X
      ▼
 Leaves populated without ever storing a Jacobian matrix.
```

### Key Systems Primitives
1. **`torch::autograd::Node`:** C++ object instantiated dynamically during the forward pass for every operation performed on tensors with `requires_grad=True`.
2. **`ctx->save_for_backward()`:** Allocates reference pointers to input or intermediate activations required during the reverse pass.
3. **Activation Memory Footprint:** The total VRAM consumption during training is dominated by these cached activations, **not** by model parameter weights alone.
4. **`backward()` Engine:** Performs a reverse topological sort on the active DAG, invoking the `apply()` method of each `Node` using CUDA kernels executing the respective VJPs.

---

## 5. Architectural Relevance to ONTOGRAPH and GRAG

Understanding VJPs and dynamic DAG execution is strictly mandatory for resolving two critical bottlenecks in your thesis:

1. **Gradient Flow Across the Dense-Sparse Boundary (Question D1):**
   In GRAG and ONTOGRAPH, soft pruning scales both text embeddings and message-passing features:
   $$\mathbf{m}_u = \text{MSG}\left(\alpha_u \mathbf{h}_u, \, \alpha_{uv} \mathbf{e}_{uv}\right)$$
   $$\alpha = \text{MLP}_{\phi_1}(\mathbf{z}_n \ominus \mathbf{z}_q)$$
   To train the MLP parameters $\phi_1, \phi_2$ end-to-end from generation loss $\mathcal{L}_{\text{QA}}$, the dynamic autograd DAG must evaluate the VJP through the custom CUDA scatter aggregation kernels (`scatter_add` / `torch_scatter`). If the message-passing kernel does not provide an exact VJP, gradients cannot backpropagate to $\phi_1, \phi_2$.

2. **Hardware Constraints on 6GB VRAM:**
   Because intermediate activations are retained via `ctx->save_for_backward()` across both the GNN layers and the LLaMA-2 transformer backbone, understanding the memory overhead of the dynamic DAG is what allows you to prevent CUDA Out-Of-Memory (OOM) crashes via gradient checkpointing and activation pruning.
