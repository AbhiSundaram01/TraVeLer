# 🧠 TraVeLer: Trajectory and Vector Field Learning

TraVeLer is a project that jointly trains two neural networks:
- A **Vector Field Network** (`vf`) that models directional dynamics over the data manifold.
- A **Graph Coarsening Network** (`model`) that extracts a low-resolution, interpretable structure (a chain) from the input data.

These models are connected by a **differentiable integration operator** `generate_integration_matrix`, which allows gradients to flow across both models jointly during training.

---

## 📊 Data Format

Let `x` be the input data (e.g. gene expression):
- Shape: `n × d` (cells × features)

A **k-NN graph** `adj` is constructed from `x`:
- Shape: `n × n`

The `model` coarsens this graph into a **chain**:
- Shape: `(n−1) × 2 × d'`  
  - ❗ **Only a chain** (not a general graph) can be used as input to `generate_integration_matrix`.
  - Note: During coarsening, the number of nodes (`n`) and features (`d`) will not necessarily match the original input data (`x`).

---

## 🔗 Key Functions

### `generate_integration_matrix(vf, chain)` → `X`
- Integrates the vector field `vf` along the input `chain`
- `vf` and `chain` must **agree on the ambient dimension**, i.e., `vf.input_features` must match `chain.shape[-1]`
- Returns `X`:

### Construction of k-NN Graph (`adj`)**

The adjacency matrix (`adj`) is built from the input data (`x`):

- Shape: `n × n`
    
- Considerations:
    
    1. What value should `k` be?
        
    2. Should the graph be **directed or undirected**?
        
    3. Instead of directly constructing a k-NN graph, **should we first build a simplicial complex and then extract its 1-skeleton**?

- **Note:** The direction of edges is crucial, as `generate_integration_matrix` is **maximized when the direction of `chain` aligns with the vector field’s direction**.

---

## 🧱 Model Overview

### Vector Field Network (`vf`)
- Learns a differentiable vector field on a manifold
- Input size: can be `d` (original features) or `2`/`3` (e.g., UMAP or layout coords)
- Output: `input_dim × c`, where `c` is number of vector field components
    - **The number of components (`c`) is a critical hyperparameter**: It can range from **1** to **the number of edges in the coarsened graph**.

### Graph Coarsening Network (`model`)
- Input: `x`, `adj`
- Output: `chain_coarsened`
- A key initialization parameter is `cluster_ratio`.
    
    - **Should `cluster_ratio` be learnable?** _(To be explored.)_
        
- The output of `model` can take multiple formats:
    
    1. **Coarsened adjacency matrix** (`adj_out`)
        
    2. **Minimum spanning tree** of `adj_out` (`adj_out_mst`), computed via:  
        `adj_out_mst = mst(adj_out)`
        
        - `mst()` **must be differentiable**.
            
    3. **Chain representation** of `adj_out` or `adj_out_mst`.

> 📌 **Design decision:** may convert to `chain` either inside `model` or in the training script.

---

## 🧠 Training Objective

The total loss combines **structure preservation** and **flow alignment**:

```python
# Original chain from input graph
chain_origin = convert_to_chain(adj, x) # Fixed during training`
X_origin = generate_integration_matrix(vf, chain_origin)

# Coarsened chain from model
chain_coarsened = model(x, adj)
X_coarsened = generate_integration_matrix(vf, chain_coarsened)

# Loss terms
L_emb = torch.norm(X_origin - X_coarsened)  # structure preservation
L_align = -X_coarsened               # alignment

L = L_emb + lambda * L_align

```

## 📐 Dimensional Compatibility

To ensure `vf` is compatible with both chains:
- Use the same representation for both `chain_origin` and `chain_coarsened` (e.g., layout coordinates)

In the future:
- Introduce a learnable diffeomorphism `ϕ` to map:
  - From high-dimensional gene space → low-dimensional layout: `vf_low = vf ∘ ϕ`
  - From layout → gene space: `vf_high = vf ∘ ϕ⁻¹`

This would allow `vf` to be trained in a low-dimensional space while remaining applicable to the high-dimensional gene space through invertible transformation.

---

## ✅ Implementation Checklist

### Data & Preprocessing
- [ ] Load `x`, build k-NN graph `adj`
- [ ] Decide `k`, directed/undirected, embedding dimension

### Neural Networks
- [ ] Implement `vf`
- [ ] Implement `model`
- [ ] Decide where `chain` conversion happens

### Core Operators
- [ ] `generate_integration_matrix(vf, chain)`
- [ ] `convert_to_chain(adj, x)` (Should different output formats (adjacency matrix, MST, chain) be part of the `model` class or handled separately in training?)
- [ ] Differentiable `mst()` operator

### Training
- [ ] Forward pass: compute `X_origin`, `X_coarsened`
- [ ] Loss: `L_emb`, `L_align`, `L`
- [ ] Backpropagation + update

### Future Work
- [ ] Make `cluster_ratio` learnable
- [ ] Add `ϕ` to bridge dimensions
- [ ] Use 1-skeleton of a simplicial complex for `adj`

---

## 📎 Notes

- The key innovation is joint training of a vector field and a graph coarsening network, mediated by a differentiable integration operation over edge chains.
- This allows interpretable vector field dynamics to guide and be guided by data-driven coarse graph structure.
- Differentiable MST and integration over chains make this pipeline end-to-end trainable.

---

## 🔧 Dependencies

- `PyTorch`
- `NetworkX`
- `scikit-learn`
- `torch-geometric`, `torch-cluster`
