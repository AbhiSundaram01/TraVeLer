from dataset import pre_process_bone_marrow_directed
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import torch
import scipy
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch_geometric.utils as pyg_utils
from main3 import setup_experiment, setup_model
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os


FILE_NAME = "../data/setty_bone_marrow.h5ad"

adata = sc.read(
    filename=FILE_NAME,
    backup_url="https://figshare.com/ndownloader/files/35826944",
)

def preprocess_data(adata):
    sc.pp.filter_genes(adata, min_counts=20)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)

    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=10)
    sc.tl.diffmap(adata, n_comps=10)
    
    return adata

def diffusion_pseudotime(adata):
    X_diffmap = adata.obsm["X_diffmap"]
    # Setting root cell as described above
    root_ixs = adata.obsm["X_diffmap"][:, 3].argmin()
    adata.uns["iroot"] = root_ixs
    sc.tl.dpt(adata)
    adata.obs["dpt"] = adata.obs["dpt_pseudotime"].copy()
    return adata.obs["dpt"], root_ixs, X_diffmap
    
def get_initial_matrices(adata):
    X = torch.FloatTensor(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X)

    # Compute directed nearest neighbors
    n_neighbors = 15  # Default k value
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Create directed connectivity matrix
    n_cells = adata.n_obs
    rows = np.repeat(np.arange(n_cells), n_neighbors)
    cols = indices.flatten()
    data = np.ones_like(cols)

    # Remove self-loops
    mask = rows != cols
    rows = rows[mask]
    cols = cols[mask]
    data = data[mask]

    # Create directed connectivity matrix without self-loops
    adata.obsp['directed_connectivities'] = scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(n_cells, n_cells)
    )

    # By default use the directed graph
    adata.obsp['connectivities'] = adata.obsp['directed_connectivities'].copy()
    # scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    # Prepare input features (x) and adjacency matrix (adj) for DiffPool
    # Extract the original features as node features
    x = torch.tensor(adata.X.toarray(), dtype=torch.float)

    # Extract the adjacency matrix
    adj = pyg_utils.to_dense_adj(
        pyg_utils.from_scipy_sparse_matrix(adata.obsp['connectivities'])[0]
    ).squeeze(0)

    # Ensure the adjacency matrix is symmetric
    # adj = (adj + adj.transpose(0, 1)) / 2

    # Convert adjacency matrix to float
    adj = adj.to(torch.float)
    return x, adj

def OT_graph_similarity(X_PCA, x_out, P):
    """
    Numerically stable Sinkhorn alignment loss between coarsened nodes
    and target cluster centers.

    Parameters
    ----------
    x_diffmap : torch.Tensor [N, d]
        Diffusion Dimensionality Reduction
    node_embeddings : torch.Tensor [N, d]
        Original node embeddings after coarsening 


    Returns
    -------
    loss : torch.Tensor (scalar)
        Differentiable alignment loss
   
    """

    N = x_out.shape[0]
    n = X_PCA.shape[0]
    # 1. Compute pairwise squared distances (cost matrix)
    C = torch.cdist(torch.tensor(np.array(X_PCA).copy(), dtype=torch.float32), x_out, p=2) ** 2
    # 2. Compute alignment loss
    
    return torch.sum(P*C) / n

def get_laplacian(X_PCA, X_PCA_nbrs):
    n = len(X_PCA_nbrs)
    A = torch.zeros(n, n)
    for i in range(n):
        A[i, X_PCA_nbrs[i]] = 1.0
    A = torch.maximum(A, A.T) #symmetrise laplacian

    degree = A.sum(dim=1)
    D_inv_sqrt = torch.diag(torch.pow(degree + 1e-8, -0.5))

    L = torch.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    return L

def train_model(model, vf, optimizer, x, adj, adata_subsampled, epochs, run_dir, logger, Laplacian):
    """Core training loop"""
    log_interval = 10
    
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Tracking metrics
    losses = []
    correlations = []
    grad_norms_vf = []
    grad_norms_model = []
    x_sums = []
    

    # Training loop
    for i in range(epochs):
        # Clear all gradients
        optimizer.zero_grad()
        
        # Forward pass through DiffPool model
        x_out, adj_out = model(x, adj)
        

        #Abhi
        P = model.cluster_matrix(x, adj)
        node_embeddings = P @ x_out

        L_emb = OT_graph_similarity(X_PCA, x_out, P) 
        L_laplacian = torch.trace(node_embeddings.T @ Laplacian @ node_embeddings) 
        L = L_emb + L_laplacian
        print(L)
        # Compute gradients
        L.backward()
        
        # Update parameters
        optimizer.step()
        
        # Store metrics
        losses.append(L.item())

        X_gnn = node_embeddings.detach().cpu().numpy()
        adata.obsm['X_gnn'] = X_gnn
        adata.uns['iroot'] = root      # same root as before
        sc.pp.neighbors(adata, use_rep='X_gnn', key_added='neighbors_gnn', n_neighbors=15)
        sc.tl.diffmap(adata, neighbors_key='neighbors_gnn', n_comps=min(15, X_gnn.shape[1]))
        sc.tl.dpt(adata, neighbors_key='neighbors_gnn')
        GNN_dpt = adata.obs['dpt_pseudotime'].copy()


        combined = pd.concat([dpt, GNN_dpt], axis=1, join='inner')
        combined.columns = ['dpt', 'gnn_dpt']
        r, p_value = spearmanr(combined['dpt'], combined['gnn_dpt'])
        print(f"Spearman correlation: {r:.3f}, p-value: {p_value:.3e}")
        correlations.append(r)

    return losses, node_embeddings, correlations

adata = preprocess_data(adata)
x, adj = get_initial_matrices(adata)
X_PCA = adata.obsm['X_pca'][:, :10] 
nbrs = NearestNeighbors(n_neighbors=16).fit(X_PCA)
_, X_PCA_nbrs = nbrs.kneighbors(X_PCA)  # returns distances and indices
X_PCA_nbrs = X_PCA_nbrs[:, 1:]          # drop self (first column)
Laplacian = get_laplacian(X_PCA, X_PCA_nbrs)
dpt, root,X_diffmap = diffusion_pseudotime(adata)
run_id, run_dir, logger = setup_experiment()
# Setup model, vector field and optimizer
model, vf, optimizer, c = setup_model(x, adj, logger)
epochs = 100
losses, node_embeddings, correlations = train_model(model, vf, optimizer, x, adj, adata, epochs, run_dir, logger, Laplacian)

# Create a figure
plt.figure(figsize=(8, 6))

# Plot loss
plt.plot(range(1, epochs + 1), losses, label='Training Loss', color='blue', marker='o')

# Plot correlation (on secondary y-axis)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(range(1, epochs + 1), correlations, label='Spearman Correlation', color='orange', marker='x')

# Labels
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='blue')
ax2.set_ylabel('Spearman Correlation', color='orange')

# Title
plt.title('GNN Training: Loss and Pseudotime Correlation (neighbours = 15)')

# Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

# Layout adjustment
plt.tight_layout()

# Save the figure to disk (HPC-friendly)
save_dir = run_dir  # or any path you want
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'loss_correlation_plot.png')
plt.savefig(save_path, dpi=300)
plt.close()  # Close figure to avoid memory issues on HPC

print(f"Plot saved to {save_path}")


# X_gnn = node_embeddings.detach().cpu().numpy()
# adata.obsm['X_gnn'] = X_gnn
# adata.uns['iroot'] = root      # same root as before
# sc.pp.neighbors(adata, use_rep='X_gnn', key_added='neighbors_gnn', n_neighbors=15)
# sc.tl.diffmap(adata, neighbors_key='neighbors_gnn', n_comps=min(15, X_gnn.shape[1]))
# sc.tl.dpt(adata, neighbors_key='neighbors_gnn')
# GNN_dpt = adata.obs['dpt_pseudotime'].copy()


# combined = pd.concat([dpt, GNN_dpt], axis=1, join='inner')
# combined.columns = ['dpt', 'gnn_dpt']
# r, p_value = spearmanr(combined['dpt'], combined['gnn_dpt'])
# print(f"Spearman correlation: {r:.3f}, p-value: {p_value:.3e}")