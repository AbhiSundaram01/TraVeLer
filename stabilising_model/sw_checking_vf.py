import sys
import os

# Add the repo root (one level above stabilising_model) to Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from dataset import pre_process_bone_marrow_directed
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import torch
import scipy
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch_geometric.utils as pyg_utils
from main3 import setup_experiment
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os
from models import DiffPool, DirectedDiffPool
from neural_k_forms.forms import NeuralOneForm
from graph_utils import get_all_edges
from neural_k_forms.chains import generate_integration_matrix
import random
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp


FILE_NAME = "data/setty_bone_marrow.h5ad"

adata = sc.read(
    filename=FILE_NAME,
    backup_url="https://figshare.com/ndownloader/files/35826944",
)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_model(x, adj, logger):
    # set_seed()
    """Initialize models, vector field and optimizer"""
    model = DirectedDiffPool(num_features=x.size(1), max_nodes=x.size(0))
    
    # # Get vector field size based on initial model output
    # with torch.no_grad():
    #     temp_x_out, _ = model(x, adj)
    #     temp_chain = soft_mst_approximation2(temp_x_out, temperature=0.1)
    #     c = temp_chain.size(0)  # number of features/columns in cochain data matrix
    #     logger.info(f"Initializing vector field with {c} components based on sample chain")
    c = 1
    
    # Initialize neural vector field
    vf_in = nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 2, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2 * c)
    )
    
    vf = NeuralOneForm(vf_in, input_dim=10, hidden_dim=128, num_cochains=c)
    model.reset_parameters()
    vf.apply(vf._init_weights)
    
    # Create joint optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 0.001, 'weight_decay': 0.001},
        {'params': vf.parameters(), 'lr': 0.001, 'weight_decay': 0.01}
    ])
    
    return model, vf, optimizer, c

def preprocess_data(adata):
    n = int(0.5 * adata.n_obs)
    np.random.seed(42)
    idx = np.random.choice(adata.n_obs, n, replace=False)
    adata_subsampled = adata[idx, :].copy()
    sc.pp.filter_genes(adata_subsampled, min_counts=20)
    sc.pp.normalize_total(adata_subsampled)
    sc.pp.log1p(adata_subsampled)
    sc.pp.highly_variable_genes(adata_subsampled)

    sc.tl.pca(adata_subsampled)
    sc.pp.neighbors(adata_subsampled, n_neighbors=50, n_pcs=10)
    sc.tl.diffmap(adata_subsampled, n_comps=10)
    
    return adata_subsampled

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

def local_constraint(X_PCA, X_PCA_nbrs, node_embeddings):
    N, k = X_PCA_nbrs.shape
    X_PCA = torch.tensor(X_PCA, dtype=torch.float)

    # Gather neighbor coordinates in PCA space
    nbrs_pca = X_PCA[X_PCA_nbrs]          # shape: [N, k, d]
    nbrs_emb = node_embeddings[X_PCA_nbrs] # shape: [N, k, d']

    # Compute pairwise distances to neighbors
    # distances in PCA space
    dist_pca = torch.norm(X_PCA.unsqueeze(1) - nbrs_pca, dim=2)  # [N, k]
    # distances in embedding space
    dist_emb = torch.norm(node_embeddings.unsqueeze(1) - nbrs_emb, dim=2)  # [N, k]

    # Compute squared difference
    loss = torch.mean((dist_pca - dist_emb) ** 2)

    return loss

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



def train_model(model, vf, optimizer, x, adj, adata_subsampled, epochs, run_dir, logger, λ_vf, Laplacian, λ_lap):
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
        chain, weights = get_all_edges(x_out, adj_out)

        X = generate_integration_matrix(vf, chain)
        X_weighted = X.squeeze() * weights.squeeze()
        L_vf = torch.mean(X_weighted, dim=0)

        #Abhi
        P = model.cluster_matrix(x, adj)
        node_embeddings = P @ x_out

        L_emb = OT_graph_similarity(X_PCA, x_out, P) 
        L_local = local_constraint(X_PCA, X_PCA_nbrs, node_embeddings)
        probs = P.clamp(min=1e-12)
        entropies = -(probs * torch.log(probs)).sum(dim=1)
        L_ent = entropies.mean()
        L_laplacian = torch.trace(node_embeddings.T @ Laplacian @ node_embeddings) 

        # if i > 99:
        #     λ_lap = 100

        L = L_emb + λ_lap * L_laplacian - λ_vf * L_vf
        print(L)
        print(L_vf)
        print(L_ent)
        print(L_laplacian)
        # Compute gradients
        L.backward()
        
        vf_grad_norm = sum(p.grad.norm().item() for p in vf.parameters() if p.grad is not None)
        model_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

        grad_norms_vf.append(vf_grad_norm)
        grad_norms_model.append(model_grad_norm)

        # Update parameters
        optimizer.step()
        
        # Store metrics
        losses.append(L.item())

        X_gnn = node_embeddings.detach().cpu().numpy()
        adata.obsm['X_gnn'] = X_gnn
        adata.uns['iroot'] = root      # same root as before
        sc.pp.neighbors(adata, use_rep='X_gnn', key_added='neighbors_gnn', n_neighbors=50)
        sc.tl.diffmap(adata, neighbors_key='neighbors_gnn', n_comps=min(15, X_gnn.shape[1]))
        sc.tl.dpt(adata, neighbors_key='neighbors_gnn')
        GNN_dpt = adata.obs['dpt_pseudotime'].copy()


        combined = pd.concat([dpt, GNN_dpt], axis=1, join='inner')
        combined.columns = ['dpt', 'gnn_dpt']
        r, p_value = spearmanr(combined['dpt'], combined['gnn_dpt'])
        print(f"Spearman correlation: {r:.3f}, p-value: {p_value:.3e}")
        correlations.append(r)

    return losses, node_embeddings, correlations, grad_norms_vf, grad_norms_model



λ_vfs = np.logspace(1.5,4.0, 6)
λ_lap = 1
adata = preprocess_data(adata)
x, adj = get_initial_matrices(adata)
X_PCA = adata.obsm['X_pca'][:, :10] 
nbrs = NearestNeighbors(n_neighbors=51).fit(X_PCA)
_, X_PCA_nbrs = nbrs.kneighbors(X_PCA)  # returns distances and indices
X_PCA_nbrs = X_PCA_nbrs[:, 1:]          # drop self (first column)
Laplacian = get_laplacian(X_PCA, X_PCA_nbrs)
dpt, root,X_diffmap = diffusion_pseudotime(adata)
run_id, run_dir, logger = setup_experiment()
# Setup model, vector field and optimizer
adata_run = adata.copy()
# Directory to save figures
fig_dir = os.path.join(run_dir, "lambda_vf_plots")
os.makedirs(fig_dir, exist_ok=True)
epochs = 150
seeds = [1,2,3,4,5]
correlation_sum = np.zeros((6,3))
for seed in seeds:
    set_seed(seed)
    correlations = np.zeros((6,3))
    for idx, λ_vf in enumerate(λ_vfs):
        # Re-initialize model for each λ_vf
        model, vf, optimizer, c = setup_model(x, adj, logger)
        epochs = 150    
        
        # Train
        losses, node_embeddings, correlation_trace, grad_norms_vf, grad_norms_model = train_model(
            model, vf, optimizer, x, adj, adata_run, epochs, run_dir, logger, λ_vf, Laplacian, λ_lap
        )
        correlations[idx][0] = correlation_trace[49]
        correlations[idx][1] = correlation_trace[99]
        correlations[idx][2] = correlation_trace[149]
    correlation_sum += correlations
correlation_mean = correlation_sum / 5


# -------------------------------
# Save data used for plotting
# -------------------------------
plot_data_dir = os.path.join(run_dir, "plot_data")
os.makedirs(plot_data_dir, exist_ok=True)

# Save as NumPy (exact reproduction)
np.savez(
    os.path.join(plot_data_dir, "correlation_vs_lambda_vf.npz"),
    lambda_vf=λ_vfs,
    correlation_mean=correlation_mean,
    epochs=np.array([50, 100, 150])
)

# Save as CSV (human-readable, easy to replot)
df_plot = pd.DataFrame(
    correlation_mean,
    columns=["epoch_50", "epoch_100", "epoch_150"]
)
df_plot.insert(0, "lambda_vf", λ_vfs)

df_plot.to_csv(
    os.path.join(plot_data_dir, "correlation_vs_lambda_vf.csv"),
    index=False
)

print(f"Saved plot data to {plot_data_dir}")


# -------------------------------
# Plot settings (global)
# -------------------------------
LABEL_FONTSIZE = 16
TITLE_FONTSIZE = 18
TICK_FONTSIZE = 14
LINEWIDTH = 2.5
MARKERSIZE = 7

labels = [
    ('early', 0, 'Epoch 50'),
    ('mid',   1, 'Epoch 100'),
    ('late',  2, 'Epoch 150'),
]

for name, idx, title in labels:
    plt.figure(figsize=(6.5, 5))
    
    plt.semilogx(
        λ_vfs,
        correlation_mean[:, idx],
        marker='o',
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE
    )

    plt.xlabel(r'$\lambda_{\mathrm{vf}}$', fontsize=LABEL_FONTSIZE)
    plt.ylabel('Spearman correlation', fontsize=LABEL_FONTSIZE)
    plt.title(
        f'Correlation vs $\\lambda_{{vf}}$, subsampled f = 0.5, ({title})',
        fontsize=TITLE_FONTSIZE
    )

    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)

    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f'correlation_vs_lambda_vf_{name}.png'),
        dpi=300
    )
    plt.close()



    # # Create figure
    # plt.figure(figsize=(8, 6))
    
    # # Plot loss
    # plt.plot(range(1, epochs+1), losses, label='Training Loss', color='blue', marker='o')
    
    # # Plot correlation on secondary y-axis
    # ax1 = plt.gca()
    # ax2 = ax1.twinx()
    # ax2.plot(range(1, epochs+1), correlations, label='Spearman Correlation', color='orange', marker='x')
    
    # # Labels
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Loss', color='blue')
    # ax2.set_ylabel('Spearman Correlation', color='orange')

    # # Vertical line at epoch 100 (λ_lap set back to 0)
    # # ax1.axvline(x=100, color='gray', linestyle='--', linewidth=1, label='λ_lap → 0')
    
    # # Title
    # plt.title(f'Training Loss and Correlation over Epochs\nλ_vf = {λ_vf:.1e}')
    
    # # Legends
    # lines_1, labels_1 = ax1.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    # # Grid and layout
    # ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    # plt.tight_layout()
    
    # # Save figure
    # save_path = os.path.join(fig_dir, f'loss_corr_lambda_vf_{λ_vf:.0e}.png')
    # plt.savefig(save_path, dpi=300)
    # plt.close()
    
    # print(f"Saved plot for λ_vf={λ_vf:.1e} at {save_path}")

    #     # ---- Gradient norm plot ----
    # plt.figure(figsize=(8, 6))

    # plt.plot(
    #     range(1, epochs + 1),
    #     grad_norms_model,
    #     label='Model gradient norm',
    #     marker='o'
    # )

    # plt.plot(
    #     range(1, epochs + 1),
    #     grad_norms_vf,
    #     label='Vector field gradient norm',
    #     marker='x'
    # )

    # # Vertical line at epoch 100
    # plt.axvline(x=100, color='gray', linestyle='--', linewidth=1, label='λ_vf → 0')

    # plt.yscale('log')  # IMPORTANT: gradients span orders of magnitude

    # plt.xlabel('Epoch')
    # plt.ylabel('Gradient norm (log scale)')
    # plt.title(f'Gradient norms over training\nλ_vf = {λ_vf:.1e}')
    # plt.legend()
    # plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # grad_save_path = os.path.join(
    #     fig_dir, f'grad_norms_lambda_vf_{λ_vf:.0e}.png'
    # )
    # plt.tight_layout()
    # plt.savefig(grad_save_path, dpi=300)
    # plt.close()

    # print(f"Saved gradient norm plot for λ_vf={λ_vf:.1e} at {grad_save_path}")




# # Convert λ_vfs to a NumPy array in case it isn't already
# λ_vfs = np.array(λ_vfs)

# # Create figure
# plt.figure(figsize=(8, 6))

# # Plot Spearman correlation vs λ_vf
# plt.plot(λ_vfs, correlations, marker='o', linestyle='-', color='blue')

# # Logarithmic x-axis
# plt.xscale('log')

# # Labels and title
# plt.xlabel(r'$\lambda_{vf}$ (vector field weight)')
# plt.ylabel('Spearman Correlation')
# plt.title('Effect of λ_vf on Pseudotime Alignment')

# # Grid for readability
# plt.grid(True, which='both', linestyle='--', alpha=0.5)

# # Save the figure
# save_path = os.path.join(run_dir, 'correlation_vs_lambda_vf.png')
# plt.savefig(save_path, dpi=300)
# plt.close()

# print(f"Correlation vs λ_vf plot saved to {save_path}")


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