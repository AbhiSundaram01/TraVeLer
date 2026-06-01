"""
Monocle Bone Marrow Evaluation
================================
Sweeps lambda_lap values with UMAP-based Laplacian regularisation.
Dataset: Setty et al. 2019 bone marrow (setty_bone_marrow.h5ad).
GNN learns to approximate UMAP embeddings; Monocle3 is run on GNN outputs
and compared (Spearman r) against control Monocle3 on UMAP embeddings.

Run from HPC:
    python evaluation.py
    TRAVELER_ROOT=/path/to/TraVeLer python evaluation.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging
import random

TRAVELER_ROOT = Path(os.environ.get("TRAVELER_ROOT",
                     str(Path(__file__).resolve().parents[2])))
DATA_FILE = Path(os.environ.get("TRAVELER_DATA",
                 str(TRAVELER_ROOT / "data"))) / "setty_bone_marrow.h5ad"
sys.path.insert(0, str(TRAVELER_ROOT))
sys.path.insert(0, str(TRAVELER_ROOT / 'py_monocle'))
os.environ.setdefault("DGL_GRAPHBOLT_DISABLE", "1")

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import torch_geometric.utils as pyg_utils
from py_monocle import learn_graph, order_cells

from models import DirectedDiffPool
from neural_k_forms.forms import NeuralOneForm
from neural_k_forms.chains import generate_integration_matrix
from graph_utils import get_all_edges


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

def setup_run(label: str = "monocle_bone_marrow"):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).parent / "runs" / f"{label}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Run: {run_id}  |  dir: {run_dir}")
    return run_id, run_dir, logger


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_and_preprocess():
    """
    Load Setty bone marrow dataset.
    Diffmap is computed for both root selection and DPT control.
    UMAP is recomputed as the GNN alignment target.
    """
    adata = sc.read(str(DATA_FILE))
    n = int(1 * adata.n_obs)
    np.random.seed(42)
    idx = np.random.choice(adata.n_obs, n, replace=False)
    adata_subsampled = adata[idx, :].copy()
    sc.pp.filter_genes(adata_subsampled, min_counts=20)
    sc.pp.normalize_total(adata_subsampled)
    sc.pp.log1p(adata_subsampled)
    sc.pp.highly_variable_genes(adata_subsampled)

    sc.tl.pca(adata_subsampled, random_state=42)
    sc.pp.neighbors(adata_subsampled, n_neighbors=50, n_pcs=10, random_state=42)
    sc.tl.diffmap(adata_subsampled, n_comps=10)
    sc.tl.umap(adata_subsampled, random_state=42)
    
    return adata_subsampled


def get_root(adata: sc.AnnData) -> int:
    """
    Root cell = argmin of diffusion component DC4 (index 3).
    This component reliably separates HSCs from committed progenitors.
    """
    return int(adata.obsm["X_diffmap"][:, 3].argmin())


def compute_monocle_control(adata: sc.AnnData, root: int):
    """Monocle3 pseudotime on UMAP embeddings (control baseline)."""
    umap = adata.obsm["X_umap"]
    sc.pp.neighbors(adata, n_neighbors=50, use_rep="X_umap")
    sc.tl.leiden(adata)
    leiden = adata.obs["leiden"].to_numpy(dtype=int)
    projected_points, mst, centroids = learn_graph(matrix=umap, clusters=leiden)
    pseudotime = order_cells(
        umap, centroids,
        mst=mst,
        projected_points=projected_points,
        root_cells=[root],
    )
    return pd.Series(pseudotime, index=adata.obs_names)


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


# ---------------------------------------------------------------------------
# Laplacian from UMAP k-NN
# ---------------------------------------------------------------------------

def build_umap_laplacian(X_umap: np.ndarray, n_neighbors: int = 20) -> torch.Tensor:
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_umap)
    _, indices = nbrs.kneighbors(X_umap)
    indices = indices[:, 1:]  # drop self
    n = len(indices)
    A = torch.zeros(n, n)
    for i in range(n):
        A[i, indices[i]] = 1.0
    A = torch.maximum(A, A.T)  # symmetrise
    degree = A.sum(dim=1)
    D_inv_sqrt = torch.diag(torch.pow(degree + 1e-8, -0.5))
    return torch.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt


# ---------------------------------------------------------------------------
# Loss terms
# ---------------------------------------------------------------------------

def ot_alignment_loss(X_target: np.ndarray, x_out: torch.Tensor,
                      P: torch.Tensor) -> torch.Tensor:
    n = X_target.shape[0]
    C = torch.cdist(
        torch.tensor(np.array(X_target), dtype=torch.float32), x_out, p=2
    ) ** 2
    return torch.sum(P * C) / n


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def setup_model(x: torch.Tensor, adj: torch.Tensor):
    set_seed()
    model = DirectedDiffPool(num_features=x.size(1), max_nodes=x.size(0))
    c = 1
    vf_in = nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 2, 128), nn.ReLU(),
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 2 * c),
    )
    vf = NeuralOneForm(vf_in, input_dim=2, hidden_dim=128, num_cochains=c)
    model.reset_parameters()
    vf.apply(vf._init_weights)
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": 1e-3, "weight_decay": 1e-3},
        {"params": vf.parameters(), "lr": 1e-3, "weight_decay": 1e-2},
    ])
    return model, vf, optimizer


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, vf, optimizer, x, adj, adata_run, X_umap, Laplacian,
          monocle_control, root, epochs, lambda_vf, lambda_lap, logger):
    losses, correlations = [], []
    X_gnn_final, gnn_pseudotime_final = None, None
    torch.autograd.set_detect_anomaly(True)

    for i in range(epochs):
        optimizer.zero_grad()

        x_out, adj_out = model(x, adj)
        chain, weights = get_all_edges(x_out, adj_out)

        X_int = generate_integration_matrix(vf, chain)
        X_weighted = X_int.squeeze() * weights.squeeze()
        L_vf = torch.mean(X_weighted, dim=0)

        P = model.cluster_matrix(x, adj)
        node_embeddings = P @ x_out

        L_emb = ot_alignment_loss(X_umap, x_out, P)
        L_lap = torch.trace(node_embeddings.T @ Laplacian @ node_embeddings)

        L = L_emb + lambda_lap * L_lap - lambda_vf * L_vf
        L.backward()

        vf_gnorm = sum(p.grad.norm().item() for p in vf.parameters() if p.grad is not None)
        mdl_gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

        if not (np.isfinite(vf_gnorm) and np.isfinite(mdl_gnorm)):
            logger.error(f"Epoch {i}: infinite gradients — stopping run.")
            break

        optimizer.step()
        losses.append(L.item())

        X_gnn = node_embeddings.detach().cpu().numpy()
        adata_run.obsm["X_gnn"] = X_gnn

        sc.pp.neighbors(adata_run, use_rep="X_gnn", n_neighbors=50)
        sc.tl.leiden(adata_run)
        leiden = adata_run.obs["leiden"].to_numpy(dtype=int)
        projected_points, mst, centroids = learn_graph(matrix=X_gnn, clusters=leiden)
        gnn_times = order_cells(
            X_gnn, centroids,
            mst=mst,
            projected_points=projected_points,
            root_cells=[root],
        )
        gnn_pseudotime = pd.Series(gnn_times)
        gnn_pseudotime.index = monocle_control.index

        try:
            combined = pd.concat([monocle_control, gnn_pseudotime], axis=1, join="inner")
            combined.columns = ["monocle", "gnn_time"]
            valid = combined.dropna()
            r, _ = spearmanr(valid["monocle"], valid["gnn_time"])
        except Exception:
            r = float("nan")
        correlations.append(r)

        X_gnn_final = X_gnn
        gnn_pseudotime_final = gnn_pseudotime

        if i % 10 == 0:
            logger.info(f"Epoch {i:3d}: loss={L.item():.4f}  r={r:.3f}  "
                        f"L_emb={L_emb.item():.4f}  L_lap={L_lap.item():.4f}")

    return losses, correlations, X_gnn_final, gnn_pseudotime_final


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plot(losses, correlations, lambda_lap, fig_dir: Path,
              title_prefix: str) -> Path:
    epochs_range = range(1, len(losses) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    ax1.plot(epochs_range, losses, color="tab:blue", linewidth=2.5,
             marker="o", markersize=3, label="Training Loss")
    ax2.plot(epochs_range, correlations, color="tab:red", linewidth=2.5,
             marker="x", markersize=4, label="Spearman r")

    ax1.set_xlabel("Epoch", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=16, color="tab:blue")
    ax2.set_ylabel("Spearman Correlation (Monocle)", fontsize=16, color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=13)
    ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=13)
    ax1.tick_params(axis="x", labelsize=13)

    plt.title(f"{title_prefix}\n$\\lambda_{{lap}}$ = {lambda_lap:.1e}", fontsize=18)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=13, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_path = fig_dir / f"loss_corr_llap_{lambda_lap:.0e}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def save_embedding_plots(X_gnn: np.ndarray, X_umap: np.ndarray,
                         gnn_pseudotime: pd.Series, monocle_control: pd.Series,
                         clusters: pd.Series, cluster_colors: list,
                         lambda_lap: float, fig_dir: Path,
                         title_prefix: str) -> Path:
    """
    Three-panel figure:
      left   — GNN embedding space coloured by cell type cluster
      centre — UMAP coloured by GNN pseudotime
      right  — UMAP coloured by Monocle3 control pseudotime (baseline)
    """
    from matplotlib.colors import ListedColormap

    gnn_pt = gnn_pseudotime.values.astype(float)
    ctrl_pt = monocle_control.values.astype(float)

    cat = clusters.astype("category")
    categories = cat.cat.categories
    cluster_codes = cat.cat.codes.values
    cmap = ListedColormap(cluster_colors)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(X_gnn[:, 0], X_gnn[:, 1], c=cluster_codes, s=1, cmap=cmap,
                    vmin=-0.5, vmax=len(categories) - 0.5)
    axes[0].set_title("GNN Embedding Space\n(cell type)", fontsize=20)
    axes[0].set_xlabel("Dim 1", fontsize=20)
    axes[0].set_ylabel("Dim 2", fontsize=20)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=cluster_colors[i],
                          markersize=5, label=cat)
               for i, cat in enumerate(categories)]
    axes[0].legend(handles=handles, fontsize=13, loc="best",
                   markerscale=1.5, framealpha=0.5)

    sc1 = axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=gnn_pt, s=1, cmap="plasma")
    axes[1].set_title("UMAP\n(GNN pseudotime)", fontsize=13)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(sc1, ax=axes[1], shrink=0.8)

    sc2 = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=ctrl_pt, s=1, cmap="plasma")
    axes[2].set_title("UMAP\n(Monocle3 control pseudotime)", fontsize=13)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.colorbar(sc2, ax=axes[2], shrink=0.8)

    fig.suptitle(f"{title_prefix}\n$\\lambda_{{lap}}$ = {lambda_lap:.1e}", fontsize=15)
    plt.tight_layout()

    save_path = fig_dir / f"embeddings_llap_{lambda_lap:.0e}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _, run_dir, logger = setup_run("monocle_bone_marrow")
    fig_dir = run_dir / "plots"
    fig_dir.mkdir(exist_ok=True)

    logger.info("Loading bone marrow data...")
    adata = load_and_preprocess()
    x, adj = get_initial_matrices(adata)

    X_umap = adata.obsm["X_umap"]
    Laplacian = build_umap_laplacian(X_umap, n_neighbors=50) #### note we use 50 neighbours for bone marrow, 20 for pancreas
    logger.info(f"UMAP Laplacian built: {Laplacian.shape}")

    root = get_root(adata)
    logger.info(f"Root cell index: {root}  DC4 value: {adata.obsm['X_diffmap'][root, 3]:.4f}")

    monocle_control = compute_monocle_control(adata, root)
    np.save(run_dir / "monocle_control.npy", monocle_control.values)
    logger.info("Control Monocle3 pseudotime computed and saved.")

    lambda_vf = 0.0
    lambda_laps = [0, 1, 100]
    epochs = 150

    for lambda_lap in lambda_laps:
        logger.info(f"\n{'='*50}")
        logger.info(f"lambda_lap = {lambda_lap}")
        adata_run = adata.copy()
        model, vf, optimizer = setup_model(x, adj)

        losses, correlations, X_gnn_final, gnn_pt_final = train(
            model, vf, optimizer, x, adj, adata_run,
            X_umap, Laplacian, monocle_control, root,
            epochs, lambda_vf, lambda_lap, logger,
        )

        p = save_plot(losses, correlations, lambda_lap, fig_dir,
                      "Monocle3 Bone Marrow Experiment with UMAP Laplacian")
        np.save(fig_dir / f"losses_llap_{lambda_lap:.0e}.npy", np.array(losses))
        np.save(fig_dir / f"corrs_llap_{lambda_lap:.0e}.npy", np.array(correlations))
        logger.info(f"Saved plot: {p}")

        if X_gnn_final is not None and gnn_pt_final is not None:
            ep = save_embedding_plots(
                X_gnn_final, X_umap, gnn_pt_final, monocle_control,
                adata.obs["clusters"], list(adata.uns["clusters_colors"]),
                lambda_lap, fig_dir,
                "Monocle3 Bone Marrow — Cell Embeddings",
            )
            logger.info(f"Saved embedding plot: {ep}")

    logger.info(f"\nAll runs complete. Results in: {run_dir}")


if __name__ == "__main__":
    main()
