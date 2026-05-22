"""
DPT Pancreas Evaluation
=======================
Sweeps lambda_lap values with PCA-based Laplacian regularisation.
GNN learns to approximate top-10 PCA embeddings; DPT is run on soft PCA embeddings (P @ X_pca)
and compared (Spearman r) against control DPT run on standard PCA embeddings.

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
                 str(TRAVELER_ROOT / "data"))) / "pancreas.h5ad"
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

def setup_run(label: str = "dpt_pancreas"):
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

def load_and_preprocess() -> sc.AnnData:
    adata = sc.read(str(DATA_FILE))

    # Use full dataset (no subsampling) - consistent with sw_pancreas.py
    sc.pp.filter_genes(adata, min_counts=20)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)

    sc.tl.pca(adata, random_state=42)
    sc.pp.neighbors(adata, n_neighbors=50, n_pcs=10, random_state=42)

    # Diffmap is required for DPT control
    sc.tl.diffmap(adata, n_comps=10)

    return adata


def get_root(adata: sc.AnnData) -> int:
    """
    Root cell from UMAP: left half (x < mean), cell with highest y.
    For pancreas this reliably selects a Ductal cell (most primitive progenitor).
    """
    umap = adata.obsm["X_umap"]
    mask = umap[:, 0] < umap[:, 0].mean()
    passed_indices = np.where(mask)[0]
    sub_argmax = umap[mask, 1].argmax()
    return int(passed_indices[sub_argmax])


def compute_dpt_control(adata: sc.AnnData, root: int) -> pd.Series:
    """Standard DPT on PCA/diffmap embeddings."""
    adata.uns["iroot"] = root
    sc.tl.dpt(adata)
    return adata.obs["dpt_pseudotime"].copy()


def get_initial_matrices(adata: sc.AnnData):
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    n_cells = adata.n_obs
    n_neighbors = 15

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    _, indices = nbrs.kneighbors(X)

    rows = np.repeat(np.arange(n_cells), n_neighbors)
    cols = indices.flatten()
    data = np.ones(len(rows), dtype=np.float32)
    mask = rows != cols
    rows, cols, data = rows[mask], cols[mask], data[mask]

    adata.obsp["connectivities"] = scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(n_cells, n_cells)
    )

    x = torch.tensor(X, dtype=torch.float)
    adj = pyg_utils.to_dense_adj(
        pyg_utils.from_scipy_sparse_matrix(adata.obsp["connectivities"])[0]
    ).squeeze(0).float()
    return x, adj


# ---------------------------------------------------------------------------
# Laplacian from PCA k-NN
# ---------------------------------------------------------------------------

def build_pca_laplacian(X_pca: np.ndarray, n_neighbors: int = 20) -> torch.Tensor:
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_pca)
    _, indices = nbrs.kneighbors(X_pca)
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
    vf = NeuralOneForm(vf_in, num_cochains=c)
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

def train(model, vf, optimizer, x, adj, adata_run, X_pca, Laplacian,
          dpt_control, root, epochs, lambda_vf, lambda_lap, logger):
    losses, correlations = [], []
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

        L_emb = ot_alignment_loss(X_pca, x_out, P)
        L_lap = torch.trace(node_embeddings.T @ Laplacian @ node_embeddings)

        L = L_emb + lambda_lap * L_lap - lambda_vf * L_vf
        L.backward()

        vf_gnorm = sum(p.grad.norm().item() for p in vf.parameters() if p.grad is not None)
        mdl_gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

        if not (np.isfinite(vf_gnorm) and np.isfinite(mdl_gnorm)):
            logger.error(f"Epoch {i}: infinite gradients — stopping run.")
            break

        torch.nn.utils.clip_grad_norm_(vf.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(L.item())

        # DPT on soft PCA embeddings (P @ X_pca), compare with control
        X_soft_pca = node_embeddings.detach().cpu().numpy()
        adata_run.obsm["X_gnn"] = X_soft_pca
        adata_run.uns["iroot"] = root
        sc.pp.neighbors(adata_run, use_rep="X_gnn", key_added="nbrs_gnn", n_neighbors=50)
        sc.tl.diffmap(adata_run, neighbors_key="nbrs_gnn",
                      n_comps=min(15, X_soft_pca.shape[1]))
        sc.tl.dpt(adata_run, neighbors_key="nbrs_gnn")
        gnn_dpt = adata_run.obs["dpt_pseudotime"].copy()

        try:
            combined = pd.concat([dpt_control, gnn_dpt], axis=1, join="inner")
            combined.columns = ["dpt", "gnn_dpt"]
            valid = combined.dropna()
            r, _ = spearmanr(valid["dpt"], valid["gnn_dpt"])
        except Exception:
            r = float("nan")
        correlations.append(r)

        if i % 10 == 0:
            logger.info(f"Epoch {i:3d}: loss={L.item():.4f}  r={r:.3f}  "
                        f"L_emb={L_emb.item():.4f}  L_lap={L_lap.item():.4f}")

    return losses, correlations


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plot(losses, correlations, lambda_lap, fig_dir: Path, title_prefix: str) -> None:
    epochs_range = range(1, len(losses) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    ax1.plot(epochs_range, losses, color="tab:blue", linewidth=2.5,
             marker="o", markersize=3, label="Training Loss")
    ax2.plot(epochs_range, correlations, color="tab:red", linewidth=2.5,
             marker="x", markersize=4, label="Spearman r")

    ax1.set_xlabel("Epoch", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=16, color="tab:blue")
    ax2.set_ylabel("Spearman Correlation (DPT)", fontsize=16, color="tab:red")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _, run_dir, logger = setup_run("dpt_pancreas")
    fig_dir = run_dir / "plots"
    fig_dir.mkdir(exist_ok=True)

    logger.info("Loading pancreas data...")
    adata = load_and_preprocess()
    x, adj = get_initial_matrices(adata)

    X_pca = adata.obsm["X_pca"][:, :10]
    Laplacian = build_pca_laplacian(X_pca, n_neighbors=20)
    logger.info(f"PCA Laplacian built: {Laplacian.shape}")

    root = get_root(adata)
    logger.info(f"Root cell index: {root}  "
                f"(cluster: {adata.obs['clusters'].iloc[root] if 'clusters' in adata.obs else 'N/A'})")

    dpt_control = compute_dpt_control(adata, root)
    np.save(run_dir / "dpt_control.npy", dpt_control.values)
    logger.info("Control DPT computed and saved.")

    lambda_vf = 0.0
    lambda_laps = [0, 0.01, 0.1, 1, 10, 100, 1000]
    epochs = 150

    for lambda_lap in lambda_laps:
        logger.info(f"\n{'='*50}")
        logger.info(f"lambda_lap = {lambda_lap}")
        adata_run = adata.copy()
        model, vf, optimizer = setup_model(x, adj)

        losses, correlations = train(
            model, vf, optimizer, x, adj, adata_run,
            X_pca, Laplacian, dpt_control, root,
            epochs, lambda_vf, lambda_lap, logger,
        )

        p = save_plot(losses, correlations, lambda_lap, fig_dir,
                      "DPT Pancreas — PCA Laplacian")
        np.save(fig_dir / f"losses_llap_{lambda_lap:.0e}.npy", np.array(losses))
        np.save(fig_dir / f"corrs_llap_{lambda_lap:.0e}.npy", np.array(correlations))
        logger.info(f"Saved plot: {p}")

    logger.info(f"\nAll runs complete. Results in: {run_dir}")


if __name__ == "__main__":
    main()
