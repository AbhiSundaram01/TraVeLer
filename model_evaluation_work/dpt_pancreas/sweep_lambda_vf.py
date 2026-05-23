"""
DPT Pancreas — Lambda VF Sweep
================================
Fixes lambda_lap (set from the lambda_lap sweep in evaluation.py) and sweeps
lambda_vf over LAMBDA_VFS. For each lambda_vf, trains over SEEDS random seeds
and records Spearman r between GNN DPT and control DPT at epochs 50, 100, 150.
Plots mean r vs lambda_vf (log scale) for each time point.

Run from HPC:
    python sweep_lambda_vf.py
    TRAVELER_ROOT=/path/to/TraVeLer python sweep_lambda_vf.py
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
# Sweep parameters — edit these based on lambda_lap sweep results
# ---------------------------------------------------------------------------

LAMBDA_LAP = 1        # fixed from lambda_lap sweep (evaluation.py)
LAMBDA_VFS = np.logspace(1.5, 4.0, 6)
SEEDS = [1, 2, 3, 4, 5]
EPOCHS = 150


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

def setup_run(label: str = "dpt_pancreas_vf_sweep"):
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
    n = int(1 * adata.n_obs)
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
    sc.tl.tsne(adata_subsampled)

    return adata_subsampled


def get_root(adata: sc.AnnData) -> int:
    umap = adata.obsm["X_umap"]
    mask = umap[:, 0] < umap[:, 0].mean()
    passed_indices = np.where(mask)[0]
    sub_argmax = umap[mask, 1].argmax()
    return int(passed_indices[sub_argmax])


def compute_dpt_control(adata: sc.AnnData, root: int) -> pd.Series:
    adata.uns["iroot"] = root
    sc.tl.dpt(adata)
    return adata.obs["dpt_pseudotime"].copy()


def get_initial_matrices(adata: sc.AnnData):
    n_neighbors = 15
    X = torch.FloatTensor(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, indices = nbrs.kneighbors(X)

    n_cells = adata.n_obs
    rows = np.repeat(np.arange(n_cells), n_neighbors)
    cols = indices.flatten()
    data = np.ones_like(cols)
    mask = rows != cols
    rows, cols, data = rows[mask], cols[mask], data[mask]

    adata.obsp['directed_connectivities'] = scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(n_cells, n_cells)
    )
    adata.obsp['connectivities'] = adata.obsp['directed_connectivities'].copy()

    x = torch.tensor(adata.X.toarray(), dtype=torch.float)
    adj = pyg_utils.to_dense_adj(
        pyg_utils.from_scipy_sparse_matrix(adata.obsp['connectivities'])[0]
    ).squeeze(0).to(torch.float)
    return x, adj


# ---------------------------------------------------------------------------
# Laplacian from PCA k-NN
# ---------------------------------------------------------------------------

def build_pca_laplacian(X_pca: np.ndarray, n_neighbors: int = 20) -> torch.Tensor:
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_pca)
    _, indices = nbrs.kneighbors(X_pca)
    indices = indices[:, 1:]
    n = len(indices)
    A = torch.zeros(n, n)
    for i in range(n):
        A[i, indices[i]] = 1.0
    A = torch.maximum(A, A.T)
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
# Model  (no set_seed call — outer loop controls seeding)
# ---------------------------------------------------------------------------

def setup_model(x: torch.Tensor, adj: torch.Tensor):
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
    vf = NeuralOneForm(vf_in, input_dim=10, hidden_dim=128, num_cochains=c)
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
    correlations = []
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

        optimizer.step()

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
            logger.info(f"  epoch {i:3d}: r={r:.3f}  L={L.item():.4f}")

    return correlations


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_correlation_vs_lambda_vf(lambda_vfs, correlation_mean, fig_dir: Path,
                                   title_prefix: str) -> None:
    labels = [
        ("early", 0, "Epoch 50"),
        ("mid",   1, "Epoch 100"),
        ("late",  2, "Epoch 150"),
    ]
    for name, idx, epoch_label in labels:
        plt.figure(figsize=(6.5, 5))
        plt.semilogx(
            lambda_vfs, correlation_mean[:, idx],
            marker="o", linewidth=2.5, markersize=7,
        )
        plt.xlabel(r"$\lambda_{\mathrm{vf}}$", fontsize=16)
        plt.ylabel("Spearman correlation", fontsize=16)
        plt.title(f"{title_prefix} ({epoch_label})", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(fig_dir / f"correlation_vs_lambda_vf_{name}.png", dpi=300)
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _, run_dir, logger = setup_run("dpt_pancreas_vf_sweep")
    fig_dir = run_dir / "plots"
    fig_dir.mkdir(exist_ok=True)
    plot_data_dir = run_dir / "plot_data"
    plot_data_dir.mkdir(exist_ok=True)

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

    logger.info(f"lambda_lap fixed at {LAMBDA_LAP}")
    logger.info(f"Sweeping lambda_vf: {LAMBDA_VFS}")
    logger.info(f"Seeds: {SEEDS}  |  Epochs: {EPOCHS}")

    correlation_sum = np.zeros((len(LAMBDA_VFS), 3))

    for seed in SEEDS:
        logger.info(f"\n{'='*50}\nSeed {seed}")
        set_seed(seed)
        for vf_idx, lambda_vf in enumerate(LAMBDA_VFS):
            logger.info(f"  lambda_vf = {lambda_vf:.3e}")
            adata_run = adata.copy()
            model, vf, optimizer = setup_model(x, adj)

            correlations = train(
                model, vf, optimizer, x, adj, adata_run,
                X_pca, Laplacian, dpt_control, root,
                EPOCHS, lambda_vf, LAMBDA_LAP, logger,
            )

            def safe_r(epoch_idx):
                return correlations[epoch_idx] if len(correlations) > epoch_idx else float("nan")

            correlation_sum[vf_idx, 0] += safe_r(49)
            correlation_sum[vf_idx, 1] += safe_r(99)
            correlation_sum[vf_idx, 2] += safe_r(149)

    correlation_mean = correlation_sum / len(SEEDS)

    np.savez(
        plot_data_dir / "correlation_vs_lambda_vf.npz",
        lambda_vf=LAMBDA_VFS,
        correlation_mean=correlation_mean,
        epochs=np.array([50, 100, 150]),
    )
    df = pd.DataFrame(correlation_mean, columns=["epoch_50", "epoch_100", "epoch_150"])
    df.insert(0, "lambda_vf", LAMBDA_VFS)
    df.to_csv(plot_data_dir / "correlation_vs_lambda_vf.csv", index=False)
    logger.info(f"Saved plot data to {plot_data_dir}")

    plot_correlation_vs_lambda_vf(
        LAMBDA_VFS, correlation_mean, fig_dir,
        f"DPT Pancreas ($\\lambda_{{lap}}={LAMBDA_LAP}$, {len(SEEDS)} seeds)",
    )
    logger.info(f"All runs complete. Results in: {run_dir}")


if __name__ == "__main__":
    main()
