import sys
from pathlib import Path
import os

from py_monocle import (
    learn_graph,
    order_cells,
    compute_cell_states,
    regression_analysis,
    differential_expression_genes,
)


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
import matplotlib.patches as mpatches

X = scipy.io.mmread("../data/GSE216999_Mancuso2022_RNAcounts.mtx").tocsr()

cells = pd.read_csv(
    "../data/GSE216999_Mancuso2022_RNAcounts_cellbarcodes.txt",
    header=None
)
cells.columns = ["cell"]
genes = pd.read_csv(
    "../data/GSE216999_Mancuso2022_RNAcounts_genes.txt",
    header=None
)
genes.columns = ["gene"]

cells = cells.iloc[1:]  # remove first row

# If the number is in the same string, split by space
cells["cell"] = cells["cell"].astype(str).str.split().str[-1]
# Remove quotes if present
cells["cell"] = cells["cell"].str.replace('"', '', regex=False)

genes = genes.iloc[1:]

adata = sc.AnnData(X.T)
adata.obs_names = cells["cell"].values
adata.var_names = genes["gene"].values

metadata = pd.read_csv("../data/GSE216999_Mancuso2022_metadata.csv")
# Select metadata rows that match the AnnData cells
metadata_matched = metadata[metadata["cell_id"].isin(adata.obs_names)].copy()
# Now assign to adata.obs safely
adata.obs = metadata_matched.set_index("cell_id").loc[adata.obs_names].copy()

pathology = adata[adata.obs["mouse.genotype"] == "APP-NLGF"]

sc.pp.filter_genes(pathology, min_cells=3)
sc.pp.normalize_total(pathology)
sc.pp.log1p(pathology)
sc.pp.highly_variable_genes(pathology)
pathology = pathology[:, pathology.var.highly_variable]
sc.tl.pca(pathology)
sc.pp.neighbors(pathology, n_neighbors=50, n_pcs=10)

states = pathology.obs["cell.state_v2.1"].astype("category").reset_index(drop=True)
umap_0 = pathology.obs["umap_1"].to_numpy()
umap_1 = pathology.obs["umap_2"].to_numpy()

states = states.astype("category")
categories = states.cat.categories.tolist()
cmap = plt.cm.get_cmap("tab10", len(categories))
color_dict = {cat: cmap(i) for i, cat in enumerate(categories)}

colors = np.array([color_dict[s] for s in states])

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(umap_0, umap_1, c=colors, s=1)

patches = [mpatches.Patch(color=color_dict[cat], label=cat) for cat in categories]
ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig("cell_states_path.png", dpi=300, bbox_inches='tight')
plt.close()

###

root = umap_0.argmax()
umap = np.column_stack([umap_0, umap_1])
pathology.obsm["X_umap"] = umap

sc.pp.neighbors(pathology, n_neighbors=50, use_rep='X_umap')
sc.tl.leiden(pathology)
leiden = pathology.obs["leiden"].to_numpy(dtype=int)
projected_points, mst, centroids = learn_graph(matrix=umap, clusters=leiden)
pseudotime = order_cells(
    umap, centroids,
    mst=mst,
    projected_points=projected_points,
    root_cells=[root],
)
plt.figure(1, (10, 6))
plt.title("Pseudotime")
plt.scatter(umap[:, 0], umap[:, 1], c=pseudotime, s=1, cmap="plasma")
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.savefig("pseudotime_path.png", dpi=300, bbox_inches='tight')
plt.close()

wildtype = adata[adata.obs["mouse.genotype"] == "APP-WT"]

sc.pp.filter_genes(wildtype, min_cells=3)
sc.pp.normalize_total(wildtype)
sc.pp.log1p(wildtype)
sc.pp.highly_variable_genes(wildtype)
wildtype = wildtype[:, wildtype.var.highly_variable]
sc.tl.pca(wildtype)
sc.pp.neighbors(wildtype, n_neighbors=50, n_pcs=10)

states = wildtype.obs["cell.state_v2.1"].astype("category").reset_index(drop=True)
umap_0 = wildtype.obs["umap_1"].to_numpy()
umap_1 = wildtype.obs["umap_2"].to_numpy()

states = states.astype("category")
categories = states.cat.categories.tolist()
cmap = plt.cm.get_cmap("tab10", len(categories))
color_dict = {cat: cmap(i) for i, cat in enumerate(categories)}

colors = np.array([color_dict[s] for s in states])

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(umap_0, umap_1, c=colors, s=1)

patches = [mpatches.Patch(color=color_dict[cat], label=cat) for cat in categories]
ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig("cell_states_wt.png", dpi=300, bbox_inches='tight')
plt.close()

###

root = umap_0.argmax()
umap = np.column_stack([umap_0, umap_1])
wildtype.obsm["X_umap"] = umap

sc.pp.neighbors(wildtype, n_neighbors=50, use_rep='X_umap')
sc.tl.leiden(wildtype)
leiden = wildtype.obs["leiden"].to_numpy(dtype=int)
projected_points, mst, centroids = learn_graph(matrix=umap, clusters=leiden)
pseudotime = order_cells(
    umap, centroids,
    mst=mst,
    projected_points=projected_points,
    root_cells=[root],
)
plt.figure(1, (10, 6))
plt.title("Pseudotime")
plt.scatter(umap[:, 0], umap[:, 1], c=pseudotime, s=1, cmap="plasma")
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.savefig("pseudotime_wt.png", dpi=300, bbox_inches='tight')
plt.close()