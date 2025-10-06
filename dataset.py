import os

import torch
import numpy as np
import pandas as pd
import scipy

import scanpy as sc
import scvelo as scv
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data

import pyVIA.datasets_via as datasets_via
import warnings
from sklearn.neighbors import NearestNeighbors

class SingleCellDataset:
    def __init__(self, adata, max_nodes):
        # Store the AnnData object and max_nodes
        self.adata = adata
        self.max_nodes = max_nodes
        
        # Convert data to PyG format and pad to max_nodes
        num_cells = min(adata.n_obs, self.max_nodes)
        
        # Pad or truncate features to max_nodes
        x_dense = adata.X.toarray()[:self.max_nodes]
        x_padded = np.zeros((self.max_nodes, x_dense.shape[1]))
        x_padded[:num_cells] = x_dense
        self.x = torch.FloatTensor(x_padded)
        
        # Pad or truncate adjacency matrix
        adj_dense = adata.obsp['connectivities'].todense()[:self.max_nodes, :self.max_nodes]
        adj_padded = np.zeros((self.max_nodes, self.max_nodes))
        adj_padded[:num_cells, :num_cells] = adj_dense
        self.adj = torch.FloatTensor(adj_padded)
        
        # Pad labels
        y_dense = pd.Categorical(adata.obs['clusters']).codes[:self.max_nodes]
        y_padded = np.zeros(self.max_nodes)
        y_padded[:num_cells] = y_dense
        self.y = torch.LongTensor(y_padded)
        
        # Create mask for valid cells
        self.mask = torch.zeros(self.max_nodes, dtype=torch.bool)
        self.mask[:num_cells] = True
        
        # Cache properties
        self._num_features = self.x.shape[1]
        self._num_classes = len(adata.obs['clusters'].unique())

    def __getitem__(self, idx):
        # Remove the extra dimension by not using unsqueeze(0)
        return Data(
            x=self.x,  # [max_nodes, num_features]
            adj=self.adj,  # [max_nodes, max_nodes]
            y=self.y,  # [max_nodes]
            mask=self.mask  # [max_nodes]
        )

    @property
    def num_features(self):
        return self._num_features
    
    @property
    def num_classes(self):
        return self._num_classes
    
    def __len__(self):
        return self.adata.n_obs
    

def preprocess_pancreas_data(file_path, subsample_frac=0.5, random_state=42, stratify_by='clusters'):
    """
    Preprocess the pancreas data for DiffPool model.
    
    Parameters:
    -----------
    file_path : str
        Path to the h5ad file containing pancreas data
    subsample_frac : float, default=0.5
        Fraction of data to keep in stratified sampling
    random_state : int, default=42
        Random state for reproducibility
    stratify_by : str, default='clusters'
        Column name in adata.obs to use for stratified sampling
        
    Returns:
    --------
    adata_subsampled : AnnData
        Preprocessed and subsampled AnnData object
    x : torch.Tensor
        Node features tensor
    adj : torch.Tensor
        Adjacency matrix tensor
    """
    # Load pancreas data
    adata = scv.datasets.pancreas(file_path=file_path)

    # # Calculate the number of observations to retain
    # n_obs_to_keep = int(adata.n_obs * subsample_frac)

    # Perform stratified sampling
    sampled_idx = adata.obs.groupby(stratify_by, group_keys=False).apply(
        lambda x: x.sample(frac=subsample_frac, random_state=random_state)
    ).index

    # Subset the AnnData object
    adata_subsampled = adata[sampled_idx].copy()

    # Filter and normalize
    scv.pp.filter_and_normalize(adata_subsampled, min_shared_counts=20, n_top_genes=2000)

    # Run PCA, compute neighbors and moments
    sc.tl.pca(adata_subsampled)
    # First run PCA and compute standard undirected KNN graph using scanpy
    sc.tl.pca(adata_subsampled)
    sc.pp.neighbors(adata_subsampled, n_neighbors=15)
    
    # Store the undirected KNN graph from scanpy
    adata_subsampled.obsp['scanpy_connectivities'] = adata_subsampled.obsp['connectivities'].copy()
    
    # Now compute directed KNN graph from original data
    X_original = adata_subsampled.X.toarray() if scipy.sparse.issparse(adata_subsampled.X) else adata_subsampled.X

    # Compute directed nearest neighbors
    n_neighbors = 15  # Default k value
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_original)
    distances, indices = nbrs.kneighbors(X_original)

    # Create directed connectivity matrix
    n_cells = adata_subsampled.n_obs
    rows = np.repeat(np.arange(n_cells), n_neighbors)
    cols = indices.flatten()
    data = np.ones_like(cols)

    # Remove self-loops
    mask = rows != cols
    rows = rows[mask]
    cols = cols[mask]
    data = data[mask]

    # Create directed connectivity matrix without self-loops
    adata_subsampled.obsp['directed_connectivities'] = scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(n_cells, n_cells)
    )
    
    # By default use the directed graph
    adata_subsampled.obsp['connectivities'] = adata_subsampled.obsp['directed_connectivities'].copy()
    # scv.pp.moments(adata_subsampled, n_pcs=None, n_neighbors=None)

    # Prepare input features (x) and adjacency matrix (adj) for DiffPool
    # Extract the original features as node features
    x = torch.tensor(adata_subsampled.X.toarray(), dtype=torch.float)

    # Extract the adjacency matrix
    adj = pyg_utils.to_dense_adj(
        pyg_utils.from_scipy_sparse_matrix(adata_subsampled.obsp['connectivities'])[0]
    ).squeeze(0)

    # Ensure the adjacency matrix is symmetric
    adj = (adj + adj.transpose(0, 1)) / 2

    # Convert adjacency matrix to float
    adj = adj.to(torch.float)

    return adata_subsampled, x, adj

def create_toy_multifurcating_data():
    """
    Create a toy multifurcating dataset using pyVIA.
    
    Returns:
    --------
    adata_toy : AnnData
        Preprocessed toy dataset
    x : torch.Tensor
        Node features tensor
    adj : torch.Tensor
        Adjacency matrix tensor
    """
    
    # Suppress warnings
    warnings.filterwarnings('ignore') 
    
    # Create toy dataset
    adata_toy = datasets_via.toy_multifurcating()
    print(f"Loaded toy multifurcating dataset with {adata_toy.n_obs} cells")
    
    # Store true labels
    true_label = adata_toy.obs['group_id'].tolist()
    
    # Process with scanpy
    ncomps = 30
    sc.tl.pca(adata_toy, svd_solver='arpack', n_comps=ncomps)
    sc.pp.neighbors(adata_toy, n_pcs=ncomps)
    sc.tl.umap(adata_toy)
    
    # Make sure clusters column exists for compatibility
    adata_toy.obs['clusters'] = adata_toy.obs['group_id'].astype('category')
    
    # Extract raw counts as node features instead of PCA
    x = torch.tensor(adata_toy.X.toarray() if scipy.sparse.issparse(adata_toy.X) else adata_toy.X, dtype=torch.float)
    
    # Extract the adjacency matrix
    adj = pyg_utils.to_dense_adj(
        pyg_utils.from_scipy_sparse_matrix(adata_toy.obsp['connectivities'])[0]
    ).squeeze(0)
    
    # Ensure the adjacency matrix is symmetric
    adj = (adj + adj.transpose(0, 1)) / 2
    
    # Convert adjacency matrix to float
    adj = adj.to(torch.float)
    
    return adata_toy, x, adj

def preprocess_bone_marrow_data(file_path):
    """
    Preprocess bone marrow data for analysis
    """
    print(f"Loading bone marrow dataset from {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    # Load the dataset
    adata = sc.read(file_path)
    
    # Apply preprocessing
    sc.pp.filter_genes(adata, min_counts=20)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    
    # Compute PCA and neighbors
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_pcs=10)
    
    # Use the highly variable genes for downstream analysis
    adata_hvg = adata[:, adata.var.highly_variable]
    
    # Convert to PyTorch tensors
    x = torch.FloatTensor(adata_hvg.X.toarray() if scipy.sparse.issparse(adata_hvg.X) else adata_hvg.X)
    
    # Get adjacency matrix from neighborhood graph
    if 'neighbors' in adata.uns:
        adj = torch.FloatTensor(adata.obsp['connectivities'].toarray())
    else:
        # If neighbors haven't been computed, create a simple adjacency
        adj = torch.eye(x.shape[0])
    
    print(f"Processed data: {x.shape[0]} cells, {x.shape[1]} genes")
    
    return adata_hvg, x, adj