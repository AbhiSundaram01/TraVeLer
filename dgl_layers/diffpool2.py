# %%
import os.path as osp
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool
# %%
max_nodes = 150

class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes


dataset = TUDataset('data', name='PROTEINS', transform=T.ToDense(max_nodes),
                    pre_filter=MyFilter())
dataset = dataset.shuffle()
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DenseDataLoader(test_dataset, batch_size=32)
val_loader = DenseDataLoader(val_dataset, batch_size=32)
train_loader = DenseDataLoader(train_dataset, batch_size=32)
# %%
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))


    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
            x = x.transpose(1, 2)  # Moves features to middle dimension [32, 64, 150]
            x = self.bns[step](x)
            x = x.transpose(1, 2)  # Restores original shape [32, 150, 64]
        

        return x


class DiffPool(torch.nn.Module):
    def __init__(self, sparse_threshold=0.1):  # Add threshold parameter
        super(DiffPool, self).__init__()
        self.sparse_threshold = sparse_threshold

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(dataset.num_features, 64, num_nodes)
        self.gnn1_embed = GNN(dataset.num_features, 64, 64)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(64, 64, num_nodes)
        self.gnn2_embed = GNN(64, 64, 64, lin=False)

        self.gnn3_embed = GNN(64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(64, 64)
        self.lin2 = torch.nn.Linear(64, dataset.num_classes)

    def sparsify_adj(self, adj):
        """Sparsify adjacency matrix by thresholding"""
        adj = torch.where(adj > self.sparse_threshold, adj, torch.zeros_like(adj))
        eye = torch.eye(adj.size(1), device=adj.device).unsqueeze(0).expand(adj.size(0), -1, -1)
        adj = adj * (1 - eye)  # Remove self-loops
        return adj

    def forward(self, x, adj, mask=None):
        # Add debugging prints
        print(f"Input shape: {x.shape}")
        
        # First level coarsening
        s = self.gnn1_pool(x, adj, mask)
        print(f"s_01 shape: {s.shape}")
        
        x = self.gnn1_embed(x, adj, mask)
        
        # First coarsened graph with sparsification
        x_1, adj_1, l1, e1 = dense_diff_pool(x, adj, s, mask)
        print(f"After first pooling: {x_1.shape}")
        adj_1 = self.sparsify_adj(adj_1)
        
        # Second level coarsening
        s = self.gnn2_pool(x_1, adj_1)
        print(f"s_12 shape: {s.shape}")
        
        x = self.gnn2_embed(x_1, adj_1)
        
        # Second coarsened graph with sparsification
        x_2, adj_2, l2, e2 = dense_diff_pool(x, adj_1, s)
        adj_2 = self.sparsify_adj(adj_2)
        
        # Store the hierarchical representations
        coarsened_graphs = {
            'level_0': {
                'x': x,        # Original node features
                'adj': adj,    # Original adjacency matrix
                'mask': mask   # Original mask
            },
            'level_1': {
                'x': x_1,      # First coarsened node features
                'adj': adj_1,  # First coarsened adjacency matrix
                's': s         # Assignment matrix for level 1
            },
            'level_2': {
                'x': x_2,      # Second coarsened node features
                'adj': adj_2,  # Second coarsened adjacency matrix
                's': s         # Assignment matrix for level 2
            }
        }
        
        # Continue with the original forward pass
        x = self.gnn3_embed(x_2, adj_2)
        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2, coarsened_graphs
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DiffPool().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _, _, _ = model(data.x, data.adj, data.mask)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.adj, data.mask)[0].max(dim=1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


@torch.no_grad()
def extract_coarsened_graphs(loader):
    model.eval()
    
    # Initialize storage for all batches
    all_coarsened = {
        'level_0': {'x': [], 'adj': [], 'mask': []},
        'level_1': {'x': [], 'adj': [], 's': []},
        'level_2': {'x': [], 'adj': [], 's': []}
    }
    all_labels = []
    
    for data in loader:
        data = data.to(device)
        # Forward pass with coarsened graph extraction
        _, _, _, coarsened = model(data.x, data.adj, data.mask)
        
        # Store each level's tensors
        for level in ['level_0', 'level_1', 'level_2']:
            for key in coarsened[level].keys():
                all_coarsened[level][key].append(coarsened[level][key].cpu())
        
        all_labels.append(data.y.cpu())
    
    # Concatenate all batches
    for level in all_coarsened:
        for key in all_coarsened[level]:
            all_coarsened[level][key] = torch.cat(all_coarsened[level][key], dim=0)
    
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_coarsened, all_labels



best_val_acc = test_acc = 0
for epoch in range(1, 10):
    train_loss = train(epoch)
    val_acc = test(val_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader)
        best_val_acc = val_acc
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
          f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    
# Usage example:
coarsened_graphs, labels = extract_coarsened_graphs(test_loader)
# %%
# draw the graph using the original graphs and the extracted coarsened graphs
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import dense_to_sparse

def draw_graph(adj, x, mask, s=None, threshold=0):  # Add threshold parameter
    """Draw graph with edge weight visualization"""
    G = nx.Graph()
    
    # Debug adjacency matrix
    print(f"Adj matrix stats: min={adj.min():.4f}, max={adj.max():.4f}")
    print(f"Number of edges above threshold: {(adj > threshold).sum().item()}")
    
    # Add nodes
    for i in range(x.size(0)):
        if mask is None or mask[i]:
            G.add_node(i, feature=x[i].mean().item())  # Use mean feature for color
    
    # Add edges with weights above threshold
    for i in range(adj.size(0)):
        for j in range(adj.size(1)):
            if adj[i,j] > threshold and (mask is None or (mask[i] and mask[j])):
                G.add_edge(i, j, weight=adj[i,j].item())
    
    # Draw
    plt.figure(figsize=(10,10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, 
            node_color=[G.nodes[node]['feature'] for node in G.nodes()],
            node_size=500,
            cmap=plt.cm.viridis,
            with_labels=True,
            width=2)
    plt.show()

# Draw the original graph
draw_graph(coarsened_graphs['level_0']['adj'][0], coarsened_graphs['level_0']['x'][0],
           coarsened_graphs['level_0']['mask'][0])

# Draw the first coarsened graph
draw_graph(coarsened_graphs['level_1']['adj'][0], coarsened_graphs['level_1']['x'][0],
           None, coarsened_graphs['level_1']['s'][0])

# Draw the second coarsened graph
draw_graph(coarsened_graphs['level_2']['adj'][0], coarsened_graphs['level_2']['x'][0],
           None, coarsened_graphs['level_2']['s'][0])
# %%
# draw the PROTEINS graph from dataset
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def draw_protein_graph(data_idx=0.75):
    """
    Draw a single graph from PROTEINS dataset
    Args:
        data_idx: index of the graph to visualize
    """
    # Get single graph data
    data = dataset[data_idx]
    
    # Create networkx graph
    G = nx.from_numpy_array(data.adj.numpy())
    
    # Get node features and prepare colors
    node_features = data.x.numpy()
    node_colors = np.mean(node_features, axis=1)  # Average feature values for coloring
    
    # Only include nodes that are not masked
    mask = data.mask.numpy()
    G = G.subgraph(np.where(mask)[0])
    
    # Setup plot
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    
    # Draw graph
    nx.draw(G, pos, 
           node_color=node_colors[mask],
           node_size=500,
           cmap=plt.cm.viridis,
           with_labels=True,
           width=2)
    
    plt.title(f'PROTEINS Graph {data_idx} (Label: {data.y.item()})')
    plt.show()

# Usage
draw_protein_graph(0)  # Draw first graph


# %%
import numpy as np
from matplotlib import pyplot as plt
import scanpy as sc
import scvelo as scv

# %%
FILE_NAME = "data/pancreas.h5ad"

adata = scv.datasets.pancreas(file_path=FILE_NAME)

# Choose the stratification criterion (e.g., 'cell_type', 'cluster', etc.)
# Ensure this column exists in adata.obs
stratify_by = 'clusters'  # Example: 'cell_type' or 'cluster'

# Calculate the number of observations to retain (half of the total observations)
n_obs_to_keep = adata.n_obs // 2

# Perform stratified sampling
# Group by the stratification criterion and sample half of the observations from each group
sampled_idx = adata.obs.groupby(stratify_by, group_keys=False).apply(
    lambda x: x.sample(frac=0.5, random_state=42)
).index

# Subset the AnnData object to keep only the sampled cells
adata_subsampled = adata[sampled_idx].copy()

scv.pp.filter_and_normalize(adata_subsampled, min_shared_counts=20, n_top_genes=2000)

sc.tl.pca(adata_subsampled)
sc.pp.neighbors(adata_subsampled)
scv.pp.moments(adata_subsampled, n_pcs=None, n_neighbors=None)

# Explicitly create and save the plot
fig, ax = plt.subplots()  # Create a new figure and axis
scv.pl.scatter(
    adata_subsampled, 
    basis="umap", 
    color="clusters", 
    show=False,  # Prevents automatic display
    ax=ax  # Explicitly use the created axis
)
fig.savefig("umap_plot.svg", format="svg")
plt.close(fig)  # Close the figure to free up memory
# %%
from ripser import ripser
from persim import plot_diagrams
import gudhi as gd

import torch
import torch.nn as nn
import torch.optim as optim
# %%
def extract_simplices(simplex_tree):
    """Extract simplices from a gudhi simplex tree.

    Parameters
    ----------
    simplex_tree: gudhi simplex tree

    Returns
    -------
    simplices: List of dictionaries, one per dimension d. The size of the dictionary
        is the number of d-simplices. The dictionary's keys are sets (of size d
        + 1) of the 0-simplices that constitute the d-simplices. The
        dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.
    """
    
    simplices = [dict() for _ in range(simplex_tree.dimension()+1)]
    for simplex, _ in simplex_tree.get_skeleton(simplex_tree.dimension()):
        k = len(simplex)
        simplices[k-1][frozenset(simplex)] = len(simplices[k-1])
    return simplices

def one_skeleton_to_chain(one_simplices, points):
    """  
    A function for turning a 1-skeleton into a chain

    Parameters
    ----------
    p : numpy array
        A set of 1-simplices extracted from the Gudhi simplex tree

    points : numpy array
        A set of vertices corresponding to the embedding

    Returns
    -------
    chain : numpy array
        A chain in R^n, represented as a numpy array of shape (p-1,2,n), where p is the number of points in the path.
        The middle index corresponds to start and endpoints of the edges in the chain.
    """

    r = len(one_simplices)

    n = points[0].shape[0]
    
    
    chain = torch.zeros((r,2,n))

    for i in range(r):

        chain[i,0,:] = torch.tensor(points[one_simplices[i][0]])
        chain[i,1,:] = torch.tensor(points[one_simplices[i][1]])
    

    return chain

# %%
# data = adata_subsampled.obsm['X_umap']

# find a good threshold for the data 
result = ripser(data, coeff=3, do_cocycles=True, maxdim =1)
diagrams = result['dgms']
cocycles = result['cocycles']
dgm1 = diagrams[1]

# identify the index of the longest interval
idx = np.argmax(dgm1[:, 1] - dgm1[:, 0])

# store the corresponding cocycle
cocycle = cocycles[1][idx]

# plot diagram with longest interval highlighted
plot_diagrams(diagrams, show = False)
plt.scatter(dgm1[idx, 0], dgm1[idx, 1], 20, 'k', 'x')
plt.title("Max 1D birth = %.3g, death = %.3g"%(dgm1[idx, 0], dgm1[idx, 1]))
plt.show()

# build a simplex tree
Rips_complex_sample = gd.RipsComplex(points = data, max_edge_length=0.5)
st = Rips_complex_sample.create_simplex_tree(max_dimension=2)

# extract simplices
simplices = extract_simplices(st)
one_simplices = [np.sort(list(elem)) for elem in simplices[1].keys()]
chain = one_skeleton_to_chain(one_simplices, data)
# %%
# Convert anndata and chain to PyTorch geometric format
def create_graph_data():
    # Convert features to torch tensor
    x = torch.FloatTensor(adata_subsampled.X.todense())
    
    # Create adjacency matrix from chain
    n = x.size(0)
    adj = torch.zeros((n, n))
    
    # Fill adjacency matrix based on chain connections
    for edge in chain:
        i = torch.argmin(torch.norm(x - edge[0].unsqueeze(0), dim=1))
        j = torch.argmin(torch.norm(x - edge[1].unsqueeze(0), dim=1))
        adj[i, j] = 1
        adj[j, i] = 1  # Make symmetric
    
    # Create mask for valid nodes
    mask = torch.ones(n, dtype=torch.bool)
    
    return x, adj, mask

# Modified DiffPool initialization
class DiffPoolSC(DiffPool):
    def __init__(self, num_features, sparse_threshold=0.1):
        super(DiffPool, self).__init__()
        self.sparse_threshold = sparse_threshold

        num_nodes = ceil(0.25 * adata_subsampled.n_obs)
        self.gnn1_pool = GNN(num_features, 64, num_nodes)
        self.gnn1_embed = GNN(num_features, 64, 64)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(64, 64, num_nodes)
        self.gnn2_embed = GNN(64, 64, 64, lin=False)

        self.gnn3_embed = GNN(64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(64, 64)
        self.lin2 = torch.nn.Linear(64, len(np.unique(adata_subsampled.obs['clusters'])))

# Training setup
x, adj, mask = create_graph_data()
num_features = x.shape[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DiffPoolSC(num_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert data to device
x = x.to(device)
adj = adj.to(device)
mask = mask.to(device)
labels = torch.tensor(pd.Categorical(adata_subsampled.obs['clusters']).codes).to(device)

# Training loop
def train_epoch():
    model.train()
    optimizer.zero_grad()
    output, l1, e1, _ = model(x, adj, mask)
    loss = F.nll_loss(output, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

# Training
for epoch in range(100):
    loss = train_epoch()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Extract hierarchical representations
model.eval()
with torch.no_grad():
    _, _, _, coarsened = model(x, adj, mask)
# %%
