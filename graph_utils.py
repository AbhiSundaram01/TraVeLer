import torch
from torch_geometric.utils import dense_to_sparse
import networkx as nx
import matplotlib.pyplot as plt

def extract_coarsened_graphs(model, loader, device):
    model.eval()
    coarsened_graphs = []
    labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, _, _, graphs = model(data.x, data.adj, data.mask)
            coarsened_graphs.append(graphs)
            labels.append(data.y)
            
    return coarsened_graphs, labels

def draw_graph(adj, x, mask=None, s=None, threshold=0):
    if mask is None:
        mask = torch.ones(adj.size(0), dtype=torch.bool)
        
    # Convert to sparse format
    edge_index, _ = dense_to_sparse(adj)
    
    # Create networkx graph
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())
    
    # Draw
    plt.figure(figsize=(8, 8))
    nx.draw(G, 
            node_color='lightblue',
            node_size=50,
            width=0.5,
            edge_color='gray',
            with_labels=False)
    plt.title(f'Graph with {G.number_of_nodes()} nodes')
    plt.show()

def convert_to_chain_format(adj, x):
    """
    Convert adjacency matrix and node features to chain format.
    
    Parameters:
    -----------
    adj : torch.Tensor
        Adjacency matrix of shape [num_nodes, num_nodes]
    x : torch.Tensor
        Node features of shape [num_nodes, feature_dim]
        
    Returns:
    --------
    torch.Tensor
        Chain format tensor of shape [num_edges, 2, feature_dim]
        where each entry contains the features of source and target nodes
    """
    # Find edges (non-zero entries in adjacency matrix)
    edges = torch.nonzero(adj > 0)  # Shape: [num_edges, 2]
    
    # If no edges found, return empty tensor
    if edges.size(0) == 0:
        return torch.zeros((0, 2, x.size(1)))
    
    # Get features for source and target nodes of each edge
    source_features = x[edges[:, 0]]  # Shape: [num_edges, feature_dim]
    target_features = x[edges[:, 1]]  # Shape: [num_edges, feature_dim]
    
    # Stack source and target features along dim=1
    chain = torch.stack([source_features, target_features], dim=1)  # Shape: [num_edges, 2, feature_dim]
    
    return chain

def create_mst_chain_from_coords(x_coords):
    """
    Create a minimum spanning tree chain using Euclidean distances.
    
    Parameters:
    -----------
    x_coords : torch.Tensor
        Node coordinates tensor of shape [num_nodes, feature_dim]
    
    Returns:
    --------
    torch.Tensor
        Chain format tensor containing only MST edges
    """
    import networkx as nx
    import numpy as np
    
    # Convert inputs to numpy for NetworkX
    node_positions = x_coords.detach().numpy()
    
    # Create a complete graph with all nodes connected
    G = nx.Graph()
    
    # Add all nodes with positions
    for i in range(len(node_positions)):
        G.add_node(i, pos=node_positions[i])
    
    # Add edges between all pairs of nodes with distance as weight
    for i in range(len(node_positions)):
        for j in range(i+1, len(node_positions)):
            # Calculate Euclidean distance between nodes
            dist = np.linalg.norm(node_positions[i] - node_positions[j])
            G.add_edge(i, j, weight=dist)
    
    # Find the minimum spanning tree
    mst = nx.minimum_spanning_tree(G)
    # print(f"Full graph: {len(G.edges)} edges")
    # print(f"MST: {len(mst.edges)} edges")
    
    # Convert MST to chain format
    mst_edges = []
    pos = nx.get_node_attributes(G, 'pos')
    
    for u, v in mst.edges():
        # Get node coordinates
        source_pos = pos[u]
        target_pos = pos[v]
        mst_edges.append([source_pos, target_pos])
    
    # Convert to PyTorch tensor in chain format
    mst_edges_tensor = torch.tensor(mst_edges, dtype=torch.float)
    
    return mst_edges_tensor

def create_mst_chain(x_coords, adjacency):
    """
    Create a chain for training that uses the minimum spanning tree of the graph.
    
    Parameters:
    -----------
    x_coords : torch.Tensor
        Node coordinates tensor of shape [num_nodes, feature_dim]
    adjacency : torch.Tensor
        Adjacency matrix of shape [num_nodes, num_nodes]
    
    Returns:
    --------
    torch.Tensor
        Chain format tensor containing only MST edges
    """
    import networkx as nx
    
    # Convert inputs to numpy for NetworkX
    node_positions = x_coords.detach().numpy()
    adj_np = adjacency.detach().numpy()
    
    # Create a networkx graph
    G = nx.Graph()
    
    # Add nodes with positions
    for i in range(len(node_positions)):
        G.add_node(i, pos=node_positions[i])
    
    # Add all edges with weights (using adjacency weights, not distances)
    for i in range(adj_np.shape[0]):
        for j in range(i+1, adj_np.shape[1]):  # Avoid duplicates (only upper triangle)
            if adj_np[i, j] > 0:  # If there's an edge
                # Use inverse of weight so stronger connections have lower "distance"
                # This ensures the MST prioritizes stronger connections
                inverse_weight = 1.0 / (adj_np[i, j] + 1e-6)
                G.add_edge(i, j, weight=inverse_weight)
    
    # Find the minimum spanning tree
    mst = nx.minimum_spanning_tree(G)
    print(f"Original graph: {len(G.edges)} edges")
    print(f"MST: {len(mst.edges)} edges")
    
    # Convert MST back to chain format
    mst_edges = []
    for u, v in mst.edges():
        # Get node coordinates
        source_pos = node_positions[u]
        target_pos = node_positions[v]
        mst_edges.append((source_pos, target_pos))
    
    # Convert to PyTorch tensor in chain format
    mst_edges_tensor = torch.tensor(mst_edges, dtype=torch.float)
    # Reshape to [num_edges, 2, feature_dim]
    chain = mst_edges_tensor.view(-1, 2, node_positions.shape[1])
    
    return chain

def soft_mst_approximation(x_out, temperature=0.1):
    """Create a differentiable approximation of MST"""
    # Calculate pairwise distances between all nodes
    n = x_out.shape[0]
    distances = torch.cdist(x_out, x_out)
    
    # Create soft edge weights using softmin
    edge_weights = torch.softmax(-distances/temperature, dim=1)
    
    # Select top k edges per node (sparsification)
    k = 2  # Adjust as needed
    values, indices = torch.topk(edge_weights, k, dim=1)
    
    # Create chain from these edges
    edges = []
    for i in range(n):
        for j in indices[i]:
            if i < j:  # Avoid duplicates
                edges.append((i, j))
    
    # Convert to chain format
    chain = torch.zeros((len(edges), 2, x_out.shape[1]), device=x_out.device)
    for idx, (i, j) in enumerate(edges):
        chain[idx, 0] = x_out[i]
        chain[idx, 1] = x_out[j]
    
    return chain

def soft_mst_approximation2(x_out, temperature=0.1, target_edges=None):
    """
    Create a differentiable approximation of MST with a fixed number of edges.
    
    Parameters:
    -----------
    x_out : torch.Tensor
        Node positions tensor
    temperature : float
        Temperature parameter for softmax (lower = sharper)
    target_edges : int or None
        If specified, exactly this many edges will be returned
    """
    # Calculate pairwise distances between all nodes
    n = x_out.shape[0]
    distances = torch.cdist(x_out, x_out)
    
    # Create soft edge weights using softmax
    edge_weights = torch.softmax(-distances/temperature, dim=1)
    
    # Create all possible edges (excluding self-loops)
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):  # Only upper triangle to avoid duplicates
            edge_list.append((i, j, edge_weights[i, j].item()))
    
    # Sort by edge weight (descending)
    edge_list.sort(key=lambda x: x[2], reverse=True)
    
    # If target_edges is specified, ensure we return exactly that many edges
    if target_edges is not None:
        if len(edge_list) > target_edges:
            # If we have too many edges, take only the top ones
            edge_list = edge_list[:target_edges]
        elif len(edge_list) < target_edges:
            # If we have too few edges, duplicate some (with small perturbation)
            additional_needed = target_edges - len(edge_list)
            extra_edges = []
            
            for idx in range(additional_needed):
                # Cycle through existing edges if needed
                source_idx = idx % len(edge_list)
                i, j, w = edge_list[source_idx]
                
                # Add small perturbation to make edges unique
                perturb = 0.01 * (idx + 1) / additional_needed
                extra_edges.append((i, j, w * (1.0 - perturb)))
            
            edge_list.extend(extra_edges)
    
    # Create chain tensor from the selected edges
    chain = torch.zeros((len(edge_list), 2, x_out.shape[1]), device=x_out.device)
    for idx, (i, j, _) in enumerate(edge_list):
        chain[idx, 0] = x_out[i]
        chain[idx, 1] = x_out[j]
    
    return chain

def weighted_edge_chain(x_out, max_edges=100):
    """Create a chain using all nodes with weighted importance"""
    n = x_out.shape[0]
    
    # Calculate all pairwise distances
    distances = torch.cdist(x_out, x_out)
    
    # Create edges (avoiding self-loops)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j, distances[i, j]))
    
    # Sort edges by distance and take top k
    edges.sort(key=lambda x: x[2])
    selected_edges = edges[:max_edges]
    
    # Create chain tensor
    chain = torch.zeros((len(selected_edges), 2, x_out.shape[1]), device=x_out.device)
    for idx, (i, j, _) in enumerate(selected_edges):
        chain[idx, 0] = x_out[i]
        chain[idx, 1] = x_out[j]
        
    return chain