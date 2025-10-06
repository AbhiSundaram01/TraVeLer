import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import networkx as nx
from datetime import datetime
from dataset import preprocess_pancreas_data
from neural_k_forms.forms import NeuralOneForm
from neural_k_forms.chains import generate_integration_matrix
from graph_utils import convert_to_chain_format

def setup_logging():
    """Setup logging and output directory"""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"test_runs/neural_oneform_test_{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f"{run_dir}/test.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    logger.info(f"Starting NeuralOneForm test {run_id}")
    
    return logger, run_dir

def plot_knn_graph1(adj, coords, save_path, logger):
    """Plot KNN graph using adjacency matrix and coordinates"""
    logger.info("Plotting KNN graph...")
    
    # Create networkx graph from adjacency matrix
    G = nx.from_numpy_array(adj.numpy())
    
    plt.figure(figsize=(12, 10))
    pos = {i: coords[i] for i in range(len(coords))}
    nx.draw(G, pos, node_size=30, node_color='blue', alpha=0.6, 
           width=0.5, with_labels=False)
    plt.title("KNN Graph from Adjacency Matrix")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"KNN graph saved to {save_path}")
    return G

def plot_knn_graph(adj, save_path, logger):
    """Plot KNN graph using adjacency matrix and NetworkX layout algorithms"""
    logger.info("Plotting KNN graph using NetworkX layout...")
    
    # Create networkx graph from adjacency matrix
    G = nx.from_numpy_array(adj.numpy())
    
    plt.figure(figsize=(12, 10))
    
    # Use NetworkX layout algorithms instead of external coordinates
    # Option 1: Force-directed layout (Fruchterman-Reingold algorithm)
    pos = nx.spring_layout(G, seed=42)
    
    # Alternative layouts - uncomment to try different ones:
    # pos = nx.kamada_kawai_layout(G)  # Another force-directed layout
    # pos = nx.spectral_layout(G)      # Uses eigenvectors of graph Laplacian
    # pos = nx.circular_layout(G)      # Simple circular layout
    
    nx.draw(G, pos, 
           node_size=30, 
           node_color='blue', 
           alpha=0.6, 
           width=0.5, 
           with_labels=False)
    
    plt.title("KNN Graph with Force-Directed Layout")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"KNN graph saved to {save_path}")

    # convert pos dictionary to numpy array for later use
    layout_coords = np.zeros((len(pos), 2))
    for i, coord in pos.items():
        layout_coords[i] = coord

    return G, layout_coords

# Replace the existing prepare_chain_coords function with this:
def prepare_chain_coords(chain, coords, logger):
    """Convert index-based chain to coordinate-based chain"""
    logger.info("Converting chain indices to coordinates...")
    
    # Get the actual feature dimension (either from coords or chain)
    feature_dim = coords.shape[1]  # Should be 2 for 2D spatial coords
    logger.info(f"Feature dimension from coords: {feature_dim}")
    
    # Convert indices to coordinates
    chain_coords = torch.zeros((chain.size(0), 2, feature_dim))
    for i in range(chain.size(0)):
        idx1 = int(chain[i, 0, 0].item())
        idx2 = int(chain[i, 1, 0].item())
        chain_coords[i, 0] = torch.tensor(coords[idx1])
        chain_coords[i, 1] = torch.tensor(coords[idx2])
    
    logger.info(f"Chain coordinates shape: {chain_coords.shape}")
    return chain_coords, feature_dim

def visualize_chain(chain, coords, save_path, logger):
    """Visualize chain as edges between points"""
    logger.info("Visualizing chain...")
    
    plt.figure(figsize=(12, 10))
    
    # Plot points
    plt.scatter(coords[:, 0], coords[:, 1], s=30, c='blue', alpha=0.6)
    
    # Plot chain edges
    for i in range(chain.size(0)):
        idx1 = int(chain[i, 0, 0].item())
        idx2 = int(chain[i, 1, 0].item())
        plt.plot([coords[idx1, 0], coords[idx2, 0]], 
                 [coords[idx1, 1], coords[idx2, 1]], 'r-', alpha=0.5, linewidth=0.7)
    
    plt.title(f"Chain Visualization: {chain.size(0)} edges")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"Chain visualization saved to {save_path}")

def initialize_neural_one_form(logger, input_dim = 2):
    """Initialize a simple NeuralOneForm with linear layers"""
    logger.info("Initializing NeuralOneForm...")
    
    hidden_dim = 64
    num_cochains = 1  # Single component vector field
    
    vf = NeuralOneForm(
        model=None,
        num_cochains=num_cochains,
        input_dim=input_dim,
        hidden_dim=hidden_dim
    )
    
    logger.info(f"Created NeuralOneForm with input_dim={input_dim}, hidden_dim={hidden_dim}")
    return vf

def visualize_vector_field(vf, coords, chain_coords, save_path, logger):
    """Visualize vector field on a grid and along chain"""
    logger.info("Visualizing vector field...")
    
    plt.figure(figsize=(12, 10))
    
    # Plot original data points
    plt.scatter(coords[:, 0], coords[:, 1], s=30, c='blue', alpha=0.3)
    
    # Create a grid for vector field visualization
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    # Add padding
    pad_x = (x_max - x_min) * 0.1
    pad_y = (y_max - y_min) * 0.1
    x_min -= pad_x; x_max += pad_x
    y_min -= pad_y; y_max += pad_y
    
    grid_size = 20
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Evaluate vector field at grid points
    grid_points = np.column_stack((xx.flatten(), yy.flatten()))
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    
    with torch.no_grad():
        vf_output = vf(grid_tensor)
    
    vf_output = vf_output.reshape(-1, 2).numpy()
    
    # Plot vector field
    plt.quiver(grid_points[:, 0], grid_points[:, 1], 
               vf_output[:, 0], vf_output[:, 1], 
               color='red', width=0.003, scale=30)
    
    # Also show vector field along chain edges
    for i in range(chain_coords.size(0)):
        midpoint = (chain_coords[i, 0] + chain_coords[i, 1]) / 2
        with torch.no_grad():
            vf_at_midpoint = vf(midpoint.unsqueeze(0))
        
        # Plot vector at midpoint
        plt.quiver(midpoint[0].item(), midpoint[1].item(),
                  vf_at_midpoint[0, 0].item(), vf_at_midpoint[0, 1].item(),
                  color='green', width=0.005, scale=15)
    
    plt.title("Vector Field Visualization")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"Vector field visualization saved to {save_path}")

def visualize_integration(integration, chain, coords, save_path, logger):
    """Visualize integration values along chain edges"""
    logger.info("Visualizing integration results...")
    
    plt.figure(figsize=(12, 10))
    
    # Plot points
    plt.scatter(coords[:, 0], coords[:, 1], s=30, c='blue', alpha=0.3)
    
    # Normalize integration values for coloring
    int_values = integration.squeeze().numpy()
    norm_values = (int_values - int_values.min()) / (int_values.max() - int_values.min() + 1e-8)
    
    # Plot chain edges colored by integration value
    for i in range(chain.size(0)):
        idx1 = int(chain[i, 0, 0].item())
        idx2 = int(chain[i, 1, 0].item())
        
        edge_color = plt.cm.viridis(norm_values[i])
        
        plt.plot([coords[idx1, 0], coords[idx2, 0]], 
                 [coords[idx1, 1], coords[idx2, 1]], 
                 color=edge_color, alpha=0.7, linewidth=2)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                              norm=plt.Normalize(int_values.min(), int_values.max()))
    sm.set_array([])
    plt.colorbar(sm, label='Integration Value')
    
    plt.title("Integration Values Along Chain Edges")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"Integration visualization saved to {save_path}")

def main():
    # Setup logging
    logger, run_dir = setup_logging()
    
    # 1. Load pancreas data
    FILE_NAME = "data/pancreas.h5ad"
    logger.info(f"Loading data from {FILE_NAME}")
    adata_subsampled, x, adj = preprocess_pancreas_data(FILE_NAME)
    
    # # Use PCA or UMAP for visualization coordinates
    # if hasattr(adata_subsampled, 'obsm') and 'X_pca' in adata_subsampled.obsm:
    #     coords = adata_subsampled.obsm['X_pca'][:, :2]
    # elif hasattr(adata_subsampled, 'obsm') and 'X_umap' in adata_subsampled.obsm:
    #     coords = adata_subsampled.obsm['X_umap']
    # else:
    #     coords = x.numpy()[:, :2]

    # # Get original high-dimensional data
    # high_dim_data = x.numpy()

    # # Create 2D coordinates for visualization
    # if hasattr(adata_subsampled, 'obsm') and 'X_umap' in adata_subsampled.obsm:
    #     viz_coords = adata_subsampled.obsm['X_umap']
    # elif hasattr(adata_subsampled, 'obsm') and 'X_pca' in adata_subsampled.obsm:
    #     viz_coords = adata_subsampled.obsm['X_pca'][:, :2]
    # else:
    #     # If no embeddings available, create simple 2D projection
    #     from sklearn.decomposition import PCA
    #     pca = PCA(n_components=2)
    #     viz_coords = pca.fit_transform(high_dim_data)
    
    # 2. Plot KNN graph
    # Use viz_coords for all visualization functions
    # G = plot_knn_graph(adj, viz_coords, f"{run_dir}/knn_graph.png", logger)
    G, layout_coords = plot_knn_graph(adj, f"{run_dir}/knn_graph.png", logger)
    logger.info(f"Generated layout coordinates with shape: {layout_coords.shape}")

    # Use these layout coordinates instead of UMAP/PCA for visualization
    viz_coords = layout_coords

    # 3. Convert adjacency matrix to chain format
    logger.info("Converting adjacency matrix to chain format...")
    chain = convert_to_chain_format(adj, x)
    logger.info(f"Chain shape: {chain.shape}")
    
    # # Subsample chain if it's too large
    # max_edges = 1000
    # if chain.shape[0] > max_edges:
    #     logger.info(f"Subsampling chain from {chain.shape[0]} to {max_edges} edges")
    #     indices = torch.randperm(chain.shape[0])[:max_edges]
    #     chain = chain[indices]
    
    # Visualize chain
    # visualize_chain(chain, coords, f"{run_dir}/chain_visualization.png", logger)
    visualize_chain(chain, viz_coords, f"{run_dir}/chain_visualization.png", logger)

    # Use high_dim_data for the neural network
    chain_coords, feature_dim = prepare_chain_coords(chain, layout_coords, logger)
    vf = initialize_neural_one_form(logger, input_dim=feature_dim)
    
    logger.info(f"DEBUG - chain_coords shape: {chain_coords.shape}, feature_dim: {feature_dim}")
    logger.info(f"DEBUG - vf model first layer weight shape: {vf.model[0].weight.shape}")

    # # 4. Initialize NeuralOneForm
    # vf = initialize_neural_one_form(logger, input_dim=2000)
    
    # # 5. Convert chain to coordinate format for vector field computation
    # chain_coords, feature_dim = prepare_chain_coords(chain, coords, logger)
    
    # 6. Compute vector field on chain
    logger.info("Computing vector field on chain...")
    with torch.no_grad():
        vf_output = vf(chain_coords)
    logger.info(f"Vector field output shape: {vf_output.shape}")
    
    # 7. Calculate integration matrix
    logger.info("Calculating integration matrix...")
    integration = generate_integration_matrix(vf, chain_coords)
    logger.info(f"Integration shape: {integration.shape}")
    logger.info(f"Integration stats - Min: {integration.min().item():.4f}, Max: {integration.max().item():.4f}, Mean: {integration.mean().item():.4f}")
    
    # # 8. Visualize vector field
    # # For vector field visualization, use 2D coordinates
    # visualize_vector_field(vf, viz_coords, chain_coords, f"{run_dir}/vector_field.png", logger)
    
    # # 9. Visualize integration results
    # visualize_integration(integration, chain, viz_coords, f"{run_dir}/integration_visualization.png", logger)
    
    # # 10. Create histogram of integration values
    # plt.figure(figsize=(10, 6))
    # plt.hist(integration.flatten().numpy(), bins=30, alpha=0.7)
    # plt.title("Distribution of Integration Values")
    # plt.xlabel("Integration Value")
    # plt.ylabel("Frequency")
    # plt.savefig(f"{run_dir}/integration_histogram.png", dpi=300)
    # plt.close()
    
    logger.info(f"All visualizations saved to {run_dir}")
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()