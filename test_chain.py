import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from datetime import datetime
import logging

# Import your modules
from models import DiffPool
from dataset import preprocess_pancreas_data
from analysis import visualize_joint_embeddings
from viz import plot_pca_variance

def setup_experiment(name):
    """Set up experiment directories and logging"""
    # Create a unique run ID based on timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/diffpool_test_{name}_{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f"{run_dir}/training.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    logger.info(f"Starting test run: {name}")
    
    return run_dir, logger

def analyze_embedding_linearity(embeddings):
    """Analyze how linear the embedding distribution is"""
    # Apply PCA
    pca = PCA()
    pca.fit(embeddings)
    
    # Get explained variance ratios
    explained_variance = pca.explained_variance_ratio_
    
    # Calculate linearity metrics
    top_component_ratio = explained_variance[0]
    top_two_components_ratio = explained_variance[0] + explained_variance[1]
    linearity_score = explained_variance[0] / (explained_variance[1] + 1e-6)
    
    return {
        'top_component_ratio': top_component_ratio,
        'top_two_components_ratio': top_two_components_ratio,
        'linearity_score': linearity_score,
        'explained_variance': explained_variance
    }

def plot_embedding_analysis(embeddings, title, save_path=None):
    """Visualize embeddings and analyze their structure"""
    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(embeddings)
    
    # Create a 2x2 figure with multiple plots
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: First 2 PCA components
    plt.subplot(2, 2, 1)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
    plt.title(f"First 2 PCA Components - {title}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    
    # Plot 2: First 3 PCA components in 3D
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], alpha=0.6)
    ax.set_title(f"First 3 PCA Components - {title}")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    
    # Plot 3: Explained variance ratio
    plt.subplot(2, 2, 3)
    plt.bar(range(1, min(11, len(pca.explained_variance_ratio_)+1)), 
            pca.explained_variance_ratio_[:10] * 100)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance %")
    plt.title(f"Eigenvalue Spectrum - {title}")
    
    # Plot 4: Raw embedding dimensions (first 2)
    plt.subplot(2, 2, 4)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6)
    plt.title(f"Raw Embedding (First 2 Dims) - {title}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
    
    return fig

def train_with_link_prediction(model, x, adj, adata, epochs=300):
    """Train DiffPool with link prediction objective"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    
    # Create training and validation masks for edges
    edge_index = adj.coalesce().indices()  # Get sparse representation
    num_edges = edge_index.size(1)
    
    # Random mask for training (80% of edges)
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    perm = torch.randperm(num_edges)
    train_size = int(0.8 * num_edges)
    train_mask[perm[:train_size]] = True
    
    # Create negative edges (same number as positive edges)
    neg_adj = 1 - adj - torch.eye(adj.size(0), device=adj.device)
    neg_edge_index = neg_adj.nonzero(as_tuple=False).t()
    neg_edge_index = neg_edge_index[:, torch.randperm(neg_edge_index.size(1))[:num_edges]]
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass through DiffPool
        x_out, _ = model(x, adj)
        node_embeddings = model.compute_node_embeddings(x, adj)
        
        # Link prediction loss
        # Use dot product between node embeddings to predict edges
        pos_out = torch.sum(node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]], dim=1)
        neg_out = torch.sum(node_embeddings[neg_edge_index[0]] * node_embeddings[neg_edge_index[1]], dim=1)
        
        # Use only training edges
        pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pos_out[train_mask], torch.ones_like(pos_out[train_mask]))
        neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            neg_out[train_mask], torch.zeros_like(neg_out[train_mask]))
        
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")
    
    return losses

def train_with_reconstruction(model, x, adj, adata, epochs=300):
    """Train DiffPool with reconstruction objective"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass through DiffPool
        x_out, adj_out = model(x, adj)
        
        # Get node embeddings through the hierarchy
        node_embeddings = model.compute_node_embeddings(x, adj)
        
        # Reconstruct original features and adjacency
        x_reconstructed = node_embeddings @ torch.pinverse(node_embeddings) @ x
        adj_reconstructed = node_embeddings @ torch.pinverse(node_embeddings) @ adj @ torch.pinverse(node_embeddings).t() @ node_embeddings.t()
        
        # Reconstruction loss
        feat_loss = torch.nn.functional.mse_loss(x_reconstructed, x)
        adj_loss = torch.nn.functional.mse_loss(adj_reconstructed, adj)
        
        # Combine losses with weighting
        loss = feat_loss + 0.5 * adj_loss
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}, Feat Loss: {feat_loss.item():.4f}, Adj Loss: {adj_loss.item():.4f}")
    
    return losses

def train_with_clustering(model, x, adj, adata, epochs=300):
    """Train DiffPool with clustering quality objective"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    
    # Get cluster labels if available
    if hasattr(adata, 'obs') and 'clusters' in adata.obs:
        cluster_labels = adata.obs['clusters'].cat.codes.values
    else:
        # Create dummy labels if real ones aren't available
        cluster_labels = np.zeros(x.size(0))
    
    cluster_labels_tensor = torch.tensor(cluster_labels, device=x.device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass through DiffPool
        x_out, _ = model(x, adj)
        
        # Get assignment matrices
        with torch.no_grad():
            z_0 = model.gnn1_embed(adj, x)
            s_0 = torch.softmax(model.gnn1_pool(adj, x), dim=-1)
        
        # Compute cluster assignments
        _, cluster_assignments = torch.max(s_0, dim=1)
        
        # Cross-entropy between DiffPool clusters and original clusters
        cluster_loss = torch.nn.functional.cross_entropy(s_0, cluster_labels_tensor)
        
        # Soft assignment entropy (encourage confident assignments)
        entropy = -torch.sum(s_0 * torch.log(s_0 + 1e-10), dim=1).mean()
        
        # Combine losses
        loss = cluster_loss + 0.1 * entropy
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}, Cluster Loss: {cluster_loss.item():.4f}, Entropy: {entropy.item():.4f}")
    
    return losses

def train_with_combined_loss(model, x, adj, adata, epochs=300):
    """Train DiffPool with a combination of loss functions"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    
    # Get cluster labels if available
    if hasattr(adata, 'obs') and 'clusters' in adata.obs:
        cluster_labels = adata.obs['clusters'].cat.codes.values
    else:
        # Create dummy labels if real ones aren't available
        cluster_labels = np.zeros(x.size(0))
    
    cluster_labels_tensor = torch.tensor(cluster_labels, device=x.device)
    
    # Create adjacency edge indices for link prediction
    edge_index = adj.coalesce().indices()
    num_edges = edge_index.size(1)
    
    # Create negative edges (same number as positive edges)
    neg_adj = 1 - adj - torch.eye(adj.size(0), device=adj.device)
    neg_edge_index = neg_adj.nonzero(as_tuple=False).t()
    neg_edge_index = neg_edge_index[:, torch.randperm(neg_edge_index.size(1))[:num_edges]]
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass through DiffPool
        x_out, adj_out = model(x, adj)
        
        # Get node embeddings and assignment matrices
        node_embeddings = model.compute_node_embeddings(x, adj)
        z_0 = model.gnn1_embed(adj, x)
        s_0 = torch.softmax(model.gnn1_pool(adj, x), dim=-1)
        
        # 1. Clustering loss
        cluster_loss = torch.nn.functional.cross_entropy(s_0, cluster_labels_tensor)
        
        # 2. Link prediction loss
        pos_out = torch.sum(node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]], dim=1)
        neg_out = torch.sum(node_embeddings[neg_edge_index[0]] * node_embeddings[neg_edge_index[1]], dim=1)
        
        link_pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pos_out, torch.ones_like(pos_out))
        link_neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            neg_out, torch.zeros_like(neg_out))
        link_loss = link_pos_loss + link_neg_loss
        
        # 3. Orthogonality regularization
        # Encourage embeddings to use all dimensions
        X = x_out - x_out.mean(dim=0, keepdim=True)
        X = X / (X.norm(dim=0, keepdim=True) + 1e-8)
        corr = X.t() @ X
        mask = 1.0 - torch.eye(corr.shape[0], device=corr.device)
        ortho_loss = (corr * mask).pow(2).sum()
        
        # Combine all losses
        loss = cluster_loss + 0.5 * link_loss + 0.1 * ortho_loss
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}, Cluster: {cluster_loss.item():.4f}, "
                  f"Link: {link_loss.item():.4f}, Ortho: {ortho_loss.item():.4f}")
    
    return losses

def main():
    # Load data
    print("Loading data...")
    FILE_NAME = "data/pancreas.h5ad"
    adata_subsampled, x, adj = preprocess_pancreas_data(FILE_NAME)
    
    # Test different loss functions
    test_functions = {
        'link_prediction': train_with_link_prediction,
        'reconstruction': train_with_reconstruction,
        'clustering': train_with_clustering,
        'combined': train_with_combined_loss
    }
    
    results = {}
    
    for name, train_func in test_functions.items():
        print(f"\n\n=== Testing {name} loss function ===")
        run_dir, logger = setup_experiment(name)
        
        # Initialize a fresh model for each test
        model = DiffPool(num_features=x.size(1), max_nodes=x.size(0))
        
        # Train model with specific loss function
        losses = train_func(model, x, adj, adata_subsampled, epochs=300)
        
        # Save the model
        torch.save({
            'model_state_dict': model.state_dict(),
            'losses': losses,
        }, f"{run_dir}/model_final.pt")
        
        # Get embeddings
        model.eval()
        with torch.no_grad():
            # Final output (cluster embeddings)
            x_out, _ = model(x, adj)
            
            # Node embeddings (both versions)
            node_embeddings_inter = model.compute_node_embeddings(x, adj, full_hierarchy=False)
            node_embeddings_full = model.compute_node_embeddings(x, adj, full_hierarchy=True)
            
            # Convert to numpy for analysis
            x_out_np = x_out.cpu().numpy()
            node_inter_np = node_embeddings_inter.cpu().numpy()
            node_full_np = node_embeddings_full.cpu().numpy()
        
        # Analyze linearity of embeddings
        linearity_cluster = analyze_embedding_linearity(x_out_np)
        linearity_inter = analyze_embedding_linearity(node_inter_np)
        linearity_full = analyze_embedding_linearity(node_full_np)
        
        # Log results
        logger.info(f"Cluster embedding linearity: top component explains {linearity_cluster['top_component_ratio']*100:.2f}% variance")
        logger.info(f"Intermediate node embedding linearity: top component explains {linearity_inter['top_component_ratio']*100:.2f}% variance")
        logger.info(f"Full hierarchy node embedding linearity: top component explains {linearity_full['top_component_ratio']*100:.2f}% variance")
        
        # Visualize embeddings
        plot_embedding_analysis(x_out_np, f"{name} - Cluster Embeddings", f"{run_dir}/cluster_embedding_analysis.png")
        plot_embedding_analysis(node_inter_np, f"{name} - Intermediate Node Embeddings", f"{run_dir}/intermediate_embedding_analysis.png")
        plot_embedding_analysis(node_full_np, f"{name} - Full Hierarchy Node Embeddings", f"{run_dir}/full_embedding_analysis.png")
        
        # Create joint visualization
        fig_joint = visualize_joint_embeddings(model, x, adj, adata_subsampled, full_hierarchy=False)
        fig_joint.savefig(f"{run_dir}/joint_embeddings.png", dpi=300)
        plt.close(fig_joint)
        
        # Store summarized results
        results[name] = {
            'losses': losses,
            'linearity_cluster': linearity_cluster,
            'linearity_inter': linearity_inter,
            'linearity_full': linearity_full
        }
    
    # Compare results across different loss functions
    print("\n\n=== Comparison of Embedding Linearity Across Loss Functions ===")
    for name, result in results.items():
        cluster_linearity = result['linearity_cluster']['top_component_ratio']
        inter_linearity = result['linearity_inter']['top_component_ratio']
        full_linearity = result['linearity_full']['top_component_ratio']
        
        print(f"{name.capitalize()} Loss:")
        print(f"  - Cluster Embeddings: {cluster_linearity*100:.2f}% variance in top component")
        print(f"  - Intermediate Node Embeddings: {inter_linearity*100:.2f}% variance in top component")
        print(f"  - Full Hierarchy Node Embeddings: {full_linearity*100:.2f}% variance in top component")
    
    # Create summary visualization
    plt.figure(figsize=(12, 8))
    metrics = ['Cluster', 'Intermediate', 'Full Hierarchy']
    x_pos = np.arange(len(metrics))
    width = 0.2
    
    # Plot bars for each loss function
    for i, (name, result) in enumerate(results.items()):
        values = [
            result['linearity_cluster']['top_component_ratio'] * 100,
            result['linearity_inter']['top_component_ratio'] * 100,
            result['linearity_full']['top_component_ratio'] * 100
        ]
        plt.bar(x_pos + (i - 1.5) * width, values, width, label=name.capitalize())
    
    plt.ylabel('Variance Explained by Top Component (%)')
    plt.title('Embedding Linearity Comparison Across Loss Functions')
    plt.xticks(x_pos, metrics)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig("linearity_comparison.png", dpi=300)
    
    print("\nAnalysis complete! Check the runs directory for detailed results.")

if __name__ == "__main__":
    main()