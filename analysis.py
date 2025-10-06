import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # Fixed import

def analyze_cluster_correspondence(model, x, adj, adata_subsampled):
    """
    Analyze how DiffPool's learned clusters correspond to biological clusters.
    """
    # Get original biological clusters with their actual names
    bio_clusters_names = adata_subsampled.obs['clusters'].values
    bio_clusters = adata_subsampled.obs['clusters'].cat.codes.values
    unique_bio_clusters = np.unique(bio_clusters)
    unique_bio_names = adata_subsampled.obs['clusters'].cat.categories.tolist()
    n_bio_clusters = len(unique_bio_clusters)
    
    # Forward pass through model to get cluster assignments
    model.eval()
    with torch.no_grad():
        # Forward pass to extract s_0 and s_1 (cluster assignments)
        z_0 = model.gnn1_embed(adj, x)
        s_0 = torch.softmax(model.gnn1_pool(adj, x), dim=-1)
        
        x_1 = s_0.t() @ z_0
        adj_1 = s_0.t() @ adj @ s_0
        
        z_1 = model.gnn2_embed(adj_1, x_1)
        s_1 = torch.softmax(model.gnn2_pool(adj_1, x_1), dim=-1)
        
        # Convert to numpy
        s_0_np = s_0.numpy()
        s_1_np = s_1.numpy()
        
        # For each original node, find its primary level-1 cluster
        level1_assignments = np.argmax(s_0_np, axis=1)  
        
        # For each level-1 cluster, find its primary level-2 cluster
        level2_assignments = np.argmax(s_1_np, axis=1)  
        
        # Map original nodes to final clusters (level 2)
        final_assignments = level2_assignments[level1_assignments]  
    
    # Compute cluster correspondence metrics
    ari = adjusted_rand_score(bio_clusters, final_assignments)
    nmi = normalized_mutual_info_score(bio_clusters, final_assignments)
    
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    
    # Create correspondence matrix between biological and DiffPool clusters
    n_diffpool_clusters = s_1_np.shape[1]
    correspondence = np.zeros((n_bio_clusters, n_diffpool_clusters))
    
    for i in range(len(bio_clusters)):
        bio_cluster = bio_clusters[i]
        diff_cluster = final_assignments[i]
        correspondence[bio_cluster, diff_cluster] += 1
    
    # Normalize by row sums
    correspondence = correspondence / correspondence.sum(axis=1, keepdims=True)
    
    # Plot heatmap of correspondences with actual cluster names
    plt.figure(figsize=(12, 10))
    sns.heatmap(correspondence, annot=True, cmap='viridis', fmt='.2f',
                xticklabels=[f'DP-{i}' for i in range(n_diffpool_clusters)],
                yticklabels=unique_bio_names)
    plt.title('Correspondence between Biological and DiffPool Clusters')
    plt.xlabel('DiffPool Clusters')
    plt.ylabel('Biological Clusters')
    plt.tight_layout()
    plt.show()
    
    # Visualize hierarchical structure
    visualize_hierarchical_clusters(s_0_np, s_1_np, bio_clusters, unique_bio_names)
    
    return correspondence, ari, nmi, final_assignments

def visualize_hierarchical_clusters(s_0, s_1, bio_clusters, bio_cluster_names):
    """Visualize the hierarchical clustering structure"""
    # Get the actual number of clusters from the shape of s_0 and s_1
    num_nodes, num_clusters_l1 = s_0.shape
    num_clusters_l1, num_clusters_l2 = s_1.shape
    
    unique_bio_clusters = np.unique(bio_clusters)
    num_bio_clusters = len(unique_bio_clusters)
    
    # Create a mapping of biological clusters to consecutive indices
    bio_cluster_map = {cluster: idx for idx, cluster in enumerate(unique_bio_clusters)}
    
    # Map original biological cluster IDs to consecutive indices
    mapped_bio_clusters = np.array([bio_cluster_map[cluster] for cluster in bio_clusters])
    
    # For each level-1 cluster, count biological clusters
    level1_assignments = np.argmax(s_0, axis=1)
    level1_bio_counts = np.zeros((num_clusters_l1, num_bio_clusters))
    
    for i in range(len(mapped_bio_clusters)):
        l1_cluster = level1_assignments[i]
        bio_cluster = mapped_bio_clusters[i]
        level1_bio_counts[l1_cluster, bio_cluster] += 1
    
    # Show top 10 level-1 clusters by size (or all if less than 10)
    cluster_sizes = np.sum(level1_bio_counts, axis=1)
    top_n = min(10, num_clusters_l1)
    top_clusters = np.argsort(cluster_sizes)[-top_n:]
    
    # Plot distribution of biological clusters in top level-1 clusters
    n_cols = min(5, top_n)
    n_rows = (top_n + n_cols - 1) // n_cols
    
    plt.figure(figsize=(4*n_cols, 4*n_rows))
    for i, cluster_idx in enumerate(top_clusters):
        plt.subplot(n_rows, n_cols, i+1)
        dist = level1_bio_counts[cluster_idx]
        
        # Create bar plot with actual cluster names as x-tick labels
        plt.bar(range(len(dist)), dist)
        plt.xticks(range(len(dist)), bio_cluster_names, rotation=90)
        plt.title(f'L1 Cluster {cluster_idx}')
        plt.xlabel('Biological Cluster')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    
    # Visualize the level-2 assignment of level-1 clusters
    level2_assignments = np.argmax(s_1, axis=1)  # [num_clusters_l1]
    
    # Create mapping from level-1 clusters to level-2
    l1_to_l2 = {}
    for i in range(num_clusters_l1):
        l2 = level2_assignments[i]
        if l2 not in l1_to_l2:
            l1_to_l2[l2] = []
        l1_to_l2[l2].append(i)
    
    # Plot the hierarchy
    n_cols = min(4, num_clusters_l2)
    n_rows = (num_clusters_l2 + n_cols - 1) // n_cols
    
    plt.figure(figsize=(5*n_cols, 5*n_rows))
    for i, (l2, l1_clusters) in enumerate(l1_to_l2.items()):
        if i >= n_rows * n_cols:  # Ensure we don't exceed subplot grid
            break
        plt.subplot(n_rows, n_cols, i+1)
        plt.title(f'L2 Cluster {l2}')
        
        # Aggregate bio distributions for all l1 clusters in this l2 cluster
        bio_dist = np.zeros(num_bio_clusters)
        for l1 in l1_clusters:
            bio_dist += level1_bio_counts[l1]
        
        # Plot with actual cluster names
        plt.bar(range(len(bio_dist)), bio_dist)
        plt.xticks(range(len(bio_dist)), bio_cluster_names, rotation=90)
        plt.xlabel('Biological Cluster')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# Update the visualization of original vs diffpool clusters
def visualize_clusters_in_umap(adata_subsampled, final_assignments):
    """Visualize original biological clusters and DiffPool clusters in UMAP space"""
    plt.figure(figsize=(15, 6))

    # Original biological clusters
    plt.subplot(1, 2, 1)
    
    # Get unique categories and create a color map
    categories = list(adata_subsampled.obs['clusters'].cat.categories)  # Convert to list
    cmap = plt.cm.get_cmap('tab20', len(categories))
    
    # Create scatter plot with categorical colors
    scatter = plt.scatter(adata_subsampled.obsm['X_umap'][:, 0], 
                adata_subsampled.obsm['X_umap'][:, 1],
                c=adata_subsampled.obs['clusters'].cat.codes, 
                cmap=cmap, s=10, alpha=0.7)
    plt.title('Original Biological Clusters')
    
    # Create custom legend with actual cluster names - fixed by using list(scatter.legend_elements()[0])
    handles, labels = scatter.legend_elements()
    legend1 = plt.legend(handles=handles[:len(categories)],  # Ensure we have the right number of handles
                        labels=categories,
                        title="Cell Types",
                        loc="upper right")
    plt.gca().add_artist(legend1)

    # DiffPool clusters
    plt.subplot(1, 2, 2)
    n_diffpool_clusters = len(np.unique(final_assignments))
    diffpool_cmap = plt.cm.get_cmap('tab10', n_diffpool_clusters)
    
    scatter2 = plt.scatter(adata_subsampled.obsm['X_umap'][:, 0], 
                adata_subsampled.obsm['X_umap'][:, 1],
                c=final_assignments, 
                cmap=diffpool_cmap, s=10, alpha=0.7)
    plt.title('DiffPool-derived Clusters')
    
    # Add legend for DiffPool clusters - also fixed
    handles2, labels2 = scatter2.legend_elements()
    legend2 = plt.legend(handles=handles2[:n_diffpool_clusters],
                        labels=[f'DP-{i}' for i in range(n_diffpool_clusters)],
                        title="DiffPool Clusters",
                        loc="upper right")
    plt.gca().add_artist(legend2)

    plt.tight_layout()
    plt.show()

def visualize_diffpool_embeddings(model, x, adj, adata_subsampled):
    """
    Visualize the original data directly in the DiffPool embedding space.
    
    Instead of using UMAP coordinates, we use DiffPool's own hierarchical embedding.
    """
    # Get original biological clusters with their actual names
    if 'clusters' in adata_subsampled.obs:
        bio_clusters = adata_subsampled.obs['clusters']
    else:
        print("Warning: No 'clusters' column found, using first categorical column as clusters")
        bio_clusters = adata_subsampled.obs.select_dtypes(['category']).iloc[:, 0]
    
    # Compute DiffPool embeddings for original nodes
    model.eval()
    with torch.no_grad():
        # Compute embeddings for all original nodes
        node_embeddings = model.compute_node_embeddings(x, adj)
    
    # Create a single plot with better size for detailed visualization
    plt.figure(figsize=(12, 10))
    
    # Plot: Original nodes in DiffPool embedding space, colored by biological clusters
    embeddings = node_embeddings.cpu().numpy()
    categories = list(bio_clusters.cat.categories)
    cmap = plt.cm.get_cmap('tab20', len(categories))
    
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1],
                c=bio_clusters.cat.codes, cmap=cmap, s=15, alpha=0.7)
    plt.title('Cell Types in DiffPool Embedding Space', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    
    # Create legend with actual cluster names
    handles, labels = scatter.legend_elements()
    legend = plt.legend(handles=handles[:len(categories)],
                      labels=categories,
                      title="Cell Types",
                      loc="best",
                      fontsize=10)
    plt.setp(legend.get_title(), fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return plt.gcf()  # Return the figure for saving

def visualize_joint_embeddings(model, x, adj, adata_subsampled=None, full_hierarchy=False):
    """
    Plot both node embeddings and cluster centers in the same normalized space.
    
    Args:
        model: DiffPool model
        x: Input node features
        adj: Adjacency matrix
        adata_subsampled: Optional AnnData object with cluster info
        full_hierarchy: Whether to use full hierarchy embeddings
    
    Returns:
        fig: Matplotlib figure containing the visualization
    """
    model.eval()
    with torch.no_grad():
        # Get node embeddings
        x_out, _ = model(x, adj)  # This gives us the final cluster centers (x_out)
        
        # Get node embeddings through the hierarchy
        node_embeddings = model.compute_node_embeddings(x, adj, full_hierarchy=full_hierarchy)
        
        # Convert to numpy for processing
        node_emb_np = node_embeddings.cpu().numpy()
        cluster_emb_np = x_out.cpu().numpy()
        
        # Combine for joint normalization
        all_embeddings = np.vstack([node_emb_np, cluster_emb_np])
        
        # Normalize to zero mean and unit variance
        mean = np.mean(all_embeddings, axis=0, keepdims=True)
        std = np.std(all_embeddings, axis=0, keepdims=True) + 1e-8  # Add epsilon for stability
        
        # Apply normalization
        node_emb_norm = (node_emb_np - mean) / std
        cluster_emb_norm = (cluster_emb_np - mean) / std
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    
    # Plot node embeddings - color by biological cluster if available
    if adata_subsampled is not None and hasattr(adata_subsampled, 'obs') and 'clusters' in adata_subsampled.obs:
        bio_clusters = adata_subsampled.obs['clusters']
        categories = list(bio_clusters.cat.categories)
        cmap = plt.cm.get_cmap('tab20', len(categories))
        
        scatter1 = plt.scatter(node_emb_norm[:, 0], node_emb_norm[:, 1],
                    c=bio_clusters.cat.codes, cmap=cmap, 
                    s=30, alpha=0.6, edgecolors='none')
        
        # Add legend for biological clusters
        handles, labels = scatter1.legend_elements()
        legend1 = plt.legend(handles=handles[:len(categories)],
                          labels=categories,
                          title="Cell Types",
                          loc="upper right",
                          fontsize=9)
        plt.gca().add_artist(legend1)
    else:
        # Just plot nodes without biological cluster coloring
        plt.scatter(node_emb_norm[:, 0], node_emb_norm[:, 1],
                    c='royalblue', s=30, alpha=0.6, label="Nodes")
    
    # Plot cluster centers with distinctive markers
    scatter2 = plt.scatter(cluster_emb_norm[:, 0], cluster_emb_norm[:, 1],
                c='red', s=150, marker='*', edgecolors='black', 
                label="Cluster Centers")
    
    # Add legend for cluster centers separately
    plt.legend([scatter2], ["Cluster Centers"], 
              loc="lower right", fontsize=12)
    
    # # Draw connections from each node to nearest cluster (optional)
    # # This is computationally expensive, so we'll do it for a sample of nodes
    # if node_emb_norm.shape[0] <= 1000:  # Only for reasonably sized graphs
    #     # Calculate distances and find closest cluster for each node
    #     from scipy.spatial.distance import cdist
    #     distances = cdist(node_emb_norm, cluster_emb_norm)
    #     closest_clusters = np.argmin(distances, axis=1)
        
    #     # Draw lines for a subset (10%) of nodes to avoid clutter
    #     n_samples = max(1, int(node_emb_norm.shape[0] * 0.1))
    #     idx_sample = np.random.choice(node_emb_norm.shape[0], 
    #                                  size=min(n_samples, node_emb_norm.shape[0]), 
    #                                  replace=False)
        
    #     for idx in idx_sample:
    #         cluster_idx = closest_clusters[idx]
    #         plt.plot([node_emb_norm[idx, 0], cluster_emb_norm[cluster_idx, 0]],
    #                 [node_emb_norm[idx, 1], cluster_emb_norm[cluster_idx, 1]],
    #                 'k-', alpha=0.1)
    
    # Add title and labels
    hierarchy_type = "Full Hierarchy" if full_hierarchy else "Intermediate Hierarchy"
    plt.title(f'Joint Visualization of Node and Cluster Embeddings ({hierarchy_type})', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig