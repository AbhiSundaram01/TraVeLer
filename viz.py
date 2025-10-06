import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx as nx
from matplotlib.lines import Line2D
import logging

# Create a null handler logger for users who don't want to configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def plot_weighted_graph(node_positions, adjacency_matrix, title=None, figsize=(10, 8)):
    """
    Plot a graph with nodes and edges colored/weighted according to adjacency matrix values.
    
    Parameters:
    -----------
    node_positions : torch.Tensor or numpy.ndarray
        Node positions in 2D space, shape (n_nodes, 2)
    adjacency_matrix : torch.Tensor or numpy.ndarray
        Adjacency matrix with edge weights, shape (n_nodes, n_nodes)
    title : str, optional
        Custom title for the plot
    figsize : tuple, optional
        Figure size as (width, height)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Create a figure and axes explicitly
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure inputs are numpy arrays
    if hasattr(node_positions, 'detach'):
        node_positions = node_positions.detach().numpy()
    if hasattr(adjacency_matrix, 'detach'):
        adj_np = adjacency_matrix.detach().numpy()
    else:
        adj_np = adjacency_matrix
    
    # Plot nodes (scatter plot)
    ax.scatter(node_positions[:, 0], node_positions[:, 1], s=50, c='skyblue', 
              alpha=0.7, edgecolor='black', label='Nodes')
    
    # Get edge weights from adjacency matrix
    edge_weights = []
    edge_indices = []
    
    # Find non-zero entries in adjacency matrix (these are the edges)
    for i in range(adj_np.shape[0]):
        for j in range(adj_np.shape[1]):
            if adj_np[i, j] > 0:  # If there's an edge
                edge_weights.append(adj_np[i, j])
                edge_indices.append((i, j))
    
    if edge_weights:  # Only proceed if there are edges
        # Convert to numpy array
        edge_weights = np.array(edge_weights)
        
        # Create colormap for edge weights
        weight_colors = plt.cm.viridis(edge_weights / edge_weights.max())
        
        # Plot edges with colors based on weights from adjacency matrix
        for idx, (i, j) in enumerate(edge_indices):
            x_values = [node_positions[i, 0], node_positions[j, 0]]
            y_values = [node_positions[i, 1], node_positions[j, 1]]
            
            # Plot the edge with color based on weight
            ax.plot(x_values, y_values, color=weight_colors[idx], alpha=0.5, 
                    linewidth=1 + 3 * edge_weights[idx] / edge_weights.max())  # Width reflects weight
        
        # Add colorbar for edge weights
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=edge_weights.max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Edge Weight from Adjacency Matrix')
    
    # Add node indices for a few select nodes
    step = max(1, node_positions.shape[0] // 10)
    for i in range(0, node_positions.shape[0], step):
        ax.text(node_positions[i, 0], node_positions[i, 1], str(i), 
                fontsize=8, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Add title and labels
    if title is None:
        title = f'Graph with edges colored by adjacency matrix weights'
    ax.set_title(title)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    # Add some stats in a text box
    if edge_weights:
        stats_text = (f"Nodes: {node_positions.shape[0]}\n"
                     f"Edges: {len(edge_weights)}\n"
                     f"Avg Edge Weight: {edge_weights.mean():.3f}\n"
                     f"Max Edge Weight: {edge_weights.max():.3f}")
    else:
        stats_text = f"Nodes: {node_positions.shape[0]}\nNo edges found"
    
    ax.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
              bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
              va='top')
    
    plt.tight_layout()
    return fig, ax

def visualize_mst(node_positions, chain_fixed, show_degree_dist=True, figsize_mst=(12, 10), figsize_dist=(12, 10)):
    """
    Visualize the Minimum Spanning Tree of a graph derived from node positions and chain data.
    
    Parameters:
    -----------
    node_positions : torch.Tensor or numpy.ndarray
        Positions of nodes in space, shape (n_nodes, dim)
    chain_fixed : torch.Tensor
        Chain data representing edges, shape (n_edges, 2, dim)
    show_degree_dist : bool, optional
        Whether to show the degree distribution plot (default: True)
    figsize_mst : tuple, optional
        Figure size for the MST plot (default: (12, 10))
    figsize_dist : tuple, optional
        Figure size for the degree distribution plot (default: (12, 10))
        
    Returns:
    --------
    G : networkx.Graph
        The original graph
    mst : networkx.Graph
        The minimum spanning tree graph
    """
    # Convert tensors to numpy arrays if needed
    if hasattr(node_positions, 'detach'):
        node_positions = node_positions.detach().numpy()
    if hasattr(chain_fixed, 'detach'):
        lines = chain_fixed.detach().numpy()
    else:
        lines = chain_fixed
    
    # Create a figure
    fig, ax = plt.subplots(figsize=figsize_mst)

    # Create a networkx graph from the adjacency matrix and node positions
    G = nx.Graph()

    # Add all nodes with positions
    for i in range(len(node_positions)):
        G.add_node(i, pos=node_positions[i])

    # Add all edges with weights based on Euclidean distance
    edge_list = []

    # First, map coordinates back to node indices
    coord_to_node = {}
    for i, pos in enumerate(node_positions):
        coord_to_node[tuple(pos)] = i

    # Then add edges with weights
    for line in lines:
        point1, point2 = line
        
        # Find closest nodes to these points
        node1 = coord_to_node[tuple(point1)]
        node2 = coord_to_node[tuple(point2)]
        
        # Add edge with weight = distance
        weight = np.linalg.norm(point2 - point1)
        G.add_edge(node1, node2, weight=weight)
        edge_list.append((node1, node2, weight))

    # Find the minimum spanning tree
    mst = nx.minimum_spanning_tree(G)
    print(f"Original graph: {len(G.edges)} edges")
    print(f"MST: {len(mst.edges)} edges")

    # Get positions for all nodes
    pos = nx.get_node_attributes(G, 'pos')

    # Plot all nodes
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color='skyblue', 
                          alpha=0.8, edgecolors='black')

    # Plot MST edges with colors based on weights
    weights = [G[u][v]['weight'] for u, v in mst.edges()]
    cmap = plt.cm.plasma_r
    norm = plt.Normalize(min(weights), max(weights))

    # Draw MST edges with varying colors and widths
    for u, v, w in [(u, v, G[u][v]['weight']) for u, v in mst.edges()]:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1+3*w/max(weights),
                              alpha=0.7, edge_color=[cmap(norm(w))], style='solid')

    # Add node labels to selected nodes
    step = max(1, len(node_positions) // 15)  # Show about 15 node labels
    node_labels = {i: str(i) for i in range(0, len(node_positions), step)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8,
                           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    # Add colorbar for edge weights
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Edge Weight (Distance)')

    # Add stats to the plot
    stats_text = (f"Nodes: {len(mst.nodes)}\n"
                 f"MST Edges: {len(mst.edges)}\n"
                 f"Original Edges: {len(G.edges)}\n"
                 f"Total MST Weight: {sum(weights):.2f}")

    plt.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            va='top', fontsize=10)

    plt.title('Minimum Spanning Tree of DiffPool Graph')
    plt.axis('off')  # Hide axes
    plt.tight_layout()
    plt.show()

    # Optional: Visualize the MST structure
    if show_degree_dist:
        plt.figure(figsize=figsize_dist)
        degrees = [d for n, d in mst.degree()]
        plt.hist(degrees, bins=range(max(degrees)+2), alpha=0.7)
        plt.xlabel('Node Degree')
        plt.ylabel('Count')
        plt.title('MST Degree Distribution')
        plt.xticks(range(max(degrees)+1))
        plt.grid(alpha=0.3)
        plt.show()
        
    return G, mst

def visualize_edge_contributions(chain, edge_sums):
    """Visualize how much each edge contributes to the total integration"""
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract node positions (we'll extract them from the chain)
    all_points = chain.reshape(-1, chain.shape[2]).detach().numpy()
    node_positions = np.unique(all_points, axis=0)
    
    # Plot edges with color based on their contribution
    lines = chain.detach().numpy()
    
    # Normalize edge contributions for color mapping
    min_sum = min(edge_sums)
    max_sum = max(edge_sums)
    norm = plt.Normalize(min_sum, max_sum)
    cmap = plt.cm.coolwarm  # Blue for negative, red for positive
    
    # Plot edges with their contribution colors
    for i, (line, edge_sum) in enumerate(zip(lines, edge_sums)):
        point1, point2 = line
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        
        # Line width proportional to absolute contribution
        width = 1 + 3 * abs(edge_sum) / max(abs(min_sum), abs(max_sum))
        
        # Color based on sign and magnitude
        ax.plot(x_values, y_values, linewidth=width, 
                color=cmap(norm(edge_sum)), alpha=0.7)
    
    # Plot nodes
    for point in node_positions:
        ax.scatter(point[0], point[1], color='black', s=30)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Edge Contribution to Integration')
    
    plt.title("Edge-wise Integration Contributions")
    plt.tight_layout()
    plt.show()

def plot_fixed_chain_structure(chain_fixed_weighted, x_out_fixed):
    """
    Plot the fixed chain structure from DiffPool.

    Parameters:
    -----------
    chain_fixed_weighted : torch.Tensor
        The fixed chain structure tensor of shape [num_edges, 2, feature_dim]
    x_out_fixed : torch.Tensor
        The node positions tensor of shape [num_nodes, feature_dim]
    """
    # Create a figure and axes explicitly
    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract node positions from x_out_fixed
    node_positions = x_out_fixed.detach().numpy()

    # Plot nodes (scatter plot)
    ax.scatter(node_positions[:, 0], node_positions[:, 1], s=50, c='skyblue', 
              alpha=0.7, edgecolor='black', label='Nodes')

    # Convert chain_fixed to numpy for plotting
    lines = chain_fixed_weighted.detach().numpy()

    # Create a colormap for the edges based on their strength or other properties
    edge_strengths = np.linalg.norm(lines[:, 1, :] - lines[:, 0, :], axis=1)
    edge_colors = plt.cm.viridis(edge_strengths / edge_strengths.max())

    # Plot edges (lines) with color based on strength
    for i, line in enumerate(lines):
        point1, point2 = line
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        
        # Plot the line with color based on edge strength
        ax.plot(x_values, y_values, color=edge_colors[i], alpha=0.5)

    # Add node indices for a few select nodes
    step = max(1, node_positions.shape[0] // 10)  # Show about 10 node labels
    for i in range(0, node_positions.shape[0], step):
        ax.text(node_positions[i, 0], node_positions[i, 1], str(i), 
                fontsize=8, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    # Add title and labels
    ax.set_title(f'Fixed Chain Structure from DiffPool (Nodes: {node_positions.shape[0]}, Edges: {lines.shape[0]})')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')

    # Add colorbar for edge strengths - fixed by using fig.colorbar() with ax parameter
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=edge_strengths.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Edge Length')

    # Add some stats in a text box
    stats_text = (f"Nodes: {node_positions.shape[0]}\n"
                 f"Edges: {lines.shape[0]}\n"
                 f"Avg Edge Length: {edge_strengths.mean():.3f}\n"
                 f"Edge Density: {lines.shape[0]/(node_positions.shape[0]*(node_positions.shape[0]-1)/2):.3%}")
                 
    ax.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
              bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
              va='top')

    plt.tight_layout()
    plt.show()

def plot_component_vector_field1(f, ax, comp=0, x_range=5, y_range=5):
    """Plot a component of a vector field given by a function f: R^2 -> R^2.

    Parameters
    ----------
    f : a Pytorch Sequential object
        A function f: R^2 -> R^2, represented as a Pytorch Sequential object

    ax : matplotlib axis object
        The axis on which to plot the vector field

    comp : int
        The component of the vector field to plot

    x_range : float or tuple
        If float: the symmetric range (-x_range, x_range)
        If tuple: the explicit (min, max) range for x values 

    y_range : float or tuple
        If float: the symmetric range (-y_range, y_range)
        If tuple: the explicit (min, max) range for y values

    Returns
    -------
    None
    """
    # Handle different range formats
    if isinstance(x_range, tuple):
        x_min, x_max = x_range
    else:
        x_min, x_max = -x_range, x_range
        
    if isinstance(y_range, tuple):
        y_min, y_max = y_range
    else:
        y_min, y_max = -y_range, y_range
    
    # Create grid
    x = np.linspace(x_min, x_max, 20)
    y = np.linspace(y_min, y_max, 20)
    X, Y = np.meshgrid(x, y)

    X = torch.tensor(X).double()
    Y = torch.tensor(Y).double()

    U = np.zeros((20, 20))
    V = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            inp = np.array([X[i, j], Y[i, j]])
            inp = torch.tensor(inp).float()

            tv = f.forward(inp).reshape(2, -1)

            U[i, j] = tv[:, comp][0]
            V[i, j] = tv[:, comp][1]
    ax.quiver(X, Y, U, V)

def plot_vector_field_components_with_edges(vf, chain, x_out, num_cols=3, save_to_file=False, filepath=None, custom_logger=None):
    """
    Plot each vector field component with its corresponding edge.
    
    Parameters:
    -----------
    vf : NeuralOneForm or Sequential
        The vector field model
    chain : torch.Tensor
        The chain tensor with shape [num_edges, 2, feature_dim]
    x_out : torch.Tensor
        Node positions from DiffPool
    num_cols : int
        Number of columns in the subplot grid
    save_to_file : bool
        Whether to save the plot to a file (True) or display it (False)
    filepath : str
        Path where to save the plot (only used if save_to_file is True)
    custom_logger : logging.Logger, optional
        Custom logger to use. If None, uses module-level logger or print
    """
    # Use provided logger or fall back to module logger
    log = custom_logger or logger
    
    num_components = chain.size(0)
    num_rows = (num_components + num_cols - 1) // num_cols
    
    # Limit the number of components to visualize if there are too many
    max_components = min(num_components, 30)  # Limit to 30 components max
    if max_components < num_components:
        if save_to_file:
            logger.info(f"Limiting visualization to {max_components} of {num_components} components")
        else:
            print(f"Limiting visualization to {max_components} of {num_components} components")
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 5*num_rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Define plotting range based on node positions
    x_min, x_max = x_out[:, 0].min().item(), x_out[:, 0].max().item()
    y_min, y_max = x_out[:, 1].min().item(), x_out[:, 1].max().item()
    
    # Add padding
    padding = 0.1 * max(x_max - x_min, y_max - y_min)
    x_range = (x_min - padding, x_max + padding)
    y_range = (y_min - padding, y_max + padding)
    
    # Get node positions as numpy array
    node_positions = x_out.detach().numpy()
    lines = chain.detach().numpy()
    
    # Plot each component (limited by max_components)
    for i in range(min(max_components, len(axes))):
        ax = axes[i]
        
        # Plot vector field component
        plot_component_vector_field1(vf, ax, comp=i, x_range=x_range, y_range=y_range)
        
        # Plot nodes
        ax.scatter(node_positions[:, 0], node_positions[:, 1], s=30, 
                  color='lightgray', alpha=0.5, edgecolor='black')
        
        # Highlight the source and target nodes of this edge
        source, target = lines[i]
        ax.scatter([source[0], target[0]], [source[1], target[1]], s=80, 
                  color=['green', 'red'], edgecolor='black', zorder=5)
        
        # Plot the edge
        ax.plot([source[0], target[0]], [source[1], target[1]], 
                color='purple', linewidth=3, alpha=0.8, zorder=4)
        
        # Add labels
        ax.text(source[0], source[1], "Source", fontsize=10, ha='center', va='bottom',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        ax.text(target[0], target[1], "Target", fontsize=10, ha='center', va='bottom',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        ax.set_title(f'Component {i}')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
    
    # Hide any unused subplots
    for i in range(min(max_components, len(axes)), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Vector Field Components with Their Corresponding Edges', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save or display the plot
    if save_to_file and filepath:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved component visualization to {filepath}")
    else:
        plt.show()

def plot_combined_vector_field_with_mst(vf, chain, x_out, save_to_file=False, filepath=None, custom_logger=None):
    """
    Plot a combined vector field with all components visualized together.
    
    Parameters:
    -----------
    vf : NeuralOneForm or Sequential
        The vector field model
    chain : torch.Tensor
        The chain tensor with shape [num_edges, 2, feature_dim]
    x_out : torch.Tensor
        Node positions from DiffPool
    save_to_file : bool
        Whether to save the plot to a file (True) or display it (False)
    filepath : str
        Path where to save the plot (only used if save_to_file is True)
    custom_logger : logging.Logger, optional
        Custom logger to use. If None, uses module-level logger or print
    """
    # Use provided logger or fall back to module logger
    log = custom_logger or logger
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Extract node positions
    node_positions = x_out.detach().numpy()
    
    # Define plotting range
    x_min, x_max = node_positions[:, 0].min(), node_positions[:, 0].max()
    y_min, y_max = node_positions[:, 1].min(), node_positions[:, 1].max()
    
    # Add padding
    padding = 0.05 * max(x_max - x_min, y_max - y_min)
    x_range = (x_min - padding, x_max + padding)
    y_range = (y_min - padding, y_max + padding)
    
    # Create grid for vector field
    x = np.linspace(x_range[0], x_range[1], 30)
    y = np.linspace(y_range[0], y_range[1], 30)
    X, Y = np.meshgrid(x, y)
    
    # Initialize combined vector field
    U_combined = np.zeros_like(X)
    V_combined = np.zeros_like(Y)
    
    # Number of components
    num_components = chain.size(0)
    
    # For each grid point, evaluate and combine all vector field components
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            inp = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float)
            
            # Get vector field output for all components
            output = vf.forward(inp).reshape(2, -1)
            
            # Sum all components to get combined vector
            U_combined[i, j] = output[0, :].sum().item()
            V_combined[i, j] = output[1, :].sum().item()
    
    # Normalize vector magnitude for better visualization
    magnitudes = np.sqrt(U_combined**2 + V_combined**2)
    max_mag = np.max(magnitudes)
    if max_mag > 0:  # Avoid division by zero
        scale_factor = 1.0 / max_mag  # Normalize to unit length
        U_combined *= scale_factor
        V_combined *= scale_factor
    
    # Plot the combined vector field
    quiver = ax.quiver(X, Y, U_combined, V_combined, 
                      color='royalblue', alpha=0.7,
                      scale=25, width=0.003, headwidth=5)
    
    # Create the MST
    # Create a networkx graph from the chain data
    G = nx.Graph()
    
    # Add all nodes with positions
    for i in range(len(node_positions)):
        G.add_node(i, pos=node_positions[i])
    
    # Add edges from chain
    lines = chain.detach().numpy()
    for line in lines:
        point1, point2 = line
        
        # Find closest nodes to these points
        node1_idx = ((node_positions - point1) ** 2).sum(axis=1).argmin()
        node2_idx = ((node_positions - point2) ** 2).sum(axis=1).argmin()
        
        # Add edge with weight = distance
        weight = np.linalg.norm(point2 - point1)
        G.add_edge(node1_idx, node2_idx, weight=weight)
    
    # Find the minimum spanning tree
    mst = nx.minimum_spanning_tree(G)
    log.info(f"MST visualization: Original graph: {len(G.edges)} edges, MST: {len(mst.edges)} edges")
    
    if save_to_file:
        logger.info(f"MST visualization: Original graph: {len(G.edges)} edges, MST: {len(mst.edges)} edges")
    else:
        print(f"MST visualization: Original graph: {len(G.edges)} edges, MST: {len(mst.edges)} edges")
    
    # Plot MST edges with colors based on weights
    weights = [G[u][v]['weight'] for u, v in mst.edges()]
    cmap = plt.cm.plasma_r
    norm = plt.Normalize(min(weights), max(weights))
    
    for u, v, w in [(u, v, G[u][v]['weight']) for u, v in mst.edges()]:
        edge_x = [node_positions[u][0], node_positions[v][0]]
        edge_y = [node_positions[u][1], node_positions[v][1]]
        ax.plot(edge_x, edge_y, color=cmap(norm(w)), 
                linewidth=1.5+3*w/max(weights), alpha=0.8,
                solid_capstyle='round', zorder=1)
    
    # Plot nodes
    scatter = ax.scatter(node_positions[:, 0], node_positions[:, 1], 
                        s=80, c='skyblue', alpha=0.8, 
                        edgecolor='black', linewidth=1, zorder=2,
                        label='Nodes')
    
    # Add node labels to selected nodes
    step = max(1, len(node_positions) // 15)
    for i in range(0, len(node_positions), step):
        ax.text(node_positions[i, 0], node_positions[i, 1], str(i), 
                fontsize=8, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
                zorder=3)
    
    # Add colorbar for edge weights
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Edge Weight (Distance)')
    
    # Add stats to the plot
    stats_text = (f"Nodes: {len(mst.nodes)}\n"
                 f"MST Edges: {len(mst.edges)}\n"
                 f"Total MST Weight: {sum(weights):.2f}")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            va='top', fontsize=10)
    
    # Add legend for quiver
    ax.quiverkey(quiver, 0.9, 0.95, 1.0, "Combined Vector Field", 
                labelpos='E', coordinates='figure')
    
    # Add a title and labels
    ax.set_title('Combined Vector Field with Minimum Spanning Tree', fontsize=14)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    # Create custom legend entry for vector field
    legend_elements = [Line2D([0], [0], marker='>', color='blue', linestyle='None',
                             markersize=10, label='Vector Field')]
    legend_elements.append(Line2D([0], [0], color=cmap(0.5), lw=4, label='MST Edge'))
    legend_elements.append(scatter)
    
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    # Save or display the plot
    if save_to_file and filepath:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved combined vector field visualization to {filepath}")
    else:
        plt.show()

def plot_pca_variance(embeddings, title="PCA Explained Variance", n_components=None, 
                      figsize=(10, 6), show_cumulative=True, save_path=None, highlight_threshold=None):
    """
    Plot the explained variance ratio for PCA components of embedding data.
    
    Args:
        embeddings: Array-like embeddings data or pre-computed PCA object
        title: Title for the plot
        n_components: Number of components to display (default: all)
        figsize: Figure size tuple (width, height)
        show_cumulative: Whether to overlay cumulative explained variance
        save_path: If provided, save figure to this path
        highlight_threshold: Optional threshold (e.g., 0.9) to highlight with a horizontal line
        
    Returns:
        fig: The matplotlib figure object
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    # Determine if input is already a PCA object
    if isinstance(embeddings, PCA):
        pca = embeddings
    else:
        # Run PCA on the embeddings
        pca = PCA()
        pca.fit(embeddings)
    
    # Get explained variance ratios
    explained_var = pca.explained_variance_ratio_
    
    # Limit number of components to display if specified
    if n_components is None:
        n_components = len(explained_var)
    else:
        n_components = min(n_components, len(explained_var))
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate x positions and component values to display
    x_pos = np.arange(1, n_components + 1)
    explained_var_subset = explained_var[:n_components]
    
    # Bar plot for individual components
    bars = ax.bar(x_pos, explained_var_subset * 100, 
                  color='royalblue', alpha=0.7, width=0.6)
    
    # Highlight top component in a different color
    if len(bars) > 0:
        bars[0].set_color('firebrick')
        bars[0].set_alpha(0.9)
    
    # Add cumulative variance line
    if show_cumulative:
        cumulative = np.cumsum(explained_var_subset)
        ax.step(x_pos, cumulative * 100, where='mid', 
                color='darkred', alpha=0.7, linewidth=2,
                label=f'Cumulative ({cumulative[-1]*100:.1f}%)')
        
        # Add horizontal highlight line if specified
        if highlight_threshold is not None:
            ax.axhline(y=highlight_threshold * 100, color='green', 
                      linestyle='--', alpha=0.7, 
                      label=f'{highlight_threshold*100}% Explained')
            
            # Find where cumulative crosses the threshold
            threshold_idx = np.where(cumulative >= highlight_threshold)[0]
            if len(threshold_idx) > 0:
                first_idx = threshold_idx[0]
                ax.axvline(x=first_idx + 1, color='green', linestyle=':', alpha=0.7)
                ax.text(first_idx + 1.1, 50, 
                        f'{first_idx + 1} components\n≥ {highlight_threshold*100}%', 
                        verticalalignment='center')
    
    # Annotate top component percentage
    if len(explained_var_subset) > 0:
        ax.text(1, explained_var_subset[0] * 100 + 3, 
                f"{explained_var_subset[0]*100:.1f}%", 
                ha='center', fontweight='bold')
        
        # Annotate interesting secondary components
        for i in range(1, min(3, len(explained_var_subset))):
            if explained_var_subset[i] > 0.05:  # Only annotate if >5%
                ax.text(i+1, explained_var_subset[i] * 100 + 1, 
                        f"{explained_var_subset[i]*100:.1f}%", 
                        ha='center')
    
    # Titles and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', fontsize=12)
    
    # Add linearity metric in the top right
    if len(explained_var_subset) > 1:
        linearity = explained_var_subset[0] / (explained_var_subset[1] + 1e-8)
        ax.text(0.95, 0.95, f"Linearity: {linearity:.1f}x", 
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    # X-axis formatting
    if n_components <= 20:
        # For fewer components, show all ticks
        ax.set_xticks(x_pos)
    else:
        # For many components, show a selection
        step = max(1, n_components // 10)
        ax.set_xticks(np.arange(1, n_components + 1, step))
    
    # Add legend if needed
    if show_cumulative or highlight_threshold is not None:
        ax.legend(loc='upper right')
        
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Tighten layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_single_component_vector_field(vf, chain, x_out, save_to_file=False, filepath=None, custom_logger=None):
    """
    Plot a vector field with only one component (c=1) against multiple edges.
    
    Parameters:
    -----------
    vf : NeuralOneForm or Sequential
        The vector field model with a single component
    chain : torch.Tensor
        The chain tensor with shape [num_edges, 2, 2]
    x_out : torch.Tensor
        Node positions from DiffPool
    save_to_file : bool
        Whether to save the plot to a file (True) or display it (False)
    filepath : str
        Path where to save the plot (only used if save_to_file is True)
    custom_logger : logging.Logger, optional
        Custom logger to use. If None, uses module-level logger or print
    """
    # Use provided logger or create a basic one
    log = custom_logger or logging.getLogger(__name__)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Extract node positions
    node_positions = x_out.detach().numpy()
    
    # Define plotting range
    x_min, x_max = node_positions[:, 0].min(), node_positions[:, 0].max()
    y_min, y_max = node_positions[:, 1].min(), node_positions[:, 1].max()
    
    # Add padding
    padding = 0.1 * max(x_max - x_min, y_max - y_min)
    x_range = (x_min - padding, x_max + padding)
    y_range = (y_min - padding, y_max + padding)
    
    # Create grid for vector field
    x = np.linspace(x_range[0], x_range[1], 30)
    y = np.linspace(y_range[0], y_range[1], 30)
    X, Y = np.meshgrid(x, y)
    
    # Initialize vector field arrays
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    # For each grid point, evaluate the single vector field component
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            inp = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float)
            
            # For c=1, the vector field output shape is [1, 2]
            output = vf(inp).reshape(2, -1)
            
            # Extract the x and y components (single component case)
            U[i, j] = output[0, 0].item()  # x-component
            V[i, j] = output[1, 0].item()  # y-component
    
    # Normalize vector magnitude for better visualization
    magnitudes = np.sqrt(U**2 + V**2)
    max_mag = np.max(magnitudes)
    if max_mag > 0:  # Avoid division by zero
        scale_factor = 1.0 / max_mag  # Normalize to unit length
        U *= scale_factor
        V *= scale_factor
    
    # Plot the vector field
    quiver = ax.quiver(X, Y, U, V, 
                      color='royalblue', alpha=0.7,
                      scale=25, width=0.003, headwidth=5)
    
    # Plot all edges from chain
    lines = chain.detach().numpy()
    for i, line in enumerate(lines):
        point1, point2 = line
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 
               color='red', linewidth=1.5, alpha=0.8,
               solid_capstyle='round', zorder=1)
    
    # Plot nodes
    scatter = ax.scatter(node_positions[:, 0], node_positions[:, 1], 
                        s=80, c='skyblue', alpha=0.8, 
                        edgecolor='black', linewidth=1, zorder=2,
                        label='Nodes')
    
    # Add node labels to select nodes
    step = max(1, len(node_positions) // 15)
    for i in range(0, len(node_positions), step):
        ax.text(node_positions[i, 0], node_positions[i, 1], str(i), 
                fontsize=8, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
                zorder=3)
    
    # Add stats to the plot
    stats_text = (f"Nodes: {len(node_positions)}\n"
                 f"Edges: {len(lines)}\n"
                 f"Vector Field: Single Component")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            va='top', fontsize=10)
    
    # Add legend for quiver
    ax.quiverkey(quiver, 0.9, 0.95, 1.0, "Single Component Vector Field", 
                labelpos='E', coordinates='figure')
    
    # Add title and labels
    ax.set_title('Single-Component Vector Field with Tree Structure', fontsize=14)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    # Create custom legend entry for vector field
    legend_elements = [Line2D([0], [0], marker='>', color='blue', linestyle='None',
                             markersize=10, label='Vector Field')]
    legend_elements.append(Line2D([0], [0], color='red', lw=2, label='Edge'))
    legend_elements.append(scatter)
    
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    # Save or display the plot
    if save_to_file and filepath:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log.info(f"Saved vector field visualization to {filepath}")
    else:
        plt.show()
        
    return fig, ax