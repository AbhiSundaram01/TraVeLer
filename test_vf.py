import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from datetime import datetime
from neural_k_forms.forms import NeuralOneForm
from neural_k_forms.chains import generate_integration_matrix
from viz import plot_combined_vector_field_with_mst

def setup_experiment():
    """Set up experiment directories and logging"""
    # Create a unique run ID based on timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/synthetic_tree_{run_id}"
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
    logger.info(f"Starting synthetic tree vector field test run {run_id}")
    
    return run_id, run_dir, logger

def generate_straight_line(length=1.0, seed=42):
    """
    Generate a simple straight line to test vector field learning.
    
    Args:
        length: Length of the line
        seed: Random seed for reproducibility
        
    Returns:
        coords: Node coordinates (2, 2)
        edges: Edge connections (1, 2)
        chain: Chain representation for vector field (1, 2, 2)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create a straight horizontal line
    coords = np.array([
        [-length/2, 0.0],  # Start point
        [length/2, 0.0]    # End point
    ], dtype=np.float32)
    
    # Single edge connecting the two points
    edges = np.array([[0, 1]], dtype=np.int64)
    
    # Create chain representation for the vector field
    chain = np.zeros((1, 2, 2), dtype=np.float32)
    chain[0, 0] = coords[0]  # Start point
    chain[0, 1] = coords[1]  # End point
    
    # Convert to torch tensors
    coords = torch.tensor(coords, dtype=torch.float32)
    edges = torch.tensor(edges, dtype=torch.long)
    chain = torch.tensor(chain, dtype=torch.float32)
    
    return coords, edges, chain

def test_simple_line():
    """Test function to run vector field learning on a simple line"""
    # Setup experiment
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/simple_line_{run_id}"
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
    logger.info(f"Starting simple line vector field test run {run_id}")
    
    # Generate simple straight line
    logger.info("Generating simple straight line...")
    coords, edges, chain = generate_straight_line(length=1.0, seed=42)
    logger.info(f"Generated line with {len(coords)} nodes and {len(edges)} edges")
    
    # Visualize line structure
    visualize_tree_structure(coords, edges, run_dir)
    
    # Setup vector field (only 1 edge/component)
    logger.info("Setting up vector field model...")
    vf = setup_vector_field(num_edges=1)
    
    # Train vector field with data augmentation (fewer parallels for this simple case)
    logger.info("Training vector field with augmentation...")
    losses = train_vector_field_with_augmentation(
        vf, chain, coords, run_dir, logger,
        epochs=500, num_parallels=10, offset_range=0.1
    )
    
    # Plot training loss
    plot_training_loss(losses, run_dir)
    
    # Analyze vector field
    logger.info("Analyzing trained vector field...")
    analyze_vector_field(vf, chain, coords, run_dir, logger)
    
    logger.info(f"Simple line experiment completed. Results saved to {run_dir}")
    return vf, chain, coords, run_dir

def generate_simple_tree(seed=42):
    """
    Generate a simple tree with one bifurcation point.
    
    Structure:
        A --- B --- C
              |
              D
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        coords: Node coordinates (4, 2)
        edges: Edge connections (3, 2)
        chain: Chain representation for vector field (3, 2, 2)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create coordinates for 4 nodes
    coords = np.array([
        [-1.0, 0.0],  # A
        [0.0, 0.0],   # B (bifurcation point)
        [1.0, 0.0],   # C
        [0.0, -1.0]   # D
    ], dtype=np.float32)
    
    # Create 3 edges: A-B, B-C, B-D
    edges = np.array([
        [0, 1],  # A-B
        [1, 2],  # B-C
        [1, 3]   # B-D
    ], dtype=np.int64)
    
    # Create chain representation for the vector field
    chain = np.zeros((3, 2, 2), dtype=np.float32)
    for i, (start_idx, end_idx) in enumerate(edges):
        chain[i, 0] = coords[start_idx]  # Start point
        chain[i, 1] = coords[end_idx]    # End point
    
    # Convert to torch tensors
    coords = torch.tensor(coords, dtype=torch.float32)
    edges = torch.tensor(edges, dtype=torch.long)
    chain = torch.tensor(chain, dtype=torch.float32)
    
    return coords, edges, chain

def test_simple_tree():
    """Test function to run vector field learning on a simple tree with one bifurcation"""
    # Setup experiment
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/simple_tree_{run_id}"
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
    logger.info(f"Starting simple tree vector field test run {run_id}")
    
    # Generate simple tree
    logger.info("Generating simple tree with one bifurcation...")
    coords, edges, chain = generate_simple_tree(seed=42)
    logger.info(f"Generated tree with {len(coords)} nodes and {len(edges)} edges")
    
    # Visualize tree structure
    visualize_tree_structure(coords, edges, run_dir)
    
    # Setup vector field
    logger.info("Setting up vector field model...")
    vf = setup_vector_field(num_edges=len(edges))
    
    # Train vector field with data augmentation
    logger.info("Training vector field with augmentation...")
    losses = train_vector_field_with_augmentation(
        vf, chain, coords, run_dir, logger,
        epochs=1000, num_parallels=500, offset_range=0.8
    )
    
    # Plot training loss
    plot_training_loss(losses, run_dir)
    
    # Analyze vector field
    logger.info("Analyzing trained vector field...")
    analyze_vector_field(vf, chain, coords, run_dir, logger)

    # Visualize individual vector field components
    visualize_component_vector_fields(vf, chain, coords, run_dir, logger)
    
    logger.info(f"Simple tree experiment completed. Results saved to {run_dir}")
    return vf, chain, coords, run_dir

def test_edge_specific_vector_fields():
    """Test function to train separate vector field models for each edge of a simple tree"""
    # Setup experiment
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/edge_specific_vf_{run_id}"
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
    logger.info(f"Starting edge-specific vector field test run {run_id}")
    
    # Generate simple tree
    logger.info("Generating simple tree with one bifurcation...")
    coords, edges, chain = generate_simple_tree(seed=42)
    logger.info(f"Generated tree with {len(coords)} nodes and {len(edges)} edges")
    
    # Visualize tree structure
    visualize_tree_structure(coords, edges, run_dir)
    
    # Train separate vector field models for each edge
    logger.info("Training separate vector field models for each edge...")
    vf_models, losses_history = train_edge_specific_vector_fields(
        chain, coords, run_dir, logger,
        epochs=1000, num_parallels=500, offset_range=0.1
    )
    
    # Plot combined loss curves
    plt.figure(figsize=(10, 6))
    for i, losses in enumerate(losses_history):
        plt.plot(losses, label=f'Edge {i}')
    
    plt.title('Vector Field Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_dir}/combined_training_losses.png", dpi=300)
    plt.close()
    
    logger.info(f"Edge-specific training completed. Results saved to {run_dir}")
    return vf_models, chain, coords, run_dir

def generate_synthetic_tree(num_nodes=30, seed=42):
    """
    Generate a synthetic tree structure in 2D.
    
    Args:
        num_nodes: Number of nodes in the tree
        seed: Random seed for reproducibility
        
    Returns:
        coords: Node coordinates (num_nodes, 2)
        edges: Edge connections (num_edges, 2)
        chain: Chain representation for vector field (num_edges, 2, 2)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Start with a central node
    coords = [(0.0, 0.0)]
    edges = []
    
    # Generate main branches from center
    num_main_branches = 4
    main_branch_angles = np.linspace(0, 2*np.pi, num_main_branches, endpoint=False)
    
    # Main branch lengths will be between 0.5 and 1.0
    main_branch_lengths = 0.7 + 0.3 * np.random.random(num_main_branches)
    
    # Create main branches
    for i in range(num_main_branches):
        angle = main_branch_angles[i]
        length = main_branch_lengths[i]
        
        # End point of this branch
        end_x = length * np.cos(angle)
        end_y = length * np.sin(angle)
        
        # Add the node and connect it to center
        coords.append((end_x, end_y))
        edges.append((0, i + 1))
    
    # Create sub-branches
    nodes_left = num_nodes - (num_main_branches + 1)
    nodes_per_branch = nodes_left // num_main_branches
    
    for main_branch_idx in range(num_main_branches):
        main_node_idx = main_branch_idx + 1
        main_x, main_y = coords[main_node_idx]
        main_angle = main_branch_angles[main_branch_idx]
        
        # Create sub-branches with some randomness in angle
        for j in range(nodes_per_branch):
            if len(coords) >= num_nodes:
                break
                
            # Sub-branch angle deviates from main branch direction
            sub_angle = main_angle + np.random.uniform(-np.pi/4, np.pi/4)
            sub_length = 0.3 + 0.2 * np.random.random()
            
            sub_x = main_x + sub_length * np.cos(sub_angle)
            sub_y = main_y + sub_length * np.sin(sub_angle)
            
            # Add the node and connect to its parent
            node_idx = len(coords)
            coords.append((sub_x, sub_y))
            edges.append((main_node_idx, node_idx))
    
    # Convert to numpy arrays
    coords = np.array(coords, dtype=np.float32)
    edges = np.array(edges, dtype=np.int64)
    
    # Create chain representation for the vector field
    chain = np.zeros((len(edges), 2, 2), dtype=np.float32)
    for i, (start_idx, end_idx) in enumerate(edges):
        chain[i, 0] = coords[start_idx]  # Start point
        chain[i, 1] = coords[end_idx]    # End point
    
    # Convert to torch tensors
    coords = torch.tensor(coords, dtype=torch.float32)
    edges = torch.tensor(edges, dtype=torch.long)
    chain = torch.tensor(chain, dtype=torch.float32)
    
    return coords, edges, chain

def generate_parallel_edges(chain, num_parallels=5, offset_range=0.1):
    """
    Generate synthetic edges that are parallel to each original edge.
    
    Args:
        chain: Original chain representation (num_edges, 2, 2)
        num_parallels: Number of parallel edges to generate per original edge
        offset_range: Maximum perpendicular distance for parallel edges
        
    Returns:
        augmented_chain: Chain with synthetic parallel edges
    """
    num_edges = chain.size(0)
    parallel_edges = []
    
    for i in range(num_edges):
        # Extract start and end points of the edge
        start_point = chain[i, 0, :]
        end_point = chain[i, 1, :]
        
        # Calculate edge direction vector
        edge_vector = end_point - start_point
        edge_length = torch.norm(edge_vector)
        edge_direction = edge_vector / edge_length
        
        # Calculate perpendicular vector (rotate 90 degrees in 2D)
        perp_vector = torch.tensor([-edge_direction[1], edge_direction[0]], dtype=torch.float32)
        
        # Create parallel edges with various offsets
        for j in range(num_parallels):
            # Calculate offset (both positive and negative)
            offset = (2 * torch.rand(1) - 1) * offset_range
            
            # Create offset vector
            offset_vector = offset * perp_vector
            
            # Create new parallel edge
            new_start = start_point + offset_vector
            new_end = end_point + offset_vector
            
            # Stack to create the edge shape expected by the model
            new_edge = torch.stack([new_start, new_end], dim=0).unsqueeze(0)
            parallel_edges.append(new_edge)
    
    # Concatenate all edges
    parallel_edges = torch.cat(parallel_edges, dim=0)
    return parallel_edges

def visualize_parallel_edges(original_chain, parallel_edges, filepath):
    """
    Visualize original edges and their parallel synthetic edges.
    
    Args:
        original_chain: Original chain representation (num_edges, 2, 2)
        parallel_edges: Synthetic parallel edges
        filepath: Path to save the visualization
    """
    plt.figure(figsize=(12, 12))
    
    # Plot original edges
    for i in range(original_chain.size(0)):
        plt.plot(
            [original_chain[i, 0, 0].item(), original_chain[i, 1, 0].item()],
            [original_chain[i, 0, 1].item(), original_chain[i, 1, 1].item()],
            'r-', lw=2, zorder=2, label='Original Edge' if i == 0 else ""
        )
    
    # Plot synthetic parallel edges
    for i in range(parallel_edges.size(0)):
        plt.plot(
            [parallel_edges[i, 0, 0].item(), parallel_edges[i, 1, 0].item()],
            [parallel_edges[i, 0, 1].item(), parallel_edges[i, 1, 1].item()],
            'g-', lw=0.5, alpha=0.5, zorder=1, label='Synthetic Edge' if i == 0 else ""
        )
    
    plt.title("Original and Synthetic Parallel Edges")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.savefig(filepath, dpi=300)
    plt.close()

def setup_vector_field(num_edges=1):
    """
    Initialize vector field model.
    
    Args:
        num_edges: Number of edges in the tree
        
    Returns:
        vf: NeuralOneForm model
    """
    # Initialize neural vector field (smaller than the one used with DiffPool)
    vf_in = nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 2, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2 * num_edges)
    )
    
    vf = NeuralOneForm(vf_in, input_dim=None, hidden_dim=None, num_cochains=num_edges)
    return vf

def train_vector_field(vf, chain, coords, run_dir, logger, epochs=500):
    """
    Train the vector field to match the synthetic tree structure.
    
    Args:
        vf: NeuralOneForm model
        chain: Chain representation (num_edges, 2, 2)
        coords: Node coordinates (num_nodes, 2)
        run_dir: Directory to save results
        logger: Logger object
        epochs: Number of training epochs
        
    Returns:
        losses: Training loss history
    """
    optimizer = torch.optim.Adam(vf.parameters(), lr=0.001)
    losses = []
    
    # Setup visualization directory
    viz_dir = f"{run_dir}/training_viz"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Initial visualization
    initial_viz_path = f"{viz_dir}/initial_vector_field.png"
    try:
        plot_combined_vector_field_with_mst(
            vf, chain, coords, save_to_file=True, 
            filepath=initial_viz_path, custom_logger=logger,
            # plot_title="Initial Vector Field - Synthetic Tree"
        )
        logger.info(f"Initial vector field visualized at {initial_viz_path}")
    except Exception as e:
        logger.error(f"Error creating initial visualization: {str(e)}")
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Integration matrix
        X = generate_integration_matrix(vf, chain)
        X_sum = torch.sum(X, dim=0)
        
        # Calculate edge lengths and weights
        edge_lengths = torch.norm(chain[:, 1, :] - chain[:, 0, :], dim=1)
        epsilon = 1e-5
        weights = 1.0 / (edge_lengths + epsilon)
        weights = weights / weights.sum() * len(weights)
        
        # Apply weights to the integration results
        L = -(X_sum * weights).sum()
        
        # Compute gradients and update
        L.backward()
        optimizer.step()
        
        losses.append(L.item())
        
        # Log progress
        if (epoch + 1) % 50 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss {L.item():.4f}")
            
        # Visualize intermediate results
        if (epoch + 1) % 100 == 0:
            viz_path = f"{viz_dir}/vector_field_epoch_{epoch+1}.png"
            try:
                plot_combined_vector_field_with_mst(
                    vf, chain, coords, save_to_file=True, 
                    filepath=viz_path, custom_logger=logger,
                    # plot_title=f"Vector Field at Epoch {epoch+1}"
                )
            except Exception as e:
                logger.error(f"Error visualizing epoch {epoch+1}: {str(e)}")
    
    # Save final model
    torch.save({
        'vf_state_dict': vf.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
    }, f"{run_dir}/vf_model_final.pt")
    
    # Final visualization
    final_viz_path = f"{run_dir}/final_vector_field.png"
    try:
        plot_combined_vector_field_with_mst(
            vf, chain, coords, save_to_file=True, 
            filepath=final_viz_path, custom_logger=logger,
            # plot_title="Final Vector Field - Synthetic Tree"
        )
        logger.info(f"Final vector field visualized at {final_viz_path}")
    except Exception as e:
        logger.error(f"Error creating final visualization: {str(e)}")
    
    return losses

def train_vector_field_with_augmentation(vf, chain, coords, run_dir, logger, epochs=500, 
                                        num_parallels=5, offset_range=0.5):
    """
    Train the vector field using data augmentation with parallel edges.
    
    Args:
        vf: NeuralOneForm model
        chain: Chain representation (num_edges, 2, 2)
        coords: Node coordinates (num_nodes, 2)
        run_dir: Directory to save results
        logger: Logger object
        epochs: Number of training epochs
        num_parallels: Number of parallel edges to generate per original edge
        offset_range: Maximum perpendicular distance for parallel edges
        
    Returns:
        losses: Training loss history
    """
    optimizer = torch.optim.Adam(vf.parameters(), lr=0.001)
    losses = []
    
    # Setup visualization directory
    viz_dir = f"{run_dir}/training_viz"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate parallel edges for visualization
    parallel_edges = generate_parallel_edges(chain, num_parallels, offset_range)    
    # Visualize original and parallel edges
    parallel_viz_path = f"{viz_dir}/parallel_edges.png"
    visualize_parallel_edges(chain, parallel_edges, parallel_viz_path)
    logger.info(f"Parallel edges visualization saved to {parallel_viz_path}")
    
    # Initial visualization of vector field
    initial_viz_path = f"{viz_dir}/initial_vector_field.png"
    try:
        plot_combined_vector_field_with_mst(
            vf, chain, coords, save_to_file=True, 
            filepath=initial_viz_path, custom_logger=logger
        )
        logger.info(f"Initial vector field visualized at {initial_viz_path}")
    except Exception as e:
        logger.error(f"Error creating initial visualization: {str(e)}")
    
    # Training loop
    for epoch in range(epochs):
        # Generate new parallel edges each epoch for better augmentation
        if epoch % 2 == 0:
            parallel_edges = generate_parallel_edges(chain, num_parallels, offset_range)
        
        optimizer.zero_grad()
        
        # Create augmented chain by combining original and parallel edges
        num_edges = chain.size(0)
        
        # Create augmented labels tensor that maps each edge to its component index
        # Original edges map to their own component (0->0, 1->1, etc.)
        orig_labels = torch.arange(num_edges)
        
        # Parallel edges map to the same component as their parent edge
        parallel_labels = torch.cat([torch.full((num_parallels,), i, dtype=torch.long) 
                                    for i in range(num_edges)])
        
        # Combine original edges and parallel edges
        augmented_chain = torch.cat([chain, parallel_edges], dim=0)
        edge_labels = torch.cat([orig_labels, parallel_labels])
        
        # # Calculate edge weights based on length (for original edges)
        # edge_lengths_orig = torch.norm(chain[:, 1, :] - chain[:, 0, :], dim=1)
        # epsilon = 1e-5
        # weights_orig = 1.0 / (edge_lengths_orig + epsilon)
        # weights_orig = weights_orig / weights_orig.sum() * len(weights_orig)
        
        # Calculate edge lengths for all edges (both original and parallel)
        edge_lengths_orig = torch.norm(chain[:, 1, :] - chain[:, 0, :], dim=1)
        edge_lengths_parallel = torch.norm(parallel_edges[:, 1, :] - parallel_edges[:, 0, :], dim=1)
        
        # Combine all edge lengths
        edge_lengths_combined = torch.cat([edge_lengths_orig, edge_lengths_parallel])
        
        # Apply the same inverse length weighting strategy to all edges
        epsilon = 1e-5
        weights_combined = 1.0 / (edge_lengths_combined + epsilon)
        weights_combined = weights_combined / weights_combined.sum() * len(weights_combined)
        
        # Calculate integration matrix for combined edges
        X = generate_integration_matrix(vf, augmented_chain)
        
        # For each edge, extract the component that corresponds to its label
        loss = 0
        for i in range(len(augmented_chain)):
            component_idx = edge_labels[i].item()
            edge_score = X[i, component_idx]
            loss -= edge_score * weights_combined[i]
        
        # loss = X.sum().sum()
        
        # Compute gradients and update
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Log progress
        if (epoch + 1) % 50 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss {loss.item():.4f}")
            
        # Visualize intermediate results
        if (epoch + 1) % 100 == 0:
            viz_path = f"{viz_dir}/vector_field_epoch_{epoch+1}.png"
            try:
                plot_combined_vector_field_with_mst(
                    vf, chain, coords, save_to_file=True, 
                    filepath=viz_path, custom_logger=logger
                )
            except Exception as e:
                logger.error(f"Error visualizing epoch {epoch+1}: {str(e)}")
    
    # Save final model
    torch.save({
        'vf_state_dict': vf.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
    }, f"{run_dir}/vf_model_final.pt")
    
    # Final visualization
    final_viz_path = f"{run_dir}/final_vector_field.png"
    try:
        plot_combined_vector_field_with_mst(
            vf, chain, coords, save_to_file=True, 
            filepath=final_viz_path, custom_logger=logger
        )
        logger.info(f"Final vector field visualized at {final_viz_path}")
    except Exception as e:
        logger.error(f"Error creating final visualization: {str(e)}")
    
    return losses

def train_edge_specific_vector_fields(chain, coords, run_dir, logger, epochs=1000, num_parallels=50, offset_range=0.1):
    """
    Train multiple vector field models, one for each edge.
    
    Args:
        chain: Chain representation (num_edges, 2, 2)
        coords: Node coordinates (num_nodes, 2)
        run_dir: Directory to save results
        logger: Logger object
        epochs: Number of training epochs
        num_parallels: Number of parallel edges to generate per original edge
        offset_range: Maximum perpendicular distance for parallel edges
        
    Returns:
        vf_models: List of trained vector field models
        losses_history: Training loss history for each model
    """
    num_edges = chain.size(0)
    vf_models = []
    losses_history = []
    
    # Create a directory to save individual model results
    models_dir = f"{run_dir}/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Train a separate model for each edge
    for edge_idx in range(num_edges):
        logger.info(f"Training vector field model for edge {edge_idx}...")
        
        # Extract the single edge we're focusing on
        edge_chain = chain[edge_idx:edge_idx+1]  # Keep dimensions: (1, 2, 2)
        
        # Setup a vector field model with a single component
        vf = setup_vector_field(num_edges=1)
        
        # Directory for this specific model
        edge_dir = f"{models_dir}/edge_{edge_idx}"
        os.makedirs(edge_dir, exist_ok=True)
        
        # Train the model with data augmentation
        optimizer = torch.optim.Adam(vf.parameters(), lr=0.001)
        losses = []
        
        for epoch in range(epochs):
            # Generate new parallel edges periodically
            if epoch % 2 == 0:
                parallel_edges = generate_parallel_edges(edge_chain, num_parallels, offset_range)
            
            optimizer.zero_grad()
            
            # Combine original edge and its parallel edges
            augmented_chain = torch.cat([edge_chain, parallel_edges], dim=0)
            
            # # Calculate edge lengths for weighting
            # edge_lengths = torch.norm(augmented_chain[:, 1, :] - augmented_chain[:, 0, :], dim=1)
            # epsilon = 1e-5
            # weights = 1.0 / (edge_lengths + epsilon)
            # weights = weights / weights.sum() * len(weights)
            
            # Integration matrix for all edges
            X = generate_integration_matrix(vf, augmented_chain)
            
            # # Since this model has only 1 component, we can directly use the first column
            # component_scores = X[:, 0]
            
            # # Apply weights to the integration results
            # loss = -(component_scores * weights).sum()

            loss = torch.sum(X, dim=0)
            
            # Compute gradients and update
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # Log progress
            if (epoch + 1) % 50 == 0:
                logger.info(f"Edge {edge_idx} - Epoch {epoch+1}/{epochs}: Loss {loss.item():.4f}")
        
        vf_models.append(vf)
        losses_history.append(losses)
        
        # Save this model
        torch.save({
            'vf_state_dict': vf.state_dict(),
            'edge_idx': edge_idx,
            'losses': losses,
        }, f"{edge_dir}/vf_model_final.pt")
        
        # Plot loss curve for this edge
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(f'Edge {edge_idx} Vector Field Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f"{edge_dir}/training_loss.png", dpi=300)
        plt.close()
        
        # Visualize the trained vector field for this edge
        visualize_edge_vector_field(vf, edge_chain, coords, edge_dir, edge_idx, logger)
    
    # Visualize all trained vector fields together
    visualize_all_edge_vector_fields(vf_models, chain, coords, run_dir, logger)
    
    return vf_models, losses_history

def visualize_tree_structure(coords, edges, run_dir):
    """
    Visualize the synthetic tree structure.
    
    Args:
        coords: Node coordinates (num_nodes, 2)
        edges: Edge connections (num_edges, 2)
        run_dir: Directory to save visualization
    """
    plt.figure(figsize=(10, 10))
    
    # Plot nodes
    plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=50, zorder=2)
    
    # Plot edges
    for edge in edges:
        start, end = edge
        plt.plot(
            [coords[start, 0], coords[end, 0]], 
            [coords[start, 1], coords[end, 1]], 
            'k-', lw=1, alpha=0.7, zorder=1
        )
    
    # Highlight root node
    plt.scatter(coords[0, 0], coords[0, 1], c='red', s=100, zorder=3)
    
    plt.title("Synthetic Tree Structure")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(f"{run_dir}/tree_structure.png", dpi=300)
    plt.close()

def plot_training_loss(losses, run_dir):
    """
    Plot training loss curve.
    
    Args:
        losses: List of loss values
        run_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Vector Field Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log')
    plt.tight_layout()
    
    plt.savefig(f"{run_dir}/training_loss.png", dpi=300)
    plt.close()

def analyze_vector_field(vf, chain, coords, run_dir, logger):
    """
    Analyze the learned vector field.
    
    Args:
        vf: Trained NeuralOneForm model
        chain: Chain representation (num_edges, 2, 2)
        coords: Node coordinates (num_nodes, 2)
        run_dir: Directory to save analysis
        logger: Logger object
    """
    # Create grid to evaluate vector field
    grid_min = torch.min(coords, dim=0)[0] - 0.2
    grid_max = torch.max(coords, dim=0)[0] + 0.2
    
    grid_x = torch.linspace(grid_min[0], grid_max[0], 50)
    grid_y = torch.linspace(grid_min[1], grid_max[1], 50)
    grid_X, grid_Y = torch.meshgrid(grid_x, grid_y, indexing='ij')
    
    grid_points = torch.stack([grid_X.flatten(), grid_Y.flatten()], dim=1)
    
    # Evaluate vector field at grid points
    with torch.no_grad():
        # Get vector field components at each point
        vf_values = []
        for point in grid_points:
            # Need to reshape point for VF evaluation
            point_reshaped = point.view(1, 2)
            vf_value = vf(point_reshaped)
            vf_values.append(vf_value)
        
        vf_values = torch.cat(vf_values, dim=0)
        
        # Calculate field strength (magnitude)
        field_strength = torch.norm(vf_values, dim=1).reshape(grid_X.shape)
    
    # Plot vector field strength
    plt.figure(figsize=(10, 10))
    
    # Heatmap of field strength
    plt.pcolormesh(grid_X.numpy(), grid_Y.numpy(), field_strength.numpy(), 
                  cmap='viridis', shading='auto')
    plt.colorbar(label='Vector Field Strength')
    
    # Plot the tree structure on top
    for i in range(len(chain)):
        plt.plot(
            [chain[i, 0, 0].item(), chain[i, 1, 0].item()],
            [chain[i, 0, 1].item(), chain[i, 1, 1].item()],
            'r-', lw=2, alpha=0.8
        )
    
    # Plot nodes
    plt.scatter(coords[:, 0], coords[:, 1], c='white', edgecolors='black', s=50, zorder=3)
    
    plt.title("Vector Field Strength Analysis")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(False)
    
    plt.savefig(f"{run_dir}/vector_field_strength.png", dpi=300)
    plt.close()

def visualize_component_vector_fields(vf, chain, coords, save_dir, logger):
    """
    Visualize each component of the vector field separately.
    
    Args:
        vf: Trained NeuralOneForm model
        chain: Chain representation (num_edges, 2, 2)
        coords: Node coordinates (num_nodes, 2)
        save_dir: Directory to save visualizations
        logger: Logger object
    """
    logger.info("Visualizing individual vector field components...")
    
    # Create directory for component visualizations
    component_dir = f"{save_dir}/components"
    os.makedirs(component_dir, exist_ok=True)
    
    # Define grid for evaluation
    x = np.linspace(-1.5, 1.5, 30)
    y = np.linspace(-1.5, 0.5, 30)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    num_edges = chain.size(0)
    num_grid_points = len(grid_points)
    
    # Evaluate vector field components individually
    vf.eval()
    vector_fields = []
    
    # Process each grid point individually to extract component values
    with torch.no_grad():
        for point in torch.tensor(grid_points, dtype=torch.float32):
            # Process one point at a time and reshape properly
            point_input = point.view(1, 1, 2)
            output = vf(point_input)  # Shape will be [1, num_edges*2]
            
            # Reshape to get components (x,y) for each edge
            output = output.view(1, num_edges, 2)
            vector_fields.append(output)
        
        # Combine all points
        vector_field = torch.cat(vector_fields, dim=0)  # [num_points, num_edges, 2]
    
    fig, axes = plt.subplots(1, num_edges, figsize=(6*num_edges, 6))
    
    # Plot each component
    for i in range(num_edges):
        if num_edges == 1:
            ax = axes
        else:
            ax = axes[i]
            
        # Extract this component's vector field
        vf_component = vector_field[:, i, :].cpu().numpy()
        
        # Reshape to grid
        U = vf_component[:, 0].reshape(X.shape)
        V = vf_component[:, 1].reshape(X.shape)
        
        # Plot vector field
        ax.quiver(X, Y, U, V, alpha=0.8)
        
        # Plot tree structure
        for j in range(num_edges):
            start_point = chain[j, 0, :].cpu().numpy()
            end_point = chain[j, 1, :].cpu().numpy()
            
            # Highlight the current component
            if j == i:
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                        'r-', linewidth=3, label=f'Edge {j}')
            else:
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                        'k-', linewidth=1, alpha=0.5)
        
        # Plot nodes
        ax.scatter(coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), 
                  c='blue', s=50, zorder=5)
        
        # Label nodes
        node_labels = ['A', 'B', 'C', 'D']
        for j, (x, y) in enumerate(coords.cpu().numpy()):
            ax.text(x, y+0.1, node_labels[j], fontsize=12, ha='center')
            
        ax.set_title(f'Component {i} Vector Field')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 0.5])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_aspect('equal')
        
    plt.tight_layout()
    component_path = f"{component_dir}/component_vector_fields.png"
    plt.savefig(component_path, dpi=300)
    plt.close()
    
    # Also create a combined visualization with all components
    plt.figure(figsize=(10, 10))
    
    # Plot tree structure
    for i in range(num_edges):
        start_point = chain[i, 0, :].cpu().numpy()
        end_point = chain[i, 1, :].cpu().numpy()
        plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                'k-', linewidth=2)
    
    # Plot nodes
    plt.scatter(coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), 
               c='blue', s=50, zorder=5)
    
    # Label nodes
    node_labels = ['A', 'B', 'C', 'D']
    for i, (x, y) in enumerate(coords.cpu().numpy()):
        plt.text(x, y+0.1, node_labels[i], fontsize=12, ha='center')
        
    # Plot vector field components with different colors
    colors = ['red', 'green', 'blue']
    for i in range(num_edges):
        # Extract this component's vector field
        vf_component = vector_field[:, i, :].cpu().numpy()
        
        # Reshape to grid
        U = vf_component[:, 0].reshape(X.shape)
        V = vf_component[:, 1].reshape(X.shape)
        
        # Plot vector field with reduced density for clarity
        skip = 2
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  U[::skip, ::skip], V[::skip, ::skip], 
                  color=colors[i], alpha=0.6, label=f'Component {i}')
    
    plt.title('All Vector Field Components')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 0.5])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.axis('equal')
    
    combined_path = f"{component_dir}/combined_components.png"
    plt.savefig(combined_path, dpi=300)
    plt.close()
    
    logger.info(f"Combined component visualization saved at {combined_path}")

def visualize_vector_field_components(vf, chain, coords, run_dir, logger):
    """
    Visualize individual vector field components with their corresponding edges.
    
    Args:
        vf: Trained NeuralOneForm model
        chain: Chain representation (num_edges, 2, 2)
        coords: Node coordinates (num_nodes, 2)
        run_dir: Directory to save visualizations
        logger: Logger object
    """
    logger.info("Creating visualization of individual vector field components...")
    
    # Define the filepath for the vector field components visualization
    components_viz_path = f"{run_dir}/vector_field_components.png"
    
    # Call the visualization function from viz.py
    try:
        from viz import plot_vector_field_components_with_edges
        
        # Adjust number of columns based on total components
        num_components = chain.size(0)
        if num_components <= 3:
            num_cols = num_components
        else:
            num_cols = min(5, num_components)
            
        plot_vector_field_components_with_edges(
            vf, chain, coords,
            num_cols=num_cols,
            save_to_file=True,
            filepath=components_viz_path,
            custom_logger=logger
        )
        logger.info(f"Vector field components visualization saved to {components_viz_path}")
        
        # Also create a detailed view of the first few components
        if num_components > 5:
            # Create detailed visualizations of first 5 components
            detailed_viz_path = f"{run_dir}/vector_field_components_detail.png"
            
            # Use the same function but limit to first 5 components
            # We can do this by creating a smaller chain with just the first 5 components
            detailed_chain = chain[:5]
            
            plot_vector_field_components_with_edges(
                vf, detailed_chain, coords,
                num_cols=3,
                save_to_file=True,
                filepath=detailed_viz_path,
                custom_logger=logger
            )
            logger.info(f"Detailed vector field components visualization saved to {detailed_viz_path}")
            
    except Exception as e:
        logger.error(f"Error visualizing vector field components: {str(e)}")
        logger.exception("Visualization error details:")
    
    # Also create a visualization showing field strength for each component
    try:
        # Create a directory for individual component visualizations
        import os
        components_dir = f"{run_dir}/components"
        os.makedirs(components_dir, exist_ok=True)
        
        # Create grid to evaluate vector field
        grid_min = torch.min(coords, dim=0)[0] - 0.2
        grid_max = torch.max(coords, dim=0)[0] + 0.2
        
        grid_size = 50  # Higher resolution grid for heatmap
        grid_x = torch.linspace(grid_min[0], grid_max[0], grid_size)
        grid_y = torch.linspace(grid_min[1], grid_max[1], grid_size)
        grid_X, grid_Y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        
        grid_points = torch.stack([grid_X.flatten(), grid_Y.flatten()], dim=1)
        
        # Evaluate vector field at grid points
        with torch.no_grad():
            # Process in batches to avoid memory issues
            batch_size = 500
            all_outputs = []
            
            for i in range(0, len(grid_points), batch_size):
                batch = grid_points[i:i+batch_size]
                output = vf(batch)
                all_outputs.append(output)
            
            vf_values = torch.cat(all_outputs, dim=0)
            
            # Reshape to get component-wise outputs
            vf_values = vf_values.reshape(len(grid_points), 2, -1)
            
            # Plot heatmaps for the top 6 components (or fewer if there are less)
            num_to_plot = min(6, chain.size(0))
            
            plt.figure(figsize=(15, 10))
            for i in range(num_to_plot):
                # Calculate field magnitude for this component
                u = vf_values[:, 0, i].reshape(grid_size, grid_size).numpy()
                v = vf_values[:, 1, i].reshape(grid_size, grid_size).numpy()
                magnitude = np.sqrt(u**2 + v**2)
                
                # Plot as heatmap with edges overlaid
                plt.subplot(2, 3, i+1)
                plt.pcolormesh(grid_X.numpy(), grid_Y.numpy(), magnitude, 
                              cmap='viridis', shading='auto')
                plt.colorbar(label='Field Strength')
                
                # Plot the tree structure
                lines = chain.detach().numpy()
                
                # Only highlight this edge
                plt.plot([lines[i, 0, 0], lines[i, 1, 0]], 
                         [lines[i, 0, 1], lines[i, 1, 1]], 
                         'r-', lw=3, alpha=0.8)
                
                # Plot other edges faintly
                for j in range(len(lines)):
                    if j != i:
                        plt.plot([lines[j, 0, 0], lines[j, 1, 0]], 
                                [lines[j, 0, 1], lines[j, 1, 1]], 
                                'k-', lw=0.5, alpha=0.2)
                
                plt.title(f'Component {i} Field Strength')
                plt.axis('equal')
            
            plt.tight_layout()
            plt.savefig(f"{run_dir}/vector_field_strength_by_component.png", dpi=300)
            plt.close()
            
    except Exception as e:
        logger.error(f"Error visualizing component strengths: {str(e)}")
        logger.exception("Visualization error details:")

def visualize_edge_vector_field(vf, edge_chain, coords, edge_dir, edge_idx, logger):
    """
    Visualize the vector field trained for a specific edge.
    """
    try:
        # Create grid for vector field evaluation
        grid_min = torch.min(coords, dim=0)[0] - 0.2
        grid_max = torch.max(coords, dim=0)[0] + 0.2
        
        grid_x = torch.linspace(grid_min[0], grid_max[0], 30)
        grid_y = torch.linspace(grid_min[1], grid_max[1], 30)
        grid_X, grid_Y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        
        grid_points = torch.stack([grid_X.flatten(), grid_Y.flatten()], dim=1)
        
        # Evaluate vector field at grid points
        with torch.no_grad():
            vf_values = vf(grid_points)
            
            # Reshape to grid dimensions
            U = vf_values[:, 0].reshape(grid_X.shape)
            V = vf_values[:, 1].reshape(grid_Y.shape)
        
        # Plot the vector field
        plt.figure(figsize=(10, 10))
        
        # Plot the vector field
        plt.quiver(grid_X.numpy(), grid_Y.numpy(), U.numpy(), V.numpy(), 
                alpha=0.8, scale=50)
        
        # Plot the targeted edge
        plt.plot(
            [edge_chain[0, 0, 0].item(), edge_chain[0, 1, 0].item()],
            [edge_chain[0, 0, 1].item(), edge_chain[0, 1, 1].item()],
            'r-', lw=3, alpha=0.8
        )
        
        # Plot all coordinates
        plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=50, zorder=3)
        
        plt.title(f"Vector Field for Edge {edge_idx}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.savefig(f"{edge_dir}/vector_field.png", dpi=300)
        plt.close()
    except Exception as e:
        logger.error(f"Error visualizing vector field for edge {edge_idx}: {str(e)}")

def visualize_all_edge_vector_fields(vf_models, chain, coords, run_dir, logger):
    """
    Create a combined visualization of all edge-specific vector fields.
    """
    try:
        num_edges = len(vf_models)
        
        # Define a grid for evaluation
        grid_min = torch.min(coords, dim=0)[0] - 0.2
        grid_max = torch.max(coords, dim=0)[0] + 0.2
        
        grid_x = torch.linspace(grid_min[0], grid_max[0], 30)
        grid_y = torch.linspace(grid_min[1], grid_max[1], 30)
        grid_X, grid_Y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        
        grid_points = torch.stack([grid_X.flatten(), grid_Y.flatten()], dim=1)
        
        # Combined visualization with all vector fields
        plt.figure(figsize=(12, 12))
        
        # Plot all edges
        for i in range(num_edges):
            start_point = chain[i, 0, :].cpu().numpy()
            end_point = chain[i, 1, :].cpu().numpy()
            plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                    'k-', linewidth=2)
        
        # Plot nodes
        plt.scatter(coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), 
                    c='blue', s=50, zorder=5)
        
        # Plot vector fields from each model with different colors
        colors = ['red', 'green', 'blue']
        
        for i, vf in enumerate(vf_models):
            with torch.no_grad():
                vf_values = vf(grid_points)
                
                # Reshape to grid dimensions
                U = vf_values[:, 0].reshape(grid_X.shape)
                V = vf_values[:, 1].reshape(grid_Y.shape)
            
            # Use modulo for color selection to handle more than 3 edges
            color = colors[i % len(colors)]
            
            # Plot with reduced density for clarity
            skip = 2
            plt.quiver(grid_X[::skip, ::skip].numpy(), grid_Y[::skip, ::skip].numpy(), 
                    U[::skip, ::skip].numpy(), V[::skip, ::skip].numpy(), 
                    color=color, alpha=0.6, scale=50, label=f'Edge {i} VF')
        
        plt.title("Combined Edge-Specific Vector Fields")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.savefig(f"{run_dir}/combined_edge_vector_fields.png", dpi=300)
        plt.close()
    except Exception as e:
        logger.error(f"Error creating combined visualization: {str(e)}")

def main():
    """Main function to run the experiment"""
    # # Run the simple line test first
    # logger = logging.getLogger()
    # logger.info("Starting with simple straight line test")
    # vf_line, chain_line, coords_line, run_dir_line = test_simple_line()
    # logger.info(f"Simple line test completed. Results in {run_dir_line}")

    # # Run the simple tree test with one bifurcation
    # logger = logging.getLogger()
    # logger.info("Starting with simple tree test (one bifurcation)")
    # vf_tree, chain_tree, coords_tree, run_dir_tree = test_simple_tree()
    # logger.info(f"Simple tree test completed. Results in {run_dir_tree}")

    # Run the edge-specific vector field test
    logger = logging.getLogger()
    logger.info("Starting edge-specific vector field training test")
    vf_models, chain, coords, run_dir = test_edge_specific_vector_fields()
    logger.info(f"Edge-specific vector field test completed. Results in {run_dir}")
    
    # # Optionally continue with the full tree test
    # logger.info("Moving on to full tree test")
    
    # # Setup experiment
    # run_id, run_dir, logger = setup_experiment()
    
    # # Generate synthetic tree
    # logger.info("Generating synthetic tree structure...")
    # coords, edges, chain = generate_synthetic_tree(num_nodes=30, seed=42)
    # logger.info(f"Generated tree with {len(coords)} nodes and {len(edges)} edges")
    
    # # Visualize tree structure
    # visualize_tree_structure(coords, edges, run_dir)
    
    # # Setup vector field
    # logger.info("Setting up vector field model...")
    # vf = setup_vector_field(num_edges=len(edges))
    # # vf = setup_vector_field()
    
    # # # Train vector field
    # # logger.info("Training vector field...")
    # # losses = train_vector_field(vf, chain, coords, run_dir, logger, epochs=500)

    # # Train vector field with data augmentation
    # logger.info("Training vector field with augmentation...")
    # losses = train_vector_field_with_augmentation(
    #     vf, chain, coords, run_dir, logger, 
    #     epochs=500, num_parallels=500, offset_range=0.1
    # )
    
    # # Plot training loss
    # plot_training_loss(losses, run_dir)
    
    # # Analyze vector field
    # logger.info("Analyzing trained vector field...")
    # analyze_vector_field(vf, chain, coords, run_dir, logger)

    # # Visualize vector field components
    # logger.info("Visualizing vector field components...")
    # visualize_vector_field_components(vf, chain, coords, run_dir, logger)
    
    # logger.info(f"Experiment completed. Results saved to {run_dir}")

if __name__ == "__main__":
    main()