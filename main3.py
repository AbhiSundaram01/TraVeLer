import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from datetime import datetime
from dataset import preprocess_pancreas_data
from models import DiffPool
from neural_k_forms.forms import NeuralOneForm
from neural_k_forms.chains import generate_integration_matrix
from graph_utils import create_mst_chain_from_coords, soft_mst_approximation2, convert_to_chain_format
from viz import plot_vector_field_components_with_edges, plot_combined_vector_field_with_mst, plot_single_component_vector_field
from analysis import visualize_joint_embeddings
import networkx as nx
from test_form import prepare_chain_coords
from pathlib import Path
import requests


def setup_experiment():
    """Set up experiment directories and logging"""
    # Create a unique run ID based on timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/run_{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f"{run_dir}/training.log"),
            logging.StreamHandler()  # Also log to console
        ]
    )
    logger = logging.getLogger()
    logger.info(f"Starting training run {run_id}")
    
    return run_id, run_dir, logger


def setup_model(x, adj, logger):
    """Initialize models, vector field and optimizer"""
    model = DiffPool(num_features=x.size(1), max_nodes=x.size(0))
    
    # # Get vector field size based on initial model output
    # with torch.no_grad():
    #     temp_x_out, _ = model(x, adj)
    #     temp_chain = soft_mst_approximation2(temp_x_out, temperature=0.1)
    #     c = temp_chain.size(0)  # number of features/columns in cochain data matrix
    #     logger.info(f"Initializing vector field with {c} components based on sample chain")
    c = 1
    
    # Initialize neural vector field
    vf_in = nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 2, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2 * c)
    )
    
    vf = NeuralOneForm(vf_in, input_dim=None, hidden_dim=None, num_cochains=c)
    model.reset_parameters()
    
    # Create joint optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 0.0001, 'weight_decay': 0.001},
        {'params': vf.parameters(), 'lr': 0.001, 'weight_decay': 0.01}
    ])
    
    return model, vf, optimizer, c


def visualize_initial_state(model, vf, x, adj, run_dir, logger):
    """Visualize the initial model state"""
    logger.info("Creating visualization of initial model state...")
    
    # Generate initial output for visualization
    with torch.no_grad():
        x_out_initial, _ = model(x, adj)
        chain_initial = create_mst_chain_from_coords(x_out_initial)
    
    # Define path for initial visualization
    initial_viz_path = f"{run_dir}/initial_vector_field.png"
    
    # Generate and save/display visualization of the initial state
    if matplotlib.get_backend() == 'agg':  # Non-interactive backend (server)
        try:
            plot_combined_vector_field_with_mst(vf, chain_initial, x_out_initial, 
                                              save_to_file=True, 
                                              filepath=initial_viz_path,
                                              custom_logger=logger)
            logger.info(f"Initial vector field visualization saved to {initial_viz_path}")
        except Exception as e:
            logger.error(f"Error creating initial visualization: {str(e)}")
            logger.exception("Visualization error details:")
    else:
        logger.info("Displaying initial vector field visualization...")
        try:
            plt.figure(figsize=(15, 10))
            plt.title("Initial Vector Field (Before Training)", fontsize=16)
            
            plot_combined_vector_field_with_mst(vf, chain_initial, x_out_initial,
                                              save_to_file=False,
                                              custom_logger=logger)
        except Exception as e:
            logger.error(f"Error displaying initial visualization: {str(e)}")
            logger.exception("Visualization error details:")


def train_model(model, vf, optimizer, x, adj, adata_subsampled, epochs, run_dir, logger):
    """Core training loop"""
    log_interval = 10
    
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Tracking metrics
    losses = []
    grad_norms_vf = []
    grad_norms_model = []
    x_sums = []
    
    # Setup checkpoint directory for embedding visualization
    checkpoint_dir = f"{run_dir}/embedding_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save initial embeddings
    save_embedding_checkpoint(model, x, adj, adata_subsampled, 0, checkpoint_dir, logger)
    
    # Save metrics to CSV
    metrics_file = f"{run_dir}/metrics.csv"
    with open(metrics_file, 'w') as f:
        f.write("epoch,loss,vf_grad_norm,model_grad_norm,x_sum_mean,x_sum_min,x_sum_max\n")

    # # create chain_origin by converting adj to chain
    # chain_origin = convert_to_chain_format(adj, x)
    # logger.info(f"DEBUG - x shape: {x.shape}, chain_origin shape: {chain_origin.shape}")

    # Use NetworkX spring layout for chain_origin
    logger.info("Creating chain_origin using NetworkX spring layout...")
    # Create networkx graph from adjacency matrix
    G = nx.from_numpy_array(adj.numpy())
    
    # Use spring layout to get 2D coordinates
    pos = nx.spring_layout(G, seed=42)
    
    # Convert pos dictionary to numpy array
    layout_coords = np.zeros((len(pos), 2))
    for i, coord in pos.items():
        layout_coords[i] = coord
    
    logger.info(f"Generated layout coordinates with shape: {layout_coords.shape}")
    
    # Create chain_origin from adjacency but use layout_coords for visualization
    chain_origin = convert_to_chain_format(adj, x)
    
    # Convert chain to use layout coordinates
    chain_coords, feature_dim = prepare_chain_coords(chain_origin, layout_coords, logger)
    logger.info(f"DEBUG - x shape: {x.shape}, chain_origin shape: {chain_origin.shape}, chain_coords shape: {chain_coords.shape}")
    
    # Training loop
    for i in range(epochs):
        # Clear all gradients
        optimizer.zero_grad()
        
        # Forward pass through DiffPool model
        x_out, _ = model(x, adj)
        
        # Create chain using minimum spanning tree with soft approximation
        chain = soft_mst_approximation2(x_out, temperature=0.1)
        
        # Debug chain tensor
        if i % log_interval == 0:
            logger.info(f"Epoch {i}: Chain shape: {chain.shape}, Values min/max/mean: {chain.min():.4f}/{chain.max():.4f}/{chain.mean():.4f}")
        
        # Integration matrix
        X = generate_integration_matrix(vf, chain)
        
        if i % log_interval == 0:
            logger.info(f"X stats: shape={X.shape}, min={X.min().item():.4f}, max={X.max().item():.4f}, mean={X.mean().item():.4f}")
            logger.info(f"X non-zero elements: {torch.count_nonzero(X).item()} / {X.numel()}")
        
        X_sum = torch.sum(X, dim=0)
        
        # logger.info(f"DEBUG - About to calculate X_origin, vf input dim: {vf.input_dim}, chain_origin shape: {chain_origin.shape}")
        X_origin = generate_integration_matrix(vf, chain_coords)
        # logger.info(f"DEBUG - X_origin shape: {X_origin.shape}, X shape: {X.shape}")
        X_origin = torch.sum(X_origin, dim=0)

        L_emb = torch.norm(X_origin - X, dim=0).mean()
        # log L_emb every log_interval
        if i % log_interval == 0:    
            logger.info(f"Epoch {i}: L_emb: {L_emb.item():.4f}")

        # # Calculate edge lengths and weights
        # edge_lengths = torch.norm(chain[:, 1, :] - chain[:, 0, :], dim=1)
        # epsilon = 1e-5
        # weights = 1.0 / (edge_lengths + epsilon)
        # weights = weights / weights.sum() * len(weights)
        
        # Apply weights to the integration results
        # L = -(X_sum * weights).sum()
        L = L_emb #-X_sum
        # L = torch.exp(L * 0.05)
        
        # Compute gradients
        L.backward()
        
        # Track gradient norms
        vf_grad_norm = sum(p.grad.norm().item() for p in vf.parameters() if p.grad is not None)
        model_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        
        # Check for infinite gradients and terminate training if necessary
        if not np.isfinite(model_grad_norm) or not np.isfinite(vf_grad_norm):
            logger.error(f"TRAINING TERMINATED: Infinite gradients detected at epoch {i}")
            logger.error(f"VF gradient norm: {vf_grad_norm}, Model gradient norm: {model_grad_norm}")
            
            # Save emergency checkpoint
            emergency_checkpoint_path = f"{run_dir}/emergency_checkpoint_epoch_{i}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'vf_state_dict': vf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': i,
                'losses': losses,
                'grad_norms_vf': grad_norms_vf,
                'grad_norms_model': grad_norms_model,
                'x_sums': x_sums
            }, emergency_checkpoint_path)
            logger.info(f"Emergency checkpoint saved to {emergency_checkpoint_path}")
            
            # Save one last embedding visualization if possible
            try:
                save_embedding_checkpoint(model, x, adj, adata_subsampled, i, checkpoint_dir, logger)
            except:
                logger.error("Could not save final embedding visualization due to errors")
            
            # Break out of training loop
            break
        else:
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(vf.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if i % log_interval == 0:
            logger.info(f"VF grad norm: {vf_grad_norm:.4f}, Model grad norm: {model_grad_norm:.4f}")
        
        # Update parameters
        optimizer.step()
        
        # Store metrics
        losses.append(L.item())
        grad_norms_vf.append(vf_grad_norm)
        grad_norms_model.append(model_grad_norm)
        x_sum_value = X_sum.detach().cpu().numpy()
        x_sums.append(x_sum_value)
        
        # Log to CSV
        with open(metrics_file, 'a') as f:
            f.write(f"{i},{L.item()},{vf_grad_norm},{model_grad_norm},{np.mean(x_sum_value)},{np.min(x_sum_value)},{np.max(x_sum_value)}\n")
        
        if i % log_interval == 0:
            logger.info(f"Epoch {i}: Loss {L.item():.4f}")
        
        # Save embedding visualization checkpoints
        if i % 200 == 0 and i > 0:
            save_embedding_checkpoint(model, x, adj, adata_subsampled, i, checkpoint_dir, logger)
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vf_state_dict': vf.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses[-1],
        'epoch': epochs
    }, f"{run_dir}/model_final.pt")
    
    return losses, grad_norms_vf, grad_norms_model, x_sums


def save_embedding_checkpoint(model, x, adj, adata_subsampled, epoch, checkpoint_dir, logger):
    """Save embedding visualization at a checkpoint"""
    try:
        from analysis import visualize_diffpool_embeddings
        
        logger.info(f"Generating DiffPool embeddings for epoch {epoch}...")
        
        # Create intermediate embedding visualization
        fig1 = visualize_diffpool_embeddings(model, x, adj, adata_subsampled)
        checkpoint_path1 = f"{checkpoint_dir}/epoch_{epoch:04d}_intermediate.svg"
        fig1.savefig(checkpoint_path1)
        plt.close(fig1)
        
        # Create full hierarchy embedding visualization
        fig2 = visualize_diffpool_embeddings(model, x, adj, adata_subsampled)
        checkpoint_path2 = f"{checkpoint_dir}/epoch_{epoch:04d}_full.svg"
        fig2.savefig(checkpoint_path2)
        plt.close(fig2)
        
        logger.info(f"Saved embedding visualizations for epoch {epoch}")
    except Exception as e:
        logger.error(f"Error generating embeddings for epoch {epoch}: {str(e)}")
        logger.exception("Visualization error details:")


def plot_training_metrics(losses, grad_norms_vf, grad_norms_model, x_sums, run_dir, logger):
    """Plot and save training metrics"""
    logger.info("Creating summary plots...")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f"{run_dir}/loss_curve.png")
    plt.close()
    
    # Plot gradient norms
    plt.figure(figsize=(10, 6))
    plt.plot(grad_norms_vf, label='VF Gradients')
    plt.plot(grad_norms_model, label='DiffPool Gradients')
    plt.title('Gradient Norms')
    plt.xlabel('Iteration')
    plt.ylabel('Norm')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{run_dir}/gradient_norms.png")
    plt.close()
    
    # Plot X_sum values
    plt.figure(figsize=(10, 6))
    x_sums_array = np.array(x_sums)
    plt.plot(x_sums_array.mean(axis=1), label='Mean Integration')
    
    if len(x_sums_array.shape) > 1 and x_sums_array.shape[1] > 1:
        plt.fill_between(
            range(len(x_sums_array)), 
            x_sums_array.min(axis=1), 
            x_sums_array.max(axis=1),
            alpha=0.3, 
            label='Min/Max Range'
        )
    
    plt.title('Integration Sum Values')
    plt.xlabel('Iteration')
    plt.ylabel('Integration Sum')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{run_dir}/integration_values.png")
    plt.close()


def visualize_final_model(model, vf, x, adj, run_dir, logger):
    """Create and save final model visualizations"""
    logger.info("Creating vector field visualizations...")
    
    # Generate final model output for visualization
    with torch.no_grad():
        x_out_final, adj_out_final = model(x, adj)
        chain_final = create_mst_chain_from_coords(x_out_final)
    
    # Define paths outside the if-else statements
    component_viz_path = f"{run_dir}/vector_field_components.png"
    combined_viz_path = f"{run_dir}/combined_vector_field.png"
    
    # Generate and save visualization
    if matplotlib.get_backend() == 'agg':  # Non-interactive backend (server)
        logger.info("Generating and saving visualizations to files...")
        
        try:
            plot_vector_field_components_with_edges(vf, chain_final, x_out_final, 
                                                  save_to_file=True, 
                                                  filepath=component_viz_path,
                                                  custom_logger=logger)
            plot_combined_vector_field_with_mst(vf, chain_final, x_out_final, 
                                              save_to_file=True, 
                                              filepath=combined_viz_path,
                                              custom_logger=logger)
            logger.info(f"All visualizations saved to {run_dir}")
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            logger.exception("Visualization error details:")
    else:
        logger.info("Displaying interactive visualizations...")
        try:
            plot_vector_field_components_with_edges(vf, chain_final, x_out_final,
                                                  save_to_file=False,
                                                  custom_logger=logger)
            plot_combined_vector_field_with_mst(vf, chain_final, x_out_final,
                                              save_to_file=False,
                                              custom_logger=logger)
        except Exception as e:
            logger.error(f"Error displaying visualizations: {str(e)}")
            logger.exception("Visualization error details:")
    
    return x_out_final, chain_final

def visualize_final_model2(model, vf, x, adj, run_dir, logger):
    """Create and save final model visualizations"""
    logger.info("Creating vector field visualizations...")
    
    # Generate final model output for visualization
    with torch.no_grad():
        x_out_final, adj_out_final = model(x, adj)
        chain_final = create_mst_chain_from_coords(x_out_final)
    
    # Define paths outside the if-else statements
    combined_viz_path = f"{run_dir}/combined_vector_field.png"
    
    # Generate and save visualization
    if matplotlib.get_backend() == 'agg':  # Non-interactive backend (server)
        logger.info("Generating and saving visualizations to files...")
        
        try:
            # For single-component vector field (c=1), use specialized visualization
            plot_single_component_vector_field(vf, chain_final, x_out_final, 
                                             save_to_file=True, 
                                             filepath=combined_viz_path,
                                             custom_logger=logger)
            logger.info(f"Vector field visualization saved to {run_dir}")
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            logger.exception("Visualization error details:")
    else:
        logger.info("Displaying interactive visualizations...")
        try:
            # For single-component vector field (c=1), use specialized visualization
            plot_single_component_vector_field(vf, chain_final, x_out_final,
                                             save_to_file=False,
                                             custom_logger=logger)
        except Exception as e:
            logger.error(f"Error displaying visualizations: {str(e)}")
            logger.exception("Visualization error details:")
    
    return x_out_final, chain_final

def analyze_clusters(model, x, adj, adata_subsampled, run_dir, logger):
    """Analyze cluster correspondence"""
    logger.info("Analyzing cluster correspondence with DiffPool embeddings...")
    
    try:
        from analysis import visualize_diffpool_embeddings, analyze_cluster_correspondence
        
        # First perform standard cluster correspondence analysis
        if hasattr(adata_subsampled, 'obs') and 'clusters' in adata_subsampled.obs:
            # Get metrics
            correspondence, ari, nmi, final_assignments = analyze_cluster_correspondence(
                model, x, adj, adata_subsampled)
            
            logger.info(f"Cluster correspondence metrics - ARI: {ari:.4f}, NMI: {nmi:.4f}")
            
            # Visualize the embeddings directly - full hierarchy
            fig = visualize_diffpool_embeddings(model, x, adj, adata_subsampled)
            embedding_viz_path = f"{run_dir}/diffpool_embeddings_full.svg"
            fig.savefig(embedding_viz_path)
            plt.close(fig)
            
            # Also save intermediate version
            fig = visualize_diffpool_embeddings(model, x, adj, adata_subsampled)
            embedding_viz_path = f"{run_dir}/diffpool_embeddings_intermediate.svg"
            fig.savefig(embedding_viz_path)
            plt.close(fig)
            
            logger.info(f"DiffPool embedding visualizations saved to {run_dir}")
            
            # Save correspondence matrix
            np.save(f"{run_dir}/cluster_correspondence.npy", correspondence)
        else:
            logger.warning("Cell type annotations not found. Looking for 'clusters' in adata.obs")
    except Exception as e:
        logger.error(f"Error analyzing cluster correspondence: {str(e)}")
        logger.exception("Analysis error details:")


def create_embedding_animation(checkpoint_dir, run_dir, logger):
    """Create animated GIFs of embedding evolution"""
    logger.info("Creating embedding evolution animations...")
    try:
        import glob
        from PIL import Image
        
        # Create animated GIFs for both visualization types
        for hierarchy_type in ['intermediate', 'full']:
            # Get all SVG files for this type
            svg_files = sorted(glob.glob(f"{checkpoint_dir}/epoch_*_{hierarchy_type}.svg"))
            
            if len(svg_files) > 0:
                # Convert SVGs to PNGs first (needed for GIF creation)
                png_files = []
                for svg_file in svg_files:
                    png_file = svg_file.replace('.svg', '.png')
                    os.system(f"convert {svg_file} {png_file}")
                    png_files.append(png_file)
                
                # Create animated GIF
                frames = [Image.open(png) for png in png_files]
                animation_path = f"{run_dir}/embedding_evolution_{hierarchy_type}.gif"
                
                # Save the animation
                frames[0].save(
                    animation_path,
                    format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=500,  # milliseconds per frame
                    loop=0  # loop forever
                )
                
                logger.info(f"Created animation: {animation_path}")
                
                # Clean up PNG files (optional)
                for png_file in png_files:
                    os.remove(png_file)
        
    except Exception as e:
        logger.warning(f"Could not create animation: {str(e)}")
        logger.warning("Animation creation requires PIL and ImageMagick")


def main():
    """Main function to run the experiment"""
    # Setup experiment
    run_id, run_dir, logger = setup_experiment()
    
    # Load data
    # FILE_NAME = "data/pancreas.h5ad"
    # logger.info(f"Loading data from {FILE_NAME}")
    # adata_subsampled, x, adj = preprocess_pancreas_data(FILE_NAME)
    FILE_NAME = "data/bone_marrow.h5ad"
    logger.info(f"Loading bone marrow data from {FILE_NAME}")

    from dataset import preprocess_bone_marrow_data
    adata_subsampled, x, adj = preprocess_bone_marrow_data(FILE_NAME)
    
    # Setup model, vector field and optimizer
    model, vf, optimizer, c = setup_model(x, adj, logger)
    
    # Visualize initial state
    visualize_initial_state(model, vf, x, adj, run_dir, logger)
    
    # Train model
    epochs = 50
    losses, grad_norms_vf, grad_norms_model, x_sums = train_model(
        model, vf, optimizer, x, adj, adata_subsampled, epochs, run_dir, logger)
    
    # Plot training metrics
    plot_training_metrics(losses, grad_norms_vf, grad_norms_model, x_sums, run_dir, logger)
    
    # Visualize final model
    x_out_final, chain_final = visualize_final_model2(model, vf, x, adj, run_dir, logger)
    
    # Analyze cluster correspondence
    analyze_clusters(model, x, adj, adata_subsampled, run_dir, logger)
    
    # Create embedding animations
    checkpoint_dir = f"{run_dir}/embedding_checkpoints"
    create_embedding_animation(checkpoint_dir, run_dir, logger)

    fig1 = visualize_joint_embeddings(model, x, adj, adata_subsampled, full_hierarchy=False)
    fig1.savefig(f"{run_dir}/joint_embeddings_intermediate.png", dpi=300)
    
    logger.info(f"Training completed. Results saved to {run_dir}")

def download_bone_marrow_dataset():
    DATA_DIR = Path("data")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    FILE_PATH = DATA_DIR / "bone_marrow.h5ad"
    if not FILE_PATH.exists():
        print("Downloading bone marrow dataset...")
        url = "https://figshare.com/ndownloader/files/35826944"
        
        response = requests.get(url, stream=True)
        with open(FILE_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Dataset downloaded to {FILE_PATH}")
    else:
        print(f"Dataset already exists at {FILE_PATH}")

if __name__ == "__main__":
    download_bone_marrow_dataset()
    main()