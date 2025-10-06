# Simple script to test the DiffPool model on toy datasets

import os
import sys
import argparse
from main3 import *  # Import all functions from main3

def run_with_dataset(dataset_type, epochs=500):
    """Run DiffPool experiment with specified dataset"""
    # Setup experiment
    run_id, run_dir, logger = setup_experiment()
    logger.info(f"Running experiment with {dataset_type} dataset")
    
    # Load appropriate dataset
    if dataset_type == "toy":
        try:
            from dataset import create_toy_multifurcating_data
            adata_subsampled, x, adj = create_toy_multifurcating_data()
            logger.info(f"Using toy dataset with {x.shape[0]} cells and {x.shape[1]} features")
        except ImportError as e:
            logger.error(f"Failed to import pyVIA: {str(e)}")
            sys.exit(1)
    elif dataset_type == "pancreas":
        FILE_NAME = "data/pancreas.h5ad"
        logger.info(f"Loading data from {FILE_NAME}")
        from dataset import preprocess_pancreas_data
        adata_subsampled, x, adj = preprocess_pancreas_data(FILE_NAME)
    else:
        logger.error(f"Unknown dataset type: {dataset_type}")
        sys.exit(1)
    
    # Setup model, vector field and optimizer
    model, vf, optimizer, c = setup_model(x, adj, logger)
    
    # Visualize initial state
    visualize_initial_state(model, vf, x, adj, run_dir, logger)
    
    # Train model with specified number of epochs
    losses, grad_norms_vf, grad_norms_model, x_sums = train_model(
        model, vf, optimizer, x, adj, adata_subsampled, epochs, run_dir, logger)
    
    # Plot training metrics
    plot_training_metrics(losses, grad_norms_vf, grad_norms_model, x_sums, run_dir, logger)
    
    # Visualize final model
    x_out_final, chain_final = visualize_final_model2(model, vf, x, adj, run_dir, logger)
    
    # Analyze cluster correspondence
    analyze_clusters(model, x, adj, adata_subsampled, run_dir, logger)
    
    # Create joint embedding visualizations
    fig1 = visualize_joint_embeddings(model, x, adj, adata_subsampled, full_hierarchy=False)
    fig1.savefig(f"{run_dir}/joint_embeddings_intermediate.png", dpi=300)
    
    fig2 = visualize_joint_embeddings(model, x, adj, adata_subsampled, full_hierarchy=True)
    fig2.savefig(f"{run_dir}/joint_embeddings_full.png", dpi=300)
    
    logger.info(f"Training completed with {dataset_type} dataset. Results saved to {run_dir}")
    
    return run_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DiffPool with different datasets")
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "pancreas"],
                        help="Dataset to use for training (toy or pancreas)")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of epochs to train")
    
    args = parser.parse_args()
    run_dir = run_with_dataset(args.dataset, args.epochs)
    print(f"Results saved to {run_dir}")
