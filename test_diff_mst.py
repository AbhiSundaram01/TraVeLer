# %%
import pytest
import torch
import numpy as np
from graph_utils import soft_mst_approximation2

import matplotlib.pyplot as plt

def visualize_mst(points, edges, title="MST Approximation"):
    """
    Helper function to visualize the MST approximation.
    
    Args:
        points: Tensor of node coordinates
        edges: Tensor in chain format [num_edges, 2, dimension]
        title: Plot title
    """
    plt.figure(figsize=(8, 8))
    # Plot nodes
    plt.scatter(points[:, 0].numpy(), points[:, 1].numpy(), c='blue', s=50, label='Nodes')
    
    # Plot edges
    for i in range(edges.shape[0]):
        x_coords = [edges[i, 0, 0].item(), edges[i, 1, 0].item()]
        y_coords = [edges[i, 0, 1].item(), edges[i, 1, 1].item()]
        plt.plot(x_coords, y_coords, 'r-', alpha=0.5)
    
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def test_soft_mst_approximation2_basic():
    """Test basic functionality of soft_mst_approximation2"""
    torch.manual_seed(42)  # For reproducibility
    
    # Create a simple 2D point set
    n_points = 10
    x_out = torch.rand(n_points, 2)  # 10 points in 2D
    
    # Run the function with default parameters
    chain = soft_mst_approximation2(x_out)
    
    # Visualize the result
    visualize_mst(x_out, chain, "Basic MST Approximation")
    
    # Verify output shape: [num_edges, 2, dimension]
    assert chain.dim() == 3
    assert chain.size(1) == 2  # source and target
    assert chain.size(2) == 2  # 2D coordinates
    assert chain.size(0) > 0   # At least one edge

def test_soft_mst_approximation2_target_edges():
    """Test soft_mst_approximation2 with specific target_edges"""
    torch.manual_seed(42)
    
    # Create a simple 2D point set
    n_points = 10
    x_out = torch.rand(n_points, 2)
    
    # Test with exactly 5 edges
    target = 5
    chain = soft_mst_approximation2(x_out, target_edges=target)
    visualize_mst(x_out, chain, f"MST with {target} edges")
    assert chain.size(0) == target
    
    # Test with more edges than possible unique pairs
    target = 100  # More than n_points*(n_points-1)/2 = 45 possible edges
    chain = soft_mst_approximation2(x_out, target_edges=target)
    visualize_mst(x_out, chain, f"MST with {target} edges (duplicated)")
    assert chain.size(0) == target

def test_soft_mst_approximation2_temperature():
    """Test the effect of temperature parameter on edge selection"""
    torch.manual_seed(42)
    
    # Create a simple 2D point set
    n_points = 10
    x_out = torch.rand(n_points, 2)
    
    # Test with high temperature (more uniform weights)
    chain_high_temp = soft_mst_approximation2(x_out, temperature=1.0, target_edges=15)
    visualize_mst(x_out, chain_high_temp, "MST with High Temperature")
    
    # Test with low temperature (more concentrated weights)
    chain_low_temp = soft_mst_approximation2(x_out, temperature=0.01, target_edges=15)
    visualize_mst(x_out, chain_low_temp, "MST with Low Temperature")
    
    # Low temperature should connect closer nodes
    # Calculate average edge length for both
    def avg_edge_length(chain):
        total = 0.0
        for i in range(chain.size(0)):
            dist = torch.norm(chain[i, 0] - chain[i, 1]).item()
            total += dist
        return total / chain.size(0)
    
    high_temp_avg = avg_edge_length(chain_high_temp)
    low_temp_avg = avg_edge_length(chain_low_temp)
    
    # Lower temperature should prefer shorter edges on average
    assert low_temp_avg <= high_temp_avg

def test_edge_correctness():
    """Test if edges in the chain connect correct nodes from x_out"""
    torch.manual_seed(42)
    
    # Create a specific point set where we know what to expect
    x_out = torch.tensor([
        [0.0, 0.0],  # Node 0
        [0.0, 1.0],  # Node 1
        [1.0, 0.0],  # Node 2
        [1.0, 1.0],  # Node 3
        [2.0, 2.0]   # Node 4 (farther from others)
    ])
    
    target = 4  # Just enough for a tree
    chain = soft_mst_approximation2(x_out, temperature=0.1, target_edges=target)
    visualize_mst(x_out, chain, "MST on Square Grid")
    
    # Verify the edges actually connect points from x_out
    for i in range(chain.size(0)):
        source = chain[i, 0]
        target = chain[i, 1]
        
        # Check if source and target are points from x_out
        source_exists = any(torch.all(torch.eq(source, p)) for p in x_out)
        target_exists = any(torch.all(torch.eq(target, p)) for p in x_out)
        
        assert source_exists, f"Edge source {source} not found in original points"
        assert target_exists, f"Edge target {target} not found in original points"

if __name__ == "__main__":
    # Run tests manually if needed
    test_soft_mst_approximation2_basic()
    test_soft_mst_approximation2_target_edges()
    test_soft_mst_approximation2_temperature()
    test_edge_correctness()
# %%
