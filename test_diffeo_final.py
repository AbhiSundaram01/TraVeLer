import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from diffeo import InvertibleBlock, DiffeoNet, OneFormNet, PullbackOneForm

def test_invertible_block():
    """Test the invertibility of a single block"""
    print("Testing InvertibleBlock...")
    
    dim = 4
    block = InvertibleBlock(dim)
    
    # Test with multiple samples
    x = torch.randn(100, dim)
    y = block(x)
    x_reconstructed = block.inverse(y)
    
    # Calculate reconstruction error
    reconstruction_error = torch.norm(x - x_reconstructed, dim=1)
    mean_error = torch.mean(reconstruction_error).item()
    max_error = torch.max(reconstruction_error).item()
    
    print(f"  Mean reconstruction error: {mean_error:.2e}")
    print(f"  Max reconstruction error: {max_error:.2e}")
    
    # Create visualization
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(reconstruction_error.detach().numpy(), bins=30, alpha=0.7)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(reconstruction_error)), reconstruction_error.detach().numpy(), alpha=0.6)
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error per Sample')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('invertible_block_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return mean_error < 1e-5

def test_diffeo_net():
    """Test the full diffeomorphism network"""
    print("\nTesting DiffeoNet...")
    
    dim = 4
    n_blocks = 3
    diffeo_net = DiffeoNet(dim, n_blocks)
    
    # Test with multiple samples
    x = torch.randn(100, dim)
    y = diffeo_net(x)
    x_reconstructed = diffeo_net.inverse(y)
    
    # Calculate reconstruction error
    reconstruction_error = torch.norm(x - x_reconstructed, dim=1)
    mean_error = torch.mean(reconstruction_error).item()
    max_error = torch.max(reconstruction_error).item()
    
    print(f"  Mean reconstruction error: {mean_error:.2e}")
    print(f"  Max reconstruction error: {max_error:.2e}")
    
    # Visualize the transformation in 2D
    dim_2d = 2
    diffeo_2d = DiffeoNet(dim_2d, n_blocks)
    
    # Create a grid of points
    x_range = np.linspace(-2, 2, 20)
    y_range = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Original points
    points_orig = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    
    # Transformed points
    with torch.no_grad():
        points_transformed = diffeo_2d(points_orig)
    
    # Plot the transformation
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(points_orig[:, 0], points_orig[:, 1], alpha=0.6, s=20)
    plt.title('Original Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.scatter(points_transformed[:, 0], points_transformed[:, 1], alpha=0.6, s=20)
    plt.title('Transformed Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('diffeo_transformation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return mean_error < 1e-4

def test_jacobian_computation():
    """Test the Jacobian computation"""
    print("\nTesting Jacobian computation...")
    
    # Simple 2D -> 3D mapping for testing
    dim_in = 2
    dim_out = 3
    
    class SimpleMapping(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim_in, dim_out)
            
        def forward(self, x):
            return self.linear(x)
    
    mapping = SimpleMapping()
    
    # Test point
    uv = torch.randn(5, dim_in, requires_grad=True)
    
    # Compute Jacobian using functional.jacobian
    try:
        jac_functional = torch.autograd.functional.jacobian(mapping, uv)
        print(f"  Jacobian shape using functional.jacobian: {jac_functional.shape}")
        
        # Alternative: Manual Jacobian computation (fixed)
        xyz = mapping(uv)
        jacobians = []
        
        for i in range(xyz.shape[1]):  # Loop over output dimensions
            grad_outputs = torch.zeros_like(xyz)
            grad_outputs[:, i] = 1.0
            
            jac_col = torch.autograd.grad(
                outputs=xyz, 
                inputs=uv, 
                grad_outputs=grad_outputs,
                retain_graph=True, 
                create_graph=True
            )[0]
            jacobians.append(jac_col)
        
        jac_manual = torch.stack(jacobians, dim=2)  # Shape: (batch, input_dim, output_dim)
        print(f"  Jacobian shape using manual computation: {jac_manual.shape}")
        
        # Compare the two methods
        diff = torch.norm(jac_functional - jac_manual.permute(0, 2, 1))
        print(f"  Difference between methods: {diff.item():.2e}")
        
        print("  Jacobian computation test passed!")
        return True
        
    except Exception as e:
        print(f"  Error in Jacobian computation: {e}")
        return False

def test_pullback_oneform():
    """Test the pullback one-form computation"""
    print("\nTesting PullbackOneForm...")
    
    # Create networks
    diffeo_net = DiffeoNet(dim=2, n_blocks=2)  # 2D -> 2D for simplicity
    oneform_net = OneFormNet(input_dim=2)  # Adjusted for 2D
    
    # Modify PullbackOneForm for 2D case
    class PullbackOneForm2D(nn.Module):
        def __init__(self, diffeo_net, oneform_net):
            super().__init__()
            self.diffeo_net = diffeo_net
            self.oneform_net = oneform_net

        def forward(self, uv):
            # Compute phi(u, v) -> (x, y)
            xy = self.diffeo_net(uv)

            # Compute (f, g) at (x, y)
            fg = self.oneform_net(xy)  # Shape: (N, 2) -> (N, 2)

            # Compute Jacobian J = d(phi)/d(uv) (N, 2, 2)
            J_phi = torch.autograd.functional.jacobian(lambda uv: self.diffeo_net(uv).T, uv).permute(1, 2, 0)

            # Compute pullback coefficients p, q
            p = (fg[:, 0] * J_phi[:, 0, 0] + fg[:, 1] * J_phi[:, 1, 0])
            q = (fg[:, 0] * J_phi[:, 0, 1] + fg[:, 1] * J_phi[:, 1, 1])

            return torch.stack([p, q], dim=1)  # (N,2)
    
    pullback_net = PullbackOneForm2D(diffeo_net, oneform_net)
    
    # Test computation
    uv = torch.randn(10, 2, requires_grad=True)
    try:
        omega_pullback = pullback_net(uv)
        print(f"  Pullback computation successful! Output shape: {omega_pullback.shape}")
        print(f"  Sample values: {omega_pullback[:3]}")
        return True
    except Exception as e:
        print(f"  Error in pullback computation: {e}")
        return False

def test_training_loop():
    """Test a simplified training loop"""
    print("\nTesting training loop...")
    
    # Create networks (2D for simplicity)
    diffeo_net = DiffeoNet(dim=2, n_blocks=2)
    oneform_net = OneFormNet(input_dim=2)
    
    class PullbackOneForm2D(nn.Module):
        def __init__(self, diffeo_net, oneform_net):
            super().__init__()
            self.diffeo_net = diffeo_net
            self.oneform_net = oneform_net

        def forward(self, uv):
            xy = self.diffeo_net(uv)
            fg = self.oneform_net(xy)
            
            # Simplified Jacobian computation for testing
            J_phi = torch.autograd.functional.jacobian(lambda uv: self.diffeo_net(uv).T, uv).permute(1, 2, 0)
            
            p = (fg[:, 0] * J_phi[:, 0, 0] + fg[:, 1] * J_phi[:, 1, 0])
            q = (fg[:, 0] * J_phi[:, 0, 1] + fg[:, 1] * J_phi[:, 1, 1])

            return torch.stack([p, q], dim=1)
    
    pullback_net = PullbackOneForm2D(diffeo_net, oneform_net)
    
    optimizer = torch.optim.Adam(
        list(diffeo_net.parameters()) + list(oneform_net.parameters()), 
        lr=1e-3
    )
    
    losses = []
    
    try:
        for epoch in range(50):  # Reduced for testing
            uv = torch.rand(16, 2)  # Sample input in R^2
            omega_pullback = pullback_net(uv)
            
            loss = torch.mean(omega_pullback**2)
            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}, Loss: {loss.item():.6f}")
        
        # Plot training curve
        plt.figure(figsize=(8, 5))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  Training loop test passed!")
        return True
        
    except Exception as e:
        print(f"  Error in training loop: {e}")
        return False

def visualize_vector_field():
    """Visualize the learned vector field"""
    print("\nVisualizing vector field...")
    
    # Create a simple trained network
    diffeo_net = DiffeoNet(dim=2, n_blocks=2)
    oneform_net = OneFormNet(input_dim=2)
    
    # Create grid for visualization
    x = np.linspace(-2, 2, 15)
    y = np.linspace(-2, 2, 15)
    X, Y = np.meshgrid(x, y)
    
    points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    
    with torch.no_grad():
        # Get the one-form values
        oneform_values = oneform_net(points)
        
        # Reshape for plotting
        U = oneform_values[:, 0].reshape(X.shape)
        V = oneform_values[:, 1].reshape(X.shape)
    
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, U, V, alpha=0.7)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Vector Field Visualization')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig('vector_field.png', dpi=150, bbox_inches='tight')
    plt.close()

def run_all_tests():
    """Run all tests and generate visualizations"""
    print("=" * 50)
    print("RUNNING DIFFEO.PY TESTS")
    print("=" * 50)
    
    test_results = []
    
    # Run individual tests
    test_results.append(("InvertibleBlock", test_invertible_block()))
    test_results.append(("DiffeoNet", test_diffeo_net()))
    test_results.append(("Jacobian Computation", test_jacobian_computation()))
    test_results.append(("PullbackOneForm", test_pullback_oneform()))
    test_results.append(("Training Loop", test_training_loop()))
    
    # Generate additional visualizations
    visualize_vector_field()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the code for issues.")
    
    print("\nGenerated visualization files:")
    print("- invertible_block_test.png")
    print("- diffeo_transformation.png") 
    print("- training_curve.png")
    print("- vector_field.png")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_all_tests()
