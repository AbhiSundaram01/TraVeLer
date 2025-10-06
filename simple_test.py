import torch
import torch.nn as nn
from diffeo import InvertibleBlock, DiffeoNet, OneFormNet

def test_basic_functionality():
    print("Testing basic functionality...")
    
    # Test InvertibleBlock
    print("1. Testing InvertibleBlock...")
    dim = 4
    block = InvertibleBlock(dim)
    x = torch.randn(10, dim)
    y = block(x)
    x_reconstructed = block.inverse(y)
    error = torch.norm(x - x_reconstructed).item()
    print(f"   Reconstruction error: {error:.2e}")
    
    # Test DiffeoNet
    print("2. Testing DiffeoNet...")
    diffeo_net = DiffeoNet(dim=4, n_blocks=3)
    x = torch.randn(10, 4)
    y = diffeo_net(x)
    x_reconstructed = diffeo_net.inverse(y)
    error = torch.norm(x - x_reconstructed).item()
    print(f"   Reconstruction error: {error:.2e}")
    
    # Test OneFormNet
    print("3. Testing OneFormNet...")
    oneform_net = OneFormNet(input_dim=3, hidden_dim=32)
    x = torch.randn(10, 3)
    output = oneform_net(x)
    print(f"   Input shape: {x.shape}, Output shape: {output.shape}")
    
    # Test simple Jacobian computation
    print("4. Testing Jacobian computation...")
    uv = torch.randn(5, 2, requires_grad=True)
    diffeo_2d = DiffeoNet(dim=2, n_blocks=2)
    
    try:
        jac = torch.autograd.functional.jacobian(diffeo_2d, uv)
        print(f"   Jacobian shape: {jac.shape}")
        print("   Jacobian computation successful!")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("Basic tests completed!")

if __name__ == "__main__":
    torch.manual_seed(42)
    test_basic_functionality()
