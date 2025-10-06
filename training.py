import torch
import torch.nn.functional as F
from neural_k_forms.chains import generate_integration_matrix

def custom_train(model, vf, train_loader, optimizer, device, num_iterations=1000):
    """
    Custom training loop using integration matrix and vector fields with gradient tracking fixes
    """
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    model.train()
    total_loss = 0.0
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        batch_loss = 0.0
        
        # Get graph representations
        with torch.no_grad():
            _, _, _, graphs = model(data.x, data.adj, data.mask)
            coarsened_graph = graphs['level_2']
            
            # Extract adjacency matrix and node features
            adj = coarsened_graph['adj'].clone()  # Clone to avoid inplace modifications
            x = coarsened_graph['x'].clone()      # Clone to avoid inplace modifications
            
            # Convert adjacency matrix to edge list format
            batch_size = adj.size(0)
            chains = []
            
            for b in range(batch_size):
                edges = torch.nonzero(adj[b] > 0).to(device)
                if edges.size(0) > 0:  # Check if there are any edges
                    start_nodes = x[b, edges[:, 0]]
                    end_nodes = x[b, edges[:, 1]]
                    chain = torch.stack([start_nodes, end_nodes], dim=1)
                    chains.append(chain)
            
            if not chains:  # Skip if no valid chains
                continue
                
            # Combine chains safely
            chain_tensor = torch.cat(chains, dim=0).to(device)

        # Training iterations
        for i in range(num_iterations):
            # Create fresh tensor for optimization
            chain_tensor_iter = chain_tensor.clone().requires_grad_(True)
            
            # Zero gradients at the start of each iteration
            optimizer.zero_grad()
            
            try:
                # Calculate integration matrix
                X = generate_integration_matrix(vf, chain_tensor_iter)
                
                # Sum reduction without inplace operation
                X_sum = torch.sum(X)
                
                # Calculate loss
                loss = torch.abs(4294967295.0 - X_sum)
                
                # Backward pass
                loss.backward()
                
                # Optimizer step
                optimizer.step()
                
                # Accumulate loss (detached from computation graph)
                batch_loss += loss.detach().item()
                
                if i % 10 == 0:  # Print less frequently
                    print(f"Batch {batch_idx}, Iteration {i}, Loss: {loss.item():.4f}")
                    
            except RuntimeError as e:
                print(f"Error in iteration {i}: {str(e)}")
                continue
        
        # Average batch loss over iterations
        if num_iterations > 0:
            batch_loss /= num_iterations
            total_loss += batch_loss
    
    # Disable anomaly detection
    torch.autograd.set_detect_anomaly(False)
    
    # Return average loss over all batches
    return total_loss / len(train_loader) if len(train_loader) > 0 else float('inf')

def train(model, train_loader, optimizer, device):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _, _, _ = model(data.x, data.adj, data.mask)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    
    return loss_all / len(train_loader.dataset)

@torch.no_grad()
def test(model, loader, device):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        output, _, _, _ = model(data.x, data.adj, data.mask)
        pred = output.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)