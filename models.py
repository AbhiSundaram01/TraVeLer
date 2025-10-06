from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool

import torch.nn as nn
from dgl.nn.pytorch import DenseSAGEConv

class DiffPool(torch.nn.Module):
    def __init__(self, num_features, max_nodes, hidden_dim=16, output_dim=2, n_layers=2, cluster_ratio=0.06):
        super(DiffPool, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim # 2
        self.n_layers = n_layers
        self.cluster_ratio = cluster_ratio

        num_clusters_1 = ceil(cluster_ratio * max_nodes)
        num_clusters_2 = ceil(cluster_ratio * cluster_ratio * max_nodes)
        # num_clusters_1 = 64
        # num_clusters_2 = 8

        self.gnn1_embed = DenseSAGEConv(num_features, hidden_dim, norm=nn.BatchNorm1d(hidden_dim))
        self.gnn1_pool = DenseSAGEConv(num_features, num_clusters_1, norm=nn.BatchNorm1d(num_clusters_1))

        self.gnn2_embed = DenseSAGEConv(hidden_dim, output_dim, norm=nn.BatchNorm1d(output_dim))
        self.gnn2_pool = DenseSAGEConv(hidden_dim, num_clusters_2, norm=nn.BatchNorm1d(num_clusters_2))

        # Initialize weights
        self.reset_parameters()

        self.projection = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x_0, adj_0):
        # # Add batch dimension if not present
        # if x_0.dim() == 2:
        #     x_0 = x_0.unsqueeze(0)
        #     adj_0 = adj_0.unsqueeze(0)
        #     unbatch_output = True
        # else:
        #     unbatch_output = False

        z_0 = self.gnn1_embed(adj_0, x_0) # (1848, 1848) x (1848, 2) x (2, 16) = (1848, 16)
        s_0 = torch.softmax(self.gnn1_pool(adj_0, x_0), dim=-1) # (1848, 1848) x (1848, 2) x (2, 64) = (1848, 64)

        x_1 = s_0.t() @ z_0 # (1848, 64)' x (1848, 16) = (64, 16)
        adj_1 = s_0.t() @ adj_0 @ s_0 # (1848, 64)' x (1848, 1848) x (1848, 64) = (64, 64)

        z_1 = self.gnn2_embed(adj_1, x_1) # (64, 64) x (64, 16) x (16, 2) = (64, 2)
        s_1 = torch.softmax(self.gnn2_pool(adj_1, x_1), dim=-1) # (64, 64) x (64, 16) x (16, 8) = (64, 8)

        x_2 = s_1.t() @ z_1 # (64, 8)' x (64, 2) = (8, 2)
        adj_2 = s_1.t() @ adj_1 @ s_1 # (64, 8)' x (64, 64) x (64, 8) = (8, 8)

        # # Remove batch dimension if we added it
        # if unbatch_output:
        #     x_2 = x_2.squeeze(0)
        #     adj_2 = adj_2.squeeze(0)

        return x_2, adj_2
    
    def compute_node_embeddings(self, x_0, adj_0, full_hierarchy=False):
        """
        Compute embeddings for each original node using the hierarchical structure.
        
        Args:
            x_0: Input node features
            adj_0: Input adjacency matrix
            full_hierarchy: If True, project through the complete hierarchy (s_0 @ s_1 @ x_2)
                            If False (default), use intermediate embedding (s_0 @ z_1)
        
        Returns:
            node_embeddings: Tensor of shape (num_original_nodes, output_dim) containing
                            embeddings of original nodes in the final embedding space.
        """
        with torch.no_grad():
            # First level - same as in forward
            z_0 = self.gnn1_embed(adj_0, x_0)  # (N, hidden_dim)
            s_0 = torch.softmax(self.gnn1_pool(adj_0, x_0), dim=-1)  # (N, num_clusters_1)
            
            # Intermediate representations
            x_1 = s_0.t() @ z_0  # (num_clusters_1, hidden_dim)
            adj_1 = s_0.t() @ adj_0 @ s_0  # (num_clusters_1, num_clusters_1)
            
            if full_hierarchy:
                # Second level - same as in forward
                z_1 = self.gnn2_embed(adj_1, x_1)  # (num_clusters_1, output_dim)
                s_1 = torch.softmax(self.gnn2_pool(adj_1, x_1), dim=-1)  # (num_clusters_1, num_clusters_2)
                
                # Final coarsened representation
                x_2 = s_1.t() @ z_1  # (num_clusters_2, output_dim)
                
                # Map original nodes all the way to final embedding space through both hierarchical levels
                # This directly relates each node to the final coarsened clusters
                node_embeddings = s_0 @ s_1 @ x_2  # (N, num_clusters_2) @ (num_clusters_2, output_dim)
            else:
                # Original approach - project to intermediate embedding space
                z_1 = self.gnn2_embed(adj_1, x_1)  # (num_clusters_1, output_dim)
                
                # Map original nodes to the intermediate embedding space
                node_embeddings = s_0 @ z_1  # (N, output_dim)

            node_embeddings = self.projection(node_embeddings)
            
            return node_embeddings
    
    def reset_parameters(self):
        """Initialize model weights using Xavier initialization"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif hasattr(m, 'fc'):  # For DenseSAGEConv layers which have fc attribute
                # Xavier initialization for the internal linear layer
                nn.init.xavier_uniform_(m.fc.weight)
                if m.fc.bias is not None:
                    nn.init.zeros_(m.fc.bias)
        
        # Apply initialization to all modules
        self.apply(init_weights)
    
class GNN2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 n_layers, normalize=False, lin=True):
        super(GNN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        for i in range(n_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))

        self.bns = torch.nn.ModuleList()
        for i in range(n_layers):
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))


    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        
        for layer_idx in range(len(self.convs)):
            x = self.bns[layer_idx](F.relu(self.convs[layer_idx](x, adj, mask)))
            
        return x
    
class DiffPool2(torch.nn.Module):
    def __init__(self, num_features, max_nodes, output_dim=2):
        super(DiffPool, self).__init__()
				#---------------------------------------------#
        input_dim = num_features
        output_dim = output_dim
        hidden_dim = 64
        n_layers = 2
        #---------------------------------------------#
        num_cluster1 = ceil(0.25 * max_nodes)
        num_cluster2 = ceil(0.25 * 0.25 * max_nodes)
        #---------------------------------------------#
        self.gnn1_embed = GNN(input_dim, hidden_dim, hidden_dim, 
                              n_layers)
        self.gnn2_embed = GNN(hidden_dim, hidden_dim, hidden_dim, 
                              n_layers, lin=False)
        self.gnn3_embed = GNN(hidden_dim, hidden_dim, hidden_dim, 
                              n_layers, lin=False)
				#---------------------------------------------#
        self.gnn1_pool = GNN(input_dim, hidden_dim, num_cluster1, 
                             n_layers)
        self.gnn2_pool = GNN(hidden_dim, hidden_dim, num_cluster2, 
                             n_layers)
				#---------------------------------------------#
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)
				#---------------------------------------------#
        
    def forward(self, x0, adj0, mask=None):
      	# s : cluster assignment matrix
        
        s0 = self.gnn1_pool(x0, adj0, mask)
        z0 = self.gnn1_embed(x0, adj0, mask)
        x1, adj1, l1, e1 = dense_diff_pool(z0, adj0, s0, mask)
        # dense_diff_pool : 아래의 연산을 수행
        #x_1 = s_0.t() @ z_0
        #adj_1 = s_0.t() @ adj_0 @ s_0
        
        s1 = self.gnn2_pool(x1, adj1)
        z1 = self.gnn2_embed(x1, adj1)
        x2, adj2, l2, e2 = dense_diff_pool(z1, adj1, s1)
        
        z2 = self.gnn3_embed(x2, adj2)

        return x2, adj2

class GNN3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))


    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
            x = x.transpose(1, 2)  # Moves features to middle dimension [32, 64, 150]
            x = self.bns[step](x)
            x = x.transpose(1, 2)  # Restores original shape [32, 150, 64]
        

        return x


class DiffPool3(torch.nn.Module):
    def __init__(self, num_features, num_classes, max_nodes, sparse_threshold=0.1):
        super(DiffPool, self).__init__()
        self.sparse_threshold = sparse_threshold
        self.max_nodes = max_nodes

        # Calculate number of nodes for first pooling layer
        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(num_features, 64, num_nodes)
        self.gnn1_embed = GNN(num_features, 64, 64)

        # Calculate number of nodes for second pooling layer
        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(64, 64, num_nodes)
        self.gnn2_embed = GNN(64, 32, 2, lin=False)

        self.gnn3_embed = GNN(2, 2, 2, lin=False)

        self.lin1 = torch.nn.Linear(2, 2)
        self.lin2 = torch.nn.Linear(2, num_classes)

    def sparsify_adj(self, adj):
        """Sparsify adjacency matrix by thresholding"""
        adj = torch.where(adj > self.sparse_threshold, adj, torch.zeros_like(adj))
        eye = torch.eye(adj.size(1), device=adj.device).unsqueeze(0).expand(adj.size(0), -1, -1)
        adj = adj * (1 - eye)  # Remove self-loops
        return adj

    def forward(self, x, adj, mask=None):
        # Add debugging prints
        print(f"Input shape: {x.shape}")
        
        # First level coarsening
        s = self.gnn1_pool(x, adj, mask)
        print(f"s_01 shape: {s.shape}")
        
        x = self.gnn1_embed(x, adj, mask)
        
        # First coarsened graph with sparsification
        x_1, adj_1, l1, e1 = dense_diff_pool(x, adj, s, mask)
        print(f"After first pooling: {x_1.shape}")
        adj_1 = self.sparsify_adj(adj_1)
        
        # Second level coarsening
        s = self.gnn2_pool(x_1, adj_1)
        print(f"s_12 shape: {s.shape}")
        
        x = self.gnn2_embed(x_1, adj_1)
        
        # Second coarsened graph with sparsification
        x_2, adj_2, l2, e2 = dense_diff_pool(x, adj_1, s)
        adj_2 = self.sparsify_adj(adj_2)
        
        # Store the hierarchical representations
        coarsened_graphs = {
            'level_0': {
                'x': x,        # Original node features
                'adj': adj,    # Original adjacency matrix
                'mask': mask   # Original mask
            },
            'level_1': {
                'x': x_1,      # First coarsened node features
                'adj': adj_1,  # First coarsened adjacency matrix
                's': s         # Assignment matrix for level 1
            },
            'level_2': {
                'x': x_2,      # Second coarsened node features
                'adj': adj_2,  # Second coarsened adjacency matrix
                's': s         # Assignment matrix for level 2
            }
        }
        
        # Continue with the original forward pass
        x = self.gnn3_embed(x_2, adj_2)
        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2, coarsened_graphs