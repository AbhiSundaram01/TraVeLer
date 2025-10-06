import unittest
from unittest.mock import MagicMock
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DenseDataLoader
from training import custom_train

class TestCustomTrain(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock()
        self.vf = MagicMock()
        self.optimizer = MagicMock()
        self.device = torch.device('cpu')
        self.num_iterations = 10

        # Create a dummy dataset with proper PyG Data objects
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return Data(
                    x=torch.randn(5, 2),        # [num_nodes, num_features]
                    adj=torch.randn(5, 5),      # [num_nodes, num_nodes]
                    mask=torch.ones(5, dtype=torch.bool),  # [num_nodes]
                    y=torch.tensor(1)
                )

        # Create proper chain structure that matches the expected shapes
        num_edges = 3
        d = 5  # Number of discretization points
        features = 2
        chain_tensor = torch.randn(num_edges, 2, features)  # [num_edges, 2, features]
        
        # Mock the model's forward pass
        self.model.return_value = (
            chain_tensor,  # output is chain tensor
            None,         # l1
            None,         # e1
            {            # coarsened graphs
                'level_2': {
                    'adj': torch.randn(2, 5, 5),
                    'x': torch.randn(2, 5, 2)
                }
            }
        )
        
        # Mock vector field to return correct shape
        def vf_side_effect(x):
            # Input: [num_edges * d, features]
            # Output: [num_edges * d, 1]
            batch_size = x.shape[0]
            return torch.randn(batch_size, 1)
        
        self.vf.side_effect = vf_side_effect
        self.train_loader = DenseDataLoader(DummyDataset(), batch_size=2)

    def test_custom_train(self):
        try:
            loss = custom_train(
                self.model, 
                self.vf, 
                self.train_loader, 
                self.optimizer, 
                self.device, 
                self.num_iterations
            )
            self.assertIsInstance(loss, float)
            self.assertGreaterEqual(loss, 0)
        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")

if __name__ == '__main__':
    unittest.main()