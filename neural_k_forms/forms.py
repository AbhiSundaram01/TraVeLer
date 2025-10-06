"""Represent neural $k$-forms in a simple interface."""

import torch
import torch.nn as nn


class NeuralOneForm(nn.Module):
    """Simple neural $1$-form neural network.

    This represents a neural $1$-form as a simple neural network with
    a single hidden layer.
    """

    def __init__(self, model, num_cochains, input_dim=None, hidden_dim=None):
        super().__init__()

        self.input_dim = input_dim
        self.num_cochains = num_cochains

        if input_dim:
            output_dim = input_dim * num_cochains

            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim),
            )

            self.output_dim = output_dim
        else:
            self.model = model
        
            # Initialize weights
            self.apply(self._init_weights)

            # extract self.model output_dim
            self.output_dim = self.model[-1].out_features

    def _init_weights(self, m):
        """Initialize weights for better starting values"""
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.1)  # Non-zero bias
    
    def forward(self, x):
        """Forward pass with proper reshaping for all input dimensions"""
        original_shape = x.shape
        print(f"DEBUG - NeuralOneForm input shape: {original_shape}")

        # Check if using built-in sequential model with linear layers
        using_linear_model = isinstance(self.model[0], nn.Linear)
        
        if using_linear_model:
            # Handle reshaping for Linear layers (need batch, features format)
            if x.dim() == 1:  # Single point [2]
                x = x.unsqueeze(0)  # -> [1, 2]
            elif x.dim() == 2:  # Already in correct format [batch, features]
                pass
            elif x.dim() == 3:  # 3D format from chain [r, d, n]
                r, d, n = x.shape
                if n != self.input_dim:
                    # Log the dimension mismatch
                    print(f"WARNING: Feature dimension mismatch - got {n}, expected {self.input_dim}")
                    # Take only the needed features or pad if necessary
                    if n > self.input_dim:
                        x = x[:, :, :self.input_dim]
                    else:
                        x = torch.nn.functional.pad(x, (0, self.input_dim - n))
                # Reshape to [r*d, input_dim] for linear layers
                x = x.reshape(-1, self.input_dim)
                print(f"DEBUG - After reshape: {x.shape}")
        else:
            # Original reshaping logic for non-linear models
            if x.dim() == 1:  # Single point [2]
                x = x.unsqueeze(0).unsqueeze(0)  # -> [1, 1, 2]
            elif x.dim() == 2:  # Batch of points [batch, 2]
                x = x.unsqueeze(1)  # -> [batch, 1, 2]
            elif x.dim() == 3:  # Already in (r, d, n) format from chain
                r, d, n = x.shape
                x = x.reshape(-1, 1, n)  # Combine first two dims, add channel dim
        
        # Forward through the model
        out = self.model(x)
        
        # Reshape back based on original input dimensions
        if using_linear_model:
            if original_shape == (2,):  # Single point
                return out.squeeze(0)
            elif len(original_shape) == 2:  # Batch of points
                return out
            elif len(original_shape) == 3:  # From chain tensor
                r, d, _ = original_shape
                out = out.reshape(r, d, -1)
        else:
            if original_shape == (2,):  # Single point
                return out
            elif len(original_shape) == 2:  # Batch of points
                return out
            elif len(original_shape) == 3:  # From chain tensor
                # Reshape back to match expected output shape
                r, d, n = original_shape
                out = out.reshape(r, d, -1)  # The -1 will capture all output features
            
        return out
