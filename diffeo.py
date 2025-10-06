import torch
import torch.nn as nn
import torch.optim as optim

class InvertibleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.s = nn.Sequential(nn.Linear(dim // 2, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 2))
        self.t = nn.Sequential(nn.Linear(dim // 2, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 2))

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1
        y2 = x2 * torch.exp(self.s(x1)) + self.t(x1)
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=-1)
        x1 = y1
        x2 = (y2 - self.t(y1)) * torch.exp(-self.s(y1))
        return torch.cat([x1, x2], dim=-1)

class DiffeoNet(nn.Module):
    def __init__(self, dim, n_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([InvertibleBlock(dim) for _ in range(n_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def inverse(self, y):
        for block in reversed(self.blocks):
            y = block.inverse(y)
        return y
    
class PullbackOneForm(nn.Module):
    def __init__(self, diffeo_net, oneform_net):
        super().__init__()
        self.diffeo_net = diffeo_net
        self.oneform_net = oneform_net

    def forward(self, uv):
        # Compute phi(u, v) -> (x, y, z)
        xyz = self.diffeo_net(uv)

        # Compute (f, g, h) at (x, y, z)
        fgh = self.oneform_net(xyz)  # Shape: (N, 3) -> (N, 3)

        # Compute Jacobian J = d(phi)/d(uv) (N, 3, 2)
        J_phi = torch.autograd.functional.jacobian(lambda uv: self.diffeo_net(uv).T, uv).permute(1, 2, 0)

        # Compute pullback coefficients p, q
        p = (fgh[:, 0] * J_phi[:, 0, 0] + fgh[:, 1] * J_phi[:, 1, 0] + fgh[:, 2] * J_phi[:, 2, 0])
        q = (fgh[:, 0] * J_phi[:, 0, 1] + fgh[:, 1] * J_phi[:, 1, 1] + fgh[:, 2] * J_phi[:, 2, 1])

        return torch.stack([p, q], dim=1)  # (N,2)

class OneFormNet(nn.Module):
    """Neural network that outputs a 1-form (3 components for 3D space)"""
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.net(x)

