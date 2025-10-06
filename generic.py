#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Synthetic data (Y-shape snapshot) --------------------
def sample_y_snapshot(n_per_branch=300, noise=0.03, seed=42):
    rng = np.random.default_rng(seed)
    # Stem (x from -1 to 0)
    x_stem = np.linspace(-1,0,n_per_branch)
    y_stem = np.zeros_like(x_stem)
    stem = np.stack([x_stem, y_stem], axis=1)
    # Up branch
    t = np.linspace(0,1,n_per_branch)
    up = np.stack([t, t], axis=1)
    # Down branch
    down = np.stack([t, -t], axis=1)
    data = np.concatenate([stem, up, down], axis=0)
    data = data + noise * rng.standard_normal(data.shape)
    return torch.tensor(data, dtype=torch.float32)

device = "cpu"
data = sample_y_snapshot(n_per_branch=200, noise=0.02, seed=0)
data_np = data.numpy()

# -------------------- GENERIC components --------------------
class EnergyNet(nn.Module):
    """Scalar energy E(x)."""
    def __init__(self, d, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

class EntropyNet(nn.Module):
    """Scalar entropy S(x)."""
    def __init__(self, d, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

class LNet(nn.Module):
    """Skew-symmetric operator L(x)."""
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.A = nn.Linear(d, d*d)  # unconstrained
    def forward(self, x):
        B = self.A(x).view(-1, self.d, self.d)
        return B - B.transpose(1,2)

class MNet(nn.Module):
    """Symmetric positive semidefinite operator M(x)."""
    def __init__(self, d, hidden=32, rank=4):
        super().__init__()
        self.d, self.rank = d, rank
        self.V = nn.Linear(d, d*rank)
    def forward(self, x):
        V = self.V(x).view(-1, self.d, self.rank)
        return torch.bmm(V, V.transpose(1,2))  # PSD

class GenericVNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.E = EnergyNet(d)
        self.S = EntropyNet(d)
        self.L = LNet(d)
        self.M = MNet(d)
    def forward(self, x):
        gradE = torch.autograd.grad(self.E(x).sum(), x, create_graph=True)[0]
        gradS = torch.autograd.grad(self.S(x).sum(), x, create_graph=True)[0]
        Lx = self.L(x)
        Mx = self.M(x)
        rev = torch.bmm(Lx, gradE.unsqueeze(-1)).squeeze(-1)
        irr = torch.bmm(Mx, gradS.unsqueeze(-1)).squeeze(-1)
        return rev + irr

# -------------------- Simple trajectory net --------------------
class TrajectoryNet(nn.Module):
    def __init__(self, d=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, d)
        )
    def forward(self, t, b):
        inp = torch.stack([t, b], dim=1)
        return self.net(inp)

# -------------------- Loss functions --------------------
def chamfer_loss(x, y):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    dist = ((x-y)**2).sum(-1)
    return dist.min(1)[0].mean() + dist.min(0)[0].mean()

def train_joint(num_epochs=200, lr=1e-3):
    d = 2
    Tnet = TrajectoryNet(d).to(device)
    Vnet = GenericVNet(d).to(device)
    opt = torch.optim.Adam(list(Tnet.parameters()) + list(Vnet.parameters()), lr=lr)

    losses = []
    for epoch in range(num_epochs):
        # Sample trajectory points
        t = torch.rand(256, device=device, requires_grad=True)
        b = torch.randint(0,3,(256,), device=device).float()
        traj = Tnet(t,b)

        # Shape loss
        L_shape = chamfer_loss(traj, data)

        # Alignment + consistency
        traj_t = torch.autograd.grad(traj.sum(), t, create_graph=True)[0]  # gives (256,) - scalar derivative for each point
        V = Vnet(traj)  # This gives (256, 2) - vector field
        
        # compute dt/dx for each component
        dtraj_dt = []
        for i in range(2):  # for x and y components
            grad_i = torch.autograd.grad(traj[:, i].sum(), t, create_graph=True)[0]
            dtraj_dt.append(grad_i)
        dtraj_dt = torch.stack(dtraj_dt, dim=1)  # Shape: (256, 2)
        
        L_cons = ((dtraj_dt - V)**2).mean()
        cos = F.cosine_similarity(V, dtraj_dt, dim=-1)
        L_align = (1 - cos).mean()

        loss = L_shape + 0.1*L_cons + 0.1*L_align
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
    return Tnet, Vnet, losses

Tnet, Vnet, losses = train_joint(num_epochs=50)

# -------------------- Visualization --------------------
plt.figure(figsize=(12,4))

# data and trajectories
plt.subplot(1,3,1)
plt.scatter(data_np[:,0], data_np[:,1], s=5, alpha=0.3, label="data")
t = torch.linspace(0,1,200)
for b in [0,1,2]:
    traj = Tnet(t, torch.full_like(t,float(b))).detach().numpy()
    plt.plot(traj[:,0], traj[:,1])
plt.title("Data + T-net trajectories")
plt.legend()

# vector field
plt.subplot(1,3,2)
grid_x, grid_y = np.meshgrid(np.linspace(-1,1,20), np.linspace(-1,1,20))
grid = torch.tensor(np.stack([grid_x.ravel(), grid_y.ravel()], axis=1), dtype=torch.float32, requires_grad=True)
V = Vnet(grid).detach().numpy()
grid_detached = grid.detach().numpy()
plt.quiver(grid_detached[:,0], grid_detached[:,1], V[:,0], V[:,1], angles="xy", scale=20)
plt.scatter(data_np[:,0], data_np[:,1], s=5, alpha=0.2)
plt.title("GENERIC V-net field")

# loss
plt.subplot(1,3,3)
plt.plot(losses)
plt.title("Training loss")

plt.tight_layout()
plt.show()

# %%
