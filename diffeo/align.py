import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()
print(f"Using device: {device}")

# =====================================================
# Source vector field
# =====================================================

def f(x):
    x1,x2 = x[:,0],x[:,1]
    return torch.stack([-x2,x1],dim=1)


# =====================================================
# Target vector field
# =====================================================

def g(y):
    y1,y2 = y[:,0],y[:,1]
    return torch.stack([-0.5*y2 + y1**3, y1],dim=1)


# =====================================================
# RealNVP coupling layer
# =====================================================

class Coupling(nn.Module):

    def __init__(self, mask):
        super().__init__()

        self.register_buffer("mask", mask)

        self.s = nn.Sequential(
            nn.Linear(2,64),
            nn.Tanh(),
            nn.Linear(64,2),
            nn.Tanh()
        )

        self.t = nn.Sequential(
            nn.Linear(2,64),
            nn.Tanh(),
            nn.Linear(64,2)
        )

    def forward(self,x):

        x_masked = x*self.mask

        s = self.s(x_masked)*(1-self.mask)
        t = self.t(x_masked)*(1-self.mask)

        y = x_masked + (1-self.mask)*(x*torch.exp(s)+t)

        return y


# =====================================================
# Flow (diffeomorphism)
# =====================================================

class Flow(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = nn.ModuleList([
            Coupling(torch.tensor([1.0, 0.0], dtype=torch.float32)),
            Coupling(torch.tensor([0.0, 1.0], dtype=torch.float32)),
            Coupling(torch.tensor([1.0, 0.0], dtype=torch.float32)),
            Coupling(torch.tensor([0.0, 1.0], dtype=torch.float32))
        ])

    def forward(self,x):

        for layer in self.layers:
            x = layer(x)

        return x


H = Flow().to(device)


# =====================================================
# Pushforward computation
# =====================================================

from torch.func import jvp

def pushforward(x):

    fx = f(x)

    y, pushed = jvp(H, (x,), (fx,))

    return y, pushed


# =====================================================
# Alignment loss
# =====================================================

def alignment_loss(x):

    y,p = pushforward(x)

    gy = g(y)

    p = p/(p.norm(dim=1,keepdim=True)+1e-8)
    gy = gy/(gy.norm(dim=1,keepdim=True)+1e-8)

    return ((p-gy)**2).mean()


# =====================================================
# Training
# =====================================================

opt = torch.optim.Adam(H.parameters(),lr=1e-3)

for step in range(3000):

    x = torch.randn(512, 2, device=device)

    loss = alignment_loss(x)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step%500==0:
        print(step,loss.item())


# =====================================================
# Visualization
# =====================================================

grid = torch.linspace(-2, 2, 20, device=device)
X, Y = torch.meshgrid(grid, grid, indexing='ij')

pts = torch.stack([X.flatten(), Y.flatten()], dim=1)

# source field
Vf = f(pts)

plt.figure()
plt.quiver(
    pts[:, 0].detach().cpu(),
    pts[:, 1].detach().cpu(),
    Vf[:, 0].detach().cpu(),
    Vf[:, 1].detach().cpu(),
)
plt.title("Source vector field f(x)")
plt.show()

# pushforward field
y,p = pushforward(pts)

plt.figure()
plt.quiver(
    y[:, 0].detach().cpu(),
    y[:, 1].detach().cpu(),
    p[:, 0].detach().cpu(),
    p[:, 1].detach().cpu(),
)
plt.title("Pushforward field H_* f")
plt.show()

# target field
Vg = g(pts)

plt.figure()
plt.quiver(
    pts[:, 0].detach().cpu(),
    pts[:, 1].detach().cpu(),
    Vg[:, 0].detach().cpu(),
    Vg[:, 1].detach().cpu(),
)
plt.title("Target vector field g(y)")
plt.show()
