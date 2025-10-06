# Code Citations

## GNN Implementations

### License: unknown
https://github.com/achew012/diffpool-template/blob/2533bdbcb153d432e1f6828b2258fe61d162a7cb/src/model/GNN.py

```
class GNN(torch.nn.Module):
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
            x = F.relu(self.convs[
```

### License: unknown
https://github.com/Novmaple/GNN-Facade/blob/ea448c29f84887fbcc26bf315ffa2da82c8117fd/test.py

```
class GNN(torch.nn.Module):
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
            x = F.relu(self.convs[
```

### License: unknown
https://github.com/Nicolo-Giacopelli/Proteins_MLNS/blob/dcf205151fc6d3c259f12d9bc666a912684d0df7/models.py

```
class GNN(torch.nn.Module):
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
```

### License: unknown
https://github.com/achew012/diffpool-template/blob/2533bdbcb153d432e1f6828b2258fe61d162a7cb/DiffPool.py

```
64)

        num_nodes = ceil(0.25 * num_nodes)
```

### License: unknown
https://github.com/achew012/diffpool-template/blob/2533bdbcb153d432e1f6828b2258fe61d162a7cb/DiffPool.py

```
64)

        self.gnn2_pool
```

## Divergence-Free Neural Networks

### License: unknown
https://github.com/bmda-unibas/LagrangianFlowNetworks/blob/fc022b7b4fcaa0108dff86b78ebc749f604eb5be/experiments/optimal_transport/DFNN/divfree.py

```
""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
from functorch import make_functional
from functorch import vmap
from functorch import jacrev


def div(u):
    """Accepts a function u:R^D -> R^D."""
    J = jacrev(u)
    return lambda x: torch.trace(J(x))


def build_divfree_vector_field(module):
    """Returns an unbatched vector field, i.e. assumes input is a 1D tensor."""

    F_fn, params = make_functional(module)

    J_fn = jacrev(F_fn, argnums=1)

    def A_fn(params, x):
        J = J_fn(params, x)
        A = J - J.T
        return A

    def A_flat_fn(params, x):
        A = A_fn(params, x)
        A_flat = A.reshape(-1)
        return A_flat

    def ddF(params, x):
        D = x.nelement()
        dA_flat = jacrev(A_flat_fn, argnums=1)(params, x)
        Jac_all = dA_flat.reshape(D, D, D)
        ddF = vmap(torch.trace)(Jac_all)
        return ddF

    return ddF, params, A_fn


if __name__ == "__main__":

    torch.manual_seed(0)

    bsz =
```

### License: unknown
https://github.com/ZakariaJarraya/DivFree/blob/5b6da9ec5d6af7148cc64c3191e81b8ebe5ecda0/divergence_free.py

```
""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
from functorch import make_functional
from functorch import vmap
from functorch import jacrev


def div(u):
    """Accepts a function u:R^D -> R^D."""
    J = jacrev(u)
    return lambda x: torch.trace(J(x))


def build_divfree_vector_field(module):
    """Returns an unbatched vector field, i.e. assumes input is a 1D tensor."""

    F_fn, params = make_functional(module)

    J_fn = jacrev(F_fn, argnums=1)

    def A_fn(params, x):
        J = J_fn(params, x)
        A = J - J.T
        return A

    def A_flat_fn(params, x):
        A = A_fn(params, x)
        A_flat = A.reshape(-1)
        return A_flat

    def ddF(params, x):
        D = x.nelement()
        dA_flat = jacrev(A_flat_fn, argnums=1)(params, x)
        Jac_all = dA_flat.reshape(D, D, D)
        ddF = vmap(torch.trace)(Jac_all)
        return ddF

    return ddF, params, A_fn


if __name__ == "__main__":

    torch.manual_seed(0)

    bsz =
```

### License: unknown
https://github.com/ZakariaJarraya/DivFree/blob/5b6da9ec5d6af7148cc64c3191e81b8ebe5ecda0/divergence_free.py

```
.Linear(ndim, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, ndim),
    )

    u_fn, params, A_fn = build_divfree_vector_field(module)

    x = torch.randn(bsz, ndim)
    A = vmap(A_fn, in_dims=(None, 0))(params, x)
    print("A should be antisymmetric:")
    print(A.shape)
    print(A)

    u = vmap(u_fn, in_dims=(None, 0))(params, x)
    print("vector field u:")
    print(u.shape)
    print(u)

    div_u = div(lambda x: u_fn(params, x))
    d = vmap(div_u)(x)
    print("Divergence(u) should be zero:")
```

