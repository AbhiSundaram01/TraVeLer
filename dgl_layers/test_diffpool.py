import torch
import pytest
from .diffpool import BatchedDiffPool

# test_diffpool.py


@pytest.mark.parametrize("link_pred,entropy", [(False, False), (True, False), (False, True), (True, True)])
def test_batched_diffpool(link_pred, entropy):
    batch_size = 2
    num_nodes = 5
    in_features = 4
    next_nodes = 3
    hidden_dim = 6
    
    model = BatchedDiffPool(
        nfeat=in_features,
        nnext=next_nodes,
        nhid=hidden_dim,
        link_pred=link_pred,
        entropy=entropy
    )

    x = torch.randn(batch_size, num_nodes, in_features)
    adj = torch.randn(batch_size, num_nodes, num_nodes)

    xnext, anext = model(x, adj, log=True)
    assert xnext.shape == (batch_size, next_nodes, hidden_dim), "Output xnext has incorrect shape"
    assert anext.shape == (batch_size, next_nodes, next_nodes), "Output anext has incorrect shape"

    if link_pred or entropy:
        # Verify that the regularization losses were recorded
        assert len(model.reg_loss) > 0, "Model did not register any regularization layers"
        for loss_name, value in model.loss_log.items():
            assert value is not None, f"Loss value for {loss_name} was not computed"
    
    # Check logging
    assert 's' in model.log, "Assignment matrix not logged"
    assert 'a' in model.log, "Coarsened adjacency not logged"