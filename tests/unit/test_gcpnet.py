import torch
from omegaconf import OmegaConf

from stok.models.gcpnet import GCPInteractions


def test_tuple_dimensions_preserve_scalar_vector_outputs():
    cfg = OmegaConf.create(
        {"nonlinearities": ["silu", "silu"], "default_bottleneck": 1, "bottleneck": 1}
    )
    layer = GCPInteractions((4, 1), (2, 1), cfg, num_message_layers=2)
    nodes = (torch.randn(3, 4), torch.randn(3, 1, 3))
    edges = (torch.randn(2, 2), torch.randn(2, 1, 3))
    output = layer(
        nodes, edges, torch.tensor([[0, 1], [1, 2]]), torch.eye(3).repeat(2, 1, 1)
    )
    assert output.scalar.shape == (3, 4)
    assert output.vector.shape == (3, 1, 3)
    assert torch.isfinite(output.scalar).all()
    assert torch.isfinite(output.vector).all()
