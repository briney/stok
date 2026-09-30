import torch
from omegaconf import OmegaConf

from stok.models.gcpnet import GCPInteractions


def test_tuple_dimensions_preserve_scalar_vector_outputs():
    cfg = OmegaConf.create(
        {
            "nonlinearities": ["silu", "silu"],
            "default_bottleneck": 1,
            "scalar_gate": 0,
            "vector_gate": True,
            "enable_e3_equivariance": False,
        }
    )
    layer_cfg = OmegaConf.create(
        {
            "pre_norm": True,
            "mp_cfg": {"self_message": True, "num_message_layers": 1},
            "use_scalar_message_attention": True,
            "use_gcp_norm": True,
            "use_gcp_dropout": True,
            "num_feedforward_layers": 1,
        }
    )
    layer = GCPInteractions((4, 1), (2, 1), cfg, layer_cfg)
    nodes = (torch.randn(3, 4), torch.randn(3, 1, 3))
    edges = (torch.randn(2, 2), torch.randn(2, 1, 3))
    output, positions = layer(
        nodes, edges, torch.tensor([[0, 1], [1, 2]]), torch.eye(3).repeat(2, 1, 1)
    )
    assert output.scalar.shape == (3, 4)
    assert output.vector.shape == (3, 1, 3)
    assert torch.isfinite(output.scalar).all()
    assert torch.isfinite(output.vector).all()
    assert positions is None
