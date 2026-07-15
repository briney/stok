import torch

from experiments.gcp_mdlm.stage1.crosscheck import build_crosscheck_model


def test_prototype_and_independent_models_forward():
    codebook = torch.randn(16, 8)
    proto = build_crosscheck_model(
        "codebook",
        vocab_size=32,
        pad_id=1,
        num_classes=16,
        codebook=codebook,
        d_model=32,
        n_heads=4,
        n_layers=2,
    )
    indep = build_crosscheck_model(
        "mlm",
        vocab_size=32,
        pad_id=1,
        num_classes=16,
        codebook=None,
        d_model=32,
        n_heads=4,
        n_layers=2,
    )
    # Token ids are placeholder input ids for this shape test; the independent
    # ("mlm") arm sizes its input embedding to num_classes (see crosscheck.py
    # docstring), so ids must stay < num_classes for both arms to be valid.
    tokens = torch.randint(0, 16, (2, 10))
    labels = torch.randint(0, 16, (2, 10))
    assert proto(tokens, labels=labels)["logits"].shape == (2, 10, 16)
    assert indep(tokens, labels=labels)["logits"].shape == (2, 10, 16)
