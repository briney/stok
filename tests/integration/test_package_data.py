import importlib.resources as r


def test_packaged_configs_and_checkpoints_exist():
    """Test that packaged configs and checkpoint files exist."""
    root = r.files("stok")
    assert (root / "configs" / "config.yaml").is_file()
    assert (root / "checkpoints" / "codebook" / "base.pt").is_file()
    assert (root / "checkpoints" / "codebook" / "lite.pt").is_file()
    for preset in ("lite", "large"):
        assert (root / "configs" / "gcp_vqvae" / f"{preset}.yaml").is_file()

    assert (root / "configs/model/mdlm_150m.yaml").is_file()
    assert (root / "configs/train/mdlm_pilot.yaml").is_file()
