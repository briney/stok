
import torch

from stok.utils.codebook import load_codebook


def test_path_overrides_preset(tmp_path):
    """Test that custom path overrides preset when loading codebook."""
    # Create a small, known tensor and save it
    custom = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    custom_path = tmp_path / "custom_codebook.pt"
    torch.save(custom, custom_path)

    # Call loader with both preset and path; path should override preset
    loaded = load_codebook(preset="base", path=str(custom_path))

    assert torch.allclose(loaded, custom)
    assert loaded.shape == (3, 4)


def test_preset_lite_loads():
    """Test that the lite preset codebook loads successfully."""
    # Ensure we can load the lite preset from package resources
    loaded = load_codebook(preset="lite")
    assert loaded.dim() == 2
    C, D = loaded.shape
    assert C > 0 and D > 0
