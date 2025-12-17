"""Unit tests for the metric registry and factory."""

import pytest
from omegaconf import OmegaConf

from stok.eval.base import MetricBase
from stok.eval.registry import (
    METRIC_REGISTRY,
    build_metrics,
    get_registered_metrics,
    register_metric,
)


def test_metric_registry_not_empty():
    """Test that the registry contains metrics after import."""
    # Import to trigger registration
    import stok.eval.metrics  # noqa: F401

    assert len(METRIC_REGISTRY) > 0
    assert "accuracy" in METRIC_REGISTRY
    assert "perplexity" in METRIC_REGISTRY


def test_register_metric_decorator():
    """Test the register_metric decorator."""
    # Create a unique name to avoid conflicts
    test_name = "_test_metric_decorator"

    @register_metric(test_name)
    class TestMetric(MetricBase):
        name = test_name
        objectives = {"test"}

        def update(self, outputs, tokens, labels, coords, cfg):
            pass

        def compute(self):
            return {}

        def reset(self):
            pass

    assert test_name in METRIC_REGISTRY
    assert METRIC_REGISTRY[test_name] is TestMetric

    # Cleanup
    del METRIC_REGISTRY[test_name]


def test_register_metric_duplicate_raises():
    """Test that registering a duplicate name raises an error."""
    test_name = "_test_duplicate"

    @register_metric(test_name)
    class TestMetric1(MetricBase):
        name = test_name

        def update(self, outputs, tokens, labels, coords, cfg):
            pass

        def compute(self):
            return {}

        def reset(self):
            pass

    with pytest.raises(ValueError, match="already registered"):

        @register_metric(test_name)
        class TestMetric2(MetricBase):
            name = test_name

            def update(self, outputs, tokens, labels, coords, cfg):
                pass

            def compute(self):
                return {}

            def reset(self):
                pass

    # Cleanup
    del METRIC_REGISTRY[test_name]


def test_get_registered_metrics():
    """Test that get_registered_metrics returns a copy of the registry."""
    registry = get_registered_metrics()
    assert isinstance(registry, dict)
    assert len(registry) > 0

    # Should be a copy, not the original
    registry["_fake"] = None
    assert "_fake" not in METRIC_REGISTRY


def test_build_metrics_filters_by_objective():
    """Test that build_metrics filters metrics by objective."""
    cfg = OmegaConf.create(
        {
            "train": {"eval": {"metrics": {}}},
            "data": {},
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    # Build for codebook objective
    codebook_metrics = build_metrics(cfg, objective="codebook")
    codebook_names = {type(m).__name__ for m in codebook_metrics}

    # Build for mlm objective
    mlm_metrics = build_metrics(cfg, objective="mlm")
    mlm_names = {type(m).__name__ for m in mlm_metrics}

    # AccuracyMetric should only be in codebook
    assert "AccuracyMetric" in codebook_names
    assert "AccuracyMetric" not in mlm_names

    # MaskedAccuracyMetric should only be in mlm
    assert "MaskedAccuracyMetric" in mlm_names
    assert "MaskedAccuracyMetric" not in codebook_names

    # Perplexity should be in both
    assert "PerplexityMetric" in codebook_names
    assert "PerplexityMetric" in mlm_names


def test_build_metrics_respects_enabled_flag():
    """Test that build_metrics respects the enabled flag."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "accuracy": {"enabled": False},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {},
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    metrics = build_metrics(cfg, objective="codebook")
    metric_names = {type(m).__name__ for m in metrics}

    # Accuracy disabled, perplexity enabled
    assert "AccuracyMetric" not in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_filters_by_decoder_requirement():
    """Test that metrics requiring decoder are filtered when decoder is None."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "lddt": {"enabled": True},
                        "accuracy": {"enabled": True},
                    }
                }
            },
            "data": {},
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    # Without decoder
    metrics_no_decoder = build_metrics(cfg, objective="codebook", decoder=None)
    names_no_decoder = {type(m).__name__ for m in metrics_no_decoder}

    # LDDTMetric requires decoder
    assert "LDDTMetric" not in names_no_decoder
    assert "AccuracyMetric" in names_no_decoder


def test_build_metrics_filters_by_coords_requirement():
    """Test that metrics requiring coords are filtered when coords unavailable."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "lddt": {"enabled": True},
                        "accuracy": {"enabled": True},
                    }
                }
            },
            "data": {},
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    # Create a mock decoder
    class MockDecoder:
        pass

    decoder = MockDecoder()

    # Without coords
    metrics_no_coords = build_metrics(
        cfg, objective="codebook", decoder=decoder, has_coords=False
    )
    names_no_coords = {type(m).__name__ for m in metrics_no_coords}

    # LDDTMetric requires coords
    assert "LDDTMetric" not in names_no_coords


def test_build_metrics_passes_config_params():
    """Test that metric-specific config params are passed to constructor."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {
                            "enabled": True,
                            "contact_threshold": 6.0,
                            "min_seq_sep": 8,
                        },
                    }
                }
            },
            "data": {"load_coords": True},
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    metrics = build_metrics(cfg, objective="mlm", has_coords=True)
    p_at_l = next((m for m in metrics if type(m).__name__ == "PrecisionAtLMetric"), None)

    assert p_at_l is not None
    assert p_at_l.contact_threshold == 6.0
    assert p_at_l.min_seq_sep == 8


def test_build_metrics_only_whitelist():
    """Test that 'only' list whitelists specific metrics for a dataset."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "accuracy": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "eval": {
                    "seq_val": {
                        "path": "/path/to/seq_val.parquet",
                        "metrics": {
                            "only": ["accuracy"],  # Only accuracy for this dataset
                        },
                    }
                }
            },
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    # Build for seq_val dataset with 'only' whitelist
    metrics = build_metrics(cfg, objective="codebook", eval_name="seq_val")
    metric_names = {type(m).__name__ for m in metrics}

    # Only accuracy should be enabled
    assert "AccuracyMetric" in metric_names
    assert "PerplexityMetric" not in metric_names


def test_build_metrics_only_whitelist_multiple():
    """Test 'only' list with multiple metrics."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "accuracy": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "eval": {
                    "test_val": {
                        "path": "/path/to/test.parquet",
                        "metrics": {
                            "only": ["accuracy", "perplexity"],  # Both allowed
                        },
                    }
                }
            },
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    metrics = build_metrics(cfg, objective="codebook", eval_name="test_val")
    metric_names = {type(m).__name__ for m in metrics}

    # Both should be enabled
    assert "AccuracyMetric" in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_only_with_override():
    """Test that per-metric enabled can override 'only' list."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "accuracy": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "eval": {
                    "hybrid_val": {
                        "path": "/path/to/hybrid.parquet",
                        "metrics": {
                            "only": ["accuracy"],  # Whitelist only accuracy
                            "perplexity": {"enabled": True},  # But explicitly enable perplexity
                        },
                    }
                }
            },
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    metrics = build_metrics(cfg, objective="codebook", eval_name="hybrid_val")
    metric_names = {type(m).__name__ for m in metrics}

    # Both should be enabled (perplexity via override)
    assert "AccuracyMetric" in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_only_disable_override():
    """Test that per-metric enabled=false can override 'only' list inclusion."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "accuracy": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "eval": {
                    "selective_val": {
                        "path": "/path/to/selective.parquet",
                        "metrics": {
                            "only": ["accuracy", "perplexity"],  # Both in whitelist
                            "accuracy": {"enabled": False},  # But disable accuracy
                        },
                    }
                }
            },
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    metrics = build_metrics(cfg, objective="codebook", eval_name="selective_val")
    metric_names = {type(m).__name__ for m in metrics}

    # Only perplexity should be enabled
    assert "AccuracyMetric" not in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_per_dataset_has_coords_load_coords():
    """Test per-dataset load_coords overrides global has_coords."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "load_coords": False,  # Global: no coords
                "eval": {
                    "struct_val": {
                        "path": "/path/to/struct.parquet",
                        "load_coords": True,  # Per-dataset: has coords
                    }
                },
            },
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    # Build for struct_val with per-dataset load_coords=True
    metrics = build_metrics(
        cfg, objective="mlm", has_coords=False, eval_name="struct_val"
    )
    metric_names = {type(m).__name__ for m in metrics}

    # PrecisionAtLMetric requires coords, should be enabled due to per-dataset override
    assert "PrecisionAtLMetric" in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_per_dataset_has_coords_explicit():
    """Test per-dataset has_coords key (alternative to load_coords)."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "load_coords": True,  # Global: has coords
                "eval": {
                    "seq_val": {
                        "path": "/path/to/seq.parquet",
                        "has_coords": False,  # Per-dataset: no coords
                    }
                },
            },
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    # Build for seq_val with per-dataset has_coords=False
    metrics = build_metrics(
        cfg, objective="mlm", has_coords=True, eval_name="seq_val"
    )
    metric_names = {type(m).__name__ for m in metrics}

    # PrecisionAtLMetric requires coords, should be excluded
    assert "PrecisionAtLMetric" not in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_no_per_dataset_override_uses_global():
    """Test that without per-dataset override, global has_coords is used."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                    }
                }
            },
            "data": {
                "load_coords": True,  # Global has coords
                "eval": {
                    "default_val": {
                        "path": "/path/to/default.parquet",
                        # No load_coords or has_coords override
                    }
                },
            },
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    # Build with global has_coords=True
    metrics = build_metrics(
        cfg, objective="mlm", has_coords=True, eval_name="default_val"
    )
    metric_names = {type(m).__name__ for m in metrics}

    # Should use global has_coords=True
    assert "PrecisionAtLMetric" in metric_names


def test_build_metrics_combined_only_and_has_coords():
    """Test combining 'only' whitelist with per-dataset has_coords."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "accuracy": {"enabled": True},
                        "perplexity": {"enabled": True},
                        "p_at_l": {"enabled": True},
                    }
                }
            },
            "data": {
                "eval": {
                    "seq_only": {
                        "path": "/path/to/seq.parquet",
                        "load_coords": False,
                        "metrics": {
                            "only": ["accuracy", "perplexity"],
                        },
                    },
                    "struct_only": {
                        "path": "/path/to/struct.parquet",
                        "load_coords": True,
                        "metrics": {
                            "only": ["accuracy", "p_at_l"],
                        },
                    },
                },
            },
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    # Sequence-only dataset
    seq_metrics = build_metrics(
        cfg, objective="mlm", has_coords=False, eval_name="seq_only"
    )
    seq_names = {type(m).__name__ for m in seq_metrics}

    assert "AccuracyMetric" not in seq_names  # Not in mlm objective
    assert "MaskedAccuracyMetric" not in seq_names  # Not in 'only' list
    assert "PerplexityMetric" in seq_names  # In 'only' list

    # Structure dataset
    struct_metrics = build_metrics(
        cfg, objective="mlm", has_coords=False, eval_name="struct_only"
    )
    struct_names = {type(m).__name__ for m in struct_metrics}

    assert "PrecisionAtLMetric" in struct_names  # In 'only' list, has coords via override


def test_build_metrics_structure_format_has_coords():
    """Test that format='structure' automatically enables has_coords."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "load_coords": False,  # Global: no coords
                "eval": {
                    "cameo": {
                        "path": "/path/to/pdb_folder",
                        "format": "structure",  # Structure folder format
                        # No explicit load_coords or has_coords
                    }
                },
            },
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    # Build for cameo dataset with format="structure"
    metrics = build_metrics(
        cfg, objective="mlm", has_coords=False, eval_name="cameo"
    )
    metric_names = {type(m).__name__ for m in metrics}

    # PrecisionAtLMetric requires coords, should be enabled due to format="structure"
    assert "PrecisionAtLMetric" in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_auto_detect_structure_folder(tmp_path):
    """Test auto-detection of structure folder by checking for PDB files."""
    # Create a temporary folder with PDB files
    pdb_folder = tmp_path / "pdb_eval"
    pdb_folder.mkdir()
    (pdb_folder / "test1.pdb").write_text("ATOM placeholder")
    (pdb_folder / "test2.pdb").write_text("ATOM placeholder")

    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "load_coords": False,  # Global: no coords
                "eval": {
                    "pdb_val": {
                        "path": str(pdb_folder),
                        # No format specified - should auto-detect
                    }
                },
            },
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    # Build for pdb_val dataset (should auto-detect structure folder)
    metrics = build_metrics(
        cfg, objective="mlm", has_coords=False, eval_name="pdb_val"
    )
    metric_names = {type(m).__name__ for m in metrics}

    # PrecisionAtLMetric requires coords, should be enabled due to auto-detection
    assert "PrecisionAtLMetric" in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_auto_detect_mmcif_folder(tmp_path):
    """Test auto-detection of structure folder with mmCIF files."""
    # Create a temporary folder with mmCIF files
    cif_folder = tmp_path / "cif_eval"
    cif_folder.mkdir()
    (cif_folder / "structure.cif").write_text("_entry.id placeholder")

    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                    }
                }
            },
            "data": {
                "load_coords": False,
                "eval": {
                    "cif_val": {
                        "path": str(cif_folder),
                    }
                },
            },
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    metrics = build_metrics(
        cfg, objective="mlm", has_coords=False, eval_name="cif_val"
    )
    metric_names = {type(m).__name__ for m in metrics}

    # Should auto-detect mmCIF folder
    assert "PrecisionAtLMetric" in metric_names


def test_build_metrics_no_auto_detect_for_parquet_folder(tmp_path):
    """Test that parquet folders are not auto-detected as structure folders."""
    # Create a temporary folder with parquet files (not structure files)
    parquet_folder = tmp_path / "parquet_eval"
    parquet_folder.mkdir()
    (parquet_folder / "data.parquet").write_bytes(b"parquet placeholder")

    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                    }
                }
            },
            "data": {
                "load_coords": False,
                "eval": {
                    "parquet_val": {
                        "path": str(parquet_folder),
                    }
                },
            },
            "model": {"classifier": {"ignore_index": -100}},
        }
    )

    metrics = build_metrics(
        cfg, objective="mlm", has_coords=False, eval_name="parquet_val"
    )
    metric_names = {type(m).__name__ for m in metrics}

    # Should NOT auto-detect as structure folder (no PDB/CIF files)
    assert "PrecisionAtLMetric" not in metric_names

