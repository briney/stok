"""An explicit local oracle is required; a skip is not parity evidence."""

import os
from pathlib import Path

import pytest

from tests.reference.generate_gcp_vqvae import verify_fixture_set


def test_published_weight_oracle_inventory():
    configured = os.environ.get("STOK_GCP_REFERENCE_FIXTURES")
    if not configured:
        pytest.skip(
            "Set STOK_GCP_REFERENCE_FIXTURES to a full locally generated oracle; published-weight parity is unverified"
        )
    manifest = verify_fixture_set(Path(configured), require_models=True)
    assert manifest["environment"]["dtype"] == "float32"
