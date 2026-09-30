import json
from pathlib import Path

import numpy as np
import pytest
import torch

FIXTURES = Path(__file__).parents[1] / "test_data/gcp_vqvae"


def test_reference_preparation_matches_offline_file_oracle():
    from stok.data.structure_encoding import prepare_reference_structure

    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for case in manifest["cases"]:
        if case["preset"] != "lite" or len(case.get("sources", [])) != 1:
            continue
        source = case["sources"][0]
        graph, residues, tokens = prepare_reference_structure(
            FIXTURES / source["path"], chain_id=source["chain_id"]
        )
        with np.load(FIXTURES / case["fixture"]["path"]) as arrays:
            length = case["lengths"][0]
            torch.testing.assert_close(
                graph.prepared_coordinates,
                torch.from_numpy(arrays["prepared_coordinates"]),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                graph.parsed_coordinates,
                torch.from_numpy(arrays["parsed_coordinates"]),
                rtol=0,
                atol=0,
                equal_nan=True,
            )
            assert residues.shape == tokens.shape == (1, 1280)
            assert torch.equal(
                residues[:, :64], torch.from_numpy(arrays["residue_mask"])
            )
            assert torch.equal(tokens[:, :64], torch.from_numpy(arrays["token_mask"]))
            assert not residues[:, length:].any() and not tokens[:, length:].any()
            assert torch.equal(graph.residue_index, torch.arange(length))
            assert not (graph.edge_index[0] == graph.edge_index[1]).any()
            assert torch.bincount(graph.edge_index[1]).tolist() == [16] * length


@pytest.mark.parametrize(
    "name, reason",
    [("too_short", "chains_too_short"), ("too_missing", "missing_ratio_exceeded")],
)
def test_reference_filtering_has_categorized_exclusions(name, reason):
    from stok.data.structure_encoding import (
        StructureExclusion,
        prepare_reference_structure,
    )

    with pytest.raises(StructureExclusion) as caught:
        prepare_reference_structure(FIXTURES / "inputs" / f"{name}.pdb", chain_id="A")
    assert caught.value.reason == reason
