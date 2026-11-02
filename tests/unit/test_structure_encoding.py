import json
from dataclasses import replace
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


def example_polymer(name="complete_pdb", *, missing=False):
    from stok.utils.structure_parser import parse_polymer_structure

    structure = parse_polymer_structure(
        FIXTURES / "inputs" / f"{name}.pdb", chain_id="A", allow_observed_sequence=True
    )
    if missing:
        coordinates = structure.coordinates.copy()
        coordinates[4, 3] = np.nan
        coordinates[12] = np.nan
        mapping = [dict(row) for row in structure.residue_map]
        mapping[12].update(observed_monomer_id=None, observed_one_letter=None)
        structure = replace(
            structure,
            coordinates=coordinates,
            atom_mask=np.isfinite(coordinates).all(-1),
            residue_map=tuple(mapping),
        )
    return structure


def test_polymer_preparation_retains_immutable_observations_and_distinct_masks():
    from graphein.protein.resi_atoms import STANDARD_AMINO_ACIDS
    from stok.data.structure_encoding import prepare_structure
    from stok.utils.featurizer import ProteinFeaturiser

    source = example_polymer(missing=True)
    original = source.coordinates.copy()
    native, residues, tokens = prepare_structure(
        source, sequence_mode="native", imputation="reference"
    )
    unknown, other_residues, other_tokens = prepare_structure(
        source, sequence_mode="unknown", imputation="reference"
    )
    length = len(source.sequence)
    assert residues.shape == tokens.shape == (1, 1280)
    assert residues[0, :length].all() and not residues[0, length:].any()
    assert not tokens[0, [4, 12]].any() and tokens[0].sum() == length - 2
    assert native.atom_mask.shape == (1, 1280, 4)
    assert native.atom_mask[0, 4].tolist() == [True, True, True, False]
    assert native.geometry_mask[0, 4] and not native.geometry_mask[0, 12]
    assert native.graph_node_mask[0, :length].all()
    assert native.residue_type[4].item() == STANDARD_AMINO_ACIDS.index(
        source.sequence[4]
    )
    assert native.residue_type[12].item() == STANDARD_AMINO_ACIDS.index("X")
    assert (unknown.residue_type == STANDARD_AMINO_ACIDS.index("X")).all()
    assert torch.equal(native.prepared_coordinates, unknown.prepared_coordinates)
    assert torch.equal(residues, other_residues) and torch.equal(tokens, other_tokens)
    encoded = ProteinFeaturiser()(unknown.clone()).x[:, 16:39]
    assert (
        encoded[:, STANDARD_AMINO_ACIDS.index("X")].eq(1).all()
        and encoded.sum(-1).eq(1).all()
    )
    assert np.array_equal(source.coordinates, original, equal_nan=True)
    assert source.sequence[4] != "X" and source.sequence[12] != "X"


def test_high_level_inference_preserves_length_holes_order_and_singleton_context():
    from stok.data.structure_encoding import tokenize_structures
    from stok.models.gcp_vqvae import GCPVQTokenizer
    from tests.unit.test_gcp_vqvae import tiny_config

    config = tiny_config()
    config["max_length"] = 1280
    model = GCPVQTokenizer(config).eval()
    sources = [example_polymer(missing=True), example_polymer("shorter")]
    state = {key: value.clone() for key, value in model.state_dict().items()}
    batched = tokenize_structures(
        model, sources, sequence_mode="native", imputation="reference"
    )
    singles = [
        tokenize_structures(
            model, [source], sequence_mode="native", imputation="reference"
        )[0]
        for source in sources
    ]
    assert [len(ids) for ids in batched] == [40, 32]
    assert batched[0][4].item() == batched[0][12].item() == -1
    assert all(torch.equal(batch, single) for batch, single in zip(batched, singles))
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in state.items()
    )
    short = replace(
        sources[1],
        sequence=sources[1].sequence[:4],
        coordinates=sources[1].coordinates[:4],
        atom_mask=sources[1].atom_mask[:4],
        residue_map=sources[1].residue_map[:4],
    )
    assert tokenize_structures(
        model, [short], sequence_mode="native", imputation="reference", min_length=4
    )[0].shape == (4,)


def test_unusable_structure_and_incomplete_metadata_are_rejected():
    from stok.data.structure_encoding import StructureExclusion, prepare_structure

    source = example_polymer()
    coordinates = np.full_like(source.coordinates, np.nan)
    missing = replace(
        source, coordinates=coordinates, atom_mask=np.zeros_like(source.atom_mask)
    )
    with pytest.raises(StructureExclusion, match="no_usable_structure"):
        prepare_structure(missing, sequence_mode="native", imputation="reference")
    with pytest.raises(ValueError, match="metadata"):
        prepare_structure(
            replace(source, source={}), sequence_mode="native", imputation="reference"
        )
    with pytest.raises(ValueError, match="metadata"):
        prepare_structure(
            replace(source, source=dict(source.source) | {"sha256": ""}),
            sequence_mode="native",
            imputation="reference",
        )
    for mode, imputation in (("invalid", "reference"), ("native", "invalid")):
        with pytest.raises(ValueError):
            prepare_structure(source, sequence_mode=mode, imputation=imputation)


def test_coverage_limits_can_be_disabled_without_disabling_length_or_usable_checks():
    from stok.data.structure_encoding import StructureExclusion, prepare_structure

    source = example_polymer()
    coordinates = source.coordinates.copy()
    coordinates[5:25] = np.nan
    source = replace(
        source, coordinates=coordinates, atom_mask=np.isfinite(coordinates).all(-1)
    )
    disabled = dict(max_missing_ratio=None, max_missing_block=None)
    graph, residues, tokens = prepare_structure(
        source, sequence_mode="native", imputation="reference", **disabled
    )
    assert torch.isfinite(graph.prepared_coordinates).all()
    assert residues.sum() == 40 and tokens.sum() == 20
    assert not tokens[0, 5:25].any()
    for limits, reason in (
        (dict(disabled, max_missing_ratio=0.49), "missing_ratio_exceeded"),
        (dict(disabled, max_missing_block=19), "missing_block_exceeded"),
        (dict(disabled, min_length=41), "chains_too_short"),
        (dict(disabled, max_length=39), "chains_too_long"),
    ):
        with pytest.raises(StructureExclusion, match=reason):
            prepare_structure(
                source, sequence_mode="native", imputation="reference", **limits
            )
    missing = replace(
        source,
        coordinates=np.full_like(coordinates, np.nan),
        atom_mask=np.zeros_like(source.atom_mask),
    )
    with pytest.raises(StructureExclusion, match="no_usable_structure"):
        prepare_structure(
            missing, sequence_mode="native", imputation="reference", **disabled
        )
    sparse_coordinates = np.full_like(coordinates, np.nan)
    sparse_coordinates[:3] = source.coordinates[:3]
    sparse = replace(
        source,
        coordinates=sparse_coordinates,
        atom_mask=np.isfinite(sparse_coordinates).all(-1),
    )
    with pytest.raises(StructureExclusion, match="too_few_graph_nodes"):
        prepare_structure(
            sparse, sequence_mode="native", imputation="observed_only", **disabled
        )


def test_manifest_paths_and_defaults_are_explicit(tmp_path):
    from stok.data.structure_encoding import iter_structure_manifest

    source = tmp_path / "input.pdb"
    source.write_text((FIXTURES / "inputs/complete_pdb.pdb").read_text())
    manifest = tmp_path / "chains.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sequence_id": "chain-1",
                "path": "input.pdb",
                "chain_id": "A",
                "source_namespace": "pdb",
                "source_accession": "example",
            }
        )
        + "\n"
    )
    (row,) = iter_structure_manifest(manifest)
    assert row["path"] == str(source.resolve())
    assert row["model_index"] == 0 and row["chain_namespace"] == "author"


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate_parent",
        "source_namespace",
        "source_accession",
        "id",
        "path",
        "namespace",
        "namespace_type",
        "model",
        "sequence",
        "chain",
        "unknown",
        "json",
    ],
)
def test_manifest_invalid_rows_include_path_and_line_context(tmp_path, fault):
    from stok.data.structure_encoding import iter_structure_manifest

    row = {
        "sequence_id": "one",
        "path": str(FIXTURES / "inputs/complete_pdb.pdb"),
        "source_namespace": "pdb",
        "source_accession": "example",
    }
    field_values = {
        "source_namespace": ("source_namespace", ""),
        "source_accession": ("source_accession", " example "),
        "id": ("sequence_id", 3),
        "path": ("path", "missing.pdb"),
        "namespace": ("chain_namespace", "other"),
        "namespace_type": ("chain_namespace", []),
        "model": ("model_index", True),
        "sequence": ("sequence", 1),
        "chain": ("chain_id", 3),
        "unknown": ("crop_length", 30),
    }
    if fault in field_values:
        field, value = field_values[fault]
        row[field] = value
    text = json.dumps(row) + "\n"
    if fault == "duplicate_parent":
        row["parent_ids"] = ["a" * 64] * 2
        text = json.dumps(row) + "\n"
    elif fault == "json":
        text = "{invalid}\n"
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(text)
    with pytest.raises(ValueError, match=f"{manifest.name}:[12]"):
        list(iter_structure_manifest(manifest))


def test_linear_filling_preserves_observed_atoms_and_rigid_transforms():
    from stok.data.structure_encoding import impute_linear_coordinates

    torch.manual_seed(8)
    original = torch.randn(16, 4, 3)
    original[:2] = float("nan")
    original[5:8] = float("nan")
    original[-2:] = float("nan")
    original[4, 3] = float("nan")
    mask = torch.isfinite(original).all(-1)
    filled = impute_linear_coordinates(original)
    assert torch.equal(filled[mask], original[mask])
    assert torch.isfinite(filled).all()
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    translation = torch.tensor([11.0, -8.0, 3.0])
    transformed = original @ rotation + translation
    torch.testing.assert_close(
        impute_linear_coordinates(transformed),
        filled @ rotation + translation,
        rtol=1e-5,
        atol=1e-5,
    )
    coincident = torch.ones(8, 4, 3)
    coincident[2:5] = float("nan")
    assert torch.equal(
        impute_linear_coordinates(coincident), torch.ones_like(coincident)
    )
    with pytest.raises(ValueError, match="anchors"):
        impute_linear_coordinates(torch.full_like(original, float("nan")))


def test_observed_only_graphs_keep_positions_and_mask_all_cross_gap_stencils():
    from stok.data.structure_encoding import prepare_structure
    from stok.utils.featurizer import ProteinFeaturiser

    source = example_polymer(missing=True)
    graph, residues, tokens = prepare_structure(
        source, sequence_mode="native", imputation="observed_only"
    )
    assert graph.num_nodes == 38 and torch.equal(graph.graph_node_mask, tokens)
    positions = graph.residue_index
    assert not (positions == 4).any() and not (positions == 12).any()
    assert torch.equal(positions, torch.where(tokens[0])[0])
    assert torch.equal(graph.seq_pos[:, 0], positions)
    assert not (graph.edge_index[0] == graph.edge_index[1]).any()
    # Spatial edges still connect nodes across missing sequence positions.
    edge_positions = positions[graph.edge_index]
    assert ((edge_positions[0] < 12) & (edge_positions[1] > 12)).any()
    features = ProteinFeaturiser()(graph.clone())
    before_gap = (positions == 11).nonzero().item()
    after_gap = (positions == 13).nonzero().item()
    assert not features.x_vector_attr[before_gap, 0].any()
    assert not features.x_vector_attr[after_gap, 1].any()
    assert not features.x[before_gap, 39:41].any()  # alpha crosses i+1
    assert not features.x[before_gap, 45:49].any()  # psi/omega cross i+1
    assert not features.x[after_gap, 43:45].any()  # phi crosses i-1
    # Every kappa/alpha stencil crossing the gap is suppressed, including
    # an omitted intermediate residue that is not itself a kappa anchor.
    assert not features.x[before_gap, 41:43].any()
    assert residues[0, 12] and not tokens[0, 12]


def test_reference_fill_displacement_and_orientation_dependence_are_recorded_baseline():
    from stok.data.structure_encoding import impute_reference_coordinates

    source = example_polymer(missing=True)
    original = torch.tensor(source.coordinates)
    propagated = original.clone()
    propagated[~torch.isfinite(propagated).all((1, 2))] = float("nan")
    filled, _ = impute_reference_coordinates(propagated)
    atom_mask = torch.isfinite(original).all(-1)
    assert (filled[atom_mask] - original[atom_mask]).norm(dim=-1).max() > 0.1
    # A short missing segment's arc uses a fixed Cartesian reference axis.
    original = torch.tensor(source.coordinates)
    original[12:15] = float("nan")
    original[4] = torch.tensor(example_polymer().coordinates[4])
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    filled, _ = impute_reference_coordinates(original)
    rotated, _ = impute_reference_coordinates(original @ rotation)
    assert not torch.allclose(rotated, filled @ rotation, rtol=1e-5, atol=1e-5)
