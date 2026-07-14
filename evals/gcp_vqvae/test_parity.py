from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import torch

from stok.models.decoder import load_pretrained_decoder
from stok.models.structure_encoder import load_pretrained_encoder
from stok.utils.decoding import decode_coords
from stok.utils.structure_loader import NoAcceptedStructuresError, load_structures

EVAL_DIR = Path(__file__).resolve().parent
ORACLE_PATH = EVAL_DIR / "oracle_base.pt"
FIXTURE_ARCHIVE = EVAL_DIR / "cif_500.tar.gz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_dir(tmp_path: Path) -> Path:
    override = os.environ.get("STOK_GCP_VQVAE_CIF_DIR")
    if override:
        return Path(override).expanduser()

    assert FIXTURE_ARCHIVE.is_file(), f"Missing bundled fixtures: {FIXTURE_ARCHIVE}"
    shutil.unpack_archive(FIXTURE_ARCHIVE, tmp_path)
    return tmp_path / "cif_500"


def _configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def test_stok_encoder_matches_cached_gcp_vqvae_outputs(tmp_path: Path) -> None:
    assert ORACLE_PATH.is_file(), (
        f"Missing {ORACLE_PATH.name}; regenerate it with "
        "python -m evals.gcp_vqvae.generate_oracle"
    )

    fixture_dir = _fixture_dir(tmp_path)
    assert fixture_dir.is_dir(), f"Fixture directory does not exist: {fixture_dir}"
    assert torch.cuda.is_available(), "This local eval requires a CUDA GPU"
    oracle = torch.load(ORACLE_PATH, map_location="cpu", weights_only=True)
    assert oracle["schema_version"] == 2
    assert oracle["preset"] == "base"

    _configure_determinism()
    device = torch.device("cuda:0")
    checkpoint = os.environ.get("STOK_ENCODER_CHECKPOINT")
    encoder = load_pretrained_encoder(
        "base",
        path=checkpoint,
        device=device,
        freeze=True,
        progress=False,
    )
    actual_codebook = encoder.vector_quantizer._codebook.embed.detach().cpu()
    torch.testing.assert_close(actual_codebook, oracle["codebook"], rtol=0, atol=0)

    accepted = 0
    rejected = 0
    for record in oracle["samples"]:
        path = fixture_dir / record["filename"]
        assert path.is_file(), f"Missing fixture: {path}"
        assert _sha256(path) == record["sha256"], f"Fixture changed: {path.name}"
        try:
            loaded = load_structures(path, max_length=encoder.max_length, device=device)
        except NoAcceptedStructuresError:
            assert not record["accepted"], f"STōk rejected upstream sample {path.name}"
            rejected += 1
            continue

        assert record["accepted"], f"STōk accepted upstream-rejected sample {path.name}"
        assert loaded.sequences == [record["sequence"]], f"Sequence mismatch: {path.name}"
        with torch.inference_mode():
            output = encoder(loaded.graph, loaded.mask, loaded.nan_mask)

        length = len(record["sequence"])
        actual_valid = output["valid"][0, :length].detach().cpu()
        actual_indices = output["indices"][0, :length].detach().cpu()
        torch.testing.assert_close(actual_valid, record["valid"], rtol=0, atol=0)
        torch.testing.assert_close(
            actual_indices[actual_valid], record["indices"][record["valid"]], rtol=0, atol=0
        )
        expected_embeddings = oracle["codebook"][0, record["indices"][record["valid"]]]
        actual_embeddings = output["embeddings"][0, :length].detach().cpu()[actual_valid]
        torch.testing.assert_close(actual_embeddings, expected_embeddings, rtol=0, atol=0)
        accepted += 1

    assert accepted == 497
    assert rejected == 3


def test_stok_decoder_matches_cached_gcp_vqvae_outputs() -> None:
    assert torch.cuda.is_available(), "This local eval requires a CUDA GPU"
    oracle = torch.load(ORACLE_PATH, map_location="cpu", weights_only=True)
    assert oracle["schema_version"] == 2
    assert oracle["preset"] == "base"
    assert oracle["decoder_sample_count"] == 32
    assert oracle["decoder_max_length"] == 1280

    records = [record for record in oracle["samples"] if record["decoder_coords"].numel()]
    assert len(records) == 32
    lengths = [len(record["sequence"]) for record in records]
    assert (min(lengths), max(lengths)) == (35, 1055)
    assert sum(bool((~record["valid"]).any()) for record in records) == 13

    _configure_determinism()
    device = torch.device("cuda:0")
    checkpoint = os.environ.get("STOK_DECODER_CHECKPOINT")
    decoder = load_pretrained_decoder(
        "base",
        path=checkpoint,
        device=device,
        freeze=True,
        progress=False,
    )
    codebook = oracle["codebook"][0].to(device)
    max_length = int(oracle["decoder_max_length"])

    for record in records:
        length = len(record["sequence"])
        valid = record["valid"].bool()
        sample_indices = record["indices"].clone().long()
        sample_indices[~valid] = 0

        indices = torch.zeros((1, max_length), dtype=torch.long, device=device)
        mask = torch.zeros((1, max_length), dtype=torch.bool, device=device)
        indices[0, :length] = sample_indices.to(device)
        mask[0, :length] = valid.to(device)

        with torch.inference_mode():
            actual = decode_coords(decoder, codebook[indices], mask)[0, :length].cpu()
        expected = record["decoder_coords"]
        assert torch.isfinite(expected[valid]).all(), record["filename"]
        torch.testing.assert_close(
            actual[valid],
            expected[valid],
            rtol=0,
            atol=1e-5,
            msg=record["filename"],
        )
