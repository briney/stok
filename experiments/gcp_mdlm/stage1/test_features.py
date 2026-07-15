import numpy as np
import pytest

from experiments.gcp_mdlm.stage1 import provenance
from experiments.gcp_mdlm.stage1.features import CachedFeatures, write_feature_cache
from stok.data.paired_records import PairedRecord


def test_sha256_array_is_stable():
    a = np.arange(6, dtype=np.int64)
    assert provenance.sha256_array(a) == provenance.sha256_array(a.copy())


def test_write_and_load_cache_roundtrip(tmp_path):
    d_model = 5
    records = [
        PairedRecord("p1", "MKV", np.array([1, 2, 3]), np.array([True, True, True])),
        PairedRecord("p2", "AAAA", np.array([4, -1, 6, 7]), np.array([True, False, True, True])),
    ]

    def fake_encode(seq_tokens, key_padding_mask):
        # deterministic per-position features: value = token id, broadcast to d_model
        b, ll = seq_tokens.shape
        feats = seq_tokens.float().unsqueeze(-1).repeat(1, 1, d_model)
        return feats

    class _Tok:
        def encode(self, seq, add_special_tokens=False):
            return [ord(c) for c in seq]

    manifest = write_feature_cache(
        tmp_path, records, fake_encode, _Tok(), d_model=d_model, batch_size=8
    )
    # only valid residues are cached: 3 + 3 = 6
    assert manifest["n_residues"] == 6
    assert manifest["n_proteins"] == 2

    cache = CachedFeatures.load(tmp_path)
    assert cache.features.shape == (6, d_model)
    np.testing.assert_array_equal(cache.token_ids, np.array([1, 2, 3, 4, 6, 7]))
    assert cache.protein_ranges == [("p1", 0, 3), ("p2", 3, 3)]


def test_write_feature_cache_raises_on_tokenizer_residue_length_mismatch(tmp_path):
    d_model = 5
    records = [
        PairedRecord("bad-record", "MKV", np.array([1, 2, 3]), np.array([True, True, True])),
    ]

    def fake_encode(seq_tokens, key_padding_mask):
        b, ll = seq_tokens.shape
        return seq_tokens.float().unsqueeze(-1).repeat(1, 1, d_model)

    class _DroppingTok:
        """Stub tokenizer that silently collapses a character (e.g. into <unk>)."""

        def encode(self, seq, add_special_tokens=False):
            # Drops the last character, producing fewer tokens than residues.
            return [ord(c) for c in seq[:-1]]

    with pytest.raises(ValueError, match="bad-record"):
        write_feature_cache(
            tmp_path, records, fake_encode, _DroppingTok(), d_model=d_model, batch_size=8
        )


def test_build_run_manifest_assembles_four_hashes(tmp_path):
    corpus_path = tmp_path / "corpus.parquet"
    codebook_path = tmp_path / "codebook.pt"
    backbone_checkpoint = tmp_path / "backbone.ckpt"
    decoder_checkpoint = tmp_path / "decoder.ckpt"
    for path, content in (
        (corpus_path, b"corpus"),
        (codebook_path, b"codebook"),
        (backbone_checkpoint, b"backbone"),
        (decoder_checkpoint, b"decoder"),
    ):
        path.write_bytes(content)

    manifest = provenance.build_run_manifest(
        corpus_path=corpus_path,
        codebook_path=codebook_path,
        backbone_checkpoint=backbone_checkpoint,
        decoder_checkpoint=decoder_checkpoint,
    )

    assert manifest == {
        "corpus_sha256": provenance.sha256_file(corpus_path),
        "codebook_sha256": provenance.sha256_file(codebook_path),
        "backbone_checkpoint_sha256": provenance.sha256_file(backbone_checkpoint),
        "decoder_checkpoint_sha256": provenance.sha256_file(decoder_checkpoint),
    }
