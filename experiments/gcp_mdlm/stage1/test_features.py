import numpy as np

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
