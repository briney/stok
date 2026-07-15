from experiments.gcp_mdlm.corpus.manifest import build_corpus_manifest, sha256_file


def test_sha256_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    import hashlib

    assert sha256_file(f) == hashlib.sha256(b"abc").hexdigest()


def test_build_manifest_has_required_keys(tmp_path):
    ckpt = tmp_path / "enc.pt"
    ckpt.write_bytes(b"w")
    m = build_corpus_manifest(
        encoder_checkpoint=ckpt,
        codebook_checkpoint=None,
        preset="base",
        min_mean_plddt=70.0,
        max_length=1280,
        mmseqs_params={"min_seq_id": 0.3, "coverage": 0.8, "version": "18"},
        split_seed=0,
        val_size=5000,
        test_size=5000,
    )
    assert m["preset"] == "base" and m["min_mean_plddt"] == 70.0
    assert m["encoder_checkpoint_sha256"] == sha256_file(ckpt)
    assert m["codebook_checkpoint_sha256"] == "<none>"
    assert m["mmseqs_params"]["min_seq_id"] == 0.3
    assert m["split"] == {"seed": 0, "val_size": 5000, "test_size": 5000}
