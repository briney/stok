from experiments.gcp_mdlm.corpus.cluster_split import assign_splits, build_split_manifest


def _write_tsv(tmp_path, pairs):
    tsv = tmp_path / "clusters.tsv"
    tsv.write_text("".join(f"{rep}\t{mem}\n" for rep, mem in pairs))
    return tsv


def test_assign_splits_whole_cluster_no_crossing(tmp_path):
    # 3 clusters: {a,a2,a3}, {b,b2}, {c}
    tsv = _write_tsv(
        tmp_path, [("a", "a"), ("a", "a2"), ("a", "a3"), ("b", "b"), ("b", "b2"), ("c", "c")]
    )
    split = assign_splits(tsv, val_size=1, test_size=1, seed=0)
    # every member of a cluster shares one split
    for cluster in (["a", "a2", "a3"], ["b", "b2"], ["c"]):
        assert len({split[m] for m in cluster}) == 1
    assert set(split.values()) <= {"train", "val", "test"}
    assert sum(1 for v in split.values() if v == "val") >= 1
    assert sum(1 for v in split.values() if v == "test") >= 1


def test_assign_splits_deterministic(tmp_path):
    tsv = _write_tsv(tmp_path, [("a", "a"), ("b", "b"), ("c", "c"), ("d", "d")])
    assert assign_splits(tsv, val_size=1, test_size=1, seed=0) == assign_splits(
        tsv, val_size=1, test_size=1, seed=0
    )


def test_build_split_manifest_has_required_keys():
    # hand-built split map: {a, a2} -> train, {b} -> val, {c} -> test
    split = {"a": "train", "a2": "train", "b": "val", "c": "test"}
    manifest = build_split_manifest(
        min_seq_id=0.3,
        coverage=0.8,
        seed=0,
        val_size=1,
        test_size=1,
        n_sequences=4,
        split=split,
        version="18.8cc5c",
    )
    assert manifest["min_seq_id"] == 0.3
    assert manifest["coverage"] == 0.8
    assert manifest["mmseqs_version"] == "18.8cc5c"
    assert manifest["seed"] == 0
    assert manifest["val_size"] == 1
    assert manifest["test_size"] == 1
    assert manifest["n_sequences"] == 4
    assert manifest["split_counts"] == {"train": 2, "val": 1, "test": 1}


def test_mmseqs_version_falls_back_to_unknown_on_failure(monkeypatch):
    import subprocess

    from experiments.gcp_mdlm.corpus import cluster_split

    def _raise(*args, **kwargs):
        raise FileNotFoundError("mmseqs not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert cluster_split.mmseqs_version() == "unknown"
