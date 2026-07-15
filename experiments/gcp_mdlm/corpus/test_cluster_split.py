from experiments.gcp_mdlm.corpus.cluster_split import assign_splits


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
