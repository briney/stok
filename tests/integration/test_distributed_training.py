import json
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import os
import signal
import socket
import subprocess
import time

from tests.integration.test_training_progress import training_command, training_env


def run_distributed(command, *, timeout=30, env=None):
    """Launch real rank processes without torchrun's hostname/DNS dependency."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    workers = []
    deadline = time.monotonic() + timeout
    try:
        for rank in range(2):
            rank_env = {
                **(env or training_env()),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "WORLD_SIZE": "2",
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "LOCAL_WORLD_SIZE": "2",
            }
            workers.append(
                subprocess.Popen(
                    command,
                    env=rank_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
            )
        results = []
        for worker in workers:
            stdout, stderr = worker.communicate(
                timeout=max(0.1, deadline - time.monotonic())
            )
            results.append(
                subprocess.CompletedProcess(command, worker.returncode, stdout, stderr)
            )
        return results
    finally:
        for worker in workers:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGKILL)
            worker.communicate()


def test_two_rank_checkpoints_complete(tmp_path):
    project = tmp_path / "run"
    results = run_distributed(training_command(project))
    for result in results:
        assert result.returncode == 0, result.stdout + result.stderr
    assert (project / "checkpoints/step_00000002.pt").is_file()


@pytest.mark.parametrize(
    "failure,context",
    [
        ("checkpoint", "Checkpoint failed"),
        ("directories", "Creating project directories failed"),
        ("snapshot", "Saving configuration failed"),
        ("log", "Opening training log failed"),
        ("log-write", "Opening training log failed"),
    ],
)
def test_project_output_failure_reaches_every_rank(tmp_path, failure, context):
    from tests.integration.test_mdlm_training import training_fixture

    if failure == "log-write" and not os.path.exists("/dev/full"):
        pytest.skip("requires failing write sink")
    training_fixture(tmp_path)
    command = [
        sys.executable,
        "-m",
        "tests.utils.distributed_probe",
        "--case",
        "mdlm-uneven",
        "--output",
        str(tmp_path),
        "--failure",
        failure,
    ]
    for result in run_distributed(command):
        assert result.returncode != 0
        assert context in result.stderr, result.stdout + result.stderr
        assert (
            "injected output failure" in result.stderr
            if failure in {"checkpoint", "directories", "snapshot"}
            else True
        )


def write_probe_data(root, n=17, eval_n=5):
    rows = [
        {"sequence_id": str(i), "sequence": "LAG", "structure_tokens": [i] * 3}
        for i in range(n)
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, root / "train.parquet")
    (root / "shards").mkdir()
    pq.write_table(table.slice(0, 7), root / "shards/a.parquet")
    pq.write_table(table.slice(7), root / "shards/b.parquet")
    pq.write_table(table.slice(0, eval_n), root / "eval.parquet")
    (root / "eval_shards").mkdir()
    pq.write_table(table.slice(0, eval_n), root / "eval_shards/a.parquet")


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("source", ["map", "iterable", "map-mixture", "mixed-mixture"])
def test_native_training_stream_matches_single_rank(tmp_path, workers, source):
    write_probe_data(tmp_path)
    command = [
        sys.executable,
        "-m",
        "tests.utils.distributed_probe",
        "--case",
        "coverage",
        "--output",
        str(tmp_path),
        "--source",
        source,
        "--workers",
        str(workers),
    ]
    # Reference draws are ordered with no workers, for exact truncation before partitioning.
    result = subprocess.run(
        [*command[:-1], "0"],
        env=training_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    results = run_distributed(command)
    for result in results:
        assert result.returncode == 0, result.stderr
    records = [json.loads((tmp_path / f"rank_{r}.json").read_text()) for r in range(2)]
    assert records[0]["micro_steps"] == records[1]["micro_steps"]
    reference = json.loads((tmp_path / "reference.json").read_text())["ids"]
    total = len(reference) // (4 * max(1, workers)) * (4 * max(1, workers))
    assert total > 0
    assert sorted(records[0]["ids"] + records[1]["ids"]) == sorted(reference[:total])


@pytest.mark.parametrize("eval_n", [1, 5])
@pytest.mark.parametrize("source", ["map", "iterable"])
def test_native_eval_has_no_duplicates_or_dropped_tail(tmp_path, eval_n, source):
    write_probe_data(tmp_path, eval_n=eval_n)
    command = [
        sys.executable,
        "-m",
        "tests.utils.distributed_probe",
        "--case",
        "eval-tail",
        "--output",
        str(tmp_path),
        "--source",
        source,
        "--workers",
        "2",
    ]
    for result in run_distributed(command):
        assert result.returncode == 0, result.stderr
    ids = sum(
        [
            json.loads((tmp_path / f"rank_{r}.json").read_text())["ids"]
            for r in range(2)
        ],
        [],
    )
    assert sorted(ids) == list(range(eval_n))


@pytest.mark.parametrize(
    "coverage", ["uneven", "empty-modality", "empty-rank", "unused-head"]
)
def test_mdlm_two_rank_update_matches_global_reference(tmp_path, coverage):
    import torch
    from tests.integration.test_mdlm_training import training_fixture
    from tests.utils.synthetic import make_mdlm_rows, declare_synthetic_source

    rows = []
    for i in range(8):
        row = {
            **make_mdlm_rows()[1],
            "sequence_id": str(i),
            "sequence": "ACD",
            "structure_tokens": [0, 1, 2],
        }
        if i % 2 == 0:
            row["structure_tokens"] = (
                [0, None, None] if coverage == "uneven" else [None] * 3
            )
            if coverage == "empty-rank":
                row["sequence"] = "XXX"
        rows.append(declare_synthetic_source(row, source_accession=f"training-{i}"))
    training_fixture(tmp_path, rows=rows)
    command = [
        sys.executable,
        "-m",
        "tests.utils.distributed_probe",
        "--case",
        "mdlm-" + coverage,
        "--output",
        str(tmp_path),
        "--accum",
        "2",
    ]
    result = subprocess.run(
        command, env=training_env(), capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reference = torch.load(
        tmp_path / "run/model/final.pt", weights_only=False, map_location="cpu"
    )
    (tmp_path / "run").rename(tmp_path / "reference-run")
    for result in run_distributed(command):
        assert result.returncode == 0, result.stdout + result.stderr
    distributed = torch.load(
        tmp_path / "run/model/final.pt", weights_only=False, map_location="cpu"
    )
    assert distributed["global_step"] == 2 and distributed["micro_step"] == 4
    assert distributed["executed_positions"] == reference["executed_positions"] == 64
    assert distributed["residues_seen"] == reference["residues_seen"] == 24
    for name in reference["model"]:
        torch.testing.assert_close(
            distributed["model"][name], reference["model"][name], atol=3e-6, rtol=3e-5
        )
    if coverage == "unused-head":
        assert torch.count_nonzero(distributed["model"]["structure_bias"]) == 0
        assert all(
            s["step"].item() == 2 for s in distributed["optimizer"]["state"].values()
        )


@pytest.mark.parametrize(
    "case,context",
    [
        ("mdlm-bad-prepare", "Preparing MDLM window failed"),
        ("mdlm-bad-forward", "Training forward failed"),
    ],
)
def test_mdlm_rank_local_failure_terminates_every_rank(tmp_path, case, context):
    from tests.integration.test_mdlm_training import training_fixture

    training_fixture(tmp_path)
    command = [
        sys.executable,
        "-m",
        "tests.utils.distributed_probe",
        "--case",
        case,
        "--output",
        str(tmp_path),
    ]
    for result in run_distributed(command, timeout=20):
        assert result.returncode != 0
        assert context in result.stderr, result.stdout + result.stderr
        sentinel = "preparation" if case == "mdlm-bad-prepare" else "forward"
        assert f"injected rank-local MDLM {sentinel} failure" in result.stderr
    assert not list((tmp_path / "run/checkpoints").glob("step_*.pt"))
