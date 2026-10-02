import json
import sys
import pandas as pd
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
    results = run_distributed(training_command(tmp_path))
    for result in results:
        assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "checkpoints/step_00000002.pt").is_file()


def test_two_rank_checkpoint_write_failure_exits(tmp_path):
    (tmp_path / "checkpoints/step_00000001.pt").mkdir(parents=True)
    results = run_distributed(training_command(tmp_path))
    for result in results:
        assert result.returncode != 0
        assert "Checkpoint failed" in result.stderr


@pytest.mark.parametrize(
    "target,context",
    [
        ("checkpoints", "Creating project directories failed"),
        ("configs/run.yaml", "Saving configuration failed"),
        ("logs/train.log", "Opening training log failed"),
    ],
)
def test_project_output_failure_reaches_every_rank(tmp_path, target, context):
    path = tmp_path / target
    if target == "checkpoints":
        path.write_text("not a directory")
    else:
        path.mkdir(parents=True)
    results = run_distributed(training_command(tmp_path), timeout=15)
    for result in results:
        assert result.returncode != 0
        assert context in result.stderr, result.stdout + result.stderr


@pytest.mark.skipif(
    not os.path.exists("/dev/full"), reason="requires a failing write sink"
)
def test_initial_log_write_failure_reaches_every_rank(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/train.log").symlink_to("/dev/full")
    results = run_distributed(training_command(tmp_path), timeout=15)
    for result in results:
        assert result.returncode != 0
        assert "Opening training log failed" in result.stderr, (
            result.stdout + result.stderr
        )
        assert "No space left on device" in result.stderr


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
    for rank in range(2):
        metrics = json.loads((tmp_path / f"rank_{rank}.json").read_text())["metrics"]
        assert metrics["acc"] == pytest.approx(1 / eval_n)
        assert metrics["acc/num_valid"] == 3 * eval_n
        assert metrics["ppl/num_valid"] == 3 * eval_n


def test_two_rank_undersized_worker_stream_fails_promptly(tmp_path):
    write_probe_data(tmp_path, n=3, eval_n=1)
    results = run_distributed(
        training_command(
            tmp_path / "run", f"data.train={tmp_path / 'shards'}", "data.num_workers=2"
        )
    )
    for result in results:
        assert result.returncode != 0
        assert "complete batches" in result.stderr


def test_empty_label_rank_matches_global_reference(tmp_path):
    import torch

    write_probe_data(tmp_path, n=8)
    order = torch.randperm(8, generator=torch.Generator().manual_seed(1337)).tolist()
    frame = pd.read_parquet(tmp_path / "train.parquet")
    frame["structure_tokens"] = [
        [None] * 3 if i in order[::2] else [0, 1, 2] for i in range(8)
    ]
    frame.to_parquet(tmp_path / "train.parquet", index=False)
    command = [
        sys.executable,
        "-m",
        "tests.utils.distributed_probe",
        "--case",
        "empty-labels",
        "--output",
        str(tmp_path),
        "--accum",
        "2",
    ]
    result = subprocess.run(
        command, env=training_env(), capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    reference = torch.load(
        tmp_path / "run/model/final.pt", weights_only=False, map_location="cpu"
    )
    for result in run_distributed(command):
        assert result.returncode == 0, result.stderr
    distributed = torch.load(
        tmp_path / "run/model/final.pt", weights_only=False, map_location="cpu"
    )
    assert distributed["global_step"] == 2 and distributed["micro_step"] == 4
    assert json.loads((tmp_path / "rank_0.json").read_text())["supervised_tokens"] == 0
    assert json.loads((tmp_path / "rank_1.json").read_text())["supervised_tokens"] > 0
    for name in reference["model"]:
        torch.testing.assert_close(
            distributed["model"][name], reference["model"][name], atol=3e-6, rtol=3e-5
        )


def test_rank_local_bad_input_exits_all_ranks(tmp_path):
    source = tmp_path / "bad.parquet"
    pd.DataFrame(
        {
            "sequence_id": ["a", "b"],
            "sequence": ["LAG"] * 2,
            "structure_tokens": [[9999, 0, 1], [0, 1, 2]],
        }
    ).to_parquet(source, index=False)
    for result in run_distributed(
        training_command(tmp_path / "run", f"data.train={source}", "data.batch_size=1"),
        timeout=15,
    ):
        assert result.returncode != 0
        assert "Loading training window failed" in result.stderr


def test_two_rank_globally_empty_pass_does_not_checkpoint(tmp_path):
    source = tmp_path / "empty.parquet"
    pq.write_table(
        pa.table(
            {
                "sequence_id": ["a", "b"],
                "sequence": ["LAG"] * 2,
                "structure_tokens": pa.array(
                    [[None] * 3] * 2, type=pa.list_(pa.int64())
                ),
            }
        ),
        source,
    )
    for result in run_distributed(
        training_command(tmp_path / "run", f"data.train={source}", "data.batch_size=1")
    ):
        assert result.returncode != 0
        assert "no successful optimizer update" in result.stderr
    assert not list((tmp_path / "run/checkpoints").glob("step_*.pt"))


@pytest.mark.parametrize("case", ["eval-error", "eval-empty"])
def test_evaluation_failure_reaches_every_rank(tmp_path, case):
    write_probe_data(tmp_path)
    if case == "eval-empty":
        frame = pd.read_parquet(tmp_path / "eval.parquet")
        pq.write_table(
            pa.table(
                {
                    "sequence_id": frame.sequence_id.tolist(),
                    "sequence": frame.sequence.tolist(),
                    "structure_tokens": pa.array(
                        [[None] * 3] * len(frame), type=pa.list_(pa.int64())
                    ),
                }
            ),
            tmp_path / "eval.parquet",
        )
    command = [
        sys.executable,
        "-m",
        "tests.utils.distributed_probe",
        "--case",
        case,
        "--output",
        str(tmp_path),
    ]
    for result in run_distributed(command):
        assert result.returncode != 0
        expected = (
            "injected rank-local evaluation failure"
            if case == "eval-error"
            else "num_valid=0"
        )
        assert expected in result.stderr


def test_logreg_budget_failure_reaches_every_rank(tmp_path):
    write_probe_data(tmp_path, eval_n=5)
    command = [
        sys.executable,
        "-m",
        "tests.utils.distributed_probe",
        "--case",
        "eval-budget",
        "--output",
        str(tmp_path),
    ]
    for result in run_distributed(command):
        assert result.returncode != 0
        assert "logreg_max_feature_bytes" in result.stderr
        assert "Evaluation dataset default" in result.stderr


def test_logreg_variable_state_gather_preserves_all_proteins(tmp_path):
    write_probe_data(tmp_path, eval_n=5)
    command = [
        sys.executable,
        "-m",
        "tests.utils.distributed_probe",
        "--case",
        "eval-logreg",
        "--output",
        str(tmp_path),
    ]
    for result in run_distributed(command):
        assert result.returncode == 0, result.stderr
    for rank in range(2):
        metrics = json.loads((tmp_path / f"rank_{rank}.json").read_text())["metrics"]
        assert metrics["p_at_l"] == 1.0
        assert metrics["p_at_l/num_valid"] == 5
        assert metrics["p_at_l/fallback"] == 1.0


def test_fitted_logreg_single_and_two_rank_scores_match(tmp_path):
    write_probe_data(tmp_path, eval_n=5)
    command = [
        sys.executable,
        "-m",
        "tests.utils.distributed_probe",
        "--case",
        "logreg-fit",
        "--output",
        str(tmp_path),
    ]
    reference = subprocess.run(
        command, env=training_env(), capture_output=True, text=True, timeout=30
    )
    assert reference.returncode == 0, reference.stderr
    expected = json.loads((tmp_path / "reference.json").read_text())["metrics"]
    assert expected["p_at_l/num_valid"] == 8
    assert "p_at_l/fallback" not in expected
    for result in run_distributed(command):
        assert result.returncode == 0, result.stderr
    for rank in range(2):
        assert (
            json.loads((tmp_path / f"rank_{rank}.json").read_text())["metrics"]
            == expected
        )


@pytest.mark.parametrize(
    "coverage", ["uneven", "empty-modality", "empty-rank", "unused-head"]
)
def test_mdlm_two_rank_update_matches_global_reference(tmp_path, coverage):
    import torch
    from tests.integration.test_mdlm_training import training_fixture
    from tests.utils.synthetic import make_mdlm_rows

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
        rows.append(row)
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
    assert not list((tmp_path / "run/checkpoints").glob("step_*.pt"))
