import os
import signal
import socket
import subprocess
import time

from tests.integration.test_training_progress import training_command, training_env


def run_distributed(command, *, timeout=30, env=None):
    """Launch real rank processes without torchrun's hostname/DNS dependency."""
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    workers = []
    deadline = time.monotonic() + timeout
    try:
        for rank in range(2):
            rank_env = {**(env or training_env()), 'MASTER_ADDR': '127.0.0.1',
                        'MASTER_PORT': str(port), 'WORLD_SIZE': '2',
                        'RANK': str(rank), 'LOCAL_RANK': str(rank),
                        'LOCAL_WORLD_SIZE': '2'}
            workers.append(subprocess.Popen(command, env=rank_env, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, text=True,
                                            start_new_session=True))
        results = []
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=max(.1, deadline-time.monotonic()))
            results.append(subprocess.CompletedProcess(command, worker.returncode, stdout, stderr))
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
    assert (tmp_path / 'checkpoints/step_00000002.pt').is_file()


def test_two_rank_checkpoint_write_failure_exits(tmp_path):
    (tmp_path / 'checkpoints/step_00000001.pt').mkdir(parents=True)
    results = run_distributed(training_command(tmp_path))
    for result in results:
        assert result.returncode != 0
        assert 'Checkpoint failed' in result.stderr


import json
import sys
import pandas as pd
import pytest


def write_probe_data(root, n=17, eval_n=5):
    rows = [{'pid': str(i), 'protein_sequence': 'LAG', 'indices': [i]*3} for i in range(n)]
    frame = pd.DataFrame(rows)
    frame.assign(indices=frame.indices.map(lambda x: ' '.join(map(str, x)))).to_csv(root/'train.csv', index=False)
    (root/'shards').mkdir()
    frame.iloc[:7].to_parquet(root/'shards/a.parquet', index=False)
    frame.iloc[7:].to_parquet(root/'shards/b.parquet', index=False)
    frame.iloc[:eval_n].assign(indices=lambda f: f.indices.map(lambda x: ' '.join(map(str, x)))).to_csv(root/'eval.csv', index=False)
    (root/'eval_shards').mkdir()
    frame.iloc[:eval_n].to_parquet(root/'eval_shards/a.parquet', index=False)


@pytest.mark.parametrize('workers', [0, 2])
@pytest.mark.parametrize('source', ['map', 'iterable', 'map-mixture', 'mixed-mixture'])
def test_native_training_stream_matches_single_rank(tmp_path, workers, source):
    write_probe_data(tmp_path)
    command = [sys.executable, '-m', 'tests.utils.distributed_probe', '--case', 'coverage',
               '--output', str(tmp_path), '--source', source, '--workers', str(workers)]
    # Reference draws are ordered with no workers, for exact truncation before partitioning.
    result = subprocess.run([*command[:-1], '0'], env=training_env(), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    results = run_distributed(command)
    for result in results:
        assert result.returncode == 0, result.stderr
    records = [json.loads((tmp_path/f'rank_{r}.json').read_text()) for r in range(2)]
    assert records[0]['micro_steps'] == records[1]['micro_steps']
    reference = json.loads((tmp_path/'reference.json').read_text())['ids']
    total = len(reference) // (4 * max(1, workers)) * (4 * max(1, workers))
    assert total > 0
    assert sorted(records[0]['ids'] + records[1]['ids']) == sorted(reference[:total])


@pytest.mark.parametrize('eval_n', [1, 5])
@pytest.mark.parametrize('source', ['map', 'iterable'])
def test_native_eval_has_no_duplicates_or_dropped_tail(tmp_path, eval_n, source):
    write_probe_data(tmp_path, eval_n=eval_n)
    command = [sys.executable, '-m', 'tests.utils.distributed_probe', '--case', 'eval-tail',
               '--output', str(tmp_path), '--source', source, '--workers', '2']
    for result in run_distributed(command):
        assert result.returncode == 0, result.stderr
    ids = sum([json.loads((tmp_path/f'rank_{r}.json').read_text())['ids'] for r in range(2)], [])
    assert sorted(ids) == list(range(eval_n))
    for rank in range(2):
        metrics = json.loads((tmp_path/f'rank_{rank}.json').read_text())['metrics']
        assert metrics['acc'] == pytest.approx(1/eval_n)


def test_two_rank_undersized_worker_stream_fails_promptly(tmp_path):
    write_probe_data(tmp_path, n=3, eval_n=1)
    results = run_distributed(training_command(tmp_path/'run',
        f'data.train={tmp_path/"shards"}', 'data.num_workers=2'))
    for result in results:
        assert result.returncode != 0
        assert 'complete batches' in result.stderr
