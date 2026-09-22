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
