"""Fresh-process continuation: data views, dropout, optimizer, and all-rank state."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from tests.integration.test_mdlm_training import mdlm_config, training_fixture
from tests.integration.test_training_progress import training_env
from tests.integration.test_distributed_training import run_distributed
from tests.utils.synthetic import make_mdlm_rows, write_dataset

PROBE = r"""
import os, sys, torch
from pathlib import Path
from omegaconf import OmegaConf
from stok.cli import train
from stok.training import tasks
from stok.models.stok import STokModel
cfg = OmegaConf.load(sys.argv[1])
stop = int(sys.argv[2])
rank = int(os.environ.get("RANK", 0))
trace = []
original_corrupt = tasks.corrupt_mdlm_batch
original_forward = STokModel.forward
original_save = train._save_checkpoint

def corrupt(batch, *args, **kwargs):
    result = original_corrupt(batch, *args, **kwargs)
    trace.append({'seeds': kwargs['seeds'], 'batch': batch, 'corruption': result})
    return result

def forward(self, tokens, *args, **kwargs):
    trace.append(tokens.detach().cpu().clone())
    return original_forward(self, tokens, *args, **kwargs)

def save(*args, **kwargs):
    original_save(*args, **kwargs)
    if kwargs['global_step'] == stop:
        raise InterruptedError('intentional interruption after completed checkpoint')

tasks.corrupt_mdlm_batch = corrupt
STokModel.forward = forward
train._save_checkpoint = save
if os.environ.get('RESUME_SKIP'):
    original_accelerator = train._maybe_get_accelerator
    def accelerator(precision=None):
        acc = original_accelerator(precision)
        acc.scaler = torch.amp.GradScaler('cpu')
        return acc
    train._maybe_get_accelerator = accelerator
    from stok.models.mdlm import STokMDLM
    init = STokMDLM.__init__
    def mdlm_init(self, **kwargs):
        init(self, **kwargs)
        calls = [0]
        def overflow(grad):
            calls[0] += 1
            return torch.full_like(grad, float('inf')) if calls[0] == 1 and not cfg.train.get('resume_from') else grad
        self.sequence_bias.register_hook(overflow)
    STokMDLM.__init__ = mdlm_init
    mdlm_forward = STokMDLM.forward
    def rank_forward(self, *args, **kwargs):
        torch.rand(rank + 1)  # ensure distinct per-rank RNG streams really restore
        return mdlm_forward(self, *args, **kwargs)
    STokMDLM.forward = rank_forward
if os.environ.get('CHECK_PROGRESS_CONTRACT'):
    from typing import is_typeddict
    from stok.utils.checkpoint import TrainingProgress
    original_restore = train.restore_training_state
    def restore(*args, **kwargs):
        result = original_restore(*args, **kwargs)
        assert is_typeddict(TrainingProgress), 'TrainingProgress must be TypedDict'
        assert type(result) is dict, 'TrainingProgress must be an ordinary dict'
        assert {'epoch', 'batches_in_epoch', 'global_step', 'micro_step', 'residues_seen', 'executed_positions', 'running_loss', 'running_updates', 'mdlm_running'} <= result.keys()
        assert result['epoch'] == 0 and result['global_step'] == 1
        return result
    train.restore_training_state = restore
if os.environ.get('RESUME_TINY_DECODER'):
    from stok.models.decoder import _DECODER_ARCH
    _DECODER_ARCH['lite'] = dict(d_model=16, n_heads=2, n_layers=1, ffn_mult=1, max_length=32, num_memory_tokens=0, attn_kv_heads=1)
    original_decoder = train.load_pretrained_decoder
    decoder_calls = []
    def load_decoder(**kwargs):
        decoder_calls.append(1)
        assert len(decoder_calls) == 1, 'Decoder constructed more than once'
        return original_decoder(**kwargs)
    train.load_pretrained_decoder = load_decoder
try:
    train.run_training(cfg)
finally:
    torch.save(trace, str(sys.argv[1]) + f'.rank{rank}.trace.pt')
"""


def execute(
    cfg, path, *, stop=-1, distributed=False, extra_env=None, ok=True, probe=None
):
    OmegaConf.save(cfg, path)
    command = [
        sys.executable,
        "-c",
        PROBE if probe is None else probe,
        str(path),
        str(stop),
    ]
    env = {**training_env(), **(extra_env or {})}
    results = (
        run_distributed(command, timeout=60, env=env)
        if distributed
        else [
            subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
        ]
    )
    for result in results:
        assert (result.returncode == 0) == ok, result.stdout + result.stderr
        if stop >= 0:
            assert "intentional interruption" in result.stderr, result.stderr
    return results


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


def config_for(tmp_path, objective="mdlm", source_kind="sharded", workers=0):
    source, codebook = training_fixture(tmp_path, n=20)
    cfg = mdlm_config(
        tmp_path / "full",
        source,
        codebook,
        **{
            "train.max_steps": 6,
            "train.gradient_accumulation_steps": 3,
            "model.encoder.dropout": 0.2,
            "data.num_workers": workers,
            "train.log_every": 3,
            "train.decay_steps": None,
            "train.warmup_steps": 1,
            "data.shuffle_rows": True,
            "data.shuffle_shards": True,
        },
    )
    if source_kind == "mixed":
        rows = [{**make_mdlm_rows()[i % 2], "sequence_id": f"b{i}"} for i in range(20)]
        second = write_dataset(tmp_path / "second", rows)
        cfg.data.train.other = {"path": str(second), "fraction": 0.4}
        split = tmp_path / "split.jsonl"
        split.write_text(
            "".join(
                json.dumps(
                    {
                        "dataset": name,
                        "sequence_id": f"{prefix}{i}",
                        "split": "train",
                        "cluster_id": f"{i % 2}",
                    }
                )
                + "\n"
                for name, prefix in [("local", ""), ("other", "b")]
                for i in range(20)
            )
        )
        cfg.data.split_manifest = str(split)
    if objective in {"mlm", "codebook"}:
        cfg.train.objective = objective
        cfg.train.mlm.mask_prob = 0.7
        if source_kind == "map":
            cfg.data.train.local.path = str(source / "part-000000.parquet")
        elif source_kind == "mixed":
            cfg.data.train.other.path = str(second / "part-000000.parquet")
    return cfg


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize(
    "objective,source_kind,stop",
    [
        ("mdlm", "sharded", 2),
        ("mdlm", "sharded", 4),
        ("mdlm", "mixed", 2),
        ("mlm", "map", 2),
        ("mlm", "sharded", 4),
        ("mlm", "mixed", 2),
        ("codebook", "map", 2),
    ],
)
def test_resume_matches_uninterrupted_training(
    tmp_path, workers, objective, source_kind, stop
):
    cfg = config_for(tmp_path, objective, source_kind, workers)
    execute(cfg, tmp_path / "full.yaml")
    cfg.train.output_dir = str(tmp_path / "interrupted")
    execute(cfg, tmp_path / "interrupted.yaml", stop=stop, ok=False)
    cfg.train.resume_from = str(
        tmp_path / f"interrupted/checkpoints/step_{stop:08d}.pt"
    )
    execute(cfg, tmp_path / "resumed.yaml")
    full = torch.load(tmp_path / "full/model/final.pt", weights_only=True)
    resumed = torch.load(tmp_path / "interrupted/model/final.pt", weights_only=True)
    assert full["format_version"] == resumed["format_version"] == 2
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "global_step",
        "micro_step",
        "residues_seen",
        "executed_positions",
        "rank_states",
    ):
        equal(full[key], resumed[key])

    def trace(name):
        return torch.load(tmp_path / f"{name}.yaml.rank0.trace.pt", weights_only=True)

    equal(trace("full"), trace("interrupted") + trace("resumed"))


@pytest.mark.parametrize("workers", [0, 2])
def test_two_rank_continuation_with_actual_cpu_scaler_skip(tmp_path, workers):
    cfg = config_for(tmp_path, workers=workers)
    cfg.train.max_steps = 4
    env = {"RESUME_SKIP": "1"}
    execute(cfg, tmp_path / "full.yaml", distributed=True, extra_env=env)
    cfg.train.output_dir = str(tmp_path / "interrupted")
    execute(
        cfg,
        tmp_path / "interrupted.yaml",
        stop=1,
        distributed=True,
        extra_env=env,
        ok=False,
    )
    cfg.train.resume_from = str(tmp_path / "interrupted/checkpoints/step_00000001.pt")
    saved = torch.load(cfg.train.resume_from, weights_only=True)
    assert len(saved["rank_states"]) == 2
    assert (
        saved["micro_step"] > 3
    )  # overflow consumed a whole window without a successful update
    assert all(r["scaler"]["scale"] == 32768 for r in saved["rank_states"])
    assert (
        saved["rank_states"][0]["batches_in_epoch"]
        == saved["rank_states"][1]["batches_in_epoch"]
    )
    assert not torch.equal(
        saved["rank_states"][0]["rng"]["torch"], saved["rank_states"][1]["rng"]["torch"]
    )
    execute(cfg, tmp_path / "resumed.yaml", distributed=True, extra_env=env)
    full = torch.load(tmp_path / "full/model/final.pt", weights_only=True)
    resumed = torch.load(tmp_path / "interrupted/model/final.pt", weights_only=True)
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "rank_states",
        "global_step",
        "micro_step",
        "residues_seen",
        "executed_positions",
    ):
        equal(full[key], resumed[key])
    for rank in (0, 1):

        def trace(name):
            return torch.load(
                tmp_path / f"{name}.yaml.rank{rank}.trace.pt", weights_only=True
            )

        equal(trace("full"), trace("interrupted") + trace("resumed"))


def snapshot(path):
    return {
        str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("train.objective", "mlm"),
        ("train.seed", 9),
        ("train.gradient_accumulation_steps", 2),
        ("train.lr", 0.001),
        ("train.max_steps", 10),
        ("train.warmup_steps", 2),
        ("train.mixed_precision", "bf16"),
        ("train.batch_size", 1),
        ("data.num_workers", 2),
        ("model.encoder.dropout", 0.1),
        ("train.mdlm.placement", "span"),
    ],
)
def test_rejected_resume_preserves_every_artifact(tmp_path, key, value):
    cfg = config_for(tmp_path)
    execute(cfg, tmp_path / "original.yaml", stop=1, ok=False)
    project = Path(cfg.train.output_dir)
    cfg.train.resume_from = str(project / "checkpoints/step_00000001.pt")
    before = snapshot(project)
    OmegaConf.update(cfg, key, value)
    results = execute(cfg, tmp_path / "rejected.yaml", ok=False)
    assert "signature mismatch" in results[0].stderr
    assert snapshot(project) == before


@pytest.mark.parametrize(
    "damage", ["legacy", "version", "rank", "world_size", "data", "codebook"]
)
def test_invalid_checkpoint_or_identity_preserves_artifacts(tmp_path, damage):
    cfg = config_for(tmp_path)
    execute(cfg, tmp_path / "original.yaml", stop=1, ok=False)
    project = Path(cfg.train.output_dir)
    checkpoint = project / "checkpoints/step_00000001.pt"
    payload = torch.load(checkpoint, weights_only=True)
    if damage == "data":
        # An appended byte changes content identity even if Arrow still reads the rows.
        path = Path(cfg.data.train.local.path) / "part-000000.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
    elif damage == "codebook":
        torch.save({"codebook": torch.ones(32, 2)}, cfg.model.codebook.path)
    else:
        if damage == "legacy":
            payload = {"model": payload["model"]}
        elif damage == "version":
            payload["format_version"] = 99
        elif damage == "rank":
            payload["rank_states"] = []
        else:
            payload["signature"]["execution"]["world_size"] = 2
        torch.save(payload, checkpoint)
    cfg.train.resume_from = str(checkpoint)
    before = snapshot(project)
    execute(cfg, tmp_path / "rejected.yaml", ok=False)
    assert snapshot(project) == before


@pytest.mark.parametrize("failure", ["read", "write", "state", "cuda_rng"])
def test_rank_local_checkpoint_failures_reach_all_ranks(tmp_path, failure):
    cfg = config_for(tmp_path)
    execute(cfg, tmp_path / "original.yaml", stop=1, distributed=True, ok=False)
    project = Path(cfg.train.output_dir)
    checkpoint = project / "checkpoints/step_00000001.pt"
    cfg.train.resume_from = str(checkpoint)
    before = snapshot(project)
    path = tmp_path / "failure.yaml"
    OmegaConf.save(cfg, path)
    if failure == "read":
        patch = "original_read = train.read_training_checkpoint\ndef read(path):\n    if rank == 1: raise OSError('injected rank-local read failure')\n    return original_read(path)\ntrain.read_training_checkpoint = read\n"
    elif failure == "cuda_rng":
        patch = "original_read = train.read_training_checkpoint\ndef read(path):\n    payload = original_read(path)\n    if rank == 1: payload['rank_states'][1]['rng']['cuda'] = []\n    return payload\ntrain.read_training_checkpoint = read\n"
    elif failure == "write":
        patch = "original_torch_save = torch.save\ndef fail_save(value, path, *args, **kwargs):\n    if rank == 0 and isinstance(value, dict) and 'format_version' in value:\n        with open(path, 'wb') as out: out.write(b'partial')\n        raise OSError('injected rank-zero write failure')\n    return original_torch_save(value, path, *args, **kwargs)\ntorch.save = fail_save\n"
    else:
        patch = "original_rng = train._collect_rng_state\ndef collect_rng(**kwargs):\n    if rank == 1: raise OSError('injected rank-local state failure')\n    return original_rng(**kwargs)\ntrain._collect_rng_state = collect_rng\n"
    probe = PROBE.replace(
        "try:\n    train.run_training(cfg)", patch + "try:\n    train.run_training(cfg)"
    )
    results = run_distributed(
        [sys.executable, "-c", probe, str(path), "-1"], timeout=60
    )
    for result in results:
        assert result.returncode != 0
        assert (
            "CUDA RNG" if failure == "cuda_rng" else "injected rank-"
        ) in result.stderr, result.stderr
    if failure in {"read", "cuda_rng"}:
        assert snapshot(project) == before
    else:
        assert checkpoint.read_bytes() == before["checkpoints/step_00000001.pt"]
        assert not list((project / "checkpoints").glob(".*.pt.*"))


@pytest.mark.parametrize("objective", ["mlm", "codebook"])
def test_legacy_invalid_resume_does_not_touch_artifacts(tmp_path, objective):
    cfg = config_for(tmp_path, "mlm", "map")
    cfg.train.objective = objective
    execute(cfg, tmp_path / "original.yaml", stop=1, ok=False)
    project = Path(cfg.train.output_dir)
    cfg.train.resume_from = str(project / "checkpoints/step_00000001.pt")
    cfg.train.lr *= 2
    before = snapshot(project)
    execute(cfg, tmp_path / "rejected.yaml", ok=False)
    assert snapshot(project) == before


@pytest.mark.parametrize(
    "damage",
    [
        "logging",
        "logging_value",
        "optimizer",
        "optimizer_partial",
        "optimizer_empty_entry",
        "rng",
        "scaler",
        "cursor",
    ],
)
def test_incomplete_rank_state_rejected_before_output(tmp_path, damage):
    cfg = config_for(tmp_path)
    execute(cfg, tmp_path / "original.yaml", stop=1, ok=False)
    project = Path(cfg.train.output_dir)
    checkpoint = project / "checkpoints/step_00000001.pt"
    payload = torch.load(checkpoint, weights_only=True)
    rank = payload["rank_states"][0]
    if damage == "logging":
        del rank["logging"]["running_loss"]
    elif damage == "logging_value":
        rank["logging"]["running_loss"] = "broken"
    elif damage == "optimizer":
        payload["optimizer"]["state"].clear()
    elif damage == "optimizer_partial":
        payload["optimizer"]["state"].pop(next(iter(payload["optimizer"]["state"])))
    elif damage == "optimizer_empty_entry":
        payload["optimizer"]["state"][next(iter(payload["optimizer"]["state"]))] = {}
    elif damage == "rng":
        del rank["rng"]["torch"]
    elif damage == "scaler":
        rank["scaler"] = {"scale": 1}
    else:
        rank["batches_in_epoch"] = 1  # pending partial accumulation forbidden
    torch.save(payload, checkpoint)
    cfg.train.resume_from = str(checkpoint)
    before = snapshot(project)
    execute(cfg, tmp_path / "rejected.yaml", ok=False)
    assert snapshot(project) == before


WANDB_MOCK = r"""
from types import SimpleNamespace
import json
record = {'init': [], 'history': [], 'summary': {}}
wb = SimpleNamespace()
def init(**kwargs):
    import random, numpy as np
    random.random(); np.random.rand(); torch.rand(7)
    record['init'].append(kwargs)
    if os.environ.get('WB_INIT_FAIL'): raise RuntimeError('visible W&B initialization failure')
    wb.run = SimpleNamespace(id=kwargs.get('id') or 'fixed-run-id', step=int(os.environ.get('WB_WATERMARK', '-1'))+1, summary=record['summary'])
    if os.environ.get('WB_HISTORY_FAIL'):
        class Step:
            def __int__(self): raise RuntimeError('visible W&B history inspection failure')
        wb.run.step = Step()
    return wb.run
def log(payload, *, step):
    import random, numpy as np
    random.random(); np.random.rand(); torch.rand(5)
    if os.environ.get('WB_LOG_FAIL'): raise RuntimeError('visible W&B history failure')
    record['history'].append(step)
wb.init, wb.log = init, log
sys.modules['wandb'] = wb
import atexit
atexit.register(lambda: Path(str(sys.argv[1])+'.wandb.json').write_text(json.dumps(record)))
"""


def execute_wandb(cfg, path, *, stop=-1, env=None, ok=True):
    OmegaConf.save(cfg, path)
    probe = PROBE.replace(
        "try:\n    train.run_training(cfg)",
        WANDB_MOCK + "try:\n    train.run_training(cfg)",
    )
    result = subprocess.run(
        [sys.executable, "-c", probe, str(path), str(stop)],
        env={**training_env(), **(env or {})},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
    return result, json.loads(Path(str(path) + ".wandb.json").read_text())


def test_wandb_id_and_rollback_suppress_only_history(tmp_path):
    cfg = config_for(tmp_path)
    cfg.train.log_every = 1
    cfg.train.wandb.enabled = True
    cfg.train.wandb.mode = "offline"
    execute_wandb(cfg, tmp_path / "full.yaml")
    cfg.train.output_dir = str(tmp_path / "interrupted")
    execute_wandb(cfg, tmp_path / "interrupted.yaml", stop=2, ok=False)
    cfg.train.resume_from = str(tmp_path / "interrupted/checkpoints/step_00000002.pt")
    payload = torch.load(cfg.train.resume_from, weights_only=True)
    assert payload["wandb_run_id"] == "fixed-run-id"
    _, record = execute_wandb(cfg, tmp_path / "resumed.yaml", env={"WB_WATERMARK": "4"})
    assert len(record["init"]) == 1
    assert record["init"][0]["id"] == "fixed-run-id"
    assert record["init"][0]["resume"] == "must"
    assert record["init"][0]["mode"] == "offline"
    assert record["history"] == [5, 6]
    assert record["summary"]["resume/rollback"] == {
        "checkpoint_step": 2,
        "history_watermark": 4,
    }
    full = torch.load(tmp_path / "full/model/final.pt", weights_only=True)
    resumed = torch.load(tmp_path / "interrupted/model/final.pt", weights_only=True)
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "global_step",
        "micro_step",
        "rank_states",
    ):
        equal(full[key], resumed[key])

    def trace(name):
        return torch.load(tmp_path / f"{name}.yaml.rank0.trace.pt", weights_only=True)

    equal(trace("full"), trace("interrupted") + trace("resumed"))
    assert (
        "Resume/rollback: checkpoint step 2; remote history watermark 4"
        in (tmp_path / "interrupted/logs/train.log").read_text()
    )


@pytest.mark.parametrize("failure", ["WB_INIT_FAIL", "WB_HISTORY_FAIL", "WB_LOG_FAIL"])
def test_wandb_failure_visible_without_new_run_fallback(tmp_path, failure):
    cfg = config_for(tmp_path)
    cfg.train.wandb.enabled = True
    cfg.train.log_every = 1
    execute_wandb(cfg, tmp_path / "original.yaml", stop=1, ok=False)
    cfg.train.resume_from = str(
        Path(cfg.train.output_dir) / "checkpoints/step_00000001.pt"
    )
    before = snapshot(Path(cfg.train.output_dir))
    result, record = execute_wandb(
        cfg, tmp_path / "failed.yaml", env={failure: "1"}, ok=False
    )
    if failure != "WB_LOG_FAIL":
        assert snapshot(Path(cfg.train.output_dir)) == before
    assert "visible W&B" in result.stderr
    assert len(record["init"]) == 1
    assert record["init"][0]["id"] == "fixed-run-id"
    assert not (Path(cfg.train.output_dir) / "model/final.pt").exists()


def test_actual_wandb_offline_id_continues_without_network(tmp_path):
    cfg = config_for(tmp_path)
    cfg.train.wandb.enabled = True
    cfg.train.max_steps = 2
    cfg.train.log_every = 1
    env = {"WANDB_MODE": "offline", "WANDB_SILENT": "true"}
    execute(cfg, tmp_path / "original.yaml", stop=1, extra_env=env, ok=False)
    project = Path(cfg.train.output_dir)
    cfg.train.resume_from = str(project / "checkpoints/step_00000001.pt")
    saved = torch.load(cfg.train.resume_from, weights_only=True)
    assert saved["wandb_run_id"]
    execute(cfg, tmp_path / "resumed.yaml", extra_env=env)
    final = torch.load(project / "model/final.pt", weights_only=True)
    assert final["wandb_run_id"] == saved["wandb_run_id"]
    assert final["global_step"] == 2
    assert list((project / "logs/wandb").glob("offline-run-*"))


def test_no_eligible_windows_are_in_resume_cursor(tmp_path):
    rows = [
        {
            **make_mdlm_rows()[1],
            "sequence_id": str(i),
            "structure_tokens": [None] * 3 if i < 4 else [0, 1, 2],
        }
        for i in range(12)
    ]
    source, codebook = training_fixture(tmp_path, rows=rows)
    cfg = mdlm_config(
        tmp_path / "full",
        source,
        codebook,
        **{
            "train.max_steps": 3,
            "train.gradient_accumulation_steps": 2,
            "model.encoder.dropout": 0.2,
            "train.mdlm.regime_weights": {"structure_only": 1},
        },
    )
    execute(cfg, tmp_path / "full.yaml")
    cfg.train.output_dir = str(tmp_path / "interrupted")
    execute(cfg, tmp_path / "interrupted.yaml", stop=1, ok=False)
    cfg.train.resume_from = str(tmp_path / "interrupted/checkpoints/step_00000001.pt")
    saved = torch.load(cfg.train.resume_from, weights_only=True)
    assert saved["micro_step"] == saved["rank_states"][0]["batches_in_epoch"] == 4
    assert saved["executed_positions"] == 32
    execute(cfg, tmp_path / "resumed.yaml")
    full = torch.load(tmp_path / "full/model/final.pt", weights_only=True)
    final = torch.load(tmp_path / "interrupted/model/final.pt", weights_only=True)
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "rank_states",
        "micro_step",
        "residues_seen",
        "executed_positions",
    ):
        equal(full[key], final[key])


def test_explicit_epochs_recurse_and_preserve_implicit_legacy_iteration(tmp_path):
    from stok.data.dataset import (
        IterableTokenizedDataset,
        MapAsIterableDataset,
        TokenizedDataset,
        InterleavedIterableDataset,
        set_dataset_epoch,
    )

    source, _ = training_fixture(tmp_path, n=20)

    def build():
        shard = IterableTokenizedDataset(str(source), max_length=8)
        mapped = MapAsIterableDataset(
            TokenizedDataset(str(source / "part-000000.parquet"), max_length=8)
        )
        child = InterleavedIterableDataset([shard, mapped], [0.5, 0.5])
        return InterleavedIterableDataset([child], [1])

    implicit, explicit = build(), build()

    def ids(ds):
        return [row["sequence_id"] for row in ds]

    first, second = ids(implicit), ids(implicit)
    assert first != second
    set_dataset_epoch(explicit, 0)
    assert ids(explicit) == ids(explicit) == first
    set_dataset_epoch(explicit, 1)
    assert ids(explicit) == second
    assert explicit.datasets[0].datasets[1]._epoch == 1


def test_resume_allows_output_logging_evaluation_and_checkpoint_overrides(tmp_path):
    cfg = config_for(tmp_path)
    execute(cfg, tmp_path / "full.yaml")
    cfg.train.output_dir = str(tmp_path / "interrupted")
    execute(cfg, tmp_path / "interrupted.yaml", stop=2, ok=False)
    cfg.train.resume_from = str(tmp_path / "interrupted/checkpoints/step_00000002.pt")
    cfg.train.output_dir = str(tmp_path / "new-output")
    cfg.train.log_every = 1
    cfg.train.save_every = 3
    cfg.train.eval.steps = 999
    cfg.train.eval.seed = 88
    cfg.train.wandb.tags = ["resumed"]
    execute(cfg, tmp_path / "resumed.yaml")
    full = torch.load(tmp_path / "full/model/final.pt", weights_only=True)
    final = torch.load(tmp_path / "new-output/model/final.pt", weights_only=True)
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "micro_step",
        "residues_seen",
        "executed_positions",
    ):
        equal(full[key], final[key])
    equal(full["rank_states"][0]["rng"], final["rank_states"][0]["rng"])


def test_training_progress_is_flat_typed_dict(tmp_path):
    cfg = config_for(tmp_path)
    execute(cfg, tmp_path / "original.yaml", stop=1, ok=False)
    cfg.train.resume_from = str(
        Path(cfg.train.output_dir) / "checkpoints/step_00000001.pt"
    )
    execute(cfg, tmp_path / "resumed.yaml", extra_env={"CHECK_PROGRESS_CONTRACT": "1"})


def fape_config(tmp_path):
    from stok.models.decoder import GeometricDecoder

    cfg = config_for(tmp_path, "codebook", "map")
    decoder = GeometricDecoder(
        d_code=2,
        d_model=16,
        n_heads=2,
        n_layers=1,
        ffn_mult=1,
        max_length=32,
        num_memory_tokens=0,
        attn_kv_heads=1,
    )
    cfg.model.decoder.path = str(tmp_path / "decoder.pt")
    cfg.model.decoder.preset = "lite"
    torch.save(decoder.state_dict(), cfg.model.decoder.path)
    cfg.data.load_coords = True
    cfg.train.fape.enabled = True
    cfg.train.fape.start_step = 0
    cfg.train.fape.weight = 1
    cfg.train.max_steps = 3
    cfg.train.log_every = 1
    return cfg


def test_changed_training_decoder_rejected_before_artifacts(tmp_path):
    cfg = fape_config(tmp_path)
    env = {"RESUME_TINY_DECODER": "1"}
    execute(cfg, tmp_path / "original.yaml", stop=1, ok=False, extra_env=env)
    project = Path(cfg.train.output_dir)
    cfg.train.resume_from = str(project / "checkpoints/step_00000001.pt")
    before = snapshot(project)
    decoder = torch.load(cfg.model.decoder.path, weights_only=True)
    decoder["projector_in.weight"].add_(0.25)
    torch.save(decoder, cfg.model.decoder.path)
    results = execute(cfg, tmp_path / "rejected.yaml", extra_env=env, ok=False)
    assert "signature mismatch" in results[0].stderr
    assert snapshot(project) == before


def test_fape_resume_with_same_decoder_matches_uninterrupted(tmp_path):
    cfg = fape_config(tmp_path)
    env = {"RESUME_TINY_DECODER": "1"}
    execute(cfg, tmp_path / "full.yaml", extra_env=env)
    cfg.train.output_dir = str(tmp_path / "interrupted")
    execute(cfg, tmp_path / "interrupted.yaml", stop=1, ok=False, extra_env=env)
    cfg.train.resume_from = str(tmp_path / "interrupted/checkpoints/step_00000001.pt")
    execute(cfg, tmp_path / "resumed.yaml", extra_env=env)
    full = torch.load(tmp_path / "full/model/final.pt", weights_only=True)
    resumed = torch.load(tmp_path / "interrupted/model/final.pt", weights_only=True)
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "global_step",
        "micro_step",
        "rank_states",
    ):
        equal(full[key], resumed[key])
    assert " | fape " in (tmp_path / "full/logs/train.log").read_text()


def test_optimizer_coverage_preserves_unused_and_frozen_parameters(tmp_path):
    from stok.cli.train import _save_checkpoint
    from stok.utils.checkpoint import read_training_checkpoint, restore_training_state

    model = torch.nn.Module()
    model.unused = torch.nn.Parameter(torch.ones(1))
    model.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    model.used = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    model.used(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    logging = dict.fromkeys(
        [
            "running_loss",
            "running_updates",
            "running_cls_loss",
            "running_cls_count",
            "running_fape_loss",
            "running_fape_count",
            "running_pred_nan_frac_sum",
            "running_pred_nan_frac_count",
            "running_masked_acc_sum",
            "running_masked_acc_count",
            "total_missing_structure",
            "total_noncanonical_sequence",
        ],
        0,
    )
    logging["mdlm_running"] = torch.zeros(5, 2, dtype=torch.float64)
    path = tmp_path / "checkpoint.pt"
    _save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=1,
        micro_step=1,
        cfg=OmegaConf.create({"train": {"objective": "codebook"}}),
        accelerator=None,
        training_state={
            "signature": {},
            "wandb_run_id": None,
            "local": {
                "epoch": 0,
                "batches_in_epoch": 1,
                "logging": logging,
                "loader_generator_state": torch.Generator().get_state(),
            },
        },
    )
    payload = read_training_checkpoint(path)
    assert len(payload["optimizer_initialized"]) == 2
    assert len(payload["optimizer"]["param_groups"][0]["params"]) == 4
    restored_optimizer = torch.optim.AdamW(model.parameters())
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda _: 1
    )
    progress = restore_training_state(
        payload,
        model=model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        accelerator=None,
    )
    assert type(progress) is dict and progress["batches_in_epoch"] == 1
    assert model.unused not in restored_optimizer.state
    assert model.frozen not in restored_optimizer.state
    assert len(restored_optimizer.state) == 2
    equal(optimizer.state_dict(), restored_optimizer.state_dict())


def test_eval_only_decoder_does_not_constrain_resume_signature(tmp_path):
    from stok.utils.checkpoint import resume_signature

    cfg = config_for(tmp_path, "codebook", "map")
    cfg.train.fape.enabled = False
    decoder = torch.nn.Linear(2, 3)
    first = resume_signature(
        cfg, sources=[], codebook=None, accelerator=None, training_decoder=decoder
    )
    with torch.no_grad():
        decoder.weight.add_(1)
    second = resume_signature(
        cfg, sources=[], codebook=None, accelerator=None, training_decoder=decoder
    )
    assert first == second


@pytest.fixture(scope="module")
def rng_checkpoint(tmp_path_factory):
    from stok.cli.train import run_training

    root = tmp_path_factory.mktemp("rng-checkpoint")
    cfg = config_for(root)
    cfg.train.max_steps = 1
    run_training(cfg)
    return cfg, torch.load(root / "full/model/final.pt", weights_only=True)


@pytest.mark.parametrize(
    "damage", ["missing", "empty", "truncated", "dtype", "shape", "length", "active"]
)
def test_cuda_rng_rejected_before_artifacts_or_wandb(
    tmp_path, monkeypatch, rng_checkpoint, damage
):
    import copy
    from stok.cli import train

    original_cfg, original = rng_checkpoint
    cfg, payload = copy.deepcopy(original_cfg), copy.deepcopy(original)
    payload["signature"]["execution"].update(
        device="cuda", cuda_rng_state_sizes=[16, 16]
    )
    rng = payload["rank_states"][0]["rng"]
    rng.update(
        cuda=[torch.zeros(16, dtype=torch.uint8) for _ in range(2)], cuda_device=1
    )
    if damage == "missing":
        del rng["cuda"]
    elif damage == "empty":
        rng["cuda"] = []
    elif damage == "truncated":
        rng["cuda"].pop()
    elif damage == "dtype":
        rng["cuda"][1] = rng["cuda"][1].float()
    elif damage == "shape":
        rng["cuda"][1] = rng["cuda"][1].view(4, 4)
    elif damage == "length":
        rng["cuda"][1] = rng["cuda"][1][:-1]
    else:
        rng["cuda_device"] = 2
    project = tmp_path / "run"
    project.mkdir()
    (project / "keep").write_bytes(b"existing run artifacts")
    checkpoint = project / "resume.pt"
    torch.save(payload, checkpoint)
    cfg.train.output_dir, cfg.train.resume_from = str(project), str(checkpoint)
    before = snapshot(project)
    monkeypatch.setattr(
        train, "resume_signature", lambda *a, **kw: payload["signature"]
    )
    monkeypatch.setattr(
        train,
        "_maybe_init_wandb",
        lambda *a, **kw: pytest.fail("W&B before RNG validation"),
    )
    with pytest.raises((ValueError, RuntimeError), match="CUDA RNG"):
        train.run_training(cfg)
    assert snapshot(project) == before
