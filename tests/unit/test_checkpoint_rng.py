"""Saved CUDA inventory is checked offline; current generators only on resume."""

import copy
from types import SimpleNamespace

import pytest
import torch

from stok.utils import checkpoint as ck
from tests.integration.test_mdlm_training import mdlm_config, training_fixture


@pytest.fixture
def cuda_resume():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    rng = ck.collect_rng_state()
    rng.update(
        cuda=[torch.arange(16, dtype=torch.uint8) for _ in range(2)], cuda_device=1
    )
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
    logging["mdlm_running"] = torch.zeros(5, 2)
    rank = dict(
        rank=0,
        rng=rng,
        scaler=None,
        epoch=0,
        batches_in_epoch=0,
        loader_generator_state=torch.Generator().get_state(),
        logging=logging,
    )
    signature = {
        "execution": {
            "device": "cuda",
            "world_size": 2,
            "cuda_rng_state_sizes": [16, 16],
        }
    }
    payload = dict(
        signature=signature,
        rank_states=[copy.deepcopy(rank), {**rank, "rank": 1}],
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        optimizer_initialized=[],
        scheduler=scheduler.state_dict(),
        config={"train": {"objective": "codebook"}},
        global_step=0,
        micro_step=0,
        residues_seen=0,
        executed_positions=0,
    )
    accelerator = SimpleNamespace(
        device=torch.device("cuda:1"),
        process_index=1,
        scaler=None,
        unwrap_model=lambda model: model,
    )
    return payload, dict(
        model=model, optimizer=optimizer, scheduler=scheduler, accelerator=accelerator
    )


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "empty",
        "truncated",
        "extra",
        "dtype",
        "shape",
        "length",
        "nontensor",
        "device_missing",
        "device_negative",
        "device_outside",
        "inventory_missing",
        "inventory_empty",
        "inventory_invalid",
    ],
)
def test_incomplete_cuda_inventory_rejected_offline(cuda_resume, monkeypatch, damage):
    payload, _ = cuda_resume
    rng = payload["rank_states"][1]["rng"]
    execution = payload["signature"]["execution"]
    if damage == "missing":
        del rng["cuda"]
    elif damage == "empty":
        rng["cuda"] = []
    elif damage == "truncated":
        rng["cuda"].pop()
    elif damage == "extra":
        rng["cuda"].append(rng["cuda"][0])
    elif damage == "dtype":
        rng["cuda"][1] = rng["cuda"][1].float()
    elif damage == "shape":
        rng["cuda"][1] = rng["cuda"][1].view(4, 4)
    elif damage == "length":
        rng["cuda"][1] = rng["cuda"][1][:-1]
    elif damage == "nontensor":
        rng["cuda"][1] = list(range(16))
    elif damage == "device_missing":
        del rng["cuda_device"]
    elif damage.startswith("device_"):
        rng["cuda_device"] = -1 if damage == "device_negative" else 2
    elif damage == "inventory_missing":
        del execution["cuda_rng_state_sizes"]
    else:
        execution["cuda_rng_state_sizes"] = (
            [] if damage == "inventory_empty" else [16, 0]
        )

    def forbidden(*args, **kwargs):
        pytest.fail("Offline validation initialized CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    with pytest.raises(ValueError, match="CUDA RNG"):
        ck.validate_resume_signature(payload, payload["signature"])


@pytest.mark.parametrize("active,count", [(0, 2), (1, 1)])
def test_resume_rejects_changed_current_cuda_inventory(
    cuda_resume, monkeypatch, active, count
):
    payload, kwargs = cuda_resume
    ck.validate_resume_signature(payload, payload["signature"])
    monkeypatch.setattr(torch.cuda, "current_device", lambda: active)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: payload["rank_states"][1]["rng"]["cuda"][:count],
    )
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: None)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="CUDA RNG"):
        ck.restore_training_state(payload, **kwargs)


def test_complete_cuda_inventory_restores_active_device_one(cuda_resume, monkeypatch):
    payload, kwargs = cuda_resume
    ck.validate_resume_signature(payload, payload["signature"])
    states = payload["rank_states"][1]["rng"]["cuda"]
    restored = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: states)
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all", lambda value: restored.append(value)
    )
    progress = ck.restore_training_state(payload, **kwargs)
    restored.clear()
    ck.restore_rng_state(progress["rng"])
    assert len(restored) == 1 and len(restored[0]) == 2
    torch.testing.assert_close(restored[0][1], states[1])


@pytest.mark.parametrize("initialized", [False, True])
def test_cpu_record_does_not_initialize_unused_cuda(tmp_path, monkeypatch, initialized):
    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(tmp_path / "run", source, codebook)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU state initialized unused CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    rng = ck.collect_rng_state(device=torch.device("cpu"))
    if not initialized:
        assert "cuda" not in ck.collect_rng_state()
    signature = ck.resume_signature(cfg, sources=[], codebook=None, accelerator=None)
    assert "cuda" not in rng
    assert signature["execution"]["cuda_rng_state_sizes"] == []
    ck.restore_rng_state(rng)
