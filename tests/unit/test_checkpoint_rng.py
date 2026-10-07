"""Saved CUDA inventory is checked offline; current generators only on resume."""

import copy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from stok.utils import checkpoint as ck
from tests.integration.test_mdlm_training import mdlm_config, training_fixture


def test_resume_signature_rejects_nonmapping_configuration():
    with pytest.raises(
        ValueError, match="Resume configuration must resolve to a mapping"
    ):
        ck.resume_signature(
            OmegaConf.create([]),
            sources=[],
            codebook=None,
            accelerator=None,
            identity={},
        )


def test_resume_rejects_changed_current_budget(cuda_resume):
    payload, _ = cuda_resume
    expected = copy.deepcopy(payload["signature"])
    ck.validate_resume_signature(payload, expected)
    expected["config"]["train"]["max_steps"] += 1
    with pytest.raises(ValueError, match="signature mismatch"):
        ck.validate_resume_signature(payload, expected)


@pytest.fixture
def cuda_resume(tmp_path):
    from stok.data.mdlm import validate_mdlm_sources

    source, artifact = training_fixture(tmp_path)
    codebook = torch.load(artifact, weights_only=True)["codebook"]
    cfg = mdlm_config(tmp_path / "run", source, artifact)
    identity = validate_mdlm_sources(
        {"local": {"path": str(source)}}, {}, codebook=codebook, split_manifest=None
    )
    model = torch.nn.Linear(2, 1)
    model.register_buffer("structure_codebook", codebook)
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
            "total_missing_structure",
            "total_noncanonical_sequence",
        ],
        0,
    )
    logging["mdlm_running"] = torch.zeros(5, 2, dtype=torch.float64)
    rank = dict(
        rank=0,
        rng=rng,
        scaler=None,
        epoch=0,
        batches_in_epoch=0,
        loader_generator_state=torch.Generator().get_state(),
        logging=logging,
    )
    signature = ck.resume_signature(
        cfg, sources=[], codebook=codebook, accelerator=None, identity=identity
    )
    signature["execution"].update(
        device="cuda", world_size=2, cuda_rng_state_sizes=[16, 16]
    )
    payload = dict(
        runtime={
            "components": {
                "objective": "mdlm",
                "model": "stok_mdlm",
                "sequence_tokenizer": "native",
                "structure_representation": "frozen_vq",
                "optimizer": "adamw",
                "scheduler": "warmup_linear",
            },
            "effective_precision": "no",
            "execution": signature["execution"],
            "source": signature["source"],
            "software": signature["software"],
            "mdlm_identity": identity,
        },
        signature=signature,
        rank_states=[copy.deepcopy(rank), {**rank, "rank": 1}],
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        optimizer_initialized=[],
        scheduler=scheduler.state_dict(),
        config=OmegaConf.to_container(cfg, resolve=True),
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
@pytest.mark.parametrize("with_accelerator", [False, True])
def test_cpu_record_does_not_initialize_unused_cuda(
    tmp_path, monkeypatch, initialized, with_accelerator
):
    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(tmp_path / "run", source, codebook)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU state initialized unused CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.backends.cudnn, "version", forbidden)
    rng = ck.collect_rng_state(device=torch.device("cpu"))
    if not initialized:
        assert "cuda" not in ck.collect_rng_state()
    accelerator = (
        SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            mixed_precision="no",
            distributed_type=SimpleNamespace(name="NO"),
        )
        if with_accelerator
        else None
    )
    signature = ck.resume_signature(
        cfg,
        sources=[],
        codebook=None,
        accelerator=accelerator,
        identity={"training_signature": {}},
    )
    assert "cuda" not in rng
    assert signature["execution"]["cuda_rng_state_sizes"] == []
    assert signature["execution"]["cudnn"] is None
    ck.restore_rng_state(rng)


@pytest.mark.parametrize("old", [None, 1, 2, 99])
def test_reader_rejects_unsupported_checkpoints_clearly(tmp_path, old):
    path = tmp_path / "old.pt"
    torch.save({"format_version": old, "model": {}}, path)
    with pytest.raises(ValueError, match="version 3"):
        ck.read_training_checkpoint(path)


def test_package_source_identity_is_location_independent_and_content_sensitive(
    tmp_path,
):
    import shutil

    checkout = tmp_path / "checkout/stok"
    checkout.mkdir(parents=True)
    (checkout / "model.py").write_text("x = 1\n")
    (checkout / "configs").mkdir()
    (checkout / "configs/config.yaml").write_text("x: 1\n")
    wheel = tmp_path / "site-packages/stok"
    shutil.copytree(checkout, wheel)
    digest = ck.package_source_sha256(checkout)
    assert digest == ck.package_source_sha256(wheel)
    (wheel / "__pycache__").mkdir()
    (wheel / "__pycache__/ignored.py").write_text("cache")
    (wheel / "build").mkdir()
    (wheel / "build/ignored.py").write_text("build")
    assert digest == ck.package_source_sha256(wheel)
    (wheel / "configs/config.yaml").write_text("x: 2\n")
    assert digest != ck.package_source_sha256(wheel)
    (wheel / "configs/config.yaml").write_text("x: 1\n")
    (wheel / "model.py").rename(wheel / "other.py")
    assert digest != ck.package_source_sha256(wheel)


@pytest.mark.parametrize(
    "getter,setter",
    [
        ("flash_sdp_enabled", "enable_flash_sdp"),
        ("math_sdp_enabled", "enable_math_sdp"),
        ("mem_efficient_sdp_enabled", "enable_mem_efficient_sdp"),
        ("cudnn_sdp_enabled", "enable_cudnn_sdp"),
        ("fp16_bf16_reduction_math_sdp_allowed", "allow_fp16_bf16_reduction_math_sdp"),
    ],
)
def test_attention_policy_changes_resume_identity_without_cuda(
    tmp_path, monkeypatch, getter, setter
):
    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(tmp_path / "run", source, codebook)
    monkeypatch.setattr(
        torch.cuda, "_lazy_init", lambda: pytest.fail("CUDA initialized")
    )
    read, toggle = (
        getattr(torch.backends.cuda, getter),
        getattr(torch.backends.cuda, setter),
    )
    original = read()
    kwargs = dict(
        sources=[],
        codebook=None,
        accelerator=None,
        identity={"training_signature": "unused"},
    )
    try:
        before = ck.resume_signature(cfg, **kwargs)
        toggle(not original)
        after = ck.resume_signature(cfg, **kwargs)
        assert before != after
        with pytest.raises(ValueError, match="signature mismatch.*execution"):
            ck.validate_resume_signature({"signature": before}, after)
    finally:
        toggle(original)
    assert read() == original
