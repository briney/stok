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
            "evaluation_protocol": None,
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


@pytest.mark.parametrize("old", [None, 1, 2, 3, 99])
def test_reader_rejects_unsupported_checkpoints_clearly(tmp_path, old):
    path = tmp_path / "old.pt"
    torch.save({"format_version": old, "model": {}}, path)
    with pytest.raises(ValueError, match="version 4"):
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


@pytest.mark.parametrize(
    "field",
    [
        "canonical_population_sha256",
        "representation_sha256",
        "replay_sha256",
        "protocol",
    ],
)
def test_format4_reader_binds_canonical_case_and_protocol_metadata(
    tmp_path, cuda_resume, field
):
    payload, _ = cuda_resume
    payload.update(format_version=4, wandb_run_id=None)
    payload["runtime"]["evaluation_protocol"] = None
    path = tmp_path / "current.pt"
    torch.save(payload, path)
    assert ck.read_training_checkpoint(path)["format_version"] == 4
    identity = payload["runtime"]["mdlm_identity"]
    if field == "canonical_population_sha256":
        identity[field] = "f" * 64
    elif field == "protocol":
        payload["runtime"]["evaluation_protocol"] = {"protocol_sha256": "f" * 64}
    else:
        identity["sources"]["local"][field] = "f" * 64
    torch.save(payload, path)
    with pytest.raises(ValueError):
        ck.read_training_checkpoint(path)


@pytest.fixture(scope="module")
def canonical_checkpoint(tmp_path_factory):
    import shutil
    from tests.integration.test_mdlm_evaluation import evaluation_training_fixture
    from stok.training.engine import run_training

    root = tmp_path_factory.mktemp("canonical-checkpoint")
    cfg = evaluation_training_fixture(root)
    cfg.train.max_steps = 1
    run_training(cfg)
    checkpoint = root / "run/model/final.pt"
    payload = ck.read_training_checkpoint(checkpoint)
    for name in ("data", "heldout", "cases"):
        shutil.rmtree(root / name)
    (root / "codebook.pt").unlink()
    (root / "splits.jsonl").unlink()
    # A complete reader has no reason to reobserve disappeared original artifacts.
    assert ck.read_training_checkpoint(checkpoint)["global_step"] == 1
    return payload


@pytest.mark.parametrize(
    "damage",
    [
        "seed",
        "mask",
        "map",
        "case_request",
        "family_selection",
        "sampler",
        "protocol",
        "numerical",
        "population",
        "representation",
        "replay",
    ],
)
def test_format4_reader_and_sampler_validate_frozen_metadata_offline(
    tmp_path, canonical_checkpoint, damage
):
    from click.testing import CliRunner
    from stok.cli.cli import cli

    payload = copy.deepcopy(canonical_checkpoint)
    identity = payload["runtime"]["mdlm_identity"]
    case = identity["shared_cases"]["cases"][0]
    if damage == "seed":
        case["seed"] += 1
    elif damage == "mask":
        case["masked"][0][0] = not case["masked"][0][0]
    elif damage == "map":
        case["residue_map_sha256"] = "f" * 64
    elif damage == "case_request":
        identity["shared_cases"]["request"]["seed"] += 1
    elif damage == "family_selection":
        payload["config"]["train"]["eval"]["mdlm"]["families"] = ["absent"]
    elif damage == "sampler":
        payload["config"]["train"]["eval"]["mdlm"]["generation"]["sampling_steps"] += 1
    elif damage == "protocol":
        payload["runtime"]["evaluation_protocol"]["settings"]["generation"][
            "sampling_steps"
        ] += 1
    elif damage == "numerical":
        for owner in (payload["runtime"], payload["signature"]):
            owner["execution"]["threads"] = 9
    elif damage == "population":
        identity["canonical_population_sha256"] = "f" * 64
    else:
        identity["sources"]["validation"][damage + "_sha256"] = "f" * 64
    checkpoint = tmp_path / "damaged.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError):
        ck.read_training_checkpoint(checkpoint)
    inputs, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    inputs.write_text('{"sequence_id":"request","length":3}\n')
    result = CliRunner().invoke(
        cli,
        [
            "sample",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(inputs),
            "--output",
            str(output),
            "--mode",
            "joint",
            "--device",
            "cuda",
        ],
    )
    assert result.exit_code != 0
    assert not output.exists()
    assert "CUDA" not in result.output


@pytest.mark.parametrize("damage", ["coverage_reason", "source_policy"])
def test_format4_reader_binds_saved_representation_coverage(
    tmp_path, canonical_checkpoint, damage
):
    payload = copy.deepcopy(canonical_checkpoint)
    identity = payload["runtime"]["mdlm_identity"]
    if damage == "coverage_reason":
        key = identity["shared_cases"]["cases"][0]["canonical_id"]
        identity["coverage"][key]["reason"] = "fabricated exclusion"
    else:
        identity["sources"]["validation"]["policy"] = {"sequence_mode": "fabricated"}
    checkpoint = tmp_path / "damaged.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError):
        ck.read_training_checkpoint(checkpoint)


@pytest.mark.parametrize("damage", ["unsupported", "missing", "bool", "nonmapping"])
def test_format4_reader_and_sampler_reject_saved_case_wrapper_offline(
    tmp_path, canonical_checkpoint, monkeypatch, damage
):
    import builtins
    import io
    import os
    from pathlib import Path
    from click.testing import CliRunner
    from stok.cli.cli import cli

    payload = copy.deepcopy(canonical_checkpoint)
    identity = payload["runtime"]["mdlm_identity"]
    shared = identity["shared_cases"]
    blocked = [Path(source["path"]) for source in identity["sources"].values()]
    blocked += [Path(ref["directory"]) for ref in shared["canonical_inventories"]]
    blocked += [
        Path(payload["config"]["train"]["eval"]["mdlm"]["case_manifest"]),
        Path(payload["config"]["data"]["split_manifest"]),
        Path(payload["config"]["model"]["codebook"]["path"]),
    ]
    assert all(not path.exists() for path in blocked)
    original_open = io.open

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(file)).resolve()
            if any(path == root or path.is_relative_to(root) for root in blocked):
                raise AssertionError("checkpoint reader reopened an original artifact")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(
        torch.cuda,
        "_lazy_init",
        lambda: pytest.fail("saved metadata validation initialized CUDA"),
    )
    for path in blocked:
        with pytest.raises(AssertionError, match="original artifact"):
            path.open("rb")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)
    assert ck.read_training_checkpoint(checkpoint)["global_step"] == 1
    if damage == "unsupported":
        shared["schema_version"] = 99
    elif damage == "missing":
        del shared["schema_version"]
    elif damage == "bool":
        shared["schema_version"] = True
    else:
        identity["shared_cases"] = []
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="case artifact"):
        ck.read_training_checkpoint(checkpoint)
    inputs, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    inputs.write_text('{"sequence_id":"request","length":3}\n')
    result = CliRunner().invoke(
        cli,
        [
            "sample",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(inputs),
            "--output",
            str(output),
            "--mode",
            "joint",
            "--device",
            "cuda",
        ],
    )
    assert result.exit_code != 0 and "case artifact" in result.output
    assert not output.exists()
    assert "CUDA" not in result.output
