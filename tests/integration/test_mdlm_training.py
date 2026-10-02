"""Real paired-Parquet training through the shared optimizer lifecycle."""

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from stok.cli.train import _build_dataloaders, run_training
from stok.data.mdlm import CANONICAL_AA, prepare_mdlm_batch
from stok.models.mdlm import STokMDLM
from stok.utils.mdlm import corrupt_mdlm_batch
from stok.utils.tokenizer import Tokenizer
from tests.utils.synthetic import make_mdlm_rows, write_dataset


def mdlm_config(project, source, codebook_path, **options):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "src/stok/configs"),
        version_base=None,
    ):
        cfg = compose(config_name="config")
    OmegaConf.set_struct(cfg, False)
    cfg.model.encoder.d_model = 16
    cfg.model.encoder.n_heads = 2
    cfg.model.encoder.n_layers = 1
    cfg.model.encoder.ffn_mult = 1
    cfg.model.encoder.dropout = cfg.model.encoder.attn_dropout = 0
    cfg.model.codebook.path = str(codebook_path)
    cfg.data.train = {"local": {"path": str(source)}}
    cfg.data.eval = {}
    cfg.data.batch_size = 2
    cfg.data.max_len = 8
    cfg.data.pin_memory = False
    cfg.data.num_workers = 0
    cfg.data.shuffle_shards = cfg.data.shuffle_rows = False
    cfg.train.objective = "mdlm"
    cfg.train.precision = "no"
    cfg.train.num_steps = 2
    cfg.train.optimizer.lr = 0.002
    cfg.train.scheduler.warmup_steps = cfg.train.scheduler.decay_steps = 0
    cfg.train.grad_clip_norm = 0
    cfg.train.log_steps = 1
    cfg.train.checkpoint_steps = 1
    cfg.train.wandb.enabled = cfg.train.console.enabled = False
    cfg.train.project_path = str(project)
    cfg.train.mdlm = {
        "placement": "token",
        "span_mean": 8,
        "regime_weights": {"joint_independent": 1},
        "noise": {"name": "linear", "power": 2, "min_mask_probability": 1e-4},
        "sequence_loss_weight": 1,
        "structure_loss_weight": 1,
    }
    for key, value in options.items():
        OmegaConf.update(cfg, key, value, force_add=True, merge=False)
    return cfg


def training_fixture(tmp_path, n=8, rows=None):
    codebook = torch.arange(64, dtype=torch.float32).reshape(32, 2)
    path = tmp_path / "codebook.pt"
    torch.save({"codebook": codebook}, path)
    if rows is None:
        rows = []
        originals = make_mdlm_rows()
        for i in range(n):
            row = originals[i % 2].copy()
            row["sequence_id"] = str(i)
            rows.append(row)
    return write_dataset(tmp_path / "data", rows, codebook=codebook), path


def checkpoint(cfg):
    run_training(cfg)
    return torch.load(
        Path(cfg.train.project_path) / "model/final.pt",
        map_location="cpu",
        weights_only=False,
    )


def test_paired_parquet_update_both_heads_and_compute_accounting(tmp_path):
    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(tmp_path / "run", source, codebook)
    state = checkpoint(cfg)
    assert state["global_step"] == state["scheduler"]["last_epoch"] == 2
    names = [
        name for name, value in state["model"].items() if name != "structure_codebook"
    ]
    states = dict(zip(names, state["optimizer"]["state"].values()))
    for head in ("sequence_bias", "structure_bias"):
        assert torch.count_nonzero(states[head]["exp_avg"]) > 0
    assert state["executed_positions"] == 2 * 2 * 8
    assert state["residues_seen"] == 2 * (6 + 3)
    assert state["config"]["train"]["effective_precision"] == "no"
    assert state["config"]["train"]["mdlm_identity"]["training_signature"]
    log = (tmp_path / "run/logs/train.log").read_text()
    assert "diffusion_loss" in log and "sequence_ce" in log and "structure_ce" in log
    assert "ppl" not in log
    assert "noncanonical_sequence 2" in log
    params = sum(
        value.numel()
        for name, value in state["model"].items()
        if name != "structure_codebook"
    )
    assert f"flops_actual={6 * params * 32}" in log


def test_raw_loader_defers_crop_and_reads_full_coordinates_when_requested(tmp_path):
    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(tmp_path / "run", source, codebook, **{"data.load_coords": True})
    from stok.data.mdlm import validate_mdlm_sources

    cfg.train.mdlm_identity = validate_mdlm_sources(
        {"local": str(source)},
        {},
        codebook=torch.load(codebook, weights_only=True)["codebook"],
        split_manifest=None,
    )
    loader, _ = _build_dataloaders(cfg, codebook_size=32, pad_id=1, objective="mdlm")
    batch = next(iter(loader))
    assert isinstance(batch, list) and isinstance(batch[0], dict)
    assert loader.collate_fn is list
    assert batch[0]["coords"].shape[0] == len(batch[0]["sequence"])
    assert batch[0]["dataset"] == cfg.train.mdlm_identity.sample_key_namespaces.local
    cfg.data.load_coords = False
    loader, _ = _build_dataloaders(cfg, codebook_size=32, pad_id=1, objective="mdlm")
    assert "coords" not in next(iter(loader))[0]
    with pytest.raises(ValueError, match="[Cc]onflict"):
        _build_dataloaders(
            cfg, codebook_size=32, pad_id=1, objective="mdlm", is_mlm=True
        )


def test_explicit_non_mlm_flag_conflicts_with_mlm_objective(tmp_path):
    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(tmp_path / "run", source, codebook)
    with pytest.raises(ValueError, match="[Cc]onflict"):
        _build_dataloaders(
            cfg, codebook_size=32, pad_id=1, objective="mlm", is_mlm=False
        )


def test_mdlm_partial_window_flushes(tmp_path):
    source, codebook = training_fixture(tmp_path, n=10)
    cfg = mdlm_config(
        tmp_path / "run",
        source,
        codebook,
        **{"train.epochs": 1, "train.grad_accum_steps": 4},
    )
    state = checkpoint(cfg)
    assert state["global_step"] == state["scheduler"]["last_epoch"] == 2
    assert state["micro_step"] == 5
    assert state["executed_positions"] == 80


def test_mdlm_accumulation_matches_full_batch(tmp_path):
    source, codebook = training_fixture(tmp_path)
    small = checkpoint(
        mdlm_config(
            tmp_path / "small", source, codebook, **{"train.grad_accum_steps": 2}
        )
    )
    large = checkpoint(
        mdlm_config(tmp_path / "large", source, codebook, **{"data.batch_size": 4})
    )
    for name in small["model"]:
        torch.testing.assert_close(
            small["model"][name], large["model"][name], atol=3e-6, rtol=3e-5
        )
    assert small["executed_positions"] == large["executed_positions"]
    assert small["residues_seen"] == large["residues_seen"]


def test_tiny_paired_subset_learns(tmp_path):
    rows = [
        {
            **make_mdlm_rows()[1],
            "sequence": "AAA",
            "structure_tokens": [0, 0, 0],
            "sequence_id": str(i),
        }
        for i in range(2)
    ]
    source, codebook = training_fixture(tmp_path, rows=rows)
    states = []
    for steps in (0, 60):
        cfg = mdlm_config(
            tmp_path / f"run{steps}",
            source,
            codebook,
            **{
                "train.num_steps": steps,
                "train.optimizer.lr": 0.02,
                "train.mdlm.noise.min_mask_probability": 0.99,
            },
        )
        states.append(checkpoint(cfg))
    model = STokMDLM(
        vocab_size=32,
        pad_id=1,
        codebook=torch.load(codebook, weights_only=True)["codebook"],
        d_model=16,
        n_heads=2,
        n_layers=1,
        ffn_mult=1,
        dropout=0,
        attn_dropout=0,
    )
    batch = prepare_mdlm_batch(
        rows, Tokenizer(), max_len=8, codebook_size=32, crop="center", seeds=[0, 0]
    )
    corruption = corrupt_mdlm_batch(
        batch,
        cfg.train.mdlm,
        seeds=[0, 0],
        regime="joint_independent",
        mask_probability=1,
    )
    from stok.utils.mdlm import mdlm_loss_terms

    losses = []
    for state in states:
        model.load_state_dict(state["model"])
        terms = mdlm_loss_terms(
            model(corruption["sequence_tokens"], corruption["structure_tokens"]),
            batch,
            corruption,
            canonical_aa_ids=torch.tensor(
                Tokenizer().convert_tokens_to_ids(list(CANONICAL_AA))
            ),
        )
        losses.append(
            float(terms["ce_sum"].sum().detach() / terms["masked_count"].sum())
        )
    assert losses[1] < losses[0] * 0.2


@pytest.mark.parametrize(
    "weights", [(-1, 1), (float("nan"), 1), (1, float("inf")), (0, 0)]
)
def test_invalid_loss_weights_fail_before_artifacts(tmp_path, weights):
    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(
        tmp_path / "run",
        source,
        codebook,
        **{
            "train.mdlm.sequence_loss_weight": weights[0],
            "train.mdlm.structure_loss_weight": weights[1],
        },
    )
    with pytest.raises(RuntimeError, match="weight"):
        run_training(cfg)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "regime,weights", [("structure_only", (1, 0)), ("sequence_only", (0, 1))]
)
def test_enabled_regime_needs_positively_weighted_modality(tmp_path, regime, weights):
    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(
        tmp_path / "run",
        source,
        codebook,
        **{
            "train.mdlm.regime_weights": {regime: 1},
            "train.mdlm.sequence_loss_weight": weights[0],
            "train.mdlm.structure_loss_weight": weights[1],
        },
    )
    with pytest.raises(RuntimeError, match="regime"):
        run_training(cfg)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("train.fape.enabled", True),
        ("train.mlm.random_token_prob", 0.2),
        ("train.mlm.mask_prob", 0.2),
        ("train.mlm.mask_token_prob", 1),
        ("train.mlm.enabled", True),
        ("train.mdlm.noise.name", "bad"),
    ],
)
def test_unsupported_config_fails_before_artifacts(tmp_path, key, value):
    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(tmp_path / "run", source, codebook, **{key: value})
    with pytest.raises(RuntimeError):
        run_training(cfg)
    assert not (tmp_path / "run").exists()


def test_bad_source_fails_before_wandb_or_artifacts(tmp_path, monkeypatch):
    source, codebook = training_fixture(tmp_path)
    manifest = source / "manifest.json"
    data = json.loads(manifest.read_text())
    data["status"] = "incomplete"
    manifest.write_text(json.dumps(data))
    from stok.cli import train

    def forbidden(*args, **kwargs):
        pytest.fail("W&B started before preflight")

    monkeypatch.setattr(train, "_maybe_wandb_login", forbidden)
    with pytest.raises((ValueError, RuntimeError)):
        run_training(mdlm_config(tmp_path / "run", source, codebook))
    assert not (tmp_path / "run").exists()


def test_no_eligible_window_skips_but_preserves_consumed_cursor(tmp_path):
    rows = [
        {
            **make_mdlm_rows()[1],
            "sequence_id": str(i),
            "structure_tokens": [None] * 3 if i < 2 else [0, 1, 2],
        }
        for i in range(4)
    ]
    source, codebook = training_fixture(tmp_path, rows=rows)
    state = checkpoint(
        mdlm_config(
            tmp_path / "run",
            source,
            codebook,
            **{
                "train.num_steps": 1,
                "train.mdlm.regime_weights": {"structure_only": 1},
            },
        )
    )
    assert state["global_step"] == state["scheduler"]["last_epoch"] == 1
    assert state["micro_step"] == 2
    assert state["residues_seen"] == 12
    assert state["executed_positions"] == 16
    assert all(s["step"].item() == 1 for s in state["optimizer"]["state"].values())


def test_all_unusable_mdlm_pass_fails_without_update(tmp_path):
    rows = [
        {**make_mdlm_rows()[1], "sequence_id": str(i), "structure_tokens": [None] * 3}
        for i in range(4)
    ]
    source, codebook = training_fixture(tmp_path, rows=rows)
    cfg = mdlm_config(
        tmp_path / "run",
        source,
        codebook,
        **{"train.mdlm.regime_weights": {"structure_only": 1}},
    )
    with pytest.raises(RuntimeError, match="no successful optimizer update"):
        run_training(cfg)
    assert not list((tmp_path / "run/checkpoints").glob("step_*.pt"))


@pytest.mark.parametrize("samples", [2, 8])
def test_real_cpu_amp_skip_preserves_cursor_without_advancing_schedule(
    tmp_path, monkeypatch, samples
):
    from stok.cli import train

    original = train._maybe_get_accelerator
    scaler = torch.amp.GradScaler("cpu")
    assert scaler.is_enabled()

    def cpu_scaler(precision=None):
        accelerator = original(precision)
        accelerator.scaler = scaler
        return accelerator

    monkeypatch.setattr(train, "_maybe_get_accelerator", cpu_scaler)
    init = STokMDLM.__init__
    calls = []

    def instrument(self, **kwargs):
        init(self, **kwargs)

        def overflow_once(grad):
            calls.append(1)
            return torch.full_like(grad, float("inf")) if len(calls) == 1 else grad

        self.sequence_bias.register_hook(overflow_once)

    monkeypatch.setattr(STokMDLM, "__init__", instrument)
    source, codebook = training_fixture(tmp_path, n=samples)
    state = checkpoint(mdlm_config(tmp_path / "run", source, codebook))
    assert len(calls) == 3
    assert scaler.get_scale() == 32768  # actual overflow detected by GradScaler
    assert state["global_step"] == state["scheduler"]["last_epoch"] == 2
    assert state["micro_step"] == 3
    assert state["executed_positions"] == 48
    assert all(s["step"].item() == 2 for s in state["optimizer"]["state"].values())


def test_real_zero_mask_draw_still_updates_adamw(tmp_path):
    original = make_mdlm_rows()[1]
    rows = [
        {
            **original,
            "sequence_id": str(i),
            "sequence": "A",
            "structure_tokens": [None],
            "coordinates": original["coordinates"][:1],
            "residue_map": original["residue_map"][:1],
        }
        for i in range(2)
    ]
    source, codebook = training_fixture(tmp_path, rows=rows)
    initial = checkpoint(
        mdlm_config(
            tmp_path / "initial",
            source,
            codebook,
            **{"train.num_steps": 0, "train.seed": 1},
        )
    )
    cfg = mdlm_config(
        tmp_path / "run",
        source,
        codebook,
        **{
            "train.num_steps": 1,
            "train.seed": 1,
            "train.mdlm.regime_weights": {"sequence_only": 1},
        },
    )
    state = checkpoint(cfg)
    log = (tmp_path / "run/logs/train.log").read_text()
    assert "sequence_masked 0/2" in log and "structure_masked 0/0" in log
    assert (
        state["global_step"]
        == state["scheduler"]["last_epoch"]
        == state["micro_step"]
        == 1
    )
    assert all(s["step"].item() == 1 for s in state["optimizer"]["state"].values())
    assert all(
        torch.count_nonzero(s["exp_avg"]) == 0
        for s in state["optimizer"]["state"].values()
    )
    # Connected zero invokes ordinary AdamW decay even though no targets were masked.
    torch.testing.assert_close(
        state["model"]["embed.weight"],
        initial["model"]["embed.weight"] * (1 - 0.002 * 0.01),
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("model.encoder.n_heads", 3),
        ("model.codebook.trainable", True),
        ("model.decoder.freeze", False),
        ("train.scheduler.decay", "bad"),
        ("train.scheduler.warmup_steps", -1),
        ("train.optimizer.lr", -1),
    ],
)
def test_invalid_mdlm_model_configuration_leaves_no_artifacts(tmp_path, key, value):
    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(tmp_path / "run", source, codebook, **{key: value})
    with pytest.raises(RuntimeError):
        run_training(cfg)
    assert not (tmp_path / "run").exists()


def test_mdlm_has_no_legacy_classifier_configuration_dependency(tmp_path):
    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(tmp_path / "run", source, codebook, **{"train.num_steps": 1})
    del cfg.model.classifier
    state = checkpoint(cfg)
    assert state["global_step"] == 1


def test_training_attaches_saved_regime_weights_for_sampling(tmp_path, monkeypatch):
    source, codebook = training_fixture(tmp_path)
    cfg = mdlm_config(tmp_path / "run", source, codebook)
    seen = []
    original = STokMDLM.forward

    def forward(self, *args, **kwargs):
        seen.append(getattr(self, "mdlm_regime_weights", None))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(STokMDLM, "forward", forward)
    state = checkpoint(cfg)
    assert seen and all(weights == {"joint_independent": 1} for weights in seen)
    assert state["config"]["train"]["mdlm"]["regime_weights"] == {
        "joint_independent": 1
    }
