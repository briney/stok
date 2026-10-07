from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from click.testing import CliRunner

from stok.cli.cli import cli
from stok.utils.codebook import load_codebook


pytest.importorskip("pyarrow")
pytest.importorskip("x_transformers")


def _make_coords(L: int) -> list[list[list[float]]]:
    out = []
    for i in range(L):
        out.append([[float(i), 0.0, 0.0], [float(i), 1.0, 0.0], [float(i), 0.0, 1.0]])
    return out


def _write_parquet_with_coords(
    path: Path, n_rows: int, seq_min_len: int, seq_max_len: int, indices_len: int
):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_rows):
        L = np.random.randint(seq_min_len, seq_max_len + 1)
        seq = "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=L))
        rows.append(
            {
                "sequence_id": f"pc{i}",
                "sequence": seq,
                "structure_tokens": [0] * len(seq),
                "coordinates": _make_coords(L),
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def _make_decoder_ckpt(tmp_path: Path, preset: str = "lite") -> Path:
    from stok.models.decoder import GeometricDecoder

    # match d_code to codebook preset
    codebook = load_codebook(preset=preset)
    d_code = int(codebook.shape[1])
    from stok.models.decoder import _DECODER_ARCH

    arch = _DECODER_ARCH[preset]
    model = GeometricDecoder(
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        ffn_mult=arch["ffn_mult"],
        max_length=arch["max_length"],
        d_code=d_code,
        num_memory_tokens=arch["num_memory_tokens"],
        attn_kv_heads=arch["attn_kv_heads"],
    )
    ckpt_path = tmp_path / f"decoder-{preset}.pt"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def test_training_with_decoder_and_fape(tmp_path, monkeypatch):
    from stok.models.decoder import _DECODER_ARCH

    monkeypatch.setitem(
        _DECODER_ARCH,
        "lite",
        dict(
            d_model=32,
            ffn_mult=1.0,
            n_layers=1,
            n_heads=2,
            attn_kv_heads=1,
            num_memory_tokens=0,
            max_length=32,
        ),
    )
    runner = CliRunner()

    max_len = 16
    indices_len = max_len - 2  # align with token positions excluding BOS/EOS

    train_pq = tmp_path / "train.parquet"
    eval_pq = tmp_path / "eval.parquet"
    _write_parquet_with_coords(
        train_pq, n_rows=4, seq_min_len=12, seq_max_len=18, indices_len=indices_len
    )
    _write_parquet_with_coords(
        eval_pq, n_rows=2, seq_min_len=12, seq_max_len=18, indices_len=indices_len
    )

    ckpt_path = _make_decoder_ckpt(tmp_path, preset="lite")

    overrides = [
        f"data.train={train_pq.as_posix()}",
        f"data.eval={eval_pq.as_posix()}",
        # tiny model for speed
        "model.encoder.d_model=64",
        "model.encoder.n_layers=2",
        "model.encoder.n_heads=4",
        "model.encoder.ffn_mult=1.0",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
        # small codebook preset
        "model.codebook.preset=lite",
        # enable decoder + fape
        "model.decoder.enabled=true",
        f"model.decoder.path={ckpt_path.as_posix()}",
        "train.fape.enabled=true",
        "train.fape.start_step=0",
        "train.decoding.eval_enabled=true",
        # small data loader
        "data.load_coords=true",
        "train.batch_size=2",
        f"data.max_len={max_len}",
        "data.num_workers=0",
        "data.pin_memory=false",
        # short run and ensure eval triggers
        "train.max_steps=2",
        "train.lr=0.01",
        "train.warmup_steps=0",
        "train.log_every=1",
        "train.eval.steps=2",
        # disable external logging
        "train.wandb.enabled=false",
        # write artifacts to temp dir
        f"train.output_dir={tmp_path.as_posix()}",
    ]

    states = []
    for weight in (0.0, 10.0):
        result = runner.invoke(
            cli, ["train", *overrides, f"train.fape.weight={weight}"]
        )
        assert result.exit_code == 0, (result.output, result.exception)
        assert "fape " in result.output
        state = torch.load(
            tmp_path / "model/final.pt", weights_only=False, map_location="cpu"
        )["model"]
        assert all(torch.isfinite(v).all() for v in state.values())
        states.append(state)
    assert any(not torch.equal(states[0][key], states[1][key]) for key in states[0])


def test_fape_only_missing_coordinates_produces_finite_update(tmp_path, monkeypatch):
    from hydra import compose, initialize_config_dir
    from stok.cli.train import run_training
    from stok.models.decoder import _DECODER_ARCH
    from tests.integration.test_training_progress import training_command
    from types import SimpleNamespace

    payloads = []
    monkeypatch.setattr(
        "stok.training.engine._maybe_init_wandb",
        lambda *args, **kwargs: SimpleNamespace(
            log=lambda data, **kwargs: payloads.append(data)
        ),
    )
    monkeypatch.setitem(
        _DECODER_ARCH,
        "lite",
        dict(
            d_model=32,
            ffn_mult=1.0,
            n_layers=1,
            n_heads=2,
            attn_kv_heads=1,
            num_memory_tokens=0,
            max_length=32,
        ),
    )
    decoder_path = _make_decoder_ckpt(tmp_path)
    source = tmp_path / "structure.parquet"
    coords = np.asarray(_make_coords(4))
    coords[2] = np.nan
    pq.write_table(
        pa.table(
            {
                "sequence_id": [str(i) for i in range(4)],
                "sequence": ["LAGV"] * 4,
                "structure_tokens": pa.array(
                    [[None] * 4] * 4, type=pa.list_(pa.int64())
                ),
                "coordinates": [coords.tolist()] * 4,
            }
        ),
        source,
    )
    overrides = training_command(
        tmp_path / "run",
        f"data.train={source}",
        "data.load_coords=true",
        f"model.decoder.path={decoder_path}",
        "train.fape.enabled=true",
        "train.gradient_accumulation_steps=2",
        "train.warmup_steps=0",
        "train.log_every=1",
    )[3:]
    states = []
    for updates in (0, 1):
        with initialize_config_dir(
            config_dir=str(Path(__file__).resolve().parents[2] / "src/stok/configs"),
            version_base=None,
        ):
            cfg = compose(
                config_name="config",
                overrides=[*overrides, f"train.max_steps={updates}"],
            )
        run_training(cfg)
        checkpoint = torch.load(
            tmp_path / "run/model/final.pt", weights_only=False, map_location="cpu"
        )
        assert checkpoint["global_step"] == updates
        states.append(checkpoint["model"])
    assert all(torch.isfinite(value).all() for value in states[1].values())
    assert any(not torch.equal(states[0][name], states[1][name]) for name in states[0])
    assert " | acc unavailable" in (tmp_path / "run/logs/train.log").read_text()
    assert len(payloads) == 1
    assert payloads[0]["train/acc/num_valid"] == 0
    assert payloads[0]["train/fape_loss/num_valid"] == 4
    assert (
        not {"train/acc", "train/mask_acc", "train/cls_loss", "train/ppl"}
        & payloads[0].keys()
    )

    # Empty CE cannot make an inactive geometric reduction an optimizer update.
    for case in ("zero_weight", "before_start", "missing_coordinates"):
        cfg.train.output_dir = str(tmp_path / case)
        cfg.train.fape.weight = 0.0 if case == "zero_weight" else 1.0
        cfg.train.fape.start_step = 1 if case == "before_start" else 0
        if case == "missing_coordinates":
            table = pq.read_table(source)
            rows = table.to_pylist()
            for row in rows:
                row["coordinates"] = np.full((4, 3, 3), np.nan).tolist()
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), source)
        with pytest.raises(RuntimeError, match="no successful optimizer update"):
            run_training(cfg)
        assert not (Path(cfg.train.output_dir) / "model/final.pt").exists()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/ROCm accelerator required"
)
def test_real_decoder_autocast_finite_gradients(dtype):
    from stok.models.decoder import GeometricDecoder
    from stok.utils.decoding import decode_token_aligned_coords
    from stok.utils.losses import fape_loss

    decoder = (
        GeometricDecoder(
            d_model=32,
            n_heads=2,
            n_layers=1,
            ffn_mult=1.0,
            max_length=16,
            d_code=8,
            num_memory_tokens=0,
            attn_kv_heads=1,
        )
        .cuda()
        .eval()
    )
    decoder.requires_grad_(False)
    codes = torch.nn.Parameter(torch.randn(1, 6, 8, device="cuda"))
    optimizer = torch.optim.AdamW([codes], lr=0.01)
    mask = torch.tensor([[False, True, True, True, True, False]], device="cuda")
    true = torch.tensor(_make_coords(6), device="cuda")[None]
    true[:, 2] = float("nan")
    before = codes.detach().clone()
    with torch.autocast("cuda", dtype=dtype):
        pred = decode_token_aligned_coords(decoder, codes, mask)
        loss = fape_loss(pred, true, residue_mask=mask)
    loss.backward()
    assert torch.isfinite(codes.grad).all() and codes.grad.abs().sum() > 0
    optimizer.step()
    assert torch.isfinite(codes).all() and not torch.equal(codes, before)


@pytest.mark.parametrize(
    "train,eligible,diagnostics,accuracy_key",
    [
        (
            {"objective": "CODEBOOK", "fape": {"enabled": True, "start_step": 0}},
            1,
            True,
            "acc",
        ),
        ({"fape": {"enabled": True, "start_step": 0}}, 1, True, "acc"),
        ({"objective": "codebook", "fape": {"start_step": 0}}, 0, True, "acc"),
        ({"objective": "codebook"}, 0, True, "acc"),
        ({}, 0, True, "acc"),
        (
            {"objective": "MLM", "fape": {"log_pred_nan_frac": True}},
            0,
            False,
            "mask_acc",
        ),
        ({"objective": "mlm"}, 0, False, "mask_acc"),
        (
            {
                "objective": "codebook",
                "fape": {"enabled": True, "start_step": 0, "log_pred_nan_frac": False},
            },
            1,
            False,
            "acc",
        ),
    ],
)
def test_classification_task_preserves_legacy_startup_defaults(
    train, eligible, diagnostics, accuracy_key
):
    from omegaconf import OmegaConf
    from stok.training.tasks import ClassificationTask

    cfg = OmegaConf.create(
        {
            "train": train,
            "model": {"encoder": {"pad_id": 1}, "classifier": {"ignore_index": -100}},
        }
    )
    authored = OmegaConf.to_yaml(cfg)
    OmegaConf.set_readonly(cfg, True)
    task = ClassificationTask(cfg, decoder=torch.nn.Identity(), codebook=None)
    batch = (
        torch.tensor([[0, 3, 2]]),
        torch.full((1, 3), -100),
        torch.zeros(1, 3, 3, 3),
    )
    window = task.prepare_window(
        [batch], epoch=0, micro_step=0, global_step=0, rank=0, world_size=1
    )
    assert window.counts.tolist() == [0, eligible, 3]
    assert task.log_pred_nan_frac is diagnostics
    # Logging selection is a startup contract even without a FAPE mapping;
    # nonempty training still requires the legacy fape.weight field.
    state = task.logging_state()
    state.update(
        running_updates=1, running_masked_acc_count=1, running_masked_acc_sum=1
    )
    task.restore_logging_state(state)
    _, payload = task.format_log(
        step=1,
        max_steps=1,
        micro_step=1,
        epoch=0.0,
        lr=0.01,
        flops=0,
        residues_seen=0,
        executed_positions=3,
    )
    assert payload[f"train/{accuracy_key}/num_valid"] == 1.0
    assert payload[f"train/{accuracy_key}"] == 1.0
    assert f"train/{'acc' if accuracy_key == 'mask_acc' else 'mask_acc'}" not in payload
    assert OmegaConf.to_yaml(cfg) == authored
