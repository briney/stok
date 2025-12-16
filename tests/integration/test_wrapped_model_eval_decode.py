import pytest
import torch
import torch.nn as nn
from importlib.resources import as_file, files

pytest.importorskip("x_transformers")


def test_eval_decode_with_wrapped_model(monkeypatch, tmp_path):
    from hydra import compose, initialize_config_dir
    from stok.cli.train import run_training
    from stok.utils.codebook import load_codebook
    from stok.models.decoder import GeometricDecoder

    # Build a tiny decoder ckpt matching the codebook preset
    codebook = load_codebook(preset="lite")
    d_code = int(codebook.shape[1])
    dec = GeometricDecoder(
        d_model=1024,
        n_heads=8,
        n_layers=12,
        ffn_mult=4.0,
        max_length=1280,
        d_code=d_code,
        num_memory_tokens=0,
        attn_kv_heads=2,
    )
    ckpt = tmp_path / "decoder-lite.pt"
    torch.save(dec.state_dict(), ckpt)

    class WrappedModel(nn.Module):
        def __init__(self, m: nn.Module):
            super().__init__()
            # Avoid any nn.Module interception during assignment
            object.__setattr__(self, "_m", m)

        def forward(self, *a, **kw):
            return object.__getattribute__(self, "_m")(*a, **kw)

        def parameters(self, recurse: bool = True):
            return object.__getattribute__(self, "_m").parameters(recurse=recurse)

        def train(self, mode: bool = True):
            object.__getattribute__(self, "_m").train(mode)
            return self

        def eval(self):
            object.__getattribute__(self, "_m").eval()
            return self

        def __getattr__(self, name: str):
            if name == "classifier":
                raise AttributeError("Wrapped model hides classifier")
            return getattr(object.__getattribute__(self, "_m"), name)

    class FakeAccelerator:
        device = torch.device("cpu")
        is_main_process = True
        num_processes = 1
        print = print

        def prepare(self, *objs):
            out = []
            for o in objs:
                if isinstance(o, nn.Module):
                    out.append(WrappedModel(o))
                else:
                    out.append(o)
            return tuple(out)

        def unwrap_model(self, model):
            return getattr(model, "_m", model)

        def backward(self, loss):
            loss.backward()

        def clip_grad_norm_(self, params, max_norm):
            nn.utils.clip_grad_norm_(list(params), max_norm)

        def gather_for_metrics(self, t):
            return t

        def wait_for_everyone(self):
            pass

    monkeypatch.setattr("stok.cli.train._maybe_get_accelerator", lambda: FakeAccelerator())

    overrides = [
        # tiny model for speed
        "model.encoder.d_model=64",
        "model.encoder.n_layers=2",
        "model.encoder.n_heads=4",
        "model.encoder.ffn_mult=1.0",
        "model.encoder.dropout=0.0",
        "model.encoder.attn_dropout=0.0",
        # small codebook preset
        "model.codebook.preset=lite",
        # enable decoder and eval-time decoding
        "model.decoder.enabled=true",
        f"model.decoder.path={ckpt.as_posix()}",
        "train.decoding.eval_enabled=true",
        "train.eval.steps=1",
        # short run
        "train.num_steps=2",
        "train.log_steps=1",
        "train.grad_accum_steps=1",
        # small data loader
        "data.batch_size=2",
        "data.max_len=64",
        "data.num_workers=0",
        "data.pin_memory=false",
        # disable external logging and write artifacts to temp dir
        "train.wandb.enabled=false",
        f"train.project_path={tmp_path.as_posix()}",
    ]

    with as_file(files("stok").joinpath("configs")) as cfg_dir:
        with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
            cfg = compose(config_name="config", overrides=overrides)
    run_training(cfg)

