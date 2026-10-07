"""Bounded production continuation/profile; run from repository root."""

import json
import os
from pathlib import Path
import subprocess
import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from tests.integration.test_mdlm_resume import PROBE, equal

import time
from stok.data.mdlm import CANONICAL_AA
from stok.eval.mdlm import validate_mdlm_decoder
from stok.models.decoder import load_pretrained_decoder
from stok.models.mdlm import STokMDLM
from stok.utils.decoding import decode_token_aligned_coords
from stok.utils.mdlm import build_mask_groups, corrupt_mdlm_batch, mdlm_loss_terms
from stok.utils.sampling import inference_context, sample_mdlm
from stok.utils.tokenizer import Tokenizer


root = Path(__file__).resolve().parents[3]
out = Path(os.environ["STOK_MDLM_OUTPUT"])
out.mkdir(parents=True, exist_ok=False)
with initialize_config_dir(
    config_dir=str(root / "src/stok/configs"), version_base=None
):
    cfg = compose(
        config_name="config",
        overrides=[
            "model=mdlm_150m",
            "train=mdlm_pilot",
            f"model.codebook.path={os.environ['STOK_MDLM_ARCHIVE']}",
            f"+data.train.public_diagnostic.path={os.environ['STOK_MDLM_SOURCE']}",
            "train.max_steps=4",
            "train.warmup_steps=0",
            "train.save_every=1",
            "train.log_every=1",
            "train.wandb.enabled=false",
            "train.console.enabled=false",
            "train.eval.mdlm.enabled=false",
            "train.eval.mdlm.generation.enabled=false",
        ],
    )
profile = r"""
import json, time
metrics = {'updates': [], 'checkpoints': []}
start = [None]
original_corrupt_profile = tasks.corrupt_mdlm_batch
original_step = torch.optim.AdamW.step
original_save_profile = train._save_checkpoint
original_restore_profile = train.restore_training_state
original_read_profile = train.read_training_checkpoint

def read_profile(*args, **kwargs):
    started = time.perf_counter()
    result = original_read_profile(*args, **kwargs)
    metrics['checkpoint_read_seconds'] = time.perf_counter()-started
    return result

def restore_profile(*args, **kwargs):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = original_restore_profile(*args, **kwargs)
    torch.cuda.synchronize()
    metrics['restore'] = {'seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    return result

def corrupt_profile(batch, *args, **kwargs):
    torch.cuda.synchronize()
    start[0] = time.perf_counter()
    result = original_corrupt_profile(batch, *args, **kwargs)
    metrics['current_residues'] = int(batch['residue_mask'].sum())
    metrics['current_positions'] = batch['sequence_tokens'].numel()
    metrics['shape'] = list(batch['sequence_tokens'].shape)
    return result

def step(self, *args, **kwargs):
    params = [p for group in self.param_groups for p in group['params']]
    assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
    heads = [p for p in params if p.ndim == 1 and p.numel() in (int(cfg.model.encoder.vocab_size), 4096)]
    assert len(heads) == 2 and all(p.grad.abs().sum() > 0 for p in heads)
    before = [p.detach().clone() for p in heads]
    result = original_step(self, *args, **kwargs)
    assert all(not torch.equal(a, b) for a, b in zip(before, heads))
    torch.cuda.synchronize()
    metrics['updates'].append({'seconds': time.perf_counter()-start[0], 'residues': metrics['current_residues'], 'positions': metrics['current_positions'], 'shape': metrics['shape']})
    return result

def save_profile(*args, **kwargs):
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        return original_save_profile(*args, **kwargs)
    finally:
        torch.cuda.synchronize()
        metrics['checkpoints'].append({'step': kwargs['global_step'], 'seconds': time.perf_counter()-started})
        metrics['checkpoints'][-1].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        metrics['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        metrics['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        metrics['device'] = torch.cuda.get_device_name()
        metrics['torch'] = torch.__version__
        metrics['hip'] = torch.version.hip
        Path(str(sys.argv[1])+'.metrics.json').write_text(json.dumps(metrics, indent=2))

tasks.corrupt_mdlm_batch = corrupt_profile
torch.optim.AdamW.step = step
train._save_checkpoint = save_profile
train.restore_training_state = restore_profile
train.read_training_checkpoint = read_profile
"""
probe = PROBE.replace(
    "try:\n    train.run_training(cfg)", profile + "\ntry:\n    train.run_training(cfg)"
)
for name, stop in (("full", -1), ("interrupted", 1), ("resumed", -1)):
    cfg.train.output_dir = str(out / ("full" if name == "full" else "interrupted"))
    if name == "resumed":
        cfg.train.resume_from = str(out / "interrupted/checkpoints/step_00000001.pt")
    path = out / f"{name}.yaml"
    OmegaConf.save(cfg, path)
    with (out / f"{name}.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, "-c", probe, str(path), str(stop)],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
    assert (result.returncode == 0) == (stop == -1), (
        name,
        (out / f"{name}.log").read_text(),
    )
    if stop >= 0:
        assert "intentional interruption" in (out / f"{name}.log").read_text()
    for path in Path(cfg.train.output_dir).glob("checkpoints/step_*.pt"):
        if name == "full" or path.name != "step_00000001.pt":
            path.unlink()  # only our redundant diagnostics
full = torch.load(out / "full/model/final.pt", weights_only=True, map_location="cpu")
resumed = torch.load(
    out / "interrupted/model/final.pt", weights_only=True, map_location="cpu"
)
for key in (
    "global_step",
    "micro_step",
    "residues_seen",
    "executed_positions",
    "rank_states",
    "scheduler",
):
    equal(full[key], resumed[key])


def trace(name):
    return torch.load(out / f"{name}.yaml.rank0.trace.pt", weights_only=True)


equal(trace("full"), trace("interrupted") + trace("resumed"))
max_error = 0.0


def compare(left, right):
    global max_error
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        if left.is_floating_point():
            max_error = max(max_error, float((left - right).abs().max()))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for item in left:
            compare(left[item], right[item])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            compare(a, b)
    else:
        assert left == right


for key in ("model", "optimizer"):
    compare(full[key], resumed[key])
metrics = json.loads((out / "full.yaml.metrics.json").read_text())
resumed_metrics = json.loads((out / "resumed.yaml.metrics.json").read_text())
metrics["restore"] = resumed_metrics["restore"]
metrics["checkpoint_read_seconds"] = resumed_metrics["checkpoint_read_seconds"]
steady = metrics["updates"][1:]
metrics.update(
    steady_residues_per_second=sum(s["residues"] for s in steady)
    / sum(s["seconds"] for s in steady),
    steady_positions_per_second=sum(s["positions"] for s in steady)
    / sum(s["seconds"] for s in steady),
    continuation_max_absolute_error=max_error,
    continuation_trace_exact=True,
    continuation_rng_exact=True,
    full_architecture_updates_total=8,
    full_model_finite_gradients_and_both_head_updates=True,
    counters={
        key: full[key]
        for key in ("global_step", "micro_step", "residues_seen", "executed_positions")
    },
    precision=full["config"]["train"]["effective_precision"],
    signature=full["signature"],
    trainable_parameters=sum(
        v.numel() for k, v in full["model"].items() if k != "structure_codebook"
    ),
)
(out / "qualification.json").write_text(json.dumps(metrics, indent=2))
# Training-subset diagnostic using production denoising/sampling/geometry utilities.

tokenizer = Tokenizer()
canonical = torch.tensor(
    tokenizer.convert_tokens_to_ids(list(CANONICAL_AA)), device="cuda"
)
enc = full["config"]["model"]["encoder"]
model = STokMDLM(
    vocab_size=enc["vocab_size"],
    pad_id=enc["pad_id"],
    codebook=full["model"]["structure_codebook"],
    d_model=enc["d_model"],
    n_heads=enc["n_heads"],
    n_layers=enc["n_layers"],
    ffn_mult=enc["ffn_mult"],
    dropout=enc["dropout"],
    attn_dropout=enc["attn_dropout"],
    norm_type=enc["norm"],
).cuda()
model.load_state_dict(full["model"])
model.mdlm_regime_weights = full["config"]["train"]["mdlm"]["regime_weights"]
batch = trace("full")[-1]["batch"]
batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
settings = OmegaConf.create(full["config"]["train"]["mdlm"])
corruption = corrupt_mdlm_batch(
    batch,
    settings,
    seeds=[1729, 1730],
    regime="joint_independent",
    mask_probability=0.5,
)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
with inference_context(model), torch.autocast("cuda", dtype=torch.bfloat16):
    outputs = model(corruption["sequence_tokens"], corruption["structure_tokens"])
    terms = mdlm_loss_terms(outputs, batch, corruption, canonical_aa_ids=canonical)
    ce = terms["ce_sum"] / terms["masked_count"]
assert torch.isfinite(ce).all()
torch.cuda.synchronize()
metrics["denoising"] = {
    "seconds": time.perf_counter() - start,
    "masked_ce": ce.tolist(),
    "masked_count": terms["masked_count"].tolist(),
    "shape": list(batch["sequence_tokens"].shape),
    "population": "training_subset",
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
}
generate = torch.zeros(
    (*batch["residue_mask"].shape, 2), dtype=torch.bool, device="cuda"
)
generate[..., 1] = batch["residue_mask"]
groups = torch.stack(
    [
        build_mask_groups(
            g,
            r,
            placement="token",
            tied=False,
            span_mean=8,
            generator=torch.Generator().manual_seed(1729),
        )
        for g, r in zip(generate, batch["residue_mask"])
    ]
)
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
with torch.autocast("cuda", dtype=torch.bfloat16):
    sampled = sample_mdlm(
        model,
        batch,
        generate_mask=generate,
        group_ids=groups,
        schedule=settings.noise,
        steps=4,
        seeds=[1729, 1730],
        canonical_aa_ids=canonical,
    )
assert torch.equal(sampled["sequence_tokens"], batch["sequence_tokens"])
labels = sampled["structure_tokens"][batch["residue_mask"]]
assert ((labels >= 0) & (labels < model.codebook_size)).all()
torch.cuda.synchronize()
metrics["generation"] = {
    "seconds": time.perf_counter() - start,
    "steps": 4,
    "condition_preserved": True,
    "shape": list(batch["sequence_tokens"].shape),
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
}
decoder = load_pretrained_decoder(
    "large", path=os.environ["STOK_MDLM_ARCHIVE"], device="cuda", freeze=True
)
validate_mdlm_decoder(
    decoder,
    model.structure_codebook,
    full["config"]["train"]["mdlm_identity"]["codebook_sha256"],
)
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
with inference_context(decoder), torch.autocast("cuda", enabled=False):
    coords = decode_token_aligned_coords(
        decoder,
        model.structure_codebook[
            sampled["structure_tokens"].clamp(0, model.codebook_size - 1)
        ],
        batch["residue_mask"],
    )
assert torch.isfinite(coords[batch["residue_mask"]]).all()
torch.cuda.synchronize()
metrics["decode"] = {
    "seconds": time.perf_counter() - start,
    "shape": list(coords.shape),
    "finite_residues": int(batch["residue_mask"].sum()),
    "precision": "float32",
    "frozen": all(not p.requires_grad for p in decoder.parameters()),
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
}
metrics["evaluation_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
metrics["evaluation_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
(out / "qualification.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
