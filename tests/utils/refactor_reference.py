"""Immutable two-update reference; subprocesses import the selected source tree."""

import argparse
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from stok.utils.pretrained import file_sha256 as sha256
from tests.integration.test_mdlm_resume import PROBE, equal, execute
from tests.integration.test_training_progress import training_env

REVISION = "7a3e0fc"
KEYS = (
    "model",
    "optimizer",
    "scheduler",
    "rank_states",
    "global_step",
    "micro_step",
    "residues_seen",
    "executed_positions",
)
ROOT = Path(__file__).resolve().parents[2]

# Frozen old-source instrumentation: keep this owner valid at 7a3e0fc.
# Hooks observe existing calls only; CPU clones and metadata do not draw RNG.
INSTRUMENT = r"""
import importlib.metadata, json, platform
import stok
expected_source = Path(os.environ['REFERENCE_SOURCE']).resolve()
assert Path(stok.__file__).resolve().is_relative_to(expected_source / 'src'), stok.__file__
metadata = {
    'package_path': stok.__file__, 'python': platform.python_version(),
    'dependencies': {name: importlib.metadata.version(name) for name in
        ('torch', 'accelerate', 'hydra-core', 'omegaconf', 'numpy', 'pyarrow', 'click', 'x-transformers', 'vector-quantize-pytorch')},
    'device': 'cpu', 'torch_threads': torch.get_num_threads(),
    'torch_interop_threads': torch.get_num_interop_threads(),
    'environment': {key: os.environ.get(key) for key in
        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'TOKENIZERS_PARALLELISM', 'ACCELERATE_USE_CPU', 'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')},
}
Path(str(sys.argv[1]) + '.environment.json').write_text(json.dumps(metadata, indent=2))
original_terms = train.mdlm_loss_terms
original_step = torch.optim.AdamW.step
weights = torch.tensor([cfg.train.mdlm.get('sequence_loss_weight', 1), cfg.train.mdlm.get('structure_loss_weight', 1)], dtype=torch.float64)
weights /= weights.max()
def loss_terms(*args, **kwargs):
    result = original_terms(*args, **kwargs)
    trace.append({'event': 'loss', 'terms': {key: value.detach().cpu().clone() for key, value in result.items()}})
    return result
def optimizer_step(self, *args, **kwargs):
    window = trace[next((i + 1 for i in range(len(trace)-1, -1, -1) if isinstance(trace[i], dict) and trace[i].get('event') == 'optimizer'), 0):]
    counts = sum((row['corruption']['eligible'].sum((0, 1)) for row in window if isinstance(row, dict) and 'corruption' in row), torch.zeros(2, dtype=torch.long))
    denominator = float((counts * weights).sum())
    trace.append({'event': 'optimizer', 'eligible_counts': counts, 'weights': weights.clone(), 'denominator': denominator,
        'gradients': [None if p.grad is None else p.grad.detach().cpu().clone() for group in self.param_groups for p in group['params']]})
    return original_step(self, *args, **kwargs)
train.mdlm_loss_terms = loss_terms
torch.optim.AdamW.step = optimizer_step
"""


def instrument(probe):
    marker = "try:\n    train.run_training(cfg)"
    assert marker in probe, "Resume probe entry point changed"
    return probe.replace(marker, INSTRUMENT + marker)


def candidate_probe():
    """Adapt candidate hook owners here when tasks/engine move; keep capture frozen."""
    return instrument(PROBE).replace("train.mdlm_loss_terms", "tasks.mdlm_loss_terms")


def identity(source):
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    files = subprocess.check_output(
        ["git", "-C", str(source), "ls-files", "src"], text=True
    ).splitlines()
    return {
        "root": str(source),
        "revision": revision,
        "files": {name: sha256(source / name) for name in files},
    }


def destination(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Refusing populated output destination: {path}")


def validate_config(cfg):
    if cfg.train.max_steps != 2 or cfg.train.save_every != 1:
        raise ValueError(
            "Reference requires exactly two updates and checkpoint cadence one"
        )
    if cfg.train.get("resume_from") or cfg.train.wandb.enabled:
        raise ValueError("Reference requires fresh training and disabled W&B")
    evaluation = cfg.train.get("eval", {})
    mdlm_evaluation = evaluation.get("mdlm", {})
    if (
        cfg.train.get("benchmark", False)
        or cfg.train.get("benchmark_mode", False)
        or mdlm_evaluation.get("enabled", False)
        or mdlm_evaluation.get("generation", {}).get("enabled", False)
    ):
        raise ValueError("Reference requires disabled benchmark controls")
    if cfg.train.objective != "mdlm" or cfg.train.mixed_precision != "no":
        raise ValueError("Reference requires MDLM with no mixed precision on CPU")


def run(cfg, directory, name, source, probe, *, stop=-1):
    env = {
        **training_env(),
        "PYTHONPATH": str(source / "src"),
        "REFERENCE_SOURCE": str(source),
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    results = execute(
        cfg,
        directory / f"{name}.yaml",
        stop=stop,
        extra_env=env,
        ok=stop < 0,
        probe=probe,
    )
    (directory / f"{name}.stdout.txt").write_text(results[0].stdout)
    (directory / f"{name}.stderr.txt").write_text(results[0].stderr)


def checkpoint(directory, name):
    return torch.load(
        directory / name / "model/final.pt", weights_only=True, map_location="cpu"
    )


def trace(directory, name):
    result = torch.load(
        directory / f"{name}.yaml.rank0.trace.pt", weights_only=True, map_location="cpu"
    )
    for row in result:
        if isinstance(row, dict) and row.get("event") == "optimizer":
            assert row["denominator"] == float(
                (row["eligible_counts"] * row["weights"]).sum()
            )
    return result


def compare(left, right):
    assert left["format_version"] == right["format_version"] == 2
    assert left["global_step"] == right["global_step"] == 2
    for key in KEYS:
        equal(left[key], right[key])


def sample(directory, checkpoint_path, source, *, inputs=None):
    payload = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    identity = payload["config"]["train"]["mdlm_identity"]
    if inputs is None:
        inputs = {
            "folding": {"sequence_id": "reference-folding", "sequence": "LAGV"},
            "inverse_folding": {
                "sequence_id": "reference-inverse",
                "structure_tokens": [0, 1, 2, 3],
                "tokenizer_sha256": identity["tokenizer_sha256"],
                "codebook_sha256": identity["codebook_sha256"],
            },
            "joint": {"sequence_id": "reference-joint", "length": 4},
        }
    outputs = {}
    env = {
        **training_env(),
        "PYTHONPATH": str(source / "src"),
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for mode, row in inputs.items():
        manifest = directory / f"sample-{mode}.input.jsonl"
        manifest.write_text(json.dumps(row) + "\n")
        output = directory / f"sample-{mode}.jsonl"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from stok.cli.sample import sample_cmd; sample_cmd()",
                "--checkpoint",
                str(checkpoint_path),
                "--input",
                str(manifest),
                "--output",
                str(output),
                "--mode",
                mode,
                "--steps",
                "4",
                "--seed",
                "1729",
                "--device",
                "cpu",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        content = json.loads(output.read_text())
        # Serialization of an equivalent checkpoint can differ; its digest is provenance.
        del content["provenance"]["checkpoint_sha256"]
        outputs[mode] = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return inputs, outputs


def capture_reference(cfg: DictConfig, directory: Path, *, source_root: Path) -> None:
    directory, source_root = directory.resolve(), source_root.resolve()
    destination(directory)
    validate_config(cfg)
    source_identity = identity(source_root)
    # Current-source captures exist only for the harness plumbing test.
    assert source_identity["revision"].startswith(REVISION) or source_root == ROOT, (
        "Migration reference must use 7a3e0fc"
    )
    if source_identity["revision"].startswith(REVISION):
        assert not subprocess.check_output(
            ["git", "-C", str(source_root), "diff", "HEAD", "--", "src"], text=True
        ), "Historical source has uncommitted product edits"
    tree = ast.parse(
        (source_root / "tests/integration/test_mdlm_resume.py").read_text()
    )
    old_probe = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PROBE"
            for target in node.targets
        )
    )
    probe = candidate_probe() if source_root == ROOT else instrument(old_probe)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "capture-probe.py").write_text(probe)
    OmegaConf.save(cfg, directory / "authored.yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(cfg, directory / "resolved.yaml")
    with initialize_config_dir(
        config_dir=str(source_root / "src/stok/configs"), version_base=None
    ):
        baseline = compose(
            config_name="config", overrides=["model=mdlm_150m", "train=mdlm_pilot"]
        )
    OmegaConf.save(baseline, directory / "scientific-baseline.yaml", resolve=True)
    cfg.train.output_dir = str(directory / "full")
    run(cfg, directory, "full", source_root, probe)
    cfg.train.output_dir = str(directory / "interrupted")
    run(cfg, directory, "interrupted", source_root, probe, stop=1)
    final = checkpoint(directory, "full")
    assert final["format_version"] == 2 and final["global_step"] == 2
    retained = torch.load(
        directory / "interrupted/checkpoints/step_00000001.pt", weights_only=True
    )
    assert retained["format_version"] == 2 and retained["global_step"] == 1
    equal(
        trace(directory, "full")[: len(trace(directory, "interrupted"))],
        trace(directory, "interrupted"),
    )
    inputs, outputs = sample(directory, directory / "full/model/final.pt", source_root)
    manifest = {
        "format_version": 1,
        "baseline": {
            "name": "gcp_large_paired_mdlm_tied_v1",
            "composition": ["model=mdlm_150m", "train=mdlm_pilot"],
            "parameter_count": 144797472,
        },
        "software_reference": "tiny_cpu",
        "historical_reference": source_identity["revision"].startswith(REVISION),
        "source": source_identity,
        "signature": final["signature"],
        "environment": json.loads(
            (directory / "full.yaml.environment.json").read_text()
        ),
        "sampling": {"inputs": inputs, "outputs": outputs},
        "checksums": {
            str(path.relative_to(directory)): sha256(path)
            for path in directory.rglob("*")
            if path.is_file()
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )


def check_reference(directory: Path, output: Path) -> None:
    directory, output = directory.resolve(), output.resolve()
    destination(output)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["checksums"].items():
        assert sha256(directory / name) == digest, (
            f"Reference checksum mismatch: {name}"
        )
    equal(identity(Path(manifest["source"]["root"])), manifest["source"])
    cfg = OmegaConf.load(directory / "resolved.yaml")
    validate_config(cfg)
    output.mkdir(parents=True, exist_ok=True)
    probe = candidate_probe()
    (output / "candidate-probe.py").write_text(probe)
    cfg.train.output_dir = str(output / "full")
    run(cfg, output, "full", ROOT, probe)
    candidate_environment = json.loads(
        (output / "full.yaml.environment.json").read_text()
    )
    equal(
        {
            key: value
            for key, value in manifest["environment"].items()
            if key != "package_path"
        },
        {
            key: value
            for key, value in candidate_environment.items()
            if key != "package_path"
        },
    )
    compare(checkpoint(directory, "full"), checkpoint(output, "full"))
    equal(trace(directory, "full"), trace(output, "full"))
    retained = output / "step_00000001.pt"
    shutil.copyfile(directory / "interrupted/checkpoints/step_00000001.pt", retained)
    cfg.train.output_dir = str(output / "continued")
    cfg.train.resume_from = str(retained)
    run(cfg, output, "continued", ROOT, probe)
    compare(checkpoint(directory, "full"), checkpoint(output, "continued"))
    equal(
        trace(directory, "full"),
        trace(directory, "interrupted") + trace(output, "continued"),
    )
    _, outputs = sample(
        output,
        output / "full/model/final.pt",
        ROOT,
        inputs=manifest["sampling"]["inputs"],
    )
    equal(manifest["sampling"]["outputs"], outputs)
    (output / "result.json").write_text(
        json.dumps(
            {"matched": True, "reference": str(directory), "candidate": identity(ROOT)},
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--source-root", type=Path, required=True)
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("--reference", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "capture":
        capture_reference(
            OmegaConf.load(args.config), args.output, source_root=args.source_root
        )
    else:
        check_reference(args.reference, args.output)


if __name__ == "__main__":
    main()
