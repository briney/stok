# Baseline extraction reference

`gcp_large_paired_mdlm_tied_v1` names the existing Hydra composition
`model=mdlm_150m train=mdlm_pilot`: 144,797,472 trainable parameters, learned
structure embeddings, and tied linear categorical predictions. Its frozen GCP
prototypes retain identity/decoding duties. No production defaults or config
selectors were added to name this baseline.

The migration reference is a separate tiny CPU software diagnostic: width 16,
one layer, two heads, FFN multiplier 1, dropout 0.2, context 8, batch 2,
accumulation 2, workers 2, seed 1729, no mixed precision, warmup/clipping zero,
two successful updates, checkpoint cadence 1, and log cadence 2. Existing mixed
availability fixtures exercise the accumulation denominator and a nonempty
step-1 logging window. Real-data diagnostics use context 66 and the matching
Large codebook while retaining the same bounded update recipe.

## Commands and artifacts

From the candidate checkout, select the qualified environment and local IPC:

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES=''
export PYTHONPATH="$PWD/src"
/tmp/stok-config-env/bin/python -m tests.utils.refactor_reference capture \
  --source-root /home/bryanbriney/git/stok/.worktrees/refactor-reference \
  --config /home/bryanbriney/git/stok/downloads/refactor-baseline-2026-10-06/synthetic-inputs-v1/config.yaml \
  --output /home/bryanbriney/git/stok/downloads/refactor-baseline-2026-10-06/synthetic-reference-v1
/tmp/stok-config-env/bin/python -m tests.utils.refactor_reference check \
  --reference /home/bryanbriney/git/stok/downloads/refactor-baseline-2026-10-06/synthetic-reference-v1 \
  --output /home/bryanbriney/git/stok/downloads/refactor-baseline-2026-10-06/synthetic-candidate-final
/tmp/stok-config-env/bin/python -m pytest \
  tests/integration/test_refactor_reference.py tests/integration/test_mdlm_resume.py -q
```

Every output destination must be empty or absent. Use a new directory for a new
comparison; never overwrite or regenerate the reference. Capture requires the
retained `7a3e0fc` source, except that the local plumbing test may capture its own
checkout and explicitly provides no cross-version evidence. The subprocess
checks its actual imported package path. Keep the retained source through final
qualification. Capture's frozen hooks and candidate hook owner are separate so
future task/engine relocation can update candidate instrumentation alone.

Capture runs two uninterrupted updates and one interrupted update. The manifest
retains source revision/file hashes, authored/resolved YAML, scientific baseline
composition, signatures, environment/package/device/thread versions, checksums,
the exact capture probe, both traces, full final checkpoint, and the interrupted
step-1 version-2 checkpoint. Hooks observe existing corruption/loss/optimizer
calls, saving detached loss terms and pre-step gradients without drawing RNG.
The eligible-token denominator is reconstructed from integer window counts and
normalized float64 modality weights.

Check verifies immutable artifact checksums and retained source identity, then
runs two fresh candidate updates and one continuation update from a copy of the
old step-1 checkpoint. Model, optimizer, scheduler, counters, rank states, full
traces and concatenated interrupted/continued traces must match exactly. Both
runs keep original data paths, budget and execution settings. Four-step seeded
folding, inverse-folding-like and joint CLI generation uses unique fixed input
IDs and records biological outputs. Comparison uses canonical JSON bytes for
all deterministic output fields; the checkpoint-file digest remains provenance.

The initial historical synthetic and real captures completed successfully in
`synthetic-reference-v1` and `real-reference-v1`, respectively. Real capture spent
three successful updates; the remaining three are reserved for final candidate
qualification. Frozen FP32 geometry decode completed for all three modes with
the matching full Large decoder; outputs and `decode-manifest.json` are retained
under `real-reference-v1`. Candidate decode equality remains a Task 6 gate.

## Gate inventory

| Gate | Inputs and limits |
|---|---|
| Synthetic software | Durable mixed-availability export/codebook and old-source reference under `downloads/refactor-baseline-2026-10-06`; required before extraction and at final comparison. |
| Real data | Completed `downloads/gcp-vqvae-public/export-large-fixed-padding-groups3`; 13 chains / 4,964 residues, matching codebook semantic digest `29d21936c951fc1421ce9bc0bcb4ebb3d6dc2391ede165d03db0be05cb958141`. Full Large archive available at `/home/bryanbriney/projects/stok/data/weights/gcp-vqvae-large/gcp-large-7d1d43950a29834e7f702409bf957e9ffb75eb3cb3952074ba4c545bb4130eaf.pth`; file digest matches its name. Root preflight verified strict loading and semantic codebook identity. |
| Real budget | Approved plan execution includes six tiny successful CPU updates total: capture 2+1, final check 2+1, plus fixed four-step sampling/decode. Use `real-config.yaml` and separate `real-reference-v1` / `real-candidate-final` destinations under the durable root. Optional between tasks; mandatory for final real-data acceptance. Preserve split status: training-subset diagnostics, never held-out scores. |
| Hardware | CPU evidence does not qualify GPU precision/topology, FP16 overflow or multi-GPU behavior. Existing device gates remain separate; no GPU job is launched by this harness. |

The qualification report's previous `/tmp` archive/checkpoint paths were absent.
Use the supplied persistent archive, not relaxed artifact validation or old
checkpoints lacking repaired RNG metadata. The qualified common environment is
`/tmp/stok-config-env/bin/python`, Python 3.12.14 and Torch 2.14.1+cu130 selected
for CPU; manifests record all dependency versions. Do not change dependencies
between capture and check. Existing upstream warnings remain visible.
