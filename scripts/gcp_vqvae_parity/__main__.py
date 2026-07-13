from __future__ import annotations

import argparse
from pathlib import Path

from .runner import QualificationConfig, run_qualification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qualify STōk encoder parity against gcp-vqvae")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=("base", "lite"), default="base")
    parser.add_argument("--hf-repo-id", default="Mahdip72/gcp-vqvae-large")
    parser.add_argument("--hf-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size != 1:
        raise SystemExit("The primary qualification requires --batch-size 1")
    config = QualificationConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        preset=args.preset,
        hf_repo_id=args.hf_repo_id,
        hf_revision=args.hf_revision,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        max_samples=args.max_samples,
        rtol=args.rtol,
        atol=args.atol,
        resume=not args.no_resume,
    )
    summary = run_qualification(config)
    print(f"qualification_status={summary.status.value}")
    return 0 if summary.core_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
