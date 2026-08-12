#!/usr/bin/env python3
"""Orchestrator: run all preprocessing stages for ConflictNet.

Pipeline:
  1. compute_prosody_stats.py
  2. run_mfa_alignment.py (optional, can skip)
  3. compute_difficulties.py (optional, can skip)

Saves a manifest of all generated files.

Usage:
    python scripts/preprocess_all.py \
        --dataset_root /data/iemocap \
        --output_dir preprocessed
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Run all ConflictNet preprocessing stages")
    p.add_argument("--iemocap_root", type=str, required=True)
    p.add_argument("--mustard_root", type=str, default=None)
    p.add_argument("--cremad_root", type=str, default=None)
    p.add_argument("--meld_root", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="preprocessed")
    p.add_argument("--skip_mfa", action="store_true", help="Skip MFA alignment")
    p.add_argument("--skip_difficulties", action="store_true", help="Skip difficulty computation")
    p.add_argument("--skip_prosody", action="store_true", help="Skip prosody statistics")
    p.add_argument("--checkpoint", type=str, default=None, help="Checkpoint for difficulty computation")
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def _script_path(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def run_stage(name: str, cmd: list) -> None:
    logger.info(f"[Preprocess] Stage: {name}")
    logger.info(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"[Preprocess] Stage '{name}' failed:\n{result.stderr[:500]}")
        raise RuntimeError(f"Preprocessing stage '{name}' failed")
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            logger.info(f"  {line}")
    logger.info(f"[Preprocess] Stage '{name}' completed")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    generated_files = []

    # Stage 1: Prosody stats
    if not args.skip_prosody:
        prosody_out = out_dir / "prosody_stats.json"
        cmd = [
            sys.executable, str(script_dir / "compute_prosody_stats.py"),
            "--output_file", str(prosody_out),
        ]
        if args.iemocap_root:
            cmd += ["--iemocap_root", args.iemocap_root]
        if args.mustard_root:
            cmd += ["--mustard_root", args.mustard_root]
        if args.cremad_root:
            cmd += ["--cremad_root", args.cremad_root]
        if args.meld_root:
            cmd += ["--meld_root", args.meld_root]
        run_stage("prosody_stats", cmd)
        generated_files.append(str(prosody_out))
    else:
        logger.info("[Preprocess] Skipping prosody stats")

    # Stage 2: MFA alignment
    if not args.skip_mfa:
        mfa_out = out_dir / "mfa_corpus"
        cmd = [
            sys.executable, str(script_dir / "run_mfa_alignment.py"),
            "--dataset_root", args.iemocap_root,
            "--output_dir", str(mfa_out),
        ]
        run_stage("mfa_alignment", cmd)
        manifest = mfa_out / "alignment_manifest.json"
        if manifest.exists():
            generated_files.append(str(manifest))
    else:
        logger.info("[Preprocess] Skipping MFA alignment")

    # Stage 3: Difficulties
    if not args.skip_difficulties:
        diff_out = out_dir / "difficulties.json"
        cmd = [
            sys.executable, str(script_dir / "compute_difficulties.py"),
            "--output_file", str(diff_out),
            "--batch_size", str(args.batch_size),
        ]
        if args.iemocap_root:
            cmd += ["--iemocap_root", args.iemocap_root]
        if args.checkpoint:
            cmd += ["--checkpoint", args.checkpoint]
        if args.mustard_root:
            cmd += ["--mustard_root", args.mustard_root]
        if args.cremad_root:
            cmd += ["--cremad_root", args.cremad_root]
        if args.meld_root:
            cmd += ["--meld_root", args.meld_root]
        run_stage("difficulties", cmd)
        generated_files.append(str(diff_out))
    else:
        logger.info("[Preprocess] Skipping difficulties")

    # Save manifest
    manifest = {
        "output_dir": str(out_dir),
        "generated_files": generated_files,
        "stages": {
            "prosody_stats": not args.skip_prosody,
            "mfa_alignment": not args.skip_mfa,
            "difficulties": not args.skip_difficulties,
        },
    }
    manifest_path = out_dir / "preprocess_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"[Preprocess] All stages complete. Manifest: {manifest_path}")
    logger.info(f"[Preprocess] Generated {len(generated_files)} files: {generated_files}")


if __name__ == "__main__":
    main()
