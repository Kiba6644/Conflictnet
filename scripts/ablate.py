#!/usr/bin/env python3
"""Ablation study: run ConflictNet training with different configs and compare.

Usage:
    python scripts/ablate.py --config_dir configs/ --iemocap_root /data/iemocap --output_dir ablation_results
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
    p = argparse.ArgumentParser(description="Run ablation studies for ConflictNet")
    p.add_argument("--config_dir", type=str, default="configs/")
    p.add_argument("--iemocap_root", type=str, required=True)
    p.add_argument("--mustard_root", type=str, default=None)
    p.add_argument("--cremad_root", type=str, default=None)
    p.add_argument("--meld_root", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="ablation_results")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def collect_metrics(results: dict) -> dict:
    """Extract summary metrics from a run's output."""
    return {
        "macro_f1": results.get("macro_f1", 0.0),
        "binary_f1": results.get("binary_f1", 0.0),
        "severity_mae": results.get("severity_mae", float("inf")),
    }


def print_table(results: list) -> None:
    """Print a comparison table of ablation results."""
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="Ablation Study Results")
        table.add_column("Config", style="cyan")
        table.add_column("macro_f1", justify="right")
        table.add_column("binary_f1", justify="right")
        table.add_column("severity_mae", justify="right")
        for r in results:
            table.add_row(
                r["config"],
                f"{r['macro_f1']:.4f}",
                f"{r['binary_f1']:.4f}",
                f"{r['severity_mae']:.4f}",
            )
        console.print(table)
    except ImportError:
        print("\n" + "=" * 60)
        print(f"{'Config':20s} {'macro_f1':>10s} {'binary_f1':>10s} {'severity_mae':>12s}")
        print("=" * 60)
        for r in results:
            print(f"{r['config']:20s} {r['macro_f1']:10.4f} {r['binary_f1']:10.4f} {r['severity_mae']:12.4f}")
        print("=" * 60 + "\n")


def main():
    args = parse_args()
    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_files = sorted(config_dir.glob("*.yaml"))
    if not config_files:
        logger.error(f"No YAML configs found in {config_dir}")
        sys.exit(1)

    logger.info(f"Found {len(config_files)} configs: {[c.name for c in config_files]}")

    script_path = Path(__file__).resolve().parent / "train.py"
    if not script_path.exists():
        logger.error(f"train.py not found at {script_path}")
        sys.exit(1)

    # Map YAML config keys to train.py CLI flags
    YAML_TO_CLI = {
        ("speaker_norm", "enabled"): (False, "--no_speaker_norm"),
        ("temporal", "enabled"): (False, "--no_temporal"),
        ("cross_attn", "enabled"): (False, "--no_cross_attn_injection"),
        ("speaker_adaptive_threshold", "enabled"): (False, "--no_speaker_adaptive_threshold"),
        ("baseline_subtract", "enabled"): (False, "--no_baseline_subtract"),
        ("training", "pretrain_epochs"): (0, "--pretrain_epochs", str),
        ("classifier", "word_div_dim"): (0, "--no_word_divergence"),
    }

    results = []
    for cfg in config_files:
        cfg_name = cfg.stem
        run_out = output_dir / cfg_name
        logger.info(f"[Ablate] Running config: {cfg_name}")

        # Parse the YAML config to extract overrides
        overrides = []
        _yaml = None
        try:
            import importlib
            _yaml = importlib.import_module('yaml')
        except ImportError:
            pass
        if _yaml is not None:
            try:
                with open(cfg) as fh:
                    cfg_data = _yaml.safe_load(fh)
                for yaml_keys, (trigger_val, *cli_spec) in YAML_TO_CLI.items():
                    val = cfg_data
                    try:
                        for key in yaml_keys:
                            val = val[key]
                    except (KeyError, TypeError):
                        continue
                    if val == trigger_val:
                        if len(cli_spec) == 1:
                            overrides.append(cli_spec[0])
                        else:
                            flag, fmt = cli_spec[0], cli_spec[1]
                            assert callable(fmt)
                            overrides.extend([flag, fmt(trigger_val)])
            except Exception:
                pass

        cmd = [
            sys.executable, str(script_path),
            *overrides,
            "--iemocap_root", args.iemocap_root,
            "--output_dir", str(run_out),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--device", args.device,
        ]
        if args.mustard_root:
            cmd.extend(["--mustard_root", args.mustard_root])
        if args.cremad_root:
            cmd.extend(["--cremad_root", args.cremad_root])
        if args.meld_root:
            cmd.extend(["--meld_root", args.meld_root])

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"[Ablate] Config {cfg_name} failed: {e.stderr[:200]}")
            results.append({"config": cfg_name, "macro_f1": 0.0, "binary_f1": 0.0, "severity_mae": float("inf")})
            continue

        # Load metrics from training output
        metrics_file = run_out / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                run_metrics = json.load(f)
        else:
            # Try evaluating the trained model
            ckpt = run_out / "best_model.safetensors"
            if ckpt.exists():
                eval_cmd = [
                    sys.executable, str(Path(__file__).resolve().parent / "evaluate.py"),
                    "--checkpoint", str(ckpt),
                    "--iemocap_root", args.iemocap_root,
                    "--output_dir", str(run_out / "eval"),
                    "--device", args.device,
                ]
                try:
                    subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
                    with open(run_out / "eval" / "metrics.json") as f:
                        run_metrics = json.load(f)
                except Exception:
                    run_metrics = {}
            else:
                run_metrics = {}

        row = {"config": cfg_name, **collect_metrics(run_metrics)}
        results.append(row)
        logger.info(f"  → {cfg_name}: macro_f1={row['macro_f1']:.4f}, binary_f1={row['binary_f1']:.4f}")

    # Output comparison table
    print_table(results)

    # Save results
    out_path = output_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"[Ablate] Results saved to {out_path}")


if __name__ == "__main__":
    main()
