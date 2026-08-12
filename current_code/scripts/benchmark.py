#!/usr/bin/env python3
"""Benchmark pipeline: evaluate a trained ConflictNet across all datasets.

Runs the model on each evaluation dataset separately, computes per-dataset
and aggregate metrics, optionally generates calibration plots and fairness
audits, and saves a comprehensive JSON report.

Usage:
    # Evaluate a trained checkpoint on all datasets
    python scripts/benchmark.py \\
        --checkpoint checkpoints/best_model.pt \\
        --iemocap_root /data/iemocap \\
        --mustard_root /data/mustard \\
        --case_root /data/case2026 \\
        --meld_root /data/meld \\
        --output_dir benchmark_results

    # Also run all 7 ablations (requires separate checkpoints)
    python scripts/benchmark.py --checkpoint checkpoints/best_model.pt \\
        --ablation_dir checkpoints/ablations/ \\
        ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TYPE_NAMES = ["sarcasm", "suppression", "deception"]

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Tuple] = {}


def _register_datasets():
    """Lazy import datasets to avoid torchaudio dependency at module level."""
    from data.datasets import IEMOCAPDataset, MUStARDDataset, MELDDataset, CMUMOSEIDataset, CASEDataset

    DATASET_REGISTRY.update({
        "iemocap": (IEMOCAPDataset, {"sessions": [5]}),
        "mustard++": (MUStARDDataset, {"split": "val"}),
        "meld": (MELDDataset, {"split": "val"}),
        "cmu-mosei": (CMUMOSEIDataset, {"split": "val"}),
        "case2026": (CASEDataset, {"split": "val"}),
    })


# ---------------------------------------------------------------------------
# Inference on a single dataset
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_on_dataset(
    model: torch.nn.Module,
    dataset_name: str,
    dataset_root: str,
    batch_size: int,
    device: str,
    prosody_lookup: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run inference on a single evaluation dataset.

    Returns dict with probs, labels, severity, genders, speaker_ids,
    and computed metrics.
    """
    if not DATASET_REGISTRY:
        _register_datasets()

    cls, kwargs = DATASET_REGISTRY.get(dataset_name, (None, {}))
    if cls is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_REGISTRY.keys())}")

    from data.datasets import make_collate_fn

    ds = cls(root=dataset_root, **kwargs)
    if len(ds) == 0:
        logger.warning(f"[Bench] {dataset_name}: empty dataset, skipping")
        return {}

    collate = make_collate_fn(prosody_lookup=prosody_lookup)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device != "cpu"), collate_fn=collate,
    )

    all_probs, all_labels = [], []
    all_sev_pred, all_sev_true = [], []
    all_genders, all_speakers = [], []

    for batch in loader:
        batch_gpu = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        out = model(
            audio=batch_gpu["audio"],
            input_ids=batch_gpu["input_ids"],
            attention_mask=batch_gpu["attention_mask"],
            audio_attention_mask=batch_gpu.get("audio_attention_mask"),
            prosody_z=batch_gpu.get("prosody_z"),
        )
        all_probs.append(out.probs_type.cpu().numpy())
        all_labels.append(batch["conflict_type_labels"].numpy())
        if out.severity is not None:
            all_sev_pred.append(out.severity.squeeze(-1).cpu().numpy())
        if "severity" in batch:
            all_sev_true.append(batch["severity"].squeeze(-1).numpy())
        all_genders.extend(batch.get("genders", [None] * batch["audio"].size(0)))
        all_speakers.extend(batch.get("speaker_ids", [None] * batch["audio"].size(0)))

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    sev_pred = np.concatenate(all_sev_pred) if all_sev_pred else None
    sev_true = np.concatenate(all_sev_true) if all_sev_true else None

    from evaluation.metrics import compute_all_metrics

    metrics = compute_all_metrics(
        probs, labels,
        severity_pred=sev_pred, severity_true=sev_true,
        type_names=TYPE_NAMES,
    )

    return {
        "dataset": dataset_name,
        "n_samples": len(ds),
        "probs": probs.tolist(),
        "labels": labels.tolist(),
        "severity_pred": sev_pred.tolist() if sev_pred is not None else None,
        "severity_true": sev_true.tolist() if sev_true is not None else None,
        "genders": all_genders,
        "speaker_ids": all_speakers,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Build model from checkpoint
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load a trained ConflictNet from checkpoint.

    Reads architecture config from the sidecar ``_meta.json`` file
    (not from the safetensors weight dict, which contains no config keys).
    """
    import json as _json
    from pathlib import Path as _Path
    from models.conflictnet import ConflictNet
    from models.checkpoint_utils import load_checkpoint_state, extract_model_state

    ckpt = load_checkpoint_state(checkpoint_path, device=device)
    model_state = extract_model_state(ckpt)

    # Read architecture config from sidecar _meta.json (B1 fix)
    audio_encoder = "emotion2vec"
    embed_dim = 256
    ckpt_path = _Path(checkpoint_path)
    meta_path = ckpt_path.parent / f"{ckpt_path.stem}_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = _json.load(f)
        exp_cfg = meta.get("experiment_config", {})
        audio_encoder = exp_cfg.get("audio_encoder", audio_encoder)
        embed_dim = exp_cfg.get("embed_dim", embed_dim)
        logger.info(f"[Bench] Config from meta.json: audio_encoder={audio_encoder}, embed_dim={embed_dim}")
    else:
        logger.warning(f"[Bench] No _meta.json found at {meta_path}, using defaults (audio_encoder={audio_encoder})")

    model = ConflictNet(
        audio_encoder_name=audio_encoder,
        embed_dim=embed_dim,
    )
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    logger.info(f"[Bench] Loaded checkpoint from {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def aggregate_metrics(dataset_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across datasets (mean, std, min, max)."""
    keys = [k for k in dataset_results[0] if k.startswith(("macro_", "binary_", "f1_", "ap_", "auc_", "wacc", "severity_"))]
    agg = {}
    for key in keys:
        vals = [r[key] for r in dataset_results if key in r and isinstance(r[key], (int, float))]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
            agg[f"{key}_min"] = float(np.min(vals))
            agg[f"{key}_max"] = float(np.max(vals))
    return agg


def print_benchmark_table(results: List[Dict[str, Any]], title: str = "Benchmark Results"):
    """Print a comparison table of per-dataset metrics."""
    from evaluation.metrics import print_metrics

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    for r in results:
        ds_name = r.get("dataset", "unknown")
        n = r.get("n_samples", 0)
        print(f"\n  [{ds_name}] ({n} samples)")
        row = {k: v for k, v in r.items() if not isinstance(v, list)}
        print_metrics(row, prefix="")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="ConflictNet benchmark pipeline")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    p.add_argument("--iemocap_root", type=str, default=None)
    p.add_argument("--mustard_root", type=str, default=None)
    p.add_argument("--meld_root", type=str, default=None)
    p.add_argument("--mosei_root", type=str, default=None)
    p.add_argument("--case_root", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="benchmark_results")
    p.add_argument("--prosody_stats", type=str, default=None)
    p.add_argument("--calibrate", action="store_true", help="Run multi-source calibration")
    p.add_argument("--fairness", action="store_true", help="Run fairness audit")
    p.add_argument("--latency", action="store_true", help="Run latency benchmark")
    p.add_argument("--plot", action="store_true", help="Generate reliability diagrams")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Register available datasets
    _register_datasets()

    # Map CLI args to dataset roots
    dataset_roots: Dict[str, str] = {}
    for cli_key, ds_name in [("iemocap_root", "iemocap"), ("mustard_root", "mustard++"),
                               ("meld_root", "meld"), ("mosei_root", "cmu-mosei"),
                               ("case_root", "case2026")]:
        root = getattr(args, cli_key, None)
        if root:
            dataset_roots[ds_name] = root

    if not dataset_roots:
        raise ValueError("Provide at least one dataset root")

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Load prosody z-scores if available
    prosody_lookup = None
    if args.prosody_stats:
        prosody_path = Path(args.prosody_stats)
        zscores_p = prosody_path.parent / f"{prosody_path.stem}.zscores.json"
        if not zscores_p.exists():
            zscores_p = prosody_path.with_suffix(".zscores.json")
        if zscores_p.exists():
            with open(zscores_p) as f:
                raw = json.load(f)
            prosody_lookup = {k: torch.tensor(v) for k, v in raw.items()}
            logger.info(f"[Bench] loaded {len(prosody_lookup)} z-score entries")

    # Run per-dataset evaluation
    dataset_results = []
    for ds_name in sorted(dataset_roots.keys()):
        root = dataset_roots[ds_name]
        logger.info(f"[Bench] Evaluating on {ds_name}...")
        t0 = time.time()
        result = evaluate_on_dataset(model, ds_name, root, args.batch_size, args.device, prosody_lookup)
        elapsed = time.time() - t0
        if result:
            result["eval_time_s"] = round(elapsed, 2)
            logger.info(f"[Bench] {ds_name}: {result['n_samples']} samples, "
                        f"macro_f1={result.get('macro_f1', 'N/A'):.4f}, time={elapsed:.1f}s")
            dataset_results.append(result)

    if not dataset_results:
        logger.error("[Bench] No datasets produced results")
        return

    # Aggregate across datasets
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": args.checkpoint,
        "datasets_evaluated": len(dataset_results),
        "per_dataset": {},
        "aggregate": aggregate_metrics(dataset_results),
    }

    for r in dataset_results:
        ds_name = r.pop("dataset")
        # Strip raw prediction arrays for the summary report
        report_entry = {k: v for k, v in r.items() if not isinstance(v, list)}
        summary["per_dataset"][ds_name] = report_entry

    # Print table
    print_benchmark_table(dataset_results)

    # ── Calibration ────────────────────────────────────────────────────
    if args.calibrate and len(dataset_results) >= 2:
        logger.info("[Bench] Running multi-source calibration...")
        from evaluation.calibration import calibrate_multi_source, plot_reliability_diagram

        source_probs = {r["dataset"]: np.array(r["probs"]) for r in dataset_results}
        source_labels = {r["dataset"]: np.array(r["labels"]) for r in dataset_results}

        for strategy in ["mean", "median", "pooled", "per_source"]:
            calib_result = calibrate_multi_source(
                source_probs, source_labels, strategy=strategy, metric="macro_f1"
            )
            summary.setdefault("calibration", {})[strategy] = {
                "threshold": calib_result.get("threshold", None),
                "best_macro_f1": calib_result.get("best_macro_f1", None),
                "per_source": calib_result.get("per_source", None),
            }
            logger.info(f"[Bench] Calibration ({strategy}): "
                        f"threshold={calib_result.get('threshold', 'N/A')}, "
                        f"best_f1={calib_result.get('best_macro_f1', 'N/A'):.4f}")

        # Reliability diagram
        if args.plot:
            try:
                import matplotlib
                matplotlib.use("Agg")
                for ds_name, r in zip(dataset_roots.keys(), dataset_results):
                    probs = np.array(r["probs"])
                    labels = np.array(r["labels"])
                    plot_path = str(out_dir / f"reliability_{ds_name.replace('+', 'p')}.png")
                    plot_reliability_diagram(
                        probs, labels, save_path=plot_path,
                        type_names=TYPE_NAMES,
                    )
                logger.info(f"[Bench] Reliability diagrams saved to {out_dir}/")
            except Exception as e:
                logger.warning(f"[Bench] Reliability diagram failed: {e}")

    # ── Fairness ───────────────────────────────────────────────────────
    if args.fairness:
        from evaluation.fairness import fairness_audit

        for r in dataset_results:
            ds_name = r["dataset"]
            genders = r.get("genders", [])
            if not genders or all(g is None for g in genders):
                continue
            probs = np.array(r["probs"])
            labels = np.array(r["labels"])
            binary_pred = (probs >= 0.5).any(axis=1).astype(int)
            binary_true = labels.any(axis=1).astype(int)
            valid_genders = [g if g in ("M", "F") else "unknown" for g in genders]
            audit = fairness_audit(binary_pred, binary_true, valid_genders)
            summary.setdefault("fairness", {})[ds_name] = audit
            logger.info(f"[Bench] Fairness ({ds_name}): disparity={audit['disparity']:.4f}")

    # ── Latency ────────────────────────────────────────────────────────
    if args.latency:
        from evaluation.latency import benchmark_latency

        # Create a dummy batch matching model input shape
        dummy = {
            "audio": torch.randn(args.batch_size, 16000),
            "input_ids": torch.randint(0, 100, (args.batch_size, 128)),
            "attention_mask": torch.ones(args.batch_size, 128, dtype=torch.long),
        }
        latency = benchmark_latency(model, dummy, device=args.device)
        summary["latency"] = latency
        logger.info(f"[Bench] Latency: avg={latency['avg_ms']}ms, throughput={latency['throughput']}/s")

    # ── Save ───────────────────────────────────────────────────────────
    report_path = out_dir / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Also save a human-readable summary
    summary_lines = [
        "=" * 60,
        "  ConflictNet Benchmark Report",
        f"  Checkpoint: {args.checkpoint}",
        f"  Datasets: {', '.join(sorted(summary['per_dataset'].keys()))}",
        "=" * 60,
    ]
    agg = summary.get("aggregate", {})
    for key in sorted(agg.keys()):
        summary_lines.append(f"  {key:25s}: {agg[key]:.4f}")
    summary_lines.append("=" * 60)

    report_txt = out_dir / "benchmark_summary.txt"
    with open(report_txt, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))

    # Print per-dataset macro F1
    print("\nPer-dataset macro-F1:")
    for ds_name, entry in sorted(summary["per_dataset"].items()):
        print(f"  {ds_name:15s}: {entry.get('macro_f1', 'N/A'):.4f}")

    logger.info(f"[Bench] Report saved to {report_path}")


if __name__ == "__main__":
    main()
