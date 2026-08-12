"""Multi-source calibration: find the optimal divergence threshold across datasets.

ConflictNet's ``conflict_flag`` depends on a sigmoid threshold
(``type_threshold`` in ``ConflictClassifier``). The optimal threshold
varies across domains (MUStARD++ sarcasm vs. CMU-MOSEI suppression vs.
CASE mixed types).  This module calibrates a single robust threshold
(optionally per-type) by pooling validation data from all available
sources.

Flow:
    1. Run inference on each source's validation split.
    2. For each source, sweep thresholds in [0.05, 0.95] and record
       macro-F1.
    3. Aggregate: produce a single threshold that maximises mean macro-F1
       (or median, or per-type weighted average).
    4. Optionally: produce per-type thresholds for the classifier's
       ``type_threshold`` (which is a scalar applied uniformly) or for the
       downstream evaluation.

Usage::

    python evaluation/calibration.py --checkpoint checkpoints/best_model.pt \\
        --mustard_root /data/mustard \\
        --case_root /data/case2026 \\
        --output_dir calibration_results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader



sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def sweep_threshold(
    probs_type: np.ndarray,
    labels_type: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, np.ndarray]:
    """Sweep binary conflict thresholds and return metrics per threshold.

    Args:
        probs_type: (N, n_types) sigmoid probabilities.
        labels_type: (N, n_types) multi-hot ground truth.
        thresholds: List of thresholds to sweep (default linspace 0.05–0.95).

    Returns:
        Dict with:
            thresholds: (T,) threshold values.
            macro_f1: (T,) macro-F1 at each threshold.
            binary_f1: (T,) any-vs-none binary F1 at each threshold.
    """
    thresholds_list = np.linspace(0.05, 0.95, 91).tolist() if thresholds is None else thresholds

    macro_f1_list = []
    binary_f1_list = []

    from sklearn.metrics import f1_score  # type: ignore  # pyre-ignore

    for th in thresholds_list:
        binary_pred = (probs_type >= th).any(axis=1).astype(int)
        binary_true = labels_type.any(axis=1).astype(int)

        # Per-type F1 then macro average
        type_f1s = []
        for t in range(probs_type.shape[1]):
            tp = (probs_type[:, t] >= th).astype(int)
            if tp.sum() == 0 and labels_type[:, t].sum() == 0:
                type_f1s.append(1.0)
            elif tp.sum() == 0:
                type_f1s.append(0.0)
            else:
                type_f1s.append(f1_score(labels_type[:, t], tp, zero_division=0))
        macro_f1_list.append(float(np.mean(type_f1s)))
        binary_f1_list.append(float(f1_score(binary_true, binary_pred, zero_division=0)))

    return {
        "thresholds": np.array(thresholds_list),
        "macro_f1": np.array(macro_f1_list),
        "binary_f1": np.array(binary_f1_list),
    }


def find_best_threshold(
    probs_type: np.ndarray,
    labels_type: np.ndarray,
    metric: str = "macro_f1",
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """Find the threshold that maximises a given metric.

    Returns:
        (best_threshold, best_metric_value).
    """
    sweep = sweep_threshold(probs_type, labels_type, thresholds)
    values = sweep[metric]
    best_idx = int(np.argmax(values))
    return float(sweep["thresholds"][best_idx]), float(values[best_idx])


def calibrate_multi_source(
    source_probs: Dict[str, np.ndarray],
    source_labels: Dict[str, np.ndarray],
    strategy: str = "mean",
    metric: str = "macro_f1",
    thresholds: Optional[List[float]] = None,
) -> Dict:
    """Calibrate threshold across multiple data sources.

    Args:
        source_probs: ``{source_name: (N_i, n_types) probs}``.
        source_labels: ``{source_name: (N_i, n_types) labels}``.
        strategy: One of:
            - ``"mean"``: maximise mean macro-F1 across sources.
            - ``"median"``: maximise median macro-F1 across sources.
            - ``"pooled"``: pool all sources then maximise.
            - ``"per_source"``: return per-source optimal thresholds.
        metric: Metric to optimise (``"macro_f1"`` or ``"binary_f1"``).
        thresholds: Candidate thresholds (default linspace 0.05–0.95).

    Returns:
        Dict with calibration results.
    """
    source_names = sorted(source_probs.keys())

    if strategy == "per_source":
        per_source = {}
        for name in source_names:
            best_th, best_val = find_best_threshold(
                source_probs[name], source_labels[name], metric, thresholds
            )
            per_source[name] = {"threshold": best_th, f"best_{metric}": best_val}
        return {"strategy": "per_source", "per_source": per_source}

    if strategy == "pooled":
        all_probs = np.concatenate([source_probs[n] for n in source_names], axis=0)
        all_labels = np.concatenate([source_labels[n] for n in source_names], axis=0)
        best_th, best_val = find_best_threshold(all_probs, all_labels, metric, thresholds)
        return {
            "strategy": "pooled",
            "threshold": best_th,
            f"best_{metric}": best_val,
            "n_total": all_probs.shape[0],
        }

    # Mean / median: sweep together
    thresholds_list = np.linspace(0.05, 0.95, 91).tolist() if thresholds is None else thresholds

    agg_values = []
    for th in thresholds_list:
        source_vals = []
        for name in source_names:
            _, val = find_best_threshold(source_probs[name], source_labels[name], metric, [th])
            source_vals.append(val)
        if strategy == "median":
            agg_values.append(float(np.median(source_vals)))
        else:
            agg_values.append(float(np.mean(source_vals)))

    agg_values = np.array(agg_values)
    best_idx = int(np.argmax(agg_values))
    best_th = thresholds_list[best_idx]

    # Per-source F1 at calibrated threshold
    per_source = {}
    for name in source_names:
        _, val = find_best_threshold(source_probs[name], source_labels[name], metric, [best_th])
        per_source[name] = {f"{metric}_at_calibrated": val}

    return {
        "strategy": strategy,
        "threshold": best_th,
        f"best_{metric}": float(agg_values[best_idx]),
        "per_source": per_source,
        "all_thresholds": thresholds_list,
        "agg_curve": agg_values.tolist(),
    }


# ---------------------------------------------------------------------------
# Reliability diagram (calibration curve visualisation)
# ---------------------------------------------------------------------------


def plot_reliability_diagram(
    probs_type: np.ndarray,
    labels_type: np.ndarray,
    save_path: Optional[str] = None,
    n_bins: int = 10,
    type_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> Optional[object]:
    """Plot reliability diagrams (calibration curves) for conflict predictions.

    A reliability diagram shows the relationship between predicted probabilities
    and observed frequencies. A perfectly calibrated model follows the diagonal.
    Under-confident models produce S-shaped curves; over-confident models
    produce inverted-S curves.

    Args:
        probs_type: (N, n_types) sigmoid probabilities.
        labels_type: (N, n_types) multi-hot ground truth.
        save_path: If provided, save the figure to this path.
        n_bins: Number of bins for calibration_curve (default 10).
        type_names: Names for each conflict type (default: conflict types).
        figsize: Matplotlib figure size (width, height).

    Returns:
        The matplotlib figure object, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("[Calib] matplotlib not installed — skipping reliability diagram")
        return None

    from sklearn.calibration import calibration_curve  # type: ignore

    if type_names is None:
        type_names = [f"type_{i}" for i in range(probs_type.shape[1])]

    n_types = probs_type.shape[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ── Left panel: Reliability diagram ──────────────────────────────
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Perfect calibration")

    colors = plt.cm.tab10(np.linspace(0, 1, n_types + 1))  # type: ignore[attr-defined]

    # Binary (any-type) reliability
    binary_true = labels_type.any(axis=1).astype(int)
    binary_prob = probs_type.max(axis=1)
    bin_frac_pos, bin_mean_pred = calibration_curve(
        binary_true, binary_prob, n_bins=n_bins, strategy="uniform"
    )
    ax1.plot(
        bin_mean_pred, bin_frac_pos, "o-", color=colors[0],
        linewidth=2, markersize=6, label="Binary (any type)",
    )

    # Per-type reliability
    for i in range(n_types):
        y = labels_type[:, i].astype(int)
        p = probs_type[:, i]
        # Skip if only one class present (calibration_curve would warn/error)
        if y.sum() == 0 or y.sum() == len(y):
            logger.debug(f"[Calib] Skipping {type_names[i]} reliability — only one class")
            continue
        try:
            frac_pos, mean_pred = calibration_curve(
                y, p, n_bins=n_bins, strategy="uniform"
            )
            ax1.plot(
                mean_pred, frac_pos, "o-", color=colors[i + 1],
                linewidth=2, markersize=5, alpha=0.7,
                label=type_names[i],
            )
        except Exception as e:
            logger.debug(f"[Calib] Could not compute {type_names[i]} calibration curve: {e}")

    ax1.set_xlabel("Mean predicted probability", fontsize=12)
    ax1.set_ylabel("Observed frequency", fontsize=12)
    ax1.set_title("Reliability Diagram", fontsize=14)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # ── Right panel: Prediction histogram ────────────────────────────
    n_hist_bins = 20
    ax2.hist(binary_prob, bins=n_hist_bins, color=colors[0], alpha=0.7,
             edgecolor="white", linewidth=0.5, label="Binary prob")
    for i in range(n_types):
        p = probs_type[:, i]
        ax2.hist(p, bins=n_hist_bins, color=colors[i + 1], alpha=0.3,
                 edgecolor="white", linewidth=0.5, label=type_names[i])

    ax2.set_xlabel("Predicted probability", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Prediction Distribution", fontsize=14)
    ax2.legend(loc="upper center", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"[Calib] Reliability diagram saved to {save_path}")

    return fig


def _close_figure(fig):
    """Safely close a matplotlib figure to avoid memory leaks."""
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Multi-source calibration")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--mustard_root", type=str, default=None)
    p.add_argument("--case_root", type=str, default=None)
    p.add_argument("--iemocap_root", type=str, default=None)
    p.add_argument("--mosei_root", type=str, default=None)
    p.add_argument("--strategy", type=str, default="mean",
                   choices=["mean", "median", "pooled", "per_source"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="calibration_results")
    return p.parse_args()


def _inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,  # type: ignore
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference, return (probs_type, labels_type)."""
    all_probs, all_labels = [], []
    with torch.no_grad():
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
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)  # type: ignore

    # --- Load model ---
    from models.conflictnet import ConflictNet

    checkpoint_path = args.checkpoint
    from models.checkpoint_utils import load_checkpoint_state, extract_model_state

    ckpt = load_checkpoint_state(checkpoint_path, device=device)
    model_state = extract_model_state(ckpt)

    model = ConflictNet()
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    logger.info(f"[Calib] Loaded checkpoint from {args.checkpoint}")

    # --- Inference per source ---
    from data.datasets import (
        MUStARDDataset, CASEDataset, IEMOCAPDataset,
        conflictnet_collate_fn,
    )

    sources = {}
    if args.mustard_root:
        sources["mustard++"] = (MUStARDDataset, {"root": args.mustard_root, "split": "val"})
    if args.case_root:
        sources["case2026"] = (CASEDataset, {"root": args.case_root, "split": "val"})
    if args.iemocap_root:
        sources["iemocap"] = (IEMOCAPDataset, {"root": args.iemocap_root, "sessions": [5]})
    if args.mosei_root:
        from data.datasets import CMUMOSEIDataset
        sources["cmu-mosei"] = (CMUMOSEIDataset, {"root": args.mosei_root, "split": "val"})

    if not sources:
        raise ValueError("Provide at least one of --mustard_root, --case_root, --iemocap_root, --mosei_root")

    source_probs = {}
    source_labels = {}
    for name, (cls, kwargs) in sources.items():
        try:
            ds = cls(**kwargs)
            if len(ds) == 0:
                logger.warning(f"[Calib] {name}: empty dataset, skipping")
                continue
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=2, pin_memory=True, collate_fn=conflictnet_collate_fn,
            )
            probs, labels = _inference(model, loader, device)
            source_probs[name] = probs
            source_labels[name] = labels
            logger.info(f"[Calib] {name}: {len(ds)} samples")
        except Exception as e:
            logger.warning(f"[Calib] {name}: failed to load: {e}")

    if not source_probs:
        raise ValueError("No sources could be loaded for inference")

    # --- Calibrate ---
    result = calibrate_multi_source(
        source_probs, source_labels,
        strategy=args.strategy,
        metric="macro_f1",
    )
    logger.info(f"[Calib] {args.strategy} threshold: {result['threshold']:.3f}")

    # --- Save ---
    serializable = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, dict):
            serializable[k] = {
                kk: float(vv) if isinstance(vv, (np.floating,)) else vv
                for kk, vv in v.items()
            }
        else:
            serializable[k] = v

    report_path = out_dir / "calibration.json"
    with open(report_path, "w") as f:
        json.dump(serializable, f, indent=2)

    # Also save summary as YAML-formatted text
    summary_lines = [
        "Multi-source calibration report",
        f"  Strategy:     {args.strategy}",
        f"  Threshold:    {result['threshold']:.3f}",
        f"  Best macro-F1: {result.get('best_macro_f1', 'N/A')}",
        "---",
    ]
    for name in sorted(source_probs.keys()):
        ps = result.get("per_source", {}).get(name, {})
        summary_lines.append(f"  {name}: {ps}")
    logger.info("\n".join(summary_lines))

    report_path_summary = out_dir / "calibration_summary.txt"
    with open(report_path_summary, "w") as f:
        f.write("\n".join(summary_lines))

    logger.info(f"[Calib] Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
