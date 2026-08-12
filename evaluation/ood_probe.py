"""Speaker-OOD probe: evaluate a trained ConflictNet on held-out speakers.

Measures model generalisation to speakers **not seen during training**.
Reports per-speaker and aggregate metrics, plus degradation relative to
in-distribution (seen-speaker) performance.

Usage::

    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt \\
        --ood_probe --held_out_speakers speaker_042,speaker_099 \\
        --case_root /data/case2026

Or via the standalone CLI below::

    python evaluation/ood_probe.py --checkpoint checkpoints/best_model.pt \\
        --case_root /data/case2026 --held_out_speakers speaker_042,speaker_099
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Speaker-OOD probe")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--case_root", type=str, default=None)
    p.add_argument("--iemocap_root", type=str, default=None)
    p.add_argument("--mustard_root", type=str, default=None)
    p.add_argument("--held_out_speakers", type=str, required=True,
                   help="Comma-separated list of held-out speaker IDs")
    p.add_argument("--seen_speakers", type=str, default=None,
                   help="Comma-separated list of seen (training) speaker IDs for comparison")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="ood_results")
    return p.parse_args()


def _get_speaker_indices(dataset, speaker_ids: set) -> List[int]:
    """Return indices of samples belonging to any speaker in speaker_ids."""
    indices = []
    for i in range(len(dataset)):
        try:
            item = dataset[i]
            if item.get("speaker_id", "") in speaker_ids:
                indices.append(i)
        except Exception:
            continue
    return indices


def _build_dataset(root: str, dataset_cls, tokenizer_name: str = "microsoft/deberta-v3-large", **kwargs):
    """Try to build a dataset, returning None if the root is invalid."""
    try:
        ds = dataset_cls(root=root, tokenizer_name=tokenizer_name, **kwargs)
        if len(ds) > 0:
            return ds
    except Exception as e:
        logger.warning(f"Failed to load {dataset_cls.__name__} from {root}: {e}")
    return None


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)  # type: ignore
    held_out = {s.strip() for s in args.held_out_speakers.split(",")}
    seen = set()
    if args.seen_speakers:
        seen = {s.strip() for s in args.seen_speakers.split(",")}

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
    logger.info(f"[OOD-Probe] Loaded checkpoint from {args.checkpoint}")

    # --- Build dataset ---
    from data.datasets import IEMOCAPDataset, MUStARDDataset, CASEDataset
    from data.datasets import conflictnet_collate_fn

    datasets = []
    if args.iemocap_root:
        ds = _build_dataset(args.iemocap_root, IEMOCAPDataset, sessions=[5])
        if ds:
            datasets.append(ds)
    if args.mustard_root:
        ds = _build_dataset(args.mustard_root, MUStARDDataset, split="val")
        if ds:
            datasets.append(ds)
    if args.case_root:
        ds = _build_dataset(args.case_root, CASEDataset, split="val")
        if ds:
            datasets.append(ds)

    if not datasets:
        raise ValueError("No datasets could be loaded. Provide at least one of --case_root, --iemocap_root, --mustard_root")

    full_dataset = ConcatDataset(datasets)
    logger.info(f"[OOD-Probe] Full dataset: {len(full_dataset)} samples")

    # --- Filter by speaker ---
    held_out_idx = _get_speaker_indices(full_dataset, held_out)
    seen_idx = _get_speaker_indices(full_dataset, seen) if seen else []

    if not held_out_idx:
        logger.error(f"[OOD-Probe] No samples found for held-out speakers: {held_out}")
        return

    held_out_set = Subset(full_dataset, held_out_idx)
    seen_set = Subset(full_dataset, seen_idx) if seen_idx else None

    logger.info(f"[OOD-Probe] Held-out samples: {len(held_out_set)} from {len(held_out)} speakers")
    if seen_set:
        logger.info(f"[OOD-Probe] Seen samples: {len(seen_set)} from {len(seen)} speakers")

    pin = (device.type != "cpu")
    held_out_loader = DataLoader(
        held_out_set, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=pin, collate_fn=conflictnet_collate_fn,
    )

    # --- Per-speaker tracking ---
    speaker_probs: Dict[str, List[np.ndarray]] = defaultdict(list)
    speaker_labels: Dict[str, List[np.ndarray]] = defaultdict(list)
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in held_out_loader:
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
            probs_np = out.probs_type.cpu().numpy()
            labels_np = batch["conflict_type_labels"].numpy()
            all_probs.append(probs_np)
            all_labels.append(labels_np)

            for i, spk in enumerate(batch["speaker_ids"]):
                speaker_probs[spk].append(probs_np[i])
                speaker_labels[spk].append(labels_np[i])

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # --- Metrics ---
    from evaluation.metrics import compute_all_metrics, print_metrics

    overall = compute_all_metrics(all_probs, all_labels)
    print_metrics(overall, prefix="[OOD-Probe] Overall")

    # Per-speaker breakdown
    per_speaker = {}
    for spk in sorted(speaker_probs.keys()):
        probs = np.stack(speaker_probs[spk], axis=0)
        labels = np.stack(speaker_labels[spk], axis=0)
        spk_metrics = compute_all_metrics(probs, labels)
        per_speaker[spk] = {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in spk_metrics.items()
        }
        logger.info(f"[OOD-Probe] Speaker {spk} ({len(probs)} samples): "
                    f"macro_f1={spk_metrics['macro_f1']:.4f}")

    # --- Compare with seen speakers (if provided) ---
    seen_metrics = None
    if seen_set:
        seen_loader = DataLoader(
            seen_set, batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=pin, collate_fn=conflictnet_collate_fn,
        )
        seen_probs, seen_labels = [], []
        with torch.no_grad():
            for batch in seen_loader:
                batch_gpu = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                out = model(
                    audio=batch_gpu["audio"],
                    input_ids=batch_gpu["input_ids"],
                    attention_mask=batch_gpu["attention_mask"],
                    audio_attention_mask=batch_gpu.get("audio_attention_mask"),
                    prosody_z=batch_gpu.get("prosody_z"),

                )
                seen_probs.append(out.probs_type.cpu().numpy())
                seen_labels.append(batch["conflict_type_labels"].numpy())
        seen_probs = np.concatenate(seen_probs, axis=0)
        seen_labels = np.concatenate(seen_labels, axis=0)
        seen_metrics = compute_all_metrics(seen_probs, seen_labels)

        logger.info(f"[OOD-Probe] Seen macro_f1={seen_metrics['macro_f1']:.4f}, "
                    f"OOD macro_f1={overall['macro_f1']:.4f}, "
                    f"drop={seen_metrics['macro_f1'] - overall['macro_f1']:.4f}")

    # --- Save ---
    report: Dict[str, Any] = {
        "overall": {k: float(v) if isinstance(v, (np.floating, float)) else v
                     for k, v in overall.items()},
        "per_speaker": per_speaker,
        "n_held_out_samples": len(held_out_set),
        "n_held_out_speakers": len(held_out),
    }
    if seen_metrics:
        report["seen_comparison"] = {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in seen_metrics.items()
        }
        report["ood_drop_macro_f1"] = float(seen_metrics["macro_f1"] - overall["macro_f1"])

    report_path = out_dir / "ood_probe.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"[OOD-Probe] Report saved to {report_path}")


if __name__ == "__main__":
    main()
