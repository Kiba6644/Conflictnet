#!/usr/bin/env python3
"""Evaluate trained ConflictNet checkpoint on CREMA-D validation split."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json as _json
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import CREMADDataset, make_collate_fn
from models.conflictnet import ConflictNet
from models.checkpoint_utils import load_checkpoint_state, extract_model_state
from evaluation.metrics import compute_all_metrics


import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ConflictNet on CREMA-D")
    p.add_argument("--cremad_root", type=str, default="/kaggle/working/cremad")
    p.add_argument("--checkpoint", type=str, default="/kaggle/working/best_model.safetensors")
    p.add_argument("--meta", type=str, default="/kaggle/working/best_model_meta.json")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    CREMAD = args.cremad_root
    CKPT = args.checkpoint
    META = args.meta
    DEVICE = args.device

    meta = {}
    if Path(META).exists():
        with open(META) as f:
            meta = _json.load(f)

    ec = meta.get("experiment_config", {})
    model = ConflictNet(
        audio_encoder_name=ec.get("audio_encoder", "wavlm"),
        embed_dim=ec.get("embed_dim", 256),
    )
    if Path(CKPT).exists():
        ckpt = load_checkpoint_state(CKPT, device="cpu")
        model.load_state_dict(extract_model_state(ckpt), strict=False)
    else:
        print(f"[WARN] Checkpoint not found at {CKPT}, evaluating uninitialized model")
    model.to(DEVICE).eval()
    print(f"Model: {ec.get('audio_encoder', 'wavlm')}, params: {sum(p.numel() for p in model.parameters()):,}")

    ds = CREMADDataset(CREMAD, split="val")
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True, collate_fn=make_collate_fn())

    all_probs, all_labels, all_sev_pred, all_sev_true = [], [], [], []
    t0 = time.time()

    with torch.no_grad():
        for b in loader:
            bg = {k: v.to(DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            out = model(audio=bg["audio"], input_ids=bg["input_ids"], attention_mask=bg["attention_mask"])
            all_probs.append(out.probs_type.cpu().numpy())
            all_labels.append(b["conflict_type_labels"].numpy())
            if out.severity is not None:
                all_sev_pred.append(out.severity.squeeze(-1).cpu().numpy())
            if "severity" in b:
                all_sev_true.append(b["severity"].squeeze(-1).numpy())

    elapsed = time.time() - t0
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    sev_pred = np.concatenate(all_sev_pred) if all_sev_pred else None
    sev_true = np.concatenate(all_sev_true) if all_sev_true else None
    metrics = compute_all_metrics(probs, labels, severity_pred=sev_pred, severity_true=sev_true)

    print(f"\nEvaluated {len(ds)} samples in {elapsed:.1f}s")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")

    out_dir = Path("/kaggle/working/benchmark_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"dataset": "cremad", "n_samples": len(ds), "eval_time_s": round(elapsed, 2)}
    report.update({k: v for k, v in metrics.items() if isinstance(v, float)})
    with open(out_dir / "cremad_metrics.json", "w") as f:
        _json.dump(report, f, indent=2)

    lines = [
        "=== CREMA-D Benchmark ===",
        f'Best val F1: {meta.get("best_val_f1", 0):.4f} (epoch {meta.get("epoch", 0)})',
    ]
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            lines.append(f"{k}: {v:.4f}")
    (out_dir / "summary.txt").write_text("\n".join(lines))
    print(f"\nReport saved to {out_dir}/")


if __name__ == "__main__":
    main()
