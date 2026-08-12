#!/usr/bin/env python3
"""Compute sample difficulty scores for ConflictNet training set.

For each sample, difficulty = 1 - max_sigmoid_probability.
High score = uncertain / hard sample.

Usage:
    python scripts/compute_difficulties.py \
        --checkpoint checkpoints/best_model.safetensors \
        --iemocap_root /data/iemocap \
        --output_file difficulties.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Compute sample difficulty scores")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to .safetensors / .pt / .pth checkpoint (optional; random init if omitted)")
    p.add_argument("--iemocap_root", type=str, default=None)
    p.add_argument("--mustard_root", type=str, default=None)
    p.add_argument("--cremad_root", type=str, default=None)
    p.add_argument("--meld_root", type=str, default=None)
    p.add_argument("--output_file", type=str, default="difficulties.json")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # Build datasets
    from data.datasets import IEMOCAPDataset, MUStARDDataset, CREMADDataset, MELDDataset, conflictnet_collate_fn

    train_datasets = []

    if args.iemocap_root:
        train_datasets.append(IEMOCAPDataset(args.iemocap_root, sessions=[1, 2, 3, 4]))
    if args.mustard_root:
        train_datasets.append(MUStARDDataset(args.mustard_root, split="train"))
    if args.cremad_root:
        train_datasets.append(CREMADDataset(args.cremad_root, split="train"))
    if args.meld_root:
        train_datasets.append(MELDDataset(args.meld_root, split="train"))

    if not train_datasets:
        logger.error("Provide at least one dataset root")
        sys.exit(1)

    train_set = ConcatDataset(train_datasets)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=conflictnet_collate_fn,
    )

    logger.info(f"Computing difficulties for {len(train_set)} samples")

    # Build model
    from models.conflictnet import ConflictNet

    model = ConflictNet()
    if args.checkpoint:
        from models.checkpoint_utils import load_checkpoint_state
        state = load_checkpoint_state(args.checkpoint, device=args.device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        logger.info(f"[Difficulties] Loaded checkpoint: {args.checkpoint}")
    else:
        logger.info("[Difficulties] No checkpoint — using random init")

    model.to(args.device)
    model.eval()

    difficulties = {}
    global_idx = 0

    with torch.no_grad():
        for batch in loader:
            batch_gpu = {
                k: v.to(args.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(
                audio=batch_gpu["audio"],
                input_ids=batch_gpu["input_ids"],
                attention_mask=batch_gpu["attention_mask"],
            )
            probs = out.probs_type.cpu().numpy()  # (B, n_types)
            # Difficulty: 1 - max sigmoid probability (uncertainty proxy)
            max_probs = probs.max(axis=1)
            batch_diff = 1.0 - max_probs

            for i, diff in enumerate(batch_diff):
                difficulties[global_idx] = round(float(diff), 6)
                global_idx += 1

    with open(args.output_file, "w") as f:
        json.dump(difficulties, f, indent=2)

    # Summary stats
    scores = list(difficulties.values())
    logger.info(f"Difficulty scores: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}, "
                f"min={np.min(scores):.4f}, max={np.max(scores):.4f}")
    logger.info(f"Saved {len(difficulties)} scores to {args.output_file}")


if __name__ == "__main__":
    main()
