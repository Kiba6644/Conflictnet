"""CLI: train ConflictNet v2.

Usage:
    python scripts/train.py --config configs/default.yaml \
        --iemocap_root /data/iemocap \
        --mustard_root  /data/mustard \
        --output_dir checkpoints/run1 \
        --pretrain_epochs 5 \
        --epochs 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Fix for macOS OpenMP multiple initialization error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader, ConcatDataset

# Add project root to sys.path so 'data', 'models' etc. can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train ConflictNet v2")
    p.add_argument("--iemocap_root", type=str, default=None)
    p.add_argument("--mustard_root", type=str, default=None)
    p.add_argument("--cremad_root", type=str, default=None, help="CREMA-D dataset root")
    p.add_argument("--meld_root", type=str, default=None, help="MELD dataset root")
    p.add_argument("--musan_path", type=str, default=None, help="MUSAN corpus for noise augmentation")
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--audio_encoder", type=str, default="emotion2vec",
                   choices=["emotion2vec", "wavlm", "wav2vec2"])
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--pretrain_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_speaker_norm", action="store_true")
    p.add_argument("--no_temporal", action="store_true",
                   help="Disable Transformer temporal context module")
    p.add_argument("--no_cross_attn_injection", action="store_true",
                   help="Disable cross-attention injection from temporal context into audio+text")
    p.add_argument("--no_speaker_adaptive_threshold", action="store_true",
                   help="Disable speaker-adaptive divergence threshold (use fixed threshold)")
    p.add_argument("--no_baseline_subtract", action="store_true",
                   help="Disable baseline-subtract prosody normalisation (use z-score instead)")
    p.add_argument("--no_word_divergence", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--prosody_stats", type=str, default=None,
                   help="Path to .pt file from compute_prosody_stats.py with per-utterance z-scores")
    p.add_argument("--amp", action="store_true",
                   help="Enable automatic mixed precision (fp16) training")
    return p.parse_args()


def main():
    args = parse_args(argv=None)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Build datasets ---
    from data.datasets import (
        IEMOCAPDataset, MUStARDDataset, CREMADDataset, MELDDataset,
        make_collate_fn,
    )

    # Create augmentation-aware collate functions via closure (fork-safe)
    from data.augmentation import AudioAugmentor

    # Load pre-computed prosody z-scores if available
    prosody_lookup = None
    if args.prosody_stats:
        prosody_path = Path(args.prosody_stats)
        zscores_p = prosody_path.parent / f"{prosody_path.stem}.zscores.json"
        if not zscores_p.exists():
            zscores_p = prosody_path.with_suffix(".zscores.json")
        if zscores_p.exists():
            try:
                with open(zscores_p) as f:
                    raw = json.load(f)
                prosody_lookup = {k: torch.tensor(v) for k, v in raw.items()}
                if not prosody_lookup:
                    prosody_lookup = None
                else:
                    logger.info(f"[Prosody] loaded {len(prosody_lookup)} z-score entries from {zscores_p}")
            except Exception as e:
                logger.warning(f"[Prosody] failed to load {zscores_p}: {e}")
        else:
            logger.warning(f"[Prosody] lookup file not found: {zscores_p}")

    augmentor = AudioAugmentor(
        sample_rate=16000,
        musan_path=getattr(args, "musan_path", None),
    )
    # Both collate fns use the SAME prosody lookup (z-scores computed from
    # training-data-only speaker statistics by compute_prosody_stats.py).
    # This is CORRECT — val utterances get z-scores based on training speaker
    # statistics, preventing data leakage across splits.
    train_collate = make_collate_fn(augmentor=augmentor, prosody_lookup=prosody_lookup)
    val_collate = make_collate_fn(prosody_lookup=prosody_lookup)

    train_datasets = []
    val_datasets = []

    if args.iemocap_root:
        # Leave-one-session-out: sessions 1-4 train, 5 val
        train_datasets.append(IEMOCAPDataset(args.iemocap_root, sessions=[1, 2, 3, 4]))
        val_datasets.append(IEMOCAPDataset(args.iemocap_root, sessions=[5]))

    if args.mustard_root:
        train_datasets.append(MUStARDDataset(args.mustard_root, split="train"))
        val_datasets.append(MUStARDDataset(args.mustard_root, split="val"))

    if args.cremad_root:
        train_datasets.append(CREMADDataset(args.cremad_root, split="train"))
        val_datasets.append(CREMADDataset(args.cremad_root, split="val"))

    if args.meld_root:
        train_datasets.append(MELDDataset(args.meld_root, split="train"))
        val_datasets.append(MELDDataset(args.meld_root, split="val"))

    if not train_datasets:
        raise ValueError("Provide at least one of --iemocap_root, --mustard_root, --cremad_root, or --meld_root")

    train_set = ConcatDataset(train_datasets)
    val_set = ConcatDataset(val_datasets)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=train_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=val_collate,
        pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    # --- Build model ---
    from models.conflictnet import ConflictNet

    model = ConflictNet(
        audio_encoder_name=args.audio_encoder,
        embed_dim=args.embed_dim,
        use_speaker_norm=not args.no_speaker_norm,
        use_temporal=not args.no_temporal,
        use_cross_attn_injection=not args.no_cross_attn_injection,
        use_speaker_adaptive_threshold=not args.no_speaker_adaptive_threshold,
        use_baseline_subtract=not args.no_baseline_subtract,
        use_word_divergence=not args.no_word_divergence,
        lora_r=args.lora_r,
    )

    param_counts = model.count_parameters()
    total_trainable = sum(v["trainable"] for v in param_counts.values())
    logger.info(f"Total trainable parameters: {total_trainable:,}")

    # --- Trainer ---
    from training.trainer import ConflictNetTrainer
    from models.experiment_config import ExperimentConfig

    exp_config = ExperimentConfig.from_args(args)
    cfg = exp_config.to_dict()
    trainer = ConflictNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        exp_config=exp_config,
        device=args.device,
        output_dir=args.output_dir,
    )

    start_epoch = 0
    if args.resume_from:
        start_epoch = trainer.load_checkpoint(args.resume_from)

    trainer.train(n_epochs=args.epochs, pretrain_epochs=args.pretrain_epochs, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
