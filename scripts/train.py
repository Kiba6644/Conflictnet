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

# Fix for DeBERTa v2 "fabs" TorchScript compilation bug
os.environ["PYTORCH_JIT"] = "0"

import torch
from torch.utils.data import DataLoader, ConcatDataset
import os

# Add project root to sys.path so 'data', 'models' etc. can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[FlushHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train ConflictNet v2")
    p.add_argument("--iemocap_root", type=str, default=None)
    p.add_argument("--mustard_root", type=str, default=None)
    p.add_argument("--mustard_wav_dir", type=str, default="utterances_final", help="Path to MUStARD wav files")
    p.add_argument("--cremad_root", type=str, default=None, help="CREMA-D dataset root")
    p.add_argument("--meld_root", type=str, default=None, help="MELD dataset root")
    p.add_argument("--meld_max_samples", type=int, default=None,
                   help="Cap MELD dataset to this many samples (stratified); set to 700 to match MUStARD scale")
    p.add_argument("--musan_path", type=str, default=None, help="MUSAN corpus for noise augmentation")
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--audio_encoder", type=str, default="emotion2vec",
                   choices=["emotion2vec", "wavlm", "wavlm_weighted", "wav2vec2"])
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (overrides config)")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile (overrides config)")
    p.add_argument("--pretrain_epochs", type=int, default=5)
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
    p.add_argument("--tokenizer_path", type=str, default=None,
                   help="Path to local tokenizer directory (avoids HuggingFace download)")
    p.add_argument("--target_f1", type=float, default=0.0,
                   help="Target val F1. If not met after training, resume with halved LR (0 = disable)")
    p.add_argument("--max_retries", type=int, default=0,
                   help="Max times to continue training if below target_f1")
    p.add_argument("--resume_epochs", type=int, default=10,
                   help="Additional epochs per retry")
    p.add_argument("--label_smoothing", type=float, default=0.05,
                   help="Label smoothing epsilon for conflict type BCE loss (0 = disabled)")
    return p.parse_args()


def main():
    args = parse_args(argv=None)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.device = f"cuda:{local_rank}"

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
        mustard_kwargs = {}
        if args.tokenizer_path:
            mustard_kwargs["tokenizer_name"] = args.tokenizer_path
            
        train_datasets.append(MUStARDDataset(
            root=args.mustard_root,
            wav_dir=args.mustard_wav_dir,
            split="train",
            **mustard_kwargs
        ))
        val_datasets.append(MUStARDDataset(
            root=args.mustard_root,
            wav_dir=args.mustard_wav_dir,
            split="val",
            **mustard_kwargs
        ))

    if args.cremad_root:
        tok_kwargs = {"tokenizer_name": args.tokenizer_path} if args.tokenizer_path else {}
        train_datasets.append(CREMADDataset(args.cremad_root, split="train", **tok_kwargs))
        val_datasets.append(CREMADDataset(args.cremad_root, split="val", **tok_kwargs))

    if args.meld_root:
        meld_kwargs = {"max_samples": args.meld_max_samples} if args.meld_max_samples else {}
        if args.tokenizer_path:
            meld_kwargs["tokenizer_name"] = args.tokenizer_path
        train_datasets.append(MELDDataset(args.meld_root, split="train", **meld_kwargs))
        val_datasets.append(MELDDataset(args.meld_root, split="val", **meld_kwargs))

    if not train_datasets:
        raise ValueError("Provide at least one of --iemocap_root, --mustard_root, --cremad_root, or --meld_root")

    train_set = ConcatDataset(train_datasets)
    val_set = ConcatDataset(val_datasets)

    # Dynamically determine num_workers to speed up data loading per GPU process
    # On Kaggle, /dev/shm limits cause silent deadlocks with persistent_workers.
    # Set to 0 to run in the main thread and ensure stability.
    optimal_workers = 0
    
    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(train_set) if local_rank != -1 else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if local_rank != -1 else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size or 16,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=optimal_workers,
        collate_fn=train_collate,
        pin_memory=True,
        # persistent_workers avoids re-spawning processes between epochs
        # (saves ~10-20s per epoch on Kaggle T4 with 4 workers)
        persistent_workers=(optimal_workers > 0),
        prefetch_factor=2 if optimal_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size or 16,
        shuffle=False,
        sampler=val_sampler,
        num_workers=optimal_workers,
        collate_fn=val_collate,
        pin_memory=True,
        persistent_workers=(optimal_workers > 0),
        prefetch_factor=2 if optimal_workers > 0 else None,
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
        label_smoothing=args.label_smoothing,
    )

    param_counts = model.count_parameters()
    total_trainable = sum(v["trainable"] for v in param_counts.values())
    logger.info(f"Total trainable parameters: {total_trainable:,}")

    # --- Trainer ---
    from training.trainer import ConflictNetTrainer, get_warmup_cosine_scheduler
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

    retries = 0
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.amp:
        cfg["amp"] = True
    if args.compile:
        cfg["compile"] = True

    while True:
        trainer.train(n_epochs=args.epochs or cfg.get("epochs", 30), pretrain_epochs=args.pretrain_epochs, start_epoch=start_epoch)

        if args.max_retries <= 0 or args.target_f1 <= 0:
            break

        meta_path = Path(args.output_dir) / "best_model_meta.json"
        best_f1 = 0.0
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            best_f1 = meta.get("best_val_f1", 0.0)

        logger.info(f"[Retry {retries+1}/{args.max_retries}] Best val F1 = {best_f1:.4f}, target = {args.target_f1}")

        if best_f1 >= args.target_f1:
            logger.info(f"Target F1 {args.target_f1} reached!")
            break

        if retries >= args.max_retries:
            logger.info(f"Max retries ({args.max_retries}) exhausted, best F1 = {best_f1:.4f}")
            break

        retries += 1
        args.lr = float(args.lr) / 2
        args.resume_from = str(Path(args.output_dir) / "best_model.safetensors")
        args.pretrain_epochs = 0

        start_epoch = trainer.load_checkpoint(args.resume_from)
        args.epochs = start_epoch + args.resume_epochs

        logger.info(f"Resuming (retry {retries}/{args.max_retries}, lr={args.lr:.2e}, epochs {start_epoch}–{args.epochs-1})")

        # Reset scheduler for continuation — cosine decays new_lr → 0 over resume_epochs
        for g in trainer.optimizer.param_groups:
            g["lr"] = args.lr
        steps_per_epoch = len(trainer.train_loader)
        trainer.scheduler = get_warmup_cosine_scheduler(
            trainer.optimizer,
            num_warmup_steps=0,
            num_training_steps=steps_per_epoch * args.resume_epochs,
        )


if __name__ == "__main__":
    main()
