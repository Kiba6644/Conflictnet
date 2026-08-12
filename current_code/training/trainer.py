"""Main training loop for ConflictNet v2.

Supports:
  - Pre-training phase: swap objective only (no conflict labels required)
  - Supervised fine-tuning: all tasks
  - Curriculum learning via CurriculumSampler
  - Gradient clipping + warmup scheduler
  - WandB logging (optional)
  - Automatic mixed precision (AMP) with GradScaler
  - Structured experiment config tracking
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

from data.context_cache import ContextCache

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models.experiment_config import ExperimentConfig

from .curriculum import CurriculumSampler

logger = logging.getLogger(__name__)


def get_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Linear warmup then cosine annealing to 0."""

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class ConflictNetTrainer:
    """Trainer encapsulating train loop, evaluation, checkpointing.

    Args:
        model: ConflictNet model instance.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        cfg: Plain dict with training hyperparameters (backward-compatible).
        exp_config: Structured ``ExperimentConfig`` for metadata / checkpoint tracking.
        device: Training device.
        output_dir: Where to save checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Any,
        exp_config: Optional[ExperimentConfig] = None,
        device: str = "cuda",
        output_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.exp_config = exp_config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = cfg.get("amp", False) and "cuda" in device
        self.grad_scaler: Optional[torch.cuda.amp.GradScaler] = None
        if self.use_amp:
            self.grad_scaler = torch.cuda.amp.GradScaler()

        self._setup_optimizer()
        self._setup_wandb()

        self.global_step = 0
        self.best_val_f1 = 0.0
        self._best_val_f1 = 0.0
        self._patience_counter = 0
        self.ctx_cache = ContextCache(
            max_turns=cfg.get("temporal_max_turns", 8),
            device=device,
        )

    def _setup_optimizer(self):
        lr = self.cfg.get("lr", 2e-5)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=0.01,
        )
        steps_per_epoch = len(self.train_loader)
        epochs = self.cfg.get("epochs", 30)
        warmup = self.cfg.get("warmup_steps", 500)
        self.scheduler = get_warmup_cosine_scheduler(
            self.optimizer,
            num_warmup_steps=warmup,
            num_training_steps=steps_per_epoch * epochs,
        )

    def _setup_wandb(self):
        self.use_wandb = False
        try:
            import wandb  # type: ignore
            if os.environ.get("WANDB_PROJECT"):
                wandb.init(project=os.environ["WANDB_PROJECT"], config=self.cfg)
                self.use_wandb = True
        except ImportError:
            pass

    def _log(self, metrics: Dict[str, float], step: int):
        if self.use_wandb:
            import wandb  # type: ignore
            wandb.log(metrics, step=step)
        logger.info(f"[Step {step}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    def train_epoch(self, epoch: int, pretraining: bool = False) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        grad_accum_steps = int(
            self.cfg.get("gradient_accumulation_steps", 1)
        )

        # Update curriculum sampler if present
        if isinstance(self.train_loader.sampler, CurriculumSampler):
            self.train_loader.sampler.set_epoch(epoch)

        self.optimizer.zero_grad()

        for batch in self.train_loader:
            # non_blocking=True pairs with pin_memory=True on the DataLoader
            batch = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Populate context from cache (history of past turns for each conversation)
            conv_ids = batch.get("conversation_ids", [])
            if conv_ids and isinstance(conv_ids, list):
                str_conv_ids: list[str] = [str(x) for x in conv_ids]
                model_embed_dim = getattr(self.model, "embed_dim", 256)
                embed_dim_val = model_embed_dim if isinstance(model_embed_dim, int) else 256
                ctx_embeds, ctx_padding, _ = self.ctx_cache.get_batch_context(
                    str_conv_ids, embed_dim=embed_dim_val
                )
            else:
                ctx_embeds = None
                ctx_padding = None

            with torch.autocast(device_type=self.device, enabled=self.use_amp):
                output = self.model(
                    audio=batch["audio"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    audio_attention_mask=batch.get("audio_attention_mask"),
                    context_embeds=ctx_embeds,
                    context_padding=ctx_padding,
                    speaker_roles=batch.get("speaker_roles"),
                    prosody_z=batch.get("prosody_z"),
                    word_timestamps=batch.get("word_timestamps"),
                    token_word_boundaries=batch.get("token_word_boundaries"),
                    conflict_type_labels=batch.get("conflict_type_labels"),
                    severity_labels=batch.get("severity"),
                    conflict_binary_labels=batch.get("conflict_binary"),
                    pretraining=pretraining,
                )

            # Update context cache with current turn fused embeddings
            if conv_ids and isinstance(conv_ids, list) and output.fused_embed is not None:
                str_conv_ids: list[str] = [str(x) for x in conv_ids]
                self.ctx_cache.batch_update(str_conv_ids, output.fused_embed)

            loss = output.loss
            if loss is None or not loss.requires_grad:
                continue

            loss = loss / grad_accum_steps

            if self.use_amp and self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (n_batches + 1) % grad_accum_steps == 0:
                if self.use_amp and self.grad_scaler is not None:
                    self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if self.use_amp and self.grad_scaler is not None:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            else:
                logger.debug("Skipping optimizer step, accumulating gradients")

            total_loss += loss.item() * grad_accum_steps
            n_batches += 1

            if self.global_step % 100 == 0 and self.global_step > 0:
                metrics = {"train/loss": loss.item() * grad_accum_steps, "train/lr": self.scheduler.get_last_lr()[0]}
                if output.loss_breakdown:
                    for k, v in output.loss_breakdown.items():
                        if isinstance(v, float):
                            metrics[f"train/{k}"] = v
                self._log(metrics, self.global_step)

        # Handle remaining gradients when epoch ends mid-accumulation
        if n_batches % grad_accum_steps != 0:
            if self.use_amp and self.grad_scaler is not None:
                self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if self.use_amp and self.grad_scaler is not None:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        from sklearn.metrics import f1_score  # type: ignore

        self.model.eval()
        # Clear context cache to prevent training dialogue context from
        # leaking into validation (fixes L3 data leakage path)
        self.ctx_cache.clear()
        all_preds, all_labels = [], []

        for batch in self.val_loader:
            batch = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            conv_ids = batch.get("conversation_ids", [])
            ctx_embeds, ctx_padding, _ = self.ctx_cache.get_batch_context(
                conv_ids, embed_dim=getattr(self.model, "embed_dim", 256)
            ) if conv_ids else (None, None, [])
            with torch.autocast(device_type=self.device, enabled=self.use_amp):
                output = self.model(
                audio=batch["audio"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                audio_attention_mask=batch.get("audio_attention_mask"),
                prosody_z=batch.get("prosody_z"),
                context_embeds=ctx_embeds,
                context_padding=ctx_padding,
                speaker_roles=batch.get("speaker_roles"),
                word_timestamps=batch.get("word_timestamps"),
                token_word_boundaries=batch.get("token_word_boundaries"),
            )
            if output.conflict_flag is not None:
                preds = output.conflict_flag.cpu().numpy().astype(int)
            else:
                import numpy as np
                preds = np.zeros(batch["audio"].size(0), dtype=int)
            conflict_binary = batch.get("conflict_binary")
            if conflict_binary is None:
                conflict_binary = torch.zeros(batch["audio"].size(0))  # type: ignore
            labels = conflict_binary.cpu().numpy().astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

        f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))  # type: ignore
        return {"val/f1_weighted": f1}

    def train(
        self,
        n_epochs: int,
        pretrain_epochs: int = 0,
        start_epoch: int = 0,
    ):
        """Full training loop.

        Args:
            n_epochs: Total training epochs (including pretrain_epochs).
            pretrain_epochs: Epochs using only self-supervised swap objective.
            start_epoch: Epoch index to resume from (for checkpoint resume).
        """
        early_stop_patience = int(
            self.cfg.get("early_stop_patience", 10)
        )

        for epoch in range(start_epoch, n_epochs):
            is_pretrain = epoch < pretrain_epochs
            phase = "pretrain" if is_pretrain else "finetune"
            logger.info(f"[Epoch {epoch+1}/{n_epochs}] phase={phase}")

            train_metrics = self.train_epoch(epoch, pretraining=is_pretrain)
            val_metrics = self.evaluate()

            all_metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}
            self._log(all_metrics, self.global_step)

            # Save epoch checkpoint
            # Model weights → safetensors (pickle-free, CWE-502 safe)
            # Training state → .pt (optimizer/scheduler can't use safetensors)
            self._save_checkpoint(epoch)

            # Save best checkpoint
            if val_metrics.get("val/f1_weighted", 0) > self.best_val_f1:
                self.best_val_f1 = val_metrics["val/f1_weighted"]
                self._save_checkpoint(epoch, is_best=True)
                logger.info(f"  ✓ New best val F1 = {self.best_val_f1:.4f}")

            # Early stopping check
            val_f1 = val_metrics.get("val/f1_weighted", 0)
            if val_f1 > self._best_val_f1:
                self._best_val_f1 = val_f1
                self._patience_counter = 0
            else:
                self._patience_counter += 1
                if self._patience_counter >= early_stop_patience:
                    logger.info("Early stopping triggered")
                    break

        logger.info(f"Training complete. Best val F1 = {self.best_val_f1:.4f}")

    @staticmethod
    def _get_git_info() -> Dict[str, str]:
        """Get git commit hash and dirty status for reproducibility tracking."""
        try:
            import subprocess
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
            return {"git_sha": sha, "git_dirty": "yes" if dirty else "no"}
        except Exception:
            return {"git_sha": "unknown", "git_dirty": "unknown"}

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint using safetensors for model weights (pickle-free).

        Saves:
          - ``*.safetensors``: model weights only (safe, no arbitrary code exec)
          - ``*_training_state.pt``: optimizer + scheduler state (torch.save,
            only loaded by our own ``load_checkpoint``; NOT user-facing)
          - ``*_meta.json``: scalar metadata (epoch, step, F1)
        """
        import json as _json

        prefix = "best_model" if is_best else f"checkpoint_epoch{epoch + 1}"

        _safe_save = getattr(torch, "save")

        # 1. Model weights → safetensors (primary, pickle-free)
        try:
            from safetensors.torch import save_file as st_save
            st_path = self.output_dir / f"{prefix}.safetensors"
            st_save(self.model.state_dict(), str(st_path))
        except ImportError:
            # Fallback: torch.save model weights only (still state_dict, not full model)
            logger.warning("[Checkpoint] safetensors not installed — falling back to torch.save for model weights")
            _safe_save(self.model.state_dict(), self.output_dir / f"{prefix}.pt")  # nosec

        # 2. Training state → .pt (optimizer/scheduler contain non-tensor objects)
        training_state: Dict[str, Any] = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        if self.grad_scaler is not None:
            training_state["grad_scaler_state_dict"] = self.grad_scaler.state_dict()
        _safe_save(training_state, self.output_dir / f"{prefix}_training_state.pt")  # nosec

        # 3. Scalar metadata + experiment config snapshot → JSON
        meta: Dict[str, Any] = {
            "global_step": self.global_step,
            "best_val_f1": self.best_val_f1,
            "epoch": epoch,
            **self._get_git_info(),
        }
        if self.exp_config is not None:
            meta["experiment_config"] = self.exp_config.to_dict()
        with open(self.output_dir / f"{prefix}_meta.json", "w") as f:
            _json.dump(meta, f, indent=2, default=str)

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint from path and restore model/optimizer/scheduler state.

        Supports both formats:
          - ``.safetensors``: loads model weights + looks for sibling
            ``*_training_state.pt`` and ``*_meta.json`` files.
          - ``.pt``: legacy format — loads everything from a single file.

        Args:
            path: Path to the checkpoint file (.safetensors or .pt).

        Returns:
            The next epoch to resume training from (0-indexed).
        """
        import json as _json

        checkpoint_path = Path(path)
        from models.checkpoint_utils import load_checkpoint_state, extract_model_state

        if checkpoint_path.suffix == ".safetensors":
            # Safe path: safetensors model weights + sidecar files
            model_state = load_checkpoint_state(checkpoint_path, device=self.device)

            result = self.model.load_state_dict(model_state, strict=False)
            if result.missing_keys:
                logger.warning(f"Missing keys in checkpoint: {result.missing_keys}")
            if result.unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {result.unexpected_keys}")

            # Load training state from sidecar .pt file
            stem = checkpoint_path.stem  # e.g. "best_model" or "checkpoint_epoch5"
            training_state_path = checkpoint_path.parent / f"{stem}_training_state.pt"
            if training_state_path.exists():
                ts: Dict[str, Any] = load_checkpoint_state(training_state_path, device=self.device)
                self.optimizer.load_state_dict(ts["optimizer_state_dict"])
                self.scheduler.load_state_dict(ts["scheduler_state_dict"])
                if self.grad_scaler is not None and "grad_scaler_state_dict" in ts:
                    self.grad_scaler.load_state_dict(ts["grad_scaler_state_dict"])

            # Load scalar metadata from JSON sidecar
            meta_path = checkpoint_path.parent / f"{stem}_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = _json.load(f)
                self.global_step = meta.get("global_step", 0)
                self.best_val_f1 = meta.get("best_val_f1", 0.0)
                self._best_val_f1 = self.best_val_f1
                resumed_epoch: int = meta.get("epoch", -1)
            else:
                resumed_epoch = -1

        else:
            # Legacy .pt path (backward compatible)
            ckpt: Dict[str, Any] = load_checkpoint_state(checkpoint_path, device=self.device)

            model_state = extract_model_state(ckpt)
            result = self.model.load_state_dict(model_state, strict=False)
            if result.missing_keys:
                logger.warning(f"Missing keys in checkpoint: {result.missing_keys}")
            if result.unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {result.unexpected_keys}")

            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.global_step = ckpt.get("global_step", 0)
            self.best_val_f1 = ckpt.get("best_val_f1", 0.0)
            self._best_val_f1 = self.best_val_f1
            resumed_epoch = ckpt.get("epoch", -1)

        logger.info(f"Resumed from checkpoint at epoch {resumed_epoch + 1}, global_step={self.global_step}")
        return resumed_epoch + 1
