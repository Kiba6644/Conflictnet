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
import sys
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

        # PyTorch 2.x compile support
        self.use_compile = cfg.get("compile", False) and hasattr(torch, "compile")
        if self.use_compile:
            logger.info("Using torch.compile() for model speedup.")
            self.model = torch.compile(self.model)

        # Multi-GPU DataParallel / DDP support
        if "cuda" in device:
            import os
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
            if local_rank != -1 and torch.distributed.is_initialized():
                logger.info(f"Using DistributedDataParallel on GPU {local_rank}.")

                # Fail fast if ranks built different models (e.g. one rank hit a
                # fallback encoder because a HuggingFace/SpeechBrain download raced
                # across torchrun workers). Otherwise DDP crashes deep inside with a
                # cryptic "inconsistent params" error that is hard to diagnose.
                world_size = torch.distributed.get_world_size()
                n_trainable_tensors = torch.tensor(
                    [sum(1 for p in self.model.parameters() if p.requires_grad)],
                    device=device,
                )
                gathered = [torch.zeros_like(n_trainable_tensors) for _ in range(world_size)]
                torch.distributed.all_gather(gathered, n_trainable_tensors)
                counts = [int(count.item()) for count in gathered]
                if any(count != counts[0] for count in counts):
                    raise RuntimeError(
                        f"Ranks built different models (trainable tensor counts: {counts}). "
                        "A pretrained-model fallback differed between ranks. Ensure every "
                        "rank uses the shared Hugging Face and ModelScope caches."
                    )

                logger.info(f"[DDP] Rank {local_rank}: parameter check passed; wrapping model.")
                self.model = nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    # Must be True because some components (like swap_objective) 
                    # are only used during pre-training phases.
                    find_unused_parameters=True,
                )
                logger.info(f"[DDP] Rank {local_rank}: model wrapper initialized.")
            elif torch.cuda.device_count() > 1:
                logger.info(f"Using DataParallel across {torch.cuda.device_count()} GPUs.")
                self.model = nn.DataParallel(self.model)

        # cuDNN benchmark for faster convolutions (useful if input sizes are static)
        if "cuda" in device:
            torch.backends.cudnn.benchmark = True

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.exp_config = exp_config
        self.device = device
        self._autocast_device_type = torch.device(device).type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = cfg.get("amp", False) and "cuda" in device
        self.grad_scaler: Optional[torch.amp.GradScaler] = None
        if self.use_amp:
            self.grad_scaler = torch.amp.GradScaler("cuda")

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
        # Use fused=True if supported (PyTorch 2.0+)
        kwargs = {}
        if hasattr(torch.optim.AdamW, "__init__") and "fused" in AdamW.__init__.__code__.co_varnames:
            kwargs["fused"] = True

        # Standard practice: do NOT apply weight decay to bias terms and
        # LayerNorm parameters — this follows the BERT/DeBERTa fine-tuning
        # recipe and typically yields slightly better accuracy.
        no_decay = {"bias", "layer_norm.weight", "layernorm.weight", "LayerNorm.weight"}

        # Encoder params (LoRA adapters in DeBERTa)
        encoder_params_wd  = [p for n, p in self.model.named_parameters()
                               if p.requires_grad and "text_encoder" in n
                               and not any(nd in n for nd in no_decay)]
        encoder_params_nwd = [p for n, p in self.model.named_parameters()
                               if p.requires_grad and "text_encoder" in n
                               and any(nd in n for nd in no_decay)]
        # Newly-initialized heads (projection, fusion, temporal, classifier)
        head_params_wd  = [p for n, p in self.model.named_parameters()
                            if p.requires_grad and "text_encoder" not in n
                            and not any(nd in n for nd in no_decay)]
        head_params_nwd = [p for n, p in self.model.named_parameters()
                            if p.requires_grad and "text_encoder" not in n
                            and any(nd in n for nd in no_decay)]

        self.optimizer = AdamW([
            {"params": encoder_params_wd,  "lr": lr,       "weight_decay": 0.01, "name": "encoder"},
            {"params": encoder_params_nwd, "lr": lr,       "weight_decay": 0.0,  "name": "encoder_nwd"},
            {"params": head_params_wd,     "lr": lr * 10,  "weight_decay": 0.01, "name": "heads"},
            {"params": head_params_nwd,    "lr": lr * 10,  "weight_decay": 0.0,  "name": "heads_nwd"},
        ], **kwargs)
        grad_accum_steps = int(self.cfg.get("gradient_accumulation_steps", 1))
        steps_per_epoch = max(1, len(self.train_loader) // grad_accum_steps)
        epochs = int(self.cfg.get("epochs") or 50)
        warmup_steps = int(self.cfg.get("warmup_steps") or 0)
        self.scheduler = get_warmup_cosine_scheduler(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, steps_per_epoch * epochs),
        )

    def _setup_wandb(self):
        self.use_wandb = False
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank != 0:
            return
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
        logger.info(
            f"[Step {step}] "
            + " | ".join(
                f"{k}={v:.2e}" if k == "lr" else f"{k}={v:.4f}"
                for k, v in metrics.items()
            )
        )
        sys.stdout.flush()

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
        elif hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        self.optimizer.zero_grad()
        self.ctx_cache.clear()

        try:
            from tqdm.auto import tqdm
            loader_iter = tqdm(
                self.train_loader, 
                desc=f"Train Epoch {epoch+1}", 
                disable=torch.distributed.is_initialized() and int(os.environ.get("LOCAL_RANK", -1)) != 0,
                file=sys.stdout,
                miniters=5
            )
        except ImportError:
            loader_iter = self.train_loader

        for batch in loader_iter:
            # non_blocking=True pairs with pin_memory=True on the DataLoader
            batch = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Populate context from cache (history of past turns for each conversation)
            conv_ids = batch.get("conversation_ids", [])
            turn_indices = batch.get("turn_indices", None)
            if isinstance(turn_indices, torch.Tensor):
                turn_indices = turn_indices.cpu().tolist()
            if conv_ids and isinstance(conv_ids, list):
                str_conv_ids: list[str] = [str(x) for x in conv_ids]
                model_embed_dim = getattr(self.model, "embed_dim", 256)
                embed_dim_val = model_embed_dim if isinstance(model_embed_dim, int) else 256
                ctx_embeds, ctx_padding, _ = self.ctx_cache.get_batch_context(
                    str_conv_ids, embed_dim=embed_dim_val, turn_indices=turn_indices
                )
            else:
                ctx_embeds = None
                ctx_padding = None
            # Extract FunASR embeddings outside DDP scope to prevent NCCL timeout
            # (Fixes Bug 10 / the DDP broadcast timeout reported by user)
            _model_inner = getattr(self.model, "module", self.model)
            precomputed_audio_embed = None
            if hasattr(_model_inner, "audio_encoder") and getattr(_model_inner.audio_encoder, "_backend", None) == "funasr":
                with torch.no_grad():
                    precomputed_audio_embed = _model_inner.audio_encoder(batch["audio"])

            with torch.autocast(device_type=self._autocast_device_type, enabled=self.use_amp):
                output = self.model(
                    audio=batch["audio"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    audio_attention_mask=batch.get("audio_attention_mask"),
                    precomputed_audio_embed=precomputed_audio_embed,
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
                    dataset_names=batch.get("dataset_names"),
                )

            # Update context cache with current turn fused embeddings
            if conv_ids and isinstance(conv_ids, list) and output.fused_embed is not None:
                str_conv_ids: list[str] = [str(x) for x in conv_ids]
                self.ctx_cache.batch_update(str_conv_ids, output.fused_embed, turn_indices=turn_indices)

            loss = output.loss
            if loss is None or not loss.requires_grad:
                continue

            loss = loss / grad_accum_steps

            is_sync_step = (n_batches + 1) % grad_accum_steps == 0
            
            # Use no_sync() if accumulating gradients in DDP to avoid premature all_reduce syncs
            import contextlib
            from torch.nn.parallel import DistributedDataParallel as DDP
            sync_context = self.model.no_sync() if not is_sync_step and isinstance(self.model, DDP) else contextlib.nullcontext()

            with sync_context:
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

            if self.global_step % 10 == 0 and self.global_step > 0:
                metrics = {
                    "loss": loss.item() * grad_accum_steps,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if output.loss_breakdown:
                    for k, v in output.loss_breakdown.items():
                        if isinstance(v, float) and k in ("bce_type", "bce_binary", "swap_loss", "severity"):
                            metrics[k] = v
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
            # BUG FIX: zero_grad() was missing here. Without it, stale gradients
            # from the end of this epoch survive into the first batch of the next
            # epoch and get double-accumulated, corrupting the first update.
            self.optimizer.zero_grad()
            self.global_step += 1

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

        is_ddp = torch.distributed.is_initialized()
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        metric_keys = ("val/f1_binary", "val/auc_binary", "val/macro_ap", "val/f1_weighted")
        if is_ddp and local_rank != 0:
            # Rank 0 owns the complete validation loader. Waiting here avoids
            # duplicate work and ensures every rank receives the same metric for
            # checkpointing and early stopping.
            metric_values = torch.zeros(len(metric_keys), device=self.device)
            torch.distributed.broadcast(metric_values, src=0)
            return {key: float(value) for key, value in zip(metric_keys, metric_values.cpu().tolist())}

        self.model.eval()
        # Clear context cache to prevent training dialogue context from
        # leaking into validation (fixes L3 data leakage path)
        self.ctx_cache.clear()
        import numpy as np
        all_probs = []
        all_labels = []
        all_binary = []

        try:
            try:
                from tqdm.auto import tqdm
                loader_iter = tqdm(
                    self.val_loader,
                    desc="Eval",
                    disable=torch.distributed.is_initialized() and int(os.environ.get("LOCAL_RANK", -1)) != 0,
                    file=sys.stdout,
                    miniters=5
                )
            except ImportError:
                loader_iter = self.val_loader

            for batch in loader_iter:
                batch = {
                    k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                conv_ids = batch.get("conversation_ids", [])
                turn_indices = batch.get("turn_indices", None)
                if isinstance(turn_indices, torch.Tensor):
                    turn_indices = turn_indices.cpu().tolist()
                str_conv_ids = [str(x) for x in conv_ids] if conv_ids else []
                # Use .module to get the true embed_dim when wrapped by DDP/DataParallel.
                # Calling getattr on the wrapper returns the default (256) instead of
                # the actual value, causing a shape mismatch crash.
                _model_inner = getattr(self.model, "module", self.model)
                _embed_dim = getattr(_model_inner, "embed_dim", 256)
                ctx_embeds, ctx_padding, _ = self.ctx_cache.get_batch_context(
                    str_conv_ids,
                    embed_dim=_embed_dim,
                    turn_indices=turn_indices,
                ) if str_conv_ids else (None, None, [])
                # FunASR bypass for evaluate as well
                precomputed_audio_embed = None
                if hasattr(_model_inner, "audio_encoder") and getattr(_model_inner.audio_encoder, "_backend", None) == "funasr":
                    precomputed_audio_embed = _model_inner.audio_encoder(batch["audio"])

                with torch.autocast(device_type=self._autocast_device_type, enabled=self.use_amp):
                    output = self.model(
                    audio=batch["audio"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    audio_attention_mask=batch.get("audio_attention_mask"),
                    precomputed_audio_embed=precomputed_audio_embed,
                    prosody_z=batch.get("prosody_z"),
                    context_embeds=ctx_embeds,
                    context_padding=ctx_padding,
                    speaker_roles=batch.get("speaker_roles"),
                    word_timestamps=batch.get("word_timestamps"),
                    token_word_boundaries=batch.get("token_word_boundaries"),
                )
                if str_conv_ids and output.fused_embed is not None:
                    self.ctx_cache.batch_update(
                        str_conv_ids, output.fused_embed, turn_indices=turn_indices
                    )
                all_probs.append(output.probs_type.float().cpu().numpy())
                all_labels.append(batch["conflict_type_labels"].cpu().numpy())
                all_binary.append(batch["conflict_binary"].cpu().numpy())

            probs = np.concatenate(all_probs)    # (N, n_classes)
            labels = np.concatenate(all_labels)  # (N, n_classes)
            binary = np.concatenate(all_binary)  # (N,)

            # --- Binary conflict F1 / AUC (dataset-agnostic) ---
            # Max probability across conflict emotion slots (anger, disgust, fear = indices 0,1,2)
            conflict_prob = probs[:, :3].max(axis=1)
            bin_pred = (conflict_prob > 0.5).astype(int)
            binary_int = binary.astype(int)
            f1_binary = f1_score(binary_int, bin_pred, zero_division=0)
            try:
                auc_binary = roc_auc_score(binary_int, conflict_prob)
            except ValueError:
                auc_binary = 0.5  # degenerate split with only one class present

            # --- Per-class AP, averaged over classes that have at least one positive ---
            per_class_ap = []
            for c in range(labels.shape[1]):
                if labels[:, c].sum() > 0:
                    try:
                        per_class_ap.append(average_precision_score(labels[:, c], probs[:, c]))
                    except ValueError:
                        pass
            macro_ap = float(np.mean(per_class_ap)) if per_class_ap else 0.0

            # Weighted (macro) F1 over all classes — used for best-checkpoint
            # and early-stopping.  Using binary_f1 here was incorrect: it
            # collapsed the multi-class signal to a single conflict/non-conflict
            # decision and ignored per-emotion class performance.
            from sklearn.metrics import f1_score as _f1
            f1_macro = _f1(labels, (probs >= 0.5).astype(int), average="macro", zero_division=0)

            metrics = {
                "val/f1_binary": float(f1_binary),
                "val/auc_binary": float(auc_binary),
                "val/macro_ap": float(macro_ap),
                "val/f1_macro": float(f1_macro),
                # val/f1_weighted drives best-checkpoint and early-stopping logic.
                # BUG FIX: was incorrectly set to f1_binary (binary conflict F1)
                # which ignores per-emotion class accuracy. Now uses macro F1.
                "val/f1_weighted": float(f1_macro),
            }
            is_ddp = torch.distributed.is_initialized() and int(os.environ.get("LOCAL_RANK", -1)) != -1
            if is_ddp:
                metric_keys = sorted(list(metrics.keys()))
                metric_values = torch.tensor([metrics[key] for key in metric_keys], device=self.device)
                torch.distributed.broadcast(metric_values, src=0)
                if int(os.environ.get("LOCAL_RANK", -1)) != 0:
                    metrics = {k: v.item() for k, v in zip(metric_keys, metric_values)}
            return metrics

        except Exception:
            # BUG FIX: broadcast a -1 sentinel to unblock non-zero DDP ranks that
            # are blocked on torch.distributed.broadcast() before re-raising.
            # Without this, a rank-0 OOM / exception causes a permanent deadlock.
            if is_ddp:
                sentinel = torch.full((len(metric_keys),), -1.0, device=self.device)
                torch.distributed.broadcast(sentinel, src=0)
            raise
        finally:
            # BUG FIX: always restore training mode. evaluate() called model.eval()
            # but never restored model.train(), leaving the model in eval mode if
            # evaluate() is called outside the train() loop (e.g. from a script).
            self.model.train()


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

            train_metrics = self.train_epoch(epoch, pretraining=is_pretrain)
            val_metrics = self.evaluate()

            all_metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}
            if self.use_wandb:
                import wandb  # type: ignore
                wandb.log(all_metrics, step=self.global_step)

            is_rank_zero = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

            # Checkpoints must have one writer. Concurrent rank writes can
            # produce a weights file and training-state file from different
            # steps, making resume silently incorrect.
            if is_rank_zero:
                self._save_checkpoint(epoch, is_latest=True)

            val_f1 = val_metrics.get("val/f1_weighted", 0.0)
            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
                if is_rank_zero:
                    self._save_checkpoint(epoch, is_best=True)

            # Clean, compact 1-line epoch summary
            tag = " ⭐ (New Best)" if is_best else ""
            if is_rank_zero:
                logger.info(
                    f"[Epoch {epoch+1:02d}/{n_epochs:02d}] {phase.upper():<8} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val F1: {val_f1:.4f}{tag}"
                )

            # Early stopping check (only during finetune phase)
            if not is_pretrain:
                if torch.distributed.is_initialized():
                    val_f1_t = torch.tensor([val_f1], dtype=torch.float, device=self.device)
                    torch.distributed.all_reduce(val_f1_t, op=torch.distributed.ReduceOp.AVG)
                    val_f1 = val_f1_t.item()

                if val_f1 > self._best_val_f1:
                    self._best_val_f1 = val_f1
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1
                    if self._patience_counter >= early_stop_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1} (patience={early_stop_patience})")
                        break

        logger.info(f"✅ Training complete. Best val F1 = {self.best_val_f1:.4f}")

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

    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_latest: bool = False):
        """Save checkpoint using safetensors for model weights (pickle-free).

        Saves:
          - ``*.safetensors``: model weights only (safe, no arbitrary code exec)
          - ``*_training_state.pt``: optimizer + scheduler state (torch.save,
            only loaded by our own ``load_checkpoint``; NOT user-facing)
          - ``*_meta.json``: scalar metadata (epoch, step, F1)
        """
        import json as _json

        if is_best:
            prefix = "best_model"
        elif is_latest:
            prefix = "latest_model"
        else:
            prefix = f"checkpoint_epoch{epoch + 1}"

        _safe_save = getattr(torch, "save")

        # 1. Model weights → safetensors (primary, pickle-free)
        # Save the underlying module so a checkpoint made by DDP can be loaded
        # by single-GPU evaluation/serving without every key being prefixed with
        # ``module.``.
        model_for_state = self.model.module if isinstance(
            self.model, nn.parallel.DistributedDataParallel
        ) else self.model
        model_state = model_for_state.state_dict()
        try:
            from safetensors.torch import save_file as st_save
            st_path = self.output_dir / f"{prefix}.safetensors"
            st_save(model_state, str(st_path))
        except ImportError:
            # Fallback: torch.save model weights only (still state_dict, not full model)
            logger.warning("[Checkpoint] safetensors not installed — falling back to torch.save for model weights")
            _safe_save(model_state, self.output_dir / f"{prefix}.pt")  # nosec

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

        model_for_state = self.model.module if isinstance(
            self.model, nn.parallel.DistributedDataParallel
        ) else self.model

        def _load_model_state(state: Dict[str, Any]):
            # Support historical DDP checkpoints while writing new checkpoints
            # in the portable, unwrapped format above.
            if state and all(key.startswith("module.") for key in state):
                state = {key.removeprefix("module."): value for key, value in state.items()}
            return model_for_state.load_state_dict(state, strict=False)

        if checkpoint_path.suffix == ".safetensors":
            # Safe path: safetensors model weights + sidecar files
            model_state = load_checkpoint_state(checkpoint_path, device=self.device)

            result = _load_model_state(model_state)
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
            result = _load_model_state(model_state)
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
