"""Text encoder: DeBERTa-v3-large with optional LoRA.

Falls back to a simple embedding + transformer when HuggingFace is unavailable.
"""

import logging
import os
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DeBERTaEncoder(nn.Module):
    """DeBERTa-v3 text encoder with optional LoRA fine-tuning.

    Args:
        gradient_checkpointing: Enable gradient checkpointing to reduce VRAM ~25%
            at cost of ~15% extra compute. Recommended for 2x T4 training.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        vocab_size: int = 128000,
        embed_dim: int = 1024,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        local_path = os.environ.get("CONFLICTNET_DEBERTA_PATH")
        if local_path:
            model_name = local_path
            logger.info(f"[DeBERTa] Using local path: {local_path}")
        self.output_dim = embed_dim
        self._use_lora = use_lora
        self._lora_r = lora_r
        self._lora_alpha = lora_alpha
        self.encoder = None
        self.tokenizer = None

        try:
            from transformers import AutoModel, AutoTokenizer
            self.encoder = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.output_dim = self.encoder.config.hidden_size
            # Gradient checkpointing: reduces peak VRAM ~25% on T4 (helpful for DDP)
            if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
                self.encoder.gradient_checkpointing_enable()
                logger.info("[DeBERTa] Gradient checkpointing enabled")
            if use_lora:
                self._apply_lora(lora_r, lora_alpha)
            return
        except Exception as e:
            logger.warning(
                f"DeBERTa model {model_name} unavailable ({e}), using fallback text encoder"
            )

        self.encoder = _TextFallbackEncoder(vocab_size, embed_dim)
        self.tokenizer = None  # tokenizer provided externally

    def _apply_lora(self, r: int, alpha: int):
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=r,
                lora_alpha=alpha,
                target_modules=["query_proj", "value_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, config)
            trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.encoder.parameters())
            logger.info(f"[LoRA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        except Exception as e:
            logger.warning(
                f"[WARN] peft unavailable or incompatible ({e}). "
                f"Ensure peft==0.14.0 is installed in your setup script. Freezing DeBERTa parameters."
            )
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_tokens: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.encoder, _TextFallbackEncoder):
            return self.encoder(input_ids, attention_mask, return_tokens)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :].float()
        if return_tokens:
            return pooled, out.last_hidden_state.float()
        return pooled


class _TextFallbackEncoder(nn.Module):
    """Simple embedding + transformer fallback (no HuggingFace needed)."""

    def __init__(self, vocab_size: int = 128000, embed_dim: int = 1024):
        super().__init__()
        self.output_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = nn.Parameter(torch.randn(1, 512, embed_dim) * 0.02)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4,
                dropout=0.1, activation="gelu", batch_first=True,
            ),
            num_layers=6,
        )

    def forward(self, input_ids, attention_mask=None, return_tokens=False):
        x = self.embed(input_ids)
        seq_len = x.size(1)
        x = x + self.pos_enc[:, :seq_len, :]
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.transformer(x)
        pooled = x[:, 0, :]
        if return_tokens:
            return pooled, x
        return pooled