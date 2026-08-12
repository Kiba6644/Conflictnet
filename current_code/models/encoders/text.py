"""Text encoder: DeBERTa-v3-large with optional LoRA."""

from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class DeBERTaEncoder(nn.Module):
    """DeBERTa-v3 text encoder with optional LoRA fine-tuning."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.output_dim = self.encoder.config.hidden_size  # 1024 for large

        if use_lora:
            self._apply_lora(lora_r, lora_alpha)

    def _apply_lora(self, r: int, alpha: int):
        try:
            from peft import LoraConfig, get_peft_model
            config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=["query_proj", "value_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, config)  # type: ignore[arg-type]
            trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.encoder.parameters())
            print(f"[LoRA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        except ImportError:
            print("[WARN] peft not installed, full fine-tuning")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_tokens: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]  # (B, hidden_size) [CLS]
        if return_tokens:
            return pooled, out.last_hidden_state  # (B, hidden_size), (B, L, hidden_size)
        return pooled
