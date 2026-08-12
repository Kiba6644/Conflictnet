"""Audio encoders: Emotion2Vec, WavLM, wav2vec2.

Each returns (batch, encoder_dim) from raw waveform input.
All encoders accept an optional attention_mask for padded audio.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, WavLMModel

logger = logging.getLogger(__name__)


class Wav2Vec2Encoder(nn.Module):
    """Baseline audio encoder from HuBERT-CLAP."""

    def __init__(self, model_name: str = "facebook/wav2vec2-large-960h", freeze: bool = True):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.output_dim: int = self.encoder.config.hidden_size  # 1024
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(
        self, audio: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
        return_frames: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(audio, attention_mask=attention_mask)
        hs = out.last_hidden_state  # (B, T, D)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hs.mean(dim=1)
        if return_frames:
            return pooled, hs
        return pooled


class WavLMEncoder(nn.Module):
    """WavLM — denoising-pretrained, strong on emotion + speaker."""

    def __init__(self, model_name: str = "microsoft/wavlm-large", freeze: bool = True):
        super().__init__()
        self.encoder = WavLMModel.from_pretrained(model_name)
        self.output_dim: int = self.encoder.config.hidden_size  # 1024
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(
        self, audio: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
        return_frames: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(audio, attention_mask=attention_mask)
        hs = out.last_hidden_state  # (B, T, D)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hs.mean(dim=1)
        if return_frames:
            return pooled, hs
        return pooled


class Emotion2VecEncoder(nn.Module):
    """Emotion2Vec / Emotion2Vec+ — emotion-specific self-supervised encoder.

    Supports two backends:
      1. funasr (recommended) — iic/emotion2vec_plus_large
      2. transformers — falls back to WavLM if funasr unavailable

    The funasr API varies by model version. This implementation handles
    both funasr >= 0.5 and earlier versions.
    """

    def __init__(
        self,
        model_name: str = "iic/emotion2vec_plus_large",
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = 768
        self._freeze = freeze
        self._model = None
        self._backend = None
        self._try_init_funasr()
        # Verify output dimension at runtime and log a warning if mismatch
        self._verify_output_dim()

    def _try_init_funasr(self):
        try:
            from funasr import AutoModel
            self._model = AutoModel(
                model=self.model_name,
                disable_update=True,
                disable_pipeline=True,
            )
            self._backend = "funasr"
            logger.info(f"[Emotion2Vec] Loaded funasr backend: {self.model_name}")
        except Exception as e:
            logger.warning(f"[Emotion2Vec] funasr init failed ({e}), falling back to WavLM")
            self._model = WavLMEncoder(freeze=self._freeze)
            self._backend = "fallback"
            self.output_dim = self._model.output_dim

    def forward(
        self, audio: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
        return_frames: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert self._model is not None, "self._model is not initialized"
        if self._backend == "funasr":
            pooled = self._forward_funasr(audio)
            if return_frames:
                return pooled, None
            return pooled
        result = self._model(audio, attention_mask=attention_mask, return_frames=return_frames)  # type: ignore[operator]
        if return_frames:
            return result  # already (pooled, frames) tuple
        return result

    def _forward_funasr(self, audio: torch.Tensor) -> torch.Tensor:
        # Convert (B, T) torch tensor to list of numpy arrays
        audio_np_list = [wav.cpu().numpy().astype(np.float32) for wav in audio]

        # funasr AutoModel.generate returns list of dicts
        # Key depends on model config: 'feats', 'feature', 'embeddings', etc.
        assert self._model is not None, "self._model is not initialized"
        results = self._model.generate(  # type: ignore[union-attr]
            audio_np_list,
            output_dir=False,
            granularity="utterance",
        )

        feats_list = []
        for r in results:
            feat = self._extract_feature(r)
            feats_list.append(feat)

        return torch.stack(feats_list, dim=0)

    def _verify_output_dim(self):
        """Log actual output dimension from the loaded model for verification."""
        if self._backend == "fallback":
            return
        try:
            with torch.no_grad():
                dummy = torch.randn(1, 16000)
                out = self._forward_funasr(dummy)
                actual_dim = out.shape[-1]
                if actual_dim != self.output_dim:
                    logger.warning(
                        f"[Emotion2Vec] Model {self.model_name} has output_dim={actual_dim}, "
                        f"but hardcoded output_dim={self.output_dim}. Update the default."
                    )
                else:
                    logger.info(f"[Emotion2Vec] Verified output_dim={actual_dim} for {self.model_name}")
        except Exception as e:
            logger.debug(f"[Emotion2Vec] Could not verify output dim: {e}")

    def _extract_feature(self, result: dict) -> torch.Tensor:
        """Extract pooled feature from a funasr result dict."""
        # Try known keys in order of likelihood
        for key in ("feats", "feature", "embeddings", "embedding", "hidden_states"):
            val = result.get(key)
            if val is not None:
                tensor = torch.tensor(val, dtype=torch.float32)
                if tensor.dim() > 1:
                    tensor = tensor.mean(dim=0)
                elif tensor.dim() == 0:
                    tensor = tensor.unsqueeze(0)
                return tensor

        # Last resort: stack all non-None numeric values
        logger.warning(f"[Emotion2Vec] Unknown result keys: {list(result.keys())}")
        tensors = []
        for v in result.values():
            if isinstance(v, (list, np.ndarray)):
                t = torch.tensor(v, dtype=torch.float32)
                if t.numel() > 0:
                    tensors.append(t.mean())
        if tensors:
            return torch.stack(tensors)
        return torch.zeros(self.output_dim)

    @property
    def device(self) -> torch.device:
        if self._backend == "funasr" and self._model is not None and hasattr(self._model, "device"):
            return self._model.device  # type: ignore[union-attr]
        assert self._model is not None, "self._model is not initialized"
        return next(self._model.parameters()).device if self._backend == "fallback" else torch.device("cpu")  # type: ignore[union-attr]


def build_audio_encoder(name: str = "emotion2vec", **kwargs) -> nn.Module:
    """Factory function.

    Args:
        name: 'emotion2vec' | 'wavlm' | 'wav2vec2'
        **kwargs: Passed to the encoder constructor (e.g. freeze=True).
    """
    encoders = {
        "wav2vec2": Wav2Vec2Encoder,
        "wavlm": WavLMEncoder,
        "emotion2vec": Emotion2VecEncoder,
    }
    if name not in encoders:
        raise ValueError(f"Unknown audio encoder: {name}. Choose from {list(encoders.keys())}")
    return encoders[name](**kwargs)
