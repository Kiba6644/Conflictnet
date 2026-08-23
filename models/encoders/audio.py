"""Audio encoders: Emotion2Vec, WavLM, wav2vec2.

Each returns (batch, encoder_dim) from raw waveform input.
All encoders accept an optional attention_mask for padded audio.
When HuggingFace models are unavailable (e.g. Kaggle without internet),
falls back to a simple spectrogram-based encoder.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from transformers.utils import logging as tf_logging
    tf_logging.set_verbosity_error()
    tf_logging.disable_progress_bar()
except Exception:
    pass


class _SpectrogramEncoder(nn.Module):
    """Simple spectrogram + CNN audio encoder fallback (no external models needed)."""

    def __init__(self, output_dim: int = 1024):
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("hann", torch.hann_window(512))
        self.conv = nn.Sequential(
            nn.Conv1d(257, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(256),
            nn.Conv1d(256, output_dim, kernel_size=1),
        )

    def forward(self, audio, attention_mask=None, return_frames=False):
        # BUG FIX: torch.stft does not support bfloat16 (raised under AMP).
        # Cast both audio and the Hann window to float32 explicitly.
        spec = torch.stft(audio.float(), n_fft=512, hop_length=160, window=self.hann.float(),
                          return_complex=True).abs()
        conv_out = self.conv(spec)
        pooled = conv_out.mean(dim=-1)
        if return_frames:
            frames = conv_out.permute(0, 2, 1)
            return pooled, frames
        return pooled


class Wav2Vec2Encoder(nn.Module):
    """Baseline audio encoder from HuBERT-CLAP, falls back to spectrogram."""

    def __init__(self, model_name: str = "facebook/wav2vec2-large-960h", freeze: bool = True):
        super().__init__()
        self._encoder = None
        self.output_dim = 768
        try:
            from transformers import Wav2Vec2Model
            enc = Wav2Vec2Model.from_pretrained(model_name)
            if freeze:
                for p in enc.parameters():
                    p.requires_grad = False
            self._encoder = enc
            self.output_dim = enc.config.hidden_size
            logger.info(f"Loaded Wav2Vec2: {model_name} (dim={self.output_dim})")
        except Exception as e:
            logger.warning(f"Wav2Vec2 unavailable ({e}), using spectrogram fallback")
            self._encoder = _SpectrogramEncoder(output_dim=self.output_dim)

    def forward(self, audio, attention_mask=None, return_frames=False):
        if isinstance(self._encoder, _SpectrogramEncoder):
            return self._encoder(audio, attention_mask, return_frames)
        out = self._encoder(audio, attention_mask=attention_mask)
        hs = out.last_hidden_state.float()
        if attention_mask is not None:
            feat_lengths = self._encoder._get_feat_extract_output_lengths(
                attention_mask.sum(dim=1)
            )
            max_time = hs.size(1)
            feat_mask = torch.arange(max_time, device=hs.device).unsqueeze(0) < feat_lengths.unsqueeze(1)
            feat_mask = feat_mask.unsqueeze(-1).float()
            pooled = (hs * feat_mask).sum(dim=1) / feat_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hs.mean(dim=1)
        if return_frames:
            return pooled, hs
        return pooled


class WavLMEncoder(nn.Module):
    """WavLM, falls back to spectrogram.

    Uses only the last hidden state. For better emotion accuracy use
    WavLMWeightedEncoder which learns a weighted average over all layers.
    """

    def __init__(self, model_name: str = "microsoft/wavlm-large", freeze: bool = True,
                 gradient_checkpointing: bool = False):
        super().__init__()
        local_path = os.environ.get("CONFLICTNET_WAVLM_PATH")
        if local_path:
            model_name = local_path
            logger.info(f"[WavLM] Using local path: {local_path}")
        self._encoder = None
        # microsoft/wavlm-large exposes 1024 features.  The old 768-d fallback
        # made a rank that missed the cached checkpoint construct a different
        # projection head from a rank that loaded it successfully.
        self.output_dim = 1024
        requested_backend = os.environ.get("CONFLICTNET_WAVLM_BACKEND", "auto")
        if requested_backend == "spectrogram":
            self._encoder = _SpectrogramEncoder(output_dim=self.output_dim)
            self._backend = "spectrogram"
            logger.info("[WavLM] Using DDP-selected spectrogram fallback")
            return
        try:
            from transformers import WavLMModel
            enc = WavLMModel.from_pretrained(model_name)
            if freeze:
                for p in enc.parameters():
                    p.requires_grad = False
            # Gradient checkpointing reduces VRAM ~30% at cost of ~15% extra compute
            if gradient_checkpointing and hasattr(enc, "gradient_checkpointing_enable"):
                enc.gradient_checkpointing_enable()
                logger.info("[WavLM] Gradient checkpointing enabled")
            self._encoder = enc
            self.output_dim = enc.config.hidden_size
            self._backend = "pretrained"
            logger.info(f"Loaded WavLM: {model_name} (dim={self.output_dim})")
        except Exception as e:
            if requested_backend == "pretrained":
                raise RuntimeError(
                    "Rank 0 selected the pretrained WavLM backend, but this rank "
                    f"could not load it: {e}"
                ) from e
            logger.warning(f"WavLM unavailable ({e}), using spectrogram fallback")
            self._encoder = _SpectrogramEncoder(output_dim=self.output_dim)
            self._backend = "spectrogram"

    def _masked_pool(self, hs: torch.Tensor, attention_mask=None) -> torch.Tensor:
        if attention_mask is not None:
            feat_lengths = self._encoder._get_feat_extract_output_lengths(
                attention_mask.sum(dim=1)
            )
            max_time = hs.size(1)
            feat_mask = torch.arange(max_time, device=hs.device).unsqueeze(0) < feat_lengths.unsqueeze(1)
            feat_mask = feat_mask.unsqueeze(-1).float()
            return (hs * feat_mask).sum(dim=1) / feat_mask.sum(dim=1).clamp(min=1)
        return hs.mean(dim=1)

    def forward(self, audio, attention_mask=None, return_frames=False):
        if isinstance(self._encoder, _SpectrogramEncoder):
            return self._encoder(audio, attention_mask, return_frames)
        out = self._encoder(audio, attention_mask=attention_mask)
        hs = out.last_hidden_state.float()
        pooled = self._masked_pool(hs, attention_mask)
        if return_frames:
            return pooled, hs
        return pooled


class WavLMWeightedEncoder(nn.Module):
    """WavLM with learnable weighted average over all hidden states.

    WavLM-large has 24 transformer layers + 1 embedding layer = 25 states.
    Emotion discriminability peaks in layers 8-12 (shown in probing studies).
    Learning a softmax-weighted combination over all layers consistently
    outperforms last-layer pooling by ~2-4% weighted F1 on emotion tasks.

    The layer weights are *trainable* even when the WavLM backbone is frozen,
    so this adds only 25 parameters while providing a significant accuracy boost.

    Args:
        model_name: HuggingFace model ID or local path.
        freeze: If True, freeze WavLM backbone weights (only layer_weights train).
        gradient_checkpointing: Enable gradient checkpointing on WavLM to save VRAM.
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        freeze: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        local_path = os.environ.get("CONFLICTNET_WAVLM_PATH")
        if local_path:
            model_name = local_path
            logger.info(f"[WavLMWeighted] Using local path: {local_path}")
        self._encoder = None
        self.output_dim = 1024  # WavLM-large hidden size
        # BUG FIX: _n_layers used to be set to 25 AND then nn.Parameter was
        # created, then inside the try block it was set again from enc.config
        # and another nn.Parameter was created. That double-init is fragile
        # (different state_dict shape if the try block fails mid-way).
        # Instead, set _n_layers once here as a default, update it if WavLM
        # loads, then create layer_weights exactly once after the try/except.
        self._n_layers = 25    # WavLM-large default: 1 embedding + 24 layers

        try:
            from transformers import WavLMModel
            enc = WavLMModel.from_pretrained(
                model_name,
                output_hidden_states=True,
            )
            if freeze:
                for p in enc.parameters():
                    p.requires_grad = False
            if gradient_checkpointing and hasattr(enc, "gradient_checkpointing_enable"):
                enc.gradient_checkpointing_enable()
                logger.info("[WavLMWeighted] Gradient checkpointing enabled")
            self._encoder = enc
            self.output_dim = enc.config.hidden_size
            self._n_layers = enc.config.num_hidden_layers + 1  # +1 for embedding layer
            logger.info(
                f"[WavLMWeighted] Loaded {model_name} (dim={self.output_dim}, "
                f"n_layers={self._n_layers}, layer_weights=trainable)"
            )
        except Exception as e:
            logger.warning(f"[WavLMWeighted] WavLM unavailable ({e}), using spectrogram fallback")
            self._encoder = _SpectrogramEncoder(output_dim=self.output_dim)

        # Learnable scalar per hidden state — created once with the final layer count
        self.layer_weights = nn.Parameter(torch.zeros(self._n_layers))

    def _masked_pool(self, hs: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Masked mean pooling over time dimension."""
        if attention_mask is not None and not isinstance(self._encoder, _SpectrogramEncoder):
            feat_lengths = self._encoder._get_feat_extract_output_lengths(
                attention_mask.sum(dim=1)
            )
            max_time = hs.size(1)
            feat_mask = torch.arange(max_time, device=hs.device).unsqueeze(0) < feat_lengths.unsqueeze(1)
            feat_mask = feat_mask.unsqueeze(-1).float()
            return (hs * feat_mask).sum(dim=1) / feat_mask.sum(dim=1).clamp(min=1)
        return hs.mean(dim=1)

    def forward(self, audio, attention_mask=None, return_frames=False):
        if isinstance(self._encoder, _SpectrogramEncoder):
            return self._encoder(audio, attention_mask, return_frames)

        out = self._encoder(
            audio,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # hidden_states: tuple of (B, T, D) — one per layer including embedding
        hidden_states = out.hidden_states  # tuple of n_layers tensors

        # Stack: (n_layers, B, T, D)
        stacked = torch.stack([h.float() for h in hidden_states], dim=0)

        # Softmax-weighted sum over layers: (B, T, D)
        weights = F.softmax(self.layer_weights, dim=0)  # (n_layers,)
        weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)  # (B, T, D)

        pooled = self._masked_pool(weighted, attention_mask)  # (B, D)

        if return_frames:
            return pooled, weighted
        return pooled


class Emotion2VecEncoder(nn.Module):
    """Emotion2Vec via FunASR, with WavLM only as an actual fallback.

    Do not initialise WavLM before FunASR.  Emotion2Vec uses the FunASR output
    when available, so eagerly constructing an otherwise-unused WavLM added a
    second pretrained download and could give DDP ranks different registered
    parameter sets when only one rank fell back to a spectrogram encoder.
    """

    def __init__(
        self,
        model_name: str = "iic/emotion2vec_plus_large",
        freeze: bool = True,
        embedding_dim: int = 1024,
    ):
        super().__init__()
        local_path = os.environ.get("CONFLICTNET_EMOTION2VEC_PATH")
        if local_path:
            model_name = local_path
            logger.info(f"[Emotion2Vec] Using local path: {local_path}")
            
        self.model_name = model_name
        self._freeze = freeze
        self.output_dim = embedding_dim
        self._model = None
        self._funasr_wrapper = []
        requested_backend = os.environ.get("CONFLICTNET_EMOTION2VEC_BACKEND", "auto")
        self._backend = "funasr"
        if requested_backend == "fallback_wavlm":
            self._model = WavLMEncoder(freeze=freeze)
            self.proj = nn.Linear(self._model.output_dim, embedding_dim) if self._model.output_dim != embedding_dim else nn.Identity()
            self.output_dim = embedding_dim
            self._backend = "fallback_wavlm"
            logger.info("[Emotion2Vec] Using DDP-selected WavLM fallback")
            return
        try:
            self._try_funasr()
        except Exception as e:
            if requested_backend == "funasr":
                raise RuntimeError(
                    "Rank 0 selected the FunASR Emotion2Vec backend, but this rank "
                    f"could not load it: {e}"
                ) from e
            logger.warning(
                f"[Emotion2Vec] FunASR unavailable ({e}); using WavLM fallback"
            )
            self._model = WavLMEncoder(freeze=freeze)
            self.proj = nn.Linear(self._model.output_dim, embedding_dim) if self._model.output_dim != embedding_dim else nn.Identity()
            self.output_dim = embedding_dim
            self._backend = "fallback_wavlm"

    def _try_funasr(self):
        from funasr import AutoModel
        
        device_str = "cpu"
        if torch.cuda.is_available():
            # Explicitly bind to the current local rank device, otherwise FunASR 
            # defaults to cuda:0 for all ranks, causing massive GPU deadlocks.
            device_str = f"cuda:{torch.cuda.current_device()}"
            
        funasr_model = AutoModel(
            model=self.model_name,
            device=device_str,
            disable_update=True,
            disable_pipeline=True,
            disable_pbar=True,
            disable_log=True,
        )
        self._funasr_wrapper = [funasr_model]
        self._backend = "funasr"
        logger.info(f"[Emotion2Vec] funasr backend: {self.model_name} on {device_str}")

    def forward(self, audio, attention_mask=None, return_frames=False):
        if self._backend == "funasr":
            pooled = self._forward_funasr(audio)
            if return_frames:
                return pooled, None
            return pooled
        out = self._model(audio, attention_mask=attention_mask, return_frames=return_frames)
        if return_frames:
            pooled, frames = out
            return self.proj(pooled), frames
        return self.proj(out)

    def _forward_funasr(self, audio):
        audio_np = audio.cpu().numpy()
        results = []
        for i in range(audio_np.shape[0]):
            emb = self._funasr_wrapper[0].generate(
                input=audio_np[i], 
                output_dir=None,  # Prevent DDP file I/O race conditions
                disable_pbar=True,
                disable_log=True
            )
            if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], dict):
                emb = emb[0].get("feats", emb[0])
            emb = np.mean(emb, axis=0) if emb.ndim > 1 else emb
            results.append(torch.from_numpy(emb).float())
        return torch.stack(results).to(audio.device)

    @property
    def device(self) -> torch.device:
        if self._backend == "funasr":
            # FunASR accepts NumPy input and its model is deliberately kept out
            # of nn.Module registration, so the caller's input device is used.
            return torch.device("cpu")
        if self._model is None:
            return torch.device("cpu")
        return next(self._model.parameters()).device


class WhisperEncoder(nn.Module):
    """Whisper audio encoder (openai/whisper-large-v3)."""

    def __init__(self, model_name: str = "openai/whisper-large-v3", freeze: bool = True):
        super().__init__()
        self.output_dim = 1280
        try:
            from transformers import WhisperModel, WhisperFeatureExtractor
            self._feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
            enc = WhisperModel.from_pretrained(model_name).encoder
            if freeze:
                for p in enc.parameters():
                    p.requires_grad = False
            self._encoder = enc
            self.output_dim = enc.config.d_model
            logger.info(f"Loaded Whisper: {model_name} (dim={self.output_dim})")
        except Exception as e:
            logger.warning(f"Whisper unavailable ({e}), using spectrogram fallback")
            self._encoder = _SpectrogramEncoder(output_dim=self.output_dim)
            self._feature_extractor = None

    def forward(self, audio, attention_mask=None, return_frames=False):
        if isinstance(self._encoder, _SpectrogramEncoder):
            return self._encoder(audio, attention_mask, return_frames)
        
        device = audio.device
        audio_np = audio.cpu().numpy()
        features = self._feature_extractor(audio_np, sampling_rate=16000, return_tensors="pt").input_features
        features = features.to(device)
        
        out = self._encoder(features)
        hs = out.last_hidden_state.float()
        
        pooled = hs.mean(dim=1)
        if return_frames:
            return pooled, hs
        return pooled


class DualAudioEncoder(nn.Module):
    """Dual Audio Encoder Fusion (Emotion2Vec + WavLM)."""

    def __init__(self, freeze: bool = True, output_dim: int = 1024):
        super().__init__()
        self.emotion2vec = Emotion2VecEncoder(freeze=freeze)
        self.wavlm = WavLMEncoder(freeze=freeze)
        in_dim = self.emotion2vec.output_dim + self.wavlm.output_dim
        self.output_dim = output_dim
        self.proj = nn.Linear(in_dim, output_dim)

    def forward(self, audio, attention_mask=None, return_frames=False):
        out1 = self.emotion2vec(audio, attention_mask, return_frames=return_frames)
        out2 = self.wavlm(audio, attention_mask, return_frames=return_frames)
        
        if return_frames:
            pool1, frame1 = out1
            pool2, frame2 = out2
            pool = torch.cat([pool1, pool2], dim=-1)
            pool = self.proj(pool)
            return pool, None
        
        pool = torch.cat([out1, out2], dim=-1)
        return self.proj(pool)


def build_audio_encoder(name: str = "emotion2vec", **kwargs) -> nn.Module:
    encoders = {
        "wav2vec2": Wav2Vec2Encoder,
        "wavlm": WavLMEncoder,
        "wavlm_weighted": WavLMWeightedEncoder,
        "emotion2vec": Emotion2VecEncoder,
        "whisper": WhisperEncoder,
        "dual": DualAudioEncoder,
    }
    if name not in encoders:
        raise ValueError(f"Unknown audio encoder: {name}. Choose from {list(encoders.keys())}")
    return encoders[name](**kwargs)
