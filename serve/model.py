"""Model wrapper for serving — loads checkpoint, runs predictions."""

from __future__ import annotations

import io
import json as _json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
import torchaudio.functional as AF
from transformers import AutoTokenizer

from models.checkpoint_utils import load_checkpoint_state, extract_model_state
from models.conflictnet import ConflictNet

logger = logging.getLogger(__name__)


class ServeModel:
    """Thread-safe model wrapper for inference.

    Loads the model once, keeps a tokenizer, and exposes a clean
    ``predict()`` method that accepts raw audio + text.
    """

    SAMPLE_RATE = 16000
    MAX_AUDIO_LEN = 10.0

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
        self.model: Optional[ConflictNet] = None
        self.tokenizer: Optional[AutoTokenizer] = None

    def load(self) -> None:
        ckpt_path = Path(self.cfg.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        logger.info(f"Loading checkpoint from {ckpt_path}")
        state = load_checkpoint_state(ckpt_path, device=str(self.device))
        model_state = extract_model_state(state)

        logger.info("Building model")

        # Warn if serving config differs from training config (B1 fix)
        meta_path = ckpt_path.parent / f"{ckpt_path.stem}_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = _json.load(f)
            exp_cfg = meta.get("experiment_config", {})
            for key, val in [("audio_encoder", self.cfg.audio_encoder),
                             ("embed_dim", self.cfg.embed_dim),
                             ("lora_r", self.cfg.lora_r)]:
                trained_val = exp_cfg.get(key)
                if trained_val is not None and str(trained_val) != str(val):
                    logger.warning(f"Serving {key}={val} differs from training {key}={trained_val}")

        self.model = ConflictNet(
            audio_encoder_name=self.cfg.audio_encoder,
            embed_dim=self.cfg.embed_dim,
            use_speaker_norm=self.cfg.use_speaker_norm,
            use_temporal=self.cfg.use_temporal,
            use_word_divergence=self.cfg.use_word_divergence,
            use_cross_attn_injection=self.cfg.use_cross_attn_injection,
            use_speaker_adaptive_threshold=self.cfg.use_speaker_adaptive_threshold,
            use_baseline_subtract=self.cfg.use_baseline_subtract,
            lora_r=self.cfg.lora_r,
            temporal_max_turns=self.cfg.temporal_max_turns,
        )

        result = self.model.load_state_dict(model_state, strict=False)
        if result.missing_keys:
            logger.warning(f"Missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            logger.warning(f"Unexpected keys: {result.unexpected_keys}")

        self.model.to(self.device)
        self.model.eval()

        if getattr(self.cfg, "compile", False) and hasattr(torch, "compile"):
            logger.info("Compiling model for faster inference")
            self.model = torch.compile(self.model)

        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            logger.info(f"Using DataParallel across {torch.cuda.device_count()} GPUs for inference")
            self.model = torch.nn.DataParallel(self.model)

        logger.info("Loading tokenizer (microsoft/deberta-v3-large)")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

    @torch.no_grad()
    def predict(
        self,
        audio_bytes: bytes,
        text: str,
        context_embeds: Optional[List[List[float]]] = None,
        prosody_z: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        assert self.model is not None and self.tokenizer is not None, "Model not loaded — call load()"

        waveform = self._load_audio(audio_bytes)
        input_ids, attn_mask = self._tokenize(text)

        waveform = waveform.unsqueeze(0).to(self.device)
        input_ids = input_ids.unsqueeze(0).to(self.device)
        attn_mask = attn_mask.unsqueeze(0).to(self.device)

        ctx = None
        ctx_pad = None
        if context_embeds is not None:
            ctx = torch.tensor(context_embeds, dtype=torch.float, device=self.device).unsqueeze(0)
            ctx_pad = torch.zeros(1, ctx.size(1), dtype=torch.bool, device=self.device)

        pz = None
        if prosody_z is not None:
            pz = torch.tensor(prosody_z, dtype=torch.float, device=self.device).unsqueeze(0)

        use_amp = getattr(self.cfg, "amp", False) and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            output = self.model(
                audio=waveform,
                input_ids=input_ids,
                attention_mask=attn_mask,
                prosody_z=pz,
                context_embeds=ctx,
                context_padding=ctx_pad,
            )

        probs = output.probs_type.squeeze(0).cpu().tolist()
        conflict = bool(output.conflict_flag.squeeze(0).cpu().item())
        severity = float(output.severity.squeeze(0).cpu().item()) if output.severity is not None else 0.0

        type_names = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]
        pred_type = type_names[probs.index(max(probs))] if conflict else "none"

        fused_embed_list = output.fused_embed.squeeze(0).cpu().tolist()

        return {
            "conflict": conflict,
            "probs": {
                "anger": probs[0],
                "disgust": probs[1],
                "fear": probs[2],
                "happiness": probs[3],
                "neutral": probs[4],
                "sadness": probs[5],
            },
            "severity": severity,
            "predicted_type": pred_type,
            "fused_embed": fused_embed_list,
        }

    @torch.no_grad()
    def predict_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        assert self.model is not None and self.tokenizer is not None
        results: List[Dict[str, Any]] = []

        for item in items:
            result = self.predict(
                audio_bytes=item["audio"],
                text=item["text"],
                context_embeds=item.get("context_embeds"),
                prosody_z=item.get("prosody_z"),
            )
            results.append(result)

        return results

    def _load_audio(self, audio_bytes: bytes) -> torch.Tensor:
        stream = io.BytesIO(audio_bytes)
        waveform, sr = torchaudio.load(stream)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.SAMPLE_RATE:
            waveform = AF.resample(waveform, sr, self.SAMPLE_RATE)
        max_samples = int(self.MAX_AUDIO_LEN * self.SAMPLE_RATE)
        waveform = waveform[:, :max_samples]
        return waveform.squeeze(0)

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.tokenizer is not None
        enc = self.tokenizer(  # type: ignore[operator]
            text,
            max_length=self.cfg.max_text_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
