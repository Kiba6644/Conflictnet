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
import numpy as np
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
            
            # Load per-class thresholds from meta if available
            calib = meta.get("calibration", {})
            if "per_class_thresholds" in calib:
                self.per_class_thresholds = np.array(calib["per_class_thresholds"])
                logger.info("Loaded per-class thresholds from meta.")

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

    def build_faiss_index(self, embeddings: np.ndarray, labels: np.ndarray):
        """Build FAISS index for retrieval-augmented detection."""
        try:
            import faiss
            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings)
            self.faiss_labels = labels
            logger.info(f"Built FAISS index with {len(embeddings)} samples.")
        except ImportError:
            logger.warning("FAISS not installed, skipping index building.")
            self.faiss_index = None

    def _get_tta_variants(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Generate test-time augmentation variants of the audio."""
        variants = [waveform]
        # Pitch shift up
        try:
            variants.append(AF.pitch_shift(waveform, self.SAMPLE_RATE, 2))
        except Exception:
            pass
        # Pitch shift down
        try:
            variants.append(AF.pitch_shift(waveform, self.SAMPLE_RATE, -2))
        except Exception:
            pass
        # Add noise
        noise = torch.randn_like(waveform) * 0.005
        variants.append(waveform + noise)
        # Scale volume
        variants.append(waveform * 0.8)
        return variants[:5]

    @torch.no_grad()
    def predict(
        self,
        audio_bytes: bytes,
        text: str,
        context_embeds: Optional[List[List[float]]] = None,
        prosody_z: Optional[List[float]] = None,
        use_tta: bool = False,
        use_retrieval: bool = False,
    ) -> Dict[str, Any]:
        assert self.model is not None and self.tokenizer is not None, "Model not loaded — call load()"

        base_waveform = self._load_audio(audio_bytes)
        input_ids, attn_mask = self._tokenize(text)

        waveforms = self._get_tta_variants(base_waveform) if use_tta else [base_waveform]
        
        all_probs = []
        all_severities = []
        all_fused_embeds = []

        for waveform in waveforms:
            waveform_t = waveform.unsqueeze(0).to(self.device)
            input_ids_t = input_ids.unsqueeze(0).to(self.device)
            attn_mask_t = attn_mask.unsqueeze(0).to(self.device)

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
                    audio=waveform_t,
                    input_ids=input_ids_t,
                    attention_mask=attn_mask_t,
                    prosody_z=pz,
                    context_embeds=ctx,
                    context_padding=ctx_pad,
                )

            probs = output.probs_type.squeeze(0).cpu().numpy()
            all_probs.append(probs)
            
            if output.severity is not None:
                all_severities.append(output.severity.squeeze(0).cpu().item())
                
            all_fused_embeds.append(output.fused_embed.squeeze(0).cpu().numpy())

        # Average over TTA variants
        avg_probs = np.mean(all_probs, axis=0)
        avg_severity = np.mean(all_severities) if all_severities else 0.0
        avg_fused_embed = np.mean(all_fused_embeds, axis=0)
        
        # Retrieval augmentation
        if use_retrieval and getattr(self, "faiss_index", None) is not None:
            import faiss
            query_emb = avg_fused_embed.reshape(1, -1).copy()
            faiss.normalize_L2(query_emb)
            distances, indices = self.faiss_index.search(query_emb, k=5)
            # Soft pseudo-labels (average of top-K)
            retrieved_labels = self.faiss_labels[indices[0]]
            weights = np.exp(distances[0]) # simple weighting
            weights = weights / np.sum(weights)
            soft_labels = np.sum(retrieved_labels * weights[:, None], axis=0)
            
            # Augment original probabilities (e.g., 0.8 * original + 0.2 * retrieval)
            avg_probs = 0.8 * avg_probs + 0.2 * soft_labels

        type_names = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]
        
        # Get threshold if calibration is available (for simplicity we use 0.5 or max)
        thresholds = getattr(self, "per_class_thresholds", np.ones(6) * 0.5)
        
        # Check if any probability exceeds its threshold
        conflict = bool((avg_probs >= thresholds).any())
        pred_type = type_names[np.argmax(avg_probs)] if conflict else "none"

        return {
            "conflict": conflict,
            "probs": {
                "anger": float(avg_probs[0]),
                "disgust": float(avg_probs[1]),
                "fear": float(avg_probs[2]),
                "happiness": float(avg_probs[3]),
                "neutral": float(avg_probs[4]),
                "sadness": float(avg_probs[5]),
            },
            "severity": float(avg_severity),
            "predicted_type": pred_type,
            "fused_embed": avg_fused_embed.tolist(),
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
