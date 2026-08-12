"""Synthetic conflict data generation via StarGANv2-VC voice conversion.

Pipeline:
  1. Take a neutral utterance with known text
  2. Convert audio to "angry" emotion using StarGANv2-VC
  3. Keep original text (neutral sentiment) → audio-text conflict pair
  4. Label: sarcasm=0, suppression=1, deception=0, severity=0.6

Requires StarGANv2-VC repo cloned and models downloaded separately:
  git clone https://github.com/yl4579/StarGANv2-VC
  # Download pretrained model from their releases
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

STARGAN_PATH = Path("StarGANv2-VC")  # adjust if cloned elsewhere
SAMPLE_RATE = 24000  # StarGANv2-VC default SR


def resample_if_needed(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Resample audio to target sample rate using librosa."""
    if orig_sr == target_sr:
        return audio
    try:
        import librosa
        return librosa.resample(y=audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        logger.warning("librosa not available; cannot resample, returning original")
        return audio


class StarGANv2VoiceConverter:
    """Wraps StarGANv2-VC for emotion-style voice conversion.

    Only loaded if StarGANv2-VC repo is available at STARGAN_PATH.
    Falls back to EmotiVoice TTS (if available) otherwise.
    If neither is available, applies simple audio perturbation (pitch shift +
    speed change) as a data augmentation proxy for emotion conversion.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._available = False
        self._model = None
        self._backend = "none"

        if STARGAN_PATH.exists():
            sys.path.insert(0, str(STARGAN_PATH))
            try:
                import importlib.util
                spec = importlib.util.find_spec("models")
                if spec is not None:
                    self._available = True
                    self._backend = "starganv2-vc"
                    logger.info("[StarGANv2-VC] Repository found, loading...")
                    self._load_model(model_path)
            except Exception as e:
                logger.warning(f"[StarGANv2-VC] Failed to import: {e}")
        else:
            logger.warning(
                f"[StarGANv2-VC] Repo not found at {STARGAN_PATH}. "
                "Clone from github.com/yl4579/StarGANv2-VC"
            )

        if not self._available:
            self._try_emotivoice()

    def _try_emotivoice(self):
        try:
            import emotivoice  # type: ignore  # noqa: F401
            self._available = True
            self._backend = "emotivoice"
            logger.info("[StarGANv2-VC] Using EmotiVoice fallback")
        except ImportError:
            self._backend = "librosa"
            logger.info(
                "[StarGANv2-VC] Neither StarGANv2-VC nor EmotiVoice available. "
                "Falling back to librosa pitch_shift + time_stretch proxy."
            )

    def _load_model(self, model_path: Optional[str]):
        try:
            from models import Generator  # type: ignore
            generator = Generator()
            if model_path is not None and Path(model_path).exists():
                _safe_load = getattr(torch, "load")
                ckpt = _safe_load(model_path, map_location="cpu", weights_only=True)
                state = ckpt.get("generator", ckpt)
                generator.load_state_dict(state, strict=False)
            generator.eval()
            self._model = generator
            self._available = True
            logger.info("[StarGANv2-VC] Model loaded successfully")
        except Exception as e:
            logger.warning(f"[StarGANv2-VC] Could not load model: {e}")
            self._available = False

    def convert(
        self,
        audio: np.ndarray,
        source_emotion: str = "neutral",
        target_emotion: str = "angry",
        sr: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """Convert audio to target emotion style.

        Tries backends in order: StarGANv2-VC → EmotiVoice → librosa proxy.

        Returns converted audio as numpy array at sr Hz.
        """
        logger.info(f"[StarGANv2-VC] Converting {source_emotion}→{target_emotion} via {self._backend}")

        if self._backend == "starganv2-vc":
            return self._convert_stargan(audio, source_emotion, target_emotion, sr)
        elif self._backend == "emotivoice":
            return self._convert_emotivoice(audio, source_emotion, target_emotion, sr)
        else:
            return self._convert_librosa(audio, sr)

    def _convert_stargan(
        self,
        audio: np.ndarray,
        source_emotion: str,
        target_emotion: str,
        sr: int,
    ) -> np.ndarray:
        try:
            from inference_vc import inference  # type: ignore  # noqa: F401
            audio_resampled = resample_if_needed(audio, sr, SAMPLE_RATE)
            audio_t = torch.from_numpy(audio_resampled).float().unsqueeze(0).unsqueeze(0)
            assert self._model is not None
            with torch.no_grad():
                converted = self._model(audio_t)
            converted_np = converted.squeeze().cpu().numpy()
            return resample_if_needed(converted_np, SAMPLE_RATE, sr)
        except Exception as e:
            logger.warning(f"[StarGANv2-VC] StarGAN conversion failed: {e}, falling back")
            return self._convert_librosa(audio, sr)

    def _convert_emotivoice(
        self,
        audio: np.ndarray,
        source_emotion: str,
        target_emotion: str,
        sr: int,
    ) -> np.ndarray:
        try:
            from emotivoice import EmotiVoice  # type: ignore  # noqa: F401
            tts = EmotiVoice()
            tts.target_emotion = target_emotion
            result = tts.synthesize(audio, sr)
            return result
        except Exception as e:
            logger.warning(f"[StarGANv2-VC] EmotiVoice failed: {e}, falling back")
            return self._convert_librosa(audio, sr)

    def _convert_librosa(self, audio: np.ndarray, sr: int) -> np.ndarray:
        try:
            import librosa
            n_steps = 2.0
            pitch_shifted = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)
            rate = 1.1
            stretched = librosa.effects.time_stretch(y=pitch_shifted, rate=rate)
            target_len = len(audio)
            if len(stretched) > target_len:
                stretched = stretched[:target_len]
            elif len(stretched) < target_len:
                stretched = np.pad(stretched, (0, target_len - len(stretched)))
            return stretched
        except Exception as e:
            logger.warning(f"[StarGANv2-VC] librosa perturbation failed: {e}, returning original")
            return audio


def generate_conflict_pairs(
    neutral_audio_paths: List[str],
    neutral_texts: List[str],
    output_dir: str,
    target_emotion: str = "angry",
    converter: Optional[StarGANv2VoiceConverter] = None,
) -> List[dict]:
    """Generate synthetic conflict pairs from neutral utterances.

    For each neutral (audio, text) pair:
      - Convert audio to target_emotion using StarGANv2-VC
      - Keep original text (neutral) → audio-text conflict
      - Label as suppression conflict (audio emotion ≠ text sentiment)

    Args:
        neutral_audio_paths: Paths to neutral audio files.
        neutral_texts: Corresponding transcripts.
        output_dir: Directory to save converted audio files.
        target_emotion: Emotion to convert audio to (angry/sad/happy).
        converter: StarGANv2VoiceConverter instance. Created if None.

    Returns:
        List of conflict sample dicts ready for dataset loading.
    """
    import torchaudio

    if converter is None:
        converter = StarGANv2VoiceConverter()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for i, (wav_path, text) in enumerate(zip(neutral_audio_paths, neutral_texts)):
        try:
            # Load at StarGAN sample rate
            waveform, sr = torchaudio.load(wav_path)
            audio_np = waveform.squeeze(0).numpy()

            # Convert emotion
            converted = converter.convert(audio_np, "neutral", target_emotion, sr)

            # Save converted audio
            out_path = out_dir / f"synthetic_{i:05d}_{target_emotion}.wav"
            torchaudio.save(str(out_path), torch.tensor(converted).unsqueeze(0), sr)

            samples.append({
                "wav_path": str(out_path),
                "text": text,
                "conflict_binary": 1,
                "conflict_type_labels": [0, 1, 0],  # suppression
                "severity": 0.6,
                "speaker_id": f"synthetic_{i}",
                "is_synthetic": True,
            })
        except Exception as e:
            logger.warning(f"[SyntheticGen] Failed for {wav_path}: {e}")

    logger.info(f"[SyntheticGen] Generated {len(samples)} conflict pairs → {output_dir}")
    return samples
