"""Audio augmentation pipeline for ConflictNet.

Provides on-the-fly augmentations using audiomentations:
  - Speed perturbation (0.9×–1.1×)
  - Additive noise (Gaussian or MUSAN corpus)
  - Pitch shift
  - Time masking (SpecAugment-style)

Usage:
    from data.augmentation import AudioAugmentor

    augmentor = AudioAugmentor(sample_rate=16000)
    augmented_np = augmentor(audio_np)     # numpy in, numpy out
    augmented_tensor = augmentor.augment_tensor(audio_tensor)  # tensor in, tensor out
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class AudioAugmentor:
    """On-the-fly audio augmentation using audiomentations.

    Falls back gracefully to no-op if audiomentations is not installed.

    Args:
        sample_rate: Audio sample rate.
        speed_perturb: Enable speed perturbation (0.9×–1.1×).
        additive_noise: Enable Gaussian noise augmentation.
        musan_path: Path to MUSAN corpus for realistic noise augmentation.
        pitch_shift: Enable pitch shifting (±2 semitones).
        time_mask: Enable random time masking (SpecAugment-style).
        p: Global probability of applying any augmentation.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        speed_perturb: bool = False,  # Disabled by default due to severe CPU bottleneck
        additive_noise: bool = True,
        musan_path: Optional[str] = None,
        pitch_shift: bool = False,
        time_mask: bool = True,
        p: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self._available = False
        self._transform = None

        try:
            import audiomentations as am  # type: ignore

            augmentations = []

            if speed_perturb:
                augmentations.append(
                    am.TimeStretch(
                        min_rate=0.9,
                        max_rate=1.1,
                        p=0.5,
                    )
                )

            if additive_noise:
                if musan_path and Path(musan_path).exists():
                    augmentations.append(
                        am.AddBackgroundNoise(
                            sounds_path=musan_path,
                            min_snr_db=10,
                            max_snr_db=30,
                            p=0.3,
                        )
                    )
                else:
                    augmentations.append(
                        am.AddGaussianNoise(
                            min_amplitude=0.001,
                            max_amplitude=0.015,
                            p=0.3,
                        )
                    )

            if pitch_shift:
                augmentations.append(
                    am.PitchShift(
                        min_semitones=-2,
                        max_semitones=2,
                        p=0.3,
                    )
                )

            if time_mask:
                augmentations.append(
                    am.TimeMask(
                        min_band_size=0.05,
                        max_band_size=0.15,
                        p=0.3,
                    )
                )

            if augmentations:
                self._transform = am.Compose(augmentations, p=p)
                self._available = True
                logger.info(
                    f"[Augmentation] {len(augmentations)} transforms active "
                    f"(p={p})"
                )

        except ImportError:
            logger.warning(
                "[Augmentation] audiomentations not installed — "
                "augmentation disabled. Install with: pip install audiomentations"
            )

    @property
    def available(self) -> bool:
        return self._available

    def __call__(self, audio_np: np.ndarray) -> np.ndarray:
        """Augment a numpy waveform. Returns augmented copy or original if disabled."""
        if not self._available or self._transform is None:
            return audio_np
        # audiomentations expects float32, mono
        audio = audio_np.astype(np.float32)
        augmented = self._transform(samples=audio, sample_rate=self.sample_rate)
        return augmented

    def augment_tensor(self, audio_tensor: "torch.Tensor") -> "torch.Tensor":
        """Convenience: augment a PyTorch tensor (CPU). Returns same dtype/device."""
        import torch

        device = audio_tensor.device
        audio_np = audio_tensor.cpu().numpy()
        augmented_np = self(audio_np)
        return torch.tensor(augmented_np, dtype=audio_tensor.dtype, device=device)
