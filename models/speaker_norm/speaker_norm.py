"""Speaker normalization via ECAPA-TDNN + prosody z-score.

Flow:
  1. SpeechBrain ECAPA-TDNN → 192-d speaker embedding per utterance
  2. Per-speaker prosody stats (pitch / energy / rate) via parselmouth
  3. Z-score normalize → speaker-invariant prosody residuals
  4. Cold-start fallback when speaker has < min_utts references
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prosody extraction helpers
# ---------------------------------------------------------------------------

def extract_prosody_stats(
    audio_np: np.ndarray,
    sr: int = 16000,
) -> Dict[str, float]:
    """Extract pitch, energy, and speaking-rate statistics.

    Returns a dict with keys: f0_mean, f0_std, energy_mean, energy_std,
    speaking_rate (syllables per second, approximated via voiced fraction).
    Falls back gracefully if parselmouth is unavailable.
    """
    stats: Dict[str, float] = {}
    try:
        import parselmouth  # type: ignore

        snd = parselmouth.Sound(audio_np, sampling_frequency=sr)
        pitch = snd.to_pitch()
        f0_values = pitch.selected_array["frequency"]
        f0_voiced = f0_values[f0_values > 0]
        stats["f0_mean"] = float(np.mean(f0_voiced)) if len(f0_voiced) else 0.0
        stats["f0_std"] = float(np.std(f0_voiced)) if len(f0_voiced) else 1.0

        intensity = snd.to_intensity()
        intensities = intensity.values.T.squeeze()
        stats["energy_mean"] = float(np.mean(intensities))
        stats["energy_std"] = float(np.std(intensities)) if np.std(intensities) > 0 else 1.0

        # speaking rate approximated as fraction of voiced frames × frames/sec
        duration = len(audio_np) / sr
        stats["speaking_rate"] = len(f0_voiced) / max(duration, 1e-6)

    except ImportError:
        logger.warning("parselmouth not installed — using librosa fallback for prosody")
        import librosa  # type: ignore
        f0, voiced_flag, _ = librosa.pyin(
            audio_np.astype(np.float32),
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=sr,
        )
        f0_voiced = f0[voiced_flag]
        stats["f0_mean"] = float(np.nanmean(f0_voiced)) if len(f0_voiced) else 0.0
        stats["f0_std"] = float(np.nanstd(f0_voiced)) if len(f0_voiced) > 1 else 1.0

        rms = librosa.feature.rms(y=audio_np.astype(np.float32))[0]
        stats["energy_mean"] = float(np.mean(rms))
        stats["energy_std"] = float(np.std(rms)) if np.std(rms) > 0 else 1.0

        duration = len(audio_np) / sr
        stats["speaking_rate"] = len(f0_voiced) / max(duration, 1e-6)

    return stats


# ---------------------------------------------------------------------------
# Per-speaker running statistics tracker
# ---------------------------------------------------------------------------

class SpeakerStats:
    """Online running mean/std tracker for a single speaker's prosody.

    Supports two normalisation modes:
      - ``z_score``: standard (x - mean) / std
      - ``baseline_normalize``: residual from neutral-speaker EMA baseline,
        then scaled by speaker's overall std. Falls back to z-score when
        neutral baseline is unavailable (< 2 samples seen).
    """

    def __init__(self):
        self.n: int = 0
        self.mean = np.zeros(3, dtype=np.float64)   # f0_mean, energy_mean, rate
        self.M2 = np.zeros(3, dtype=np.float64)      # for Welford online variance
        self.neutral_baseline: Optional[np.ndarray] = None  # EMA of neutral utterances

    def update(self, f0_mean: float, energy_mean: float, speaking_rate: float):
        x = np.array([f0_mean, energy_mean, speaking_rate], dtype=np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(3, dtype=np.float64)
        return np.sqrt(self.M2 / (self.n - 1))

    def z_score(self, f0_mean: float, energy_mean: float, speaking_rate: float) -> np.ndarray:
        x = np.array([f0_mean, energy_mean, speaking_rate], dtype=np.float64)
        std = self.std
        std = np.where(std < 1e-6, 1.0, std)
        return (x - self.mean) / std

    def update_baseline(
        self,
        f0_mean: float,
        energy_mean: float,
        speaking_rate: float,
        lr: float = 0.1,
    ):
        """Update the neutral-speaking baseline via exponential moving average.

        Call this for utterances where ``conflict_flag`` is False (i.e. the
        speaker is speaking neutrally).  The EMA tracks the centroid of
        non-conflict prosody so that ``baseline_normalize`` measures *departure*
        from the speaker's neutral speaking style.

        Args:
            f0_mean: Mean F0 of the utterance (Hz).
            energy_mean: Mean energy/intensity (dB).
            speaking_rate: Voiced-frame fraction (proxy for rate).
            lr: EMA learning rate (default 0.1).
        """
        x = np.array([f0_mean, energy_mean, speaking_rate], dtype=np.float64)
        if self.neutral_baseline is None:
            self.neutral_baseline = x.copy()
        else:
            self.neutral_baseline = (1.0 - lr) * self.neutral_baseline + lr * x

    def baseline_normalize(
        self,
        f0_mean: float,
        energy_mean: float,
        speaking_rate: float,
    ) -> np.ndarray:
        """Deviation from neutral-speaking baseline, scaled by speaker std.

        ``baseline_normalize`` = (x - neutral_centroid) / overall_std.

        When the neutral baseline is unavailable (< 2 observations) this
        falls back to standard ``z_score``.

        Returns:
            3-d array: [f0_deviation, energy_deviation, rate_deviation].
        """
        if self.neutral_baseline is not None and self.n >= 2:
            x = np.array([f0_mean, energy_mean, speaking_rate], dtype=np.float64)
            residual = x - self.neutral_baseline
            std = self.std
            std = np.where(std < 1e-6, 1.0, std)
            return residual / std
        return self.z_score(f0_mean, energy_mean, speaking_rate)


# ---------------------------------------------------------------------------
# Cold-start fallback using speaker cluster centroids
# ---------------------------------------------------------------------------

class ColdStartFallback:
    """Handles speakers with insufficient reference utterances.

    Strategy hierarchy (from spec):
      1. If speaker has ≥ min_ref_utts → use their own SpeakerStats
      2. If speaker has < min_ref_utts but gender is known → use gender cluster
      3. Else → use global corpus statistics
    """

    def __init__(self, n_clusters: int = 20, min_ref_utts: int = 5):
        self.n_clusters = n_clusters
        self.min_ref_utts = min_ref_utts
        self._global_stats = SpeakerStats()
        self._gender_stats: Dict[str, SpeakerStats] = {"M": SpeakerStats(), "F": SpeakerStats()}
        # cluster centroids fitted from VoxCeleb embeddings (populated externally)
        self._cluster_stats: List[SpeakerStats] = [SpeakerStats() for _ in range(n_clusters)]
        self._cluster_centroids: Optional[np.ndarray] = None  # (n_clusters, 192)

    def register_utterance(
        self,
        f0_mean: float,
        energy_mean: float,
        speaking_rate: float,
        gender: Optional[str] = None,
    ):
        """Update global and gender-group statistics."""
        self._global_stats.update(f0_mean, energy_mean, speaking_rate)
        if gender in self._gender_stats:
            self._gender_stats[gender].update(f0_mean, energy_mean, speaking_rate)

    def set_cluster_centroids(self, centroids: np.ndarray):
        """Provide pre-computed k-means centroids (n_clusters × 192)."""
        assert centroids.shape == (self.n_clusters, 192), \
            f"Expected ({self.n_clusters}, 192), got {centroids.shape}"
        self._cluster_centroids = centroids

    def get_stats(
        self,
        speaker_stats: SpeakerStats,
        speaker_embedding: Optional[np.ndarray] = None,
        gender: Optional[str] = None,
    ) -> SpeakerStats:
        """Return the best available statistics object for normalization."""
        if speaker_stats.n >= self.min_ref_utts:
            return speaker_stats
        if gender in self._gender_stats and self._gender_stats[gender].n >= self.min_ref_utts:
            return self._gender_stats[gender]
        if self._cluster_centroids is not None and speaker_embedding is not None:
            dists = np.linalg.norm(self._cluster_centroids - speaker_embedding, axis=1)
            nearest = int(np.argmin(dists))
            if self._cluster_stats[nearest].n >= self.min_ref_utts:
                return self._cluster_stats[nearest]
        return self._global_stats


# ---------------------------------------------------------------------------
# Standalone preprocessing function (NumPy — runs in data loader, NOT forward)
# ---------------------------------------------------------------------------

def compute_prosody_z_scores(
    audio_np_list: List[np.ndarray],
    speaker_ids: List[str],
    speaker_registry: "defaultdict[str, SpeakerStats]",
    cold_start: "ColdStartFallback",
    genders: Optional[List[Optional[str]]] = None,
    use_baseline_subtract: bool = False,
    conflict_flags: Optional[List[bool]] = None,
    sr: int = 16000,
) -> torch.Tensor:
    """Compute per-speaker prosody features (z-score or baseline-subtracted).

    This is a **module-level function**, intentionally kept outside any
    ``nn.Module`` so that NumPy / parselmouth operations are never part of
    the PyTorch autograd graph or ONNX export path.

    Call this from your collate_fn or Dataset.__getitem__, then pass the
    resulting CPU tensor to ``SpeakerNormalizer.forward(prosody_z=...)``.

    Args:
        audio_np_list: Per-utterance numpy waveforms.
        speaker_ids: Corresponding speaker ID strings.
        speaker_registry: The normalizer's ``_speaker_registry`` defaultdict.
        cold_start: The normalizer's ``ColdStartFallback`` instance.
        genders: Optional list of 'M' / 'F' / None per utterance.
        use_baseline_subtract: If True, use ``baseline_normalize`` (deviation
            from neutral centroid) instead of standard z-score.
        conflict_flags: Per-utterance bool — used to update neutral baseline
            for non-conflict utterances when ``use_baseline_subtract`` is True.
        sr: Audio sample rate.

    Returns:
        CPU float32 tensor of shape (B, 3) — prosody features.
    """
    safe_genders: List[Optional[str]] = genders if genders is not None else [None] * len(audio_np_list)
    cf: List[bool] = conflict_flags if conflict_flags is not None else [False] * len(audio_np_list)
    
    z_scores = []
    for audio_np, spk_id, gender, is_conflict in zip(audio_np_list, speaker_ids, safe_genders, cf):
        stats_raw = extract_prosody_stats(audio_np, sr=sr)
        spk_stats = speaker_registry[spk_id]
        spk_stats.update(
            stats_raw["f0_mean"], stats_raw["energy_mean"], stats_raw["speaking_rate"]
        )
        cold_start.register_utterance(
            stats_raw["f0_mean"], stats_raw["energy_mean"], stats_raw["speaking_rate"], gender
        )

        # Update neutral baseline for non-conflict (neutral) utterances
        if use_baseline_subtract and not is_conflict:
            spk_stats.update_baseline(
                stats_raw["f0_mean"], stats_raw["energy_mean"], stats_raw["speaking_rate"]
            )

        best_stats = cold_start.get_stats(spk_stats, gender=gender)

        if use_baseline_subtract:
            z = best_stats.baseline_normalize(
                stats_raw["f0_mean"], stats_raw["energy_mean"], stats_raw["speaking_rate"]
            )
        else:
            z = best_stats.z_score(
                stats_raw["f0_mean"], stats_raw["energy_mean"], stats_raw["speaking_rate"]
            )
        z_scores.append(z)

    return torch.tensor(np.stack(z_scores), dtype=torch.float32)  # CPU tensor


# ---------------------------------------------------------------------------
# Main speaker normalization module
# ---------------------------------------------------------------------------

class SpeakerNormalizer(nn.Module):
    """Full speaker normalization pipeline.

    1. ECAPA-TDNN speaker embedding (192-d)
    2. Per-speaker prosody z-score (3-d: f0, energy, rate)
    3. Project speaker embedding + z-scores into embed_dim via linear layer
    4. Returns (speaker_embed, prosody_z) to be concatenated / gated upstream

    Args:
        embed_dim: Output projection size (matches shared ConflictNet space).
        model_source: SpeechBrain ECAPA-TDNN model ID.
        cold_start_clusters: K-means clusters for cold-start speakers.
        min_ref_utts: Minimum utterances before using speaker's own stats.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        cold_start_clusters: int = 20,
        min_ref_utts: int = 5,
        use_baseline_subtract: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self._spk_embed_dim = 192
        self._prosody_dim = 3
        self.use_baseline_subtract = use_baseline_subtract

        # Speaker embedding model (lazy-loaded to avoid SpeechBrain import at import time)
        self._model_source = model_source
        self._spk_model = None

        # Prosody projection: [192 + 3] → embed_dim
        self.spk_proj = nn.Sequential(
            nn.Linear(self._spk_embed_dim + self._prosody_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # Per-speaker statistics registry  {speaker_id → SpeakerStats}
        self._speaker_registry: defaultdict[str, SpeakerStats] = defaultdict(SpeakerStats)
        self.cold_start = ColdStartFallback(
            n_clusters=cold_start_clusters,
            min_ref_utts=min_ref_utts,
        )

        # Eagerly load speaker model during init (before DDP wrapping) to prevent DDP deadlock
        self._load_spk_model()

    def _load_spk_model(self):
        if self._spk_model is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
                self._spk_model = EncoderClassifier.from_hparams(
                    source=self._model_source,
                    savedir=f"pretrained_models/{self._model_source.replace('/', '_')}",
                )
                logger.info(f"[SpeakerNorm] Loaded ECAPA-TDNN from {self._model_source}")
            except Exception as e:
                logger.warning(f"[SpeakerNorm] SpeechBrain unavailable ({e}) — speaker embedding disabled")
                self._spk_model = "disabled"

    def _get_spk_model(self):
        if self._spk_model is None:
            self._load_spk_model()
        return self._spk_model

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_speaker(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract 192-d ECAPA speaker embedding. Input: (B, T) waveform."""
        model = self._get_spk_model()
        if isinstance(model, str) or model is None:
            return torch.zeros(audio.size(0), self._spk_embed_dim, device=audio.device)
        embeddings = model.encode_batch(audio)  # (B, 1, 192)
        return embeddings.squeeze(1)            # (B, 192)

    def precompute_prosody_z(
        self,
        audio_np_list: List[np.ndarray],
        speaker_ids: List[str],
        genders: Optional[List[Optional[str]]] = None,
        use_baseline_subtract: bool = False,
        conflict_flags: Optional[List[bool]] = None,
        sr: int = 16000,
    ) -> torch.Tensor:
        """**Preprocessing helper** — call this in the data loader / collate_fn,
        NOT inside the model forward pass.

        Computes per-speaker prosody features using NumPy/parselmouth and
        returns a plain CPU float32 tensor ready to be moved to the target
        device before calling ``forward()``.

        Args:
            audio_np_list: List of numpy waveforms (one per batch item).
            speaker_ids: Corresponding speaker IDs.
            genders: Optional gender labels ('M' / 'F' / None).
            use_baseline_subtract: If True, uses ``baseline_normalize``
                (deviation from neutral centroid) instead of z-score.
            conflict_flags: Per-utterance bool used to update neutral baseline
                for non-conflict utterances.
            sr: Sample rate.

        Returns:
            CPU tensor of shape (B, 3) — prosody features.
        """
        return compute_prosody_z_scores(
            audio_np_list=audio_np_list,
            speaker_ids=speaker_ids,
            speaker_registry=self._speaker_registry,
            cold_start=self.cold_start,
            genders=genders,
            use_baseline_subtract=use_baseline_subtract,
            conflict_flags=conflict_flags,
            sr=sr,
        )

    # ------------------------------------------------------------------
    # Forward pass  (pure-torch — no NumPy inside)
    # ------------------------------------------------------------------

    def forward(
        self,
        audio: torch.Tensor,
        prosody_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (spk_embedding: B×192, speaker_feat: B×embed_dim).

        ``speaker_feat`` is the projected [spk_embed ∥ prosody_z] vector.

        Args:
            audio: (B, T) waveform tensor on the target device.
            prosody_z: Pre-computed prosody z-scores (B, 3) on the target
                device.  Compute this *before* calling forward using
                ``precompute_prosody_z()`` in your collate_fn / data pipeline.
                If None, prosody z-scores default to zeros (inference mode
                without speaker history).
        """
        spk_embed = self.encode_speaker(audio)  # (B, 192) — pure torch

        if prosody_z is None:
            prosody_z = torch.zeros(
                audio.size(0), self._prosody_dim, device=audio.device
            )

        combined = torch.cat([spk_embed, prosody_z], dim=-1)  # (B, 195)
        speaker_feat = self.spk_proj(combined)                # (B, embed_dim)
        return spk_embed, speaker_feat
