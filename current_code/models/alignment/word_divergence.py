"""Word-level audio-text divergence features via MFA alignment.

Pipeline:
  1. Run MFA offline to produce .TextGrid files
  2. This module parses TextGrids → per-word audio spans
  3. For each word: extract audio embedding + text token embedding
  4. Compute per-word cosine divergence = 1 - cos_sim(audio_word, text_word)
  5. Aggregate into a fixed-size feature vector for the classifier

Two computation paths:
  - ``forward_from_precomputed``: word embeddings already extracted
  - ``forward_from_encoder_hidden``: uses encoder hidden states + word timestamps
    to extract per-word embeddings without re-forwarding the encoder.

MFA must be run offline before training:
  mfa align /path/to/corpus english_us_mfa english_us_mfa /path/to/textgrids
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MFA_TIER_NAMES = ("words", "English words", "Word", "word", "wrd")


def parse_textgrid(tg_path: str) -> List[Tuple[str, float, float]]:
    """Parse a Praat TextGrid file into word-level (word, start, end) tuples.

    Handles both MFA v1 and v2 output formats and multiple tier naming
    conventions.  Falls back to the first interval tier if no tier named
    'words' is found.

    Args:
        tg_path: Path to the .TextGrid file.

    Returns:
        List of (word, start_seconds, end_seconds) tuples, excluding silence.
    """
    words: List[Tuple[str, float, float]] = []
    with open(tg_path, "r", encoding="utf-8-sig") as f:
        lines = [line.rstrip("\n") for line in f]

    tiers = _split_tiers(lines)
    target = _find_word_tier(tiers)
    if target is None:
        logger.warning(f"[TextGrid] No words tier found in {tg_path}")
        return words

    words = _parse_interval_tier(target)
    return words


def _split_tiers(lines: List[str]) -> List[List[str]]:
    """Split a TextGrid into per-tier line groups.

    Groups lines by ``item [N]:`` boundaries. Each tier contains its
    header lines (name, class) plus all intervals.
    """
    tiers: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"item\s*\[", stripped):
            if current:
                tiers.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        tiers.append(current)
    return tiers


def _find_word_tier(tiers: List[List[str]]) -> Optional[List[str]]:
    """Find the interval tier whose name matches 'words'."""
    for tier in tiers:
        tier_text = "\n".join(tier).lower()
        for name in MFA_TIER_NAMES:
            if f'name = "{name.lower()}"' in tier_text or f"name = '{name.lower()}'" in tier_text:
                return tier
    for tier in tiers:
        tier_text = "\n".join(tier)
        if "class = \"IntervalTier\"" in tier_text.lower():
            return tier
    return None


def _parse_interval_tier(lines: List[str]) -> List[Tuple[str, float, float]]:
    """Extract (word, start, end) from an interval tier's lines."""
    words: List[Tuple[str, float, float]] = []
    current: Dict[str, float | str] = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("xmin ="):
            current["start"] = float(stripped.split("=")[1].strip())
        elif stripped.startswith("xmax ="):
            current["end"] = float(stripped.split("=")[1].strip())
        elif stripped.startswith("text ="):
            text = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            current["text"] = text
            if text and text.lower() not in ("", "sp", "sil", "<eps>", "silence", "pau"):
                words.append((str(current.get("text", "")), float(current.get("start", 0)), float(current.get("end", 0))))
            current = {}
    return words


def extract_word_audio_spans(
    audio: np.ndarray,
    sr: int,
    word_spans: List[Tuple[str, float, float]],
) -> List[np.ndarray]:
    """Extract audio numpy slices for each word given (word, start, end) spans."""
    clips = []
    for _, start, end in word_spans:
        s = int(start * sr)
        e = int(end * sr)
        clip = audio[s:e]
        clips.append(clip if len(clip) > 0 else np.zeros(sr // 10))
    return clips


class WordLevelDivergence(nn.Module):
    """Compute per-word audio-text divergence features.

    Two computation paths:
      1. ``forward_from_precomputed(word_audio_embeds, word_text_embeds)`` —
         use pre-extracted word-level embeddings (you handle extraction).
      2. ``forward_from_encoder_hidden(audio_frame_embeds, word_timestamps,
         text_token_embeds, token_word_map)`` — extracts word embeddings from
         full-utterance encoder hidden states using MFA timestamps.

    Aggregation into a fixed-dim feature vector (DIVERGENCE_FEAT_DIM = 8):
      - [max_div, mean_div, std_div, n_conflict_words_ratio,
         top3_divergent_positions (normalised), total_n_words_normalised]

    Args:
        embed_dim: Shared embedding dim.
        divergence_threshold: Per-word divergence above this is "conflict word".
    """

    DIVERGENCE_FEAT_DIM = 8

    def __init__(
        self,
        embed_dim: int = 256,
        divergence_threshold: float = 0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.divergence_threshold = divergence_threshold

        # Lightweight projection of word-level features into shared space
        self.word_audio_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.word_text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def compute_word_divergences(
        self,
        word_audio_embeds: torch.Tensor,
        word_text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Per-word cosine divergence in [0, 2]."""
        a = F.normalize(self.word_audio_proj(word_audio_embeds), dim=-1)
        t = F.normalize(self.word_text_proj(word_text_embeds), dim=-1)
        cos_sim = (a * t).sum(dim=-1)
        return 1.0 - cos_sim

    def aggregate(self, divergences: torch.Tensor) -> torch.Tensor:
        """Collapse variable-length per-word divergences to DIVERGENCE_FEAT_DIM vector."""
        n = divergences.size(0)
        if n == 0:
            return torch.zeros(self.DIVERGENCE_FEAT_DIM, device=divergences.device)

        max_div = divergences.max()
        mean_div = divergences.mean()
        std_div = divergences.std() if n > 1 else torch.zeros(1, device=divergences.device)
        n_conflict = (divergences > self.divergence_threshold).float().mean()

        # Top-3 divergent word positions (normalised by sequence length)
        top_k = min(3, n)
        _, top_idx = torch.topk(divergences, top_k)
        top_positions = top_idx.float() / max(n - 1, 1)
        if top_k < 3:
            pad = torch.zeros(3 - top_k, device=divergences.device)
            top_positions = torch.cat([top_positions, pad])

        n_words_norm = torch.tensor(min(n / 50.0, 1.0), device=divergences.device)

        return torch.stack([
            max_div, mean_div, std_div.squeeze(), n_conflict,
            top_positions[0], top_positions[1], top_positions[2],
            n_words_norm,
        ])

    def forward_from_precomputed(
        self,
        word_audio_embeds_list: List[torch.Tensor],
        word_text_embeds_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute divergence features for a batch of utterances.

        Args:
            word_audio_embeds_list: Per-utterance word audio embeddings.
            word_text_embeds_list:  Per-utterance word text embeddings.

        Returns:
            (B, DIVERGENCE_FEAT_DIM) divergence feature matrix.
        """
        feats = []
        for wa, wt in zip(word_audio_embeds_list, word_text_embeds_list):
            if wa.size(0) == 0:
                feats.append(torch.zeros(self.DIVERGENCE_FEAT_DIM, device=wa.device))
                continue
            div = self.compute_word_divergences(wa, wt)
            feats.append(self.aggregate(div))
        return torch.stack(feats)

    def forward_from_encoder_hidden(
        self,
        audio_frame_embeds: torch.Tensor,
        word_timestamps: List[List[Tuple[float, float]]],
        text_token_embeds: torch.Tensor,
        token_word_boundaries: List[List[Tuple[int, int]]],
        frame_rate: float = 50.0,
    ) -> torch.Tensor:
        """Extract per-word embeddings from encoder hidden states.

        This avoids re-encoding each word through the audio encoder.
        Instead, it uses the frame-level outputs and MFA timestamps
        to extract per-word audio features, then matches them against
        per-word text token averages.

        Args:
            audio_frame_embeds: (B, T_frames, D) — full encoder hidden states.
            word_timestamps: Per-utterance list of (start_sec, end_sec).
            text_token_embeds: (B, L_text) — DeBERTa hidden states per token.
            token_word_boundaries: Per-utterance list of (start_idx, end_idx)
                spans mapping token positions to word boundaries.
            frame_rate: Number of encoder frames per second (typically 50 for
                16kHz audio with standard CNN stride).

        Returns:
            (B, DIVERGENCE_FEAT_DIM) divergence features.
        """
        B = audio_frame_embeds.size(0)
        feats = []
        for b in range(B):
            n_words = len(word_timestamps[b])
            if n_words == 0:
                feats.append(torch.zeros(self.DIVERGENCE_FEAT_DIM, device=audio_frame_embeds.device))
                continue

            frames = audio_frame_embeds[b]  # (T_frames, D)
            word_audio = []
            word_text = []

            for w_idx, (w_start, w_end) in enumerate(word_timestamps[b]):
                # Map time to frame indices
                f_start = max(0, int(w_start * frame_rate))
                f_end = min(frames.size(0), int(w_end * frame_rate))
                if f_end <= f_start:
                    continue
                word_audio.append(frames[f_start:f_end].mean(dim=0))

                # Map to text token span
                if w_idx < len(token_word_boundaries[b]):
                    t_start, t_end = token_word_boundaries[b][w_idx]
                    if t_end > t_start:
                        word_text.append(text_token_embeds[b, t_start:t_end].mean(dim=0))

            if not word_audio or not word_text:
                feats.append(torch.zeros(self.DIVERGENCE_FEAT_DIM, device=audio_frame_embeds.device))
                continue

            wa = torch.stack(word_audio)
            wt = torch.stack(word_text)
            n_words_audio, n_words_text = wa.size(0), wt.size(0)
            if n_words_audio != n_words_text:
                logger.warning(
                    "Word count mismatch: audio_words=%d, text_words=%d. Truncating to %d.",
                    n_words_audio, n_words_text, min(n_words_audio, n_words_text),
                )
            n = min(n_words_audio, n_words_text)
            div = self.compute_word_divergences(wa[:n], wt[:n])
            feats.append(self.aggregate(div))

        return torch.stack(feats)
