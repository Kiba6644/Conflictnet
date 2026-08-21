"""Multi-label conflict subtype classifier + severity regression head.

Outputs:
  - conflict_types: (B, 6) sigmoid probabilities — [anger, disgust, fear, happiness, neutral, sadness]
  - severity:       (B, 1) regression in [0, 1] — conflict intensity
  - conflict_flag:  (B,) — binary conflict indicator (any conflict-class present)

The word-level divergence features from MFA alignment are optionally fused
as an additional input channel before classification.

Speaker-adaptive threshold (Gap 3): when ``speaker_feat`` is provided to
``forward``, the ``conflict_flag`` threshold is adjusted by a per-sample
offset predicted from the speaker representation.  This allows the model
to learn that some speakers are prosodically more "expressive" and require
a higher bar for flagging emotional conflict, while others are more
"monotone" and need a lower threshold.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


# CREMA-D 6 emotion categories — index order matches conflict_type_labels in datasets.py
CONFLICT_TYPES = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]


class SpeakerAdaptiveThreshold(nn.Module):
    """Predict a per-sample threshold offset from speaker features.

    Learns to map the speaker representation (which encodes both speaker
    identity and utterance-level prosody) to an offset in [0, max_offset]
    that is added to the global ``type_threshold``.

    This lets the model dynamically adjust the conflict detection bar:
    expressive speakers with wide prosody variance get a higher threshold
    (fewer false positives), while monotone speakers get a lower threshold
    (fewer false negatives).

    Args:
        embed_dim: Speaker feature dimension.
        max_offset: Maximum threshold offset (default 0.3).
    """

    def __init__(self, embed_dim: int = 256, max_offset: float = 0.3):
        super().__init__()
        self.max_offset = max_offset
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, speaker_feat: torch.Tensor) -> torch.Tensor:
        """Return per-sample offset in [0, max_offset].

        Args:
            speaker_feat: (B, embed_dim) speaker representation.

        Returns:
            (B,) per-sample threshold offsets.
        """
        offset = torch.tanh(self.net(speaker_feat)).squeeze(-1)  # (B,) in [-1, 1]
        return offset * (self.max_offset / 2)


class ConflictClassifier(nn.Module):
    """Joint multi-label subtype classifier and severity regression head.

    Input:
        fused_embed: (B, embed_dim) — the fused audio+text+context representation
        word_div:    (B, word_div_dim) — optional per-word divergence features

    Output:
        ConflictOutput namedtuple with fields:
            logits_type:   (B, n_types)  raw logits for BCE
            probs_type:    (B, n_types)  sigmoid probabilities
            severity:      (B, 1)        [0, 1] severity score
            conflict_flag: (B,)          bool — any type threshold exceeded

    Args:
        embed_dim: Input fused embedding dimension.
        n_types: Number of emotion classes (default 6 — CREMA-D: anger, disgust, fear, happiness, neutral, sadness).
        hidden_dims: MLP hidden layer sizes.
        word_div_dim: Dimension of word-level divergence features (0 = disabled).
        severity_head: Whether to include the severity regression head.
        type_threshold: Sigmoid threshold for conflict_flag activation.
        dropout: Dropout before classification heads.
        speaker_adaptive_threshold: If True, learns a per-sample threshold
            offset from speaker features (Gap 3).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_types: int = 6,
        hidden_dims: Tuple[int, ...] = (512, 256),
        word_div_dim: int = 0,
        severity_head: bool = True,
        type_threshold: float = 0.5,
        dropout: float = 0.1,
        speaker_adaptive_threshold: bool = True,
    ):
        super().__init__()
        self.n_types = n_types
        self.severity_head = severity_head
        self.type_threshold = type_threshold
        self.speaker_adaptive_threshold = speaker_adaptive_threshold
        self.word_div_dim = word_div_dim

        input_dim = embed_dim + word_div_dim

        # Shared feature extractor MLP
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.LayerNorm(h_dim),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.shared_mlp = nn.Sequential(*layers)

        # Multi-label subtype head (sigmoid — NOT softmax)
        self.type_head = nn.Linear(in_dim, n_types)

        # DEDICATED sarcasm head — deeper, with its own dropout
        # Trained exclusively on MUStARD++ via gated BCE in forward()
        self.sarcasm_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # Severity regression head (sigmoid → [0, 1])
        self.severity_proj = nn.Linear(in_dim, 1) if severity_head else None

        # Speaker-adaptive threshold (Gap 3)
        self.threshold_net = SpeakerAdaptiveThreshold(embed_dim=embed_dim) if speaker_adaptive_threshold else None

    def forward(
        self,
        fused_embed: torch.Tensor,
        word_div: Optional[torch.Tensor] = None,
        speaker_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Returns: (logits_type, probs_type, severity, conflict_flag)
          - logits_type:   (B, n_types)
          - probs_type:    (B, n_types)
          - severity:      (B, 1)  or None if severity_head=False
          - conflict_flag: (B,) bool
        """
        if word_div is not None:
            x = torch.cat([fused_embed, word_div], dim=-1)
        elif self.word_div_dim > 0:
            zeros = torch.zeros(fused_embed.size(0), self.word_div_dim, device=fused_embed.device, dtype=fused_embed.dtype)
            x = torch.cat([fused_embed, zeros], dim=-1)
        else:
            x = fused_embed

        feat = self.shared_mlp(x)

        logits_type = self.type_head(feat)       # (B, n_types)
        probs_type = torch.sigmoid(logits_type)  # (B, n_types)

        severity = None
        if self.severity_proj is not None:
            severity = torch.sigmoid(self.severity_proj(feat))  # (B, 1)

        # Effective threshold: base + (optional) per-sample speaker offset
        if self.threshold_net is not None and speaker_feat is not None:
            offset = self.threshold_net(speaker_feat)  # (B,)
            threshold = self.type_threshold + offset.unsqueeze(-1).expand_as(probs_type)
        else:
            threshold = self.type_threshold

        conflict_flag = (probs_type > threshold).any(dim=-1)  # (B,)

        return logits_type, probs_type, severity, conflict_flag
