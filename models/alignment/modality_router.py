"""Adaptive Modality Router for ConflictNet.

Learned scalar gate α ∈ [0,1] that decides per-utterance how much to trust
text vs. audio embeddings before fusion.  Also provides corruption utilities
for the eval_corruption.py evaluation script.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Modality Router
# ---------------------------------------------------------------------------

class ModalityRouter(nn.Module):
    """Lightweight learned scalar gate α ∈ [0,1] that decides how much to
    trust text vs audio for each utterance.

    Architecture:
        Input: cat([text_embed, audio_embed]) → (B, 2*embed_dim)
        FC1: 2D → D//2, GELU + LayerNorm
        FC2: D//2 → D//8, GELU
        FC3: D//8 → 1, Sigmoid
        Output: α (B, 1)

    Fused: h = α * h_text + (1-α) * h_audio

    ~6,000 trainable parameters for embed_dim=256.
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        d_in = embed_dim * 2
        d_mid = embed_dim // 2
        d_small = embed_dim // 8

        self.fc1 = nn.Linear(d_in, d_mid)
        self.norm1 = nn.LayerNorm(d_mid)
        self.fc2 = nn.Linear(d_mid, d_small)
        self.fc3 = nn.Linear(d_small, 1)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Initialise fc3 bias to 0 so α starts near 0.5 (equal weighting)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, text_embed: torch.Tensor, audio_embed: torch.Tensor) -> torch.Tensor:
        """Compute gate weight α.

        Args:
            text_embed:  (B, embed_dim)
            audio_embed: (B, embed_dim)

        Returns:
            alpha: (B, 1) in [0, 1]  — weight applied to text_embed.
        """
        x = torch.cat([text_embed, audio_embed], dim=-1)  # (B, 2*D)
        x = self.norm1(self.act(self.fc1(x)))              # (B, D//2)
        x = self.act(self.fc2(x))                          # (B, D//8)
        alpha = self.sigmoid(self.fc3(x))                  # (B, 1)
        return alpha


# ---------------------------------------------------------------------------
# Entropy regularisation
# ---------------------------------------------------------------------------

def entropy_regularization_loss(alpha: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Entropy regularisation that penalises the router for always collapsing
    to 0 or 1 (i.e. completely ignoring one modality).

    Maximising entropy keeps α spread across [0,1], so the router explores
    partial-trust decisions rather than hard-switching.

    Formula:
        L_entropy = -mean( α·log(α+ε) + (1-α)·log(1-α+ε) )

    A *higher* value means *more* uniform α → router is uncertain / exploring.
    Minimising the negative entropy (i.e. adding this loss with a positive
    weight) discourages collapse to the extremes.

    Args:
        alpha: (B, 1) gate weights output by ModalityRouter.
        eps:   Small constant to avoid log(0).

    Returns:
        Scalar entropy loss (lower is *more* collapsed, higher is *more* uniform).
    """
    entropy = -(alpha * torch.log(alpha + eps) + (1.0 - alpha) * torch.log(1.0 - alpha + eps))
    return entropy.mean()


# ---------------------------------------------------------------------------
# Corruption utilities (used by eval_corruption.py)
# ---------------------------------------------------------------------------

def corrupt_audio_waveform(
    audio: torch.Tensor,
    mode: str,
    snr_db: float = 15.0,
) -> torch.Tensor:
    """Apply a corruption to a raw audio waveform tensor.

    Args:
        audio:  (B, T) or (T,) waveform tensor (any dtype).
        mode:   Corruption mode.
                  • ``'gaussian_noise'`` — add white noise at the given SNR.
                  • ``'zeroed'``         — replace the entire waveform with zeros.
        snr_db: Signal-to-noise ratio in dB (only used for ``'gaussian_noise'``).

    Returns:
        Corrupted waveform, same shape and device as *audio*.
    """
    if mode == "zeroed":
        return torch.zeros_like(audio)

    if mode == "gaussian_noise":
        # Compute signal power per-sample (or scalar for 1-D input)
        if audio.dim() == 2:
            signal_power = audio.pow(2).mean(dim=-1, keepdim=True)  # (B, 1)
        else:
            signal_power = audio.pow(2).mean()

        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise_std = torch.sqrt(noise_power.clamp(min=1e-12))
        noise = torch.randn_like(audio) * noise_std
        return audio + noise

    raise ValueError(
        f"Unknown audio corruption mode '{mode}'. "
        "Supported: 'gaussian_noise', 'zeroed'."
    )


def corrupt_text_tokens(
    input_ids: torch.Tensor,
    mode: str,
    unk_token_id: int = 100,
    corrupt_frac: float = 0.3,
) -> torch.Tensor:
    """Apply a corruption to tokenised text input IDs.

    Args:
        input_ids:     (B, L) integer token-ID tensor.
        mode:          Corruption mode.
                         • ``'random_replace'`` — randomly replace *corrupt_frac*
                           of tokens (excluding the first/last special tokens)
                           with *unk_token_id*.
                         • ``'all_unk'``        — replace every token with
                           *unk_token_id*.
        unk_token_id:  Token ID used as the replacement token (default 100 =
                       ``[UNK]`` in most HF tokenisers).
        corrupt_frac:  Fraction of tokens to replace for ``'random_replace'``
                       (ignored for ``'all_unk'``).

    Returns:
        Corrupted input_ids, same shape and device as the input.
    """
    corrupted = input_ids.clone()

    if mode == "all_unk":
        corrupted[:] = unk_token_id
        return corrupted

    if mode == "random_replace":
        B, L = corrupted.shape
        # Create a per-token mask with the requested fraction
        mask = torch.rand(B, L, device=corrupted.device) < corrupt_frac
        corrupted = corrupted.masked_fill(mask, unk_token_id)
        return corrupted

    raise ValueError(
        f"Unknown text corruption mode '{mode}'. "
        "Supported: 'random_replace', 'all_unk'."
    )
