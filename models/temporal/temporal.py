"""Transformer temporal context module.

Takes a sequence of per-turn fused embeddings and produces
context-aware representations using a causal Transformer encoder.

Key design decisions:
- Causal masking: each turn can only attend to prior turns (no future leakage)
- Positional encoding: learned (not fixed sinusoidal) — turn order ≠ time
- Speaker role tokens: optional [SPK_A] / [SPK_B] type embeddings
- Returns both per-turn and pooled (mean) representations
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings for turn position."""

    def __init__(self, max_turns: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_turns, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) — adds position embedding for each of the T turns."""
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, T)
        return x + self.embedding(positions)  # broadcast over batch


class SpeakerRoleEmbedding(nn.Module):
    """Optional type embedding: speaker A (0) vs speaker B (1)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(2, embed_dim)  # 0=SPK_A, 1=SPK_B

    def forward(self, x: torch.Tensor, speaker_roles: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        speaker_roles: (B, T) — 0 or 1 indicating which speaker each turn belongs to
        """
        return x + self.embedding(speaker_roles)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPathTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, drop_path_prob=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_path_prob = drop_path_prob

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None, is_causal: bool = False) -> torch.Tensor:
        # Pre-LN architecture
        x = src
        if self.norm_first:
            attn_out = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + drop_path(attn_out, self.drop_path_prob, self.training)
            ff_out = self._ff_block(self.norm2(x))
            x = x + drop_path(ff_out, self.drop_path_prob, self.training)
        else:
            attn_out = self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            x = self.norm1(x + drop_path(attn_out, self.drop_path_prob, self.training))
            ff_out = self._ff_block(x)
            x = self.norm2(x + drop_path(ff_out, self.drop_path_prob, self.training))
        return x

class TransformerTemporalContext(nn.Module):
    """Causal Transformer encoder over a dialogue turn sequence.

    Architecture:
        Input: sequence of fused embeddings (B, T, embed_dim)
        → Positional encoding (learned)
        → Optional speaker role embedding
        → N × Transformer encoder layers (causal mask)
        → Output: per-turn context embeddings (B, T, embed_dim)
                  + pooled context (B, embed_dim)

    Args:
        embed_dim: Dimensionality of turn embeddings (must match projection heads).
        n_layers: Number of Transformer encoder layers.
        n_heads: Number of attention heads. Must divide embed_dim evenly.
        ff_dim: Feed-forward hidden size (default 4× embed_dim).
        dropout: Dropout probability.
        max_turns: Maximum number of turns in a dialogue context window.
        use_speaker_roles: Whether to add speaker A/B role embeddings.
        causal: If True, applies causal (autoregressive) attention mask.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_turns: int = 16,
        use_speaker_roles: bool = True,
        causal: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_turns = max_turns
        self.causal = causal

        ff_dim = ff_dim or embed_dim * 4
        
        self.pos_encoding = LearnedPositionalEncoding(max_turns, embed_dim)
        self.speaker_role_emb = SpeakerRoleEmbedding(embed_dim) if use_speaker_roles else None

        encoder_layer = DropPathTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (B, T, D) convention throughout
            norm_first=True,   # Pre-LN for training stability
            drop_path_prob=0.1,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal attention mask (True = masked/ignored)."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()  # (T, T)

    def forward(
        self,
        turn_embeds: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        speaker_roles: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            turn_embeds: (B, T, embed_dim) — fused audio+text embedding per turn.
            padding_mask: (B, T) bool — True where turn is padding (no utterance).
            speaker_roles: (B, T) int — 0/1 role per turn. Required if use_speaker_roles.

        Returns:
            per_turn: (B, T, embed_dim) — contextualised per-turn embeddings.
            pooled:   (B, embed_dim)    — mean over non-padded turns.
        """
        B, T, D = turn_embeds.shape
        assert D == self.embed_dim, \
            f"Expected embed_dim={self.embed_dim}, got {D}"
        assert T <= self.max_turns, \
            f"Sequence length {T} exceeds max_turns={self.max_turns}"

        x = self.pos_encoding(turn_embeds)  # (B, T, D)

        if self.speaker_role_emb is not None and speaker_roles is not None:
            x = self.speaker_role_emb(x, speaker_roles)

        # Skip causal mask when T == 1 (single turn cannot see future turns anyway;
        # passing a (1,1) mask alongside src_key_padding_mask triggers a known
        # PyTorch SDPA kernel deadlock on Turing/T4 GPUs).
        attn_mask = self._causal_mask(T, turn_embeds.device) if (self.causal and T > 1) else None

        # Omit src_key_padding_mask if no elements in the batch are actually padded
        if padding_mask is not None and not padding_mask.any():
            padding_mask = None

        per_turn = self.transformer(
            x,
            mask=attn_mask,
            src_key_padding_mask=padding_mask,
        )  # (B, T, D)
        per_turn = self.output_norm(per_turn)

        # Mean pooling over non-padded turns
        if padding_mask is not None:
            valid = (~padding_mask).float().unsqueeze(-1)  # (B, T, 1)
            pooled = (per_turn * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            pooled = per_turn.mean(dim=1)  # (B, D)

        return per_turn, pooled
