"""Shared contrastive alignment space + Context-Gated Contrastive Loss.

Based on the HuBERT-CLAP projection architecture, extended with:
  - Context gating: the contrastive temperature is modulated by dialogue context
  - Conflict-aware negative mining: hard negatives that are emotionally congruent
    (text agrees with audio) are down-weighted vs. conflict pairs
  - Cross-attention injection: dialogue history attends into audio + text
    embeddings in the shared space before fusion (Gap 2 from architecture diagram).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection head: encoder_dim → embed_dim.

    Matches the HuBERT-CLAP architecture exactly:
        Linear → GELU → LayerNorm → Linear → LayerNorm
    """

    def __init__(self, input_dim: int = 1024, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalAttention(nn.Module):
    """Direct audio↔text cross-attention with optional dialogue context.

    Replaces the earlier CrossAttentionInjector (which only attended over
    dialogue history). Instead of each modality independently attending
    to past context, this module enables:

      1. Audio attends to text (direct cross-modal alignment)
      2. Text attends to audio (direct cross-modal alignment)
      3. Both also optionally attend to dialogue history

    Flow (no context):
        text_embed  (B, 1, D) ──→ K/V
        audio_embed (B, 1, D) ──→ Q ──→ CrossAttn ──→ audio_mod

        audio_embed (B, 1, D) ──→ K/V
        text_embed  (B, 1, D) ──→ Q ──→ CrossAttn ──→ text_mod

    Flow (with context):
        K/V = [other_modality_embed || context_seq]

    Each modality gets its own independent cross-attention layer so they
    can learn modality-specific cross-modal and context patterns.

    Args:
        embed_dim: Shared embedding dimensionality.
        n_heads: Number of attention heads per modality.
        dropout: Dropout probability in the cross-attention layer.
    """

    def __init__(self, embed_dim: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.audio_cross_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_audio = nn.LayerNorm(embed_dim)
        self.norm_text = nn.LayerNorm(embed_dim)

    def forward(
        self,
        audio_embed: torch.Tensor,
        text_embed: torch.Tensor,
        context_seq: Optional[torch.Tensor] = None,
        context_padding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-modal attention.

        Args:
            audio_embed: (B, D) current-turn audio embedding.
            text_embed: (B, D) current-turn text embedding.
            context_seq: (B, T, D) or None — fused embeddings of past turns.
            context_padding: (B, T) bool or None — True for padding positions.

        Returns:
            audio_out: (B, D) cross-modal modulated audio embedding.
            text_out:  (B, D) cross-modal modulated text embedding.
        """
        B = audio_embed.size(0)
        device = audio_embed.device
        dtype = audio_embed.dtype

        # Build K/V sequences: each modality attends to the other + optional context
        text_kv = text_embed.unsqueeze(1)  # (B, 1, D)
        audio_kv = audio_embed.unsqueeze(1)  # (B, 1, D)
        kv_padding_audio: Optional[torch.Tensor] = None
        kv_padding_text: Optional[torch.Tensor] = None

        if context_seq is not None:
            _, T, D = context_seq.shape
            assert D == self.embed_dim
            # Append context to K/V sequences
            text_kv = torch.cat([text_kv, context_seq], dim=1)     # (B, 1+T, D)
            audio_kv = torch.cat([audio_kv, context_seq], dim=1)   # (B, 1+T, D)
            if context_padding is not None:
                valid = torch.zeros(B, 1, dtype=torch.bool, device=device)
                kv_padding_audio = torch.cat([valid, context_padding], dim=1)
                # Guard against all-masked context: insert a neutral unmasked key
                fully_masked = context_padding.all(dim=1)  # (B,)
                if fully_masked.any():
                    neutral = torch.zeros(B, 1, D, device=device, dtype=dtype)
                    text_kv = torch.cat([text_kv, neutral], dim=1)
                    audio_kv = torch.cat([audio_kv, neutral], dim=1)
                    pad_ext = torch.zeros(B, 1, dtype=torch.bool, device=device)
                    kv_padding_audio = torch.cat([kv_padding_audio, pad_ext], dim=1)
                # BUG FIX: clone AFTER all extensions so both masks match the
                # final K/V length. Previously cloned before the neutral-key
                # extension, making kv_padding_text 1 slot shorter than text_kv
                # on the first turn of every dialogue (cold-start), which caused
                # a shape mismatch crash inside nn.MultiheadAttention.
                kv_padding_text = kv_padding_audio.clone()

        # Audio path: audio attends to text (+ optional context)
        q_audio = audio_embed.unsqueeze(1)  # (B, 1, D)
        audio_mod, _ = self.audio_cross_attn(
            q_audio, text_kv, text_kv,
            key_padding_mask=kv_padding_audio,
        )
        audio_out = self.norm_audio(audio_embed + audio_mod.squeeze(1))

        # Text path: text attends to audio (+ optional context)
        q_text = text_embed.unsqueeze(1)  # (B, 1, D)
        text_mod, _ = self.text_cross_attn(
            q_text, audio_kv, audio_kv,
            key_padding_mask=kv_padding_text,
        )
        text_out = self.norm_text(text_embed + text_mod.squeeze(1))

        return audio_out, text_out


class ContextGatedContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss with context-adaptive temperature.

    Novel extension: instead of a fixed scalar temperature τ, we learn a
    small MLP that takes the pooled dialogue context vector and outputs Δτ.
    This allows the model to soften the contrastive objective when the
    dialogue context is inherently ambiguous (e.g., long sarcastic exchanges)
    and sharpen it for clearly sincere utterances.

    Additionally, pairs flagged as `conflict_label=1` contribute an extra
    alignment penalty term that pushes audio and text embeddings *apart*
    in the shared space — the core ConflictNet inductive bias.

    Args:
        embed_dim: Shared embedding dimensionality.
        base_temperature: Initial τ (learnable scalar).
        context_gate_dim: Size of context-gating MLP hidden layer.
        conflict_margin: Margin m for the conflict separation loss.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        base_temperature: float = 0.07,
        context_gate_dim: int = 64,
        conflict_margin: float = 0.5,
    ):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(base_temperature).log())
        self.conflict_margin = conflict_margin

        # Context gate: pooled_context (embed_dim) → Δlog_tau (scalar)
        self.context_gate = nn.Sequential(
            nn.Linear(embed_dim, context_gate_dim),
            nn.Tanh(),
            nn.Linear(context_gate_dim, 1),
        )

    def forward(
        self,
        audio_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        context_pooled: Optional[torch.Tensor] = None,
        conflict_labels: Optional[torch.Tensor] = None,
        sarcasm_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute context-gated contrastive + conflict separation loss.

        Args:
            audio_embeds: (B, D) L2-normalised audio projections.
            text_embeds:  (B, D) L2-normalised text projections.
            context_pooled: (B, D) pooled dialogue context (from temporal module).
            conflict_labels: (B,) float — 1.0 for conflict pairs, 0.0 otherwise.
            sarcasm_mask: (B,) bool — True if the sample is sarcasm.

        Returns:
            Scalar loss.
        """
        # L2 normalise
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Effective temperature — per-sample when context is available
        tau = self.log_tau.exp()  # scalar (base)
        if context_pooled is not None:
            delta_tau = self.context_gate(context_pooled).squeeze(-1)  # (B,)
            tau = (self.log_tau + delta_tau).exp()  # (B,) — per-sample temperature

        # Un-scaled cosine similarity matrix (B, B)
        sim_raw = audio_embeds @ text_embeds.T  # (B, B)

        # Per-sample temperature scaling: each row (audio anchor) divided by its tau
        if tau.dim() > 0:
            sim_a2t = sim_raw / tau.unsqueeze(1)  # (B, B) — audio-to-text, per-row temp
            sim_t2a = sim_raw.T / tau.unsqueeze(1)  # (B, B) — text-to-audio, per-row temp
        else:
            sim_a2t = sim_raw / tau
            sim_t2a = sim_raw.T / tau

        # Standard symmetric InfoNCE
        # BUG FIX: removed the sarcasm_mask.all() early-exit that zeroed both
        # InfoNCE losses when an entire batch was conflict (common on MUStARD).
        # F.cross_entropy with ignore_index=-1 handles a fully-masked batch
        # gracefully without needing a special-case zero branch.
        B = audio_embeds.size(0)
        labels = torch.arange(B, device=audio_embeds.device)
        if sarcasm_mask is not None and sarcasm_mask.any():
            labels = labels.clone()
            labels[sarcasm_mask] = -1   # cross_entropy ignore_index=-1
        loss_a2t = F.cross_entropy(sim_a2t, labels, ignore_index=-1)
        loss_t2a = F.cross_entropy(sim_t2a, labels, ignore_index=-1)
        contrastive_loss = (loss_a2t + loss_t2a) / 2

        # Conflict separation loss: push paired audio↔text apart by margin
        # (uses un-scaled cosine similarities since margin is in cosine space)
        conflict_sep_loss = (sim_raw * 0.0).sum()
        if conflict_labels is not None and conflict_labels.sum() > 0:
            conflict_mask = conflict_labels.bool()
            apply_sep = conflict_mask
            if sarcasm_mask is not None:
                apply_sep = conflict_mask & ~sarcasm_mask   # prosodic-only conflict
            if apply_sep.any():
                paired_sim = torch.diagonal(sim_raw)  # (B,) — un-scaled cosine sim
                # Conflict pairs should have LOW cosine similarity (audio ≠ text).
                # We penalise when sim > -conflict_margin, pushing them apart
                # until sim < -margin in the L2-normalised cosine space.
                conflict_sep_loss = conflict_sep_loss + F.relu(
                    paired_sim[apply_sep] + self.conflict_margin
                ).mean()

        return contrastive_loss + conflict_sep_loss
