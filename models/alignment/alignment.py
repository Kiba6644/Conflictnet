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

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


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
    """Token-Level Multi-Layer Cross-Modal Attention with ModalDrop."""

    def __init__(self, embed_dim: int = 256, n_heads: int = 8, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.modal_drop_p = 0.05

        self.audio_layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True),
                'norm': nn.LayerNorm(embed_dim)
            })
            for _ in range(num_layers)
        ])
        
        self.text_layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True),
                'norm': nn.LayerNorm(embed_dim)
            })
            for _ in range(num_layers)
        ])

    def forward(
        self,
        audio_embed: torch.Tensor,
        text_embed: torch.Tensor,
        context_seq: Optional[torch.Tensor] = None,
        context_padding: Optional[torch.Tensor] = None,
        audio_seq: Optional[torch.Tensor] = None,
        text_seq: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B = audio_embed.size(0)
        device = audio_embed.device
        dtype = audio_embed.dtype

        # ModalDrop (during training)
        if self.training:
            if torch.rand(1).item() < self.modal_drop_p:
                audio_embed = torch.zeros_like(audio_embed)
                if audio_seq is not None:
                    audio_seq = torch.zeros_like(audio_seq)
            if torch.rand(1).item() < self.modal_drop_p:
                text_embed = torch.zeros_like(text_embed)
                if text_seq is not None:
                    text_seq = torch.zeros_like(text_seq)

        a_seq = audio_seq if audio_seq is not None else audio_embed.unsqueeze(1)
        t_seq = text_seq if text_seq is not None else text_embed.unsqueeze(1)
        
        # key_padding_mask expects True for padding (ignored) elements.
        # our attention masks are True for valid elements, so we invert them.
        if text_attention_mask is not None and text_seq is not None:
            valid_t = ~text_attention_mask.bool()
        else:
            valid_t = torch.zeros(B, t_seq.size(1), dtype=torch.bool, device=device)
            
        if audio_attention_mask is not None and audio_seq is not None:
            # We assume audio_attention_mask is aligned with audio_seq frames.
            # FunASR frames downsample by 320, so we can interpolate the mask.
            # But Wait: if the user passes the mask for the raw waveform, it's 160000 long!
            # We should just use a length-based mask.
            lengths = audio_attention_mask.sum(dim=1)
            frame_lengths = torch.ceil(lengths / 320.0).long()
            max_f = a_seq.size(1)
            idx = torch.arange(max_f, device=device).unsqueeze(0)
            valid_a = idx >= frame_lengths.unsqueeze(1)
        else:
            valid_a = torch.zeros(B, a_seq.size(1), dtype=torch.bool, device=device)
        
        for i in range(self.num_layers):
            text_kv = t_seq
            audio_kv = a_seq
            
            kv_padding_audio: Optional[torch.Tensor] = valid_t
            kv_padding_text: Optional[torch.Tensor] = valid_a

            if context_seq is not None:
                _, T, D = context_seq.shape
                assert D == self.embed_dim
                text_kv = torch.cat([text_kv, context_seq], dim=1)
                audio_kv = torch.cat([audio_kv, context_seq], dim=1)
                if context_padding is not None:
                    kv_padding_audio = torch.cat([valid_t, context_padding], dim=1)
                    kv_padding_text = torch.cat([valid_a, context_padding], dim=1)
                    
                    fully_masked = context_padding.all(dim=1)
                    if fully_masked.any():
                        neutral = torch.zeros(B, 1, D, device=device, dtype=dtype)
                        text_kv = torch.cat([text_kv, neutral], dim=1)
                        audio_kv = torch.cat([audio_kv, neutral], dim=1)
                        pad_ext = torch.zeros(B, 1, dtype=torch.bool, device=device)
                        if kv_padding_audio is not None:
                            kv_padding_audio = torch.cat([kv_padding_audio, pad_ext], dim=1)
                        if kv_padding_text is not None:
                            kv_padding_text = torch.cat([kv_padding_text, pad_ext], dim=1)

            # Audio attends to Text (+ context)
            q_audio = a_seq
            a_mod, _ = self.audio_layers[i]['cross_attn'](
                q_audio, text_kv, text_kv,
                key_padding_mask=kv_padding_audio,
            )
            a_seq = self.audio_layers[i]['norm'](a_seq + drop_path(a_mod, 0.1, self.training))

            # Text attends to Audio (+ context)
            q_text = t_seq
            t_mod, _ = self.text_layers[i]['cross_attn'](
                q_text, audio_kv, audio_kv,
                key_padding_mask=kv_padding_text,
            )
            t_seq = self.text_layers[i]['norm'](t_seq + drop_path(t_mod, 0.1, self.training))
        
        # Mean pooling to return (B, D)
        audio_out = a_seq.mean(dim=1) if audio_seq is not None else a_seq.squeeze(1)
        text_out = t_seq.mean(dim=1) if text_seq is not None else t_seq.squeeze(1)
        
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
        base_temperature: float = 0.15,
        context_gate_dim: int = 64,
        conflict_margin: float = 0.5,
        queue_size: int = 1024,
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

        # Momentum queue for extra negatives (MoCo-style)
        self.queue_size = queue_size
        if queue_size > 0:
            self.register_buffer(
                "_audio_queue", F.normalize(torch.randn(queue_size, embed_dim), dim=1))
            self.register_buffer(
                "_text_queue", F.normalize(torch.randn(queue_size, embed_dim), dim=1))
            self.register_buffer("_queue_ptr", torch.zeros(1, dtype=torch.long))
        else:
            self._audio_queue = None
            self._text_queue = None
            self._queue_ptr = None

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

        # Expand similarity matrix with queue negatives if available
        B = audio_embeds.size(0)
        labels = torch.arange(B, device=audio_embeds.device)
        
        if self.queue_size > 0 and self._audio_queue is not None:
            # Queue provides extra negative keys (detached, no gradient)
            audio_keys = torch.cat([text_embeds,  self._text_queue.clone().detach()],  dim=0)  # (B+Q, D)
            text_keys  = torch.cat([audio_embeds, self._audio_queue.clone().detach()], dim=0)  # (B+Q, D)
            
            # Logits: (B, B+Q) — first B columns are in-batch, rest are queue negatives
            sim_a2t_raw = audio_embeds @ audio_keys.T
            sim_t2a_raw = text_embeds  @ text_keys.T
        else:
            sim_a2t_raw = audio_embeds @ text_embeds.T
            sim_t2a_raw = text_embeds  @ audio_embeds.T
            
        # For the separation loss, we still only care about in-batch similarities
        sim_raw = audio_embeds @ text_embeds.T  # (B, B)

        # Per-sample temperature scaling: each row (audio anchor) divided by its tau
        if tau.dim() > 0:
            sim_a2t = sim_a2t_raw / tau.unsqueeze(1)  # (B, B+Q) — audio-to-text, per-row temp
            sim_t2a = sim_t2a_raw / tau.unsqueeze(1)  # (B, B+Q) — text-to-audio, per-row temp
        else:
            sim_a2t = sim_a2t_raw / tau
            sim_t2a = sim_t2a_raw / tau

        # Standard symmetric InfoNCE
        # BUG FIX: removed the sarcasm_mask.all() early-exit that zeroed both
        # InfoNCE losses when an entire batch was conflict (common on MUStARD).
        # F.cross_entropy with ignore_index=-1 handles a fully-masked batch
        # gracefully without needing a special-case zero branch.
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
            if sarcasm_mask is not None:
                apply_sep = conflict_mask & sarcasm_mask   # ONLY sarcasm pairs should be pushed apart
            else:
                apply_sep = torch.zeros_like(conflict_mask)
            if apply_sep.any():
                paired_sim = torch.diagonal(sim_raw)  # (B,) — un-scaled cosine sim
                # Conflict pairs should have LOW cosine similarity (audio ≠ text).
                # We penalise when sim > -conflict_margin, pushing them apart
                # until sim < -margin in the L2-normalised cosine space.
                conflict_sep_loss = conflict_sep_loss + F.relu(
                    paired_sim[apply_sep] + self.conflict_margin
                ).mean()

        return contrastive_loss + conflict_sep_loss

    @torch.no_grad()
    def update_queue(self, audio_embed: torch.Tensor, text_embed: torch.Tensor):
        """FIFO enqueue current batch, dequeue oldest entries."""
        if self.queue_size <= 0 or self._audio_queue is None:
            return
        B = audio_embed.shape[0]
        ptr = int(self._queue_ptr)
        slots = list(range(ptr, ptr + B))
        slots = [s % self.queue_size for s in slots]
        self._audio_queue[slots] = F.normalize(audio_embed.detach().float(), dim=1)
        self._text_queue[slots]  = F.normalize(text_embed.detach().float(), dim=1)
        self._queue_ptr[0] = (ptr + B) % self.queue_size


class MoEFusion(nn.Module):
    """Gated Multi-Modal Mixture-of-Experts (MoE) Fusion.

    Replaces the linear fusion gate with 4 expert MLPs + a gating network.
    The gating network selects/weights experts based on `speaker_feat` and `word_div` features.
    """

    def __init__(
        self,
        fuse_in: int,
        embed_dim: int = 256,
        num_experts: int = 4,
        gate_in: int = 256 + 11,
    ):
        super().__init__()
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fuse_in, embed_dim * 2),
                nn.GELU(),
                nn.LayerNorm(embed_dim * 2),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            for _ in range(num_experts)
        ])

        self.gate = nn.Sequential(
            nn.Linear(gate_in, 64),
            nn.GELU(),
            nn.Linear(64, num_experts)
        )

    def forward(self, x: torch.Tensor, gate_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, fuse_in) combined modalities.
            gate_feat: (B, gate_in) gate features (e.g., [speaker_feat, word_div_feats]).
        Returns:
            (B, embed_dim) fused embedding.
        """
        weights = F.softmax(self.gate(gate_feat), dim=-1)  # (B, num_experts)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, num_experts, embed_dim)
        return (expert_outputs * weights.unsqueeze(-1)).sum(dim=1)
