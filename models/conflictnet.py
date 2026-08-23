"""ConflictNet v2 — full model assembly.

Combines all components into a single nn.Module:
  1. Audio encoder (Emotion2Vec / WavLM / wav2vec2)
  2. Text encoder  (DeBERTa-v3 + LoRA)
  3. Projection heads → shared 256-d space
  4. Speaker normalizer (ECAPA-TDNN + prosody z-score)
  5. Temporal context module (causal Transformer)
  6. Conflict classifier (multi-label subtype + severity)
  7. Word-level divergence features (optional, requires MFA)

Forward pass returns a ConflictNetOutput dataclass.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import build_audio_encoder, DeBERTaEncoder
from .speaker_norm import SpeakerNormalizer
from .temporal import TransformerTemporalContext
from .alignment import ProjectionHead, ContextGatedContrastiveLoss, CrossModalAttention
from .alignment.word_divergence import WordLevelDivergence
from .classifier import ConflictClassifier

logger = logging.getLogger(__name__)

def focal_bce_loss(logits, targets, alpha=0.75, gamma=2.0, pos_weight=None):
    """Focal loss: down-weights easy negatives, focuses on hard sarcasm cases."""
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1 - probs)   # p_t
    focal_weight = (1 - pt) ** gamma
    
    if pos_weight is not None:
        alpha_weight = torch.where(targets > 0.5, pos_weight, torch.ones_like(targets))
    else:
        alpha_weight = torch.where(targets > 0.5,
                                   torch.full_like(targets, alpha),
                                   torch.full_like(targets, 1 - alpha))
    return (alpha_weight * focal_weight * bce).mean()

# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ConflictNetOutput:
    """All outputs from a ConflictNet forward pass."""
    # Core predictions
    logits_type: torch.Tensor       # (B, n_types) — raw BCE logits
    probs_type: torch.Tensor        # (B, n_types) — sigmoid probabilities
    severity: Optional[torch.Tensor]          # (B, 1) or None
    conflict_flag: torch.Tensor     # (B,) bool

    # Embeddings for loss computation and attribution
    audio_embed: torch.Tensor       # (B, embed_dim) projected audio
    text_embed: torch.Tensor        # (B, embed_dim) projected text
    speaker_feat: torch.Tensor      # (B, embed_dim) speaker projection
    fused_embed: torch.Tensor       # (B, embed_dim) post-fusion, pre-temporal
    context_pooled: torch.Tensor    # (B, embed_dim) temporal context pooled

    # Per-turn context (when operating in dialogue mode)
    per_turn_context: Optional[torch.Tensor]  # (B, T, embed_dim)

    # Word divergence features (if MFA available)
    word_div_feats: Optional[torch.Tensor]    # (B, 8)

    # Loss (computed if labels provided)
    loss: Optional[torch.Tensor] = None
    loss_breakdown: Optional[Dict[str, torch.Tensor]] = None


# ---------------------------------------------------------------------------
# Self-supervised swap pre-training objective
# ---------------------------------------------------------------------------

class SwapPretrainingObjective(nn.Module):
    """Self-supervised objective: detect swapped audio↔text pairs.

    Randomly swaps audio OR text (with equal probability) so paired
    audio and text come from *different* utterances. The model must
    classify each pair as matched (0) or swapped (1). This forces
    cross-modal alignment without any conflict labels.

    Using both audio-swap and text-swap prevents the model from
    learning a trivial text-only shortcut.

    Applied during pre-training epochs only.
    """

    def __init__(self, embed_dim: int = 256, swap_prob: float = 0.3):
        super().__init__()
        self.swap_prob = swap_prob
        self.swap_classifier = nn.Linear(embed_dim * 2, 1)

    def forward(
        self,
        audio_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Return BCE loss for swap detection (audio-swap or text-swap)."""
        B = audio_embeds.size(0)
        device = audio_embeds.device
        swap_mask = torch.rand(B, device=device) < self.swap_prob
        swap_labels = swap_mask.float()

        if B < 2:
            swap_mask.fill_(False)
            swap_labels.fill_(0.0)

        if not swap_mask.any():
            pair_feat = torch.cat([audio_embeds, text_embeds], dim=-1)
            logits = self.swap_classifier(pair_feat).squeeze(-1)
            return F.binary_cross_entropy_with_logits(logits, swap_labels)

        # Shift by 1 to ensure a derangement (no self-swapping)
        perm = (torch.arange(B, device=device) + 1) % B

        # SPEED FIX: replaced Python loop with vectorized torch.where ops.
        # Old loop built pair_feats item-by-item (O(B) Python overhead).
        # Randomly choose audio-swap or text-swap for swapped positions.
        use_audio_swap = (torch.rand(B, device=device) < 0.5) & swap_mask
        use_text_swap  = (~use_audio_swap) & swap_mask

        # Build audio side: swap with perm[i] where use_audio_swap, else keep original
        audio_side = torch.where(
            use_audio_swap.unsqueeze(-1).expand_as(audio_embeds),
            audio_embeds[perm],
            audio_embeds,
        )
        # Build text side: swap with perm[i] where use_text_swap, else keep original
        text_side = torch.where(
            use_text_swap.unsqueeze(-1).expand_as(text_embeds),
            text_embeds[perm],
            text_embeds,
        )

        pair_feat = torch.cat([audio_side, text_side], dim=-1)
        logits = self.swap_classifier(pair_feat).squeeze(-1)
        return F.binary_cross_entropy_with_logits(logits, swap_labels)


# ---------------------------------------------------------------------------
# Multi-task uncertainty loss balancing (Kendall et al. 2018)
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """Kendall 2018 uncertainty-based multi-task loss weighting.

    Learns log(σ²) per task. Loss = Σ (1/σ²_i) * L_i + log(σ_i).
    No manual weighting needed — σ adapts during training.
    """

    def __init__(self, n_tasks: int = 4):
        super().__init__()
        init = torch.zeros(n_tasks)
        if n_tasks > 2:
            init[2] = 5.0   # severity: e^(-5) ≈ 0.007 weight, effectively disabled
        self.log_vars = nn.Parameter(init)

    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Use log_vars.device as canonical; move each loss to it to avoid
        # device mismatches (fallback tensors may be created on audio.device).
        total = self.log_vars.new_zeros(())  # scalar, same device as parameters
        weights = {}
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            # Always accumulate the log(sigma) regularisation term so log_vars[i]
            # always receives a gradient. Previously a loss.item()==0.0 early-exit
            # was skipping this for disabled tasks (e.g. severity), leaving
            # log_vars[2] frozen at its init value of 5.0 throughout training.
            loss_i = loss.to(self.log_vars.device)
            total = total + precision * loss_i + 0.5 * self.log_vars[i]
            weights[f"sigma_task_{i}"] = torch.exp(self.log_vars[i] * 0.5).item()
        return total, weights


# ---------------------------------------------------------------------------
# ConflictNet full model
# ---------------------------------------------------------------------------

class ConflictNet(nn.Module):
    """ConflictNet v2 — speaker-normalised cross-modal conflict detector.

    Args:
        audio_encoder_name: 'emotion2vec' | 'wavlm' | 'wav2vec2'
        embed_dim: Shared embedding dimensionality (256).
        n_conflict_types: Number of emotion classes (6 — CREMA-D: anger, disgust, fear, happiness, neutral, sadness).
        temporal_n_layers: Layers in the temporal Transformer.
        temporal_n_heads: Attention heads in the temporal Transformer.
        temporal_max_turns: Max dialogue turns in context window.
        use_speaker_norm: Enable ECAPA-TDNN speaker normalization.
        use_word_divergence: Enable MFA word-level divergence features.
        use_swap_pretraining: Enable self-supervised swap objective.
        lora_r: LoRA rank for DeBERTa (0 = full fine-tuning).
        lora_alpha: LoRA scaling.
    """

    def __init__(
        self,
        audio_encoder_name: str = "emotion2vec",
        embed_dim: int = 256,
        n_conflict_types: int = 6,
        temporal_n_layers: int = 2,
        temporal_n_heads: int = 4,
        temporal_max_turns: int = 16,
        use_speaker_norm: bool = True,
        use_temporal: bool = True,
        use_word_divergence: bool = True,
        use_swap_pretraining: bool = True,
        use_cross_attn_injection: bool = True,
        use_speaker_adaptive_threshold: bool = True,
        use_baseline_subtract: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        label_smoothing: float = 0.05,  # aligned with CLI --label_smoothing default
        sarcasm_pos_weight: float = 8.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_speaker_norm = use_speaker_norm
        self.use_temporal = use_temporal
        self.use_word_divergence = use_word_divergence
        self.use_swap_pretraining = use_swap_pretraining
        self.use_cross_attn_injection = use_cross_attn_injection
        self.use_speaker_adaptive_threshold = use_speaker_adaptive_threshold
        self.use_baseline_subtract = use_baseline_subtract
        self.label_smoothing = label_smoothing

        # Uniform pos_weight across all emotion slots — each is a minority class.
        # Old code used sarcasm_pos_weight=8.0 on slot 0 only, which was wrong
        # for MELD/CREMA-D where slot 0 is anger (different class balance).
        pos_w = torch.full((n_conflict_types,), 3.0)
        self.register_buffer("pos_weight", pos_w)

        # 1. Encoders
        self.audio_encoder = build_audio_encoder(audio_encoder_name)
        self.text_encoder = DeBERTaEncoder(
            use_lora=(lora_r > 0),
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        audio_enc_dim = self.audio_encoder.output_dim
        text_enc_dim = self.text_encoder.output_dim

        # 2. Projection heads → shared space
        self.audio_proj = ProjectionHead(input_dim=int(audio_enc_dim), embed_dim=embed_dim)  # type: ignore[arg-type]
        self.text_proj = ProjectionHead(input_dim=int(text_enc_dim), embed_dim=embed_dim)  # type: ignore[arg-type]

        # 3. Speaker normalization (baseline-subtract is an ablation flag)
        self.speaker_norm = SpeakerNormalizer(
            embed_dim=embed_dim,
            use_baseline_subtract=use_baseline_subtract,
        ) if use_speaker_norm else None

        # Gating network: fuse (audio_proj + text_proj + speaker_feat) → fused_embed
        # Input: [audio_proj ∥ text_proj ∥ speaker_feat] = 3 × embed_dim
        fuse_in = embed_dim * 3 if use_speaker_norm else embed_dim * 2
        self.fusion_gate = nn.Sequential(
            nn.Linear(fuse_in, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # 4. Temporal context (optional — disabled in ablation)
        self.temporal = TransformerTemporalContext(
            embed_dim=embed_dim,
            n_layers=temporal_n_layers,
            n_heads=temporal_n_heads,
            max_turns=temporal_max_turns,
        ) if use_temporal else None

        # 4b. Cross-modal attention: audio↔text + optional dialogue history
        #     (requires temporal context when context_seq is needed;
        #      disabled by --no_cross_attn_injection or --no_temporal)
        #     Note: CrossModalAttention works with or without context_seq,
        #     so cross-modal alignment still fires even without temporal.
        self.cross_modal_attn = CrossModalAttention(
            embed_dim=embed_dim,
            n_heads=temporal_n_heads,
        ) if use_cross_attn_injection else None

        # 5. Word-level divergence
        self.word_divergence = WordLevelDivergence(embed_dim=embed_dim) if use_word_divergence else None
        self._word_div_warned = False
        word_div_dim = WordLevelDivergence.DIVERGENCE_FEAT_DIM if use_word_divergence else 0

        # 6. Classifier
        self.classifier = ConflictClassifier(
            embed_dim=embed_dim,
            n_types=n_conflict_types,
            word_div_dim=word_div_dim,
            speaker_adaptive_threshold=use_speaker_adaptive_threshold,
        )

        # 7. Contrastive loss
        self.contrastive_loss_fn = ContextGatedContrastiveLoss(embed_dim=embed_dim)

        # 8. Self-supervised swap objective
        self.swap_objective = SwapPretrainingObjective(embed_dim=embed_dim) if use_swap_pretraining else None

        # 9. Multi-task loss balancing
        # Tasks: [contrastive, conflict_type, severity, swap]
        n_tasks = 4 if use_swap_pretraining else 3
        self.multi_task_loss = MultiTaskLoss(n_tasks=n_tasks)

        logger.info(
            f"[ConflictNet] audio={audio_encoder_name} | embed_dim={embed_dim} | "
            f"speaker_norm={use_speaker_norm} | word_div={use_word_divergence} | "
            f"temporal={temporal_n_layers}L×{temporal_n_heads}H"
        )

    # ------------------------------------------------------------------
    # Single-utterance forward (no dialogue context)
    # ------------------------------------------------------------------

    def encode(
        self,
        audio: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        prosody_z: Optional[torch.Tensor] = None,
        return_frames: bool = False,
        return_tokens: bool = False,
        precomputed_audio_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode audio and text to shared space, apply speaker normalization.

        Args:
            audio: (B, T) waveform tensor.
            input_ids, attention_mask: Tokenised text.
            audio_attention_mask: (B, T) bool — True for valid audio samples.
            prosody_z: Pre-computed prosody z-scores (B, 3).
            return_frames: If True, also return audio frame-level embeddings.
            return_tokens: If True, also return text token-level embeddings.
            precomputed_audio_embed: (B, audio_enc_dim) — pre-extracted audio
                embedding to bypass the audio encoder (used when FunASR is
                pre-extracted outside DDP scope to prevent NCCL timeout).

        Returns:
            audio_embed: (B, embed_dim)
            text_embed:  (B, embed_dim)
            speaker_feat: (B, embed_dim)
            audio_frames: (B, T_audio, D) or None
            text_tokens:  (B, L_text, D) or None
        """
        # Audio path
        if precomputed_audio_embed is not None:
            # Use pre-extracted embedding; frame-level features are unavailable
            # in this path (FunASR does not return frame-level hidden states).
            audio_raw = precomputed_audio_embed
            audio_frames = None
        else:
            audio_raw = self.audio_encoder(audio, attention_mask=audio_attention_mask, return_frames=return_frames)
            if return_frames:
                audio_raw, audio_frames = audio_raw
            else:
                audio_frames = None
        audio_embed = self.audio_proj(audio_raw)        # (B, embed_dim)

        # Project frame-level embeddings to embed_dim for word divergence
        if return_frames and audio_frames is not None:
            audio_frames = self.audio_proj(audio_frames)  # (B, T, D_raw) -> (B, T, embed_dim)

        # Text path
        text_raw = self.text_encoder(input_ids, attention_mask, return_tokens=return_tokens)
        if return_tokens:
            text_raw, text_tokens = text_raw
        else:
            text_tokens = None
        text_embed = self.text_proj(text_raw)                     # (B, embed_dim)

        # Project token-level embeddings to embed_dim for word divergence
        if return_tokens and text_tokens is not None:
            text_tokens = self.text_proj(text_tokens)  # (B, L, D_raw) -> (B, L, embed_dim)

        # Speaker path — pure-torch, prosody_z is pre-computed
        if self.speaker_norm is not None:
            _, speaker_feat = self.speaker_norm(
                audio=audio,
                prosody_z=prosody_z,  # may be None → uses zeros inside
            )
        else:
            speaker_feat = torch.zeros_like(audio_embed)

        return audio_embed, text_embed, speaker_feat, audio_frames, text_tokens

    def fuse(
        self,
        audio_embed: torch.Tensor,
        text_embed: torch.Tensor,
        speaker_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse audio, text, speaker embeddings via gated MLP."""
        if self.use_speaker_norm:
            combined = torch.cat([audio_embed, text_embed, speaker_feat], dim=-1)
        else:
            combined = torch.cat([audio_embed, text_embed], dim=-1)
        return self.fusion_gate(combined)  # (B, embed_dim)

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        # Per-utterance inputs (current turn)
        audio: torch.Tensor,              # (B, T_audio)
        input_ids: torch.Tensor,          # (B, seq_len)
        attention_mask: torch.Tensor,     # (B, seq_len)
        audio_attention_mask: Optional[torch.Tensor] = None,  # (B, T_audio) bool
        # Dialogue context (optional)
        context_embeds: Optional[torch.Tensor] = None,    # (B, T_turns, embed_dim)
        context_padding: Optional[torch.Tensor] = None,   # (B, T_turns) bool
        speaker_roles: Optional[torch.Tensor] = None,     # (B, T_turns) int
        # Speaker normalization — pass pre-computed tensor from collate_fn
        prosody_z: Optional[torch.Tensor] = None,         # (B, 3) on device
        # Word-level divergence inputs (optional, requires MFA alignment)
        word_timestamps: Optional[List[List[Tuple[float, float]]]] = None,
        token_word_boundaries: Optional[List[List[Tuple[int, int]]]] = None,
        # Supervision
        conflict_type_labels: Optional[torch.Tensor] = None,  # (B, n_types) multi-hot
        severity_labels: Optional[torch.Tensor] = None,       # (B, 1)
        conflict_binary_labels: Optional[torch.Tensor] = None, # (B,) for contrastive
        pretraining: bool = False,
        dataset_names: Optional[List[str]] = None,
        # Pre-extracted audio embedding (bypasses audio encoder inside DDP scope)
        # Set by trainer when audio_encoder._backend == "funasr" to prevent NCCL
        # timeout: FunASR's sequential per-sample inference holds the thread long
        # enough that the faster rank's gradient all-reduce times out.
        precomputed_audio_embed: Optional[torch.Tensor] = None,
    ) -> ConflictNetOutput:

        # 1. Encode (all pure-torch — numpy preprocessing done in collate_fn)
        need_frames = self.word_divergence is not None and word_timestamps is not None
        audio_embed, text_embed, speaker_feat, audio_frames, text_tokens = self.encode(
            audio, input_ids, attention_mask,
            audio_attention_mask=audio_attention_mask,
            prosody_z=prosody_z,
            return_frames=need_frames,
            return_tokens=need_frames,
            precomputed_audio_embed=precomputed_audio_embed,
        )

        # 2. Cross-modal attention: audio↔text BEFORE fusion (+ optional dialogue context)
        #    context_embeds is (B, T, D) or None; CrossModalAttention handles cold-start.
        if self.cross_modal_attn is not None:
            audio_embed, text_embed = self.cross_modal_attn(
                audio_embed, text_embed,
                context_seq=context_embeds,
                context_padding=context_padding,
            )

        # 3. Fuse current turn
        fused_embed = self.fuse(audio_embed, text_embed, speaker_feat)  # (B, D)

        # 4. Temporal context (optional — skip if disabled for ablation)
        if self.temporal is not None:
            current_turn = fused_embed.unsqueeze(1)  # (B, 1, D)
            if context_embeds is not None:
                turn_seq = torch.cat([context_embeds.to(device=fused_embed.device), current_turn], dim=1)
                if context_padding is not None:
                    curr_pad = torch.zeros(fused_embed.size(0), 1, dtype=torch.bool, device=fused_embed.device)
                    pad_mask = torch.cat([context_padding.to(device=fused_embed.device), curr_pad], dim=1)
                else:
                    pad_mask = None
            else:
                turn_seq = current_turn
                pad_mask = None

            per_turn_ctx, context_pooled = self.temporal(
                turn_seq, padding_mask=pad_mask, speaker_roles=speaker_roles
            )
            current_ctx = per_turn_ctx[:, -1, :]  # last position
        else:
            per_turn_ctx = fused_embed.unsqueeze(1)
            context_pooled = fused_embed
            current_ctx = fused_embed

        # 4. Word-level divergence features
        word_div_feats = None
        if (
            self.word_divergence is not None
            and need_frames
            and audio_frames is not None
            and text_tokens is not None
        ):
            assert word_timestamps is not None and token_word_boundaries is not None
            word_div_feats = self.word_divergence.forward_from_encoder_hidden(
                audio_frame_embeds=audio_frames,
                text_token_embeds=text_tokens,
                word_timestamps=word_timestamps,
                token_word_boundaries=token_word_boundaries,
            )
        elif self.word_divergence is not None and not self._word_div_warned:
            logger.warning(
                "WordDivergence unavailable for this batch (missing alignments or "
                "encoder frame features); using zero divergence features."
            )
            self._word_div_warned = True

        # 5. Classify (with speaker-adaptive threshold when speaker_feat available)
        logits_type, probs_type, severity, conflict_flag = self.classifier(
            fused_embed=current_ctx,
            word_div=word_div_feats,
            speaker_feat=speaker_feat,
        )

        # 6. Compute losses if labels provided
        loss = None
        loss_breakdown = None
        if conflict_type_labels is not None or pretraining:
            losses = []

            # 6a. Contrastive loss
            # sarcasm_mask used to exclude sarcasm pairs from InfoNCE (they're
            # intentionally audio≠text, so forcing alignment would be wrong).
            # BUG FIX: was conflict_type_labels[:,0] which is the anger slot —
            # for MELD, any angry (conflict=1) sample got incorrectly excluded.
            # Fixed: use conflict_binary_labels which is dataset-agnostic.
            sarcasm_mask = None
            if conflict_binary_labels is not None:
                sarcasm_mask = conflict_binary_labels.bool()

            cl = self.contrastive_loss_fn(
                audio_embed, text_embed,
                context_pooled=context_pooled,
                conflict_labels=conflict_binary_labels,
                sarcasm_mask=sarcasm_mask,
            )
            losses.append(cl)

            # 6b. Multi-label BCE loss for conflict types (with label smoothing)
            # BUG FIX: old code gated the entire loss on dataset_names == "mustard",
            # giving MELD/CREMA-D/IEMOCAP samples a ZERO gradient on their emotion
            # labels. Now all datasets use the same focal BCE path uniformly.
            if conflict_type_labels is not None:
                eps = self.label_smoothing
                smooth_labels = conflict_type_labels.float().clamp(eps, 1.0 - eps)

                # Focal loss on all conflict-emotion slots (anger, disgust, fear = 0,1,2)
                # which are the minority classes across all datasets.
                type_loss = focal_bce_loss(
                    logits_type,
                    smooth_labels,
                    alpha=0.75,
                    gamma=2.0,
                    # self.pos_weight is already a registered buffer — PyTorch
                    # moves it to the model device automatically via .to(device).
                    # Calling .to(audio.device) inside forward() created a
                    # temporary tensor every step (minor but wasteful).
                    pos_weight=self.pos_weight,
                )
                losses.append(type_loss)
            else:
                losses.append((logits_type * 0.0).sum())


            # 6c. Severity MSE loss
            if severity is not None and severity_labels is not None:
                sev_target = severity_labels.float().view(-1)
                sev_pred = severity.view(-1)
                sev_loss = nn.functional.mse_loss(sev_pred, sev_target)
                losses.append(sev_loss)
            else:
                losses.append((severity * 0.0).sum() if severity is not None else torch.tensor(0.0, device=audio.device))

            # 6d. Self-supervised swap loss (pre-training only)
            if self.swap_objective is not None:
                swap_loss = self.swap_objective(audio_embed, text_embed)
                losses.append(swap_loss)

            loss, sigma_weights = self.multi_task_loss(losses)
            loss_breakdown = {
                "contrastive": losses[0].detach().item(),
                "type_bce": losses[1].detach().item(),
                "severity_mse": losses[2].detach().item(),
                **sigma_weights,
            }
            if self.swap_objective is not None:
                loss_breakdown["swap"] = losses[3].detach().item()

        return ConflictNetOutput(
            logits_type=logits_type,
            probs_type=probs_type,
            severity=severity,
            conflict_flag=conflict_flag,
            audio_embed=audio_embed,
            text_embed=text_embed,
            speaker_feat=speaker_feat,
            fused_embed=fused_embed,
            context_pooled=context_pooled,
            per_turn_context=per_turn_ctx,
            word_div_feats=word_div_feats,
            loss=loss,
            loss_breakdown=loss_breakdown,
        )

    def count_parameters(self) -> Dict[str, Dict[str, int]]:
        """Return parameter counts per module."""
        result = {}
        for name, module in self.named_children():
            n_total = sum(p.numel() for p in module.parameters())
            n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            result[name] = {"total": n_total, "trainable": n_train}
        return result
