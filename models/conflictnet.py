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
from .alignment import ProjectionHead, ContextGatedContrastiveLoss, CrossModalAttention, MoEFusion
from .alignment.modality_router import ModalityRouter, entropy_regularization_loss
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

from transformers.utils import ModelOutput

@dataclasses.dataclass
class ConflictNetOutput(ModelOutput):
    """All outputs from a ConflictNet forward pass."""
    # Core predictions
    logits_type: Optional[torch.Tensor] = None       # (B, n_types) — raw BCE logits
    probs_type: Optional[torch.Tensor] = None        # (B, n_types) — sigmoid probabilities
    severity: Optional[torch.Tensor] = None          # (B, 1) or None
    conflict_flag: Optional[torch.Tensor] = None     # (B,) bool

    # Embeddings for loss computation and attribution
    audio_embed: Optional[torch.Tensor] = None       # (B, embed_dim) projected audio
    text_embed: Optional[torch.Tensor] = None        # (B, embed_dim) projected text
    speaker_feat: Optional[torch.Tensor] = None      # (B, embed_dim) speaker projection
    fused_embed: Optional[torch.Tensor] = None       # (B, embed_dim) post-fusion, pre-temporal
    context_pooled: Optional[torch.Tensor] = None    # (B, embed_dim) temporal context pooled

    # Per-turn context (when operating in dialogue mode)
    per_turn_context: Optional[torch.Tensor] = None  # (B, T, embed_dim)

    # Word divergence features (if MFA available)
    word_div_feats: Optional[torch.Tensor] = None    # (B, 8)

    # Adaptive modality router gate weight (None when router is disabled)
    router_alpha: Optional[torch.Tensor] = None  # (B, 1) gate weight α

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

    def __init__(self, n_tasks: int = 4, classification_weight_boost: float = 1.5):
        super().__init__()
        init = torch.zeros(n_tasks)
        if n_tasks > 2:
            init[2] = 5.0   # severity: e^(-5) ≈ 0.007 weight, effectively disabled
        self.log_vars = nn.Parameter(init)
        self.classification_weight_boost = classification_weight_boost

    def forward(
        self,
        losses: List[torch.Tensor],
        active_mask: Optional[List[bool]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Use log_vars.device as canonical; move each loss to it to avoid
        # device mismatches (fallback tensors may be created on audio.device).
        total = self.log_vars.new_zeros(())  # scalar, same device as parameters
        weights = {}
        for i, loss in enumerate(losses):
            weights[f"sigma_task_{i}"] = float(torch.exp(self.log_vars[i] * 0.5).detach().item())
            loss_i = loss.to(self.log_vars.device)
            # Skip inactive tasks (e.g. missing severity or inactive swap) so their
            # uncertainty regulariser terms (0.5 * log_var) do not corrupt total loss
            is_active = (active_mask[i] if active_mask is not None else True) and (loss_i.requires_grad or loss_i.item() > 0)
            if not is_active:
                continue

            # Clamp log_vars to prevent exp() overflow in FP16 (>11 overflows float16)
            log_var_clamped = self.log_vars[i].clamp(min=-6.0, max=6.0)
            precision = torch.exp(-log_var_clamped)

            # Boost emotion classification task (task index 1) so it drives representation learning
            task_mult = self.classification_weight_boost if i == 1 else 1.0

            total = total + task_mult * (precision * loss_i + 0.5 * log_var_clamped)
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
        lora_r: int = 32,
        lora_alpha: int = 32,
        label_smoothing: float = 0.05,  # aligned with CLI --label_smoothing default
        sarcasm_pos_weight: float = 8.0,
        gradient_checkpointing: bool = False,
        unfreeze_audio_layers: int = 0,
        use_adaptive_router: bool = False,
        router_entropy_reg: float = 0.01,
        modality_dropout_prob: float = 0.15,
        use_cross_entropy: bool = True,
        use_class_weights: bool = True,
        contrastive_loss_scale: float = 0.25,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.contrastive_loss_scale = contrastive_loss_scale
        self.use_speaker_norm = use_speaker_norm
        self.use_temporal = use_temporal
        self.use_word_divergence = use_word_divergence
        self.use_swap_pretraining = use_swap_pretraining
        self.use_cross_attn_injection = use_cross_attn_injection
        self.use_speaker_adaptive_threshold = use_speaker_adaptive_threshold
        self.use_baseline_subtract = use_baseline_subtract
        self.label_smoothing = label_smoothing
        self.modality_dropout_prob = modality_dropout_prob
        self.use_cross_entropy = use_cross_entropy
        self.use_class_weights = use_class_weights

        # Balanced pos_weight for MELD/CREMA-D classes: [anger, disgust, fear, joy, neutral, sadness]
        # MELD frequencies: Anger 11%, Disgust 3%, Fear 3%, Joy 17%, Neutral 47%, Sadness 7%
        # pos_weight = (1 - freq) / freq
        # For Neutral: (1 - 0.47) / 0.47 = ~1.1
        # For Disgust: (1 - 0.03) / 0.03 = ~32.0
        # For Anger: (1 - 0.11) / 0.11 = ~8.0
        # Capped inverse-frequency weights for [anger, disgust, fear, joy, neutral, sadness]
        # After surprise->neutral remapping, neutral is ~59% so its weight is reduced.
        # Caps at 3.0 to prevent extreme logit bias under focal loss + argmax eval.
        pos_w = torch.tensor([2.0, 3.0, 3.0, 1.5, 0.6, 2.5])
        # If n_conflict_types is different (e.g. MUStARD + sarcasm), pad with 3.0
        if n_conflict_types != 6:
            pos_w_padded = torch.full((n_conflict_types,), 3.0)
            pos_w_padded[:min(6, n_conflict_types)] = pos_w[:min(6, n_conflict_types)]
            pos_w = pos_w_padded
            
        self.register_buffer("pos_weight", pos_w)

        # Multi-class cross-entropy class weights for [anger, disgust, fear, joy, neutral, sadness]
        # Inverse square-root based for MELD: preserves Neutral recall without sacrificing minority emotion sensitivity
        ce_w = torch.tensor([1.5, 3.0, 3.0, 1.2, 0.75, 1.9])
        if n_conflict_types != 6:
            ce_w_padded = torch.full((n_conflict_types,), 1.0)
            ce_w_padded[:min(6, n_conflict_types)] = ce_w[:min(6, n_conflict_types)]
            ce_w = ce_w_padded
        self.register_buffer("class_weights", ce_w)

        # 1. Encoders
        self.audio_encoder = build_audio_encoder(
            audio_encoder_name,
            gradient_checkpointing=gradient_checkpointing,
            unfreeze_last_n_layers=unfreeze_audio_layers,
        )
        self.text_encoder = DeBERTaEncoder(
            use_lora=(lora_r > 0),
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            gradient_checkpointing=gradient_checkpointing,
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

        # Adaptive modality router (optional — disabled by default)
        self.use_adaptive_router = use_adaptive_router
        self.router_entropy_reg = router_entropy_reg
        self.modality_router = ModalityRouter(embed_dim) if use_adaptive_router else None

        # Gating network: fuse (audio_proj + text_proj + speaker_feat) → fused_embed
        # Input: [audio_proj ∥ text_proj ∥ speaker_feat] = 3 × embed_dim
        fuse_in = embed_dim * 3 if use_speaker_norm else embed_dim * 2
        gate_in = 0
        if use_speaker_norm:
            gate_in += embed_dim
        if use_word_divergence:
            gate_in += 11

        if gate_in > 0:
            self.fusion_gate = MoEFusion(
                fuse_in=fuse_in,
                embed_dim=embed_dim,
                num_experts=4,
                gate_in=gate_in,
            )
        else:
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

        # 6. Classifier (with multimodal skip highway)
        self.classifier = ConflictClassifier(
            embed_dim=embed_dim,
            n_types=n_conflict_types,
            word_div_dim=word_div_dim,
            speaker_adaptive_threshold=use_speaker_adaptive_threshold,
            use_skip_highway=True,
        )

        # 7. Contrastive loss
        self.contrastive_loss_fn = ContextGatedContrastiveLoss(embed_dim=embed_dim)

        # 8. Self-supervised swap objective
        self.swap_objective = SwapPretrainingObjective(embed_dim=embed_dim) if use_swap_pretraining else None

        # 9. Multi-task loss balancing
        # Tasks: [contrastive, conflict_type, severity, swap]
        n_tasks = 4 if use_swap_pretraining else 3
        self.multi_task_loss = MultiTaskLoss(n_tasks=n_tasks, classification_weight_boost=1.5)

        logger.info(
            f"[ConflictNet] audio={audio_encoder_name} | embed_dim={embed_dim} | "
            f"speaker_norm={use_speaker_norm} | word_div={use_word_divergence} | "
            f"temporal={temporal_n_layers}L×{temporal_n_heads}H | "
            f"contrastive_loss_scale={contrastive_loss_scale}"
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
        precomputed_speaker_embed: Optional[torch.Tensor] = None,
        precomputed_audio_frames: Optional[torch.Tensor] = None,
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
            precomputed_audio_frames: (B, T_frames, audio_enc_dim) — pre-extracted
                audio frames to prevent loss of cross-modal alignment accuracy.

        Returns:
            audio_embed: (B, embed_dim)
            text_embed:  (B, embed_dim)
            speaker_feat: (B, embed_dim)
            audio_frames: (B, T_audio, D) or None
            text_tokens:  (B, L_text, D) or None
        """
        # Audio path
        if precomputed_audio_embed is not None:
            # Use pre-extracted embedding and frames
            audio_raw = precomputed_audio_embed
            audio_frames = precomputed_audio_frames
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
            if precomputed_speaker_embed is not None:
                # Bypass ECAPA-TDNN inside DDP by directly providing the precomputed
                # 192-d speaker embedding. This prevents SpeechBrain from deadlocking
                # DDP's asynchronous buffer broadcast.
                spk_embed = precomputed_speaker_embed
                if prosody_z is None:
                    prosody_z = torch.zeros(audio.size(0), self.speaker_norm._prosody_dim, device=audio.device)
                combined = torch.cat([spk_embed, prosody_z], dim=-1)
                speaker_feat = self.speaker_norm.spk_proj(combined)
            else:
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
        word_div_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fuse audio, text, speaker embeddings via gated MLP or MoE.

        When ``self.modality_router`` is enabled, a learned scalar α ∈ [0,1]
        weights text vs audio *before* the fusion gate:
            audio_embed  ← (1 - α) * audio_embed
            text_embed   ←       α * text_embed

        Returns:
            fused_embed: (B, embed_dim)
            router_alpha: (B, 1) or None if router is disabled.
        """
        alpha = None
        if self.modality_router is not None:
            alpha = self.modality_router(text_embed, audio_embed)  # (B, 1)
            # Weighted combination before MoE
            audio_embed = (1 - alpha) * audio_embed
            text_embed = alpha * text_embed

        # Modality Dropout: regularize audio/text representations to prevent text dominance
        if self.training and self.modality_dropout_prob > 0.0:
            B = audio_embed.size(0)
            r = torch.rand(B, 1, device=audio_embed.device)
            p = self.modality_dropout_prob
            audio_mask = (r >= p).float()
            text_mask = ((r < p) | (r >= 2 * p)).float()
            audio_embed = audio_embed * audio_mask
            text_embed = text_embed * text_mask

        if self.use_speaker_norm:
            combined = torch.cat([audio_embed, text_embed, speaker_feat], dim=-1)
        else:
            combined = torch.cat([audio_embed, text_embed], dim=-1)

        if isinstance(self.fusion_gate, MoEFusion):
            gate_feats = []
            if self.use_speaker_norm:
                gate_feats.append(speaker_feat)
            if self.use_word_divergence:
                if word_div_feats is None:
                    word_div_feats = torch.zeros(audio_embed.size(0), 11, device=audio_embed.device)
                gate_feats.append(word_div_feats)
            gate_feat_tensor = torch.cat(gate_feats, dim=-1)
            return self.fusion_gate(combined, gate_feat_tensor), alpha
        else:
            return self.fusion_gate(combined), alpha  # (B, embed_dim), alpha

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
        speaker_roles: Optional[torch.Tensor] = None,     # (B,) or (B, T_turns) int
        context_speaker_roles: Optional[torch.Tensor] = None,  # (B, T_turns) int
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
        # Pre-extracted speaker embedding (bypasses ECAPA-TDNN inside DDP scope)
        precomputed_speaker_embed: Optional[torch.Tensor] = None,
        precomputed_audio_frames: Optional[torch.Tensor] = None,
    ) -> ConflictNetOutput:

        # 1. Encode (all pure-torch — numpy preprocessing done in collate_fn)
        need_frames = (self.word_divergence is not None and word_timestamps is not None) or self.cross_modal_attn is not None
        audio_embed, text_embed, speaker_feat, audio_frames, text_tokens = self.encode(
            audio, input_ids, attention_mask,
            audio_attention_mask=audio_attention_mask,
            prosody_z=prosody_z,
            return_frames=need_frames,
            return_tokens=need_frames,
            precomputed_audio_embed=precomputed_audio_embed,
            precomputed_speaker_embed=precomputed_speaker_embed,
            precomputed_audio_frames=precomputed_audio_frames,
        )

        # 2. Cross-modal attention: audio↔text BEFORE fusion (+ optional dialogue context)
        # BUG FIX: previously cross-modal attention was not masking padding frames/tokens,
        # treating zeros as valid features. We now pass attention masks to prevent this.
        if self.cross_modal_attn is not None:
            audio_embed, text_embed = self.cross_modal_attn(
                audio_embed, text_embed,
                context_seq=context_embeds,
                context_padding=context_padding,
                audio_seq=audio_frames if need_frames else None,
                text_seq=text_tokens if need_frames else None,
                audio_attention_mask=audio_attention_mask,
                text_attention_mask=attention_mask,
            )

        # 3. Word-level divergence features (needed for MoEFusion gate)
        word_div_feats = None
        if (
            self.word_divergence is not None
            and need_frames
            and audio_frames is not None
            and text_tokens is not None
            and word_timestamps is not None
            and token_word_boundaries is not None
        ):
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

        # 4. Fuse current turn
        fused_embed, router_alpha = self.fuse(audio_embed, text_embed, speaker_feat, word_div_feats)  # (B, D)

        # 5. Temporal context (optional — skip if disabled for ablation)
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

            # Multi-party speaker role alignment across dialogue sequence
            full_speaker_roles = None
            if speaker_roles is not None:
                if speaker_roles.dim() == 1:
                    curr_spk = speaker_roles.unsqueeze(1)
                elif speaker_roles.dim() == 2 and speaker_roles.size(1) == 1:
                    curr_spk = speaker_roles
                elif speaker_roles.dim() == 2 and speaker_roles.size(1) == turn_seq.size(1):
                    full_speaker_roles = speaker_roles
                    curr_spk = None
                else:
                    curr_spk = speaker_roles

                if full_speaker_roles is None and curr_spk is not None:
                    if context_embeds is not None:
                        if context_speaker_roles is not None:
                            full_speaker_roles = torch.cat(
                                [context_speaker_roles.to(fused_embed.device), curr_spk.to(fused_embed.device)], dim=1
                            )
                        else:
                            ctx_spk = torch.zeros(
                                curr_spk.size(0), context_embeds.size(1), dtype=curr_spk.dtype, device=fused_embed.device
                            )
                            full_speaker_roles = torch.cat([ctx_spk, curr_spk.to(fused_embed.device)], dim=1)
                    else:
                        full_speaker_roles = curr_spk

            per_turn_ctx, context_pooled = self.temporal(
                turn_seq, padding_mask=pad_mask, speaker_roles=full_speaker_roles
            )
            current_ctx = per_turn_ctx[:, -1, :]  # last position
        else:
            per_turn_ctx = fused_embed.unsqueeze(1)
            context_pooled = fused_embed
            current_ctx = fused_embed

        # 5. Classify (with multimodal skip highway and speaker-adaptive threshold)
        logits_type, probs_type, severity, conflict_flag = self.classifier(
            fused_embed=current_ctx,
            word_div=word_div_feats,
            speaker_feat=speaker_feat,
            dataset_names=dataset_names,
            audio_embed=audio_embed,
            text_embed=text_embed,
        )

        # 6. Compute losses if labels provided
        loss = None
        loss_breakdown = None
        if conflict_type_labels is not None or pretraining:
            losses = []

            # 6a. Supervised Multimodal Contrastive loss (SupCon) / InfoNCE
            # sarcasm_mask used to exclude sarcasm pairs from InfoNCE (they're
            # intentionally audio≠text, so forcing alignment would be wrong).
            # BUG FIX: was conflict_type_labels[:,0] which is the anger slot —
            # for MELD, any angry (conflict=1) sample got incorrectly excluded.
            # Fixed: use conflict_binary_labels which is dataset-agnostic.
            sarcasm_mask = None
            if conflict_binary_labels is not None:
                if dataset_names is not None:
                    # Only MUStARD and CASE contain sarcasm pairs (intentionally mismatched).
                    # For MELD/CREMA-D, angry/fearful speech is aligned, so do NOT mask it.
                    is_sarcasm = torch.tensor(
                        [n in ("mustard", "case") for n in dataset_names],
                        device=conflict_binary_labels.device
                    )
                    sarcasm_mask = conflict_binary_labels.bool() & is_sarcasm
                else:
                    sarcasm_mask = torch.zeros_like(conflict_binary_labels, dtype=torch.bool)

            cl = self.contrastive_loss_fn(
                audio_embed, text_embed,
                context_pooled=context_pooled,
                conflict_labels=conflict_binary_labels,
                sarcasm_mask=sarcasm_mask,
                emotion_labels=conflict_type_labels,
            )
            cl = cl * self.contrastive_loss_scale
            losses.append(cl)

            # 6b. Classification loss for conflict types
            # For single-label emotion datasets (MELD, CREMA-D, IEMOCAP), use multi-class
            # Cross-Entropy loss with inverse-frequency class weighting and label smoothing.
            # For multi-label datasets (MUStARD, CASE), preserve multi-label Focal BCE loss.
            if conflict_type_labels is not None:
                is_single_label_batch = False
                if dataset_names is not None:
                    is_single_label_batch = all(n in ("meld", "cremad", "iemocap") for n in dataset_names)
                else:
                    is_single_label_batch = bool((conflict_type_labels.sum(dim=-1) <= 1.0 + 1e-4).all())

                if is_single_label_batch and self.use_cross_entropy:
                    target_cls = conflict_type_labels.argmax(dim=-1)
                    weights = self.class_weights if self.use_class_weights else None
                    type_loss = nn.functional.cross_entropy(
                        logits_type,
                        target_cls,
                        weight=weights,
                        label_smoothing=self.label_smoothing,
                    )
                    losses.append(type_loss)
                elif dataset_names is not None and any(n in ("meld", "cremad", "iemocap") for n in dataset_names) and self.use_cross_entropy:
                    sl_mask = torch.tensor(
                        [n in ("meld", "cremad", "iemocap") for n in dataset_names],
                        device=logits_type.device,
                        dtype=torch.bool,
                    )
                    weights = self.class_weights if self.use_class_weights else None
                    ce_loss = nn.functional.cross_entropy(
                        logits_type[sl_mask],
                        conflict_type_labels[sl_mask].argmax(dim=-1),
                        weight=weights,
                        label_smoothing=self.label_smoothing,
                    )
                    ml_mask = ~sl_mask
                    eps = self.label_smoothing
                    smooth_labels = conflict_type_labels[ml_mask].float().clamp(eps, 1.0 - eps)
                    bce_loss = focal_bce_loss(
                        logits_type[ml_mask],
                        smooth_labels,
                        alpha=0.75,
                        gamma=2.0,
                        pos_weight=self.pos_weight,
                    )
                    losses.append((ce_loss * sl_mask.sum() + bce_loss * ml_mask.sum()) / logits_type.size(0))
                else:
                    eps = self.label_smoothing
                    smooth_labels = conflict_type_labels.float().clamp(eps, 1.0 - eps)
                    type_loss = focal_bce_loss(
                        logits_type,
                        smooth_labels,
                        alpha=0.75,
                        gamma=2.0,
                        pos_weight=self.pos_weight,
                    )
                    losses.append(type_loss)
            else:
                losses.append((logits_type * 0.0).sum())


            # 6c. Severity MSE loss
            has_real_severity = any(d in ("iemocap", "cremad") for d in (dataset_names or []))
            if severity is not None and severity_labels is not None and has_real_severity:
                sev_target = severity_labels.float().view(-1)
                sev_pred = severity.view(-1)
                sev_loss = nn.functional.mse_loss(sev_pred, sev_target)
                losses.append(sev_loss)
            else:
                losses.append(torch.zeros(1, device=audio.device if audio is not None else "cpu").squeeze())

            # 6d. Self-supervised swap loss (pre-training phase only)
            if self.swap_objective is not None and pretraining:
                swap_loss = self.swap_objective(audio_embed, text_embed)
                losses.append(swap_loss)
            elif self.swap_objective is not None:
                swap_loss = (audio_embed * 0.0).sum() + sum((p * 0.0).sum() for p in self.swap_objective.parameters())
                losses.append(swap_loss)

            active_mask = [
                True,  # 0: contrastive
                conflict_type_labels is not None,  # 1: classification
                bool(has_real_severity and severity is not None and severity_labels is not None),  # 2: severity
            ]
            if self.swap_objective is not None:
                active_mask.append(bool(pretraining))

            loss, sigma_weights = self.multi_task_loss(losses, active_mask=active_mask)
            loss_breakdown = {
                "contrastive": losses[0].detach(),
                "type_bce": losses[1].detach(),
                "severity_mse": losses[2].detach(),
                **sigma_weights,
            }
            if self.swap_objective is not None:
                loss_breakdown["swap"] = losses[3].detach()

            # 6e. Router entropy regularisation (penalises hard 0/1 collapse)
            if self.modality_router is not None and router_alpha is not None:
                router_ent_loss = entropy_regularization_loss(router_alpha)
                # Subtract to MAXIMIZE entropy: prevents alpha from collapsing to 0 or 1
                # which would zero out one entire modality (audio or text).
                loss = loss - self.router_entropy_reg * router_ent_loss
                loss_breakdown["router_entropy"] = router_ent_loss.detach()

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
            router_alpha=router_alpha,
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
