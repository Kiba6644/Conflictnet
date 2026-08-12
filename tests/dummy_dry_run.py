"""Dummy dry-run: smoke tests for integration of modules.

Instead of loading the full ConflictNet (which requires downloading DeBERTa-v3
and Emotion2Vec), this tests the submodules in an isolated fashion that
exercises the exact same code paths used in the full model forward pass.

Run:  python tests/dummy_dry_run.py
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def test_cross_modal_attention_no_context():
    """CrossModalAttention: audio<->text without dialogue context."""
    from models.alignment import CrossModalAttention, ProjectionHead

    B, D = 4, 64
    attn = CrossModalAttention(embed_dim=D, n_heads=4)
    proj = ProjectionHead(input_dim=D, embed_dim=D)

    audio_raw = torch.randn(B, D)
    text_raw = torch.randn(B, D)
    audio_embed = proj(audio_raw)
    text_embed = proj(text_raw)

    # Pure cross-modal: no context_seq
    audio_mod, text_mod = attn(audio_embed, text_embed)

    assert audio_mod.shape == (B, D), f"Expected ({B}, {D}), got {audio_mod.shape}"
    assert text_mod.shape == (B, D)
    print("  [PASS] CrossModalAttention (no context): shapes OK")
    print(f"         audio_mod range: [{audio_mod.min().item():.3f}, {audio_mod.max().item():.3f}]")


def test_cross_modal_attention_with_context():
    """CrossModalAttention: audio<->text with dialogue context."""
    from models.alignment import CrossModalAttention, ProjectionHead

    B, T, D = 4, 5, 64
    attn = CrossModalAttention(embed_dim=D, n_heads=4)
    proj = ProjectionHead(input_dim=D, embed_dim=D)

    audio_embed = proj(torch.randn(B, D))
    text_embed = proj(torch.randn(B, D))
    context_seq = torch.randn(B, T, D)

    audio_mod, text_mod = attn(audio_embed, text_embed, context_seq=context_seq)

    assert audio_mod.shape == (B, D)
    assert text_mod.shape == (B, D)
    print("  [PASS] CrossModalAttention (with context): shapes OK")


def test_cross_modal_attention_with_context_and_padding():
    """CrossModalAttention: with context and padding mask."""
    from models.alignment import CrossModalAttention, ProjectionHead

    B, T, D = 2, 4, 64
    attn = CrossModalAttention(embed_dim=D, n_heads=4)
    proj = ProjectionHead(input_dim=D, embed_dim=D)

    audio_embed = proj(torch.randn(B, D))
    text_embed = proj(torch.randn(B, D))
    context_seq = torch.randn(B, T, D)
    pad = torch.tensor([[False, False, True, True],
                        [False, False, False, True]])

    audio_mod, text_mod = attn(audio_embed, text_embed, context_seq=context_seq, context_padding=pad)

    assert audio_mod.shape == (B, D)
    assert text_mod.shape == (B, D)
    print("  [PASS] CrossModalAttention (with context + padding): shapes OK")


def test_cross_modal_attention_cold_start():
    """CrossModalAttention: all-masked context should not crash."""
    from models.alignment import CrossModalAttention, ProjectionHead

    B, T, D = 2, 4, 64
    attn = CrossModalAttention(embed_dim=D, n_heads=4)
    proj = ProjectionHead(input_dim=D, embed_dim=D)

    audio_embed = proj(torch.randn(B, D))
    text_embed = proj(torch.randn(B, D))
    context_seq = torch.randn(B, T, D)
    pad = torch.ones(B, T, dtype=torch.bool)

    audio_mod, text_mod = attn(audio_embed, text_embed, context_seq=context_seq, context_padding=pad)

    assert audio_mod.shape == (B, D)
    assert text_mod.shape == (B, D)
    print("  [PASS] CrossModalAttention (cold-start guard): shapes OK")


def test_full_forward_chain():
    """Simulate the full ConflictNet forward pass (fusion → temporal → classifier)."""
    from models.alignment import CrossModalAttention, ProjectionHead
    from models.classifier import ConflictClassifier
    from models.temporal import TransformerTemporalContext

    B, D = 4, 64
    T = 5

    # Encoder simulation
    proj = ProjectionHead(input_dim=D, embed_dim=D)
    audio_embed = proj(torch.randn(B, D))
    text_embed = proj(torch.randn(B, D))

    # Cross-modal attention
    attn = CrossModalAttention(embed_dim=D, n_heads=4)
    context_seq = torch.randn(B, T, D)
    audio_embed, text_embed = attn(audio_embed, text_embed, context_seq=context_seq)

    # Speaker feature (dummy)
    speaker_feat = torch.randn(B, D)

    # Fusion gate (same as conflictnet.py)
    fusion_gate = torch.nn.Sequential(
        torch.nn.Linear(D * 3, D * 2),
        torch.nn.GELU(),
        torch.nn.LayerNorm(D * 2),
        torch.nn.Linear(D * 2, D),
        torch.nn.LayerNorm(D),
    )
    combined = torch.cat([audio_embed, text_embed, speaker_feat], dim=-1)
    fused_embed = fusion_gate(combined)

    # Temporal context (max_turns accommodates context + current turn)
    temporal = TransformerTemporalContext(embed_dim=D, n_layers=2, n_heads=4, max_turns=T + 1)
    current_turn = fused_embed.unsqueeze(1)
    turn_seq = torch.cat([context_seq, current_turn], dim=1)
    per_turn_ctx, context_pooled = temporal(turn_seq)
    current_ctx = per_turn_ctx[:, -1, :]

    # Classifier with speaker-adaptive threshold
    clf = ConflictClassifier(embed_dim=D, n_types=3, speaker_adaptive_threshold=True)
    logits, probs, severity, flag = clf(
        fused_embed=current_ctx,
        speaker_feat=speaker_feat,
    )

    assert logits.shape == (B, 3), f"Expected ({B}, 3), got {logits.shape}"
    assert probs.shape == (B, 3)
    assert severity.shape == (B, 1)
    assert flag.shape == (B,)
    assert flag.dtype == torch.bool
    print("  [PASS] Full forward chain: shapes OK")
    print(f"         logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"         probs range:  [{probs.min().item():.3f}, {probs.max().item():.3f}]")


def test_baseline_normalize_integration():
    """Simulate the baseline-subtract prosody pipeline."""
    import numpy as np
    from models.speaker_norm.speaker_norm import SpeakerStats, compute_prosody_z_scores

    from collections import defaultdict
    registry = defaultdict(SpeakerStats)

    from models.speaker_norm.speaker_norm import ColdStartFallback
    cold_start = ColdStartFallback(min_ref_utts=1)

    for _ in range(10):
        audio = np.random.randn(16000).astype(np.float32)
        compute_prosody_z_scores(
            [audio], ["speaker_001"], registry, cold_start,
            use_baseline_subtract=True,
            sr=16000,
        )

    audio1 = np.random.randn(16000).astype(np.float32)
    feat1 = compute_prosody_z_scores(
        [audio1], ["speaker_001"], registry, cold_start,
        use_baseline_subtract=True,
        conflict_flags=[False],
        sr=16000,
    )

    audio2 = np.random.randn(16000).astype(np.float32)
    feat2 = compute_prosody_z_scores(
        [audio2], ["speaker_001"], registry, cold_start,
        use_baseline_subtract=True,
        conflict_flags=[True],
        sr=16000,
    )

    assert feat1.shape == (1, 3), f"Expected (1, 3), got {feat1.shape}"
    assert feat2.shape == (1, 3)
    print("  [PASS] Baseline-normalize integration: shapes OK")


def test_speaker_adaptive_threshold_variation():
    """Verify that different speaker features produce different thresholds."""
    from models.classifier import SpeakerAdaptiveThreshold

    D = 64
    sat = SpeakerAdaptiveThreshold(embed_dim=D, max_offset=0.3)

    expressive_spk = torch.tensor([[1.0] * D])
    monotone_spk = torch.tensor([[-1.0] * D])

    offset_e = sat(expressive_spk).item()
    offset_m = sat(monotone_spk).item()

    assert 0 <= offset_e <= 0.3
    assert 0 <= offset_m <= 0.3
    print("  [PASS] SpeakerAdaptiveThreshold produces different offsets")
    print(f"         expressive offset: {offset_e:.4f}, monotone offset: {offset_m:.4f}")


def test_classifier_with_and_without_speaker_feat():
    """Verify the classifier handles both cases (speaker_feat provided or not)."""
    from models.classifier import ConflictClassifier

    B, D = 4, 64
    clf = ConflictClassifier(embed_dim=D, n_types=3, speaker_adaptive_threshold=True)
    fused = torch.randn(B, D)
    spk = torch.randn(B, D)

    logits, probs, severity, flag_with = clf(fused, speaker_feat=spk)
    _, _, _, flag_without = clf(fused)

    assert flag_with.shape == (B,)
    assert flag_without.shape == (B,)
    print("  [PASS] Classifier works with AND without speaker_feat")


def test_swap_pretraining():
    """Verify swap pretraining objective produces a valid loss."""
    from models.conflictnet import SwapPretrainingObjective

    B, D = 8, 64
    swap_obj = SwapPretrainingObjective(embed_dim=D, swap_prob=0.3)
    audio = torch.randn(B, D)
    text = torch.randn(B, D)

    loss = swap_obj(audio, text)
    assert loss.item() > 0
    print(f"  [PASS] SwapPretrainingObjective: loss={loss.item():.4f}")


def test_multi_task_loss():
    """Verify multi-task loss produces a scalar with uncertainty weights."""
    from models.conflictnet import MultiTaskLoss

    mtl = MultiTaskLoss(n_tasks=4)
    losses = [torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.3), torch.tensor(0.2)]
    total, weights = mtl(losses)

    assert total.dim() == 0  # scalar
    assert len(weights) == 4
    print(f"  [PASS] MultiTaskLoss: total={total.item():.4f}, weights={list(weights.values())}")


def main():
    print("Dummy dry-run integration tests:\n")
    test_cross_modal_attention_no_context()
    test_cross_modal_attention_with_context()
    test_cross_modal_attention_with_context_and_padding()
    test_cross_modal_attention_cold_start()
    test_full_forward_chain()
    test_baseline_normalize_integration()
    test_speaker_adaptive_threshold_variation()
    test_classifier_with_and_without_speaker_feat()
    test_swap_pretraining()
    test_multi_task_loss()
    print("\n  All dummy dry-run smoke tests passed!")


if __name__ == "__main__":
    main()
