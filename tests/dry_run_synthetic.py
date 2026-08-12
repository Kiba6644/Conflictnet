"""Synthetic end-to-end dry-run: verifies model initialisation + forward pass.

Run with:  python tests/dry_run_synthetic.py

No GPU, no real data, no internet required.  Exits with code 0 on success.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

DEVICE = "cpu"
B, D, T_AUDIO, T_TEXT = 4, 256, 32000, 64


class _MockEncoder(torch.nn.Module):
    """Encoder stub — ignores inputs, produces (B, D) or ((B, D), (B, T, D))."""
    def __init__(self, embed_dim: int = D):
        super().__init__()
        self.output_dim = embed_dim
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        out = torch.randn(B, self.embed_dim)
        if kwargs.get("return_frames", False):
            frames = torch.randn(B, 8, self.embed_dim)  # fake (B, T, D)
            return out, frames
        return out


def _mock_audio_encoder(encoder_name: str = "emotion2vec"):
    return _MockEncoder()


def _mock_text_encoder(**kwargs):
    return _MockEncoder()


def test_full_forward():
    from models.conflictnet import ConflictNet, ConflictNetOutput

    with patch("models.conflictnet.build_audio_encoder", _mock_audio_encoder), \
         patch("models.conflictnet.DeBERTaEncoder", _mock_text_encoder):
        model = ConflictNet(
            audio_encoder_name="emotion2vec",
            embed_dim=D,
            use_speaker_norm=True,
            use_temporal=True,
            use_word_divergence=True,
            use_cross_attn_injection=True,
            use_speaker_adaptive_threshold=True,
            use_baseline_subtract=True,
            lora_r=8,
            temporal_max_turns=8,
        ).to(DEVICE).eval()

    audio = torch.randn(B, T_AUDIO)
    input_ids = torch.randint(0, 100, (B, T_TEXT))
    attention_mask = torch.ones(B, T_TEXT, dtype=torch.long)

    # Forward without labels
    out: ConflictNetOutput = model(audio=audio, input_ids=input_ids, attention_mask=attention_mask)
    assert out.logits_type.shape == (B, 3), f"logits_type: {out.logits_type.shape}"
    assert out.probs_type.shape == (B, 3), f"probs_type: {out.probs_type.shape}"
    assert out.severity is not None
    assert out.severity.shape == (B, 1) or out.severity.shape == (B,), f"severity: {out.severity.shape}"
    assert out.conflict_flag.shape == (B,), f"conflict_flag: {out.conflict_flag.shape}"
    assert out.loss is None, "No labels → no loss"
    print("  [PASS] Full forward (no labels): output OK")

    # Forward with labels
    type_labels = torch.rand(B, 3)
    sev_labels = torch.rand(B)
    bin_labels = torch.randint(0, 2, (B,)).float()
    out2 = model(
        audio=audio, input_ids=input_ids, attention_mask=attention_mask,
        conflict_type_labels=type_labels, severity_labels=sev_labels,
        conflict_binary_labels=bin_labels,
    )
    assert out2.loss is not None, "Labels → loss computed"
    assert out2.loss_breakdown is not None
    assert "contrastive" in out2.loss_breakdown
    assert "type_bce" in out2.loss_breakdown
    assert "severity_mse" in out2.loss_breakdown
    print(f"  [PASS] Full forward (with labels): loss={out2.loss.item():.4f}")
    print(f"         Breakdown: {out2.loss_breakdown}")


def test_ablation_flags():
    from models.conflictnet import ConflictNet

    for disable_norm in [True, False]:
        for disable_temporal in [True, False]:
            for disable_word_div in [True, False]:
                with patch("models.conflictnet.build_audio_encoder", _mock_audio_encoder), \
                     patch("models.conflictnet.DeBERTaEncoder", _mock_text_encoder):
                    model = ConflictNet(
                        use_speaker_norm=not disable_norm,
                        use_temporal=not disable_temporal,
                        use_word_divergence=not disable_word_div,
                    ).to(DEVICE).eval()
                audio = torch.randn(2, 16000)
                input_ids = torch.randint(0, 100, (2, 32))
                attention_mask = torch.ones(2, 32, dtype=torch.long)
                out = model(audio=audio, input_ids=input_ids, attention_mask=attention_mask)
                assert out.probs_type.shape == (2, 3)
    print("  [PASS] All 8 ablation combinations produce correct output shape")


def test_swap_pretraining():
    from models.conflictnet import ConflictNet

    with patch("models.conflictnet.build_audio_encoder", _mock_audio_encoder), \
         patch("models.conflictnet.DeBERTaEncoder", _mock_text_encoder):
        model = ConflictNet().to(DEVICE).eval()
    audio = torch.randn(2, 16000)
    input_ids = torch.randint(0, 100, (2, 32))
    attention_mask = torch.ones(2, 32, dtype=torch.long)

    out = model(
        audio=audio, input_ids=input_ids, attention_mask=attention_mask,
        pretraining=True,
    )
    assert out.loss is not None
    assert out.loss_breakdown is not None
    assert "swap" in out.loss_breakdown
    print("  [PASS] Pre-training mode: swap loss included")



def test_temperature_per_sample():
    """Verify the B3 fix: context gate produces per-sample tau."""
    from models.alignment.alignment import ContextGatedContrastiveLoss

    loss_fn = ContextGatedContrastiveLoss(embed_dim=64, context_gate_dim=32)
    B = 4
    audio = torch.randn(B, 64).div_(5).requires_grad_()
    text = torch.randn(B, 64).div_(5).requires_grad_()
    ctx = torch.randn(B, 64)

    # Without context — scalar tau
    loss_noctx = loss_fn(audio, text, context_pooled=None)
    assert loss_noctx is not None

    # With context — per-sample tau
    loss_ctx = loss_fn(audio, text, context_pooled=ctx)
    assert loss_ctx is not None

    # Both should be different (context should modulate tau)
    ctx_delta = loss_fn.context_gate(ctx).squeeze(-1)
    assert ctx_delta.shape == (B,), f"ctx_delta shape {ctx_delta.shape} should be per-sample"
    print(f"  [PASS] Context gate: per-sample deltas shape {ctx_delta.shape}")
    print(f"         delta range: [{ctx_delta.min().item():.4f}, {ctx_delta.max().item():.4f}]")


def main():
    print("=" * 60)
    print("  ConflictNet Synthetic Dry-Run")
    print("=" * 60)

    test_full_forward()
    test_ablation_flags()
    test_swap_pretraining()
    test_temperature_per_sample()

    print("\n  All dry-run tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
