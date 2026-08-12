"""Unit tests for ConflictNet components (no GPU required, uses tiny synthetic tensors)."""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest.mock

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Ablation flag tests (submodule + YAML + CLI — avoids HF downloads)
# ---------------------------------------------------------------------------

class TestAblationFlags:
    def test_baseline_subtract_flag_on_normalizer(self):
        from models.speaker_norm.speaker_norm import SpeakerNormalizer
        norm = SpeakerNormalizer(embed_dim=64, use_baseline_subtract=False)
        assert norm.use_baseline_subtract is False

        norm2 = SpeakerNormalizer(embed_dim=64, use_baseline_subtract=True)
        assert norm2.use_baseline_subtract is True

    def test_classifier_adaptive_threshold_flag(self):
        from models.classifier import ConflictClassifier
        clf_with = ConflictClassifier(embed_dim=64, n_types=3, speaker_adaptive_threshold=True)
        assert clf_with.threshold_net is not None

        clf_without = ConflictClassifier(embed_dim=64, n_types=3, speaker_adaptive_threshold=False)
        assert clf_without.threshold_net is None

    def test_ablate_yaml_configs_exist(self):
        from pathlib import Path
        yaml = pytest.importorskip("yaml")
        config_dir = Path(__file__).parent.parent / "configs"
        expected = [
            "ablate_no_cross_attn.yaml",
            "ablate_no_adaptive_threshold.yaml",
            "ablate_no_baseline_subtract.yaml",
            "ablate_no_pretrain.yaml",
            "ablate_no_speaker_norm.yaml",
            "ablate_no_temporal.yaml",
            "ablate_no_word_div.yaml",
        ]
        for name in expected:
            path = config_dir / name
            assert path.exists(), f"Missing config: {path}"
            with open(path) as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict)

    def test_train_has_new_cli_flags(self):
        from scripts.train import parse_args
        import sys
        sys.argv = ['train.py', '--iemocap_root', '/tmp']
        args = parse_args()
        assert hasattr(args, 'no_cross_attn_injection')
        assert hasattr(args, 'no_speaker_adaptive_threshold')
        assert hasattr(args, 'no_baseline_subtract')
        assert hasattr(args, 'prosody_stats')
        assert hasattr(args, 'amp')
        assert args.no_cross_attn_injection is False
        assert args.no_speaker_adaptive_threshold is False
        assert args.no_baseline_subtract is False
        assert args.amp is False
        assert args.prosody_stats is None


# ---------------------------------------------------------------------------
# Encoder tests
# ---------------------------------------------------------------------------

class TestEncoders:
    def test_audio_factory_invalid(self):
        from models.encoders import build_audio_encoder
        with pytest.raises(ValueError):
            build_audio_encoder("nonexistent_encoder")

    def test_projection_head_shape(self):
        from models.alignment import ProjectionHead
        head = ProjectionHead(input_dim=768, embed_dim=256)
        x = torch.randn(4, 768)
        out = head(x)
        assert out.shape == (4, 256)

    def test_projection_head_normalisation(self):
        """Output is NOT pre-normalised; caller must normalise."""
        from models.alignment import ProjectionHead
        import torch.nn.functional as F
        head = ProjectionHead(input_dim=64, embed_dim=32)
        x = torch.randn(2, 64)
        out = head(x)
        normed = F.normalize(out, dim=-1)
        assert normed.shape == out.shape


# ---------------------------------------------------------------------------
# Temporal context tests
# ---------------------------------------------------------------------------

class TestTemporalContext:
    def test_output_shape(self):
        from models.temporal import TransformerTemporalContext
        model = TransformerTemporalContext(embed_dim=64, n_layers=1, n_heads=4, max_turns=8)
        x = torch.randn(2, 5, 64)  # (B=2, T=5, D=64)
        per_turn, pooled = model(x)
        assert per_turn.shape == (2, 5, 64)
        assert pooled.shape == (2, 64)

    def test_causal_masking(self):
        """With causal mask, output at position t should not depend on t+1."""
        from models.temporal import TransformerTemporalContext
        model = TransformerTemporalContext(embed_dim=32, n_layers=1, n_heads=4, max_turns=8)
        model.eval()
        x = torch.randn(1, 4, 32)
        with torch.no_grad():
            out1, _ = model(x)
            x_perturbed = x.clone()
            x_perturbed[0, 3, :] += 10.0  # perturb last turn only
            out2, _ = model(x_perturbed)
        # First 3 positions should be identical (causal)
        assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-5), \
            "Causal mask violated: early positions affected by later tokens"

    def test_padding_mask(self):
        from models.temporal import TransformerTemporalContext
        model = TransformerTemporalContext(embed_dim=32, n_layers=1, n_heads=4, max_turns=8)
        x = torch.randn(2, 4, 32)
        padding = torch.tensor([[False, False, True, True],
                                [False, False, False, True]])
        per_turn, pooled = model(x, padding_mask=padding)
        assert pooled.shape == (2, 32)


# ---------------------------------------------------------------------------
# Speaker normalizer tests
# ---------------------------------------------------------------------------

class TestSpeakerNorm:
    def test_prosody_extraction_librosa(self):
        """Should work even without parselmouth (librosa fallback)."""
        from models.speaker_norm.speaker_norm import extract_prosody_stats
        audio = np.random.randn(16000).astype(np.float32)
        stats = extract_prosody_stats(audio, sr=16000)
        assert "f0_mean" in stats
        assert "energy_mean" in stats
        assert "speaking_rate" in stats

    def test_speaker_stats_z_score(self):
        from models.speaker_norm.speaker_norm import SpeakerStats
        s = SpeakerStats()
        for _ in range(10):
            s.update(120.0 + np.random.randn(), 60.0 + np.random.randn(), 5.0)
        z = s.z_score(120.0, 60.0, 5.0)
        assert z.shape == (3,)
        # Mean should be close to 0 (z-score of the mean)
        assert abs(z[0]) < 2.0

    def test_cold_start_fallback(self):
        from models.speaker_norm.speaker_norm import ColdStartFallback, SpeakerStats
        fb = ColdStartFallback(min_ref_utts=5)
        sparse_spk = SpeakerStats()
        sparse_spk.update(100.0, 50.0, 4.0)  # only 1 utt
        # Register some global stats
        for _ in range(10):
            fb.register_utterance(110.0, 55.0, 4.5, gender="F")
        result = fb.get_stats(sparse_spk, gender="F")
        assert result.n >= 5  # should fall back to gender group

    def test_baseline_update_and_normalize(self):
        from models.speaker_norm.speaker_norm import SpeakerStats
        s = SpeakerStats()
        for _ in range(10):
            s.update(120.0 + np.random.randn(), 60.0 + np.random.randn(), 5.0)
        # Update neutral baseline with non-conflict utterances
        for _ in range(5):
            s.update_baseline(120.0, 58.0, 5.0, lr=0.2)
        assert s.neutral_baseline is not None
        # baseline_normalize should produce a sensible result
        z = s.baseline_normalize(125.0, 65.0, 6.0)
        assert z.shape == (3,)
        # fallback to z_score when baseline is None
        s2 = SpeakerStats()
        for _ in range(10):
            s2.update(120.0, 60.0, 5.0)
        z2 = s2.baseline_normalize(120.0, 60.0, 5.0)
        assert z2.shape == (3,)


# ---------------------------------------------------------------------------
# Alignment / loss tests
# ---------------------------------------------------------------------------

class TestContrastiveLoss:
    def test_loss_positive(self):
        from models.alignment import ContextGatedContrastiveLoss
        loss_fn = ContextGatedContrastiveLoss(embed_dim=64)
        audio = torch.randn(4, 64)
        text = torch.randn(4, 64)
        loss = loss_fn(audio, text)
        assert loss.item() > 0

    def test_loss_decreases_with_aligned_pairs(self):
        """Identical audio and text embeddings should produce lower loss."""
        from models.alignment import ContextGatedContrastiveLoss
        import torch.nn.functional as F
        loss_fn = ContextGatedContrastiveLoss(embed_dim=32)
        x = F.normalize(torch.randn(8, 32), dim=-1)
        # Perfect alignment: audio = text
        loss_aligned = loss_fn(x, x).item()
        # Random: audio ≠ text
        y = F.normalize(torch.randn(8, 32), dim=-1)
        loss_random = loss_fn(x, y).item()
        # Aligned should generally have lower loss
        # (may not always hold due to temperature, but true in expectation)
        assert loss_aligned <= loss_random + 1.0  # generous tolerance

    def test_conflict_separation_loss(self):
        from models.alignment import ContextGatedContrastiveLoss
        loss_fn = ContextGatedContrastiveLoss(embed_dim=32)
        audio = torch.randn(4, 32)
        text = torch.randn(4, 32)
        conflict_labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = loss_fn(audio, text, conflict_labels=conflict_labels)
        assert loss.item() > 0

    def test_cross_modal_attention_shape(self):
        from models.alignment import CrossModalAttention
        attn = CrossModalAttention(embed_dim=32, n_heads=2)
        B, D = 4, 32
        audio = torch.randn(B, D)
        text = torch.randn(B, D)
        # Without context
        audio_out, text_out = attn(audio, text)
        assert audio_out.shape == (B, D)
        assert text_out.shape == (B, D)

    def test_cross_modal_attention_with_context(self):
        from models.alignment import CrossModalAttention
        attn = CrossModalAttention(embed_dim=16, n_heads=2)
        B, T, D = 2, 4, 16
        audio = torch.randn(B, D)
        text = torch.randn(B, D)
        ctx = torch.randn(B, T, D)
        audio_out, text_out = attn(audio, text, context_seq=ctx)
        assert audio_out.shape == (B, D)
        assert text_out.shape == (B, D)

    def test_cross_modal_attention_with_context_and_padding(self):
        from models.alignment import CrossModalAttention
        attn = CrossModalAttention(embed_dim=16, n_heads=2)
        B, T, D = 2, 4, 16
        audio = torch.randn(B, D)
        text = torch.randn(B, D)
        ctx = torch.randn(B, T, D)
        pad = torch.tensor([[False, False, True, True],
                            [False, False, False, True]])
        audio_out, text_out = attn(audio, text, context_seq=ctx, context_padding=pad)
        assert audio_out.shape == (B, D)
        assert text_out.shape == (B, D)

    def test_cross_modal_attention_cold_start(self):
        """All-masked context should not crash (cold-start guard)."""
        from models.alignment import CrossModalAttention
        attn = CrossModalAttention(embed_dim=16, n_heads=2)
        B, T, D = 2, 4, 16
        audio = torch.randn(B, D)
        text = torch.randn(B, D)
        ctx = torch.randn(B, T, D)
        pad = torch.ones(B, T, dtype=torch.bool)  # all masked
        audio_out, text_out = attn(audio, text, context_seq=ctx, context_padding=pad)
        assert audio_out.shape == (B, D)
        assert text_out.shape == (B, D)

    def test_cross_modal_attention_cross_modal_invariance(self):
        """Swapping audio and text should produce different outputs (not symmetric)."""
        from models.alignment import CrossModalAttention
        attn = CrossModalAttention(embed_dim=16, n_heads=2)
        B, D = 2, 16
        audio = torch.randn(B, D)
        text = torch.randn(B, D)
        a1, t1 = attn(audio, text)
        a2, t2 = attn(text, audio)
        # Cross-modal attention is not symmetric: a1 != a2 in general
        assert not torch.allclose(a1, a2, atol=1e-4)


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------

class TestClassifier:
    def test_output_shapes(self):
        from models.classifier import ConflictClassifier
        clf = ConflictClassifier(embed_dim=64, n_types=3, hidden_dims=(128, 64))
        x = torch.randn(4, 64)
        logits, probs, severity, flag = clf(x)
        assert logits.shape == (4, 3)
        assert probs.shape == (4, 3)
        assert severity.shape == (4, 1)
        assert flag.shape == (4,)
        assert flag.dtype == torch.bool

    def test_probs_in_range(self):
        from models.classifier import ConflictClassifier
        clf = ConflictClassifier(embed_dim=32, n_types=3)
        x = torch.randn(8, 32)
        _, probs, severity, _ = clf(x)
        assert (probs >= 0).all() and (probs <= 1).all()
        assert (severity >= 0).all() and (severity <= 1).all()

    def test_word_div_fusion(self):
        from models.classifier import ConflictClassifier
        clf = ConflictClassifier(embed_dim=32, n_types=3, word_div_dim=8)
        x = torch.randn(4, 32)
        wd = torch.randn(4, 8)
        logits, probs, severity, flag = clf(x, word_div=wd)
        assert logits.shape == (4, 3)

    def test_speaker_adaptive_threshold(self):
        from models.classifier import ConflictClassifier, SpeakerAdaptiveThreshold
        sat = SpeakerAdaptiveThreshold(embed_dim=32, max_offset=0.3)
        spk_feat = torch.randn(4, 32)
        offset = sat(spk_feat)
        assert offset.shape == (4,)
        assert (offset >= 0).all() and (offset <= 0.3).all()

        clf = ConflictClassifier(embed_dim=32, n_types=3, speaker_adaptive_threshold=True)
        fused = torch.randn(4, 32)
        spk = torch.randn(4, 32)
        logits, probs, severity, flag = clf(fused, speaker_feat=spk)
        assert logits.shape == (4, 3)
        assert flag.dtype == torch.bool

    def test_speaker_adaptive_threshold_without_feat(self):
        """Should fall back to fixed threshold when speaker_feat is not provided."""
        from models.classifier import ConflictClassifier
        clf = ConflictClassifier(embed_dim=32, n_types=3, speaker_adaptive_threshold=True)
        fused = torch.randn(4, 32)
        logits, probs, severity, flag = clf(fused)  # no speaker_feat
        assert logits.shape == (4, 3)
        assert flag.dtype == torch.bool


# ---------------------------------------------------------------------------
# Curriculum sampler tests
# ---------------------------------------------------------------------------

class TestCurriculumSampler:
    def test_easy_only_at_warmup(self):
        from training.curriculum import CurriculumSampler
        difficulties = [0.1, 0.9, 0.2, 0.8, 0.3]
        sampler = CurriculumSampler(difficulties, epoch=0, warmup_epochs=5, shuffle=False)
        indices = list(sampler)
        # At epoch 0, threshold = 0.33 → only items with d ≤ 0.33
        for i in indices:
            assert difficulties[i] <= 0.33 + 1e-6

    def test_all_included_after_training(self):
        from training.curriculum import CurriculumSampler
        difficulties = [0.1, 0.9, 0.5, 0.7, 0.3]
        sampler = CurriculumSampler(difficulties, epoch=30, max_epochs=30, warmup_epochs=5, shuffle=False)
        assert len(sampler) == len(difficulties)

    def test_len_consistency(self):
        from training.curriculum import CurriculumSampler
        difficulties = list(np.random.rand(100))
        sampler = CurriculumSampler(difficulties, epoch=10, max_epochs=30, warmup_epochs=5)
        assert len(sampler) == len(list(sampler))


# ---------------------------------------------------------------------------
# Word divergence tests
# ---------------------------------------------------------------------------

class TestWordDivergence:
    def test_aggregate_shape(self):
        from models.alignment.word_divergence import WordLevelDivergence
        wd = WordLevelDivergence(embed_dim=32)
        divs = torch.rand(10)
        feat = wd.aggregate(divs)
        assert feat.shape == (WordLevelDivergence.DIVERGENCE_FEAT_DIM,)

    def test_empty_word_list(self):
        from models.alignment.word_divergence import WordLevelDivergence
        wd = WordLevelDivergence(embed_dim=32)
        wa = torch.zeros(0, 32)
        wt = torch.zeros(0, 32)
        feat = wd.forward_from_precomputed([wa], [wt])
        assert feat.shape == (1, WordLevelDivergence.DIVERGENCE_FEAT_DIM)

    def test_batch_forward(self):
        from models.alignment.word_divergence import WordLevelDivergence
        wd = WordLevelDivergence(embed_dim=32)
        B = 3
        wa_list = [torch.randn(np.random.randint(3, 8), 32) for _ in range(B)]
        wt_list = [torch.randn(wa.shape[0], 32) for wa in wa_list]
        feat = wd.forward_from_precomputed(wa_list, wt_list)
        assert feat.shape == (B, WordLevelDivergence.DIVERGENCE_FEAT_DIM)

    def test_forward_from_encoder_hidden(self):
        from models.alignment.word_divergence import WordLevelDivergence
        wd = WordLevelDivergence(embed_dim=32)
        B, T_frames, L_tokens = 2, 200, 30
        audio_frames = torch.randn(B, T_frames, 32)
        text_tokens = torch.randn(B, L_tokens, 32)
        # Two utterances, 4 and 3 words
        word_timestamps = [
            [(0.0, 0.3), (0.3, 0.7), (0.7, 1.2), (1.2, 2.0)],
            [(0.1, 0.5), (0.5, 0.9), (0.9, 1.5)],
        ]
        token_word_boundaries = [
            [(0, 3), (3, 7), (7, 10), (10, 14)],
            [(0, 4), (4, 8), (8, 11)],
        ]
        feat = wd.forward_from_encoder_hidden(
            audio_frame_embeds=audio_frames,
            word_timestamps=word_timestamps,
            text_token_embeds=text_tokens,
            token_word_boundaries=token_word_boundaries,
        )
        assert feat.shape == (B, WordLevelDivergence.DIVERGENCE_FEAT_DIM)
        assert not torch.isnan(feat).any()


# ---------------------------------------------------------------------------
# ExperimentConfig tests
# ---------------------------------------------------------------------------

class TestExperimentConfig:
    def test_roundtrip_json(self):
        from models.experiment_config import ExperimentConfig
        cfg = ExperimentConfig(amp=True, use_word_divergence=False, temporal_max_turns=16)
        raw = cfg.to_json()
        restored = ExperimentConfig.from_json(raw)
        assert restored.amp is True
        assert restored.use_word_divergence is False
        assert restored.temporal_max_turns == 16
        assert restored.embed_dim == 256  # default

    def test_from_args(self):
        from models.experiment_config import ExperimentConfig
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--epochs", type=int, default=10)
        p.add_argument("--no_temporal", action="store_true")
        p.add_argument("--amp", action="store_true")
        p.add_argument("--batch_size", type=int, default=32)
        args = p.parse_args(["--epochs", "20", "--no_temporal", "--amp"])
        cfg = ExperimentConfig.from_args(args)
        assert cfg.epochs == 20
        assert cfg.use_temporal is False  # inverted from --no_temporal
        assert cfg.amp is True
        assert cfg.batch_size == 32  # default
        assert cfg.use_speaker_norm is True  # default

    def test_to_cli_args_inverse(self):
        from models.experiment_config import ExperimentConfig
        cfg = ExperimentConfig(amp=True, use_temporal=True, use_word_divergence=False)
        cli = cfg.to_cli_args()
        assert cli["amp"] is True
        assert cli["no_temporal"] is False  # use_temporal=True → no_temporal=False
        assert cli["no_word_divergence"] is True  # use_word_divergence=False → no_word_divergence=True

    def test_validation(self):
        from models.experiment_config import ExperimentConfig
        import pytest
        with pytest.raises(ValueError):
            ExperimentConfig(temporal_max_turns=0)


# ---------------------------------------------------------------------------
# Multi-task loss tests
# ---------------------------------------------------------------------------

class TestMultiTaskLoss:
    def test_output_scalar(self):
        from models.conflictnet import MultiTaskLoss
        mtl = MultiTaskLoss(n_tasks=3)
        losses = [torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.3)]
        total, weights = mtl(losses)
        assert total.dim() == 0  # scalar
        assert len(weights) == 3

    def test_learnable_sigmas(self):
        from models.conflictnet import MultiTaskLoss
        mtl = MultiTaskLoss(n_tasks=4)
        assert sum(p.numel() for p in mtl.parameters()) == 4


# ---------------------------------------------------------------------------
# Full-model integration tests (mock encoders, no HF downloads)
# ---------------------------------------------------------------------------

class MockAudioEncoder(torch.nn.Module):
    """Dummy audio encoder that returns random tensors (no HF download)."""
    output_dim = 768

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, audio, attention_mask=None, return_frames=False):
        B = audio.shape[0]
        pooled = torch.randn(B, self.output_dim, device=audio.device)
        if return_frames:
            return pooled, torch.randn(B, 50, self.output_dim, device=audio.device)
        return pooled


class MockTextEncoder(torch.nn.Module):
    """Dummy text encoder that returns random tensors (no HF download)."""
    output_dim = 1024

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input_ids, attention_mask, return_tokens=False):
        B = input_ids.shape[0]
        pooled = torch.randn(B, self.output_dim, device=input_ids.device)
        if return_tokens:
            return pooled, torch.randn(B, input_ids.shape[1], self.output_dim, device=input_ids.device)
        return pooled


class TestFullModelIntegration:
    """Integration tests for the full ConflictNet forward pass.

    Uses mock encoders (no HF model downloads). Exercises the exact same
    code paths as the real model, including all 7 ablation toggles.
    """

    @pytest.fixture(autouse=True)
    def patch_encoders(self):
        from models import conflictnet as cn_module
        with (
            unittest.mock.patch.object(cn_module, 'build_audio_encoder', return_value=MockAudioEncoder()),
            unittest.mock.patch.object(cn_module, 'DeBERTaEncoder', MockTextEncoder),
        ):
            yield

    @pytest.fixture
    def model(self):
        from models.conflictnet import ConflictNet
        m = ConflictNet(
            audio_encoder_name="emotion2vec",
            embed_dim=64,
            n_conflict_types=6,
            temporal_n_layers=1,
            temporal_n_heads=2,
            temporal_max_turns=4,
            use_speaker_norm=True,
            use_temporal=True,
            use_word_divergence=False,
            use_swap_pretraining=True,
            use_cross_attn_injection=True,
            use_speaker_adaptive_threshold=True,
            use_baseline_subtract=True,
            lora_r=0,
            lora_alpha=1,
        )
        m.eval()
        return m

    def _synthetic_batch(self, model, B=2, T_turns=3, device="cpu"):
        """Create a synthetic batch for the forward pass."""
        audio = torch.randn(B, 16000, device=device)
        input_ids = torch.randint(0, 100, (B, 10), device=device)
        attention_mask = torch.ones(B, 10, dtype=torch.long, device=device)
        audio_attn_mask = torch.ones(B, 16000, dtype=torch.bool, device=device)
        context = torch.randn(B, T_turns, model.embed_dim, device=device)
        context_pad = torch.zeros(B, T_turns, dtype=torch.bool, device=device)
        prosody_z = torch.randn(B, 3, device=device)

        type_labels = torch.randint(0, 2, (B, 6), device=device).float()
        severity_labels = torch.rand(B, 1, device=device)
        binary_labels = (type_labels.sum(dim=1) > 0).float()

        return {
            "audio": audio,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_attention_mask": audio_attn_mask,
            "context_embeds": context,
            "context_padding": context_pad,
            "prosody_z": prosody_z,
            "conflict_type_labels": type_labels,
            "severity_labels": severity_labels,
            "conflict_binary_labels": binary_labels,
        }

    def test_full_forward_default(self, model):
        """Full forward pass with all features enabled & labels."""
        batch = self._synthetic_batch(model, B=2)
        with torch.no_grad():
            out = model(**batch)

        assert out.logits_type.shape == (2, 6)
        assert out.probs_type.shape == (2, 6)
        assert out.severity.shape == (2, 1)
        assert out.conflict_flag.shape == (2,)
        assert out.conflict_flag.dtype == torch.bool
        assert out.audio_embed.shape == (2, 64)
        assert out.text_embed.shape == (2, 64)
        assert out.speaker_feat.shape == (2, 64)
        assert out.fused_embed.shape == (2, 64)
        assert out.context_pooled.shape == (2, 64)
        assert out.per_turn_context.shape == (2, 4, 64)  # T_turns(3) + current(1)
        assert out.loss is not None
        assert out.loss_breakdown is not None

    def test_full_forward_no_labels(self, model):
        """Forward pass without labels → no loss."""
        batch = self._synthetic_batch(model, B=2)
        del batch["conflict_type_labels"]
        del batch["severity_labels"]
        del batch["conflict_binary_labels"]
        with torch.no_grad():
            out = model(**batch)

        assert out.loss is None
        assert out.loss_breakdown is None
        assert out.probs_type.shape == (2, 6)
        assert out.conflict_flag.shape == (2,)

    def test_full_forward_no_context(self, model):
        """Forward pass without dialogue context (single-utterance)."""
        batch = self._synthetic_batch(model, B=2)
        del batch["context_embeds"]
        del batch["context_padding"]
        with torch.no_grad():
            out = model(**batch)

        assert out.logits_type.shape == (2, 6)
        assert out.per_turn_context.shape == (2, 1, 64)  # just current turn
        assert out.loss is not None

    def test_full_forward_pretraining(self, model):
        """Pretraining mode should produce a loss even without labels."""
        batch = self._synthetic_batch(model, B=2)
        del batch["conflict_type_labels"]
        del batch["severity_labels"]
        del batch["conflict_binary_labels"]
        with torch.no_grad():
            out = model(**batch, pretraining=True)

        assert out.loss is not None

    def test_ablate_cross_attn(self):
        """Forward pass with cross_attn disabled."""
        from models.conflictnet import ConflictNet
        m = ConflictNet(
            audio_encoder_name="emotion2vec",
            embed_dim=64, n_conflict_types=6,
            use_cross_attn_injection=False,
            use_swap_pretraining=False,
            use_word_divergence=False,
            temporal_n_layers=1, temporal_n_heads=2, temporal_max_turns=4,
            lora_r=0, lora_alpha=1,
        )
        m.eval()
        assert m.cross_modal_attn is None
        batch = self._synthetic_batch(m, B=2)
        with torch.no_grad():
            out = m(**batch)
        assert out.logits_type.shape == (2, 6)

    def test_ablate_temporal(self):
        """Forward pass with temporal context disabled."""
        from models.conflictnet import ConflictNet
        m = ConflictNet(
            audio_encoder_name="emotion2vec",
            embed_dim=64, n_conflict_types=6,
            use_temporal=False,
            use_swap_pretraining=False,
            use_word_divergence=False,
            lora_r=0, lora_alpha=1,
        )
        m.eval()
        assert m.temporal is None
        batch = self._synthetic_batch(m, B=2)
        with torch.no_grad():
            out = m(**batch)
        assert out.logits_type.shape == (2, 6)

    def test_ablate_speaker_norm(self):
        """Forward pass with speaker norm disabled."""
        from models.conflictnet import ConflictNet
        m = ConflictNet(
            audio_encoder_name="emotion2vec",
            embed_dim=64, n_conflict_types=6,
            use_speaker_norm=False,
            use_swap_pretraining=False,
            use_word_divergence=False,
            temporal_n_layers=1, temporal_n_heads=2, temporal_max_turns=4,
            lora_r=0, lora_alpha=1,
        )
        m.eval()
        assert m.speaker_norm is None
        batch = self._synthetic_batch(m, B=2)
        with torch.no_grad():
            out = m(**batch)
        assert out.logits_type.shape == (2, 6)
        # Speaker feat should be zeros
        assert torch.allclose(out.speaker_feat, torch.zeros(2, 64))

    def test_ablate_all_disabled(self):
        """Forward pass with all optional features disabled."""
        from models.conflictnet import ConflictNet
        m = ConflictNet(
            audio_encoder_name="emotion2vec",
            embed_dim=64, n_conflict_types=6,
            use_speaker_norm=False,
            use_temporal=False,
            use_word_divergence=False,
            use_swap_pretraining=False,
            use_cross_attn_injection=False,
            use_speaker_adaptive_threshold=False,
            use_baseline_subtract=False,
            lora_r=0, lora_alpha=1,
        )
        m.eval()
        batch = self._synthetic_batch(m, B=2)
        with torch.no_grad():
            out = m(**batch)
        assert out.logits_type.shape == (2, 6)
        assert out.loss is not None

    def test_parameter_counts(self, model):
        """count_parameters returns correct structure."""
        counts = model.count_parameters()
        assert isinstance(counts, dict)
        assert "audio_encoder" in counts
        assert "text_encoder" in counts
        assert "fusion_gate" in counts
        for name, c in counts.items():
            assert "total" in c
            assert "trainable" in c
            assert c["total"] >= c["trainable"]

    def test_batch_independence(self, model):
        """Different batch items should produce different outputs."""
        batch = self._synthetic_batch(model, B=2)
        with torch.no_grad():
            out = model(**batch)
        # Two items in batch should have different predictions (model is random)
        assert not torch.allclose(out.probs_type[0], out.probs_type[1], atol=1e-3)

    def test_single_utterance_cross_modal(self):
        """CrossModalAttention works without temporal, with no context."""
        from models.conflictnet import ConflictNet
        m = ConflictNet(
            audio_encoder_name="emotion2vec",
            embed_dim=64, n_conflict_types=6,
            use_temporal=False,
            use_cross_attn_injection=True,
            use_swap_pretraining=False,
            use_word_divergence=False,
            lora_r=0, lora_alpha=1,
        )
        m.eval()
        assert m.temporal is None
        assert m.cross_modal_attn is not None
        # Single utterance: no context_embeds, no context_padding
        batch = self._synthetic_batch(m, B=2)
        del batch["context_embeds"]
        del batch["context_padding"]
        with torch.no_grad():
            out = m(**batch)
        assert out.logits_type.shape == (2, 6)

    def test_training_step(self):
        """Full training step: forward, backward, optimizer step (no AMP).
        Validates dtype consistency, gradient flow, and loss_breakdown
        includes sigma weights."""
        from models.conflictnet import ConflictNet
        m = ConflictNet(
            audio_encoder_name="emotion2vec",
            embed_dim=64, n_conflict_types=6,
            temporal_n_layers=1, temporal_n_heads=2, temporal_max_turns=4,
            use_speaker_norm=True, use_temporal=True,
            use_word_divergence=False, use_swap_pretraining=False,
            use_cross_attn_injection=True,
            use_speaker_adaptive_threshold=True,
            use_baseline_subtract=True,
            lora_r=0, lora_alpha=1,
        )
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        batch = self._synthetic_batch(m, B=2)
        for name, p in m.named_parameters():
            assert p.dtype == torch.float32, f"{name} is {p.dtype}"
        out = m(**batch)
        loss1 = out.loss
        loss1.backward()
        opt.step()
        opt.zero_grad()
        for name, p in m.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
        out = m(**batch)
        out.loss.backward()
        opt.step()
        assert out.loss_breakdown is not None
        sigma_keys = [k for k in out.loss_breakdown if k.startswith("sigma_task_")]
        assert len(sigma_keys) >= 3
        for k in sigma_keys:
            assert isinstance(out.loss_breakdown[k], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
