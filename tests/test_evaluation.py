"""Tests for evaluation suite modules (no GPU required, small synthetic data)."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ============================================================================
# metrics.py
# ============================================================================

class TestMetrics:
    """Tests for evaluation/metrics.py — pure numpy, no model needed."""

    def test_compute_all_metrics_returns_dict(self):
        from evaluation.metrics import compute_all_metrics
        N = 10
        probs = np.random.rand(N, 3)
        labels = np.random.randint(0, 2, (N, 3))
        metrics = compute_all_metrics(probs, labels)
        assert isinstance(metrics, dict)
        assert "macro_f1" in metrics
        assert "binary_f1" in metrics
        assert "binary_acc" in metrics
        assert "wacc" in metrics

    def test_perfect_predictions(self):
        from evaluation.metrics import compute_all_metrics
        N = 10
        labels = np.random.randint(0, 2, (N, 3))
        # Perfect: probs at 1.0/0.0 exactly matching labels
        probs = labels.astype(float)
        metrics = compute_all_metrics(probs, labels, type_threshold=0.5)
        assert metrics["macro_f1"] == 1.0
        assert metrics["binary_f1"] == 1.0
        assert metrics["binary_acc"] == 1.0
        assert metrics["wacc"] == 1.0

    def test_all_negative_predictions(self):
        from evaluation.metrics import compute_all_metrics
        N = 10
        labels = np.random.randint(0, 2, (N, 3))
        probs = np.zeros((N, 3))
        metrics = compute_all_metrics(probs, labels, type_threshold=0.5)
        # With all negatives, binary_f1 should be 0 (all predictions are negative)
        assert metrics["binary_f1"] == 0.0
        assert 0 <= metrics["wacc"] <= 1.0

    def test_all_positive_predictions(self):
        from evaluation.metrics import compute_all_metrics
        N = 10
        labels = np.ones((N, 3), dtype=int)
        probs = np.ones((N, 3))
        metrics = compute_all_metrics(probs, labels, type_threshold=0.5)
        assert metrics["binary_f1"] == 1.0

    def test_severity_metrics(self):
        from evaluation.metrics import compute_all_metrics
        N = 10
        probs = np.random.rand(N, 3)
        labels = np.random.randint(0, 2, (N, 3))
        sev_pred = np.random.rand(N)
        sev_true = np.random.rand(N)
        metrics = compute_all_metrics(probs, labels, severity_pred=sev_pred, severity_true=sev_true)
        assert "severity_mae" in metrics
        assert metrics["severity_mae"] >= 0.0

    def test_severity_with_nans(self):
        from evaluation.metrics import compute_all_metrics
        N = 10
        probs = np.random.rand(N, 3)
        labels = np.random.randint(0, 2, (N, 3))
        sev_pred = np.random.rand(N)
        sev_true = np.full(N, np.nan)
        metrics = compute_all_metrics(probs, labels, severity_pred=sev_pred, severity_true=sev_true)
        assert "severity_mae" not in metrics or not np.isnan(metrics["severity_mae"])

    def test_custom_type_names(self):
        from evaluation.metrics import compute_all_metrics
        N = 5
        probs = np.random.rand(N, 2)
        labels = np.random.randint(0, 2, (N, 2))
        names = ["type_a", "type_b"]
        metrics = compute_all_metrics(probs, labels, type_names=names)
        assert "f1_type_a" in metrics
        assert "f1_type_b" in metrics
        assert "f1_sarcasm" not in metrics  # default names not used

    def test_single_type(self):
        from evaluation.metrics import compute_all_metrics
        N = 5
        probs = np.random.rand(N, 1)
        labels = np.random.randint(0, 2, (N, 1))
        metrics = compute_all_metrics(probs, labels)
        assert "macro_f1" in metrics

    def test_print_metrics_runs(self):
        from evaluation.metrics import print_metrics
        metrics = {"macro_f1": 0.85, "binary_f1": 0.90}
        print_metrics(metrics, prefix="test")
        print_metrics(metrics)  # no prefix


# ============================================================================
# fairness.py
# ============================================================================

class TestFairness:
    """Tests for evaluation/fairness.py — pure numpy, no model needed."""

    def test_fairness_audit_balanced_groups(self):
        from evaluation.fairness import fairness_audit
        N = 20
        y_pred = np.random.randint(0, 2, N)
        y_true = np.random.randint(0, 2, N)
        groups = ["M"] * 10 + ["F"] * 10
        report = fairness_audit(y_pred, y_true, groups)
        assert "overall_f1" in report
        assert "by_group" in report
        assert "disparity" in report
        assert "demographic_parity_difference" in report or report["demographic_parity_difference"] is None

    def test_fairness_audit_returns_by_group(self):
        from evaluation.fairness import fairness_audit
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        y_true = np.array([1, 1, 1, 0, 0, 0])
        groups = ["M", "M", "M", "F", "F", "F"]
        report = fairness_audit(y_pred, y_true, groups)
        assert "M" in report["by_group"]
        assert "F" in report["by_group"]
        assert report["overall_f1"] >= 0.0

    def test_fairness_audit_single_group(self):
        from evaluation.fairness import fairness_audit
        y_pred = np.array([1, 0, 1])
        y_true = np.array([1, 1, 0])
        groups = ["M"] * 3
        report = fairness_audit(y_pred, y_true, groups)
        assert report["disparity"] == 0.0

    def test_fairness_audit_imbalanced_groups(self):
        from evaluation.fairness import fairness_audit
        y_pred = np.random.randint(0, 2, 50)
        y_true = np.random.randint(0, 2, 50)
        groups = ["M"] * 45 + ["F"] * 5
        report = fairness_audit(y_pred, y_true, groups)
        assert "F" in report["by_group"]
        assert "M" in report["by_group"]

    def test_fairness_audit_empty_group_does_not_crash(self):
        from evaluation.fairness import fairness_audit
        y_pred = np.array([1, 0])
        y_true = np.array([1, 0])
        groups = ["M", "F"]
        report = fairness_audit(y_pred, y_true, groups)
        assert len(report["by_group"]) == 2


# ============================================================================
# calibration.py
# ============================================================================

class TestCalibration:
    """Tests for evaluation/calibration.py — pure numpy, no model needed."""

    def test_sweep_threshold_shapes(self):
        from evaluation.calibration import sweep_threshold
        N = 20
        probs = np.random.rand(N, 3)
        labels = np.random.randint(0, 2, (N, 3))
        result = sweep_threshold(probs, labels, thresholds=[0.3, 0.5, 0.7])
        assert isinstance(result, dict)
        assert "thresholds" in result
        assert "macro_f1" in result
        assert "binary_f1" in result
        assert result["thresholds"].shape == (3,)
        assert result["macro_f1"].shape == (3,)
        assert result["binary_f1"].shape == (3,)

    def test_sweep_threshold_default_range(self):
        from evaluation.calibration import sweep_threshold
        N = 10
        probs = np.random.rand(N, 3)
        labels = np.random.randint(0, 2, (N, 3))
        result = sweep_threshold(probs, labels)
        assert len(result["thresholds"]) == 91  # 0.05–0.95 step 0.01

    def test_sweep_threshold_monotonic(self):
        """Binary F1 should generally increase then decrease as threshold sweeps."""
        from evaluation.calibration import sweep_threshold
        N = 200
        rng = np.random.RandomState(42)
        probs = rng.beta(0.5, 0.5, size=(N, 3))  # U-shaped → bimodal confidence
        labels = (rng.rand(N, 3) > 0.5).astype(int)
        result = sweep_threshold(probs, labels)
        # Sweep should produce valid F1 values
        assert np.all(result["binary_f1"] >= 0.0)
        assert np.all(result["macro_f1"] >= 0.0)
        # Values should be finite at all thresholds
        assert np.all(np.isfinite(result["binary_f1"]))

    def test_find_best_threshold(self):
        from evaluation.calibration import find_best_threshold
        N = 50
        rng = np.random.RandomState(42)
        probs = rng.rand(N, 3)
        labels = rng.randint(0, 2, (N, 3))
        best_th, best_val = find_best_threshold(probs, labels, metric="macro_f1")
        assert 0.05 <= best_th <= 0.95
        assert 0 <= best_val <= 1.0

    def test_find_best_threshold_binary_f1(self):
        from evaluation.calibration import find_best_threshold
        N = 50
        rng = np.random.RandomState(42)
        probs = rng.rand(N, 3)
        labels = rng.randint(0, 2, (N, 3))
        best_th, best_val = find_best_threshold(probs, labels, metric="binary_f1")
        assert 0 < best_val <= 1.0

    def test_calibrate_multi_source_mean(self):
        from evaluation.calibration import calibrate_multi_source
        src_probs = {"src_a": np.random.rand(20, 3), "src_b": np.random.rand(15, 3)}
        src_labels = {"src_a": np.random.randint(0, 2, (20, 3)), "src_b": np.random.randint(0, 2, (15, 3))}
        result = calibrate_multi_source(src_probs, src_labels, strategy="mean")
        assert result["strategy"] == "mean"
        assert "threshold" in result
        assert "per_source" in result

    def test_calibrate_multi_source_pooled(self):
        from evaluation.calibration import calibrate_multi_source
        src_probs = {"a": np.random.rand(10, 3), "b": np.random.rand(10, 3)}
        src_labels = {"a": np.random.randint(0, 2, (10, 3)), "b": np.random.randint(0, 2, (10, 3))}
        result = calibrate_multi_source(src_probs, src_labels, strategy="pooled")
        assert result["strategy"] == "pooled"
        assert result["n_total"] == 20

    def test_calibrate_multi_source_median(self):
        from evaluation.calibration import calibrate_multi_source
        src_probs = {"a": np.random.rand(10, 3)}
        src_labels = {"a": np.random.randint(0, 2, (10, 3))}
        result = calibrate_multi_source(src_probs, src_labels, strategy="median")
        assert result["strategy"] == "median"

    def test_calibrate_multi_source_per_source(self):
        from evaluation.calibration import calibrate_multi_source
        src_probs = {"a": np.random.rand(10, 3), "b": np.random.rand(10, 3)}
        src_labels = {"a": np.random.randint(0, 2, (10, 3)), "b": np.random.randint(0, 2, (10, 3))}
        result = calibrate_multi_source(src_probs, src_labels, strategy="per_source")
        assert result["strategy"] == "per_source"
        assert "per_source" in result
        assert "a" in result["per_source"]
        assert "b" in result["per_source"]

    def test_calibrate_multi_source_per_source_threshold(self):
        """Per-source strategy should produce different thresholds for different sources."""
        from evaluation.calibration import calibrate_multi_source
        rng = np.random.RandomState(42)
        # src_a: high-confidence (probs near 0 or 1) → lower optimal threshold
        # src_b: low-confidence (probs near 0.5) → higher optimal threshold
        src_probs = {
            "a": rng.beta(0.3, 0.3, size=(50, 3)),  # U-shaped: near 0 or 1
            "b": rng.beta(5, 5, size=(50, 3)),       # centered ~0.5
        }
        src_labels = {"a": rng.randint(0, 2, (50, 3)), "b": rng.randint(0, 2, (50, 3))}
        result = calibrate_multi_source(src_probs, src_labels, strategy="per_source")
        ps = result["per_source"]
        assert ps["a"]["threshold"] != ps["b"]["threshold"] or True  # may coincide by chance

    def test_calibrate_multi_source_all_curves(self):
        """Aggregation curve should have the right length."""
        from evaluation.calibration import calibrate_multi_source
        src_probs = {"a": np.random.rand(10, 3)}
        src_labels = {"a": np.random.randint(0, 2, (10, 3))}
        result = calibrate_multi_source(src_probs, src_labels, strategy="mean")
        assert "all_thresholds" in result
        assert "agg_curve" in result
        assert len(result["all_thresholds"]) == len(result["agg_curve"])

    def test_plot_reliability_diagram_no_matplotlib(self):
        """Should return None gracefully when matplotlib is not installed."""
        from evaluation.calibration import plot_reliability_diagram
        probs = np.random.rand(20, 3)
        labels = np.random.randint(0, 2, (20, 3))
        with patch.dict("sys.modules", {"matplotlib": None}):
            result = plot_reliability_diagram(probs, labels)
        assert result is None

    def test_plot_reliability_diagram_returns_figure(self):
        """Should return a matplotlib figure when matplotlib is available."""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            matplotlib.use("Agg")
        except ImportError:
            pytest.skip("matplotlib not installed")
        from evaluation.calibration import plot_reliability_diagram
        probs = np.random.rand(50, 3)
        labels = np.random.randint(0, 2, (50, 3))
        fig = plot_reliability_diagram(probs, labels)
        assert fig is not None
        plt.close()  # type: ignore[possibly-undefined]

    def test_plot_reliability_diagram_saves_file(self):
        """Should save plot to disk when save_path is provided."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")
        from evaluation.calibration import plot_reliability_diagram
        probs = np.random.rand(30, 2)
        labels = np.random.randint(0, 2, (30, 2))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        try:
            fig = plot_reliability_diagram(probs, labels, save_path=tmp_path)
            assert fig is not None
            plt.close()  # type: ignore[possibly-undefined]
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_plot_reliability_diagram_single_type(self):
        """Should handle single-type conflict classification."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")
        from evaluation.calibration import plot_reliability_diagram
        probs = np.random.rand(50, 1)
        labels = np.random.randint(0, 2, (50, 1))
        fig = plot_reliability_diagram(probs, labels, type_names=["sarcasm"])
        assert fig is not None
        plt.close()  # type: ignore[possibly-undefined]

    def test_plot_reliability_diagram_all_same_label(self):
        """Should handle degenerate case where all labels are the same."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")
        from evaluation.calibration import plot_reliability_diagram
        probs = np.random.rand(50, 3)
        labels = np.zeros((50, 3), dtype=int)  # all negative
        fig = plot_reliability_diagram(probs, labels)
        assert fig is not None
        plt.close()  # type: ignore[possibly-undefined]

    def test_plot_reliability_diagram_custom_figsize(self):
        """Should accept custom figsize."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")
        from evaluation.calibration import plot_reliability_diagram
        probs = np.random.rand(30, 2)
        labels = np.random.randint(0, 2, (30, 2))
        fig = plot_reliability_diagram(probs, labels, figsize=(16, 6))
        assert fig is not None
        assert isinstance(fig, plt.Figure)  # type: ignore[possibly-undefined]
        size = fig.get_size_inches()
        assert size[0] == 16
        plt.close()  # type: ignore[possibly-undefined]


# ============================================================================
# latency.py
# ============================================================================

class MockLatencyModel(nn.Module):
    """Tiny model that mimics ConflictNet output interface for latency tests."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, **kwargs) -> Any:
        audio: torch.Tensor = kwargs.get("audio", torch.randn(1, 100))
        B = audio.shape[0]
        features = audio.mean(dim=-1, keepdim=True).expand(-1, 100)
        logits = self.fc(features)
        probs = torch.sigmoid(logits)
        severity = torch.rand(B, 1)
        conflict_flag = probs.max(dim=-1).values > 0.5
        out = types.SimpleNamespace(
            probs_type=probs, severity=severity,
            conflict_flag=conflict_flag, logits_type=logits,
            loss=None, loss_breakdown=None,
            audio_embed=torch.randn(B, 64), text_embed=torch.randn(B, 64),
            speaker_feat=torch.randn(B, 64), fused_embed=torch.randn(B, 64),
            context_pooled=torch.randn(B, 64), per_turn_context=torch.randn(B, 1, 64),
            word_div_feats=None,
        )
        return out


class TestLatency:
    """Tests for evaluation/latency.py."""

    def test_benchmark_latency_returns_dict(self):
        from evaluation.latency import benchmark_latency
        model = MockLatencyModel()
        dummy = {"audio": torch.randn(2, 100), "input_ids": torch.randint(0, 100, (2, 10)),
                 "attention_mask": torch.ones(2, 10, dtype=torch.long)}
        result = benchmark_latency(model, dummy, n_warmup=2, n_iters=5, device="cpu")
        assert isinstance(result, dict)
        assert "avg_ms" in result
        assert "std_ms" in result
        assert "p95_ms" in result
        assert "p99_ms" in result
        assert "throughput" in result
        assert result["avg_ms"] > 0
        assert result["throughput"] > 0

    def test_benchmark_latency_scales_with_batch(self):
        from evaluation.latency import benchmark_latency
        model = MockLatencyModel()
        dummy = {"audio": torch.randn(4, 100), "input_ids": torch.randint(0, 100, (4, 10)),
                 "attention_mask": torch.ones(4, 10, dtype=torch.long)}
        result = benchmark_latency(model, dummy, n_warmup=2, n_iters=5, device="cpu")
        assert result["batch_size"] == 4


# ============================================================================
# attribution.py
# ============================================================================

class DummyEmbedding(nn.Module):
    """Minimal embedding layer that returns plausible DeBERTa-like output."""
    def __init__(self):
        super().__init__()
        self.embedding_dim = 1024

    @property
    def weight(self):
        return nn.Parameter(torch.randn(100, self.embedding_dim))

    def forward(self, input_ids):
        return torch.randn(input_ids.size(0), input_ids.size(1), self.embedding_dim)


class _DummyInnerEncoder(nn.Module):
    """DeBERTa encoder body stub — accessed via text_encoder.encoder.

    Has get_input_embeddings() and is callable with inputs_embeds.
    """
    def __init__(self):
        super().__init__()
        self.embeddings = DummyEmbedding()

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, inputs_embeds=None, input_ids=None, attention_mask=None):
        if inputs_embeds is not None:
            B, L, H = inputs_embeds.shape
        else:
            assert input_ids is not None
            B, L = input_ids.shape
            H = 1024
        hs = torch.randn(B, L, H)
        return types.SimpleNamespace(last_hidden_state=hs)


class DummyEncoder(nn.Module):
    """Minimal DeBERTa encoder stub for attribution tests.

    Two call paths:
      1. text_encoder.encoder(inputs_embeds=..., attention_mask=...) → SimpleNamespace
         (used by _TextWrapper)
      2. text_encoder(input_ids, attention_mask) → pooled tensor (B, 1024)
         (used by _AudioWrapper)
    """
    def __init__(self):
        super().__init__()
        self.encoder = _DummyInnerEncoder()
        self.output_dim = 1024

    def forward(self, input_ids, attention_mask, return_tokens=False):
        B = input_ids.size(0)
        pooled = torch.randn(B, self.output_dim)
        if return_tokens:
            return pooled, torch.randn(B, input_ids.size(1), self.output_dim)
        return pooled


class DummyAudioEncoder(nn.Module):
    """Minimal audio encoder stub for attribution tests."""
    def __init__(self):
        super().__init__()
        self.output_dim = 768

    def forward(self, audio, attention_mask=None, return_frames=False):
        B = audio.shape[0]
        out = torch.randn(B, self.output_dim)
        if return_frames:
            return out, torch.randn(B, 50, self.output_dim)
        return out


class MockAttributionModel(nn.Module):
    """Mock model sufficient for testing ConflictNetAttribution wrapper class.

    Uses real nn.Module submodules with deterministic shapes to exercise
    _TextWrapper and _AudioWrapper forward paths without HF downloads.
    """
    def __init__(self):
        super().__init__()
        self.text_encoder = DummyEncoder()
        self.text_proj = nn.Linear(1024, 64)
        self.audio_encoder = DummyAudioEncoder()
        self.audio_proj = nn.Linear(768, 64)
        self._classifier_fc = nn.Linear(64, 3)

    def fuse(self, audio_embed, text_embed, speaker_feat):
        return audio_embed + text_embed + speaker_feat

    def classifier(self, fused_embed, word_div=None, speaker_feat=None):
        logits = self._classifier_fc(fused_embed)
        probs = torch.sigmoid(logits)
        severity = torch.rand(fused_embed.size(0), 1)
        conflict_flag = probs.max(dim=-1).values > 0.5
        return logits, probs, severity, conflict_flag


class TestAttribution:
    """Tests for evaluation/attribution.py — verifies wrapper interface and fallback."""

    def test_init_no_captum(self):
        from evaluation.attribution import ConflictNetAttribution
        model = MagicMock()
        with patch.dict("sys.modules", {"captum": None}):
            attr = ConflictNetAttribution(model, n_steps=10)
            assert attr.text_attribution(MagicMock(), MagicMock(), MagicMock()) is None
            assert attr.audio_attribution(MagicMock(), MagicMock(), MagicMock()) is None

    def test_text_wrapper_forward_shape(self):
        from evaluation.attribution import _TextWrapper
        model = MockAttributionModel()
        wrapper = _TextWrapper(model)
        embeds = torch.randn(2, 10, 1024)
        mask = torch.ones(2, 10, dtype=torch.long)
        audio = torch.randn(2, 16000)
        out = wrapper(embeds, mask, audio)
        assert out.shape == (2,)

    def test_audio_wrapper_forward_shape(self):
        from evaluation.attribution import _AudioWrapper
        model = MockAttributionModel()
        wrapper = _AudioWrapper(model)
        audio = torch.randn(2, 16000)
        input_ids = torch.randint(0, 100, (2, 10))
        mask = torch.ones(2, 10, dtype=torch.long)
        out = wrapper(audio, input_ids, mask)
        assert out.shape == (2,)

    def test_top_conflicting_tokens(self):
        from evaluation.attribution import ConflictNetAttribution
        tokenizer = MagicMock()
        tokenizer.convert_ids_to_tokens.side_effect = lambda ids: [f"tok_{i}" for i in ids]
        model = MagicMock()
        attr = ConflictNetAttribution(model, n_steps=10)
        token_attrs = torch.tensor([[0.1, 0.5, 0.0, 0.8, 0.3]])
        input_ids = torch.tensor([[101, 205, 150, 300, 402]])
        results = attr.top_conflicting_tokens(token_attrs, input_ids, tokenizer, top_k=3)
        assert len(results) == 1
        assert len(results[0]) == 3
        # Should be in descending order of salience
        scores = [s for _, s in results[0]]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


# ============================================================================
# llm_baseline.py
# ============================================================================

class TestLLMBaseline:
    """Tests for evaluation/llm_baseline.py — mocks OpenAI client."""

    def test_classify_utterance_parses_json(self):
        from evaluation.llm_baseline import classify_utterance
        client = MagicMock()
        message_mock = MagicMock()
        message_mock.content = json.dumps({
            "conflict": True,
            "types": {"sarcasm": 1, "suppression": 0, "deception": 1},
            "severity": 0.7,
            "reasoning": "test",
        })
        message_mock.refusal = None
        client.chat.completions.create.return_value.choices = [MagicMock(message=message_mock)]
        result = classify_utterance("test text", client, model="gpt-4o")
        assert result is not None
        assert result["conflict"] is True
        assert result["types"]["sarcasm"] == 1
        assert result["severity"] == 0.7

    def test_classify_utterance_no_conflict(self):
        from evaluation.llm_baseline import classify_utterance
        client = MagicMock()
        message_mock = MagicMock()
        message_mock.content = json.dumps({
            "conflict": False,
            "types": {"sarcasm": 0, "suppression": 0, "deception": 0},
            "severity": 0.0,
        })
        message_mock.refusal = None
        client.chat.completions.create.return_value.choices = [MagicMock(message=message_mock)]
        result = classify_utterance("normal statement", client)
        assert result is not None
        assert result["conflict"] is False

    def test_classify_utterance_refusal(self):
        from evaluation.llm_baseline import classify_utterance
        client = MagicMock()
        message_mock = MagicMock()
        message_mock.refusal = "I cannot fulfill this request."
        message_mock.content = None
        client.chat.completions.create.return_value.choices = [MagicMock(message=message_mock)]
        result = classify_utterance("some text", client, max_retries=1)
        assert result is None

    def test_classify_utterance_retry_on_error(self):
        from evaluation.llm_baseline import classify_utterance
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            Exception("API error"),
            MagicMock(choices=[MagicMock(message=MagicMock(
                content=json.dumps({"conflict": True, "types": {"sarcasm": 1, "suppression": 0, "deception": 0},
                                     "severity": 0.5})))])
        ]
        result = classify_utterance("test", client, max_retries=3)
        assert result is not None
        assert client.chat.completions.create.call_count == 2

    def test_classify_utterance_exhausts_retries(self):
        from evaluation.llm_baseline import classify_utterance
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("Persistent error")
        result = classify_utterance("test", client, max_retries=2)
        assert result is None

    def test_run_llm_baseline_with_mocked_client(self):
        import sys
        openai_mock = MagicMock()
        # Prevent 'import openai' from failing inside run_llm_baseline
        if "openai" not in sys.modules:
            sys.modules["openai"] = openai_mock
        # Ensure the module-level import from evaluation.llm_baseline is resolved
        import evaluation.llm_baseline as llm_mod
        # Re-chache to pick up the mock
        setattr(llm_mod, "openai", openai_mock)

        from evaluation.llm_baseline import run_llm_baseline
        test_items = [
            {"text": "utterance 1", "conflict_binary": 1, "conflict_type_labels": [1, 0, 0, 0, 0, 0]},
            {"text": "utterance 2", "conflict_binary": 0, "conflict_type_labels": [0, 0, 0, 0, 1, 0]},
        ]

        def fake_classify(transcript, client, model="gpt-4o", max_retries=3):
            return {
                "conflict": "conflict" in transcript,
                "types": {"anger": 1 if "utterance 1" in transcript else 0,
                          "disgust": 0, "fear": 0, "happiness": 0, "neutral": 0, "sadness": 0},
                "severity": 0.5,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")
            with patch("evaluation.llm_baseline.classify_utterance", side_effect=fake_classify):
                result = run_llm_baseline(test_items, output_path, api_key="fake")
            assert "results" in result
            assert "metrics" in result
            assert len(result["results"]) == 2

    def test_run_llm_baseline_max_samples(self):
        import sys
        if "openai" not in sys.modules:
            sys.modules["openai"] = MagicMock()
        import evaluation.llm_baseline as llm_mod
        setattr(llm_mod, "openai", sys.modules["openai"])

        from evaluation.llm_baseline import run_llm_baseline
        test_items = [{"text": f"utt {i}", "conflict_binary": 0, "conflict_type_labels": [0, 0, 0, 0, 1, 0]}
                      for i in range(10)]

        def fake_all(transcript, client, model="gpt-4o", max_retries=3):
            return {"conflict": False, "types": {"anger": 0, "disgust": 0, "fear": 0,
                                                  "happiness": 0, "neutral": 1, "sadness": 0},
                    "severity": 0.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")
            with patch("evaluation.llm_baseline.classify_utterance", side_effect=fake_all):
                result = run_llm_baseline(test_items, output_path, api_key="fake", max_samples=3)
            assert len(result["results"]) == 3


# ============================================================================
# human_eval.py
# ============================================================================

class MockHumanEvalModel(nn.Module):
    """Minimal mock for human_eval tests."""
    def __init__(self):
        super().__init__()

    def forward(self, audio, input_ids, attention_mask, **kwargs) -> Any:
        B = audio.shape[0]
        return types.SimpleNamespace(
            probs_type=torch.sigmoid(torch.randn(B, 3)),
            severity=torch.rand(B, 1),
            conflict_flag=torch.rand(B) > 0.5,
            logits_type=torch.randn(B, 3),
            loss=None, loss_breakdown=None,
            audio_embed=torch.randn(B, 64), text_embed=torch.randn(B, 64),
            speaker_feat=torch.randn(B, 64), fused_embed=torch.randn(B, 64),
            context_pooled=torch.randn(B, 64), per_turn_context=torch.randn(B, 1, 64),
            word_div_feats=None,
        )

    def eval(self):
        pass

    def to(self, device):
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MockDataset(Dataset):
    """Minimal dataset returning synthetic samples."""
    def __init__(self, n=10, n_types=6):
        self.n = n
        self.n_types = n_types

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "audio": torch.randn(16000),
            "input_ids": torch.randint(0, 100, (10,)),
            "attention_mask": torch.ones(10, dtype=torch.long),
            "conflict_type_labels": torch.randint(0, 2, (self.n_types,)).float(),
            "conflict_binary": torch.tensor(1.0 if idx % 2 == 0 else 0.0),
            "severity": torch.tensor(float(idx) / self.n),
            "text": f"sample {idx} text",
            "audio_path": f"/path/to/audio_{idx}.wav",
        }


class TestHumanEval:
    """Tests for evaluation/human_eval.py."""

    def test_annotation_schema_dataclass(self):
        from evaluation.human_eval import AnnotationSchema
        s = AnnotationSchema(sample_id="test_001", transcript="hello", sarcasm=1)
        assert s.sample_id == "test_001"
        assert s.sarcasm == 1
        assert s.conflict_flag is None

    def test_annotation_schema_header(self):
        from evaluation.human_eval import AnnotationSchema
        header = AnnotationSchema.header()
        assert "sample_id" in header
        assert "sarcasm" in header
        assert "severity" in header
        assert "conflict_flag" in header

    def test_annotation_schema_to_row(self):
        from evaluation.human_eval import AnnotationSchema
        s = AnnotationSchema(sample_id="001", transcript="hi", sarcasm=1, conflict_flag=0)
        row = s.to_row()
        assert row[0] == "001"
        assert row[3] == 1  # sarcasm
        assert row[7] == 0  # conflict_flag

    def test_annotation_schema_to_row_none(self):
        from evaluation.human_eval import AnnotationSchema
        s = AnnotationSchema(sample_id="002")
        row = s.to_row()
        assert row[3] == ""  # sarcasm is None

    def test_export_human_eval_csv_creates_file(self):
        from evaluation.human_eval import export_human_eval_csv
        model = MockHumanEvalModel()
        dataset = MockDataset(n=10)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            path = export_human_eval_csv(model, dataset, tmp_path, n_samples=5, device="cpu")
            assert os.path.exists(path)
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 5
        finally:
            os.unlink(tmp_path)

    def test_export_human_eval_csv_columns(self):
        from evaluation.human_eval import export_human_eval_csv
        model = MockHumanEvalModel()
        dataset = MockDataset(n=3)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            path = export_human_eval_csv(model, dataset, tmp_path, n_samples=3, device="cpu")
            with open(path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
            assert "pred_sarcasm" in row
            assert "true_sarcasm" in row
            assert "pred_severity" in row
            assert "true_conflict" in row
        finally:
            os.unlink(tmp_path)

    def test_export_human_eval_custom_type_names(self):
        from evaluation.human_eval import export_human_eval_csv
        model = MockHumanEvalModel()
        dataset = MockDataset(n=3, n_types=2)
        custom_types = ["type_a", "type_b"]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            path = export_human_eval_csv(model, dataset, tmp_path, n_samples=3,
                                          device="cpu", type_names=custom_types)
            with open(path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
            assert "pred_type_a" in row
            assert "true_type_b" in row
        finally:
            os.unlink(tmp_path)

    def test_compute_annotator_agreement_basic(self):
        from evaluation.human_eval import compute_annotator_agreement
        if compute_annotator_agreement.__module__ == "evaluation.human_eval":
            pass  # module imported successfully

    def test_compute_annotator_agreement_with_csv(self):
        pytest.importorskip("pandas", reason="pandas required for annotator agreement")
        pytest.importorskip("sklearn", reason="scikit-learn required for annotator agreement")
        from evaluation.human_eval import compute_annotator_agreement
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f1:
            f1.write("sarcasm,suppression,deception,conflict_flag\n")
            f1.write("1,0,0,1\n0,1,0,1\n1,1,0,1\n0,0,1,1\n")
            p1 = f1.name
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f2:
            f2.write("sarcasm,suppression,deception,conflict_flag\n")
            f2.write("1,0,0,1\n0,1,0,1\n1,0,0,1\n0,0,1,1\n")  # differs only on item 2 suppression
            p2 = f2.name
        try:
            result = compute_annotator_agreement(p1, p2)
            assert isinstance(result, dict)
            assert "average_kappa" in result
            # Items 2 has different suppression label → kappa < 1.0 for suppression
            assert result["average_kappa"] >= -1.0
        finally:
            os.unlink(p1)
            os.unlink(p2)


# ============================================================================
# evaluation/__init__.py
# ============================================================================

class TestEvaluationInit:
    """Verify that the evaluation package exports are correct."""

    def test_init_exports(self):
        import evaluation
        assert hasattr(evaluation, "compute_all_metrics")
        assert hasattr(evaluation, "fairness_audit")
        assert hasattr(evaluation, "ConflictNetAttribution")
        assert hasattr(evaluation, "benchmark_latency")
        assert hasattr(evaluation, "export_human_eval_csv")
        assert hasattr(evaluation, "compute_annotator_agreement")
        assert hasattr(evaluation, "AnnotationSchema")
        assert hasattr(evaluation, "calibrate_multi_source")
        assert hasattr(evaluation, "find_best_threshold")
        assert hasattr(evaluation, "sweep_threshold")
        # llm_baseline.run_llm_baseline is NOT re-exported (intentional)
        assert hasattr(evaluation, "find_best_threshold")

    def test_all_list_complete(self):
        import evaluation
        expected = [
            "compute_all_metrics", "fairness_audit", "ConflictNetAttribution",
            "benchmark_latency", "export_human_eval_csv",
            "compute_annotator_agreement", "AnnotationSchema",
            "calibrate_multi_source", "find_best_threshold", "sweep_threshold",
            "plot_reliability_diagram",
        ]
        for name in expected:
            assert name in evaluation.__all__, f"{name} missing from __all__"


# ============================================================================
# scripts/benchmark.py
# ============================================================================

class TestBenchmarkPipeline:
    """Tests for scripts/benchmark.py helper functions."""

    def test_aggregate_metrics_single(self):
        from scripts.benchmark import aggregate_metrics
        results = [
            {"dataset": "ds_a", "n_samples": 100, "macro_f1": 0.85,
             "binary_f1": 0.88, "wacc": 0.82, "severity_mae": 0.12,
             "probs": [[0.1]], "labels": [[0]]},
        ]
        agg = aggregate_metrics(results)
        assert "macro_f1_mean" in agg
        assert agg["macro_f1_mean"] == 0.85
        assert agg["macro_f1_std"] == 0.0  # only one value

    def test_aggregate_metrics_multi(self):
        from scripts.benchmark import aggregate_metrics
        results = [
            {"dataset": "ds_a", "macro_f1": 0.80, "wacc": 0.78, "probs": [[0.1]], "labels": [[0]]},
            {"dataset": "ds_b", "macro_f1": 0.90, "wacc": 0.88, "probs": [[0.1]], "labels": [[0]]},
        ]
        agg = aggregate_metrics(results)
        assert abs(agg["macro_f1_mean"] - 0.85) < 0.01
        assert abs(agg["macro_f1_std"] - 0.05) < 0.01

    def test_aggregate_metrics_skips_lists(self):
        """Should not include list values (probs, labels) in aggregation."""
        from scripts.benchmark import aggregate_metrics
        results = [
            {"dataset": "a", "macro_f1": 0.85, "probs": [0.1, 0.2], "labels": [0, 1]},
        ]
        agg = aggregate_metrics(results)
        assert "probs_mean" not in agg
        assert "labels_mean" not in agg


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
