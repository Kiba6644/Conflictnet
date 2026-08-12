"""Evaluation: metrics, fairness audit, Captum attribution, LLM baseline,
latency, human eval, OOD probe, multi-source calibration.

Modules:
  - metrics.py       → WAcc, macro-F1, per-type AP, severity MAE
  - fairness.py      → FairLearn demographic parity + equalized odds
  - attribution.py   → Captum integrated gradients for token/frame attribution
  - llm_baseline.py  → GPT-4o text-only ceiling
  - latency.py       → Inference latency benchmarking
  - human_eval.py    → Human evaluation CSV export + annotator agreement
  - ood_probe.py     → Speaker-OOD held-out evaluation (CLI: ``python evaluation/ood_probe.py``)
  - calibration.py   → Multi-source threshold calibration (CLI: ``python evaluation/calibration.py``)
"""

from .metrics import compute_all_metrics
from .fairness import fairness_audit
from .attribution import ConflictNetAttribution
from .latency import benchmark_latency
from .human_eval import export_human_eval_csv, compute_annotator_agreement, AnnotationSchema
from .calibration import calibrate_multi_source, find_best_threshold, sweep_threshold, plot_reliability_diagram

__all__ = [
    "compute_all_metrics", "fairness_audit", "ConflictNetAttribution",
    "benchmark_latency", "export_human_eval_csv",
    "compute_annotator_agreement", "AnnotationSchema",
    "calibrate_multi_source", "find_best_threshold", "sweep_threshold",
    "plot_reliability_diagram",
]
