"""Fairness audit using FairLearn.

Evaluates demographic parity and equalized odds across:
  - Gender (M / F)
  - Any other provided sensitive attribute

Usage:
    from evaluation.fairness import fairness_audit
    report = fairness_audit(preds, labels, sensitive_features=genders)
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np


def fairness_audit(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    sensitive_features: Union[List[Any], Dict[str, List[Any]]],
    metric_fn=None,
) -> Dict[str, Any]:
    """Run FairLearn fairness audit across multiple attributes.

    Args:
        y_pred: Binary predictions (N,).
        y_true: Ground truth labels (N,).
        sensitive_features: List of group membership per sample or Dict of group membership per sample (e.g. {"gender": [...], "age": [...]}).
        metric_fn: Metric function (default: f1_score).

    Returns:
        Dict with per-attribute reports, or direct report if list was passed.
    """
    from typing import Union
    
    is_list = isinstance(sensitive_features, list)
    if is_list:
        sensitive_features = {"group": sensitive_features}
        
    reports = {}
    try:
        from fairlearn.metrics import (  # type: ignore
            MetricFrame,
            demographic_parity_difference,
            equalized_odds_difference,
        )
        from sklearn.metrics import f1_score  # type: ignore

        if metric_fn is None:
            def metric_fn(y, p):
                return f1_score(y, p, zero_division=0)

        for attr_name, attr_values in sensitive_features.items():
            if not attr_values or len(attr_values) != len(y_true):
                continue
            
            mf = MetricFrame(
                metrics={"f1": metric_fn},
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=attr_values,
            )

            dpd = demographic_parity_difference(
                y_true=y_true, y_pred=y_pred, sensitive_features=attr_values
            )
            eod = equalized_odds_difference(
                y_true=y_true, y_pred=y_pred, sensitive_features=attr_values
            )

            # Handle new and old FairLearn versions
            if hasattr(mf.by_group, "to_dict"):
                by_group_raw = mf.by_group.to_dict()
                by_group = by_group_raw.get("f1", by_group_raw)
            else:
                by_group = mf.by_group.to_dict() if hasattr(mf.by_group, "to_dict") else dict(mf.by_group)
                
            if "f1" in by_group and isinstance(by_group["f1"], dict):
                by_group = by_group["f1"]

            if hasattr(mf.overall, "to_dict"):
                overall_raw = mf.overall.to_dict()
                overall = float(overall_raw.get("f1", overall_raw))
            else:
                overall = float(mf.overall["f1"]) if isinstance(mf.overall, dict) else float(mf.overall)
            diffs = mf.difference()
            if hasattr(diffs, "to_dict"):
                diffs_raw = diffs.to_dict()
                disparity = float(diffs_raw.get("f1", diffs_raw))
            else:
                disparity = float(diffs["f1"]) if isinstance(diffs, dict) or hasattr(diffs, "__getitem__") and "f1" in diffs else float(diffs)

            reports[attr_name] = {
                "overall_f1": overall,
                "by_group": by_group,
                "disparity": disparity,
                "demographic_parity_difference": float(dpd),
                "equalized_odds_difference": float(eod),
            }

        return reports["group"] if is_list else reports

    except ImportError:
        print("[WARN] fairlearn not installed. Run: pip install fairlearn")
        from sklearn.metrics import f1_score  # type: ignore

        for attr_name, attr_values in sensitive_features.items():
            if not attr_values or len(attr_values) != len(y_true):
                continue
                
            groups = sorted(set(attr_values))
            by_group = {}
            for g in groups:
                mask = np.array(attr_values) == g
                if mask.sum() > 0:
                    by_group[g] = float(f1_score(y_true[mask], y_pred[mask], zero_division=0))

            values = list(by_group.values())
            reports[attr_name] = {
                "overall_f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "by_group": by_group,
                "disparity": max(values) - min(values) if len(values) >= 2 else 0.0,
                "demographic_parity_difference": None,
                "equalized_odds_difference": None,
            }

        return reports["group"] if is_list else reports
