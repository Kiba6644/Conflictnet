"""Human evaluation framework for ConflictNet.

Provides:
  - export_human_eval_csv: export model predictions + ground truth for human annotation
  - AnnotationSchema: describes the annotation task for human raters
  - compute_annotator_agreement: Cohen's kappa between two annotators
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

try:
    from sklearn.metrics import cohen_kappa_score  # type: ignore
except ImportError:
    cohen_kappa_score = None

logger = logging.getLogger(__name__)

TYPE_NAMES = ["sarcasm", "suppression", "deception"]


@dataclass
class AnnotationSchema:
    """Describes the annotation task for human raters.

    Each sample is a dialogue utterance. The rater must label:
      - Whether the utterance contains conflict (binary)
      - Which subtypes are present (multi-label)
      - Severity of the conflict (1-5 Likert scale)
    """

    sample_id: str = ""
    audio_path: str = ""
    transcript: str = ""

    # Annotation fields
    sarcasm: Optional[int] = None
    suppression: Optional[int] = None
    deception: Optional[int] = None
    severity: Optional[int] = None
    conflict_flag: Optional[int] = None

    @classmethod
    def header(cls) -> List[str]:
        return [
            "sample_id", "audio_path", "transcript",
            "sarcasm", "suppression", "deception",
            "severity", "conflict_flag",
        ]

    def to_row(self) -> List[Any]:
        return [
            self.sample_id,
            self.audio_path,
            self.transcript,
            "" if self.sarcasm is None else self.sarcasm,
            "" if self.suppression is None else self.suppression,
            "" if self.deception is None else self.deception,
            "" if self.severity is None else self.severity,
            "" if self.conflict_flag is None else self.conflict_flag,
        ]


@torch.no_grad()
def export_human_eval_csv(
    model: torch.nn.Module,
    dataset: Dataset,
    output_path: str,
    n_samples: int = 200,
    device: str = "cuda",
    type_names: Optional[List[str]] = None,
) -> str:
    """Export model predictions + ground truth for human annotation.

    For each sample (up to n_samples), computes model predictions and writes
    them alongside ground truth labels to a CSV file.

    Args:
        model: Trained ConflictNet model.
        dataset: Evaluation dataset.
        output_path: Path for the output CSV.
        n_samples: Maximum number of samples to export.
        device: Device for inference.

    Returns:
        Path to the written CSV file.
    """
    if type_names is None:
        type_names = TYPE_NAMES

    model.eval()
    model.to(device)

    n_types = len(type_names)
    rows: List[Dict[str, Any]] = []
    n = min(n_samples, len(dataset))  # type: ignore[arg-type]

    for i in range(n):
        try:
            sample = dataset[i]
        except Exception as e:
            logger.warning(f"[HumanEval] Skipping sample {i}: {e}")
            continue

        audio = sample["audio"].unsqueeze(0).to(device)
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

        out = model(audio=audio, input_ids=input_ids, attention_mask=attention_mask)
        probs = out.probs_type.cpu().numpy()[0]
        severity_pred = float(out.severity.squeeze(-1).cpu().item()) if out.severity is not None else None
        conflict_pred = int(out.conflict_flag.cpu().item())

        true_type = sample.get("conflict_type_labels", torch.zeros(n_types)).numpy()
        true_severity = float(sample.get("severity", torch.tensor(float("nan"))).item())
        true_conflict = int(sample.get("conflict_binary", torch.tensor(0)).item())

        row: Dict[str, Any] = {
            "sample_id": i,
            "audio_path": sample.get("audio_path", ""),
            "transcript": sample.get("text", ""),
        }
        for j, name in enumerate(type_names):
            row[f"pred_{name}"] = float(probs[j])
            row[f"true_{name}"] = int(true_type[j])
        row["pred_severity"] = severity_pred
        row["pred_conflict"] = conflict_pred
        row["true_severity"] = true_severity
        row["true_conflict"] = true_conflict

        rows.append(row)

    fieldnames = [
        "sample_id", "audio_path", "transcript",
        *[f"pred_{n}" for n in type_names],
        "pred_severity", "pred_conflict",
        *[f"true_{n}" for n in type_names],
        "true_severity", "true_conflict",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"[HumanEval] Exported {len(rows)} samples to {output_path}")
    return output_path


def compute_annotator_agreement(
    csv_path1: str,
    csv_path2: str,
    label_columns: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute Cohen's kappa between two annotators.

    Args:
        csv_path1: Path to first annotator CSV.
        csv_path2: Path to second annotator CSV.
        label_columns: Column names to compare. Defaults to binary annotations.

    Returns:
        Dict mapping column name to Cohen's kappa value.
    """
    if pd is None:
        logger.error("pandas is required for annotator agreement computation")
        return {}
    if cohen_kappa_score is None:
        logger.error("scikit-learn is required for annotator agreement computation")
        return {}

    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)

    if label_columns is None:
        label_columns = [
            "sarcasm", "suppression", "deception", "conflict_flag"
        ]

    agreements = {}
    for col in label_columns:
        if col in df1.columns and col in df2.columns:
            valid = df1[col].notna() & df2[col].notna()
            if valid.sum() < 2:
                agreements[col] = float("nan")
                continue
            kappa = cohen_kappa_score(df1[col][valid], df2[col][valid])
            agreements[col] = round(kappa, 4)

    avg = np.mean([v for v in agreements.values() if not np.isnan(v)])
    agreements["average_kappa"] = round(float(avg), 4)

    logger.info(f"[HumanEval] Annotator agreement: {agreements}")
    return agreements
