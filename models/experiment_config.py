"""Structured experiment configuration for ConflictNet.

Serializable to/from JSON for checkpoint metadata, enabling full
experiment reproducibility tracing.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ExperimentConfig:
    """Immutable snapshot of all experiment parameters.

    Round-trips through JSON for checkpoint metadata.  Created from
    ``argparse.Namespace`` (the ``parse_args()`` output) via ``from_args()``.

    Fields are grouped: model architecture, training hyper-parameters,
    ablation toggles, data configuration, and runtime environment.
    """

    # ── Model architecture ──────────────────────────────────────────────
    audio_encoder: str = "emotion2vec"
    embed_dim: int = 256
    lora_r: int = 16

    # ── Training hyper-parameters ───────────────────────────────────────
    epochs: int = 30
    pretrain_epochs: int = 5
    batch_size: int = 16
    lr: float = 2e-5
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    early_stop_patience: int = 10
    seed: int = 42
    amp: bool = False

    # ── Ablation toggles ────────────────────────────────────────────────
    use_speaker_norm: bool = True
    use_temporal: bool = True
    use_cross_attn_injection: bool = True
    use_speaker_adaptive_threshold: bool = True
    use_baseline_subtract: bool = True
    use_word_divergence: bool = True

    # ── Data configuration ──────────────────────────────────────────────
    temporal_max_turns: int = 8
    prosody_stats: Optional[str] = None
    resume_from: Optional[str] = None
    train_datasets: Optional[List[str]] = None
    val_datasets: Optional[List[str]] = None

    # ── Runtime (not typically serialized, but carried for completeness) ─
    device: str = "cuda"
    output_dir: str = "checkpoints"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExperimentConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, raw: str) -> ExperimentConfig:
        return cls.from_dict(json.loads(raw))

    @classmethod
    def from_args(cls, args: Any) -> ExperimentConfig:
        """Build from an ``argparse.Namespace`` (or any object with matching attrs)."""
        kwargs: Dict[str, Any] = {}

        # Map CLI names to config field names
        cli_to_field = {
            "no_speaker_norm": "use_speaker_norm",
            "no_temporal": "use_temporal",
            "no_cross_attn_injection": "use_cross_attn_injection",
            "no_speaker_adaptive_threshold": "use_speaker_adaptive_threshold",
            "no_baseline_subtract": "use_baseline_subtract",
            "no_word_divergence": "use_word_divergence",
        }

        known = {f.name for f in dataclasses.fields(cls)}
        for attr_name in dir(args):
            if attr_name.startswith("_"):
                continue
            val = getattr(args, attr_name)
            if attr_name in known and val is not None:
                kwargs[attr_name] = val
            elif attr_name in cli_to_field:
                # Invert: CLI stores action_store_true as "True when disabled"
                kwargs[cli_to_field[attr_name]] = not val

        return cls(**kwargs)

    def to_cli_args(self) -> Dict[str, Any]:
        """Inverse of ``from_args()`` — produce CLI override dict."""
        out: Dict[str, Any] = {}
        field_to_cli = {
            "use_speaker_norm": "no_speaker_norm",
            "use_temporal": "no_temporal",
            "use_cross_attn_injection": "no_cross_attn_injection",
            "use_speaker_adaptive_threshold": "no_speaker_adaptive_threshold",
            "use_baseline_subtract": "no_baseline_subtract",
            "use_word_divergence": "no_word_divergence",
        }

        for field_name in (f.name for f in dataclasses.fields(self)):
            val = getattr(self, field_name)
            if field_name in field_to_cli:
                out[field_to_cli[field_name]] = not val
            elif field_name in {
                "device", "output_dir", "train_datasets", "val_datasets",
            }:
                continue  # runtime, not CLI-expressible
            else:
                out[field_name] = val
        return out

    def __post_init__(self):
        if self.temporal_max_turns < 1:
            raise ValueError(f"temporal_max_turns must be >= 1, got {self.temporal_max_turns}")
