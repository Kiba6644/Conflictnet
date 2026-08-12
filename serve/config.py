"""Serving configuration — loaded from environment or defaults."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ServeConfig:
    checkpoint_path: str = "checkpoints/best_model.safetensors"
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 8000
    max_audio_len: float = 10.0
    max_text_len: int = 512
    compile: bool = True
    amp: bool = True

    # Model construction args (must match training)
    audio_encoder: str = "emotion2vec"
    embed_dim: int = 256
    use_speaker_norm: bool = True
    use_temporal: bool = True
    use_word_divergence: bool = True
    use_cross_attn_injection: bool = True
    use_speaker_adaptive_threshold: bool = True
    use_baseline_subtract: bool = True
    lora_r: int = 16
    temporal_max_turns: int = 16

    # CORS
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls) -> "ServeConfig":
        import os
        kwargs: dict = {}
        for f in cls.__dataclass_fields__:
            env_val = os.environ.get(f"SERVE_{f.upper()}")
            if env_val is not None:
                field_type = cls.__dataclass_fields__[f].type
                if field_type is bool:
                    kwargs[f] = env_val.lower() in ("1", "true", "yes")
                elif field_type is int:
                    kwargs[f] = int(env_val)
                elif field_type is float:
                    kwargs[f] = float(env_val)
                elif field_type is list and "str" in str(field_type):
                    kwargs[f] = [x.strip() for x in env_val.split(",")]
                else:
                    kwargs[f] = env_val
            env_val_list = os.environ.get(f"SERVE_{f.upper()}_LIST")
            if env_val_list is not None:
                kwargs[f] = [x.strip() for x in env_val_list.split(",")]
        return cls(**kwargs)
