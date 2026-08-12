"""Pydantic schemas for ConflictNet serving API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., description="ok or error")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    device: str = Field(..., description="Compute device (cpu / cuda)")


# ── Predict (single utterance) ──────────────────────────────────────────────

class PredictResponse(BaseModel):
    conflict: bool = Field(..., description="Whether conflict is detected")
    probs: Dict[str, float] = Field(
        ..., description="Per-type sigmoid probabilities {anger, disgust, fear, happiness, neutral, sadness}"
    )
    severity: float = Field(..., ge=0.0, le=1.0, description="Conflict severity score")
    predicted_type: str = Field(
        ..., description="Predicted emotion: anger / disgust / fear / happiness / neutral / sadness / none"
    )
    fused_embed: Optional[List[float]] = Field(
        None, description="256-d fused audio-text embedding for context accumulation"
    )


# ── Predict batch ───────────────────────────────────────────────────────────

class BatchItem(BaseModel):
    audio: Any = Field(..., description="Raw WAV bytes (base64-encoded)")
    text: str = Field(..., description="Utterance text")
    context_embeds: Optional[List[List[float]]] = Field(
        None, description="Past turn embeddings (T_turns, embed_dim)"
    )
    prosody_z: Optional[List[float]] = Field(
        None, description="Prosody z-scores [f0_z, energy_z, rate_z]"
    )

class PredictBatchRequest(BaseModel):
    items: List[BatchItem] = Field(..., min_length=1, max_length=64)


class PredictBatchResponse(BaseModel):
    results: List[PredictResponse]


# ── Error ───────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
