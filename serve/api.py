"""FastAPI inference server for ConflictNet.

Endpoints:
  - ``GET  /health``        — health check (model loaded + ready)
  - ``POST /predict``       — single utterance prediction
  - ``POST /predict_batch`` — batch prediction
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, Form, HTTPException

from serve.config import ServeConfig
from serve.model import ServeModel
from serve.schemas import (
    PredictResponse,
    PredictBatchResponse,
    PredictBatchRequest,
    HealthResponse,
)

logger = logging.getLogger(__name__)


def create_app(cfg: Optional[ServeConfig] = None) -> fastapi.FastAPI:
    if cfg is None:
        cfg = ServeConfig.from_env()

    serve_model = ServeModel(cfg)

    @asynccontextmanager
    async def lifespan(app: fastapi.FastAPI):
        logger.info("Starting ConflictNet serving — loading model...")
        try:
            serve_model.load()
            logger.info("Model loaded successfully")
        except FileNotFoundError as e:
            logger.error(str(e))
            raise
        yield

    app = fastapi.FastAPI(
        title="ConflictNet v2",
        description="Multimodal emotional conflict detection",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ──────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            model_loaded=serve_model.model is not None,
            device=str(serve_model.device),
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(
        audio: UploadFile = File(..., description="WAV audio file (16kHz mono)"),
        text: str = Form(..., description="Utterance text"),
        context_embeds: Optional[str] = Form(None, description="JSON-encoded list of past turn embeddings"),
        prosody_z: Optional[str] = Form(None, description="JSON-encoded [f0_z, energy_z, rate_z]"),
    ):
        if serve_model.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        ctx = _parse_optional_json(context_embeds)
        pz = _parse_optional_json(prosody_z)

        try:
            result = serve_model.predict(
                audio_bytes=audio_bytes,
                text=text,
                context_embeds=ctx,
                prosody_z=pz,
            )
            return PredictResponse(**result)
        except Exception as e:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict_batch", response_model=PredictBatchResponse)
    async def predict_batch(
        request: PredictBatchRequest,
    ):
        if serve_model.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        items: List[Dict[str, Any]] = []
        for item in request.items:
            items.append({
                "audio": item.audio,
                "text": item.text,
                "context_embeds": item.context_embeds,
                "prosody_z": item.prosody_z,
            })

        try:
            results = serve_model.predict_batch(items)
            return PredictBatchResponse(results=results)  # type: ignore[arg-type]
        except Exception as e:
            logger.exception("Batch prediction failed")
            raise HTTPException(status_code=500, detail=str(e))

    return app


def _parse_optional_json(raw: Optional[str]) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {raw[:100]}")
