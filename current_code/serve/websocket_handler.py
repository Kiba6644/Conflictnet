"""WebSocket endpoint for real-time streaming dialogue inference.

Maintains per-connection dialogue context across turns.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from serve.config import ServeConfig
from serve.model import ServeModel

logger = logging.getLogger(__name__)


class StreamingSession:
    """Per-connection session that accumulates dialogue context."""

    def __init__(self, session_id: str, embed_dim: int, max_turns: int = 16):
        self.session_id = session_id
        self.embed_dim = embed_dim
        self.max_turns = max_turns
        self.context_embeds: List[List[float]] = []

    def update_context(self, fused_embed: List[float]) -> None:
        self.context_embeds.append(fused_embed)
        if len(self.context_embeds) > self.max_turns:
            self.context_embeds = self.context_embeds[-self.max_turns:]


class WebSocketHandler:
    """Manages WebSocket connections with per-session context tracking."""

    def __init__(self, model: ServeModel, cfg: ServeConfig):
        self.model = model
        self.sessions: Dict[str, StreamingSession] = {}

    async def handle(self, websocket: WebSocket) -> None:
        await websocket.accept()
        session_id = f"ws_{id(websocket)}"
        session = StreamingSession(
            session_id=session_id,
            embed_dim=self.model.cfg.embed_dim,
            max_turns=self.model.cfg.temporal_max_turns,
        )
        self.sessions[session_id] = session

        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)

                action = msg.get("action", "predict")
                if action == "reset":
                    session.context_embeds = []
                    await websocket.send_json({"action": "reset", "status": "ok"})
                    continue

                audio_bytes: Optional[bytes] = None
                audio_b64 = msg.get("audio")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)

                text = msg.get("text", "")

                if not audio_bytes or not text:
                    await websocket.send_json({
                        "error": "Both 'audio' (base64) and 'text' fields are required",
                    })
                    continue

                result = self.model.predict(
                    audio_bytes=audio_bytes,
                    text=text,
                    context_embeds=session.context_embeds if session.context_embeds else None,
                    prosody_z=msg.get("prosody_z"),
                )

                session.update_context(result["fused_embed"])

                await websocket.send_json(result)

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {session_id}")
        except Exception as e:
            logger.exception(f"WebSocket error [{session_id}]")
            try:
                await websocket.send_json({"error": str(e)})
            except Exception:
                pass
        finally:
            self.sessions.pop(session_id, None)
