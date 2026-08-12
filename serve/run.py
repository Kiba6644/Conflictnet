"""Entry point: run ``python -m serve.run`` to start the server."""

from __future__ import annotations

import logging
import os

import uvicorn

from serve.config import ServeConfig

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    cfg = ServeConfig.from_env()
    logger.info(f"Starting ConflictNet server on {cfg.host}:{cfg.port}")

    from serve.api import create_app
    app = create_app(cfg)

    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
