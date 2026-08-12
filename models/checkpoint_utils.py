"""Safe checkpoint loading utilities.

Centralises all checkpoint I/O so that:
  1. ``.safetensors`` files are preferred (pickle-free, CWE-502 safe).
  2. Legacy ``.pt`` / ``.pth`` files are loaded with ``weights_only=True``
     to prevent arbitrary code execution via pickle.
  3. A single call-site makes auditing straightforward.

Usage::

    from models.checkpoint_utils import load_checkpoint_state

    state = load_checkpoint_state("checkpoints/best_model.pt", device="cpu")
    model.load_state_dict(state["model_state_dict"], strict=False)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Union

import torch

logger = logging.getLogger(__name__)


def load_checkpoint_state(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",  # type: ignore[arg-type]
) -> Dict[str, Any]:
    """Load checkpoint weights from *path* safely.

    Supports two formats:

    * ``.safetensors`` — loaded via ``safetensors.torch.load_file``
      (no pickle, no arbitrary code execution).
    * ``.pt`` / ``.pth`` / other — loaded via ``torch.load`` with
      ``weights_only=True`` (restricts unpickling to tensor data only,
      blocking arbitrary object instantiation).

    For legacy ``.pt`` files the returned dict is the raw checkpoint.
    For ``.safetensors`` the returned dict is the flat state-dict (no
    ``model_state_dict`` wrapper), equivalent to
    ``torch.load(...)["model_state_dict"]``.

    Args:
        path: Filesystem path to the checkpoint file.
        device: Target device (``"cpu"`` / ``"cuda"`` / ``torch.device``).

    Returns:
        A dictionary of tensors (state-dict or full checkpoint dict).

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If the file cannot be deserialised safely.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device_str = str(device)

    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file as st_load

        logger.debug("Loading safetensors checkpoint: %s", checkpoint_path)
        return st_load(str(checkpoint_path), device=device_str)

    # Legacy .pt / .pth — weights_only=True prevents arbitrary pickle execution
    logger.debug("Loading legacy .pt checkpoint: %s (weights_only=True)", checkpoint_path)
    return torch.load(  # noqa: S301 — weights_only=True prevents arbitrary code execution
        str(checkpoint_path),
        map_location=device_str,
        weights_only=True,
    )


def extract_model_state(
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract the model state-dict from a checkpoint.

    Handles both flat state-dicts (e.g. from safetensors) and wrapped
    checkpoints containing a ``model_state_dict`` key.
    """
    return checkpoint.get("model_state_dict", checkpoint)
