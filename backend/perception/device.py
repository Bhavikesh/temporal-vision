"""
perception/device.py
--------------------
Select the best available compute device without hard-coding CUDA paths
or making NVIDIA-only assumptions.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def select_device(preference: str = "auto") -> str:
    """
    Return a torch device string.

    Priority:  CUDA  >  MPS  >  CPU
    The caller may override with preference="cuda" | "mps" | "cpu".
    """
    import torch

    if preference != "auto":
        p = preference.lower()
        if p == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available; falling back to auto-select.")
        elif p == "mps" and not torch.backends.mps.is_available():
            log.warning("MPS requested but not available; falling back to auto-select.")
        else:
            log.info("Using requested device: %s", p)
            return p

    # Auto-select
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    log.info("Auto-selected device: %s", device)
    return device
