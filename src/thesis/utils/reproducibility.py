"""Reproducibility helpers: seed management, git SHA, device selection."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Set RNG seeds for Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_sha(short: bool = True) -> str | None:
    """Return the current git SHA if available."""
    try:
        cmd = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]
