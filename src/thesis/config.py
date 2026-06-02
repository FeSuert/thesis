"""Centralized configuration loading.

Environment variables are loaded from `.env` if present.
Config files are expected under `configs/` as YAML.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config, resolve env var interpolations, apply CLI overrides.

    Args:
        path: Path to a YAML config. Defaults to configs/base.yaml.
        overrides: List of Hydra-style overrides like ["seed=123", "defender.tier=7b"].

    Returns:
        Resolved OmegaConf DictConfig.
    """
    load_dotenv(REPO_ROOT / ".env", override=False)

    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    OmegaConf.resolve(cfg)
    return cfg  # type: ignore[return-value]


def get_path(cfg: DictConfig, key: str) -> Path:
    """Get a path from config, expanding and ensuring it exists as a directory."""
    raw = cfg.paths[key]
    p = Path(os.path.expandvars(raw)).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p
