"""Smoke tests that must pass in any environment (Mac CPU or H200)."""

from __future__ import annotations

import thesis


def test_package_imports() -> None:
    assert thesis.__version__


def test_reproducibility_helpers() -> None:
    from thesis.utils.reproducibility import get_device, set_seed

    set_seed(42)
    assert get_device() in {"cuda", "mps", "cpu"}


def test_config_loads() -> None:
    from thesis.config import load_config

    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.dpo.beta == 0.1
    assert "data_dir" in cfg.paths
