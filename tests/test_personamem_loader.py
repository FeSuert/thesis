"""Integration-ish test: exercises the PersonaMem-v2 loader against a real file.

This test is conditionally skipped if the local inspection cache is absent.
It does NOT download anything; to populate the cache see `docs/remote-h200.md`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thesis.data.personamem import (
    load_chat_history_from_path,
    load_persona_from_path,
)

# Use the local inspection cache populated during Phase 1 schema exploration.
CACHE = Path(__file__).resolve().parents[1] / ".hf_inspect"


def _find(glob: str) -> Path | None:
    hits = list(CACHE.rglob(glob))
    return hits[0] if hits else None


@pytest.fixture(scope="module")
def persona0_path() -> Path:
    p = _find("raw_data_250815_143643_persona0.json")
    if p is None:
        pytest.skip("Local PersonaMem-v2 inspection cache not populated.")
    return p


@pytest.fixture(scope="module")
def chat0_path() -> Path:
    p = _find("chat_history_250911_213118_persona0.json")
    if p is None:
        pytest.skip("Local PersonaMem-v2 inspection cache not populated.")
    return p


def test_persona_loads(persona0_path: Path) -> None:
    persona = load_persona_from_path(persona0_path)
    assert persona.persona_id == "0"
    assert persona.raw["name"] == "Amara Nwosu"


def test_a_user_extraction(persona0_path: Path) -> None:
    persona = load_persona_from_path(persona0_path)
    a = persona.a_user
    assert a.age == 28
    assert a.sex == "Female"
    assert "Washington" in a.loc  # work_location was "Washington, D.C."
    assert "STEM" in a.prof or "Educator" in a.prof.lower() or "Designer" in a.prof
    assert a.status  # non-empty marital_status
    assert "class" in a.income.lower()  # "Upper middle class" — qualitative socioeconomic label


def test_chat_history_loads(chat0_path: Path) -> None:
    ch = load_chat_history_from_path(chat0_path)
    assert ch.persona_id == 0
    assert len(ch) == ch.metadata["total_messages"]
    assert ch.token_count > 0


def test_attacker_context_strips_system(chat0_path: Path) -> None:
    ch = load_chat_history_from_path(chat0_path)
    ctx = ch.attacker_context()
    assert all(t["role"] != "system" for t in ctx), "System prompt leaked into attacker context"
    # chat_history has 1 system + 236 user/assistant turns for persona0
    assert len(ctx) == len(ch) - 1
