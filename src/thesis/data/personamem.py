"""PersonaMem-v2 loader (D-003).

PersonaMem-v2 directory layout on Hugging Face (`bowen-upenn/PersonaMem-v2`):

    data/raw_data/raw_data_{ts}_persona{N}.json        # persona metadata
    data/chat_history_32k/chat_history_{ts}_persona{N}.json
    data/chat_history_128k/chat_history_{ts}_persona{N}.json
    data/chat_history_multimodal_{32k|128k}/...        # not used (text-only for us)
    benchmark/text/{train,val,benchmark}.csv           # query/answer benchmarks

Each persona file is a dict with a single string key (persona index as string)
wrapping the actual persona. Chat history files contain `metadata` and
`chat_history` (list of {role, content} turns; the first turn is a system prompt
that includes the full persona JSON — strip it for attacker inputs).

Ground truth attributes (`A_user`) per D-012 are six:
    loc, prof, age, sex, status, income

Income is represented as a **categorical socioeconomic class label** (e.g.,
"Upper middle class") drawn from `demographics.socioeconomic_status`, not as a
numeric value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

REPO_ID = "bowen-upenn/PersonaMem-v2"
DEFAULT_BUCKET: Literal["32k", "128k"] = "32k"


# -------------------------------------------------------------------- Persona


@dataclass
class AUser:
    """Ground-truth attribute set `A_user` (D-012: six attributes).

    Each field is a short, canonical string description of the attribute,
    suitable for use as a label for Attacker training/evaluation.
    """

    loc: str  # coarse location (city + country typically)
    prof: str  # occupation / job title
    age: int  # integer age
    sex: str  # gender label as stated in the persona (v2 uses "Female"/"Male"/...)
    status: str  # marital/family status label
    income: str  # qualitative socioeconomic class (e.g., "Upper middle class")


@dataclass
class Persona:
    """Wrapper over a raw PersonaMem-v2 persona dict.

    Field access is via the underlying `raw` dict (22 keys at the top level).
    Convenience properties extract `A_user` attributes with a documented policy.
    """

    persona_id: str
    raw: dict[str, Any] = field(repr=False)

    # ---- A_user extraction ---------------------------------------------------

    @property
    def age(self) -> int:
        return int(self.raw["age"])

    @property
    def sex(self) -> str:
        return str(self.raw["gender"])

    @property
    def loc(self) -> str:
        """Coarse location: city + country, prefer work_location; fall back to
        address city + nationality if work_location is absent."""
        occ = self.raw.get("occupation", {})
        city = occ.get("work_location") or ""
        nationality = self.raw.get("nationality", "") or ""
        if city:
            return f"{city} ({nationality})" if nationality else city
        sens = self.raw.get("sensitive_information", {})
        return sens.get("physical_address", "") or nationality

    @property
    def prof(self) -> str:
        occ = self.raw.get("occupation", {})
        title = str(occ.get("job_title", ""))
        industry = str(occ.get("industry", ""))
        return f"{title} ({industry})" if industry else title

    @property
    def status(self) -> str:
        demo = self.raw.get("demographics", {})
        return str(demo.get("marital_status", ""))

    @property
    def income(self) -> str:
        """Qualitative socioeconomic class label (D-012).

        v2 provides `demographics.socioeconomic_status` as a short string like
        "Upper middle class". Treated as a categorical attribute, not numeric.
        """
        demo = self.raw.get("demographics", {})
        return str(demo.get("socioeconomic_status", ""))

    @property
    def a_user(self) -> AUser:
        return AUser(
            loc=self.loc,
            prof=self.prof,
            age=self.age,
            sex=self.sex,
            status=self.status,
            income=self.income,
        )

    # ---- Curated conversation snippets inside the persona file --------------

    @property
    def conversation_topics(self) -> list[str]:
        return list(self.raw.get("conversations", {}).keys())

    def conversation_snippets(self, topic: str) -> list[dict[str, Any]]:
        """Return the curated (preference, turns) snippets for a topic.

        Each snippet has: preference, pref_type, who, conversations (turns),
        user_query, correct_answer, incorrect_answers, topic_query, ..."""
        return list(self.raw.get("conversations", {}).get(topic, []))


# --------------------------------------------------------------- Chat history


@dataclass
class ChatHistory:
    """A persona's assembled long-context chat history (32k or 128k bucket)."""

    persona_id: int
    turns: list[dict[str, str]] = field(repr=False)
    metadata: dict[str, Any] = field(repr=False)

    def __len__(self) -> int:
        return len(self.turns)

    @property
    def token_count(self) -> int:
        return int(self.metadata.get("final_token_count", 0))

    def user_turns(self) -> list[str]:
        """User-side content only. This is what the Attacker sees."""
        return [t["content"] for t in self.turns if t.get("role") == "user"]

    def attacker_context(self, up_to_turn: int | None = None) -> list[dict[str, str]]:
        """Return only user+assistant turns, stripping the system prompt.

        The system prompt in PersonaMem-v2 contains the full persona JSON;
        an Attacker operating as a third-party observer must never see it.

        Args:
            up_to_turn: if given, only return turns with index < this value.
                Useful for simulating `C_{t-1}` in Algorithm 1.
        """
        filtered = [t for t in self.turns if t.get("role") != "system"]
        if up_to_turn is not None:
            filtered = filtered[:up_to_turn]
        return filtered


# -------------------------------------------------------------------- Loader


_PERSONA_FILENAME_TS = "250815_143643"
_CHAT32K_FILENAME_TS = "250911_213118"
_CHAT128K_FILENAME_TS = "250911_213509"


def persona_filename(persona_index: int) -> str:
    return f"data/raw_data/raw_data_{_PERSONA_FILENAME_TS}_persona{persona_index}.json"


def chat_history_filename(
    persona_index: int, bucket: Literal["32k", "128k"] = DEFAULT_BUCKET
) -> str:
    ts = _CHAT32K_FILENAME_TS if bucket == "32k" else _CHAT128K_FILENAME_TS
    return f"data/chat_history_{bucket}/chat_history_{ts}_persona{persona_index}.json"


def load_persona_from_path(path: str | Path) -> Persona:
    """Load a persona from a local JSON file. Unwraps the string-keyed outer dict."""
    path = Path(path)
    outer = json.loads(path.read_text())
    if len(outer) != 1:
        raise ValueError(f"Expected exactly one top-level key in {path}, got {list(outer)}")
    persona_id, inner = next(iter(outer.items()))
    return Persona(persona_id=persona_id, raw=inner)


def load_chat_history_from_path(path: str | Path) -> ChatHistory:
    path = Path(path)
    data = json.loads(path.read_text())
    meta = data.get("metadata", {})
    return ChatHistory(
        persona_id=int(meta.get("persona_id", -1)),
        turns=list(data.get("chat_history", [])),
        metadata=meta,
    )


def download_persona(persona_index: int, cache_dir: str | Path) -> Path:
    """Download a single persona JSON from HF to `cache_dir` and return the path.

    Kept thin — for bulk operations prefer `huggingface_hub.snapshot_download`
    with explicit `allow_patterns`.
    """
    from huggingface_hub import hf_hub_download  # lazy import

    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=persona_filename(persona_index),
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
    )


def download_chat_history(
    persona_index: int,
    cache_dir: str | Path,
    bucket: Literal["32k", "128k"] = DEFAULT_BUCKET,
) -> Path:
    from huggingface_hub import hf_hub_download  # lazy import

    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=chat_history_filename(persona_index, bucket),
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
    )


def load_pair(
    persona_index: int,
    cache_dir: str | Path,
    bucket: Literal["32k", "128k"] = DEFAULT_BUCKET,
) -> tuple[Persona, ChatHistory]:
    """Convenience: download (if needed) and load a persona + chat-history pair."""
    p_path = download_persona(persona_index, cache_dir)
    ch_path = download_chat_history(persona_index, cache_dir, bucket)
    return load_persona_from_path(p_path), load_chat_history_from_path(ch_path)
