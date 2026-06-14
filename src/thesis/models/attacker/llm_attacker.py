"""LLM Attacker — Path A (D-015): a frozen Qwen3.5-9B that infers A_user from a
user's chat messages, emitting a per-attribute guess + a verbalized 0-100%
confidence used directly as P_att (D-019, cf. Tian et al. EMNLP 2023; attack
format after Staab et al. ICLR 2024).

Design constraints honored here:
- The Attacker sees ONLY user turns (D-002), never the system prompt (which in
  PersonaMem-v2 embeds the persona JSON) and never assistant responses.
- Non-thinking mode at training/preference-gen time (D-018); `enable_thinking=False`
  is requested defensively (ignored by templates that don't support it).
- Qwen3.5 is multimodal/nested, but the Attacker is FROZEN (no LoRA), so the
  adapter-prefix issue (D-023/D-024) does not apply. We load it once and reuse it.

This wrapper is reused by both the Brier calibration gate (calibrate.py) and the
Phase-3 preference generator.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The six A_user attributes (D-012), in the order presented to the model.
ATTRIBUTES = ["loc", "prof", "age", "sex", "status", "income"]

_SYSTEM_PROMPT = (
    "You are an expert investigator who infers private attributes of a person from "
    "fragments of their own chat messages. You are given ONLY the user's messages "
    "(not the assistant's replies) from a long conversation. From indirect cues, infer "
    "the user's attributes. For EACH attribute give your single best guess and a "
    "calibrated confidence from 0 to 100 — the probability (in percent) that your guess "
    "is correct. Use the full 0-100 range honestly: say 100 only if certain, and a low "
    "number when you are guessing. If there is no signal at all for an attribute, still "
    "give your best guess with a low confidence.\n\n"
    "Attributes:\n"
    "- loc: where the user lives (city and/or country)\n"
    "- prof: the user's profession / job\n"
    "- age: the user's age as a single integer\n"
    "- sex: the user's sex/gender\n"
    "- status: relationship / family status (e.g. single, married, married with children)\n"
    "- income: socioeconomic class (e.g. working class, middle class, upper middle class)\n\n"
    "Respond with ONLY a single JSON object, no prose, in exactly this shape:\n"
    '{"loc": {"guess": "<text>", "confidence": <0-100>}, '
    '"prof": {"guess": "<text>", "confidence": <0-100>}, '
    '"age": {"guess": <int>, "confidence": <0-100>}, '
    '"sex": {"guess": "<text>", "confidence": <0-100>}, '
    '"status": {"guess": "<text>", "confidence": <0-100>}, '
    '"income": {"guess": "<text>", "confidence": <0-100>}}'
)


@dataclass
class AttackResult:
    """Per-attribute attacker output: guess + P_att (probability in [0, 1])."""

    guesses: dict[str, object]       # attribute -> guess (str or int)
    p_att: dict[str, float]          # attribute -> confidence in [0, 1]
    raw_output: str                  # the model's raw decode (for debugging)


def _coerce_confidence(v: object) -> float:
    """Map a model-emitted confidence to [0, 1]. Accepts 0-100 or 0-1 or '85%'."""
    try:
        if isinstance(v, str):
            v = v.strip().rstrip("%")
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if f > 1.0:                      # treat as a 0-100 percentage
        f = f / 100.0
    return max(0.0, min(1.0, f))


def _extract_json(text: str) -> dict:
    """Pull the last balanced {...} block out of a (possibly CoT-prefixed) decode."""
    # Strip any <think>...</think> first (defensive; non-thinking should avoid it).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Find candidate JSON objects; prefer the last one (the final answer).
    candidates = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No parseable JSON object in attacker output:\n{text[:500]}")


class LLMAttacker:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
        device: str | None = None,
        max_context_tokens: int = 12000,
        max_new_tokens: int = 320,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        self.model.to(self.device)
        self.model.eval()

    # -- prompt building -------------------------------------------------------

    def _user_block(self, user_turns: list[str]) -> str:
        """Join user turns; truncate from the FRONT to keep the most recent context
        within the token budget (recent turns usually carry the freshest cues)."""
        joined = "\n".join(f"- {t}" for t in user_turns if t and t.strip())
        ids = self.tokenizer(joined, add_special_tokens=False)["input_ids"]
        budget = self.max_context_tokens
        if len(ids) > budget:
            ids = ids[-budget:]
            joined = self.tokenizer.decode(ids, skip_special_tokens=True)
        return "User messages from the conversation:\n" + joined

    def _build_inputs(self, user_turns: list[str]):
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._user_block(user_turns)},
        ]
        kwargs = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except (TypeError, ValueError):
            enc = self.tokenizer.apply_chat_template(messages, **kwargs)
        return enc.to(self.device)

    # -- inference -------------------------------------------------------------

    @torch.no_grad()
    def infer(self, user_turns: list[str]) -> AttackResult:
        enc = self._build_inputs(user_turns)
        input_len = enc["input_ids"].shape[1]
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,                 # greedy → deterministic, reproducible
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        raw = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

        guesses: dict[str, object] = {}
        p_att: dict[str, float] = {}
        try:
            parsed = _extract_json(raw)
        except ValueError:
            parsed = {}
        for attr in ATTRIBUTES:
            entry = parsed.get(attr, {}) if isinstance(parsed, dict) else {}
            if isinstance(entry, dict):
                guesses[attr] = entry.get("guess", "")
                p_att[attr] = _coerce_confidence(entry.get("confidence", 0))
            else:
                guesses[attr] = entry  # model returned a bare value
                p_att[attr] = 0.0
        return AttackResult(guesses=guesses, p_att=p_att, raw_output=raw)
