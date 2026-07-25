"""Thin OpenAI-compatible chat client used by the judge and API-attacker scripts.

The RQ3 utility judge uses the **OpenAI API directly** (base_url https://api.openai.com/v1,
model gpt-5, key from the OPENAI_API_KEY token). The API attacker uses Moonshot/Kimi
(key from MOONSHOT_API_KEY). Any OpenAI-compatible endpoint works by switching --base-url
and --model. The API key is never stored in the repo: it is read from --api-key, the
OPENAI_API_KEY / MOONSHOT_API_KEY env vars, or an interactive prompt, in that order.
"""

from __future__ import annotations

import getpass
import os
import time


def resolve_api_key(cli_key: str | None, env_vars: tuple[str, ...] = ("OPENAI_API_KEY", "MOONSHOT_API_KEY")) -> str:
    if cli_key:
        return cli_key
    for var in env_vars:
        val = os.environ.get(var)
        if val:
            return val
    return getpass.getpass("API key (input hidden): ").strip()


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: str,
                 max_retries: int = 5, timeout: float = 120.0) -> None:
        from openai import OpenAI

        self.model = model
        self.max_retries = max_retries
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        m = (model or "").lower()
        # OpenAI GPT-5 / o-series reasoning: require max_completion_tokens (not max_tokens),
        # only default temperature, and spend hidden reasoning tokens against the budget.
        self._openai_reasoning = m.startswith(("gpt-5", "o1", "o3", "o4"))
        # Other providers' reasoning models (e.g. Moonshot Kimi K3) also reject a non-default
        # temperature but keep the standard max_tokens field.
        self._kimi_reasoning = "k3" in m
        self._reasoning = self._openai_reasoning or self._kimi_reasoning

    def chat(self, messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 512) -> str:
        """Single chat completion with exponential-backoff retries.

        Adapts parameters to the model family (GPT-5/o-series, Kimi K3, or standard chat).
        Non-retryable client errors (bad params, auth) fail fast instead of retrying.
        """
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict = {"model": self.model, "messages": messages}
                # Reasoning models spend hidden tokens against the cap → give headroom.
                budget = max(max_tokens, 2048) if self._reasoning else max_tokens
                if self._openai_reasoning:
                    kwargs["max_completion_tokens"] = budget
                    kwargs["reasoning_effort"] = "minimal"
                else:
                    kwargs["max_tokens"] = budget
                # Only send a custom temperature for models that permit one.
                if not self._reasoning:
                    kwargs["temperature"] = temperature
                resp = self.client.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                status = getattr(e, "status_code", None)
                # 4xx (except 408 timeout / 429 rate-limit) are permanent → fail fast.
                if status is not None and 400 <= status < 500 and status not in (408, 429):
                    raise RuntimeError(f"Non-retryable API error ({status}): {e}") from e
                wait = 2 ** attempt
                print(f"    API error (attempt {attempt + 1}/{self.max_retries}): {e} — retry in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"API call failed after {self.max_retries} retries: {last_err}")
