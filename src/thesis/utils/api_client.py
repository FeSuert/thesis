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

    def chat(self, messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 512) -> str:
        """Single chat completion with exponential-backoff retries."""
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 — retry any transient API error
                last_err = e
                wait = 2 ** attempt
                print(f"    API error (attempt {attempt + 1}/{self.max_retries}): {e} — retry in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"API call failed after {self.max_retries} retries: {last_err}")
