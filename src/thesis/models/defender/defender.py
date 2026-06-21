"""Defender wrapper (D-001 memoryless; D-024 Qwen3.5-4B merged SFT model).

Generates k candidate rewrites for a single user message. The Defender is
**memoryless** (D-001): it sees only the current user turn — never the chat
history. (History is used solely by the Attacker to compute the contextual P_att
risk signal during preference generation.)

The system prompt is the exact canonical instruction the model was SFT'd on, so
generation matches training.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Exact canonical Defender instruction from the SFT corpus (data-public/sft/*.jsonl).
DEFENDER_SYSTEM = (
    "You are a privacy-preserving rewriting assistant. Rewrite the user's message so it can be "
    "safely sent to an external AI chatbot without revealing the user's personal identity. Replace "
    "or remove specific personal details \u2014 location, profession, age, sex/gender, relationship "
    "or family status, and income/socioeconomic status \u2014 with more general expressions, while "
    "fully preserving the original meaning, intent, and natural fluency. Generalize details that "
    "are needed to answer the request (e.g. \"nurse in Boston\" \u2192 \"healthcare worker\"); drop "
    "details that are irrelevant to the request. Never insert false information, never leave blanks, "
    "placeholders, or bracketed tags, and never add explanations. If the message contains no "
    "identifying details, return it essentially unchanged. Output only the rewritten message."
)


class Defender:
    def __init__(self, model_dir: str, device: str | None = None, max_new_tokens: int = 256) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _inputs(self, user_text: str):
        messages = [
            {"role": "system", "content": DEFENDER_SYSTEM},
            {"role": "user", "content": user_text},
        ]
        kwargs = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except (TypeError, ValueError):
            enc = self.tokenizer.apply_chat_template(messages, **kwargs)
        return enc.to(self.device)

    @torch.no_grad()
    def candidates(self, user_text: str, k: int = 4, temperature: float = 0.8,
                   top_p: float = 0.95) -> list[str]:
        """Sample k diverse rewrites of `user_text` (do_sample for diversity)."""
        enc = self._inputs(user_text)
        input_len = enc["input_ids"].shape[1]
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=k,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        return [self.tokenizer.decode(o[input_len:], skip_special_tokens=True).strip() for o in out]

    @torch.no_grad()
    def rewrite(self, user_text: str) -> str:
        """Deterministic (greedy) single rewrite — used at evaluation time."""
        enc = self._inputs(user_text)
        input_len = enc["input_ids"].shape[1]
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
