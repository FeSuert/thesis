"""RQ3 step 1 — generate LLM responses for original vs. rewritten prompts.

Reads the per-turn log of an eval-v2 run (turns_<variant>.jsonl), sends both the
original and the rewritten prompt to the same local responder model, and saves the
response pairs. Step 2 (rq3_judge.py) then scores response similarity and runs the
LLM-as-judge comparison — that part needs internet, so it runs on the login node.

Responses are generated greedily so the comparison reflects the prompt difference,
not sampling noise. Each turn is answered independently (no chat history), matching
the memoryless setting the Defender operates in.

Run (fat_gpu; responder needs one GPU):
  HF_HOME=/lustre/urdeniw5/hf_cache HF_HUB_OFFLINE=1 \
    uv run python -m thesis.evaluation.rq3_responses \
      --turns outputs/eval/v2-qwen/turns_sftdpo_v2.jsonl \
      --out outputs/eval/rq3/responses_sftdpo_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from thesis.utils.reproducibility import repo_root, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ3: generate responses for original vs rewrite.")
    p.add_argument("--turns", required=True,
                   help="Per-turn JSONL from run_eval_v2 (turns_<variant>.jsonl).")
    p.add_argument("--responder", default="Qwen/Qwen3.5-9B",
                   help="Local model that answers the prompts (stands in for the chatbot).")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=0, help="Cap turns (0 = all).")
    p.add_argument("--changed-only", action="store_true",
                   help="Only answer turns where the rewrite differs from the original. "
                        "Unchanged turns get identical responses by construction.")
    p.add_argument("--out", default="outputs/eval/rq3/responses.jsonl")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class Responder:
    def __init__(self, model_name: str, max_new_tokens: int) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def respond(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except (TypeError, ValueError):
            enc = self.tokenizer.apply_chat_template(messages, **kwargs)
        enc = enc.to(self.device)
        input_len = enc["input_ids"].shape[1]
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = repo_root()

    turns_path = Path(args.turns) if Path(args.turns).is_absolute() else root / args.turns
    out_path = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    turns = [json.loads(line) for line in turns_path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if args.changed_only:
        turns = [t for t in turns if t.get("changed")]
    if args.limit > 0:
        turns = turns[: args.limit]
    print(f"[rq3] {len(turns)} turns from {turns_path.name}; responder={args.responder}")

    # Resume support: skip turns already answered in a previous partial run.
    done: set[tuple[int, int]] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["persona_id"], r["turn_idx"]))
        print(f"[rq3] resuming: {len(done)} turns already answered")

    responder = Responder(args.responder, args.max_new_tokens)

    with out_path.open("a", encoding="utf-8") as f:
        for i, t in enumerate(turns, 1):
            key = (t["persona_id"], t["turn_idx"])
            if key in done:
                continue
            resp_orig = responder.respond(t["original"])
            # Identical texts get identical greedy responses — skip the second call.
            resp_rw = resp_orig if t["rewrite"].strip() == t["original"].strip() \
                else responder.respond(t["rewrite"])
            f.write(json.dumps({
                "persona_id": t["persona_id"], "turn_idx": t["turn_idx"],
                "original": t["original"], "rewrite": t["rewrite"],
                "changed": t.get("changed"), "s_sem_prompt": t.get("s_sem"),
                "response_original": resp_orig, "response_rewrite": resp_rw,
            }, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(turns):
                print(f"    {i}/{len(turns)} turns answered")

    print(f"[rq3] responses -> {out_path}")
    print("[rq3] next: run rq3_judge.py on the login node (needs internet for the judge API).")


if __name__ == "__main__":
    main()
