"""Quick generation sanity check for the SFT Defender (Phase 2).

Loads the base model + the trained LoRA adapter and prints
    ORIGINAL  ->  REWRITE
for N prompts drawn from the validation set, so you can eyeball whether the
SFT-only model actually learned privacy-preserving rewriting (generalize/obscure,
preserve intent, leave clean prompts unchanged) — loss alone can't tell you that.

Run (on the H200, offline; needs a GPU):
    HF_HUB_OFFLINE=1 uv run python -m thesis.training.sft.infer \
        --adapter outputs/sft-qwen35-4b/final --n 15
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from thesis.config import load_config
from thesis.utils.reproducibility import repo_root, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT Defender generation sanity check.")
    p.add_argument("--adapter", default="outputs/sft-qwen35-4b/final",
                   help="Path to the trained LoRA adapter dir (contains adapter_config.json).")
    p.add_argument("--merged", default=None,
                   help="Path to a merged standalone model dir (Fix A). If set, it is loaded "
                        "directly with AutoModelForCausalLM and --adapter/--base_model are ignored. "
                        "This is the recommended path for Qwen3.5 (avoids the adapter-prefix bug).")
    p.add_argument("--base_model", default=None,
                   help="Override base model id; defaults to the SFT config's defender.base_model.")
    p.add_argument("--val_file", default="data-public/sft/val.jsonl")
    p.add_argument("--n", type=int, default=15, help="How many validation prompts to sample.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = repo_root()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.merged is not None:
        # Fix A path: a standalone merged model. No adapter, no PeftModel, so the
        # adapter-prefix mismatch (vLLM #34186 / D-023) cannot occur.
        merged_dir = Path(args.merged)
        if not merged_dir.is_absolute():
            merged_dir = root / merged_dir
        tokenizer = AutoTokenizer.from_pretrained(merged_dir)
        model = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.bfloat16)
        model.to(device)
        model.eval()
    else:
        # Legacy path: frozen base + LoRA adapter (used by the 2507 run).
        base_model = args.base_model
        if base_model is None:
            cfg = load_config(root / "configs" / "sft" / "qwen35-4b.yaml")
            base_model = cfg.defender.base_model

        adapter_dir = Path(args.adapter)
        if not adapter_dir.is_absolute():
            adapter_dir = root / adapter_dir

        # Tokenizer: prefer the one saved with the adapter (identical, but self-contained).
        tok_src = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_model
        tokenizer = AutoTokenizer.from_pretrained(tok_src)

        # Load the frozen base the SAME way training did (no device_map), attach adapter.
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.to(device)
        model.eval()

    # Sample N rows from the held-out validation set.
    val_path = Path(args.val_file)
    if not val_path.is_absolute():
        val_path = root / val_path
    rows = [json.loads(line) for line in open(val_path, encoding="utf-8") if line.strip()]
    random.shuffle(rows)
    rows = rows[: args.n]

    for i, row in enumerate(rows, 1):
        msgs = row["messages"]
        system, user = msgs[0], msgs[1]
        reference = msgs[2]["content"]  # the teacher rewrite (for comparison only)
        meta = row.get("meta", {})

        # Build the inference prompt: system + user, then ask the model to generate.
        # enable_thinking=False keeps Qwen3.5 in non-thinking mode (direct rewrite, no
        # chain-of-thought); harmlessly ignored by templates that don't support it.
        tmpl_kwargs = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = tokenizer.apply_chat_template([system, user], enable_thinking=False, **tmpl_kwargs).to(device)
        except (TypeError, ValueError):
            enc = tokenizer.apply_chat_template([system, user], **tmpl_kwargs).to(device)
        input_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,          # greedy → deterministic, reproducible
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens (strip the prompt).
        gen = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

        print(f"\n=== {i}/{len(rows)}  [has_leak={meta.get('has_leak')}  "
              f"strategy={meta.get('strategy')}  leaked={meta.get('leaked_attributes')}] ===")
        print(f"ORIGINAL : {user['content']}")
        print(f"REWRITE  : {gen}")
        print(f"(teacher): {reference}")


if __name__ == "__main__":
    main()
