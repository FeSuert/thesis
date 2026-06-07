"""SFT warm-up training for the Defender model (Phase 2; D-021 variant, D-022 corpus).

Teaches Qwen3.5-4B-Instruct privacy-preserving paraphrasing (generalize/obscure, no masking)
on the synthetic corpus, BEFORE DPO adds the adversarial privacy bias.

Run (on the H200, from repo root):
    uv run python -m thesis.training.sft.train --config configs/sft/qwen35-4b.yaml

Smoke test first (cheap; de-risks the Gated-DeltaNet + TRL/PEFT integration — D-014):
    uv run python -m thesis.training.sft.train --config configs/sft/qwen35-4b.yaml \
        --override sft.epochs=1 sft.max_steps=20 run_name=sft-smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from thesis.config import get_path, load_config
from thesis.utils.reproducibility import git_sha, repo_root, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT warm-up for the Defender.")
    # Phase-specific YAML; merged on top of configs/base.yaml below.
    p.add_argument("--config", default="configs/sft/qwen35-4b.yaml")
    # Hydra-style overrides, e.g. sft.epochs=1 sft.lr=1e-4 run_name=sft-smoke
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def build_config(args: argparse.Namespace):
    """base.yaml  ⊕  phase YAML  ⊕  CLI overrides  → one resolved config."""
    base = load_config()  # loads configs/base.yaml + .env interpolations
    phase = OmegaConf.load(repo_root() / args.config)
    cfg = OmegaConf.merge(base, phase)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    OmegaConf.resolve(cfg)  # expand ${paths.data_dir} etc. now, before use
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    s = cfg.sft

    # 1) Reproducibility: one seed for Python/NumPy/Torch(+CUDA). Logged below.
    set_seed(cfg.seed)

    # 2) Output directory for this run (checkpoints + metadata), under $OUTPUT_DIR.
    out_dir = get_path(cfg, "output_dir") / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3) Tokenizer. The dataset is in chat ("messages") format; the tokenizer's
    #    chat template turns it into the exact token stream the model expects.
    tokenizer = AutoTokenizer.from_pretrained(cfg.defender.base_model)
    if tokenizer.pad_token is None:
        # Causal LMs often lack a pad token; reuse EOS so batching/padding works.
        tokenizer.pad_token = tokenizer.eos_token

    # 4) Data. Each row already has {"messages": [system, user, assistant], "meta": {...}}.
    #    TRL reads the "messages" column and applies the chat template automatically.
    #    Data files live in data-public/ (committed); resolve them relative to the
    #    repo root so the run works regardless of the current working directory.
    def _resolve(p: str) -> str:
        path = Path(p)
        return str(path if path.is_absolute() else repo_root() / path)

    data = load_dataset(
        "json",
        data_files={
            "train": _resolve(s.train_file),
            "validation": _resolve(s.val_file),
        },
    )

    # 5) LoRA: train small low-rank adapters instead of all 4B weights.
    #    ~0.5% of params updated → fast, low VRAM, and matches the "ship a small
    #    on-device model" goal (adapters merge into the base at export time).
    peft_config = None
    if s.use_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=s.lora_r,
            lora_alpha=s.lora_alpha,
            lora_dropout=s.lora_dropout,
            target_modules=list(s.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )

    # 6) Training arguments (TRL's SFTConfig = HF TrainingArguments + SFT extras).
    from trl import SFTConfig, SFTTrainer

    sft_args = SFTConfig(
        output_dir=str(out_dir),
        seed=cfg.seed,
        # --- data shaping ---
        max_length=s.max_seq_len,        # truncate sequences longer than this
        packing=s.packing,               # False → one example per sequence (exact masking)
        # Train the loss ONLY on assistant tokens. The system prompt is identical on
        # every row and the user turn is given context — learning to "predict" them
        # would waste capacity and bias the model. Requires a recent TRL whose chat
        # template marks the assistant span; the smoke test confirms it works.
        assistant_only_loss=s.assistant_only_loss,
        # --- optimization ---
        num_train_epochs=s.epochs,
        learning_rate=float(s.lr),
        lr_scheduler_type=s.lr_scheduler,
        warmup_ratio=s.warmup_ratio,
        weight_decay=s.weight_decay,
        per_device_train_batch_size=s.per_device_batch_size,
        per_device_eval_batch_size=s.per_device_batch_size,
        gradient_accumulation_steps=s.grad_accum,
        max_grad_norm=s.max_grad_norm,
        bf16=s.bf16,                     # H100/H200 bfloat16; stable, no loss scaling
        gradient_checkpointing=s.gradient_checkpointing,  # trade compute for VRAM
        # --- eval / logging / checkpoints ---
        eval_strategy=s.eval_strategy,
        eval_steps=s.eval_steps,
        logging_steps=s.logging_steps,
        save_steps=s.save_steps,
        save_total_limit=s.save_total_limit,
        load_best_model_at_end=s.load_best_model_at_end,
        metric_for_best_model=s.metric_for_best_model,
        # --- experiment tracking (W&B) ---
        report_to=["wandb"] if cfg.tracking.mode != "disabled" else [],
        run_name=cfg.run_name,
    )
    # Optional cap honored only if passed via --override sft.max_steps=N (smoke test).
    if "max_steps" in s:
        sft_args.max_steps = int(s.max_steps)

    # 7) Reproducibility metadata frozen next to the checkpoints (docs/reproducibility.md).
    meta = {
        "run_name": cfg.run_name,
        "git_sha": git_sha(short=False),
        "seed": cfg.seed,
        "base_model": cfg.defender.base_model,
        "dataset_version": s.dataset_version,
        "train_file": str(s.train_file),
        "val_file": str(s.val_file),
        "n_train": data["train"].num_rows,
        "n_val": data["validation"].num_rows,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # 8) Trainer: ties model + data + args + LoRA together and runs the loop.
    trainer = SFTTrainer(
        model=cfg.defender.base_model,   # string → TRL loads it (bf16 from SFTConfig)
        args=sft_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,      # newer TRL name for the tokenizer
        peft_config=peft_config,
    )

    trainer.train()                      # the actual optimization loop

    # 9) Persist the final (best, if load_best_model_at_end) adapter + tokenizer.
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    print(f"[sft] done. artifacts in {out_dir}")


if __name__ == "__main__":
    main()
