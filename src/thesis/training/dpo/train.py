"""DPO training for the Defender (Phase 3; D-021 variants 3 & 4).

Trains a LoRA adapter with TRL's DPOTrainer on the preference pairs produced by
`thesis.training.preference_gen.generate`, then merges it into a standalone model
(Fix A / D-024) so inference has no adapter-prefix issue.

Variants (same config, different base):
  SFT+DPO  (deployed):  base = outputs/sft-qwen3_5-4b/merged
  DPO-only (ablation):  --override defender.base_model=Qwen/Qwen3.5-4B run_name=dpo-only-qwen3_5-4b

The preference data is conversational {prompt, chosen, rejected}; DPOTrainer reads those
columns directly. With a LoRA peft_config and ref_model=None, the reference policy is the
base model with the adapter disabled (standard TRL behavior).

Run (H100 fat_gpu):
  HF_HOME=/lustre/urdeniw5/hf_cache HF_HUB_OFFLINE=1 WANDB_MODE=offline \
    uv run python -m thesis.training.dpo.train --config configs/dpo/qwen3_5-4b.yaml
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
    p = argparse.ArgumentParser(description="DPO training for the Defender.")
    p.add_argument("--config", default="configs/dpo/qwen3_5-4b.yaml")
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def build_config(args: argparse.Namespace):
    base = load_config()
    phase = OmegaConf.load(repo_root() / args.config)
    cfg = OmegaConf.merge(base, phase)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    OmegaConf.resolve(cfg)
    return cfg


def _resolve(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else repo_root() / path)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    d = cfg.dpo

    set_seed(cfg.seed)
    out_dir = get_path(cfg, "output_dir") / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # The base model may be a HF id (DPO-only) or a local merged dir (SFT+DPO).
    base_model = cfg.defender.base_model
    base_local = Path(_resolve(base_model)) if not str(base_model).startswith("Qwen/") else base_model
    tok_src = str(base_local) if isinstance(base_local, Path) and base_local.exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Preference data: keep only the columns DPOTrainer needs (drop "meta").
    ds = load_dataset("json", data_files=_resolve(d.pref_file))["train"]
    keep = {"prompt", "chosen", "rejected"}
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    split = ds.train_test_split(test_size=float(d.val_fraction), seed=cfg.seed)
    print(f"[dpo] {ds.num_rows} pairs -> train {split['train'].num_rows} / val {split['test'].num_rows}")

    peft_config = None
    if d.use_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=d.lora_r,
            lora_alpha=d.lora_alpha,
            lora_dropout=d.lora_dropout,
            target_modules=list(d.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )

    from trl import DPOConfig, DPOTrainer

    dpo_args = DPOConfig(
        output_dir=str(out_dir),
        seed=cfg.seed,
        beta=float(d.beta),
        max_length=d.max_length,
        num_train_epochs=d.epochs,
        learning_rate=float(d.lr),
        lr_scheduler_type=d.lr_scheduler,
        warmup_ratio=d.warmup_ratio,
        weight_decay=d.weight_decay,
        per_device_train_batch_size=d.per_device_batch_size,
        per_device_eval_batch_size=d.per_device_batch_size,
        gradient_accumulation_steps=d.grad_accum,
        max_grad_norm=d.max_grad_norm,
        bf16=d.bf16,
        gradient_checkpointing=d.gradient_checkpointing,
        eval_strategy=d.eval_strategy,
        eval_steps=d.eval_steps,
        logging_steps=d.logging_steps,
        save_steps=d.save_steps,
        save_total_limit=d.save_total_limit,
        load_best_model_at_end=d.load_best_model_at_end,
        metric_for_best_model=d.metric_for_best_model,
        report_to=["wandb"] if cfg.tracking.mode != "disabled" else [],
        run_name=cfg.run_name,
    )

    meta = {
        "run_name": cfg.run_name,
        "git_sha": git_sha(short=False),
        "seed": cfg.seed,
        "base_model": str(base_model),
        "pref_file": str(d.pref_file),
        "n_pairs": ds.num_rows,
        "beta": float(d.beta),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    trainer = DPOTrainer(
        model=tok_src,                  # string/path → TRL loads it; LoRA + auto reference policy
        args=dpo_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))

    if d.use_lora and bool(d.get("merge_after_train", False)):
        merged_dir = out_dir / "merged"
        print(f"[dpo] merging LoRA adapter into base -> {merged_dir}")
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        print(f"[dpo] merged full model saved in {merged_dir}")

    print(f"[dpo] done. artifacts in {out_dir}")


if __name__ == "__main__":
    main()
