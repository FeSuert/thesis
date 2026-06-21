"""RQ1 — longitudinal risk: ASR as a function of conversation length.

For each persona, the Defender rewrites the user turns once, then the attacker is run on
GROWING PREFIXES of the (rewritten) turns at checkpoints k = 1, 3, 5, 8, 12, ... We aggregate
ASR(k) and leak(k) across personas to plot how attribute-inference risk accumulates as the chat
grows — undefended vs. defended. The thesis's central question (RQ1): undefended risk should climb
with context, and the Defender should flatten/lower that curve.

Reuses the same attacker (Qwen in-distribution, or Gemma cross-model) + scorer — no new model.

Run (fat_gpu / fat_gpu_h200):
  HF_HOME=/lustre/urdeniw5/hf_cache HF_HUB_OFFLINE=1 \
    uv run python -m thesis.evaluation.rq1_longitudinal \
      --defenders undefended=none sftdpo_v2=outputs/dpo-sftdpo-qwen3_5-4b-v2/merged \
      --attacker Qwen/Qwen3.5-9B --limit 60
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from thesis.data.personamem import load_chat_history_from_path
from thesis.models.attacker.calibrate import (
    UNKNOWN, _norm, find_chat_history, load_ground_truth, score_attribute,
)
from thesis.models.attacker.llm_attacker import ATTRIBUTES, LLMAttacker
from thesis.models.defender.defender import Defender
from thesis.utils.reproducibility import repo_root, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ1: ASR vs conversation length.")
    p.add_argument("--defenders", nargs="+",
                   default=["undefended=none", "sftdpo_v2=outputs/dpo-sftdpo-qwen3_5-4b-v2/merged"],
                   help="name=path; path 'none' = undefended, HF id, or merged dir.")
    p.add_argument("--attacker", default="Qwen/Qwen3.5-9B")
    p.add_argument("--gt", default="data-public/labels_groundtruth.jsonl")
    p.add_argument("--data-root", default="data/personamem-v2")
    p.add_argument("--bucket", default="32k", choices=["32k", "128k"])
    p.add_argument("--split", default="benchmark", choices=["benchmark", "pool", "all"])
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--checkpoints", default="1,3,5,8,12",
                   help="Comma-separated turn counts at which to measure ASR.")
    p.add_argument("--min-turn-chars", type=int, default=40)
    p.add_argument("--age-tol", type=int, default=5)
    p.add_argument("--max-context-tokens", type=int, default=12000)
    p.add_argument("--out", default="outputs/eval/rq1")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = repo_root()
    checkpoints = sorted(int(x) for x in args.checkpoints.split(","))
    max_k = max(checkpoints)

    gt_path = Path(args.gt) if Path(args.gt).is_absolute() else root / args.gt
    data_root = Path(args.data_root) if Path(args.data_root).is_absolute() else root / args.data_root
    out_dir = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_ground_truth(gt_path, args.split)
    if args.limit > 0:
        rows = rows[: args.limit]
    personas = []
    for row in rows:
        pid = int(row["persona_id"])
        gold = {a: row["attributes"][a]["value"] for a in ATTRIBUTES
                if _norm(row["attributes"].get(a, {}).get("value", "Unknown")) not in UNKNOWN}
        ch = find_chat_history(data_root, pid, args.bucket)
        if ch is None or not gold:
            continue
        turns = [t for t in load_chat_history_from_path(ch).user_turns()
                 if len(t.strip()) >= args.min_turn_chars][:max_k]
        if turns:
            personas.append({"pid": pid, "gold": gold, "turns": turns})
    print(f"[rq1] {len(personas)} personas; checkpoints={checkpoints}; attacker={args.attacker}")

    attacker = LLMAttacker(model_name=args.attacker, max_context_tokens=args.max_context_tokens)

    # results[variant][k] = {"correct": int, "total": int, "leak_sum": float}
    results: dict[str, dict[int, dict]] = {}

    for spec in args.defenders:
        name, _, path = spec.partition("=")
        print(f"\n[rq1] === variant '{name}' (defender={path or 'none'}) ===")
        defender = None
        if path and path.lower() != "none":
            dpath = path if path.startswith("Qwen/") else str(
                (Path(path) if Path(path).is_absolute() else root / path))
            defender = Defender(dpath)

        acc = {k: {"correct": 0, "total": 0, "leak_sum": 0.0} for k in checkpoints}
        for i, per in enumerate(personas, 1):
            turns = per["turns"]
            rewritten = turns if defender is None else [defender.rewrite(t) for t in turns]
            for k in checkpoints:
                if k > len(rewritten):
                    continue
                attack = attacker.infer(rewritten[:k])
                for attr, gval in per["gold"].items():
                    correct = int(score_attribute(attr, attack.guesses.get(attr, ""), gval, args.age_tol))
                    acc[k]["correct"] += correct
                    acc[k]["total"] += 1
                    acc[k]["leak_sum"] += float(attack.p_att.get(attr, 0.0))
            if i % 20 == 0:
                print(f"    {i}/{len(personas)} personas done")

        results[name] = {
            str(k): {
                "n_attr": acc[k]["total"],
                "asr": (acc[k]["correct"] / acc[k]["total"]) if acc[k]["total"] else None,
                "leak": (acc[k]["leak_sum"] / acc[k]["total"]) if acc[k]["total"] else None,
            } for k in checkpoints
        }

        if defender is not None:
            del defender
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {"attacker": args.attacker, "split": args.split, "checkpoints": checkpoints,
               "results": results}
    (out_dir / "rq1_report.json").write_text(json.dumps(payload, indent=2))

    # CSV (variant, k, asr, leak) for plotting
    csv_lines = ["variant,k,asr,leak,n_attr"]
    for name, by_k in results.items():
        for k in checkpoints:
            r = by_k[str(k)]
            csv_lines.append(f"{name},{k},{r['asr']},{r['leak']},{r['n_attr']}")
    (out_dir / "rq1_curve.csv").write_text("\n".join(csv_lines))

    # ---- print ASR(k) table ----
    def f(x):
        return "  n/a" if x is None else f"{x:.3f}"

    print("\n=== RQ1 — ASR vs conversation length ===")
    print(f"attacker={args.attacker}  split={args.split}  personas={len(personas)}")
    print(f"{'k':>4} " + " ".join(f"{n:>12}" for n in results))
    for k in checkpoints:
        print(f"{k:>4} " + " ".join(f"{f(results[n][str(k)]['asr']):>12}" for n in results))
    print(f"\n[rq1] report -> {out_dir / 'rq1_report.json'}  | curve -> {out_dir / 'rq1_curve.csv'}")
    print("[rq1] expect undefended ASR to RISE with k; the defended curve should rise slower / lower.")


if __name__ == "__main__":
    main()
