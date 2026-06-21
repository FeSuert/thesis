"""Phase 4 evaluation harness (RQ2 privacy mitigation; RQ3 utility).

Compares Defender variants (D-021) under ONE attacker on the held-out benchmark split:
  - undefended : original user turns (baseline ASR / leak — the threat we mitigate)
  - base       : raw Qwen3.5-4B as Defender (no privacy training)
  - sft        : SFT-only merged model
  - sftdpo_*   : SFT+DPO merged model(s)
  - dpoonly    : DPO-only merged model

For each variant and persona: rewrite the (capped) user turns with that Defender (memoryless,
D-001), then run the attacker on the rewritten turns and score per-attribute correctness vs the
gold answer key (skip Unknown). Reports per variant:
  - ASR  = correct-inference rate per attribute (lower = more private)
  - leak = mean P_att over known attributes (continuous privacy signal)
  - util = mean S_sem(original, rewrite) (higher = meaning preserved)

PASS 1 uses attacker=Qwen3.5-9B (in-distribution — same family the Defender trained against; the
most sensitive check that DPO reduced what it optimized). PASS 2 (thesis transferability, D-015d)
swaps --attacker for a NON-Qwen model; same code.

The attacker + evaluator load once; each Defender loads/frees in turn (so the 9B attacker + one
4B Defender coexist → run on fat_gpu / H100).

Run (fat_gpu):
  HF_HOME=/lustre/urdeniw5/hf_cache HF_HUB_OFFLINE=1 \
    uv run python -m thesis.evaluation.run_eval \
      --defenders undefended=none base=Qwen/Qwen3.5-4B sft=outputs/sft-qwen3_5-4b/merged \
        sftdpo_v1=outputs/dpo-sftdpo-qwen3_5-4b/merged \
        sftdpo_v2=outputs/dpo-sftdpo-qwen3_5-4b-v2/merged \
        dpoonly=outputs/dpo-only-qwen3_5-4b/merged \
      --limit 60
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
from thesis.models.evaluator.similarity import SemanticSimilarity
from thesis.utils.reproducibility import repo_root, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 evaluation: ASR / leak / utility per variant.")
    p.add_argument("--defenders", nargs="+", required=True,
                   help="Variants as name=path. path may be 'none' (undefended), a HF id "
                        "(e.g. Qwen/Qwen3.5-4B for base), or a merged model dir.")
    p.add_argument("--attacker", default="Qwen/Qwen3.5-9B")
    p.add_argument("--evaluator", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--gt", default="data-public/labels_groundtruth.jsonl")
    p.add_argument("--data-root", default="data/personamem-v2")
    p.add_argument("--bucket", default="32k", choices=["32k", "128k"])
    p.add_argument("--split", default="benchmark", choices=["benchmark", "pool", "all"])
    p.add_argument("--limit", type=int, default=0, help="Cap personas (0 = all in split).")
    p.add_argument("--max-turns", type=int, default=15)
    p.add_argument("--min-turn-chars", type=int, default=40)
    p.add_argument("--age-tol", type=int, default=5)
    p.add_argument("--max-context-tokens", type=int, default=12000)
    p.add_argument("--out", default="outputs/eval/pass1")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = repo_root()

    gt_path = Path(args.gt) if Path(args.gt).is_absolute() else root / args.gt
    data_root = Path(args.data_root) if Path(args.data_root).is_absolute() else root / args.data_root
    out_dir = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- assemble the fixed evaluation set (same for every variant) ----
    rows = load_ground_truth(gt_path, args.split)
    if args.limit > 0:
        rows = rows[: args.limit]
    personas = []
    for row in rows:
        pid = int(row["persona_id"])
        gold_known = {a: row["attributes"][a]["value"] for a in ATTRIBUTES
                      if _norm(row["attributes"].get(a, {}).get("value", "Unknown")) not in UNKNOWN}
        ch = find_chat_history(data_root, pid, args.bucket)
        if ch is None or not gold_known:
            continue
        turns = [t for t in load_chat_history_from_path(ch).user_turns()
                 if len(t.strip()) >= args.min_turn_chars][: args.max_turns]
        if turns:
            personas.append({"pid": pid, "gold": gold_known, "turns": turns})
    print(f"[eval] {len(personas)} personas; attacker={args.attacker}; "
          f"variants={[d.split('=')[0] for d in args.defenders]}")

    attacker = LLMAttacker(model_name=args.attacker, max_context_tokens=args.max_context_tokens)
    evaluator = SemanticSimilarity(model_name=args.evaluator)

    report: dict[str, dict] = {}

    for spec in args.defenders:
        name, _, path = spec.partition("=")
        print(f"\n[eval] === variant '{name}' (defender={path or 'none'}) ===")
        defender = None
        if path and path.lower() != "none":
            dpath = path if path.startswith("Qwen/") else str(
                (Path(path) if Path(path).is_absolute() else root / path))
            defender = Defender(dpath)

        terms = {a: [] for a in ATTRIBUTES}     # attr -> list of (correct:int, p_att:float)
        utils: list[float] = []

        for i, per in enumerate(personas, 1):
            if defender is None:
                rw = per["turns"]
            else:
                rw = [defender.rewrite(t) for t in per["turns"]]
                sims = [evaluator.cosine(o, r) for o, r in zip(per["turns"], rw)]
                utils.append(sum(sims) / len(sims))
            attack = attacker.infer(rw)
            for attr, gval in per["gold"].items():
                correct = int(score_attribute(attr, attack.guesses.get(attr, ""), gval, args.age_tol))
                terms[attr].append((correct, float(attack.p_att.get(attr, 0.0))))
            if i % 20 == 0:
                print(f"    {i}/{len(personas)} personas done")

        def _asr(a):
            t = terms[a]
            return None if not t else sum(c for c, _ in t) / len(t)

        def _leak(a):
            t = terms[a]
            return None if not t else sum(p for _, p in t) / len(t)

        all_t = [x for a in ATTRIBUTES for x in terms[a]]
        report[name] = {
            "defender": path or "none",
            "n_personas": len(personas),
            "utility_mean_ssem": (sum(utils) / len(utils)) if utils else None,
            "overall": {
                "asr": (sum(c for c, _ in all_t) / len(all_t)) if all_t else None,
                "leak": (sum(p for _, p in all_t) / len(all_t)) if all_t else None,
            },
            "per_attribute": {a: {"n": len(terms[a]), "asr": _asr(a), "leak": _leak(a)}
                              for a in ATTRIBUTES},
        }

        if defender is not None:
            del defender
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    (out_dir / "eval_report.json").write_text(json.dumps(
        {"attacker": args.attacker, "split": args.split, "variants": report}, indent=2))

    # ---- print comparison table ----
    def f(x):
        return "  n/a" if x is None else f"{x:.3f}"

    print("\n=== Phase 4 eval — variant comparison ===")
    print(f"attacker={args.attacker}  split={args.split}  personas={len(personas)}")
    print(f"{'variant':<12} {'ASR':>7} {'leak':>7} {'utility':>8}   (ASR/leak lower=better, util higher=better)")
    for name, r in report.items():
        print(f"{name:<12} {f(r['overall']['asr']):>7} {f(r['overall']['leak']):>7} "
              f"{f(r['utility_mean_ssem']):>8}")
    print("\nPer-attribute ASR:")
    print(f"{'variant':<12} " + " ".join(f"{a:>7}" for a in ATTRIBUTES))
    for name, r in report.items():
        print(f"{name:<12} " + " ".join(f"{f(r['per_attribute'][a]['asr']):>7}" for a in ATTRIBUTES))
    print(f"\n[eval] report -> {out_dir / 'eval_report.json'}")


if __name__ == "__main__":
    main()
