"""Instrumented evaluation harness — dumps per-turn and per-persona artifacts.

Same scoring logic as run_eval.py but additionally persists:
  - Per-turn: original text, rewrite, S_sem, whether text changed.
  - Per-persona: attacker raw output snippet, parse success, per-attribute breakdown.
  - Aggregates: S_sem distribution stats, % of prompts modified, attacker parse failure rate.

Supports --presidio flag to include the Presidio NER baseline without a GPU model.

Run (fat_gpu or fat_gpu_h200):
  HF_HOME=/lustre/urdeniw5/hf_cache HF_HUB_OFFLINE=1 \
    uv run python -m thesis.evaluation.run_eval_v2 \
      --defenders undefended=none sft=outputs/sft-qwen3_5-4b/merged \
        sftdpo_v2=outputs/dpo-sftdpo-qwen3_5-4b-v2/merged \
      --presidio --attacker Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
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
    p = argparse.ArgumentParser(description="Instrumented Phase 4 evaluation.")
    p.add_argument("--defenders", nargs="+", required=True,
                   help="name=path pairs. 'none'=undefended, HF id, or merged model dir.")
    p.add_argument("--presidio", action="store_true",
                   help="Include Presidio NER redaction as a baseline variant.")
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
    p.add_argument("--out", default="outputs/eval/v2")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _text_changed(original: str, rewrite: str, threshold: float = 0.95) -> bool:
    """True if the rewrite differs meaningfully from the original."""
    if original.strip() == rewrite.strip():
        return False
    from difflib import SequenceMatcher
    return SequenceMatcher(None, original, rewrite).ratio() < threshold


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = repo_root()

    gt_path = Path(args.gt) if Path(args.gt).is_absolute() else root / args.gt
    data_root = Path(args.data_root) if Path(args.data_root).is_absolute() else root / args.data_root
    out_dir = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation set
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
    print(f"[eval-v2] {len(personas)} personas; attacker={args.attacker}")

    attacker = LLMAttacker(model_name=args.attacker, max_context_tokens=args.max_context_tokens)
    evaluator = SemanticSimilarity(model_name=args.evaluator)

    # Build variant list
    variant_specs: list[tuple[str, str]] = []
    for spec in args.defenders:
        name, _, path = spec.partition("=")
        variant_specs.append((name, path or "none"))
    if args.presidio:
        variant_specs.append(("presidio", "__presidio__"))

    full_report: dict[str, dict] = {}

    for vname, vpath in variant_specs:
        print(f"\n[eval-v2] === variant '{vname}' ===")
        defender = None
        if vpath == "__presidio__":
            from thesis.evaluation.baselines.presidio_defender import PresidioDefender
            defender = PresidioDefender()
        elif vpath.lower() != "none":
            dpath = vpath if vpath.startswith("Qwen/") else str(
                (Path(vpath) if Path(vpath).is_absolute() else root / vpath))
            defender = Defender(dpath)

        turn_log = (out_dir / f"turns_{vname}.jsonl").open("w", encoding="utf-8")
        persona_rows: list[dict] = []

        all_ssem: list[float] = []
        all_changed: list[bool] = []
        parse_ok = 0
        parse_fail = 0
        terms = {a: [] for a in ATTRIBUTES}

        for i, per in enumerate(personas, 1):
            if defender is None:
                rw = per["turns"]
            else:
                rw = [defender.rewrite(t) for t in per["turns"]]

            per_ssem: list[float] = []
            per_changed: list[bool] = []

            for ti, (orig, rewrite) in enumerate(zip(per["turns"], rw)):
                ssem = evaluator.cosine(orig, rewrite) if defender is not None else 1.0
                changed = _text_changed(orig, rewrite) if defender is not None else False
                per_ssem.append(ssem)
                per_changed.append(changed)
                all_ssem.append(ssem)
                all_changed.append(changed)

                turn_log.write(json.dumps({
                    "persona_id": per["pid"], "turn_idx": ti,
                    "original": orig[:500], "rewrite": rewrite[:500],
                    "s_sem": round(ssem, 4), "changed": changed,
                }, ensure_ascii=False) + "\n")

            # Attack the rewritten turns
            attack = attacker.infer(rw)
            raw = attack.raw_output
            parsed_ok = bool(attack.guesses.get("loc"))
            if parsed_ok:
                parse_ok += 1
            else:
                parse_fail += 1

            per_correct: dict[str, int] = {}
            per_patt: dict[str, float] = {}
            for attr, gval in per["gold"].items():
                correct = int(score_attribute(attr, attack.guesses.get(attr, ""), gval, args.age_tol))
                conf = float(attack.p_att.get(attr, 0.0))
                terms[attr].append((correct, conf))
                per_correct[attr] = correct
                per_patt[attr] = conf

            persona_rows.append({
                "persona_id": per["pid"],
                "n_turns": len(per["turns"]),
                "mean_ssem": round(float(np.mean(per_ssem)), 4) if per_ssem else None,
                "pct_changed": round(sum(per_changed) / len(per_changed), 3) if per_changed else None,
                "per_attr_correct": per_correct,
                "per_attr_patt": {k: round(v, 4) for k, v in per_patt.items()},
                "attacker_parse_ok": parsed_ok,
                "attacker_raw_snippet": raw[:300],
            })

            if i % 20 == 0 or i == len(personas):
                print(f"    {i}/{len(personas)} done")

        turn_log.close()

        # Compute aggregates
        def _asr(a):
            t = terms[a]
            return round(sum(c for c, _ in t) / len(t), 4) if t else None

        def _leak(a):
            t = terms[a]
            return round(sum(p for _, p in t) / len(t), 4) if t else None

        all_t = [x for a in ATTRIBUTES for x in terms[a]]
        overall_asr = round(sum(c for c, _ in all_t) / len(all_t), 4) if all_t else None
        overall_leak = round(sum(p for _, p in all_t) / len(all_t), 4) if all_t else None

        ssem_arr = np.array(all_ssem) if all_ssem else np.array([])
        pct_changed = round(sum(all_changed) / len(all_changed), 4) if all_changed else None

        variant_result = {
            "defender": vpath,
            "n_personas": len(personas),
            "overall": {"asr": overall_asr, "leak": overall_leak},
            "utility": {
                "mean_ssem": round(float(ssem_arr.mean()), 4) if len(ssem_arr) > 0 else None,
                "median_ssem": round(float(np.median(ssem_arr)), 4) if len(ssem_arr) > 0 else None,
                "std_ssem": round(float(ssem_arr.std()), 4) if len(ssem_arr) > 0 else None,
                "min_ssem": round(float(ssem_arr.min()), 4) if len(ssem_arr) > 0 else None,
                "p5_ssem": round(float(np.percentile(ssem_arr, 5)), 4) if len(ssem_arr) > 0 else None,
                "p25_ssem": round(float(np.percentile(ssem_arr, 25)), 4) if len(ssem_arr) > 0 else None,
                "pct_changed": pct_changed,
                "n_turns_total": len(all_ssem),
            },
            "attacker_parse": {
                "ok": parse_ok,
                "fail": parse_fail,
                "fail_rate": round(parse_fail / (parse_ok + parse_fail), 4)
                if (parse_ok + parse_fail) else None,
            },
            "per_attribute": {a: {"n": len(terms[a]), "asr": _asr(a), "leak": _leak(a)}
                              for a in ATTRIBUTES},
        }
        full_report[vname] = variant_result

        # Per-persona rows for bootstrap
        (out_dir / f"personas_{vname}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in persona_rows), encoding="utf-8"
        )

        if isinstance(defender, Defender):
            del defender
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Write final report
    payload = {
        "attacker": args.attacker, "split": args.split, "n_personas": len(personas),
        "variants": full_report,
    }
    (out_dir / "eval_report_v2.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Print summary
    def f(x):
        return "  n/a" if x is None else f"{x:.3f}"

    print("\n=== eval v2 — variant comparison ===")
    print(f"attacker={args.attacker}  split={args.split}  personas={len(personas)}")
    print(f"{'variant':<12} {'ASR':>7} {'leak':>7} {'S_sem':>7} {'%chg':>6} {'parseFail':>9}")
    for name, r in full_report.items():
        u = r["utility"]
        p = r["attacker_parse"]
        print(f"{name:<12} {f(r['overall']['asr']):>7} {f(r['overall']['leak']):>7} "
              f"{f(u['mean_ssem']):>7} {f(u['pct_changed']):>6} "
              f"{f(p['fail_rate']):>9}")
    print("\nPer-attribute ASR:")
    print(f"{'variant':<12} " + " ".join(f"{a:>7}" for a in ATTRIBUTES))
    for name, r in full_report.items():
        print(f"{name:<12} " + " ".join(f"{f(r['per_attribute'][a]['asr']):>7}" for a in ATTRIBUTES))
    print(f"\nS_sem distribution (defended variants):")
    for name, r in full_report.items():
        u = r["utility"]
        if u["mean_ssem"] is not None and u["mean_ssem"] < 1.0:
            print(f"  {name}: mean={u['mean_ssem']} median={u['median_ssem']} "
                  f"std={u['std_ssem']} min={u['min_ssem']} p5={u['p5_ssem']} "
                  f"p25={u['p25_ssem']} %changed={u['pct_changed']}")
    print(f"\n[eval-v2] report -> {out_dir / 'eval_report_v2.json'}")
    print(f"[eval-v2] per-turn logs -> {out_dir}/turns_*.jsonl")
    print(f"[eval-v2] per-persona rows -> {out_dir}/personas_*.jsonl")


if __name__ == "__main__":
    main()
