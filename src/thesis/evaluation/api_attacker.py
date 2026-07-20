"""Cross-model transfer eval with an API attacker (e.g. Kimi K2.6).

Two-stage design: the rewrites were already produced on the GPU by run_eval_v2
(turns_<variant>.jsonl); this script replays them against an API attacker and
scores per-attribute correctness against the same gold answer key. No GPU needed —
runs on the login node or a laptop.

Uses the exact same attack system prompt and scoring as the local LLMAttacker, so
the numbers are directly comparable to the Qwen / Gemma passes.

One variant per turns file; run once per variant:
  uv run python -m thesis.evaluation.api_attacker \
    --turns outputs/eval/v2-qwen/turns_sftdpo_v2.jsonl --variant sftdpo_v2 \
    --base-url https://api.moonshot.ai/v1 --attacker-model kimi-k2.6 \
    --out outputs/eval/kimi
For the undefended row, point --turns at turns_undefended.jsonl (originals == rewrites).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from thesis.models.attacker.calibrate import (
    UNKNOWN, _norm, load_ground_truth, score_attribute,
)
from thesis.models.attacker.llm_attacker import ATTRIBUTES, _SYSTEM_PROMPT, _extract_json, _coerce_confidence
from thesis.utils.api_client import ChatClient, resolve_api_key
from thesis.utils.reproducibility import repo_root, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transfer eval with an API attacker.")
    p.add_argument("--turns", required=True,
                   help="Per-turn JSONL from run_eval_v2 (turns_<variant>.jsonl).")
    p.add_argument("--variant", required=True, help="Variant name for the report.")
    p.add_argument("--base-url", default="https://api.moonshot.ai/v1")
    p.add_argument("--attacker-model", default="kimi-k2.6")
    p.add_argument("--api-key", default=None,
                   help="Attacker API key. Falls back to env vars or interactive prompt.")
    p.add_argument("--gt", default="data-public/labels_groundtruth.jsonl")
    p.add_argument("--split", default="benchmark", choices=["benchmark", "pool", "all"])
    p.add_argument("--age-tol", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--limit", type=int, default=0, help="Cap personas (0 = all).")
    p.add_argument("--out", default="outputs/eval/api-attacker")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = repo_root()

    turns_path = Path(args.turns) if Path(args.turns).is_absolute() else root / args.turns
    gt_path = Path(args.gt) if Path(args.gt).is_absolute() else root / args.gt
    out_dir = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group the rewritten turns by persona, in turn order.
    by_persona: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for line in turns_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            by_persona[t["persona_id"]].append((t["turn_idx"], t["rewrite"]))
    for pid in by_persona:
        by_persona[pid].sort()

    gold_by_pid: dict[int, dict[str, str]] = {}
    for row in load_ground_truth(gt_path, args.split):
        pid = int(row["persona_id"])
        gold_by_pid[pid] = {a: row["attributes"][a]["value"] for a in ATTRIBUTES
                            if _norm(row["attributes"].get(a, {}).get("value", "Unknown"))
                            not in UNKNOWN}

    pids = [pid for pid in by_persona if gold_by_pid.get(pid)]
    if args.limit > 0:
        pids = pids[: args.limit]
    print(f"[api-attack] variant={args.variant} personas={len(pids)} "
          f"attacker={args.attacker_model}")

    per_persona_path = out_dir / f"attacks_{args.variant}.jsonl"
    done: set[int] = set()
    if per_persona_path.exists():
        for line in per_persona_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["persona_id"])
        print(f"[api-attack] resuming: {len(done)} personas already attacked")

    api_key = resolve_api_key(args.api_key, env_vars=("MOONSHOT_API_KEY", "OPENAI_API_KEY"))
    client = ChatClient(base_url=args.base_url, model=args.attacker_model, api_key=api_key)

    with per_persona_path.open("a", encoding="utf-8") as f:
        for i, pid in enumerate(pids, 1):
            if pid in done:
                continue
            turns = [text for _, text in by_persona[pid]]
            block = "User messages from the conversation:\n" + \
                    "\n".join(f"- {t}" for t in turns if t and t.strip())
            raw = client.chat([
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": block},
            ], temperature=0.0, max_tokens=args.max_new_tokens)

            try:
                parsed = _extract_json(raw)
                parse_ok = True
            except ValueError:
                parsed = {}
                parse_ok = False

            attrs: dict[str, dict] = {}
            for attr, gval in gold_by_pid[pid].items():
                entry = parsed.get(attr, {}) if isinstance(parsed, dict) else {}
                guess = entry.get("guess", "") if isinstance(entry, dict) else entry
                conf = _coerce_confidence(entry.get("confidence", 0)) \
                    if isinstance(entry, dict) else 0.0
                attrs[attr] = {
                    "gold": gval, "guess": guess, "conf": round(conf, 3),
                    "correct": bool(score_attribute(attr, guess, gval, args.age_tol)),
                }
            f.write(json.dumps({
                "persona_id": pid, "parse_ok": parse_ok,
                "attrs": attrs, "raw_snippet": raw[:300],
            }, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(pids):
                print(f"    {i}/{len(pids)} personas attacked")

    # Aggregate from the (possibly resumed) full file.
    rows = [json.loads(line) for line in per_persona_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    terms: dict[str, list[tuple[int, float]]] = {a: [] for a in ATTRIBUTES}
    parse_fail = sum(1 for r in rows if not r["parse_ok"])
    for r in rows:
        for attr, a in r["attrs"].items():
            terms[attr].append((int(a["correct"]), float(a["conf"])))

    all_t = [x for a in ATTRIBUTES for x in terms[a]]
    report = {
        "variant": args.variant,
        "attacker": args.attacker_model,
        "turns_file": str(turns_path),
        "n_personas": len(rows),
        "parse_fail": parse_fail,
        "overall": {
            "asr": round(sum(c for c, _ in all_t) / len(all_t), 4) if all_t else None,
            "leak": round(sum(p for _, p in all_t) / len(all_t), 4) if all_t else None,
        },
        "per_attribute": {
            a: {"n": len(terms[a]),
                "asr": round(sum(c for c, _ in terms[a]) / len(terms[a]), 4) if terms[a] else None,
                "leak": round(sum(p for _, p in terms[a]) / len(terms[a]), 4) if terms[a] else None}
            for a in ATTRIBUTES
        },
    }
    report_path = out_dir / f"report_{args.variant}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== API attacker report ===")
    print(f"variant={args.variant}  attacker={args.attacker_model}  "
          f"personas={len(rows)}  parse_fail={parse_fail}")
    print(f"overall ASR={report['overall']['asr']}  leak={report['overall']['leak']}")
    for a in ATTRIBUTES:
        pa = report["per_attribute"][a]
        print(f"  {a:<8} n={pa['n']:>4}  asr={pa['asr']}  leak={pa['leak']}")
    print(f"\n[api-attack] report -> {report_path}")


if __name__ == "__main__":
    main()
