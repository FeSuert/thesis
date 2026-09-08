"""Generate Staab-anonymizer rewrites for the benchmark personas.

Runs on a GPU compute node (fat_gpu), fully offline. Reads
outputs/eval/v2/turns_undefended.jsonl (from run_eval_v2) and writes
outputs/eval/v2/turns_staab.jsonl in the SAME persona/turn order, so it drops
into run_eval_v2 as a precomputed variant:  staab=outputs/eval/v2/turns_staab.jsonl

Crash-safe and resumable: every line is flushed, and re-running skips the
(persona_id, turn_idx) pairs already written. If a job times out, just resubmit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis.evaluation.baselines.staab_anonymizer import StaabAnonymizer
from thesis.utils.reproducibility import repo_root, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Staab adversarial-anonymization generator.")
    p.add_argument("--anonymizer", default="Qwen/Qwen3.5-4B")
    p.add_argument("--adversary", default="Qwen/Qwen3.5-9B")
    p.add_argument("--max-iters", type=int, default=3)
    p.add_argument("--early-stop-conf", type=float, default=0.5)
    p.add_argument("--thinking", action="store_true",
                   help="Use the thinking adversary (stronger, ~3x slower). Default: non-thinking.")
    p.add_argument("--adversary-max-new-tokens", type=int, default=1024,
                   help="Only used with --thinking. Lower (e.g. 640) to trade accuracy for speed.")
    p.add_argument("--in", dest="in_file", default="outputs/eval/v2/turns_undefended.jsonl")
    p.add_argument("--out", default="outputs/eval/v2/turns_staab.jsonl")
    p.add_argument("--return-last", action="store_true",
                   help="Return the final iteration instead of the lowest-leak one.")
    p.add_argument("--fresh", action="store_true", help="Ignore any existing output and restart.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = repo_root()

    in_file = Path(args.in_file) if Path(args.in_file).is_absolute() else root / args.in_file
    out_file = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if not in_file.exists():
        raise SystemExit(f"{in_file} not found. Run run_eval_v2 first (needs turns_undefended.jsonl).")

    lines = [json.loads(l) for l in in_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Resume: keep only the valid rows already written, and skip them this run.
    done: set[tuple] = set()
    if out_file.exists() and not args.fresh:
        kept = []
        for l in out_file.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            try:
                r = json.loads(l)                      # stop at the first corrupt (truncated) line
            except Exception:
                break
            done.add((r["persona_id"], r["turn_idx"]))
            kept.append(l)
        out_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"Loaded {len(lines)} turns; {len(done)} already done, {len(lines) - len(done)} to do.")

    mode = "adversary=thinking" if args.thinking else "adversary=non-thinking"
    print(f"Loading anonymizer={args.anonymizer} + adversary={args.adversary} ({mode}) ...")
    anon = StaabAnonymizer(
        anonymizer_model=args.anonymizer,
        adversary_model=args.adversary,
        max_iters=args.max_iters,
        early_stop_conf=args.early_stop_conf,
        adversary_thinking=args.thinking,
        adversary_max_new_tokens=args.adversary_max_new_tokens,
        return_best=not args.return_last,
    )

    diag_file = out_file.with_name(out_file.stem + "_diag.jsonl")
    fmode = "a" if done else "w"
    with out_file.open(fmode, encoding="utf-8") as out_f, diag_file.open(fmode, encoding="utf-8") as diag_f:
        for idx, item in enumerate(lines, 1):
            if (item["persona_id"], item["turn_idx"]) in done:
                continue
            rw = anon.rewrite(item["original"])
            out_f.write(json.dumps({
                "persona_id": item["persona_id"], "turn_idx": item["turn_idx"],
                "original": item["original"], "rewrite": rw,
            }, ensure_ascii=False) + "\n")
            out_f.flush()                              # crash-safe: survive a timeout/kill
            diag_f.write(json.dumps({
                "persona_id": item["persona_id"], "turn_idx": item["turn_idx"],
                "rounds": anon.last_rounds, "best_mean_conf": round(anon.last_best_conf, 4),
            }, ensure_ascii=False) + "\n")
            diag_f.flush()
            if idx % 25 == 0 or idx == len(lines):
                print(f"  {idx}/{len(lines)} processed", flush=True)

    print(f"\nSaved {out_file}\nDiagnostics -> {diag_file}")


if __name__ == "__main__":
    main()
