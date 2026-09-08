"""Generate Staab-anonymizer rewrites for the benchmark personas.

Runs on a GPU compute node (fat_gpu), fully offline. Reads
outputs/eval/v2/turns_undefended.jsonl (from run_eval_v2) and writes
outputs/eval/v2/turns_staab.jsonl in the SAME persona/turn order, so it drops
into run_eval_v2 as a precomputed variant:  staab=outputs/eval/v2/turns_staab.jsonl
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
    p.add_argument("--in", dest="in_file", default="outputs/eval/v2/turns_undefended.jsonl")
    p.add_argument("--out", default="outputs/eval/v2/turns_staab.jsonl")
    p.add_argument("--return-last", action="store_true",
                   help="Return the final iteration instead of the lowest-leak one.")
    p.add_argument("--dump-iters", action="store_true",
                   help="Also write turns_staab_iter{1..N}.jsonl for the ASR-vs-iteration figure.")
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
    print(f"Loaded {len(lines)} turns from {in_file}")

    print(f"Loading anonymizer={args.anonymizer} + adversary={args.adversary} ...")
    anon = StaabAnonymizer(
        anonymizer_model=args.anonymizer,
        adversary_model=args.adversary,
        max_iters=args.max_iters,
        early_stop_conf=args.early_stop_conf,
        return_best=not args.return_last,
    )

    diag_file = out_file.with_name(out_file.stem + "_diag.jsonl")
    iter_writers = {}
    if args.dump_iters:
        for r in range(1, args.max_iters + 1):
            iter_writers[r] = out_file.with_name(f"{out_file.stem}_iter{r}.jsonl").open(
                "w", encoding="utf-8")

    with out_file.open("w", encoding="utf-8") as out_f, diag_file.open("w", encoding="utf-8") as diag_f:
        for idx, item in enumerate(lines, 1):
            orig = item["original"]
            rw = anon.rewrite(orig)
            rec = {"persona_id": item["persona_id"], "turn_idx": item["turn_idx"],
                   "original": orig, "rewrite": rw}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            diag_f.write(json.dumps({
                "persona_id": item["persona_id"], "turn_idx": item["turn_idx"],
                "rounds": anon.last_rounds, "best_mean_conf": round(anon.last_best_conf, 4),
            }, ensure_ascii=False) + "\n")
            if args.dump_iters:
                trace = anon.last_trace  # [orig, iter1, iter2, ...]
                for r, w in iter_writers.items():
                    text_r = trace[r] if r < len(trace) else trace[-1]
                    w.write(json.dumps({**rec, "rewrite": text_r}, ensure_ascii=False) + "\n")
            if idx % 25 == 0 or idx == len(lines):
                print(f"  {idx}/{len(lines)} turns anonymized")

    for w in iter_writers.values():
        w.close()
    print(f"\nSaved {out_file}\nDiagnostics -> {diag_file}")


if __name__ == "__main__":
    main()
