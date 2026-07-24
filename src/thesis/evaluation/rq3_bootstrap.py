"""Bootstrap confidence intervals for RQ3 judge metrics.

Reads the per-item judged_<variant>/judged.jsonl files written by rq3_judge and puts
95% CIs on the utility metrics, plus paired ΔCIs vs a baseline variant (default
sftdpo_v2) so comparisons like "sftdpo_v2 loses less utility than presidio" become
statistically defensible.

Resampling unit = PERSONA (turns within a persona are correlated). Paired: one
resample of persona ids per iteration, applied to every variant, so per-variant
differences use the same draw. Variants judge different edited-turn sets, but they
share the persona universe, so pairing at the persona level is valid; a persona with
no edited turns for a variant simply contributes no items to that variant.

Metrics per variant (over the turns the defender edited):
  mean_rewrite, mean_original, mean_gap (orig-rw), rewrite_ge_rate, tie_rate, mean_sim.

No GPU / API / torch — runs in seconds on the login node:
  uv run --no-sync python -m thesis.evaluation.rq3_bootstrap \
    --judged-dir outputs/eval/rq3 --baseline sftdpo_v2 --n-boot 10000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["mean_rewrite", "mean_original", "mean_gap", "rewrite_ge_rate", "tie_rate", "mean_sim"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap CIs for RQ3 judge metrics.")
    p.add_argument("--judged-dir", default="outputs/eval/rq3",
                   help="Directory holding judged_<variant>/judged.jsonl.")
    p.add_argument("--variants", nargs="*", default=None,
                   help="Variant names. Default: auto-detect judged_*/judged.jsonl.")
    p.add_argument("--baseline", default="sftdpo_v2",
                   help="Variant for paired Δ comparisons (gap / rewrite_ge_rate).")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--ci", type=float, default=95.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_judged(path: Path) -> dict[int, list[dict]]:
    by_p: dict[int, list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        j = r.get("judge", {})
        if "score_original" not in j or "score_rewrite" not in j:
            continue  # parse-failed rows carry no scores
        sim = r.get("response_similarity")
        by_p[int(r["persona_id"])].append({
            "so": float(j["score_original"]),
            "sr": float(j["score_rewrite"]),
            "sim": float(sim) if sim is not None else np.nan,
            "tie": 1.0 if str(j.get("verdict")) == "tie" else 0.0,
        })
    return by_p


def metrics_from_items(items: list[dict]) -> dict[str, float]:
    if not items:
        return {m: float("nan") for m in METRICS}
    so = np.array([i["so"] for i in items])
    sr = np.array([i["sr"] for i in items])
    sim = np.array([i["sim"] for i in items])
    tie = np.array([i["tie"] for i in items])
    return {
        "mean_rewrite": float(sr.mean()),
        "mean_original": float(so.mean()),
        "mean_gap": float((so - sr).mean()),
        "rewrite_ge_rate": float((sr >= so).mean()),
        "tie_rate": float(tie.mean()),
        "mean_sim": float(np.nanmean(sim)),
    }


def _ci(arr: np.ndarray, lo_q: float, hi_q: float) -> list[float]:
    a = arr[~np.isnan(arr)]
    if a.size == 0:
        return [float("nan"), float("nan")]
    return [round(float(np.percentile(a, lo_q)), 4), round(float(np.percentile(a, hi_q)), 4)]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    jdir = Path(args.judged_dir)

    if args.variants:
        variants = list(args.variants)
    else:
        variants = sorted(
            os.path.basename(os.path.dirname(f))[len("judged_"):]
            for f in glob.glob(str(jdir / "judged_*" / "judged.jsonl"))
        )
    if not variants:
        raise SystemExit(f"No judged_*/judged.jsonl found under {jdir}")

    data = {v: load_judged(jdir / f"judged_{v}" / "judged.jsonl") for v in variants}
    all_pids = sorted(set().union(*(set(d.keys()) for d in data.values())))
    n = len(all_pids)

    # Flatten items per variant for point estimates.
    flat = {v: [it for p in data[v] for it in data[v][p]] for v in variants}
    point = {v: metrics_from_items(flat[v]) for v in variants}
    n_items = {v: len(flat[v]) for v in variants}

    lo_q = (100.0 - args.ci) / 2.0
    hi_q = 100.0 - lo_q

    boot = {v: {m: np.empty(args.n_boot) for m in METRICS} for v in variants}
    have_base = args.baseline in variants
    boot_dgap = {v: np.empty(args.n_boot) for v in variants if v != args.baseline}   # variant_gap - baseline_gap
    boot_dge = {v: np.empty(args.n_boot) for v in variants if v != args.baseline}    # baseline_ge - variant_ge

    pid_arr = np.array(all_pids)
    for b in range(args.n_boot):
        sampled = pid_arr[rng.integers(0, n, n)]
        per_var_m = {}
        for v in variants:
            dv = data[v]
            items: list[dict] = []
            for p in sampled:
                lst = dv.get(int(p))
                if lst:
                    items.extend(lst)
            mv = metrics_from_items(items)
            per_var_m[v] = mv
            for m in METRICS:
                boot[v][m][b] = mv[m]
        if have_base:
            bg = per_var_m[args.baseline]["mean_gap"]
            bge = per_var_m[args.baseline]["rewrite_ge_rate"]
            for v in variants:
                if v == args.baseline:
                    continue
                boot_dgap[v][b] = per_var_m[v]["mean_gap"] - bg      # >0 = variant worse (bigger gap)
                boot_dge[v][b] = bge - per_var_m[v]["rewrite_ge_rate"]  # >0 = baseline better

    report: dict = {
        "judged_dir": str(jdir), "n_personas": n, "n_boot": args.n_boot,
        "ci_pct": args.ci, "baseline": args.baseline, "seed": args.seed, "variants": {},
    }
    for v in variants:
        report["variants"][v] = {
            "n_items": n_items[v],
            **{m: {"est": round(point[v][m], 4), "ci": _ci(boot[v][m], lo_q, hi_q)} for m in METRICS},
        }
        if have_base and v != args.baseline:
            report["variants"][v]["delta_gap_vs_baseline"] = {
                "est": round(point[v]["mean_gap"] - point[args.baseline]["mean_gap"], 4),
                "ci": _ci(boot_dgap[v], lo_q, hi_q),
            }
            report["variants"][v]["delta_rewrite_ge_baseline_minus_variant"] = {
                "est": round(point[args.baseline]["rewrite_ge_rate"] - point[v]["rewrite_ge_rate"], 4),
                "ci": _ci(boot_dge[v], lo_q, hi_q),
            }

    out_path = Path(args.out) if args.out else jdir / "rq3_bootstrap_ci.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    ci = int(args.ci)
    print(f"\n=== RQ3 judge-metric bootstrap ({ci}% CI, n_boot={args.n_boot}, "
          f"personas={n}) — baseline={args.baseline} ===\n")
    print(f"{'variant':<12} {'items':>6} {'gap':>6} {'gap CI':>15} {'rw>=orig':>9} "
          f"{'rw>=orig CI':>15} {'rw_score':>8}")
    for v in variants:
        r = report["variants"][v]
        g, ge, rw = r["mean_gap"], r["rewrite_ge_rate"], r["mean_rewrite"]
        print(f"{v:<12} {r['n_items']:>6} {g['est']:>6.3f} "
              f"[{g['ci'][0]:.3f},{g['ci'][1]:.3f}]".rjust(16) +
              f" {ge['est']:>9.3f} " +
              f"[{ge['ci'][0]:.3f},{ge['ci'][1]:.3f}]".rjust(15) +
              f" {rw['est']:>8.3f}")

    if have_base:
        print(f"\nPaired Δ vs baseline ({args.baseline}):")
        print(f"{'variant':<12} {'Δgap':>7} {'Δgap 95% CI':>18}  (>0 = worse than baseline)")
        for v in variants:
            if v == args.baseline:
                continue
            d = report["variants"][v]["delta_gap_vs_baseline"]
            sig = "" if (d["ci"][0] > 0 or d["ci"][1] < 0) else "  (CI spans 0)"
            print(f"{v:<12} {d['est']:>7.3f}  [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]{sig}")

    print(f"\n[rq3-bootstrap] report -> {out_path}")


if __name__ == "__main__":
    main()
