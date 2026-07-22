"""Bootstrap confidence intervals for ASR — from eval-v2 per-persona rows.

Reads the personas_<variant>.jsonl files written by run_eval_v2 and produces, for
each variant:
  - overall ASR with a 95% CI,
  - per-attribute ASR with n and 95% CI,
and, against a chosen baseline (default: undefended), the *paired* ΔASR CI — i.e.
how much each variant reduces attack success, with a confidence interval on that
reduction. Paired because every variant is evaluated on the same personas, so we
resample persona indices once per iteration and apply them to all variants.

The resampling unit is the PERSONA (attributes within a persona are correlated), so
we draw personas with replacement at the original sample size — the standard
nonparametric bootstrap.

No GPU, no API, no torch — runs in seconds on the login node:
  uv run --no-sync python -m thesis.evaluation.bootstrap_ci \
    --personas-dir outputs/eval/v2 --baseline undefended --n-boot 10000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

# Canonical attribute order (matches the attacker); only those present are reported.
ATTRIBUTES = ["loc", "prof", "age", "sex", "status", "income"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap CIs for ASR from personas_*.jsonl.")
    p.add_argument("--personas-dir", default="outputs/eval/v2",
                   help="Directory holding personas_<variant>.jsonl files.")
    p.add_argument("--variants", nargs="*", default=None,
                   help="Variant names to include. Default: auto-detect all in the dir.")
    p.add_argument("--baseline", default="undefended",
                   help="Variant to measure reductions against (paired ΔASR).")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--ci", type=float, default=95.0, help="CI width in percent (default 95).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: <personas-dir>/bootstrap_ci.json).")
    return p.parse_args()


def load_variant(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[int(r["persona_id"])] = r
    return rows


def build_correct_matrix(rows: dict[int, dict], pids: list[int],
                         attrs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (correct, mask): shape (n_personas, n_attrs); NaN/0 where attr unknown."""
    n, m = len(pids), len(attrs)
    correct = np.full((n, m), np.nan, dtype=float)
    for i, pid in enumerate(pids):
        pac = rows[pid].get("per_attr_correct", {}) or {}
        for j, a in enumerate(attrs):
            if a in pac and pac[a] is not None:
                correct[i, j] = float(pac[a])
    mask = ~np.isnan(correct)
    return correct, mask


def _overall(correct: np.ndarray, mask: np.ndarray, idx: np.ndarray) -> float:
    sm = mask[idx]
    denom = sm.sum()
    if denom == 0:
        return float("nan")
    return float(np.nansum(correct[idx]) / denom)


def _per_attr(correct: np.ndarray, mask: np.ndarray, idx: np.ndarray) -> np.ndarray:
    sub, sm = correct[idx], mask[idx]
    num = np.nansum(sub, axis=0)
    den = sm.sum(axis=0)
    out = np.full(correct.shape[1], np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    pdir = Path(args.personas_dir)

    if args.variants:
        variants = list(args.variants)
    else:
        variants = sorted(
            os.path.basename(f)[len("personas_"):-len(".jsonl")]
            for f in glob.glob(str(pdir / "personas_*.jsonl"))
        )
    if not variants:
        raise SystemExit(f"No personas_*.jsonl found in {pdir}")

    loaded = {v: load_variant(pdir / f"personas_{v}.jsonl") for v in variants}

    # Common persona set across all variants (should be identical, but be safe).
    common = set.intersection(*(set(d.keys()) for d in loaded.values()))
    pids = sorted(common)
    n = len(pids)
    attrs = [a for a in ATTRIBUTES]  # report canonical order; missing -> masked out

    mats = {v: build_correct_matrix(loaded[v], pids, attrs) for v in variants}

    lo_q = (100.0 - args.ci) / 2.0
    hi_q = 100.0 - lo_q

    # Full-sample point estimates.
    full_idx = np.arange(n)
    point_overall = {v: _overall(*mats[v], full_idx) for v in variants}
    point_per_attr = {v: _per_attr(*mats[v], full_idx) for v in variants}
    attr_n = {v: mats[v][1].sum(axis=0).astype(int) for v in variants}

    # Bootstrap: one resample of persona indices per iteration, applied to all variants.
    boot_overall = {v: np.empty(args.n_boot) for v in variants}
    boot_per_attr = {v: np.empty((args.n_boot, len(attrs))) for v in variants}
    boot_delta = {v: np.empty(args.n_boot) for v in variants if v != args.baseline}
    have_baseline = args.baseline in variants

    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)  # with replacement, same size
        base_val = _overall(*mats[args.baseline], idx) if have_baseline else float("nan")
        for v in variants:
            ov = _overall(*mats[v], idx)
            boot_overall[v][b] = ov
            boot_per_attr[v][b] = _per_attr(*mats[v], idx)
            if have_baseline and v != args.baseline:
                # ΔASR = reduction vs baseline (positive = more private than baseline).
                boot_delta[v][b] = base_val - ov

    def ci(arr: np.ndarray) -> list[float]:
        a = arr[~np.isnan(arr)]
        return [round(float(np.percentile(a, lo_q)), 4),
                round(float(np.percentile(a, hi_q)), 4)]

    report: dict = {
        "personas_dir": str(pdir),
        "n_personas": n,
        "n_boot": args.n_boot,
        "ci_pct": args.ci,
        "baseline": args.baseline,
        "seed": args.seed,
        "variants": {},
    }
    for v in variants:
        report["variants"][v] = {
            "overall": {
                "asr": round(point_overall[v], 4),
                "ci": ci(boot_overall[v]),
            },
            "per_attribute": {
                attrs[j]: {
                    "n": int(attr_n[v][j]),
                    "asr": (round(float(point_per_attr[v][j]), 4)
                            if not np.isnan(point_per_attr[v][j]) else None),
                    "ci": ci(boot_per_attr[v][:, j]),
                }
                for j in range(len(attrs)) if attr_n[v][j] > 0
            },
        }
        if have_baseline and v != args.baseline:
            report["variants"][v]["delta_vs_baseline"] = {
                "reduction": round(point_overall[args.baseline] - point_overall[v], 4),
                "ci": ci(boot_delta[v]),
            }

    out_path = Path(args.out) if args.out else pdir / "bootstrap_ci.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Human-readable summary.
    print(f"\n=== Bootstrap ASR CIs ({int(args.ci)}%, n_boot={args.n_boot}, "
          f"n_personas={n}) ===")
    print(f"baseline = {args.baseline}\n")
    print(f"{'variant':<12} {'ASR':>7}  {'95% CI':>17}   {'ΔASR vs base':>13}  {'Δ 95% CI':>17}")
    for v in variants:
        r = report["variants"][v]
        o = r["overall"]
        ci_s = f"[{o['ci'][0]:.3f}, {o['ci'][1]:.3f}]"
        if v == args.baseline or "delta_vs_baseline" not in r:
            print(f"{v:<12} {o['asr']:>7.3f}  {ci_s:>17}   {'—':>13}  {'—':>17}")
        else:
            d = r["delta_vs_baseline"]
            d_ci = f"[{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"
            sig = "" if (d["ci"][0] > 0 or d["ci"][1] < 0) else "  (CI spans 0)"
            print(f"{v:<12} {o['asr']:>7.3f}  {ci_s:>17}   {d['reduction']:>13.3f}  "
                  f"{d_ci:>17}{sig}")

    print("\nPer-attribute ASR [95% CI] (n):")
    header = "  ".join(f"{a:>16}" for a in attrs)
    print(f"{'variant':<12} {header}")
    for v in variants:
        pa = report["variants"][v]["per_attribute"]
        cells = []
        for a in attrs:
            if a in pa and pa[a]["asr"] is not None:
                c = pa[a]
                cells.append(f"{c['asr']:.2f}[{c['ci'][0]:.2f},{c['ci'][1]:.2f}]")
            else:
                cells.append("—")
        print(f"{v:<12} " + "  ".join(f"{c:>16}" for c in cells))

    print(f"\n[bootstrap] report -> {out_path}")


if __name__ == "__main__":
    main()
