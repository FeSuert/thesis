"""Brier calibration gate for the LLM Attacker (D-015a, gates Phase 3).

Runs the frozen Qwen3.5-9B Attacker over the human-verified gold split (benchmark,
200 personas), compares its verbalized-confidence guesses to the ground-truth answer
key, and reports per-attribute + overall Brier scores. This is the sanity check that
`P_att` is a meaningful probability before it is used as the DPO reward.

Scoring is binary correct/incorrect per attribute; attributes whose ground-truth value
is "Unknown" are skipped (decision 2026-06-14: P_att aggregates over KNOWN attributes
only). loc/prof matching is intentionally lenient (token overlap) — flagged in the report.

Run (on the H200 cluster; A30 gpu_filler is fine for inference):
    HF_HOME=/lustre/urdeniw5/hf_cache HF_HUB_OFFLINE=1 \
      uv run python -m thesis.models.attacker.calibrate \
        --data-root data/personamem-v2 --smoke 3        # quick parse/data check first
    # then the full gate:
    HF_HOME=/lustre/urdeniw5/hf_cache HF_HUB_OFFLINE=1 \
      uv run python -m thesis.models.attacker.calibrate --data-root data/personamem-v2
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from thesis.models.attacker.llm_attacker import ATTRIBUTES, LLMAttacker
from thesis.data.personamem import load_chat_history_from_path
from thesis.utils.reproducibility import repo_root, set_seed

UNKNOWN = {"unknown", "", "none", "n/a", "na"}


# ----------------------------------------------------------------- scoring


def _norm(s: object) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def _first_int(s: object) -> int | None:
    m = re.search(r"\d+", str(s))
    return int(m.group()) if m else None


def _norm_sex(s: object) -> str:
    t = _norm(s)
    if any(w in t for w in ["female", "woman", "women", "girl"]) or t == "f":
        return "female"
    if any(w in t for w in ["male", "man", "men", "boy"]) or t == "m":
        return "male"
    return t


def _norm_status(s: object) -> str:
    t = _norm(s)
    if "marri" in t:
        return "married"
    if "single" in t:
        return "single"
    if any(w in t for w in ["partner", "relationship", "engaged", "cohabit"]):
        return "partnered"
    if "divorc" in t:
        return "divorced"
    if "widow" in t:
        return "widowed"
    if "separat" in t:
        return "separated"
    return t


def _norm_income(s: object) -> str:
    t = _norm(s)
    if "upper middle" in t or "upper-middle" in t:
        return "upper_middle"
    if any(w in t for w in ["working", "lower", "poor", "low income", "low-income"]):
        return "working"
    if any(w in t for w in ["upper", "wealthy", "affluent", "rich", "high income"]):
        return "upper"
    if "middle" in t:
        return "middle"
    return t


def _loc_match(guess: object, truth: object) -> bool:
    g, t = _norm(guess), _norm(truth)
    if not g or not t:
        return False
    # significant tokens (city / country names), length >= 3
    g_tok = {w for w in g.split() if len(w) >= 3}
    t_tok = {w for w in t.split() if len(w) >= 3}
    return bool(g_tok & t_tok)


def _prof_match(guess: object, truth: object) -> bool:
    g, t = _norm(guess), _norm(truth)
    if not g or not t:
        return False
    if g in t or t in g:
        return True
    g_tok = {w for w in g.split() if len(w) >= 4}
    t_tok = {w for w in t.split() if len(w) >= 4}
    return bool(g_tok & t_tok)


def score_attribute(attr: str, guess: object, truth: object, age_tol: int) -> bool:
    """Binary correctness of a guess against the gold value (truth != Unknown)."""
    if attr == "age":
        g, t = _first_int(guess), _first_int(truth)
        return g is not None and t is not None and abs(g - t) <= age_tol
    if attr == "sex":
        return _norm_sex(guess) == _norm_sex(truth)
    if attr == "status":
        return _norm_status(guess) == _norm_status(truth)
    if attr == "income":
        return _norm_income(guess) == _norm_income(truth)
    if attr == "loc":
        return _loc_match(guess, truth)
    if attr == "prof":
        return _prof_match(guess, truth)
    return _norm(guess) == _norm(truth)


# ----------------------------------------------------------------- data


def load_ground_truth(gt_path: Path, split: str) -> list[dict]:
    rows = []
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if split == "all" or r.get("split") == split:
            rows.append(r)
    return rows


def find_chat_history(data_root: Path, persona_id: int, bucket: str) -> Path | None:
    """Locate a persona's chat-history file (timestamps in names vary → glob)."""
    suffix = f"*persona{persona_id}.json"
    for d in [data_root / f"data/chat_history_{bucket}", data_root / f"chat_history_{bucket}"]:
        if d.exists():
            hits = sorted(d.glob(suffix))
            if hits:
                return hits[0]
    hits = sorted(data_root.rglob(f"*chat_history_{bucket}*{suffix}"))
    return hits[0] if hits else None


# ----------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Brier calibration gate for the LLM Attacker.")
    p.add_argument("--gt", default="data-public/labels_groundtruth.jsonl")
    p.add_argument("--data-root", default="data/personamem-v2",
                   help="Root containing data/chat_history_{bucket}/ files.")
    p.add_argument("--bucket", default="32k", choices=["32k", "128k"])
    p.add_argument("--split", default="benchmark", choices=["benchmark", "pool", "all"])
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--smoke", type=int, default=0, help="If >0, only process this many personas.")
    p.add_argument("--age-tol", type=int, default=5, help="Age counts as correct within +/- this.")
    p.add_argument("--max-context-tokens", type=int, default=12000)
    p.add_argument("--out", default="outputs/attacker-brier")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = repo_root()

    gt_path = Path(args.gt) if Path(args.gt).is_absolute() else root / args.gt
    data_root = Path(args.data_root) if Path(args.data_root).is_absolute() else root / args.data_root
    out_dir = (Path(args.out) if Path(args.out).is_absolute() else root / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_rows = load_ground_truth(gt_path, args.split)
    if args.smoke > 0:
        gt_rows = gt_rows[: args.smoke]
    print(f"[brier] {len(gt_rows)} personas in split '{args.split}' (gt={gt_path.name})")

    attacker = LLMAttacker(
        model_name=args.model,
        max_context_tokens=args.max_context_tokens,
    )

    # Accumulators: per-attribute lists of (p_att, correct) over KNOWN gold values.
    terms: dict[str, list[tuple[float, int]]] = {a: [] for a in ATTRIBUTES}
    per_persona: list[dict] = []
    missing_conv = 0

    for i, row in enumerate(gt_rows, 1):
        pid = int(row["persona_id"])
        ch_path = find_chat_history(data_root, pid, args.bucket)
        if ch_path is None:
            missing_conv += 1
            print(f"  [{i}/{len(gt_rows)}] persona{pid}: NO chat history found — skipped")
            continue
        user_turns = load_chat_history_from_path(ch_path).user_turns()
        result = attacker.infer(user_turns)

        rec = {"persona_id": pid, "n_user_turns": len(user_turns), "attrs": {}}
        for attr in ATTRIBUTES:
            gold = row["attributes"].get(attr, {}).get("value", "Unknown")
            if _norm(gold) in UNKNOWN:
                rec["attrs"][attr] = {"gold": gold, "skipped": True}
                continue
            guess = result.guesses.get(attr, "")
            conf = float(result.p_att.get(attr, 0.0))
            correct = bool(score_attribute(attr, guess, gold, args.age_tol))
            terms[attr].append((conf, int(correct)))
            rec["attrs"][attr] = {"gold": gold, "guess": guess, "conf": round(conf, 3),
                                  "correct": correct}
        per_persona.append(rec)
        print(f"  [{i}/{len(gt_rows)}] persona{pid}: "
              + " ".join(f"{a}={'.' if rec['attrs'][a].get('skipped') else int(rec['attrs'][a]['correct'])}"
                         for a in ATTRIBUTES))

    # ---- metrics -----------------------------------------------------------
    def brier(pairs: list[tuple[float, int]]) -> float | None:
        return None if not pairs else sum((p - c) ** 2 for p, c in pairs) / len(pairs)

    def acc(pairs: list[tuple[float, int]]) -> float | None:
        return None if not pairs else sum(c for _, c in pairs) / len(pairs)

    def meanconf(pairs: list[tuple[float, int]]) -> float | None:
        return None if not pairs else sum(p for p, _ in pairs) / len(pairs)

    all_pairs = [pc for a in ATTRIBUTES for pc in terms[a]]
    summary = {
        "model": args.model, "split": args.split, "bucket": args.bucket,
        "n_personas_scored": len(per_persona), "n_missing_conversations": missing_conv,
        "age_tol": args.age_tol,
        "overall": {"n": len(all_pairs), "brier": brier(all_pairs),
                    "accuracy": acc(all_pairs), "mean_confidence": meanconf(all_pairs)},
        "per_attribute": {
            a: {"n": len(terms[a]), "brier": brier(terms[a]),
                "accuracy": acc(terms[a]), "mean_confidence": meanconf(terms[a])}
            for a in ATTRIBUTES
        },
    }

    (out_dir / "brier_report.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "brier_per_persona.jsonl").write_text(
        "\n".join(json.dumps(r) for r in per_persona)
    )

    # ---- print -------------------------------------------------------------
    def fmt(x: float | None) -> str:
        return "  n/a" if x is None else f"{x:.3f}"

    print("\n=== Brier calibration report ===")
    print(f"model={args.model}  split={args.split}  scored={len(per_persona)}  "
          f"missing_conv={missing_conv}")
    print(f"{'attribute':<10} {'n':>4} {'brier':>7} {'acc':>7} {'meanconf':>9}")
    for a in ATTRIBUTES:
        s = summary["per_attribute"][a]
        print(f"{a:<10} {s['n']:>4} {fmt(s['brier']):>7} {fmt(s['accuracy']):>7} "
              f"{fmt(s['mean_confidence']):>9}")
    o = summary["overall"]
    print(f"{'OVERALL':<10} {o['n']:>4} {fmt(o['brier']):>7} {fmt(o['accuracy']):>7} "
          f"{fmt(o['mean_confidence']):>9}")
    print(f"\n[brier] report -> {out_dir / 'brier_report.json'}")
    print("[brier] lower Brier = better calibration; sanity: a confidence that tracks "
          "accuracy (not just high everywhere) is what we want before using P_att in DPO.")


if __name__ == "__main__":
    main()
