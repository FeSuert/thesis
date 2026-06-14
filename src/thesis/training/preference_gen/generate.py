"""Phase 3 preference generation (Algorithm 1 Phase 1; D-001, D-002, D-019, D-021).

For each user turn of a pool persona:
  1. The memoryless Defender (merged Qwen3.5 SFT) samples k candidate rewrites (D-001).
  2. The Evaluator scores S_sem(original, candidate) (D-016).
  3. The Attacker reads the SANITIZED history so far + the candidate (user turns only,
     D-002) and returns per-attribute verbalized confidence P_att (D-019). The leak for a
     known attribute = confidence IF the guess is correct vs ground truth, else 0; P_att is
     the MEAN over the persona's KNOWN attributes (decision 2026-06-14). This ties the reward
     to actual correct inference (ASR), not raw confidence.
  4. Reward R = lambda*S_sem - (1-lambda)*P_att. Candidates with S_sem < tau are dropped
     (utility floor, anti-reward-hacking, proposal-analysis 3.5).
  5. chosen = argmax R, rejected = argmin R; the pair is kept only if the margin > delta
     (proposal-analysis 3.6). The sanitized history is then extended with the chosen rewrite
     (D-002: history = earlier user turns only).

Output: conversational DPO pairs {prompt, chosen, rejected, meta} as JSONL, plus stats.

Run (fat_gpu — needs the 4B Defender + 9B Attacker + mpnet resident together):
    HF_HOME=/lustre/urdeniw5/hf_cache HF_HUB_OFFLINE=1 \
      uv run python -m thesis.training.preference_gen.generate \
        --defender outputs/sft-qwen3_5-4b/merged --data-root data/personamem-v2 \
        --sessions 12 --max-turns 6 --k 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis.data.personamem import load_chat_history_from_path
from thesis.models.attacker.calibrate import (
    UNKNOWN,
    _norm,
    find_chat_history,
    load_ground_truth,
    score_attribute,
)
from thesis.models.attacker.llm_attacker import ATTRIBUTES, LLMAttacker
from thesis.models.defender.defender import DEFENDER_SYSTEM, Defender
from thesis.models.evaluator.similarity import SemanticSimilarity
from thesis.utils.reproducibility import repo_root, set_seed


def aggregate_p_att(attack, gold_known: dict[str, str], age_tol: int) -> tuple[float, dict]:
    """Mean over KNOWN attributes of (confidence if the guess is correct, else 0)."""
    leaks: dict[str, float] = {}
    for attr, gval in gold_known.items():
        guess = attack.guesses.get(attr, "")
        conf = float(attack.p_att.get(attr, 0.0))
        correct = score_attribute(attr, guess, gval, age_tol)
        leaks[attr] = conf if correct else 0.0
    agg = sum(leaks.values()) / len(leaks) if leaks else 0.0
    return agg, leaks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 preference generation (pilot/scale).")
    p.add_argument("--defender", default="outputs/sft-qwen3_5-4b/merged")
    p.add_argument("--attacker", default="Qwen/Qwen3.5-9B")
    p.add_argument("--evaluator", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--gt", default="data-public/labels_groundtruth.jsonl")
    p.add_argument("--data-root", default="data/personamem-v2")
    p.add_argument("--bucket", default="32k", choices=["32k", "128k"])
    p.add_argument("--split", default="pool", choices=["pool", "benchmark", "all"])
    p.add_argument("--sessions", type=int, default=12, help="Number of personas to process.")
    p.add_argument("--max-turns", type=int, default=6, help="User turns per session to process.")
    p.add_argument("--min-turn-chars", type=int, default=40,
                   help="Skip trivial/padding user turns shorter than this.")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--lambda", dest="lam", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.6, help="Utility floor on S_sem.")
    p.add_argument("--delta", type=float, default=0.05, help="Min reward margin to keep a pair.")
    p.add_argument("--age-tol", type=int, default=5)
    p.add_argument("--max-context-tokens", type=int, default=4096)
    p.add_argument("--out", default="outputs/dpo-pref/pilot")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = repo_root()

    gt_path = Path(args.gt) if Path(args.gt).is_absolute() else root / args.gt
    data_root = Path(args.data_root) if Path(args.data_root).is_absolute() else root / args.data_root
    defender_dir = Path(args.defender) if Path(args.defender).is_absolute() else root / args.defender
    out_dir = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_rows = load_ground_truth(gt_path, args.split)[: args.sessions]
    print(f"[pref] {len(gt_rows)} personas (split={args.split}); k={args.k} "
          f"lambda={args.lam} tau={args.tau} delta={args.delta}")

    defender = Defender(str(defender_dir))
    attacker = LLMAttacker(model_name=args.attacker, max_context_tokens=args.max_context_tokens)
    evaluator = SemanticSimilarity(model_name=args.evaluator)

    pairs: list[dict] = []
    stats = {"turns_seen": 0, "turns_paired": 0, "skip_low_diversity": 0,
             "skip_utility_floor": 0, "skip_margin": 0, "skip_no_known_attrs": 0}

    pairs_f = (out_dir / "pref_pairs.jsonl").open("w", encoding="utf-8")

    for si, row in enumerate(gt_rows, 1):
        pid = int(row["persona_id"])
        gold_known = {a: row["attributes"][a]["value"]
                      for a in ATTRIBUTES
                      if _norm(row["attributes"].get(a, {}).get("value", "Unknown")) not in UNKNOWN}
        ch_path = find_chat_history(data_root, pid, args.bucket)
        if ch_path is None:
            print(f"  [{si}/{len(gt_rows)}] persona{pid}: no chat history — skipped")
            continue
        if not gold_known:
            stats["skip_no_known_attrs"] += 1
            print(f"  [{si}/{len(gt_rows)}] persona{pid}: no known gold attributes — skipped")
            continue

        user_turns = [t for t in load_chat_history_from_path(ch_path).user_turns()
                      if len(t.strip()) >= args.min_turn_chars]
        history: list[str] = []           # sanitized history C (user turns only, D-002)
        n_paired_here = 0

        for ti, u_t in enumerate(user_turns[: args.max_turns]):
            stats["turns_seen"] += 1
            cands = list(dict.fromkeys(c for c in defender.candidates(
                u_t, k=args.k, temperature=args.temperature) if c.strip()))
            if len(cands) < 2:
                stats["skip_low_diversity"] += 1
                continue

            sims = evaluator.cosine_batch(u_t, cands)
            scored = []
            for c, s in zip(cands, sims):
                if s < args.tau:           # utility floor (anti reward-hacking)
                    continue
                attack = attacker.infer(history + [c])
                p_agg, leaks = aggregate_p_att(attack, gold_known, args.age_tol)
                reward = args.lam * s - (1.0 - args.lam) * p_agg
                scored.append({"text": c, "s_sem": s, "p_att": p_agg,
                               "reward": reward, "leaks": leaks})
            if len(scored) < 2:
                stats["skip_utility_floor"] += 1
                continue

            scored.sort(key=lambda x: x["reward"])
            rejected, chosen = scored[0], scored[-1]
            history.append(chosen["text"])     # extend sanitized history with the best rewrite
            margin = chosen["reward"] - rejected["reward"]
            if margin <= args.delta:
                stats["skip_margin"] += 1
                continue

            pair = {
                "prompt": [{"role": "system", "content": DEFENDER_SYSTEM},
                           {"role": "user", "content": u_t}],
                "chosen": [{"role": "assistant", "content": chosen["text"]}],
                "rejected": [{"role": "assistant", "content": rejected["text"]}],
                "meta": {
                    "persona_id": pid, "turn_idx": ti, "context_turns": len(history) - 1,
                    "known_attrs": list(gold_known.keys()),
                    "reward_chosen": round(chosen["reward"], 4),
                    "reward_rejected": round(rejected["reward"], 4),
                    "margin": round(margin, 4),
                    "s_sem_chosen": round(chosen["s_sem"], 4),
                    "p_att_chosen": round(chosen["p_att"], 4),
                    "s_sem_rejected": round(rejected["s_sem"], 4),
                    "p_att_rejected": round(rejected["p_att"], 4),
                },
            }
            pairs_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            pairs_f.flush()
            pairs.append(pair)
            stats["turns_paired"] += 1
            n_paired_here += 1

        print(f"  [{si}/{len(gt_rows)}] persona{pid}: {n_paired_here} pairs "
              f"(known={list(gold_known.keys())})")

    pairs_f.close()

    # ---- summary -----------------------------------------------------------
    def _mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    summary = {
        "n_pairs": len(pairs),
        **stats,
        "mean_s_sem_chosen": _mean([p["meta"]["s_sem_chosen"] for p in pairs]),
        "mean_s_sem_rejected": _mean([p["meta"]["s_sem_rejected"] for p in pairs]),
        "mean_p_att_chosen": _mean([p["meta"]["p_att_chosen"] for p in pairs]),
        "mean_p_att_rejected": _mean([p["meta"]["p_att_rejected"] for p in pairs]),
        "mean_margin": _mean([p["meta"]["margin"] for p in pairs]),
        "config": {"k": args.k, "lambda": args.lam, "tau": args.tau, "delta": args.delta,
                   "temperature": args.temperature, "defender": str(defender_dir),
                   "attacker": args.attacker},
    }
    (out_dir / "stats.json").write_text(json.dumps(summary, indent=2))

    print("\n=== preference-gen summary ===")
    for kk, vv in summary.items():
        if kk != "config":
            print(f"  {kk}: {vv}")
    print(f"\n[pref] pairs -> {out_dir / 'pref_pairs.jsonl'}")
    print("[pref] reward-hacking check: chosen should keep HIGH S_sem (not collapse to generic "
          "mush) while having LOWER P_att than rejected. Eyeball a few pairs below:")
    for p in pairs[:4]:
        print(f"\n  --- persona{p['meta']['persona_id']} turn{p['meta']['turn_idx']} "
              f"(margin {p['meta']['margin']}) ---")
        print(f"  PROMPT   : {p['prompt'][1]['content'][:160]}")
        print(f"  CHOSEN   : {p['chosen'][0]['content'][:160]}  "
              f"[S_sem={p['meta']['s_sem_chosen']} P_att={p['meta']['p_att_chosen']}]")
        print(f"  REJECTED : {p['rejected'][0]['content'][:160]}  "
              f"[S_sem={p['meta']['s_sem_rejected']} P_att={p['meta']['p_att_rejected']}]")


if __name__ == "__main__":
    main()
