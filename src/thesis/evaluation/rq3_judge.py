"""RQ3 step 2 — score response pairs: embedding similarity + LLM-as-judge.

Reads the response pairs from rq3_responses.py and produces the two RQ3 metrics:
  1. Response semantic alignment: cosine similarity between the response to the
     original prompt and the response to the rewritten prompt (all-mpnet-base-v2).
  2. Helpfulness comparison via LLM-as-judge: the judge sees the user's ORIGINAL
     intent and two responses (A/B order randomized, provenance hidden) and rates
     whether each response serves that intent, 1-5, plus a verdict.

Judge bias controls: the judge never learns which response came from the rewritten
prompt, and A/B assignment is randomized per item (recorded for de-randomization).

Runs on the login node or a laptop (needs internet for the judge API; the embedding
model is small enough for CPU). The API key is passed at runtime, never stored:
  uv run python -m thesis.evaluation.rq3_judge \
    --responses outputs/eval/rq3/responses_sftdpo_v2.jsonl \
    --base-url https://api.openai.com/v1 --judge-model gpt-5 \
    --out outputs/eval/rq3/judged_sftdpo_v2
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np

from thesis.models.evaluator.similarity import SemanticSimilarity
from thesis.utils.api_client import ChatClient, resolve_api_key
from thesis.utils.reproducibility import repo_root, set_seed

JUDGE_SYSTEM = (
    "You are an impartial evaluator of AI assistant responses. You are given a user's "
    "request and two responses to it, labeled A and B. Judge how well EACH response "
    "serves the user's request: helpfulness, relevance, and completeness. Judge only "
    "quality with respect to the request — ignore style, length, and formatting "
    "differences unless they affect usefulness.\n\n"
    "Respond with ONLY a JSON object in exactly this shape:\n"
    '{"score_a": <1-5>, "score_b": <1-5>, "verdict": "<A|B|tie>", '
    '"reason": "<one short sentence>"}\n'
    "Scores: 5 = fully answers the request, 3 = partially useful, 1 = useless or "
    "off-target. verdict = the more helpful response, or tie."
)

JUDGE_USER_TEMPLATE = (
    "User request:\n{request}\n\n"
    "Response A:\n{resp_a}\n\n"
    "Response B:\n{resp_b}\n\n"
    "Which response serves the user's request better?"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ3: judge response pairs.")
    p.add_argument("--responses", required=True,
                   help="JSONL from rq3_responses.py.")
    p.add_argument("--base-url", default="https://api.openai.com/v1")
    p.add_argument("--judge-model", default="gpt-5")
    p.add_argument("--api-key", default=None,
                   help="Judge API key. If omitted, read from OPENAI_API_KEY / "
                        "MOONSHOT_API_KEY env or prompted interactively.")
    p.add_argument("--evaluator", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--limit", type=int, default=0, help="Cap items (0 = all).")
    p.add_argument("--skip-identical", action="store_true", default=True,
                   help="Identical responses are scored as ties without a judge call.")
    p.add_argument("--out", default="outputs/eval/rq3/judged")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _parse_judge(raw: str) -> dict | None:
    candidates = re.findall(r"\{.*\}", raw, flags=re.DOTALL)
    for cand in reversed(candidates):
        try:
            d = json.loads(cand)
            if "score_a" in d and "score_b" in d and "verdict" in d:
                return d
        except json.JSONDecodeError:
            continue
    return None


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)
    root = repo_root()

    resp_path = Path(args.responses) if Path(args.responses).is_absolute() else root / args.responses
    out_dir = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    items = [json.loads(line) for line in resp_path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if args.limit > 0:
        items = items[: args.limit]
    print(f"[rq3-judge] {len(items)} response pairs; judge={args.judge_model}")

    evaluator = SemanticSimilarity(model_name=args.evaluator)

    judged_path = out_dir / "judged.jsonl"
    done: set[tuple[int, int]] = set()
    if judged_path.exists():
        for line in judged_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["persona_id"], r["turn_idx"]))
        print(f"[rq3-judge] resuming: {len(done)} already judged")

    client: ChatClient | None = None  # created lazily so ties-only runs need no key
    records: list[dict] = []

    with judged_path.open("a", encoding="utf-8") as f:
        for i, it in enumerate(items, 1):
            key = (it["persona_id"], it["turn_idx"])
            if key in done:
                continue

            r_orig, r_rw = it["response_original"], it["response_rewrite"]
            resp_sim = evaluator.cosine(r_orig, r_rw)

            identical = r_orig.strip() == r_rw.strip()
            if identical and args.skip_identical:
                rec = {
                    "persona_id": it["persona_id"], "turn_idx": it["turn_idx"],
                    "changed": it.get("changed"), "s_sem_prompt": it.get("s_sem_prompt"),
                    "response_similarity": round(resp_sim, 4),
                    "judge": {"score_original": 5, "score_rewrite": 5, "verdict": "tie",
                              "reason": "identical responses", "skipped_call": True},
                }
            else:
                if client is None:
                    api_key = resolve_api_key(args.api_key)
                    client = ChatClient(base_url=args.base_url, model=args.judge_model,
                                        api_key=api_key)
                # Randomize which side is the original so the judge can't learn a
                # positional pattern.
                orig_is_a = rng.random() < 0.5
                resp_a, resp_b = (r_orig, r_rw) if orig_is_a else (r_rw, r_orig)
                raw = client.chat([
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                        request=it["original"], resp_a=resp_a, resp_b=resp_b)},
                ], temperature=0.0, max_tokens=256)
                parsed = _parse_judge(raw)
                if parsed is None:
                    rec_judge = {"parse_failed": True, "raw": raw[:300]}
                else:
                    sa, sb = float(parsed["score_a"]), float(parsed["score_b"])
                    verdict = str(parsed["verdict"]).strip().upper()
                    # De-randomize back to original/rewrite terms.
                    score_orig, score_rw = (sa, sb) if orig_is_a else (sb, sa)
                    if verdict == "TIE":
                        v = "tie"
                    elif (verdict == "A") == orig_is_a:
                        v = "original"
                    else:
                        v = "rewrite"
                    rec_judge = {"score_original": score_orig, "score_rewrite": score_rw,
                                 "verdict": v, "reason": parsed.get("reason", ""),
                                 "orig_was_a": orig_is_a}
                rec = {
                    "persona_id": it["persona_id"], "turn_idx": it["turn_idx"],
                    "changed": it.get("changed"), "s_sem_prompt": it.get("s_sem_prompt"),
                    "response_similarity": round(resp_sim, 4),
                    "judge": rec_judge,
                }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            records.append(rec)
            if i % 25 == 0 or i == len(items):
                print(f"    {i}/{len(items)} judged")

    # Reload everything (including previously-done rows) for the summary.
    records = [json.loads(line) for line in judged_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]

    sims = np.array([r["response_similarity"] for r in records])
    scored = [r for r in records if "score_original" in r.get("judge", {})]
    s_orig = np.array([r["judge"]["score_original"] for r in scored])
    s_rw = np.array([r["judge"]["score_rewrite"] for r in scored])
    verdicts = [r["judge"]["verdict"] for r in scored]
    changed = [r for r in records if r.get("changed")]

    summary = {
        "n_items": len(records),
        "n_changed_prompts": len(changed),
        "judge_model": args.judge_model,
        "response_similarity": {
            "mean": round(float(sims.mean()), 4) if len(sims) else None,
            "median": round(float(np.median(sims)), 4) if len(sims) else None,
            "p5": round(float(np.percentile(sims, 5)), 4) if len(sims) else None,
            "min": round(float(sims.min()), 4) if len(sims) else None,
        },
        "judge_scores": {
            "n_scored": len(scored),
            "mean_original": round(float(s_orig.mean()), 3) if len(s_orig) else None,
            "mean_rewrite": round(float(s_rw.mean()), 3) if len(s_rw) else None,
            "verdict_counts": {v: verdicts.count(v) for v in ("original", "rewrite", "tie")},
            "n_parse_failed": sum(1 for r in records
                                  if r.get("judge", {}).get("parse_failed")),
        },
    }
    (out_dir / "rq3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== RQ3 summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\n[rq3-judge] per-item -> {judged_path}")
    print(f"[rq3-judge] summary  -> {out_dir / 'rq3_summary.json'}")


if __name__ == "__main__":
    main()
