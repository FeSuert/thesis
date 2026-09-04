"""Generate Gemini Flash rewrites for the 200 benchmark personas."""

from __future__ import annotations

import json
from pathlib import Path

from thesis.evaluation.baselines.gemini_teacher import GeminiDefender
from thesis.models.attacker.calibrate import (
    UNKNOWN,
    _norm,
    find_chat_history,
    load_ground_truth,
)
from thesis.models.attacker.llm_attacker import ATTRIBUTES
from thesis.data.personamem import load_chat_history_from_path
from thesis.utils.reproducibility import repo_root


def load_benchmark_personas(root: Path, split: str = "benchmark", max_turns: int = 15, min_turn_chars: int = 40):
    gt_path = root / "data-public/labels_groundtruth.jsonl"
    data_root = root / "data/personamem-v2"
    
    rows = load_ground_truth(gt_path, split)
    personas = []
    for row in rows:
        pid = int(row["persona_id"])
        gold_known = {
            a: row["attributes"][a]["value"]
            for a in ATTRIBUTES
            if _norm(row["attributes"].get(a, {}).get("value", "Unknown")) not in UNKNOWN
        }
        ch = find_chat_history(data_root, pid, "32k")
        if ch is None or not gold_known:
            continue
        turns = [
            t for t in load_chat_history_from_path(ch).user_turns()
            if len(t.strip()) >= min_turn_chars
        ][:max_turns]
        if turns:
            personas.append({"pid": pid, "turns": turns})
    return personas


def main() -> None:
    root = repo_root()
    out_dir = root / "outputs/eval/v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "turns_gemini.jsonl"

    print("Initializing Gemini baseline...")
    defender = GeminiDefender()

    undef_file = out_dir / "turns_undefended.jsonl"

    if undef_file.exists():
        print(f"Reading existing turns directly from {undef_file}...")
        with undef_file.open("r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        
        print(f"Found {len(lines)} turns across benchmark personas. Generating Gemini rewrites...")
        with out_file.open("w", encoding="utf-8") as out_f:
            for idx, item in enumerate(lines, 1):
                orig = item["original"]
                rw = defender.rewrite(orig)
                
                record = {
                    "persona_id": item["persona_id"],
                    "turn_idx": item["turn_idx"],
                    "original": orig,
                    "rewrite": rw,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if idx % 50 == 0 or idx == len(lines):
                    print(f"  {idx}/{len(lines)} turns rewritten")
    else:
        print("turns_undefended.jsonl not found; loading benchmark from ground truth...")
        personas = load_benchmark_personas(root)
        print(f"Loaded {len(personas)} personas.")
        
        with out_file.open("w", encoding="utf-8") as out_f:
            total_turns = sum(len(p["turns"]) for p in personas)
            count = 0
            for per in personas:
                for ti, orig in enumerate(per["turns"]):
                    rw = defender.rewrite(orig)
                    record = {
                        "persona_id": per["pid"],
                        "turn_idx": ti,
                        "original": orig,
                        "rewrite": rw,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 50 == 0 or count == total_turns:
                        print(f"  {count}/{total_turns} turns rewritten")

    print(f"\nAll rewrites successfully saved to {out_file}")


if __name__ == "__main__":
    main()
