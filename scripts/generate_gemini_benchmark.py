import os
import json
from datasets import load_dataset
from thesis.evaluation.baselines.gemini_teacher import GeminiDefender
from thesis.config import get_path, load_config

def main():
    print("Initializing Gemini...")
    defender = GeminiDefender()
    
    # Load the 200 held-out benchmark personas
    # (Adjust the path/split if your PersonaMem-v2 loader is slightly different)
    dataset = load_dataset("json", data_files="data/benchmark_200.jsonl")["train"]
    
    output_file = "outputs/eval/baselines/gemini_rewrites.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Starting generation for {len(dataset)} personas...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, persona in enumerate(dataset):
            print(f"Processing persona {idx + 1}/{len(dataset)}...")
            
            rewritten_turns = []
            # Assuming persona["conversation"] holds the turns
            for turn in persona["conversation"]:
                if turn["role"] == "user":
                    # Only rewrite the user's outgoing prompt
                    rewritten_text = defender.rewrite(turn["text"])
                    rewritten_turns.append({
                        "role": "user",
                        "text": rewritten_text,
                        "original_text": turn["text"]
                    })
                else:
                    # Keep the LLM's responses exactly as they are
                    rewritten_turns.append(turn)
            
            # Save the persona with the newly rewritten conversation
            new_persona = persona.copy()
            new_persona["conversation"] = rewritten_turns
            f.write(json.dumps(new_persona, ensure_ascii=False) + "\n")

    print(f"\n✅ All rewrites saved to {output_file}")

if __name__ == "__main__":
    main()
