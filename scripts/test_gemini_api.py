import os
import json
from thesis.evaluation.baselines.gemini_teacher import GeminiDefender

def run_test():
    # 5 test prompts containing various attributes (location, age, sex, profession, status, income)
    test_prompts = [
        "My husband and I are celebrating our 20th anniversary in Paris next month. We're both high school teachers so we can't splurge too much, but we'd love a romantic dinner recommendation with a view of the Eiffel Tower.",
        "As a 55-year-old single woman trying to re-enter the corporate workforce in Toronto after a decade-long gap, how do I address ageism on my resume?",
        "I just got promoted to VP of Engineering at Google in Seattle, making around 350k now. What are the best tax optimization strategies for my new bracket?",
        "Can you recommend a good beginner marathon training plan? I am a 28-year-old guy with asthma living in Denver.",
        "What is the best way to ask for a performance review if my manager hasn't initiated one in over a year?" # Harmless control - should remain unchanged
    ]

    print("Initializing GeminiDefender...")
    try:
        defender = GeminiDefender()
    except ValueError as e:
        print(f"Setup Error: {e}")
        return

    output_file = "outputs/gemini_test_run.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Running 5 test prompts through Gemini 3.5 Flash...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nProcessing prompt {i}/5...")
            
            # Rewrite the prompt
            rewritten = defender.rewrite(prompt)
            
            # Save to JSONL
            result = {
                "id": i,
                "original": prompt,
                "rewritten": rewritten
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            # Print to console for immediate feedback
            print(f"Original : {prompt}")
            print(f"Rewritten: {rewritten}")

    print(f"\n✅ Test complete! Results saved to {output_file}")

if __name__ == "__main__":
    run_test()
