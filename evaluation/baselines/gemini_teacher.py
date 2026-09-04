"""Gemini Flash baseline — Cloud teacher rewrite.

Rewrites the user prompt using Gemini 3.5 Flash to evaluate whether the 
small local student (Qwen3.5-4B SFT+DPO) matches its teacher.
Uses the "generalize the specific" instruction format from Phase 1.

Requires the Google Generative AI SDK:
    uv run pip install google-generativeai
"""

from __future__ import annotations

import os
import time
import google.generativeai as genai
from google.generativeai.types import generation_types

class GeminiDefender:
    def __init__(self, model_name: str = "gemini-3.5-flash") -> None:
        """Initializes the Gemini API client and the rewriting model."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please add it to your .env file.")
        
        genai.configure(api_key=api_key)
        
        # This matches the Phase 1 "generalize the specific" SFT generation style
        self.system_instruction = (
            "You are a privacy-preserving text rewriter. Your task is to rewrite the user's message "
            "to hide sensitive personal attributes (location, profession, age, sex, relationship status, income). "
            "Rule 1: Generalize the specific. Replace sensitive details with broader equivalents "
            "(e.g., 'in Berlin' -> 'in a city', 'my husband' -> 'my partner', 'as a senior developer' -> 'as a professional'). "
            "Rule 2: Do NOT use bracket masking like <LOCATION> or [REDACTED]. Keep the text fluent and natural. "
            "Rule 3: Keep the original meaning and intent of the request intact so it remains useful. "
            "Rule 4: If the message contains no sensitive information, return it exactly unchanged."
        )
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=self.system_instruction,
            generation_config=genai.GenerationConfig(
                temperature=0.0,  # Strict, deterministic rewriting
                max_output_tokens=1024,
            )
        )

    def rewrite(self, text: str) -> str:
        """Rewrites a single message to generalize sensitive attributes."""
        # Fast exit for empty strings
        if not text or not text.strip():
            return text
            
        try:
            response = self.model.generate_content(text)
            # Handle cases where the API blocks output due to safety filters
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"Warning: Gemini blocked prompt due to safety filters. Returning original. Reason: {response.prompt_feedback.block_reason}")
                return text
                
            if response.text:
                return response.text.strip()
            return text

        except generation_types.StopCandidateException:
            print("Warning: Gemini stopped generation unexpectedly. Returning original.")
            return text
        except Exception as e:
            # Catch rate limits or network errors, wait briefly, and fail open (return original)
            if "429" in str(e):
                print("Rate limit hit. Waiting 5 seconds...")
                time.sleep(5)
                # Simple single retry
                try:
                    return self.model.generate_content(text).text.strip()
                except:
                    pass
            print(f"Error calling Gemini API: {e}. Returning original text.")
            return text
