import os
import time
from google import genai
from google.genai import types

class GeminiDefender:
    def __init__(self, model_name: str = "gemini-3.5-flash") -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please add it to your .env file or export it.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.system_instruction = (
                "You are an automated privacy-preserving text rewriter. Your ONLY job is to output the final rewritten text. "
                "Rule 1: Generalize the specific. Replace sensitive details (location, profession, age, sex, relationship status, income) with broader equivalents "
                "(e.g., 'in Berlin' -> 'in a city', 'my husband' -> 'my partner', 'as a senior developer' -> 'as a professional'). "
                "Rule 2: Do NOT use bracket masking like <LOCATION>. Keep the text fluent and natural. "
                "Rule 3: Keep the original meaning and intent of the request intact so it remains useful. "
                "Rule 4: If the message contains no sensitive information, output the exact original text. "
                "CRITICAL: Output ONLY the final rewritten text. Do NOT include preambles like 'Here is the rewrite:'. "
                "Do NOT include explanations, reasoning, or lists of changes. Start immediately with the rewritten text."
            )

    def rewrite(self, text: str) -> str:
        if not text or not text.strip():
            return text
            
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=text,
                config=types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                        temperature=0.1,  # Bumped from 0.0 to prevent truncation bugs
                        max_output_tokens=1024,
                        safety_settings=[
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                        ]
                    )
            )
            if response.text:
                return response.text.strip()
            return text
        except Exception as e:
            if "429" in str(e):
                print("Rate limit hit. Waiting 5 seconds...")
                time.sleep(5)
                try:
                    return self.client.models.generate_content(
                        model=self.model_name,
                        contents=text,
                        config=types.GenerateContentConfig(
                            system_instruction=self.system_instruction,
                            temperature=0.1,
                            max_output_tokens=1024,
                            safety_settings=[
                                types.SafetySetting(
                                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                                ),
                                types.SafetySetting(
                                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                                ),
                                types.SafetySetting(
                                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                                ),
                                types.SafetySetting(
                                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                                ),
                            ]
                        )
                    ).text.strip()
                except:
                    pass
            print(f"Error calling Gemini API: {e}. Returning original text.")
            return text
