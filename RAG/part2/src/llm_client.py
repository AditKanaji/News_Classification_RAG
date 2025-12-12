import google.generativeai as genai
import json
import time
from .config import GOOGLE_API_KEY, MODEL_NAME, GENERATION_CONFIG

class LLMClient:
    def __init__(self):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(MODEL_NAME)

    def generate_json(self, prompt, retries=3):
        """
        Generates content ensuring JSON output. 
        Includes basic retry logic for stability.
        """
        for attempt in range(retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=GENERATION_CONFIG
                )
                
                # Parse JSON immediately to ensure validity
                return json.loads(response.text)
                
            except Exception as e:
                print(f"   [LLM Error] Attempt {attempt+1}/{retries}: {e}")
                time.sleep(2) # Backoff
                
        # Return empty structure on failure to prevent crash
        return None