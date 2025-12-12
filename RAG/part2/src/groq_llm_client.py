from groq import Groq
from .groq_config import GROQ_API_KEY, GENERATION_MODEL_NAME

class LLMClientGroq:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GENERATION_MODEL_NAME

    def generate_text(self, prompt, system_instruction="You are a helpful assistant."):
        """
        Generates text using Groq (Llama 3).
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.3,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"   ❌ Groq API Error: {e}")
            return None