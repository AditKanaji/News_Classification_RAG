import google.generativeai as genai
from src.utils import get_api_key
import os

def list_models():
    api_key = get_api_key("GOOGLE")
    if not api_key:
        print("No Google API Key found.")
        return

    genai.configure(api_key=api_key)
    
    print("Listing available models:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Name: {m.name}")

if __name__ == "__main__":
    list_models()
