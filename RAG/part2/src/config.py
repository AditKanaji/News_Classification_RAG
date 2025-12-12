import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY") # UPDATED: Using Guardian Key

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# --- NEW RAG CONFIGURATION ---
GENERATION_MODEL_NAME = 'gemini-2.0-flash' 
EMBEDDING_MODEL_NAME = 'models/text-embedding-004'
CHUNK_SIZE = 500
TOP_K_RETRIEVAL = 3

# --- BACKWARD COMPATIBILITY ---
MODEL_NAME = GENERATION_MODEL_NAME 

# Generation Config
GENERATION_CONFIG = {
    "temperature": 0.3, 
    "response_mime_type": "text/plain"
}