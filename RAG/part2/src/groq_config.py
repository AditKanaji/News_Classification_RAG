import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

# Model Configuration
GENERATION_MODEL_NAME = 'llama-3.1-8b-instant' 
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Runs locally via HuggingFace

# RAG Settings
CHUNK_SIZE = 500
TOP_K_RETRIEVAL = 3