import os
import logging
from dotenv import load_dotenv

load_dotenv()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_api_key(service_name):
    key = os.getenv(f"{service_name.upper()}_API_KEY")
    if not key:
        logging.warning(f"{service_name.upper()}_API_KEY not found in environment variables.")
    return key
