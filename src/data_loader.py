import pandas as pd
import requests
import logging
from src.utils import get_api_key

logger = logging.getLogger(__name__)

def load_local_dataset(filepath):
    """Loads a local CSV dataset."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded dataset from {filepath} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def fetch_guardian_articles(query, page_size=20):
    """Fetches articles from The Guardian API."""
    api_key = get_api_key("GUARDIAN")
    if not api_key:
        return []

    url = "https://content.guardianapis.com/search"
    params = {
        'q': query,
        'api-key': api_key,
        'page-size': page_size,
        'show-fields': 'bodyText,headline,trailText'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        results = data.get('response', {}).get('results', [])
        
        articles = []
        for item in results:
            fields = item.get('fields', {})
            articles.append({
                'id': item.get('id'),
                'webTitle': item.get('webTitle'),
                'sectionName': item.get('sectionName'),
                'bodyText': fields.get('bodyText', ''),
                'trailText': fields.get('trailText', '')
            })
        
        logger.info(f"Fetched {len(articles)} articles for query '{query}'")
        return articles
    except Exception as e:
        logger.error(f"Error fetching from Guardian API: {e}")
        return []
