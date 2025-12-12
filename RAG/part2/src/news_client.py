import requests
from .config import GUARDIAN_API_KEY

class NewsClient:
    def __init__(self):
        if not GUARDIAN_API_KEY:
            print("⚠️ Warning: GUARDIAN_API_KEY not found in .env. RAG will fail.")
        self.base_url = "https://content.guardianapis.com/search"

    def fetch_articles(self, query, from_date=None, to_date=None, limit=5):
        """
        Fetches news articles from The Guardian API with date filtering.
        """
        print(f"   📡 Fetching Guardian news for: '{query}' ({from_date} to {to_date})...")
        
        # Guardian API specific parameters
        params = {
            'q': query,
            'api-key': GUARDIAN_API_KEY,
            'show-fields': 'bodyText,trailText', # Request full body text
            'page-size': limit,
            'order-by': 'relevance'
        }

        # Add date filters if provided (Format: YYYY-MM-DD)
        if from_date:
            params['from-date'] = from_date
        if to_date:
            params['to-date'] = to_date
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            # Guardian response structure is response -> results
            results = data.get('response', {}).get('results', [])
            
            for item in results:
                fields = item.get('fields', {})
                body_text = fields.get('bodyText') or fields.get('trailText') or ""
                
                if body_text:
                    articles.append({
                        'title': item.get('webTitle', 'No Title'),
                        'url': item.get('webUrl', ''),
                        'publishedAt': item.get('webPublicationDate', ''),
                        'content': body_text
                    })
            
            print(f"   ✅ Found {len(articles)} relevant Guardian articles.")
            return articles

        except Exception as e:
            print(f"   ❌ Error fetching news: {e}")
            return []