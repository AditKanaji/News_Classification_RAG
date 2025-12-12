import os
import random
import time
import re
from datetime import datetime, timedelta

# Import Groq specific modules
from src.groq_config import TOP_K_RETRIEVAL
from src.news_client import NewsClient # Reusing existing News Client
from src.data_loader import load_csv_data # Reusing existing Data Loader
from src.groq_pipeline import NuanceDetectorGroq

# Configuration
# INPUT_CSV = os.path.join("data", "stress_test", "input", "FakeData.csv")
INPUT_CSV = os.path.join("data", "stress_test", "input", "ContradictData.csv")
# INPUT_CSV = os.path.join("data", "stress_test", "input", "TrueData.csv")
# INPUT_CSV = os.path.join("data", "stress_test", "input", "original.csv")
SAMPLE_SIZE = 10
DATE_WINDOW_DAYS = 3

def clean_search_query(text):
    if not text: return ""
    cleaned = re.sub(r'[^\w\s]', ' ', text)
    return " ".join(cleaned.split())

def calculate_date_window(date_str):
    if not date_str or str(date_str).lower() == 'unknown':
        return None, None
    try:
        clean_date = str(date_str).split('T')[0].strip()[:10]
        dt = datetime.strptime(clean_date, "%Y-%m-%d")
        start_date = (dt - timedelta(days=DATE_WINDOW_DAYS)).strftime("%Y-%m-%d")
        end_date = (dt + timedelta(days=DATE_WINDOW_DAYS)).strftime("%Y-%m-%d")
        return start_date, end_date
    except ValueError:
        return None, None

def main():
    print("--- 🚀 Automated Stress Test (Groq + HuggingFace) ---")
    
    # 1. Load Data
    all_articles = load_csv_data(INPUT_CSV)
    if not all_articles: return

    # 2. Random Selection
    selected_articles = random.sample(all_articles, min(SAMPLE_SIZE, len(all_articles)))
    print(f"🎲 Selected {len(selected_articles)} random articles.\n")

    # 3. Initialize Engines
    news_client = NewsClient()
    detector = NuanceDetectorGroq()

    # 4. Processing Loop
    for i, article in enumerate(selected_articles):
        raw_title = article.get('title', 'Unknown')
        print(f"\n[{i+1}/{len(selected_articles)}] Verifying: '{raw_title}'")
        
        # A. Date Window
        start_date, end_date = calculate_date_window(article.get('published_date'))
        
        # B. Fetch Evidence
        query = clean_search_query(raw_title)
        real_news = news_client.fetch_articles(query, from_date=start_date, to_date=end_date, limit=3)
        
        if not real_news:
            print("   ⚠️ No matching real news found. Skipping.")
            continue

        # C. Ingest & Retrieve (RAG)
        detector.rag.ingest_articles(real_news)
        query_content = article.get('bodyText', '')[:500]
        context_chunks = detector.rag.retrieve(query_content, top_k=TOP_K_RETRIEVAL)
        
        # D. LLM Judge
        print("   ⚖️  Groq (Llama 3) Judging...")
        verdict = detector.judge_article(article, context_chunks)
        print(f"\n{verdict}\n")
        
        time.sleep(1)

if __name__ == "__main__":
    main()