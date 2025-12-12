import os
import sys
import random
import time
import re
from datetime import datetime, timedelta
import google.generativeai as genai

# Import local modules
from src.config import GOOGLE_API_KEY, GENERATION_MODEL_NAME, TOP_K_RETRIEVAL
from src.news_client import NewsClient
from src.rag_engine import LightweightRAGEngine
from src.data_loader import load_csv_data

# Configuration
# INPUT_CSV = os.path.join("data", "stress_test", "input", "TrueData.csv")
# INPUT_CSV = os.path.join("data", "stress_test", "input", "FakeData.csv")
INPUT_CSV = os.path.join("data", "stress_test", "input", "ContradictData.csv")
# INPUT_CSV = os.path.join("data", "stress_test", "input", "original.csv") #different words
SAMPLE_SIZE = 10 # Number of random articles to test
DATE_WINDOW_DAYS = 3 # Look for news +/- 3 days around the article date

# Initialize AI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(GENERATION_MODEL_NAME)

def clean_search_query(text):
    """
    Removes special characters/punctuation that might break the News API.
    """
    if not text:
        return ""
    # Remove anything that isn't a word character or whitespace
    # This strips smart quotes, colons, dashes, etc.
    cleaned = re.sub(r'[^\w\s]', ' ', text)
    # Collapse multiple spaces into one
    return " ".join(cleaned.split())

def calculate_date_window(date_str):
    """
    Parses date strings (handles 'YYYY-MM-DD' or ISO 'YYYY-MM-DDTHH:MM:SSZ')
    and returns (start_date, end_date) strings covering a +/- window.
    """
    if not date_str or str(date_str).lower() == 'unknown':
        return None, None

    try:
        # Clean ISO format (e.g., 2025-11-21T14:08:42Z -> 2025-11-21)
        # Taking the first 10 chars is usually safe for YYYY-MM-DD
        clean_date = str(date_str).split('T')[0].strip()[:10]
        
        # Attempt to parse date
        dt = datetime.strptime(clean_date, "%Y-%m-%d")
        start_date = (dt - timedelta(days=DATE_WINDOW_DAYS)).strftime("%Y-%m-%d")
        end_date = (dt + timedelta(days=DATE_WINDOW_DAYS)).strftime("%Y-%m-%d")
        return start_date, end_date
    except ValueError:
        print(f"   ⚠️ Could not parse date: {date_str}. Searching without date filter.")
        return None, None

def judge_article_accuracy(stress_article, evidence_chunks):
    """
    Uses LLM to compare the Stress Article against Real Evidence.
    """
    evidence_text = "\n\n".join([f"REAL NEWS ({c['source']} - {c['url']}): {c['text']}" for c in evidence_chunks])
    
    prompt = f"""
    You are a Fact-Checking Judge. 
    Compare the "SUSPICIOUS ARTICLE" against the "VERIFIED REAL NEWS" retrieved from the same time period.

    SUSPICIOUS ARTICLE:
    Title: {stress_article.get('title')}
    Date: {stress_article.get('published_date')}
    Content: {stress_article.get('bodyText')[:1000]}...

    VERIFIED REAL NEWS (Ground Truth):
    {evidence_text}

    TASK:
    1. Does the suspicious article contradict the real news?
    2. Does it invent events that didn't happen?
    3. Does it twist real facts (e.g. correct stats, wrong conclusion)?

    OUTPUT (Plain Text):
    - Verdict: [SAFE / MISLEADING / FABRICATED]
    - Confidence Score: [0-100]%
    - Explanation: (Briefly explain the discrepancy or confirmation)
    """
    
    response = model.generate_content(prompt)
    return response.text

def main():
    print("--- 📉 Automated Random Stress Test (RAG-Based) ---")
    
    # 1. Load Data
    all_articles = load_csv_data(INPUT_CSV)
    if not all_articles:
        return

    # 2. Random Selection
    selected_articles = random.sample(all_articles, min(SAMPLE_SIZE, len(all_articles)))
    print(f"🎲 Selected {len(selected_articles)} random articles for deep verification.\n")

    # 3. Initialize Engines
    news_client = NewsClient()
    rag_engine = LightweightRAGEngine()

    # 4. Processing Loop
    for i, article in enumerate(selected_articles):
        raw_title = article.get('title', 'Unknown')
        print(f"\n[{i+1}/{len(selected_articles)}] Verifying: '{raw_title}'")
        
        # A. Determine Date Window
        pub_date = article.get('published_date')
        start_date, end_date = calculate_date_window(pub_date)
        
        # B. Fetch Real News (Evidence)
        # We query the cleaned title to find corroborating reports
        # UPDATE: Clean the query to avoid 400 Bad Request
        query = clean_search_query(raw_title)
        
        real_news = news_client.fetch_articles(query, from_date=start_date, to_date=end_date, limit=3)
        
        if not real_news:
            print("   ⚠️ No matching real news found in this date window. Skipping RAG check.")
            print("   -> Verdict: UNVERIFIABLE (No Ground Truth)")
            continue

        # C. Build Index for this specific check
        rag_engine.ingest_articles(real_news)
        
        # D. Retrieve Context relevant to the *content* of the suspicious article
        # We use the first 500 chars of the body as the query to find semantic matches
        query_content = article.get('bodyText', '')[:500]
        context_chunks = rag_engine.retrieve(query_content, top_k=TOP_K_RETRIEVAL)
        
        # E. LLM Judge
        print("   ⚖️  Judging accuracy against retrieved evidence...")
        verdict = judge_article_accuracy(article, context_chunks)
        print(f"\n{verdict}\n")
        
        # Sleep to be nice to APIs
        time.sleep(1)

if __name__ == "__main__":
    main()