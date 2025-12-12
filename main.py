import argparse
import logging
from src.utils import setup_logging
from src.data_loader import load_local_dataset, fetch_guardian_articles
from src.preprocessing import preprocess_dataset, get_features
from src.models import train_classical_models, classify_with_llm
from src.evaluation import evaluate_models
from src.rag import SimpleRAG
import pandas as pd

from src.local_llm import train_local_bert, classify_with_local_llm

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="News Classification & RAG System")
    parser.add_argument('--mode', choices=['classify', 'rag'], required=True, help="Mode to run: 'classify' or 'rag'")
    parser.add_argument('--dataset', type=str, default='data/dataset.csv', help="Path to dataset for classification")
    parser.add_argument('--query', type=str, help="Query for RAG or Guardian API search")
    
    args = parser.parse_args()
    
    if args.mode == 'classify':
        logger.info("Starting Classification Pipeline...")
        df = load_local_dataset(args.dataset)
        if df is None:
            logger.error("Dataset not found. Please provide a valid CSV file.")
            return

        # Preprocessing
        df = preprocess_dataset(df)
        X_train, X_test, y_train, y_test, tfidf = get_features(df)
        
        # Train Classical Models
        trained_models = train_classical_models(X_train, y_train)
        
        # Evaluate Classical
        results = evaluate_models(trained_models, X_test, y_test)
        print("\nClassical Model Results:")
        print(results)
        
        # Train Local LLM (DistilBERT)
        print("\n" + "="*50)
        logger.info("Training Local LLM (DistilBERT)... This may take a few minutes.")
        local_model, tokenizer, id2label = train_local_bert(df)
        
        # Prepare Few-Shot Examples for Gemini
        categories = df['label'].unique().tolist()
        few_shot_examples = []
        for cat in categories:
            # Get 1 example per class from training set
            cat_indices = y_train[y_train == cat].index[:1] 
            for idx in cat_indices:
                few_shot_examples.append((df.loc[idx, 'cleaned_text'], df.loc[idx, 'label']))
        
        # Comparison Demo
        print("\n" + "="*50)
        logger.info("Running Comparison: Ground Truth vs Local LLM vs Gemini (Few-Shot)...")
        
        # Pick test samples (2 per class)
        samples_to_test = []
        for cat in categories:
            cat_indices = y_test[y_test == cat].index[:2]
            samples_to_test.extend(cat_indices)
            
        print(f"\n{'Text Snippet':<50} | {'Ground Truth':<12} | {'Local LLM':<12} | {'Gemini (Few-Shot)':<12}")
        print("-" * 100)
        
        for idx in samples_to_test:
            text = df.loc[idx, 'cleaned_text']
            true_label = df.loc[idx, 'label']
            
            # Local LLM Prediction
            local_pred = classify_with_local_llm(text, local_model, tokenizer, id2label)
            
            # Gemini Prediction (Few-Shot)
            gemini_pred = classify_with_llm(text, categories, examples=few_shot_examples)
            
            print(f"{text[:47]:<50}... | {true_label:<12} | {local_pred:<12} | {gemini_pred:<12}")

    elif args.mode == 'rag':
        logger.info("Starting RAG Pipeline...")
        if not args.query:
            logger.error("Please provide a --query for RAG mode.")
            return
            
        # Fetch articles for RAG context
        logger.info(f"Fetching articles for context on topic: {args.query}")
        articles = fetch_guardian_articles(args.query)
        
        if not articles:
            logger.error("No articles found.")
            return
            
        rag = SimpleRAG()
        rag.add_documents(articles)
        
        # Interactive Loop
        print("\nRAG System Ready. Type 'exit' to quit.")
        while True:
            user_query = input("\nAsk a question: ")
            if user_query.lower() == 'exit':
                break
            
            answer = rag.generate_answer(user_query)
            print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()
