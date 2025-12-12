"""
Standalone script to evaluate the Local LLM (DistilBERT) on the test set.
Calculates comprehensive metrics including Accuracy, Precision, Recall, F1, and Confusion Matrix.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from src.data_loader import load_local_dataset
from src.preprocessing import preprocess_dataset
from src.local_llm import train_local_bert, classify_with_local_llm
from src.utils import setup_logging
import logging
from tqdm import tqdm

def evaluate_local_llm_model(dataset_path='data/dataset.csv'):
    """Evaluates the Local LLM and prints comprehensive metrics."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Loading dataset...")
    df = load_local_dataset(dataset_path)
    if df is None:
        logger.error("Dataset not found.")
        return
    
    # Preprocessing with source information
    df = preprocess_dataset(df, include_source=True)
    
    # Split data (using same split as training for consistency)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    logger.info(f"Training set size: {len(train_df)}")
    logger.info(f"Test set size: {len(test_df)}")
    
    # Train Local LLM
    logger.info("Training Local LLM (DistilBERT)...")
    model, tokenizer, id2label = train_local_bert(train_df)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    y_true = test_df['label'].tolist()
    y_pred = []
    
    for text in tqdm(test_df['cleaned_text'].tolist(), desc="Predicting"):
        pred = classify_with_local_llm(text, model, tokenizer, id2label)
        y_pred.append(pred)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Print results
    print("\n" + "="*60)
    print("LOCAL LLM (DistilBERT) EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\n" + "-"*60)
    print("CLASSIFICATION REPORT:")
    print("-"*60)
    print(classification_report(y_true, y_pred, zero_division=0))
    
    print("\n" + "-"*60)
    print("CONFUSION MATRIX:")
    print("-"*60)
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(df['label'].unique())
    
    # Pretty print confusion matrix
    print(f"\n{'Predicted->':<15}", end="")
    for label in labels:
        print(f"{label:<12}", end="")
    print("\nActual v")
    
    for i, true_label in enumerate(labels):
        print(f"{true_label:<15}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:<12}", end="")
        print()
    
    print("="*60)
    
    # Save results to file
    with open('local_llm_evaluation.txt', 'w') as f:
        f.write("LOCAL LLM (DistilBERT) EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, zero_division=0))
    
    logger.info("Results saved to local_llm_evaluation.txt")

if __name__ == "__main__":
    evaluate_local_llm_model()
