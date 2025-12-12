import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.data_loader import load_local_dataset
from src.preprocessing import preprocess_dataset
from src.models import classify_with_llm
import logging
import time
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Load Dataset
    df = load_local_dataset('data/dataset.csv')
    if df is None:
        logger.error("Dataset not found.")
        return

    # 2. Preprocess
    df = preprocess_dataset(df)
    
    # 3. Split Data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 4. Prepare Few-Shot Examples (from training set)
    categories = df['label'].unique().tolist()
    few_shot_examples = []
    for cat in categories:
        # Get 1 example per class
        cat_indices = train_df[train_df['label'] == cat].index[:1] 
        for idx in cat_indices:
            few_shot_examples.append((train_df.loc[idx, 'cleaned_text'], train_df.loc[idx, 'label']))
            
    # 5. Select Sample for Evaluation (10 samples)
    # We'll take 5 from each class if binary, or just random 10
    sample_test_df = test_df.sample(n=10, random_state=42)
    
    logger.info(f"Evaluating on {len(sample_test_df)} samples.")
    logger.info("Rate Limit: 2 calls per minute (30s delay between calls).")
    
    y_true = sample_test_df['label'].tolist()
    y_pred = []
    
    # 6. Evaluate loop
    for i, text in enumerate(sample_test_df['cleaned_text'].tolist()):
        logger.info(f"Processing sample {i+1}/{len(sample_test_df)}...")
        
        start_time = time.time()
        pred = classify_with_llm(text, categories, examples=few_shot_examples)
        y_pred.append(pred)
        
        # Rate limiting logic
        if i < len(sample_test_df) - 1:
            logger.info("Sleeping for 30s to respect rate limit...")
            time.sleep(30)
            
    # 7. Calculate Metrics
    # Filter out errors if any
    valid_indices = [i for i, p in enumerate(y_pred) if p != "Error"]
    if len(valid_indices) < len(y_pred):
        logger.warning(f"Ignored {len(y_pred) - len(valid_indices)} errors.")
        
    final_true = [y_true[i] for i in valid_indices]
    final_pred = [y_pred[i] for i in valid_indices]
    
    acc = accuracy_score(final_true, final_pred)
    prec = precision_score(final_true, final_pred, average='weighted', zero_division=0)
    rec = recall_score(final_true, final_pred, average='weighted', zero_division=0)
    f1 = f1_score(final_true, final_pred, average='weighted', zero_division=0)
    
    print("\n" + "="*50)
    print("Gemini (Few-Shot) Evaluation Results (Sample)")
    print("="*50)
    print(f"Sample Size: {len(final_true)}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print("-" * 50)
    
    # Print individual predictions for transparency
    print("\nIndividual Predictions:")
    for t, p in zip(final_true, final_pred):
        print(f"True: {t:<10} | Pred: {p}")

if __name__ == "__main__":
    main()
