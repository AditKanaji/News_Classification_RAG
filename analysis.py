import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.data_loader import load_local_dataset
from src.preprocessing import preprocess_dataset
from src.local_llm import classify_with_local_llm
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import json
import os
from tqdm import tqdm
from statsmodels.stats.contingency_tables import mcnemar

# Setup
RESULTS_DIR = "results/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_prep_data():
    print("Loading and preprocessing data...")
    df = load_local_dataset('data/dataset.csv')
    df = preprocess_dataset(df, include_source=True)
    
    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df

def evaluate_classical_models(train_df, test_df):
    print("Evaluating Classical Models...")
    
    # Vectorize
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = tfidf.fit_transform(train_df['cleaned_text'])
    X_test = tfidf.transform(test_df['cleaned_text'])
    y_train = train_df['label']
    y_test = test_df['label']
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    
    return {
        "Logistic Regression": (y_test, y_pred_lr),
        "Decision Tree": (y_test, y_pred_dt)
    }

def evaluate_local_llm(test_df, model_path="./results"):
    print("Evaluating Local LLM...")
    # Load best model (assuming it's in results/ or search for best checkpoint)
    # Since we want to use the *best* trained state, we should ideally find the best checkpoint
    # For now, let's look for the checkpoint with the highest number that isn't excessively large
    # But usually, 'results' might just contain the latest. Let's assume the user wants checkpoitn-925 or similar?
    # Actually, let's try to load from 'results' directly if a model was saved there, 
    # OR find the checkpoint directory with the highest step count.
    
    content = os.listdir("results")
    checkpoints = [d for d in content if d.startswith("checkpoint")]
    if checkpoints:
         # simple sort by number
        checkpoints.sort(key=lambda x: int(x.split('-')[1]))
        best_checkpoint = os.path.join("results", checkpoints[-1]) # Use latest for now as best usually loaded at end
        print(f"Loading local LLM from {best_checkpoint}...")
        model_path = best_checkpoint
    else:
        print("No checkpoints found, using default path.")

    
    try:
        # Tokenizer was not saved in checkpoints, load from base
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading Local LLM: {e}")
        return None, None

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Determine ID2Label
    # The training script used sorted unique labels. We must replicate this.
    labels = sorted(test_df['label'].unique().tolist()) # Assuming test_df covers all labels, or better use unique over full df if available. 
    # But wait, load_and_prep_data returns train/test splits. 
    # Let's assume binary "Fake", "Real" or similar.
    # To be safe, let's get it from the model if valid, else infer.
    # The previous script did: 
    # labels = sorted(df[label_col].unique().tolist())
    # label2id = {l: i for i, l in enumerate(labels)}
    # Since we don't have the full df here easily without reloading, let's look at the config first.
    
    if model.config.id2label and model.config.id2label != {0: "LABEL_0", 1: "LABEL_1"}:
         id2label = model.config.id2label
    else:
         print("Config id2label invalid, reconstructing from data...")
         # We need to match the sorting used in training.
         # Ideally we load the whole dataset to be sure, or just assume the classes found in test_df are sufficient if it's a balanced split.
         # Let's inspect test_df unique labels
         unique_labels = sorted(list(set(test_df['label'])))
         id2label = {i: l for i, l in enumerate(unique_labels)}
         print(f"Reconstructed id2label: {id2label}")
    
    preds = []
    for text in tqdm(test_df['cleaned_text'].tolist(), desc="Local LLM Inference"):
         # Simple inference loop
         inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
         with torch.no_grad():
             logits = model(**inputs).logits
         predicted_id = logits.argmax().item()
         preds.append(id2label[predicted_id])
         
    return test_df['label'].tolist(), preds

def plot_metrics(results):
    print("Plotting metrics...")
    metrics_data = []
    
    for model_name, (y_true, y_pred) in results.items():
        if y_pred is None: continue
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics_data.append({"Model": model_name, "Metric": "Accuracy", "Score": acc})
        metrics_data.append({"Model": model_name, "Metric": "F1 Score", "Score": f1})
        
    df_metrics = pd.DataFrame(metrics_data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_metrics, x="Model", y="Score", hue="Metric")
    plt.title("Model Performance Comparison")
    plt.ylim(0, 1.0)
    plt.savefig(f"{RESULTS_DIR}/model_comparison.png")
    plt.close()

def plot_confusion_matrices(results):
    print("Plotting confusion matrices...")
    models_to_plot = list(results.keys())
    n_models = len(models_to_plot)
    
    if n_models == 0: return

    # Determine grid size (1 row)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    # If only 1 model, axes is not a list
    if n_models == 1: axes = [axes]
    
    for i, model_name in enumerate(models_to_plot):
        y_true, y_pred = results[model_name]
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f"{model_name}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrices.png")
    plt.close()

def plot_learning_curve(trainer_state_path):
    print("Plotting learning curve...")
    if not os.path.exists(trainer_state_path):
        print(f"Trainer state not found at {trainer_state_path}")
        return

    with open(trainer_state_path, 'r') as f:
        data = json.load(f)
        
    history = data['log_history']
    
    train_loss = []
    val_loss = []
    epochs = []
    
    # Parse history (it's a list where train and eval logs might be separate)
    # We need to align them roughly by epoch or step
    
    max_step = 0
    step_loss_map = {}
    
    for entry in history:
        step = entry.get('step')
        if 'loss' in entry: # Training loss
            if step not in step_loss_map: step_loss_map[step] = {}
            step_loss_map[step]['train'] = entry['loss']
            step_loss_map[step]['epoch'] = entry['epoch']
        if 'eval_loss' in entry: # Validation loss
             if step not in step_loss_map: step_loss_map[step] = {}
             step_loss_map[step]['val'] = entry['eval_loss']
             step_loss_map[step]['epoch'] = entry['epoch']
             
    # Convert to list
    steps = sorted(step_loss_map.keys())
    train_vals = []
    val_vals = []
    epoch_vals = []
    
    for s in steps:
        if 'train' in step_loss_map[s] and 'val' in step_loss_map[s]:
             train_vals.append(step_loss_map[s]['train'])
             val_vals.append(step_loss_map[s]['val'])
             epoch_vals.append(step_loss_map[s]['epoch'])
             
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_vals, train_vals, label='Training Loss')
    plt.plot(epoch_vals, val_vals, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Local LLM Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/learning_curve.png")
    plt.close()

def perform_mcnemar_test(y_true, pred1, pred2, model1_name, model2_name):
    print(f"\nPerforming McNemar's Test: {model1_name} vs {model2_name}")
    
    # Create contingency table
    # Yes/Yes: Both correct
    # Yes/No: M1 correct, M2 wrong
    # No/Yes: M1 wrong, M2 correct
    # No/No: Both wrong
    
    yy = 0
    yn = 0
    ny = 0
    nn = 0
    
    for t, p1, p2 in zip(y_true, pred1, pred2):
        c1 = (p1 == t)
        c2 = (p2 == t)
        
        if c1 and c2: yy += 1
        elif c1 and not c2: yn += 1
        elif not c1 and c2: ny += 1
        else: nn += 1
        
    table = [[yy, yn], [ny, nn]]
    print(f"Contingency Table:\n{table}")
    
    result = mcnemar(table, exact=True)
    print(f"statistic={result.statistic}, p-value={result.pvalue}")
    
    alpha = 0.05
    if result.pvalue < alpha:
        print("Different proportions of errors (reject H0)")
    else:
        print("Same proportions of errors (fail to reject H0)")

def main():
    train_df, test_df = load_and_prep_data()
    
    # 1. Classical
    results = evaluate_classical_models(train_df, test_df)
    
    # 2. Local LLM
    y_true_llm, y_pred_llm = evaluate_local_llm(test_df)
    if y_pred_llm:
        results["Local LLM (DistilBERT)"] = (y_true_llm, y_pred_llm)
        
    # 3. Plots
    plot_metrics(results)
    plot_confusion_matrices(results)
    
    # Find latest checkpoint for learning curve
    content = os.listdir("results")
    checkpoints = [d for d in content if d.startswith("checkpoint")]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('-')[1]))
        latest_checkpoint = checkpoints[-1]
        trainer_state = os.path.join("results", latest_checkpoint, "trainer_state.json")
        plot_learning_curve(trainer_state)
        
    # 4. Stats
    if "Logistic Regression" in results and "Local LLM (DistilBERT)" in results:
        y_true = results["Logistic Regression"][0]
        y_pred_lr = results["Logistic Regression"][1]
        y_pred_llm = results["Local LLM (DistilBERT)"][1]
        
        perform_mcnemar_test(y_true, y_pred_lr, y_pred_llm, "Logistic Regression", "Local LLM")
        
    print(f"\nAnalysis complete. Plots saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
