from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd

def evaluate_models(models, X_test, y_test):
    """Evaluates trained models."""
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })
        
    return pd.DataFrame(results)

def compare_llm_vs_classical(y_true, llm_preds, classical_preds):
    """Compares LLM predictions with classical model predictions."""
    # This is a placeholder for a more detailed comparison logic
    # In a real scenario, you'd align the indices and compare
    pass
