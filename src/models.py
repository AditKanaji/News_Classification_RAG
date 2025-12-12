import google.generativeai as genai
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
import logging
from src.utils import get_api_key
import time

logger = logging.getLogger(__name__)

def train_classical_models(X_train, y_train):
    """Trains classical ML models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "SGD Classifier (Regression-based)": SGDClassifier(loss='log_loss')
    }
    
    trained_models = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def configure_genai():
    api_key = get_api_key("GOOGLE")
    if api_key:
        genai.configure(api_key=api_key)
        return True
    return False

def classify_with_llm(text, categories, examples=None):
    """Classifies text using Gemini API with optional Few-Shot examples."""
    if not configure_genai():
        return "Error: API Key missing"
    
    model = genai.GenerativeModel('gemini-flash-latest')
    
    prompt = f"Classify the following news article into one of these categories: {', '.join(categories)}.\n"
    
    if examples:
        prompt += "\nHere are some examples:\n"
        for ex_text, ex_label in examples:
            prompt += f"Article: {ex_text[:300]}...\nCategory: {ex_label}\n\n"
            
    prompt += f"Return ONLY the category name.\n\nArticle: {text[:1000]}..."
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"LLM Classification Error: {e}")
        time.sleep(1) # Rate limit handling
        return "Error"
