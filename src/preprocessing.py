import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataset(df, text_col='text', label_col='label', include_source=True):
    """Cleans and prepares the dataset."""
    if df is None:
        return None, None, None, None
    
    # Handle missing values
    df = df.dropna(subset=[text_col, label_col])
    
    # Clean text
    df['cleaned_text'] = df[text_col].apply(clean_text)
    
    # Add source-based features if available and requested
    if include_source:
        source_features = []
        if 'author' in df.columns:
            df['author'] = df['author'].fillna('unknown')
            source_features.append('author')
        if 'site_url' in df.columns:
            df['site_url'] = df['site_url'].fillna('unknown')
            source_features.append('site_url')
        
        # Combine text with source information
        if source_features:
            for feat in source_features:
                df['cleaned_text'] = df['cleaned_text'] + ' ' + df[feat].apply(lambda x: clean_text(str(x)))
    
    return df

def get_features(df, text_col='cleaned_text', label_col='label'):
    """Generates TF-IDF features and splits data."""
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf.fit_transform(df[text_col])
    y = df[label_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, tfidf
