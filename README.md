# News Classification & RAG System

This project implements a news classification system and a RAG (Retrieval-Augmented Generation) pipeline using Google's Gemini API and The Guardian's API.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    Create a `.env` file in the root directory with your API keys:
    ```
    GOOGLE_API_KEY=your_google_api_key
    GUARDIAN_API_KEY=your_guardian_api_key
    ```

3.  **Dataset**:
    Place your classification dataset in `data/dataset.csv`. It should have at least two columns: `text` and `label`.

## Usage

### Classification Mode
Runs preprocessing, trains classical models, evaluates them, and demonstrates LLM classification.
```bash
python main.py --mode classify --dataset data/dataset.csv
```



