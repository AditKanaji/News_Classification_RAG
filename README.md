# Hybrid Fake News Classification & Verification

A hybrid system for detecting fake news that combines **text classification** (Classical ML, a fine-tuned local LLM, and a cloud LLM) with a **Retrieval-Augmented Generation (RAG) "Trust but Verify" pipeline** that checks a claim against real, verified news coverage from The Guardian.

> Author: **Adit Kanaji**

---

## Table of Contents

- [Motivation](#motivation)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Trust but Verify: Content Verification](#trust-but-verify-content-verification)
- [Key Observations](#key-observations)
- [Limitations & Future Work](#limitations--future-work)

---

## Motivation

In an era of information overload, it's critical to categorize news efficiently and retrieve reliable information. This project has three objectives:

1. **Automated classification**: automatically label news articles as *real* or *fake*.
2. **Intelligent retrieval**: answer user questions with a RAG pipeline that retrieves and synthesizes live news from The Guardian.
3. **Model comparison**: evaluate Classical ML, a local LLM, and a cloud-based LLM side by side.

---

## Features

- **Text preprocessing + TF-IDF features** for classical models
- **Classical ML classifiers**: Decision Tree, SGD Classifier, Logistic Regression
- **Local LLM**: fine-tuned **DistilBERT** classifier (Hugging Face `transformers`)
- **Cloud LLM**: **Google Gemini** few-shot classification (one example per class)
- **Side-by-side comparison** table: Ground Truth vs. Local LLM vs. Gemini
- **RAG mode**: fetch Guardian articles on a topic, then ask questions interactively
- **Claim verification**: score a suspicious headline/article against verified news from the same date window
- **Evaluation scripts and plots** (accuracy, precision, recall, F1)

---

## System Architecture

### Classification pipeline

```
                 ┌──────────────────────┐      ┌──────────────────────┐
                 │        TRAIN         │      │         TEST         │
┌─────────────┐  │  Classical Models    │      │  Classical Models    │  ┌──────────────┐
│ Source-based│─▶│  Local LLM (DistilBERT)│ ──▶ │  Local LLM (DistilBERT)│─▶│  Results &   │
│ news dataset│  │  Gemini Few-Shot     │      │  Gemini Few-Shot     │  │  Evaluation  │
└─────────────┘  └──────────────────────┘      └──────────────────────┘  └──────────────┘
```

### "Trust but Verify" pipeline

```
Test Article ──▶ Hugging Face embeddings ──▶ LLM judge ──▶ Decision (score + verdict)
                        ▲                        │
                        │                        │ date range
                        └── Retrieved articles ◀─┴── News API (The Guardian)
```

---

## Project Structure

```
News_Classification_RAG/
├── main.py                            # Entry point: classification & RAG modes
├── analysis.py                        # Analysis / plotting of results
├── evaluate_local_llm.py              # Evaluation of the DistilBERT model
├── evaluate_gemini_sample_metrics.py  # Gemini few-shot metrics on a sample
├── list_models.py                     # Lists available Gemini models for your API key
├── local_llm_evaluation.txt           # Saved local LLM evaluation output
├── requirements.txt
├── src/
│   ├── utils.py                       # Logging setup
│   ├── data_loader.py                 # Local CSV loader + Guardian API fetcher
│   ├── preprocessing.py               # Text cleaning + TF-IDF feature extraction
│   ├── models.py                      # Classical models + Gemini classification
│   ├── local_llm.py                   # DistilBERT training & inference
│   ├── evaluation.py                  # Metrics for trained models
│   └── rag.py                         # SimpleRAG implementation
├── RAG/                               # "Trust but Verify" claim-verification work
├── data/                              # Place dataset.csv here
├── docs/                              # Project documentation
└── results/plots/                     # Generated evaluation plots
```

---

## Installation

**Prerequisites:** Python 3.9+ and `pip`. A GPU is recommended (not required) for DistilBERT fine-tuning.

```bash
git clone https://github.com/AditKanaji/News_Classification_RAG.git
cd News_Classification_RAG

# (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Main dependencies: `pandas`, `numpy`, `scikit-learn`, `google-generativeai`, `requests`, `python-dotenv`, `tqdm`, `torch`, `transformers`, `accelerate`, `matplotlib`, `seaborn`.

---

## Configuration

Create a `.env` file in the project root with your API keys:

```env
GOOGLE_API_KEY=your_google_api_key
GUARDIAN_API_KEY=your_guardian_api_key
```

- **Google Gemini API key**: [Google AI Studio](https://aistudio.google.com/)
- **Guardian API key**: [The Guardian Open Platform](https://open-platform.theguardian.com/access/)

Run `python list_models.py` to check which Gemini models your key can access.

---

## Dataset

The project uses the **Kaggle Fake News Dataset** (~2,000 rows).

Place it at `data/dataset.csv`. It must contain at least these columns:

| Column   | Description                          |
|----------|--------------------------------------|
| `text`   | The content of the news article      |
| `label`  | `Real` / `Fake`                      |
| `source` | Website the article came from (optional) |

You can point to a different file with `--dataset`.

---

## Usage

### 1. Classification mode

Trains and evaluates the classical models, fine-tunes DistilBERT, then prints a comparison of Ground Truth vs. Local LLM vs. Gemini (few-shot) on sample test articles (2 per class).

```bash
python main.py --mode classify
# or with a custom dataset
python main.py --mode classify --dataset path/to/your_data.csv
```

> DistilBERT training may take a few minutes, especially on CPU.

### 2. RAG mode

Fetches Guardian articles on a topic, indexes them, and opens an interactive Q&A loop.

```bash
python main.py --mode rag --query "Ireland presidential election"
```

```
RAG System Ready. Type 'exit' to quit.

Ask a question: Who won the election?
Answer: ...
```

Type `exit` to quit.

### 3. Evaluation & analysis

```bash
python evaluate_local_llm.py               # DistilBERT metrics
python evaluate_gemini_sample_metrics.py   # Gemini few-shot metrics
python analysis.py                         # Generate plots into results/plots/
```

---

## Results

| Model                   | Accuracy | Precision | Recall | F1 Score |
|-------------------------|---------:|----------:|-------:|---------:|
| **Decision Tree**       | **97.6%** | 0.98     | 0.98   | 0.98     |
| SGD Classifier          | 85.6%    | 0.86      | 0.86   | 0.85     |
| Logistic Regression     | 78.0%    | 0.81      | 0.78   | 0.76     |
| Local LLM (DistilBERT)  | 75.4%    | 0.75      | 0.75   | 0.75     |
| Gemini (Few-Shot)       | 30.0%*   | 0.31      | 0.30   | 0.28     |

\* Gemini was evaluated on a sample of the test set (see `evaluate_gemini_sample_metrics.py`).

**Key finding:** the Decision Tree performed best, and classical ML outperformed both the local LLM and Gemini few-shot. Simpler models were far more effective for this dataset.

---

## Trust but Verify: Content Verification

### Is source classification enough?

No. A story posted on a low-credibility website can still be 100% accurate, and vice versa. To judge an article fairly, you need to verify its **content** against a large body of verified news.

### How it works

1. **Clean the input**: normalize the suspicious headline/article (strip symbols, noise, etc.).
2. **Fetch evidence**: query The Guardian API for articles in a date window around the claim.
3. **Embed & rank**: generate vector embeddings (Hugging Face, 384-dim) for the claim and the evidence, then rank evidence by cosine similarity.
4. **Judge**: send the claim plus the top evidence to an LLM acting as a *Truthfulness Scorer*, which returns:

```
SCORE: [0-100]%
REASON: [brief explanation]
```

### Experiment

The pipeline was tested with three versions of the same headline:

| Type          | Headline |
|---------------|----------|
| Original      | Catherine Connolly's landslide shakes Ireland's establishment – but not its politics |
| Changed (reworded) | Catherine Connolly's overwhelming win rattles Irish elite – yet politics unchanged |
| Contradiction | Catherine Connolly Loses Ireland Presidential Election by Historic Margin |

| Test           | Llama 3.1 8B (via Groq)  | Gemini 2.0 Flash        |
|----------------|--------------------------|-------------------------|
| Changed data   | Misleading, 80%  ❌      | **SAFE, 95%** ✅        |
| Contradiction  | Misleading, 80%  ⚠️      | **FABRICATED, 100%** ✅ |

---

## Key Observations

**Gemini 2.0 Flash** reached roughly 95–100% accuracy in testing. It correctly separated *semantic variation* (rewording) from *factual contradiction*, recognizing that downplaying an event isn't the same as lying about it.

**Llama 3.1 (8B, Groq)** is very fast and good at summarizing context, but showed a high false-positive rate. It confused tone with fact, flagging the reworded headline as misleading purely because the writing style changed.

---

## Limitations & Future Work

- Classical models may be picking up dataset-specific patterns (such as source style), so their high accuracy may not generalize to unseen sources.
- Gemini few-shot was evaluated on a small sample with only one example per class; more examples or better prompting would likely improve it.
- Verification depends on The Guardian's coverage; claims outside its scope can't be checked well.
- Possible extensions: multiple news sources, a persistent vector database, and a web UI.
