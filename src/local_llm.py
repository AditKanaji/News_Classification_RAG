import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_local_bert(df, text_col='cleaned_text', label_col='label', validation_split=0.1):
    """Fine-tunes DistilBERT for sequence classification."""
    logger.info("Starting Local LLM (DistilBERT) training...")
    
    # Encode labels
    labels = sorted(df[label_col].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    
    df['label_id'] = df[label_col].map(label2id)
    
    # Split data for validation (small split from training data)
    if validation_split > 0:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df[text_col].tolist(), df['label_id'].tolist(), test_size=validation_split, random_state=42
        )
    else:
        train_texts = df[text_col].tolist()
        train_labels = df['label_id'].tolist()
        val_texts = train_texts[:10]  # Small validation set
        val_labels = train_labels[:10]
    
    # Tokenization
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    
    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)
    
    # Model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(labels))
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=20,              # Increased epochs for early stopping
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",            # Evaluate every epoch
        save_strategy="epoch",            # Save every epoch
        load_best_model_at_end=True,      # Load best model at end
        metric_for_best_model="loss",     # Use validation loss
        use_cpu=not torch.cuda.is_available()
    )
    
    from transformers import EarlyStoppingCallback

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train()
    
    logger.info("Local LLM training complete.")
    return model, tokenizer, id2label

def classify_with_local_llm(text, model, tokenizer, id2label):
    """Classifies text using the fine-tuned local model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Move inputs to same device as model
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    return id2label[predicted_class_id]
