import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Constants
RANDOM_STATE = 42
DATA_DIR = "dataset/data/jsons"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_model = None
        self.should_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = self._save_model_state(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = self._save_model_state(model)
            self.counter = 0
    
    def _save_model_state(self, model):
        return {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    def get_best_model(self, model):
        if self.best_model is not None:
            model.load_state_dict(self.best_model)
        return model

class ArticleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_articles():
    """Load all articles from JSON files into a pandas DataFrame."""
    articles = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
                article = json.load(f)
                articles.append(article)
    return pd.DataFrame(articles)

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(data_loader, desc='Training')
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    return predictions, actual_labels, total_loss / len(data_loader)

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=['Left', 'Center', 'Right'],
                yticklabels=['Left', 'Center', 'Right'])
    plt.title('Confusion Matrix - BERT')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()

def save_model(model, tokenizer, metrics, save_dir=None):
    """Save the model, tokenizer, and experiment details to disk."""
    # Generate timestamp for the model directory
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    if save_dir is None:
        save_dir = f'saved_models/bert_model_{timestamp}'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save model and tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Save experiment details
    experiment_details = {
        'timestamp': timestamp,
        'model_type': 'BERT',
        'parameters': {
            'max_length': MAX_LENGTH,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'early_stopping_patience': 3,
            'early_stopping_min_delta': 0.001
        },
        'metrics': metrics
    }
    
    with open(os.path.join(save_dir, 'experiment_details.json'), 'w') as f:
        json.dump(experiment_details, f, indent=4)
    
    print(f"Model, tokenizer, and experiment details saved to {save_dir}")
    return save_dir

def load_model(save_dir):
    """Load the model and tokenizer from disk."""
    model = BertForSequenceClassification.from_pretrained(save_dir)
    tokenizer = BertTokenizer.from_pretrained(save_dir)
    return model, tokenizer

def predict_bias(text, model, tokenizer, device):
    """Predict bias for a given text using the loaded model."""
    model.eval()
    # Prepare the text input
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)
    
    bias_map = {0: 'left', 1: 'center', 2: 'right'}
    return bias_map[preds.item()]

def main():
    print("Loading articles...")
    df = load_articles()
    
    # Prepare data
    texts = df['content'].tolist()
    labels = df['bias'].tolist()
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    
    # Initialize tokenizer and model
    print("Initializing BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3,
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(DEVICE)
    
    # Create datasets
    train_dataset = ArticleDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = ArticleDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    
    # Training loop
    print(f"Starting training on {DEVICE}...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        print(f"Average training loss: {train_loss:.4f}")
        
        # Evaluate on validation set
        print("\nEvaluating...")
        predictions, actual_labels, val_loss = evaluate_model(model, val_loader, DEVICE)
        
        # Early stopping check
        early_stopping(val_loss, model)
        
        # Calculate metrics
        accuracy = accuracy_score(actual_labels, predictions)
        print(f"\nValidation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(actual_labels, predictions))
        
        # Plot confusion matrix
        plot_confusion_matrix(
            actual_labels, 
            predictions, 
            f'confusion_matrix_bert_epoch_{epoch+1}.png'
        )
        
        if early_stopping.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Load the best model before saving
    model = early_stopping.get_best_model(model)
    
    print("\nTraining completed!")
    print("Model evaluation and confusion matrix plots have been saved.")
    
    # Calculate final metrics
    final_metrics = {
        'final_validation_loss': val_loss,
        'final_validation_accuracy': accuracy,
        'classification_report': classification_report(actual_labels, predictions, output_dict=True),
        'early_stopping_epoch': epoch + 1
    }
    
    # Save the trained model and tokenizer with metrics
    save_dir = save_model(model, tokenizer, final_metrics)
    
    # Example of loading and using the model
    print("\nTesting model loading and prediction...")
    loaded_model, loaded_tokenizer = load_model(save_dir)
    loaded_model.to(DEVICE)
    
    # Test prediction on a sample article
    sample_text = df['content'].iloc[0]
    predicted_bias = predict_bias(sample_text, loaded_model, loaded_tokenizer, DEVICE)
    print(f"Sample prediction - Predicted bias: {predicted_bias}")
    print(f"Actual bias: {df['bias_text'].iloc[0]}")

if __name__ == "__main__":
    main()
