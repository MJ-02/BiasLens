import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
RANDOM_STATE = 42
DATA_DIR = "dataset/data/jsons"

def load_articles():
    """Load all articles from JSON files into a pandas DataFrame."""
    articles = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
                article = json.load(f)
                articles.append(article)
    return pd.DataFrame(articles)

def preprocess_data(df):
    """Preprocess the data and create features."""
    # Basic text features
    df['text_length'] = df['content'].str.len()
    df['word_count'] = df['content'].str.split().str.len()
    
    # Create topic one-hot encoding
    topic_dummies = pd.get_dummies(df['topic'], prefix='topic')
    
    # Combine features
    feature_df = pd.concat([
        df[['text_length', 'word_count']],
        topic_dummies
    ], axis=1)
    
    return feature_df

def train_and_evaluate_models(X_train, X_test, y_train, y_test, tfidf_train, tfidf_test):
    """Train and evaluate both baseline models."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Combine TF-IDF features with other features
        X_train_combined = np.hstack([tfidf_train.toarray(), X_train])
        X_test_combined = np.hstack([tfidf_test.toarray(), X_test])
        
        # Train model
        model.fit(X_train_combined, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_combined)
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Print results
        print(f"\n{name} Results:")
        print(f"Accuracy: {results[name]['accuracy']:.4f}")
        print("\nClassification Report:")
        print(results[name]['classification_report'])
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(results[name]['confusion_matrix'], 
                   annot=True, 
                   fmt='d',
                   cmap='Blues',
                   xticklabels=['Left', 'Center', 'Right'],
                   yticklabels=['Left', 'Center', 'Right'])
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()
    
    return results

def main():
    # Load data
    print("Loading articles...")
    df = load_articles()
    
    # Create features
    print("Creating features...")
    X = preprocess_data(df)
    y = df['bias']
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_features = tfidf.fit_transform(df['content'])
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test, tfidf_train, tfidf_test = train_test_split(
        X, y, tfidf_features, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, tfidf_train, tfidf_test)
    
    print("\nBaseline models training completed!")
    print("Confusion matrix plots have been saved.")

if __name__ == "__main__":
    main()
