import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, LongformerTokenizer
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)

def load_articles(data_dir="dataset/data/jsons"):
    """Load all articles from JSON files into a pandas DataFrame."""
    articles = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                article = json.load(f)
                articles.append(article)
    return pd.DataFrame(articles)

def analyze_class_distribution(df):
    """Analyze and plot the distribution of bias classes."""
    plt.figure(figsize=(10, 6))
    bias_counts = df['bias'].value_counts()
    sns.barplot(x=bias_counts.index, y=bias_counts.values)
    plt.title('Distribution of Bias Classes')
    plt.xlabel('Bias Class (0=left, 1=center, 2=right)')
    plt.ylabel('Number of Articles')
    plt.savefig('bias_distribution.png')
    plt.close()
    
    print("\nClass Distribution:")
    for bias, count in bias_counts.items():
        print(f"Class {bias}: {count} articles ({count/len(df)*100:.2f}%)")

def analyze_text_lengths(df):
    """Analyze and plot the distribution of article lengths."""
    # Character lengths
    df['char_length'] = df['content'].str.len()
    
    # Word lengths
    df['word_count'] = df['content'].apply(lambda x: len(word_tokenize(x)))
    
    # Sentence lengths
    df['sentence_count'] = df['content'].apply(lambda x: len(sent_tokenize(x)))
    
    # Calculate token lengths using both BERT and Longformer tokenizers
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    longformer_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    
    print("\nText Length Statistics:")
    print(f"Average character length: {df['char_length'].mean():.2f}")
    print(f"Average word count: {df['word_count'].mean():.2f}")
    print(f"Average sentence count: {df['sentence_count'].mean():.2f}")
    
    # Plot length distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Article Length Distributions')
    
    # Character length distribution
    sns.histplot(data=df, x='char_length', bins=50, ax=axes[0,0])
    axes[0,0].set_title('Character Length Distribution')
    axes[0,0].set_xlabel('Number of Characters')
    
    # Word count distribution
    sns.histplot(data=df, x='word_count', bins=50, ax=axes[0,1])
    axes[0,1].set_title('Word Count Distribution')
    axes[0,1].set_xlabel('Number of Words')
    
    # Sentence count distribution
    sns.histplot(data=df, x='sentence_count', bins=50, ax=axes[1,0])
    axes[1,0].set_title('Sentence Count Distribution')
    axes[1,0].set_xlabel('Number of Sentences')
    
    # Token lengths for both tokenizers
    token_lengths_bert = []
    token_lengths_longformer = []
    
    print("\nCalculating token lengths (this may take a while)...")
    for text in df['content']:
        token_lengths_bert.append(len(bert_tokenizer.encode(text)))
        token_lengths_longformer.append(len(longformer_tokenizer.encode(text)))
    
    df['bert_tokens'] = token_lengths_bert
    df['longformer_tokens'] = token_lengths_longformer
    
    # Plot token length distributions
    sns.histplot(data=df, x='bert_tokens', bins=50, ax=axes[1,1], color='green')
    axes[1,1].set_title('Token Length Distribution')
    axes[1,1].set_xlabel('Number of Tokens')
    
    plt.tight_layout()
    plt.savefig('length_distributions.png')
    plt.close()
    
    print("\nToken Length Statistics:")
    print(f"BERT - Average tokens per article: {np.mean(token_lengths_bert):.2f}")
    print(f"BERT - Articles exceeding 512 tokens: {sum(np.array(token_lengths_bert) > 512)} ({sum(np.array(token_lengths_bert) > 512)/len(df)*100:.2f}%)")
    print(f"Longformer - Average tokens per article: {np.mean(token_lengths_longformer):.2f}")
    print(f"Longformer - Articles exceeding 4096 tokens: {sum(np.array(token_lengths_longformer) > 4096)} ({sum(np.array(token_lengths_longformer) > 4096)/len(df)*100:.2f}%)")

def analyze_topics(df):
    """Analyze the distribution of topics."""
    topic_counts = df['topic'].value_counts()
    
    plt.figure(figsize=(15, 6))
    sns.barplot(x=topic_counts.values, y=topic_counts.index)
    plt.title('Distribution of Topics')
    plt.xlabel('Number of Articles')
    plt.tight_layout()
    plt.savefig('topic_distribution.png')
    plt.close()
    
    print("\nTop 10 Topics:")
    for topic, count in topic_counts.head(10).items():
        print(f"{topic}: {count} articles ({count/len(df)*100:.2f}%)")

def analyze_sources(df):
    """Analyze the distribution of news sources."""
    source_counts = df['source'].value_counts()
    
    plt.figure(figsize=(15, 6))
    sns.barplot(x=source_counts.head(20).values, y=source_counts.head(20).index)
    plt.title('Top 20 News Sources')
    plt.xlabel('Number of Articles')
    plt.tight_layout()
    plt.savefig('source_distribution.png')
    plt.close()
    
    print("\nTop 10 Sources:")
    for source, count in source_counts.head(10).items():
        print(f"{source}: {count} articles ({count/len(df)*100:.2f}%)")

def analyze_temporal_distribution(df):
    """Analyze the temporal distribution of articles."""
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Plot articles per year
    plt.figure(figsize=(12, 6))
    year_counts = df['year'].value_counts().sort_index()
    sns.barplot(x=year_counts.index, y=year_counts.values)
    plt.title('Articles per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('temporal_distribution.png')
    plt.close()
    
    print("\nTemporal Distribution:")
    print(f"Date range: from {df['date'].min()} to {df['date'].max()}")
    print("\nArticles per year:")
    for year, count in year_counts.items():
        print(f"{year}: {count} articles")

def main():
    print("Loading articles...")
    df = load_articles()
    
    print("\n=== Dataset Overview ===")
    print(f"Total number of articles: {len(df)}")
    
    print("\n=== Class Distribution Analysis ===")
    analyze_class_distribution(df)
    
    print("\n=== Text Length Analysis ===")
    analyze_text_lengths(df)
    
    print("\n=== Topic Analysis ===")
    analyze_topics(df)
    
    print("\n=== Source Analysis ===")
    analyze_sources(df)
    
    print("\n=== Temporal Analysis ===")
    analyze_temporal_distribution(df)
    
    print("\nAnalysis complete! Visualization files have been saved.")

if __name__ == "__main__":
    main()
