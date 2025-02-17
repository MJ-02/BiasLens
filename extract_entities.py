import json
import os
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from tqdm import tqdm
import pickle

def load_articles(data_path, batch_size=100):
    """Load articles in batches from the dataset"""
    jsons_path = os.path.join(data_path, 'data', 'jsons')
    json_files = os.listdir(jsons_path)
    
    for i in range(0, len(json_files), batch_size):
        batch_files = json_files[i:i + batch_size]
        batch_articles = []
        
        for json_file in batch_files:
            with open(os.path.join(jsons_path, json_file), 'r', encoding='utf-8') as f:
                article = json.load(f)
                batch_articles.append({
                    'id': article['ID'],
                    'content': article['content']
                })
        
        yield batch_articles

def extract_entities(text, tokenizer, model):
    """Extract entities using GLINER"""
    entities = defaultdict(set)
    
    # Process text in chunks of max 512 tokens to handle long texts
    max_length = 512
    words = text.split()
    chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(-1)
            
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        current_entity = []
        current_type = None
        
        for token, pred in zip(tokens, predictions[0]):
            if pred == 1:  # B-PER
                if current_entity:
                    entity_text = ' '.join(current_entity).strip().lower()
                    if current_type:
                        entities[current_type].add(entity_text)
                current_entity = [token.replace('##', '')]
                current_type = 'PERSON'
            elif pred == 2:  # B-ORG
                if current_entity:
                    entity_text = ' '.join(current_entity).strip().lower()
                    if current_type:
                        entities[current_type].add(entity_text)
                current_entity = [token.replace('##', '')]
                current_type = 'ORG'
            elif pred == 3:  # B-LOC
                if current_entity:
                    entity_text = ' '.join(current_entity).strip().lower()
                    if current_type:
                        entities[current_type].add(entity_text)
                current_entity = [token.replace('##', '')]
                current_type = 'Location'
            elif pred in [4, 5, 6]:  # I-PER, I-ORG, I-LOC
                if current_entity:
                    current_entity.append(token.replace('##', ''))
            else:  # O
                if current_entity:
                    entity_text = ' '.join(current_entity).strip().lower()
                    if current_type:
                        entities[current_type].add(entity_text)
                    current_entity = []
                    current_type = None
    
    return dict(entities)

def process_articles():
    """Process articles in batches and save extracted entities"""
    # Initialize GLINER model
    print("Loading GLINER model...")
    model_name = "EmergentMethods/gliner_medium_news-v2.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    # Create directory for saving results
    os.makedirs('extracted_entities', exist_ok=True)
    
    # Process articles in batches
    batch_num = 0
    article_entities = {}
    
    print("Processing articles...")
    for batch in tqdm(load_articles('dataset'), desc="Processing batches"):
        for article in batch:
            entities = extract_entities(article['content'], tokenizer, model)
            article_entities[article['id']] = entities
        
        # Save batch results
        batch_file = f'extracted_entities/batch_{batch_num}.pkl'
        with open(batch_file, 'wb') as f:
            pickle.dump(article_entities, f)
        
        print(f"Saved batch {batch_num} with {len(article_entities)} articles")
        batch_num += 1
        article_entities = {}  # Clear for next batch

if __name__ == "__main__":
    process_articles()
