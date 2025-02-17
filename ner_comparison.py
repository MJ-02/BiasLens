import json
import os
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from tqdm import tqdm

def load_sample_articles(data_path, sample_size=5):
    """Load a sample of articles from the dataset"""
    articles = []
    jsons_path = os.path.join(data_path, 'data', 'jsons')
    json_files = os.listdir(jsons_path)[:sample_size]
    
    for json_file in json_files:
        with open(os.path.join(jsons_path, json_file), 'r', encoding='utf-8') as f:
            article = json.load(f)
            articles.append(article)
    
    return articles

def extract_entities_gliner(text, tokenizer, model):
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

def compare_ner_systems():
    """Compare GLINER NER system on sample articles"""
    articles = load_sample_articles('dataset')
    
    # Initialize GLINER model
    model_name = "EmergentMethods/gliner_medium_news-v2.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    results = {
        'GLINER': defaultdict(list)
    }
    
    for article in tqdm(articles, desc="Processing articles"):
        text = article['content']
        
        # Extract entities using GLINER
        gliner_entities = extract_entities_gliner(text, tokenizer, model)
        
        # Store results
        for entity_type in ['PERSON', 'ORG', 'Location']:
            if entity_type in gliner_entities:
                results['GLINER'][entity_type].extend(gliner_entities[entity_type])
    
    # Print comparison results
    print("\nNER Systems Comparison:")
    print("=" * 50)
    
    for system, entities in results.items():
        print(f"\n{system} Results:")
        for entity_type, values in entities.items():
            print(f"\n{entity_type}:")
            print(f"Total unique entities: {len(set(values))}")
            print("Sample entities:", list(set(values))[:5])

if __name__ == "__main__":
    compare_ner_systems()
