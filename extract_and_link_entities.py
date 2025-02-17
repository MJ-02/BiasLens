import json
import os
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
import pickle
from entity_linking import EntityLinker

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
    entities = defaultdict(list)  # Changed to list to preserve order and context
    
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
        current_start = None
        
        for idx, (token, pred) in enumerate(zip(tokens, predictions[0])):
            if pred == 1:  # B-PER
                if current_entity:
                    entity_text = ' '.join(current_entity).strip()
                    if current_type:
                        entities[current_type].append({
                            'text': entity_text,
                            'context': chunk[max(0, current_start-100):min(len(chunk), current_start+len(entity_text)+100)]
                        })
                current_entity = [token.replace('##', '')]
                current_type = 'PERSON'
                current_start = idx
            elif pred == 2:  # B-ORG
                if current_entity:
                    entity_text = ' '.join(current_entity).strip()
                    if current_type:
                        entities[current_type].append({
                            'text': entity_text,
                            'context': chunk[max(0, current_start-100):min(len(chunk), current_start+len(entity_text)+100)]
                        })
                current_entity = [token.replace('##', '')]
                current_type = 'ORG'
                current_start = idx
            elif pred == 3:  # B-LOC
                if current_entity:
                    entity_text = ' '.join(current_entity).strip()
                    if current_type:
                        entities[current_type].append({
                            'text': entity_text,
                            'context': chunk[max(0, current_start-100):min(len(chunk), current_start+len(entity_text)+100)]
                        })
                current_entity = [token.replace('##', '')]
                current_type = 'Location'
                current_start = idx
            elif pred in [4, 5, 6]:  # I-PER, I-ORG, I-LOC
                if current_entity:
                    current_entity.append(token.replace('##', ''))
            else:  # O
                if current_entity:
                    entity_text = ' '.join(current_entity).strip()
                    if current_type:
                        entities[current_type].append({
                            'text': entity_text,
                            'context': chunk[max(0, current_start-100):min(len(chunk), current_start+len(entity_text)+100)]
                        })
                    current_entity = []
                    current_type = None
                    current_start = None
    
    return dict(entities)

def process_articles():
    """Process articles in batches, extract entities, and link them to Wikidata"""
    # Initialize GLINER model
    print("Loading GLINER model...")
    model_name = "EmergentMethods/gliner_medium_news-v2.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    # Initialize Entity Linker
    print("Initializing Entity Linker...")
    linker = EntityLinker()
    
    # Create directories for saving results
    os.makedirs('extracted_entities', exist_ok=True)
    
    # Process articles in batches
    batch_num = 0
    article_entities = {}
    
    print("Processing articles...")
    for batch in tqdm(load_articles('dataset'), desc="Processing batches"):
        for article in batch:
            # Extract entities with context
            raw_entities = extract_entities(article['content'], tokenizer, model)
            
            # Link entities to Wikidata
            linked_entities = defaultdict(list)
            for entity_type, entities in raw_entities.items():
                for entity in entities:
                    canonical_name, wikidata_id = linker.link_entity(
                        entity['text'],
                        entity['context'],
                        entity_type
                    )
                    if wikidata_id:  # Only include successfully linked entities
                        linked_entities[entity_type].append({
                            'text': entity['text'],
                            'canonical_name': canonical_name,
                            'wikidata_id': wikidata_id,
                            'context': entity['context']
                        })
            
            article_entities[article['id']] = dict(linked_entities)
        
        # Save batch results
        batch_file = f'extracted_entities/batch_{batch_num}.pkl'
        with open(batch_file, 'wb') as f:
            pickle.dump(article_entities, f)
        
        print(f"Saved batch {batch_num} with {len(article_entities)} articles")
        
        # Save entity linker cache periodically
        if batch_num % 5 == 0:
            linker.save_cache(f'extracted_entities/entity_cache_{batch_num}.json')
        
        batch_num += 1
        article_entities = {}  # Clear for next batch

if __name__ == "__main__":
    process_articles()
