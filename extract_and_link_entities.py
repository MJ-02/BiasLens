import json
import os
from collections import defaultdict
from gliner import GLiNER
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

# Initialize GLiNER model globally
model = GLiNER.from_pretrained("EmergentMethods/gliner_medium_news-v2.1")

def extract_entities(text):
    """Extract entities using GLINER"""
    entities = defaultdict(list)
    
    # Get context window size
    context_size = 100
    
    # Process the text with GLiNER
    results = model.predict(text)
    
    # Process each entity mention
    for entity in results:
        entity_text = entity['entity']
        entity_type = entity['type']
        start_idx = entity['start']
        end_idx = entity['end']
        
        # Map GLiNER types to our schema
        type_mapping = {
            'PERSON': 'PERSON',
            'ORG': 'ORG',
            'GPE': 'Location',
            'LOC': 'Location'
        }
        
        if entity_type in type_mapping:
            # Get surrounding context
            context_start = max(0, start_idx - context_size)
            context_end = min(len(text), end_idx + context_size)
            context = text[context_start:context_end]
            
            entities[type_mapping[entity_type]].append({
                'text': entity_text,
                'context': context
            })
    
    return dict(entities)

def process_articles():
    """Process articles in batches, extract entities, and link them to Wikidata"""
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
            raw_entities = extract_entities(article['content'])
            
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
