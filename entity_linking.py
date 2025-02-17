import json
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import spacy
import wikipedia
from wikimapper import WikiMapper
from tqdm import tqdm
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EntityLinker:
    def __init__(self):
        # Load spaCy with entity linker
        self.nlp = spacy.load("en_core_web_lg")
        
        # Load sentence transformer for context similarity
        self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Initialize WikiMapper for entity linking
        self.mapper = WikiMapper("index_enwiki-latest.db")
        
        # Cache for entity resolutions
        self.entity_cache = {}
        self.context_vectors = {}
        
    def get_wikidata_entity(self, title: str) -> Dict:
        """Get Wikidata entity information"""
        try:
            # First get Wikipedia page
            page = wikipedia.page(title, auto_suggest=False)
            
            # Get Wikidata ID from Wikipedia page
            wikidata_id = self.mapper.title_to_id(page.title)
            
            if not wikidata_id:
                return None
            
            # Query Wikidata API
            url = f"https://www.wikidata.org/w/api.php"
            params = {
                "action": "wbgetentities",
                "ids": wikidata_id,
                "format": "json",
                "languages": "en"
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if "entities" in data and wikidata_id in data["entities"]:
                entity = data["entities"][wikidata_id]
                
                # Extract relevant information
                result = {
                    "id": wikidata_id,
                    "title": page.title,
                    "description": page.summary,
                    "aliases": set()
                }
                
                # Get aliases
                if "aliases" in entity and "en" in entity["aliases"]:
                    for alias in entity["aliases"]["en"]:
                        result["aliases"].add(alias["value"].lower())
                
                return result
            
        except (wikipedia.exceptions.DisambiguationError,
                wikipedia.exceptions.PageError,
                requests.exceptions.RequestException):
            return None
        
        return None
    
    def get_entity_context(self, text: str, entity_text: str) -> str:
        """Extract context around entity mention"""
        # Find entity position
        start = text.lower().find(entity_text.lower())
        if start == -1:
            return text
        
        # Get surrounding context (100 chars before and after)
        context_start = max(0, start - 100)
        context_end = min(len(text), start + len(entity_text) + 100)
        
        return text[context_start:context_end]
    
    def link_entity(self, entity_text: str, context: str, entity_type: str) -> Tuple[str, str]:
        """
        Link entity mention to Wikidata entity
        Returns tuple of (canonical_name, wikidata_id)
        """
        cache_key = (entity_text, entity_type)
        if cache_key in self.entity_cache:
            return self.entity_cache[cache_key]
        
        # Get context embedding
        context_vector = self.encoder.encode(context)
        
        try:
            # Search Wikipedia for entity
            search_results = wikipedia.search(entity_text, results=5)
            
            candidates = []
            for title in search_results:
                entity = self.get_wikidata_entity(title)
                if entity:
                    # Check if entity type matches
                    if self.check_entity_type(entity, entity_type):
                        # Get description embedding
                        desc_vector = self.encoder.encode(entity["description"])
                        
                        # Calculate context similarity
                        similarity = cosine_similarity(
                            context_vector.reshape(1, -1),
                            desc_vector.reshape(1, -1)
                        )[0][0]
                        
                        candidates.append((entity, similarity))
            
            if candidates:
                # Sort by similarity score
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_match = candidates[0][0]
                
                result = (best_match["title"], best_match["id"])
                self.entity_cache[cache_key] = result
                return result
                
        except (wikipedia.exceptions.DisambiguationError,
                wikipedia.exceptions.PageError,
                requests.exceptions.RequestException):
            pass
        
        # If no good match found, return original text
        self.entity_cache[cache_key] = (entity_text, None)
        return (entity_text, None)
    
    def check_entity_type(self, entity: Dict, required_type: str) -> bool:
        """Check if Wikidata entity matches required type"""
        # This could be expanded with more sophisticated type checking
        # using Wikidata instance of (P31) and subclass of (P279) properties
        if required_type == "PERSON":
            return "person" in entity["description"].lower()
        elif required_type == "ORG":
            return any(word in entity["description"].lower() 
                      for word in ["organization", "company", "agency", "party"])
        elif required_type == "Location":
            return any(word in entity["description"].lower()
                      for word in ["country", "city", "state", "region", "location"])
        return False
    
    def save_cache(self, output_file: str):
        """Save entity cache to file"""
        cache_data = {
            str(key): value for key, value in self.entity_cache.items()
        }
        with open(output_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def load_cache(self, input_file: str):
        """Load entity cache from file"""
        with open(input_file) as f:
            cache_data = json.load(f)
            self.entity_cache = {
                eval(key): value for key, value in cache_data.items()
            }

def test_entity_linker():
    """Test the entity linker with some examples"""
    linker = EntityLinker()
    
    test_cases = [
        ("USA", "The United States has announced new trade policies.", "Location"),
        ("FBI", "The FBI is investigating the case.", "ORG"),
        ("Biden", "President Biden addressed the nation.", "PERSON"),
        ("Congress", "Congress passed the bill yesterday.", "ORG")
    ]
    
    print("\nTesting Entity Linking:")
    print("=" * 50)
    
    for entity_text, context, entity_type in test_cases:
        canonical, wikidata_id = linker.link_entity(entity_text, context, entity_type)
        print(f"\nEntity: {entity_text}")
        print(f"Context: {context}")
        print(f"Type: {entity_type}")
        print(f"Linked to: {canonical} (Wikidata: {wikidata_id})")

if __name__ == "__main__":
    test_entity_linker()
