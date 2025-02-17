import os
import pickle
import re
from rdflib import Graph, Literal, Namespace, RDF, XSD
from tqdm import tqdm

def load_extracted_entities():
    """Load all extracted entity batches"""
    entities_dir = 'extracted_entities'
    all_entities = {}
    
    print("Loading extracted entities...")
    for batch_file in tqdm(os.listdir(entities_dir), desc="Loading batches"):
        if batch_file.endswith('.pkl'):
            with open(os.path.join(entities_dir, batch_file), 'rb') as f:
                batch_entities = pickle.load(f)
                all_entities.update(batch_entities)
    
    return all_entities

def update_graph_with_entities(ttl_file, output_file):
    """Update existing knowledge graph with entity mentions"""
    # Load existing graph
    print("Loading existing knowledge graph...")
    g = Graph()
    g.parse(ttl_file, format="turtle")
    
    # Get namespaces from existing graph
    ns = None
    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    for prefix, namespace in g.namespaces():
        if prefix == "ex":
            ns = namespace
            break
    
    if ns is None:
        ns = Namespace("http://biaslens.com/")
        g.bind("ex", ns)
    
    # Load extracted entities
    article_entities = load_extracted_entities()
    
    print("Adding entity mentions to graph...")
    # Add entity mentions to graph
    for article_id, entities in tqdm(article_entities.items(), desc="Processing articles"):
        article = ns[f"article_{article_id}"]
        
        # Skip if article not in graph
        if (article, None, None) not in g:
            continue
        
        for entity_type, entity_set in entities.items():
            for entity_text in entity_set:
                # Create normalized entity ID
                entity_id = re.sub(r'\W+', '_', entity_text.lower())
                entity_node = ns[f"entity_{entity_type.lower()}_{entity_id}"]
                
                # Add entity type and label
                g.add((entity_node, RDF.type, ns[entity_type.lower()]))
                g.add((entity_node, RDFS.label, Literal(entity_text, datatype=XSD.string)))
                
                # Add mention relationship
                g.add((article, ns.mentions, entity_node))
    
    # Save updated graph
    print("Saving updated knowledge graph...")
    g.serialize(output_file, format="turtle")
    print(f"Updated graph saved to {output_file}")

if __name__ == "__main__":
    # Update the graph
    update_graph_with_entities(
        ttl_file="bias_lens_data/bias_lens_graph.ttl",
        output_file="bias_lens_data/bias_lens_graph_with_entities.ttl"
    )
