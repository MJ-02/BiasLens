import os
import pickle
import re
from rdflib import Graph, Literal, Namespace, RDF, XSD, URIRef
from tqdm import tqdm

def load_extracted_entities():
    """Load all extracted and linked entity batches"""
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
    """Update existing knowledge graph with linked entity mentions"""
    # Load existing graph
    print("Loading existing knowledge graph...")
    g = Graph()
    g.parse(ttl_file, format="turtle")
    
    # Define namespace
    ns = Namespace("http://biaslens.com/")
    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    g.bind("bl", ns)
    
    # Load extracted entities
    article_entities = load_extracted_entities()
    
    print("Adding linked entity mentions to graph...")
    # Add entity mentions to graph
    for article_id, entities in tqdm(article_entities.items(), desc="Processing articles"):
        article = ns[f"article_{article_id}"]
        
        # Skip if article not in graph
        if (article, None, None) not in g:
            continue
        
        for entity_type, entity_mentions in entities.items():
            for mention in entity_mentions:
                # Create or get existing entity node using canonical name
                entity_id = re.sub(r'\W+', '_', mention['canonical_name'].lower())
                entity_node = ns[f"entity_{entity_type.lower()}_{entity_id}"]
                
                # Check if entity already exists
                if (entity_node, RDF.type, None) not in g:
                    # Add entity type
                    g.add((entity_node, RDF.type, ns[entity_type.lower()]))
                    g.add((entity_node, RDFS.label, Literal(mention['canonical_name'], datatype=XSD.string)))
                
                # Add mention with context
                g.add((article, ns.mentions, entity_node))
                g.add((entity_node, ns.mentionedAs, Literal(mention['text'], datatype=XSD.string)))
                g.add((entity_node, ns.hasContext, Literal(mention['context'], datatype=XSD.string)))
    
    # Save updated graph
    print("Saving updated knowledge graph...")
    g.serialize(output_file, format="turtle")
    print(f"Updated graph saved to {output_file}")

    # Print some statistics
    print("\nKnowledge Graph Statistics:")
    print("=" * 50)
    
    entity_types = {
        ns.person: "People",
        ns.org: "Organizations",
        ns.location: "Locations"
    }
    
    for entity_type, label in entity_types.items():
        count = len(list(g.subjects(RDF.type, entity_type)))
        print(f"Total unique {label}: {count}")
    
    total_mentions = len(list(g.subjects(RDF.type, ns.Mention)))
    print(f"Total mentions: {total_mentions}")

if __name__ == "__main__":
    # Update the graph
    update_graph_with_entities(
        ttl_file="bias_lens_data/bias_lens_graph.ttl",
        output_file="bias_lens_data/bias_lens_graph_with_linked_entities.ttl"
    )
