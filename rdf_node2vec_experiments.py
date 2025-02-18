import networkx as nx
from rdflib import Graph
from node2vec import Node2Vec
import numpy as np
from typing import Dict, Callable
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def rdf_to_networkx(rdf_path: str) -> nx.Graph:
    """
    Convert RDF graph to NetworkX graph
    """
    # Load RDF graph
    g = Graph()
    g.parse(rdf_path)
    
    # Convert to NetworkX
    G = nx.Graph()
    
    # Add edges from RDF triples
    for s, p, o in g:
        G.add_edge(str(s), str(o), predicate=str(p))
        
    return G

def uniform_weights(G: nx.Graph) -> nx.Graph:
    """All edges have weight 1"""
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1.0
    return G

def predicate_frequency_weights(G: nx.Graph) -> nx.Graph:
    """Weight edges based on predicate frequency (inverse)"""
    pred_counts = {}
    for _, _, d in G.edges(data=True):
        pred = d['predicate']
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    
    # Inverse frequency weighting
    for edge in G.edges(data=True):
        pred = edge[2]['predicate']
        G[edge[0]][edge[1]]['weight'] = 1.0 / np.log1p(pred_counts[pred])
    return G

def degree_weights(G: nx.Graph) -> nx.Graph:
    """Weight edges based on node degrees"""
    degrees = dict(G.degree())
    for edge in G.edges():
        # Use harmonic mean of node degrees
        deg1, deg2 = degrees[edge[0]], degrees[edge[1]]
        weight = 2 / (1/deg1 + 1/deg2) if deg1 > 0 and deg2 > 0 else 1.0
        G[edge[0]][edge[1]]['weight'] = weight
    return G

def run_node2vec(G: nx.Graph, dimensions: int, walk_length: int, num_walks: int, 
                 p: float, q: float) -> Dict[str, np.ndarray]:
    """
    Run node2vec on the graph and return embeddings
    """
    # Initialize node2vec
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=4
    )
    
    # Train model
    model = node2vec.fit(window=10, min_count=1)
    
    # Get embeddings
    embeddings = {}
    for node in G.nodes():
        embeddings[node] = model.wv[node]
        
    return embeddings

def visualize_embeddings(embeddings: Dict[str, np.ndarray], node_labels: Dict[str, str], 
                        title: str, save_path: str):
    """
    Visualize embeddings using t-SNE
    """
    # Convert embeddings to matrix
    nodes = list(embeddings.keys())
    emb_matrix = np.array([embeddings[node] for node in nodes])
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    node_pos = tsne.fit_transform(emb_matrix)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(node_pos[:, 0], node_pos[:, 1], 
                         c=[float(node_labels.get(node, -1)) for node in nodes],
                         cmap='viridis')
    plt.colorbar(scatter, label='Bias Score')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def main():
    # Parameters for experiments
    dimensions = 128
    walk_length = 30
    num_walks = 200
    p_values = [0.25, 1.0, 4.0]  # Return parameter
    q_values = [0.25, 1.0, 4.0]  # In-out parameter
    
    # Load graph
    rdf_path = "bias_lens_data/bias_lens_graph.ttl"
    G = rdf_to_networkx(rdf_path)
    
    # Load bias labels
    bias_labels = pd.read_csv("bias_lens_data/URI_label_pairs.tsv", sep='\t', index_col=0)
    bias_labels = bias_labels.to_dict()[bias_labels.columns[0]]
    
    # Weight functions to test
    weight_functions = {
        'uniform': uniform_weights,
        'predicate_frequency': predicate_frequency_weights,
        'degree': degree_weights
    }
    
    # Run experiments
    results = []
    for weight_name, weight_fn in weight_functions.items():
        # Apply weights
        G_weighted = weight_fn(G.copy())
        
        for p in p_values:
            for q in q_values:
                print(f"Running experiment: weights={weight_name}, p={p}, q={q}")
                
                # Get embeddings
                embeddings = run_node2vec(G_weighted, dimensions, walk_length, 
                                        num_walks, p, q)
                
                # Visualize
                title = f"Node2Vec Embeddings (weights={weight_name}, p={p}, q={q})"
                save_path = f"figures/node2vec_{weight_name}_p{p}_q{q}.png"
                visualize_embeddings(embeddings, bias_labels, title, save_path)
                
                # Save embeddings
                emb_df = pd.DataFrame.from_dict(embeddings, orient='index')
                emb_df.to_csv(f"embeddings/node2vec_{weight_name}_p{p}_q{q}.csv")
                
                # Record parameters
                results.append({
                    'weight_method': weight_name,
                    'p': p,
                    'q': q,
                    'dimensions': dimensions,
                    'walk_length': walk_length,
                    'num_walks': num_walks
                })
    
    # Save experiment metadata
    pd.DataFrame(results).to_csv("embeddings/experiment_metadata.csv", index=False)

if __name__ == "__main__":
    main()
