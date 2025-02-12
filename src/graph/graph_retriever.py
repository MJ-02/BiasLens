from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from .enhanced_graph import EnhancedKnowledgeGraph

class GraphRetriever:
    """
    Retriever component for BiasLens GraphRAG system.
    Handles intelligent retrieval of relevant articles and context from the knowledge graph.
    """
    
    def __init__(self, graph: EnhancedKnowledgeGraph):
        """
        Initialize the graph retriever.
        
        Args:
            graph: Instance of EnhancedKnowledgeGraph
        """
        self.graph = graph
        self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
    
    def get_relevant_context(self, 
                           query: str, 
                           max_results: int = 5,
                           max_hops: int = 2) -> List[Dict]:
        """
        Retrieve relevant articles and their context based on a query.
        
        Args:
            query: The search query or article content to find context for
            max_results: Maximum number of direct results to return
            max_hops: Maximum number of hops for graph traversal
        
        Returns:
            List of relevant articles with their context and metadata
        """
        # First, find directly similar articles
        similar_articles = self.graph.find_similar_articles(query, limit=max_results)
        
        # For each similar article, get its context through graph traversal
        enriched_results = []
        for article in similar_articles:
            # Get contextual articles
            context = self.graph.get_article_context(
                article['id'], 
                max_hops=max_hops
            )
            
            # Add context to the result
            enriched_results.append({
                **article,
                'context': context
            })
        
        return enriched_results
    
    def get_bias_context(self, 
                        article_content: str, 
                        source: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Get context specifically for bias analysis, including:
        1. Similar articles from different sources
        2. Articles from the same source for source bias patterns
        
        Args:
            article_content: Content of the article to analyze
            source: Source of the article
        
        Returns:
            Tuple containing:
            - List of similar articles from different sources
            - List of articles from the same source
        """
        # Get similar articles across sources
        cross_source = self.graph.find_similar_articles(article_content, limit=5)
        
        # Filter out articles from the same source
        cross_source = [
            article for article in cross_source 
            if article['source'].lower() != source.lower()
        ]
        
        # Get articles from the same source
        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (a:Article)
                WHERE a.source = $source
                WITH a, gds.similarity.cosine(a.embedding, $query_embedding) AS score
                WHERE score > 0.3
                RETURN a.title as title,
                       a.source as source,
                       a.url as url,
                       a.bias as bias,
                       score
                ORDER BY score DESC
                LIMIT 5
                """,
                source=source,
                query_embedding=self.embeddings.encode(article_content).tolist()
            )
            same_source = [dict(record) for record in result]
        
        return cross_source, same_source
    
    def get_temporal_context(self, 
                           article_content: str, 
                           date: str,
                           window_days: int = 7) -> List[Dict]:
        """
        Get temporal context by finding similar articles within a time window.
        
        Args:
            article_content: Content of the article
            date: Date of the article (ISO format)
            window_days: Number of days before and after for context
        
        Returns:
            List of temporally relevant articles
        """
        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (a:Article)
                WHERE datetime(a.published_date) > datetime($date) - duration({days: $window})
                  AND datetime(a.published_date) < datetime($date) + duration({days: $window})
                WITH a, gds.similarity.cosine(a.embedding, $query_embedding) AS score
                WHERE score > 0.5
                RETURN a.title as title,
                       a.source as source,
                       a.url as url,
                       a.published_date as date,
                       score
                ORDER BY score DESC
                LIMIT 10
                """,
                date=date,
                window=window_days,
                query_embedding=self.embeddings.encode(article_content).tolist()
            )
            
            return [dict(record) for record in result]
    
    def get_source_bias_patterns(self, source: str) -> Dict:
        """
        Analyze bias patterns for a specific source.
        
        Args:
            source: Name of the news source
        
        Returns:
            Dictionary containing bias pattern analysis
        """
        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (a:Article)
                WHERE a.source = $source
                WITH a.bias as bias, count(*) as count
                RETURN collect({bias: bias, count: count}) as distribution,
                       avg(bias) as avg_bias,
                       stdev(bias) as std_bias
                """,
                source=source
            )
            
            return dict(result.single())
