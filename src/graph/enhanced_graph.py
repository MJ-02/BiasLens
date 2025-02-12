from typing import Dict, List, Optional
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

class EnhancedKnowledgeGraph:
    """
    Enhanced Knowledge Graph implementation using Neo4j for BiasLens.
    Handles article storage, relationship creation, and similarity search.
    """
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", 
                 password: str = "password"):
        """
        Initialize the graph database connection and embedding model.
        
        Args:
            uri: Neo4j database URI
            username: Database username
            password: Database password
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize database constraints and indexes
        self._init_db()
    
    def _init_db(self):
        """Initialize database constraints and indexes."""
        with self.driver.session() as session:
            # Create constraints
            session.run("""
                CREATE CONSTRAINT article_id IF NOT EXISTS
                FOR (a:Article) REQUIRE a.id IS UNIQUE
            """)
            
            # Create index for embeddings
            session.run("""
                CREATE INDEX article_embedding IF NOT EXISTS
                FOR (a:Article) ON a.embedding
            """)
    
    def create_article_node(self, article_data: Dict) -> str:
        """
        Create a new article node with embeddings and metadata.
        
        Args:
            article_data: Dictionary containing article information
                Required keys: 'title', 'content', 'source', 'url'
                Optional: 'authors', 'published_date'
        
        Returns:
            str: ID of created article node
        """
        # Generate embeddings for article content
        content = f"{article_data['title']} {article_data['content']}"
        embedding = self.embeddings.encode(content).tolist()
        
        # Create article node
        with self.driver.session() as session:
            result = session.run("""
                CREATE (a:Article {
                    id: randomUUID(),
                    title: $title,
                    content: $content,
                    source: $source,
                    url: $url,
                    embedding: $embedding,
                    authors: $authors,
                    published_date: $published_date,
                    created_at: datetime()
                })
                RETURN a.id as id
                """,
                title=article_data['title'],
                content=article_data['content'],
                source=article_data['source'],
                url=article_data['url'],
                embedding=embedding,
                authors=article_data.get('authors', []),
                published_date=article_data.get('published_date', None)
            )
            return result.single()['id']
    
    def create_relationships(self, article_id: str, threshold: float = 0.7):
        """
        Create relationships between articles based on content similarity.
        
        Args:
            article_id: ID of the article to create relationships for
            threshold: Similarity threshold for creating relationships
        """
        with self.driver.session() as session:
            session.run("""
                MATCH (a:Article {id: $article_id})
                MATCH (b:Article)
                WHERE a <> b
                WITH a, b, gds.similarity.cosine(a.embedding, b.embedding) AS similarity
                WHERE similarity > $threshold
                CREATE (a)-[r:SIMILAR_TO {score: similarity}]->(b)
                """,
                article_id=article_id,
                threshold=threshold
            )
    
    def find_similar_articles(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Find articles similar to a query using embedding similarity.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
        
        Returns:
            List of similar articles with similarity scores
        """
        query_embedding = self.embeddings.encode(query).tolist()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Article)
                WITH a, gds.similarity.cosine(a.embedding, $query_embedding) AS score
                WHERE score > 0.5
                RETURN a.title as title, 
                       a.source as source,
                       a.url as url,
                       score
                ORDER BY score DESC
                LIMIT $limit
                """,
                query_embedding=query_embedding,
                limit=limit
            )
            
            return [dict(record) for record in result]
    
    def get_article_context(self, article_id: str, max_hops: int = 2) -> List[Dict]:
        """
        Get contextual information about an article by traversing the graph.
        
        Args:
            article_id: ID of the article to get context for
            max_hops: Maximum number of relationship hops to traverse
        
        Returns:
            List of related articles and their relationships
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (a:Article {id: $article_id})-[r:SIMILAR_TO*1..$max_hops]-(b:Article)
                WITH b, reduce(s = 1.0, rel in r | s * rel.score) as relevance
                RETURN b.title as title,
                       b.source as source,
                       b.url as url,
                       relevance
                ORDER BY relevance DESC
                """,
                article_id=article_id,
                max_hops=max_hops
            )
            
            return [dict(record) for record in result]
    
    def close(self):
        """Close the database connection."""
        self.driver.close()
