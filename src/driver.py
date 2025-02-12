import os
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from .utils.config import BiasLensConfig
from .graph.enhanced_graph import EnhancedKnowledgeGraph
from .rag.rag_engine import BiasLensRAG

class BiasLens:
    """
    Main driver class for the BiasLens system.
    Coordinates data processing, graph operations, and bias analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize BiasLens with configuration.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = BiasLensConfig(config_path)
        
        # Initialize graph database
        self.graph = EnhancedKnowledgeGraph(
            **self.config.neo4j_config
        )
        
        # Initialize RAG engine
        self.rag = BiasLensRAG(
            self.graph,
            **self.config.openai_config
        )
    
    def process_dataset(self, dataset_path: str):
        """
        Process a dataset of articles and add them to the knowledge graph.
        
        Args:
            dataset_path: Path to the dataset CSV file
        """
        df = pd.read_csv(dataset_path)
        
        print(f"Processing {len(df)} articles...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # Create article node
            article_data = {
                'title': row['title'],
                'content': row['content'],
                'source': row['source_id'],
                'url': row['link'],
                'authors': row.get('creator', []),
                'published_date': row.get('published_date', None)
            }
            
            article_id = self.graph.create_article_node(article_data)
            
            # Create relationships with other articles
            self.graph.create_relationships(
                article_id,
                threshold=self.config.graph_config['similarity_threshold']
            )
    
    def fetch_and_process_news(self):
        """Fetch current news articles and process them."""
        import requests
        
        news_config = self.config.news_api_config
        url = (
            f"https://newsdata.io/api/1/news"
            f"?apikey={news_config['api_key']}"
            f"&country={news_config['country']}"
            f"&language={news_config['language']}"
            f"&size={news_config['max_articles']}"
        )
        
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"News API error: {response.text}")
        
        articles = response.json()['results']
        print(f"Processing {len(articles)} new articles...")
        
        for article in tqdm(articles):
            article_data = {
                'title': article['title'],
                'content': article['content'],
                'source': article['source_id'],
                'url': article['link'],
                'authors': article.get('creator', []),
                'published_date': article.get('pubDate', datetime.now().isoformat())
            }
            
            article_id = self.graph.create_article_node(article_data)
            self.graph.create_relationships(article_id)
    
    def analyze_article(self, 
                       article_content: str,
                       source: str,
                       date: Optional[str] = None) -> Dict:
        """
        Analyze bias in an article using the RAG system.
        
        Args:
            article_content: Content of the article
            source: Source of the article
            date: Optional publication date
        
        Returns:
            Dictionary containing bias analysis results
        """
        return self.rag.analyze_article_bias(
            article_content,
            source,
            date
        )
    
    def analyze_source(self,
                      source: str,
                      days: int = 30) -> Dict:
        """
        Analyze bias patterns for a news source.
        
        Args:
            source: Name of the news source
            days: Number of days to analyze
        
        Returns:
            Dictionary containing source analysis results
        """
        end_date = datetime.now().isoformat()
        start_date = (
            datetime.now() - pd.Timedelta(days=days)
        ).isoformat()
        
        return self.rag.analyze_source_evolution(
            source,
            start_date,
            end_date
        )
    
    def get_similar_articles(self,
                           article_content: str,
                           limit: int = 5) -> List[Dict]:
        """
        Find articles similar to the provided content.
        
        Args:
            article_content: Content to find similar articles for
            limit: Maximum number of results
        
        Returns:
            List of similar articles with metadata
        """
        return self.graph.find_similar_articles(article_content, limit)
    
    def close(self):
        """Clean up resources."""
        self.graph.close()

def main():
    """Main entry point for BiasLens."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BiasLens News Bias Analysis')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--dataset', help='Path to dataset CSV file')
    parser.add_argument('--fetch', action='store_true', 
                       help='Fetch and process current news')
    parser.add_argument('--analyze-source', help='Analyze a specific news source')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days for source analysis')
    
    args = parser.parse_args()
    
    try:
        biaslens = BiasLens(args.config)
        
        if args.dataset:
            biaslens.process_dataset(args.dataset)
        
        if args.fetch:
            biaslens.fetch_and_process_news()
        
        if args.analyze_source:
            analysis = biaslens.analyze_source(args.analyze_source, args.days)
            print("\nSource Analysis Results:")
            print(f"Average Bias: {analysis['avg_bias']:.2f}")
            print(f"Total Articles: {analysis['total_articles']}")
            print("\nBias Evolution (7-day rolling average):")
            for point in analysis['rolling_average']:
                print(f"{point['date']}: {point['avg_bias']:.2f}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        if 'biaslens' in locals():
            biaslens.close()

if __name__ == "__main__":
    main()
