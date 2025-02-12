import os
from typing import Dict, Optional
from dotenv import load_dotenv
import yaml

class BiasLensConfig:
    """Configuration manager for BiasLens."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from environment variables and config file.
        
        Args:
            config_path: Optional path to YAML config file
        """
        # Load environment variables
        load_dotenv()
        
        # Load config file if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
    
    @property
    def neo4j_config(self) -> Dict:
        """Get Neo4j database configuration."""
        return {
            'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
            'password': os.getenv('NEO4J_PASSWORD', 'password'),
        }
    
    @property
    def openai_config(self) -> Dict:
        """Get OpenAI configuration."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        return {
            'api_key': api_key,
            'model_name': self.config.get('openai', {}).get('model_name', 'gpt-4'),
            'temperature': self.config.get('openai', {}).get('temperature', 0.7),
            'max_tokens': self.config.get('openai', {}).get('max_tokens', 1000),
        }
    
    @property
    def news_api_config(self) -> Dict:
        """Get News API configuration."""
        api_key = os.getenv('NEWS_API_KEY')
        if not api_key:
            raise ValueError("News API key not found in environment variables")
        
        return {
            'api_key': api_key,
            'country': self.config.get('news_api', {}).get('country', 'us'),
            'language': self.config.get('news_api', {}).get('language', 'en'),
            'max_articles': self.config.get('news_api', {}).get('max_articles', 50),
        }
    
    @property
    def embedding_config(self) -> Dict:
        """Get embedding model configuration."""
        return {
            'model_name': self.config.get('embeddings', {}).get(
                'model_name', 'all-MiniLM-L6-v2'
            ),
            'cache_dir': self.config.get('embeddings', {}).get(
                'cache_dir', './.cache/embeddings'
            ),
        }
    
    @property
    def graph_config(self) -> Dict:
        """Get graph processing configuration."""
        return {
            'similarity_threshold': self.config.get('graph', {}).get(
                'similarity_threshold', 0.7
            ),
            'max_relationships': self.config.get('graph', {}).get(
                'max_relationships', 10
            ),
            'temporal_window_days': self.config.get('graph', {}).get(
                'temporal_window_days', 7
            ),
        }
    
    @property
    def analysis_config(self) -> Dict:
        """Get bias analysis configuration."""
        return {
            'min_confidence': self.config.get('analysis', {}).get(
                'min_confidence', 0.7
            ),
            'context_window': self.config.get('analysis', {}).get(
                'context_window', 5
            ),
            'max_hops': self.config.get('analysis', {}).get(
                'max_hops', 2
            ),
        }
    
    def get_default_config(self) -> Dict:
        """Get default configuration template."""
        return {
            'openai': {
                'model_name': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 1000,
            },
            'news_api': {
                'country': 'us',
                'language': 'en',
                'max_articles': 50,
            },
            'embeddings': {
                'model_name': 'all-MiniLM-L6-v2',
                'cache_dir': './.cache/embeddings',
            },
            'graph': {
                'similarity_threshold': 0.7,
                'max_relationships': 10,
                'temporal_window_days': 7,
            },
            'analysis': {
                'min_confidence': 0.7,
                'context_window': 5,
                'max_hops': 2,
            },
        }
    
    def save_default_config(self, path: str):
        """
        Save default configuration template to a file.
        
        Args:
            path: Path to save the configuration file
        """
        with open(path, 'w') as f:
            yaml.dump(self.get_default_config(), f, default_flow_style=False)
