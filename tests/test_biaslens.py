import os
import pytest
from datetime import datetime
from src.driver import BiasLens
from src.utils.config import BiasLensConfig
from src.graph.enhanced_graph import EnhancedKnowledgeGraph
from src.rag.rag_engine import BiasLensRAG

# Sample test data
SAMPLE_ARTICLE = {
    'title': 'Test Article',
    'content': '''
    This is a test article about climate change. Scientists report increasing global temperatures.
    Some politicians argue about the economic impact of environmental regulations.
    Environmental groups advocate for immediate action.
    ''',
    'source': 'Test News',
    'url': 'http://test.com/article1',
    'authors': ['John Doe'],
    'published_date': datetime.now().isoformat()
}

@pytest.fixture
def config():
    """Create a test configuration."""
    return BiasLensConfig('config/default_config.yaml')

@pytest.fixture
def graph(config):
    """Create a test graph instance."""
    return EnhancedKnowledgeGraph(
        **config.neo4j_config
    )

@pytest.fixture
def biaslens(config):
    """Create a test BiasLens instance."""
    return BiasLens('config/default_config.yaml')

def test_config_loading(config):
    """Test configuration loading."""
    assert config.neo4j_config['uri'] == 'bolt://localhost:7687'
    assert config.graph_config['similarity_threshold'] == 0.7
    assert config.analysis_config['min_confidence'] == 0.7

def test_graph_operations(graph):
    """Test basic graph operations."""
    # Create article node
    article_id = graph.create_article_node(SAMPLE_ARTICLE)
    assert article_id is not None
    
    # Create relationships
    graph.create_relationships(article_id, threshold=0.7)
    
    # Find similar articles
    similar = graph.find_similar_articles(SAMPLE_ARTICLE['content'], limit=5)
    assert isinstance(similar, list)

def test_article_analysis(biaslens):
    """Test article bias analysis."""
    analysis = biaslens.analyze_article(
        SAMPLE_ARTICLE['content'],
        SAMPLE_ARTICLE['source'],
        SAMPLE_ARTICLE['published_date']
    )
    
    assert isinstance(analysis, dict)
    assert 'bias_score' in analysis
    assert 'confidence' in analysis
    assert 'analysis' in analysis
    assert 'indicators' in analysis
    
    assert -1 <= analysis['bias_score'] <= 1
    assert 0 <= analysis['confidence'] <= 1

def test_source_analysis(biaslens):
    """Test source bias analysis."""
    analysis = biaslens.analyze_source(
        SAMPLE_ARTICLE['source'],
        days=30
    )
    
    assert isinstance(analysis, dict)
    assert 'raw_data' in analysis
    assert 'rolling_average' in analysis
    assert 'total_articles' in analysis
    assert 'avg_bias' in analysis

def test_similar_articles(biaslens):
    """Test similar articles retrieval."""
    similar = biaslens.get_similar_articles(
        SAMPLE_ARTICLE['content'],
        limit=5
    )
    
    assert isinstance(similar, list)
    assert len(similar) <= 5
    
    if similar:
        assert 'title' in similar[0]
        assert 'source' in similar[0]
        assert 'url' in similar[0]
        assert 'score' in similar[0]

def test_error_handling(biaslens):
    """Test error handling."""
    with pytest.raises(ValueError):
        biaslens.analyze_article('', '')  # Empty content and source
    
    with pytest.raises(ValueError):
        biaslens.analyze_source('')  # Empty source name

def test_cleanup(biaslens):
    """Test resource cleanup."""
    biaslens.close()
    # Verify no active connections remain
    # This might need to be adapted based on how Neo4j handles connection status

if __name__ == '__main__':
    pytest.main([__file__])
