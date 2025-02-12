# BiasLens: Enhanced News Bias Analysis

BiasLens is an advanced system for analyzing bias in news articles using graph-based Retrieval Augmented Generation (GraphRAG). The system combines knowledge graph technology with large language models to provide comprehensive and contextual bias analysis.

## Features

- **Graph-Based Analysis**: Uses Neo4j to create and traverse relationships between articles
- **Contextual Bias Detection**: Considers multiple sources and historical patterns
- **Temporal Analysis**: Tracks bias evolution over time
- **Source Pattern Recognition**: Identifies consistent bias patterns in news sources
- **Real-Time Processing**: Fetches and analyzes current news articles
- **Comprehensive API**: Flexible interface for various analysis needs

## Architecture

BiasLens uses a sophisticated architecture combining several key components:

1. **Knowledge Graph (Neo4j)**
   - Stores articles and their relationships
   - Enables semantic similarity connections
   - Supports temporal and source-based analysis

2. **GraphRAG Engine**
   - Retrieves relevant context from the knowledge graph
   - Uses GPT-4 for analysis
   - Provides explainable bias assessments

3. **Embedding System**
   - Generates semantic embeddings for articles
   - Enables similarity-based retrieval
   - Supports multiple embedding models

## Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Install Neo4j**
- Download and install [Neo4j](https://neo4j.com/download/)
- Create a new database
- Note down the connection details

3. **Configuration**
- Copy `config/default_config.yaml` to `config/config.yaml`
- Update the configuration with your settings
- Set up environment variables:
  ```bash
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USERNAME=neo4j
  NEO4J_PASSWORD=your-password
  OPENAI_API_KEY=your-openai-key
  NEWS_API_KEY=your-newsdata-key
  ```

## Usage

1. **Process Historical Dataset**
```bash
python -m src.driver --config config/config.yaml --dataset path/to/dataset.csv
```

2. **Fetch Current News**
```bash
python -m src.driver --config config/config.yaml --fetch
```

3. **Analyze Source Bias**
```bash
python -m src.driver --config config/config.yaml --analyze-source "source-name" --days 30
```

## Python API

```python
from src.driver import BiasLens

# Initialize
biaslens = BiasLens("config/config.yaml")

# Analyze an article
analysis = biaslens.analyze_article(
    article_content="Article text here",
    source="News Source Name",
    date="2024-02-12"
)

# Get similar articles
similar = biaslens.get_similar_articles(
    article_content="Article text here",
    limit=5
)

# Analyze source patterns
source_analysis = biaslens.analyze_source(
    source="News Source Name",
    days=30
)

# Clean up
biaslens.close()
```

## Output Format

The system provides detailed analysis results:

```python
{
    "bias_score": 0.25,  # Range: -1 (left) to 1 (right)
    "confidence": 0.85,  # Range: 0 to 1
    "analysis": "Detailed analysis text...",
    "indicators": [
        "Use of emotional language",
        "Source selection bias",
        # ...
    ],
    "context_used": {
        "cross_source_articles": 5,
        "source_articles": 3,
        "temporal_articles": 10
    }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original dataset from [Article Bias Prediction](https://github.com/ramybaly/Article-Bias-Prediction)
- Based on research paper: [Article Bias Detection Using Graph Neural Networks](https://arxiv.org/abs/2010.05338)
