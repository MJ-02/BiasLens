# BiasLens: Advanced News Analysis Platform

BiasLens is a sophisticated platform that combines graph-based analysis with AI to help users understand media bias in news articles. Using GraphRAG (Graph-based Retrieval Augmented Generation) technology, it provides contextual analysis of news articles while visualizing relationships between articles, authors, sources, and topics.

## Features

- **Real-Time News Analysis**: Stay updated with the latest news while understanding potential biases through our advanced analysis system
- **Interactive Visualizations**: Explore relationships between articles, authors, sources, and topics through dynamic graph visualizations
- **Contextual Analysis**: Understand articles in the context of related content and historical patterns
- **Source Bias Tracking**: Monitor and analyze bias patterns across different news sources over time
- **Topic-Based Insights**: Explore how different sources cover the same topics
- **GraphRAG Technology**: Leverage graph-based retrieval for more nuanced and contextual bias analysis

## Architecture

BiasLens uses a modern, containerized architecture:

- **Frontend**: React-based web application with Material-UI and D3.js visualizations
- **Backend**: FastAPI server with GraphRAG implementation
- **Database**: Neo4j graph database for storing and querying article relationships
- **AI Integration**: OpenAI GPT-4 for advanced text analysis
- **Embeddings**: Sentence transformers for semantic similarity analysis

## Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.9+ (for local development)
- Neo4j 4.4+
- OpenAI API key
- News API key (from newsdata.io)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/biaslens.git
cd biaslens
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh start
```

4. Access the platform:
- Frontend: http://localhost:3000
- API Documentation: http://localhost:5000/docs
- Neo4j Browser: http://localhost:7474

## Development Setup

### Frontend

```bash
cd frontend
npm install
npm start
```

### Backend

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
uvicorn src.api.server:app --reload
```

### Running Tests

```bash
pytest tests/
```

## Docker Deployment

The platform is containerized using Docker for easy deployment:

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## API Documentation

The backend API provides comprehensive endpoints for:

- Article retrieval and analysis
- Graph data access
- Bias analysis
- Topic and source statistics

Full API documentation is available at `http://localhost:5000/docs` when running the server.

## Graph Visualization

BiasLens provides two types of graph visualizations:

1. **Article Context Graph**: Shows relationships between a specific article and related entities
2. **Global Knowledge Graph**: Visualizes the entire network of articles, authors, sources, and topics

### Graph Features:
- Interactive navigation
- Node filtering
- Relationship exploration
- Temporal analysis
- Bias pattern visualization

## Configuration

Key configuration files:

- `config/default_config.yaml`: Default configuration settings
- `.env`: Environment variables
- `docker-compose.yml`: Container orchestration
- `frontend/nginx.conf`: Frontend server configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and development process.

## Monitoring

The platform includes monitoring endpoints:

- Health checks: `http://localhost:5000/health`
- Metrics: `http://localhost:9090` (when enabled)
- Neo4j metrics: Available through the Neo4j Browser

## Troubleshooting

Common issues and solutions:

1. **Neo4j Connection Issues**:
   - Verify Neo4j is running: `docker-compose ps`
   - Check credentials in `.env`
   - Ensure ports are not in use

2. **API Key Issues**:
   - Verify API keys in `.env`
   - Check API rate limits
   - Ensure proper key permissions

3. **Performance Issues**:
   - Check Neo4j memory settings
   - Verify cache configuration
   - Monitor system resources

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original dataset from [Article Bias Prediction](https://github.com/ramybaly/Article-Bias-Prediction)
- Research paper: [Article Bias Detection Using Graph Neural Networks](https://arxiv.org/abs/2010.05338)
- Neo4j team for graph database technology
- OpenAI for GPT-4 API
