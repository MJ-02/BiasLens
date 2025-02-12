from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from ..driver import BiasLens
from ..utils.config import BiasLensConfig

app = FastAPI(
    title="BiasLens API",
    description="API for the BiasLens news bias analysis platform",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize BiasLens
config = BiasLensConfig("config/config.yaml")
biaslens = BiasLens(config)

# Models
class Article(BaseModel):
    id: str
    title: str
    content: str
    source: str
    author: str
    date: datetime
    topics: List[str]
    bias_score: float
    url: str

class GraphData(BaseModel):
    nodes: List[Dict]
    links: List[Dict]

class BiasAnalysis(BaseModel):
    bias_score: float
    confidence: float
    analysis: str
    indicators: List[str]
    context_used: Dict[str, int]

# Routes
@app.get("/api/articles", response_model=List[Article])
async def get_articles(
    source: Optional[str] = None,
    topic: Optional[str] = None,
    days: int = Query(default=30, ge=1, le=90),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=10, ge=1, le=50)
):
    """Get latest articles with optional filters."""
    try:
        # Get articles from Neo4j
        articles = biaslens.get_articles(
            source=source,
            topic=topic,
            days=days,
            skip=(page - 1) * limit,
            limit=limit
        )
        return articles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/articles/{article_id}", response_model=Article)
async def get_article(article_id: str):
    """Get single article by ID."""
    try:
        article = biaslens.get_article(article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        return article
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/articles/{article_id}/analysis", response_model=BiasAnalysis)
async def get_article_analysis(article_id: str):
    """Get bias analysis for an article."""
    try:
        article = biaslens.get_article(article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        
        analysis = biaslens.analyze_article(
            article['content'],
            article['source'],
            article['date']
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/articles/{article_id}/similar", response_model=List[Article])
async def get_similar_articles(
    article_id: str,
    limit: int = Query(default=5, ge=1, le=20)
):
    """Get similar articles."""
    try:
        similar = biaslens.get_similar_articles(article_id, limit)
        return similar
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph/article/{article_id}", response_model=GraphData)
async def get_article_subgraph(article_id: str):
    """Get subgraph for an article."""
    try:
        article = biaslens.get_article(article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        
        # Get article's local graph context
        graph_data = biaslens.graph.get_article_context(article_id)
        return graph_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph/global", response_model=GraphData)
async def get_global_graph(
    days: int = Query(default=30, ge=1, le=90),
    show_articles: bool = True,
    show_authors: bool = True,
    show_outlets: bool = True,
    show_topics: bool = True
):
    """Get global graph data with filters."""
    try:
        graph_data = biaslens.get_global_graph(
            days=days,
            include_articles=show_articles,
            include_authors=show_authors,
            include_outlets=show_outlets,
            include_topics=show_topics
        )
        return graph_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/source/{source}", response_model=Dict)
async def get_source_analysis(
    source: str,
    days: int = Query(default=30, ge=1, le=90)
):
    """Get bias analysis for a source."""
    try:
        analysis = biaslens.analyze_source(source, days)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/topics", response_model=List[str])
async def get_topics():
    """Get all topics."""
    try:
        topics = biaslens.get_topics()
        return topics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sources", response_model=List[str])
async def get_sources():
    """Get all sources."""
    try:
        sources = biaslens.get_sources()
        return sources
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    biaslens.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
