from typing import Dict, List, Optional
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime
from ..graph.graph_retriever import GraphRetriever
from ..graph.enhanced_graph import EnhancedKnowledgeGraph

class BiasAnalysisPrompts:
    """Prompt templates for bias analysis."""
    
    BIAS_ANALYSIS = """
    Analyze the potential bias in the following article, considering the provided context:

    Article Content:
    {article_content}

    Source: {source}

    Context from similar articles:
    {cross_source_context}

    Historical context from same source:
    {source_context}

    Source bias patterns:
    {bias_patterns}

    Temporal context:
    {temporal_context}

    Please provide a detailed analysis including:
    1. Overall bias assessment (scale -1 to 1, where -1 is extremely left-leaning, 0 is neutral, 1 is extremely right-leaning)
    2. Key indicators of bias identified
    3. Comparison with other sources' coverage
    4. Historical bias patterns of the source
    5. Confidence score (0-1) for this assessment

    Analysis:
    """

class BiasLensRAG:
    """
    RAG (Retrieval Augmented Generation) engine for BiasLens.
    Combines graph-based retrieval with LLM analysis for comprehensive bias detection.
    """
    
    def __init__(self, 
                 graph: EnhancedKnowledgeGraph,
                 openai_api_key: str,
                 model_name: str = "gpt-4"):
        """
        Initialize the RAG engine.
        
        Args:
            graph: Instance of EnhancedKnowledgeGraph
            openai_api_key: OpenAI API key for LLM access
            model_name: Name of the OpenAI model to use
        """
        self.retriever = GraphRetriever(graph)
        self.llm = OpenAI(openai_api_key=openai_api_key, model_name=model_name)
        self.prompt = PromptTemplate(
            input_variables=[
                "article_content",
                "source",
                "cross_source_context",
                "source_context",
                "bias_patterns",
                "temporal_context"
            ],
            template=BiasAnalysisPrompts.BIAS_ANALYSIS
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _format_articles_context(self, articles: List[Dict]) -> str:
        """Format a list of articles into a readable context string."""
        context = []
        for article in articles:
            context.append(
                f"- {article['title']} (Source: {article['source']}, "
                f"Similarity: {article['score']:.2f})"
            )
        return "\n".join(context)
    
    def _format_bias_patterns(self, patterns: Dict) -> str:
        """Format bias pattern analysis into a readable string."""
        return (
            f"Average bias: {patterns['avg_bias']:.2f}\n"
            f"Standard deviation: {patterns['std_bias']:.2f}\n"
            "Distribution:\n" +
            "\n".join([
                f"- Bias {d['bias']}: {d['count']} articles"
                for d in patterns['distribution']
            ])
        )
    
    def analyze_article_bias(self, 
                           article_content: str,
                           source: str,
                           date: Optional[str] = None) -> Dict:
        """
        Perform comprehensive bias analysis on an article using RAG.
        
        Args:
            article_content: Content of the article to analyze
            source: Source of the article
            date: Publication date (ISO format), defaults to current date
        
        Returns:
            Dictionary containing:
            - bias_score: Float between -1 and 1
            - confidence: Float between 0 and 1
            - analysis: Detailed analysis text
            - indicators: List of bias indicators
            - context_used: Summary of context used in analysis
        """
        # Get various types of context
        cross_source, source_articles = self.retriever.get_bias_context(
            article_content, 
            source
        )
        
        bias_patterns = self.retriever.get_source_bias_patterns(source)
        
        date = date or datetime.now().isoformat()
        temporal_context = self.retriever.get_temporal_context(
            article_content,
            date
        )
        
        # Format context for prompt
        context = {
            "article_content": article_content,
            "source": source,
            "cross_source_context": self._format_articles_context(cross_source),
            "source_context": self._format_articles_context(source_articles),
            "bias_patterns": self._format_bias_patterns(bias_patterns),
            "temporal_context": self._format_articles_context(temporal_context)
        }
        
        # Generate analysis
        result = self.chain.run(**context)
        
        # Parse the result
        # Note: This assumes a structured output format from the LLM
        # You might need to adjust the parsing based on actual output
        lines = result.strip().split("\n")
        bias_score = float(lines[0].split(":")[1].strip())
        confidence = float(lines[1].split(":")[1].strip())
        indicators = [
            line.strip("- ") for line in lines 
            if line.startswith("- ") and ":" not in line
        ]
        analysis = "\n".join(lines[lines.index("Analysis:") + 1:])
        
        return {
            "bias_score": bias_score,
            "confidence": confidence,
            "analysis": analysis,
            "indicators": indicators,
            "context_used": {
                "cross_source_articles": len(cross_source),
                "source_articles": len(source_articles),
                "temporal_articles": len(temporal_context)
            }
        }
    
    def analyze_source_evolution(self, 
                               source: str,
                               start_date: str,
                               end_date: str) -> Dict:
        """
        Analyze how a source's bias has evolved over time.
        
        Args:
            source: Name of the news source
            start_date: Start date for analysis (ISO format)
            end_date: End date for analysis (ISO format)
        
        Returns:
            Dictionary containing temporal bias analysis
        """
        with self.retriever.graph.driver.session() as session:
            result = session.run("""
                MATCH (a:Article)
                WHERE a.source = $source
                  AND datetime(a.published_date) >= datetime($start_date)
                  AND datetime(a.published_date) <= datetime($end_date)
                WITH a.published_date as date, a.bias as bias
                ORDER BY date
                RETURN date, bias
                """,
                source=source,
                start_date=start_date,
                end_date=end_date
            )
            
            data_points = [dict(record) for record in result]
            
            # Calculate rolling averages and trends
            window_size = 7  # 7-day rolling average
            rolling_avg = []
            
            for i in range(len(data_points) - window_size + 1):
                window = data_points[i:i + window_size]
                avg_bias = sum(p['bias'] for p in window) / window_size
                rolling_avg.append({
                    'date': window[-1]['date'],
                    'avg_bias': avg_bias
                })
            
            return {
                'raw_data': data_points,
                'rolling_average': rolling_avg,
                'total_articles': len(data_points),
                'avg_bias': sum(p['bias'] for p in data_points) / len(data_points)
            }
