import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Chip,
  CircularProgress,
  Divider,
} from '@mui/material';
import ForceGraph2D from 'react-force-graph';

// Mock data - Replace with actual API call
const mockArticle = {
  id: 1,
  title: 'Climate Change Impact on Global Economy',
  content: `Climate change poses significant challenges to the global economy, according to recent studies. 
  Scientists warn that rising temperatures could lead to substantial economic losses across various sectors. 
  Industry experts suggest that immediate action is necessary to mitigate these impacts. 
  However, some economists argue about the cost-effectiveness of proposed solutions.`,
  source: 'EcoNews',
  author: 'John Smith',
  date: '2024-02-12T10:30:00',
  topics: ['Climate', 'Economy', 'Global'],
  bias_score: 0.2,
  subgraph: {
    nodes: [
      { id: 'article', label: 'Current Article', group: 'article' },
      { id: 'climate', label: 'Climate Change', group: 'topic' },
      { id: 'economy', label: 'Economy', group: 'topic' },
      { id: 'scientists', label: 'Scientists', group: 'entity' },
      { id: 'economists', label: 'Economists', group: 'entity' },
    ],
    links: [
      { source: 'article', target: 'climate' },
      { source: 'article', target: 'economy' },
      { source: 'climate', target: 'scientists' },
      { source: 'economy', target: 'economists' },
    ],
  },
};

function ArticleView() {
  const { id } = useParams();
  const [article, setArticle] = useState(null);
  const [loading, setLoading] = useState(true);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });

  useEffect(() => {
    // Replace with actual API call
    const fetchArticle = async () => {
      try {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));
        setArticle(mockArticle);
        setGraphData(mockArticle.subgraph);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching article:', error);
        setLoading(false);
      }
    };

    fetchArticle();
  }, [id]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!article) {
    return (
      <Container>
        <Typography variant="h5" sx={{ mt: 4 }}>
          Article not found
        </Typography>
      </Container>
    );
  }

  const getNodeColor = (node) => {
    switch (node.group) {
      case 'article':
        return '#1976d2';
      case 'topic':
        return '#2e7d32';
      case 'entity':
        return '#d32f2f';
      default:
        return '#757575';
    }
  };

  return (
    <Container maxWidth="xl">
      <Grid container spacing={3}>
        {/* Article Content */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h4" gutterBottom>
              {article.title}
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle1" color="text.secondary">
                {article.source} • By {article.author}
              </Typography>
              <Typography variant="subtitle2" color="text.secondary">
                {new Date(article.date).toLocaleDateString()}
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 3 }}>
              {article.topics.map((topic) => (
                <Chip
                  key={topic}
                  label={topic}
                  sx={{ backgroundColor: 'rgba(0,0,0,0.08)' }}
                />
              ))}
            </Box>

            <Typography variant="body1" paragraph>
              {article.content}
            </Typography>

            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 2,
                mt: 3,
                p: 2,
                bgcolor: 'background.default',
                borderRadius: 1,
              }}
            >
              <Typography variant="subtitle1">Bias Analysis:</Typography>
              <Box
                sx={{
                  px: 2,
                  py: 1,
                  borderRadius: 1,
                  bgcolor: article.bias_score > 0 ? 'success.light' : 'error.light',
                  color: 'white',
                }}
              >
                Score: {article.bias_score.toFixed(2)}
              </Box>
            </Box>
          </Paper>
        </Grid>

        {/* Graph Visualization */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Article Context Graph
            </Typography>
            
            <Box sx={{ height: 400, border: '1px solid rgba(0,0,0,0.1)', borderRadius: 1 }}>
              <ForceGraph2D
                graphData={graphData}
                nodeLabel="label"
                nodeColor={node => getNodeColor(node)}
                nodeRelSize={6}
                linkWidth={2}
                linkColor={() => '#999'}
                backgroundColor="#ffffff"
              />
            </Box>

            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                Legend:
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#1976d2' }} />
                  <Typography variant="body2">Current Article</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#2e7d32' }} />
                  <Typography variant="body2">Topics</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#d32f2f' }} />
                  <Typography variant="body2">Entities</Typography>
                </Box>
              </Box>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
}

export default ArticleView;
