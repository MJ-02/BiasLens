import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Paper,
  Typography,
  Box,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Switch,
  FormControlLabel,
  CircularProgress,
} from '@mui/material';
import ForceGraph3D from 'react-force-graph';

// Mock data - Replace with actual API call
const mockGraphData = {
  nodes: [
    // Articles
    { id: 'article1', label: 'Climate Change Impact', group: 'article', size: 1 },
    { id: 'article2', label: 'Economic Policy Review', group: 'article', size: 1 },
    // Authors
    { id: 'author1', label: 'John Smith', group: 'author', size: 2 },
    { id: 'author2', label: 'Jane Doe', group: 'author', size: 2 },
    // Outlets
    { id: 'outlet1', label: 'EcoNews', group: 'outlet', size: 3 },
    { id: 'outlet2', label: 'Global Times', group: 'outlet', size: 3 },
    // Topics
    { id: 'topic1', label: 'Climate', group: 'topic', size: 2 },
    { id: 'topic2', label: 'Economy', group: 'topic', size: 2 },
  ],
  links: [
    // Article - Author relationships
    { source: 'article1', target: 'author1', type: 'written_by' },
    { source: 'article2', target: 'author2', type: 'written_by' },
    // Article - Outlet relationships
    { source: 'article1', target: 'outlet1', type: 'published_by' },
    { source: 'article2', target: 'outlet2', type: 'published_by' },
    // Article - Topic relationships
    { source: 'article1', target: 'topic1', type: 'about' },
    { source: 'article2', target: 'topic2', type: 'about' },
  ],
};

function GlobalGraph() {
  const navigate = useNavigate();
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    showArticles: true,
    showAuthors: true,
    showOutlets: true,
    showTopics: true,
    timeRange: 30, // days
  });
  const [highlightNodes, setHighlightNodes] = useState(new Set());
  const [highlightLinks, setHighlightLinks] = useState(new Set());
  const [selectedNode, setSelectedNode] = useState(null);

  useEffect(() => {
    // Replace with actual API call
    const fetchGraphData = async () => {
      try {
        await new Promise(resolve => setTimeout(resolve, 1000));
        setGraphData(mockGraphData);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching graph data:', error);
        setLoading(false);
      }
    };

    fetchGraphData();
  }, []);

  const getNodeColor = useCallback(node => {
    if (highlightNodes.has(node)) {
      return '#f50057';
    }
    switch (node.group) {
      case 'article':
        return '#1976d2';
      case 'author':
        return '#2e7d32';
      case 'outlet':
        return '#ed6c02';
      case 'topic':
        return '#9c27b0';
      default:
        return '#757575';
    }
  }, [highlightNodes]);

  const handleNodeClick = useCallback(node => {
    if (node.group === 'article') {
      navigate(`/article/${node.id}`);
    }
    setSelectedNode(node);
  }, [navigate]);

  const handleNodeHover = useCallback((node) => {
    if (!node) {
      setHighlightNodes(new Set());
      setHighlightLinks(new Set());
      return;
    }

    const neighbors = new Set();
    const links = new Set();
    
    graphData.links.forEach(link => {
      if (link.source === node || link.target === node) {
        neighbors.add(link.source);
        neighbors.add(link.target);
        links.add(link);
      }
    });

    setHighlightNodes(neighbors);
    setHighlightLinks(links);
  }, [graphData]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="xl">
      <Grid container spacing={3}>
        {/* Controls */}
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Graph Controls
            </Typography>
            
            <Box sx={{ mb: 3 }}>
              <Typography gutterBottom>Time Range (Days)</Typography>
              <Slider
                value={filters.timeRange}
                min={1}
                max={90}
                onChange={(_, value) => setFilters(prev => ({ ...prev, timeRange: value }))}
                valueLabelDisplay="auto"
              />
            </Box>

            <Box sx={{ mb: 3 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={filters.showArticles}
                    onChange={(e) => setFilters(prev => ({ ...prev, showArticles: e.target.checked }))}
                  />
                }
                label="Show Articles"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={filters.showAuthors}
                    onChange={(e) => setFilters(prev => ({ ...prev, showAuthors: e.target.checked }))}
                  />
                }
                label="Show Authors"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={filters.showOutlets}
                    onChange={(e) => setFilters(prev => ({ ...prev, showOutlets: e.target.checked }))}
                  />
                }
                label="Show Outlets"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={filters.showTopics}
                    onChange={(e) => setFilters(prev => ({ ...prev, showTopics: e.target.checked }))}
                  />
                }
                label="Show Topics"
              />
            </Box>

            {selectedNode && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Selected Node:
                </Typography>
                <Typography variant="body2">
                  Type: {selectedNode.group}
                </Typography>
                <Typography variant="body2">
                  Label: {selectedNode.label}
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Graph Visualization */}
        <Grid item xs={12} md={9}>
          <Paper sx={{ height: 'calc(100vh - 200px)', position: 'relative' }}>
            <ForceGraph3D
              graphData={graphData}
              nodeLabel="label"
              nodeColor={getNodeColor}
              nodeRelSize={node => node.size * 5}
              linkWidth={link => highlightLinks.has(link) ? 2 : 1}
              linkColor={link => highlightLinks.has(link) ? '#f50057' : '#999'}
              onNodeClick={handleNodeClick}
              onNodeHover={handleNodeHover}
              enableNodeDrag={false}
              enableNavigationControls={true}
              showNavInfo={true}
            />
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
}

export default GlobalGraph;
