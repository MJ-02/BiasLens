import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Typography,
  Box,
  Chip,
  TextField,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  CircularProgress,
} from '@mui/material';
import { format } from 'date-fns';

// Mock data - Replace with actual API call
const mockArticles = [
  {
    id: 1,
    title: 'Climate Change Impact on Global Economy',
    source: 'EcoNews',
    author: 'John Smith',
    date: '2024-02-12T10:30:00',
    image: 'https://source.unsplash.com/random/800x400?climate',
    topics: ['Climate', 'Economy', 'Global'],
    bias_score: 0.2,
  },
  // Add more mock articles...
];

function getBiasColor(score) {
  // Convert -1 to 1 scale to color
  const normalizedScore = (score + 1) / 2; // Convert to 0-1 scale
  const hue = normalizedScore * 120; // 0 is red, 120 is green
  return `hsl(${hue}, 70%, 50%)`;
}

function NewsFeed() {
  const navigate = useNavigate();
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [sourceFilter, setSourceFilter] = useState('all');
  const [topicFilter, setTopicFilter] = useState('all');

  // Get unique sources and topics from articles
  const sources = [...new Set(articles.map(article => article.source))];
  const topics = [...new Set(articles.flatMap(article => article.topics))];

  useEffect(() => {
    // Replace with actual API call
    const fetchArticles = async () => {
      try {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));
        setArticles(mockArticles);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching articles:', error);
        setLoading(false);
      }
    };

    fetchArticles();
  }, []);

  // Filter articles based on search term and filters
  const filteredArticles = articles.filter(article => {
    const matchesSearch = article.title.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesSource = sourceFilter === 'all' || article.source === sourceFilter;
    const matchesTopic = topicFilter === 'all' || article.topics.includes(topicFilter);
    return matchesSearch && matchesSource && matchesTopic;
  });

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="xl">
      {/* Filters */}
      <Box sx={{ mb: 4, mt: 2 }}>
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Search Articles"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Source</InputLabel>
              <Select
                value={sourceFilter}
                label="Source"
                onChange={(e) => setSourceFilter(e.target.value)}
              >
                <MenuItem value="all">All Sources</MenuItem>
                {sources.map(source => (
                  <MenuItem key={source} value={source}>{source}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Topic</InputLabel>
              <Select
                value={topicFilter}
                label="Topic"
                onChange={(e) => setTopicFilter(e.target.value)}
              >
                <MenuItem value="all">All Topics</MenuItem>
                {topics.map(topic => (
                  <MenuItem key={topic} value={topic}>{topic}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </Box>

      {/* Articles Grid */}
      <Grid container spacing={3}>
        {filteredArticles.map((article) => (
          <Grid item xs={12} sm={6} md={4} key={article.id}>
            <Card
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                cursor: 'pointer',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  transition: 'transform 0.2s ease-in-out',
                },
              }}
              onClick={() => navigate(`/article/${article.id}`)}
            >
              <CardMedia
                component="img"
                height="200"
                image={article.image}
                alt={article.title}
              />
              <CardContent sx={{ flexGrow: 1 }}>
                <Typography gutterBottom variant="h6" component="h2">
                  {article.title}
                </Typography>
                <Box sx={{ mb: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    {article.source} • {format(new Date(article.date), 'MMM d, yyyy')}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                  {article.topics.map((topic) => (
                    <Chip
                      key={topic}
                      label={topic}
                      size="small"
                      sx={{ backgroundColor: 'rgba(0,0,0,0.08)' }}
                    />
                  ))}
                </Box>
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                  }}
                >
                  <Typography variant="body2">Bias Score:</Typography>
                  <Box
                    sx={{
                      width: 40,
                      height: 24,
                      borderRadius: 1,
                      backgroundColor: getBiasColor(article.bias_score),
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <Typography variant="body2" sx={{ color: 'white' }}>
                      {article.bias_score.toFixed(1)}
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
}

export default NewsFeed;
