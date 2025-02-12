import React from 'react';
import {
  Container,
  Typography,
  Paper,
  Box,
  Grid,
  Card,
  CardContent,
  CardMedia,
} from '@mui/material';
import {
  Timeline as TimelineIcon,
  GraphicEq as GraphIcon,
  Psychology as AIIcon,
  Language as GlobalIcon,
} from '@mui/icons-material';

function About() {
  const features = [
    {
      icon: <TimelineIcon sx={{ fontSize: 40 }} />,
      title: 'Real-Time News Analysis',
      description: 'Stay updated with the latest news while understanding potential biases in real-time through our advanced analysis system.',
    },
    {
      icon: <GraphIcon sx={{ fontSize: 40 }} />,
      title: 'Graph-Based Context',
      description: 'Visualize connections between articles, authors, sources, and topics through our interactive knowledge graph.',
    },
    {
      icon: <AIIcon sx={{ fontSize: 40 }} />,
      title: 'AI-Powered Insights',
      description: 'Leverage advanced machine learning and natural language processing to detect and analyze media bias patterns.',
    },
    {
      icon: <GlobalIcon sx={{ fontSize: 40 }} />,
      title: 'Comprehensive Coverage',
      description: 'Access news from various sources and understand different perspectives on important topics.',
    },
  ];

  return (
    <Container maxWidth="lg">
      {/* Hero Section */}
      <Box sx={{ mb: 6, textAlign: 'center' }}>
        <Typography
          variant="h2"
          gutterBottom
          sx={{
            fontWeight: 700,
            background: 'linear-gradient(45deg, #1976d2 30%, #21CBF3 90%)',
            backgroundClip: 'text',
            textFillColor: 'transparent',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          Understanding News Bias
        </Typography>
        <Typography variant="h5" color="text.secondary" paragraph>
          BiasLens helps you navigate media bias through advanced graph-based analysis
          and AI-powered insights.
        </Typography>
      </Box>

      {/* Features Grid */}
      <Grid container spacing={4} sx={{ mb: 6 }}>
        {features.map((feature, index) => (
          <Grid item xs={12} md={6} key={index}>
            <Card
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  transition: 'transform 0.2s ease-in-out',
                },
              }}
            >
              <CardContent>
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    mb: 2,
                    color: 'primary.main',
                  }}
                >
                  {feature.icon}
                  <Typography variant="h5" component="h2" sx={{ ml: 1 }}>
                    {feature.title}
                  </Typography>
                </Box>
                <Typography variant="body1" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* How It Works Section */}
      <Paper sx={{ p: 4, mb: 6 }}>
        <Typography variant="h4" gutterBottom>
          How It Works
        </Typography>
        <Typography variant="body1" paragraph>
          BiasLens uses a sophisticated graph-based Retrieval Augmented Generation (GraphRAG) 
          system to analyze news articles and detect potential biases. Our approach combines:
        </Typography>
        <Box component="ul" sx={{ pl: 2 }}>
          <Typography component="li" sx={{ mb: 1 }}>
            Knowledge Graph Technology to map relationships between articles, authors,
            sources, and topics
          </Typography>
          <Typography component="li" sx={{ mb: 1 }}>
            Advanced Natural Language Processing to understand article content and context
          </Typography>
          <Typography component="li" sx={{ mb: 1 }}>
            Machine Learning algorithms to detect patterns and analyze bias
          </Typography>
          <Typography component="li" sx={{ mb: 1 }}>
            Interactive Visualizations to help users understand complex relationships
          </Typography>
        </Box>
      </Paper>

      {/* Research Background */}
      <Paper sx={{ p: 4, mb: 6 }}>
        <Typography variant="h4" gutterBottom>
          Research Background
        </Typography>
        <Typography variant="body1" paragraph>
          BiasLens is based on cutting-edge research in media bias detection and analysis.
          Our approach builds upon the work presented in the paper "Article Bias Detection
          Using Graph Neural Networks" and extends it with advanced GraphRAG capabilities.
        </Typography>
        <Typography variant="body1">
          The system continuously learns and improves its analysis by incorporating new
          articles and user feedback, while maintaining transparency in its methodology
          and findings.
        </Typography>
      </Paper>

      {/* Team Section */}
      <Paper sx={{ p: 4 }}>
        <Typography variant="h4" gutterBottom>
          Our Mission
        </Typography>
        <Typography variant="body1" paragraph>
          We believe in promoting media literacy and critical thinking by providing tools
          that help readers understand potential biases in news coverage. Our goal is to
          empower readers to:
        </Typography>
        <Box component="ul" sx={{ pl: 2 }}>
          <Typography component="li" sx={{ mb: 1 }}>
            Make informed decisions about their news consumption
          </Typography>
          <Typography component="li" sx={{ mb: 1 }}>
            Understand different perspectives on important topics
          </Typography>
          <Typography component="li" sx={{ mb: 1 }}>
            Recognize patterns of bias in media coverage
          </Typography>
          <Typography component="li" sx={{ mb: 1 }}>
            Access a more balanced view of current events
          </Typography>
        </Box>
      </Paper>
    </Container>
  );
}

export default About;
