import React from 'react';
import { Box, Container, Typography, Link, Divider } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';

function Footer() {
  return (
    <Box
      component="footer"
      sx={{
        py: 3,
        px: 2,
        mt: 'auto',
        backgroundColor: (theme) =>
          theme.palette.mode === 'light'
            ? theme.palette.grey[200]
            : theme.palette.grey[800],
      }}
    >
      <Container maxWidth="lg">
        <Divider sx={{ mb: 3 }} />
        
        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', sm: 'row' },
            justifyContent: 'space-between',
            alignItems: { xs: 'center', sm: 'flex-start' },
            gap: 2,
          }}
        >
          {/* Links Section */}
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: { xs: 'center', sm: 'flex-start' },
            }}
          >
            <Typography variant="h6" color="text.primary" gutterBottom>
              BiasLens
            </Typography>
            <Link
              component={RouterLink}
              to="/"
              color="text.secondary"
              sx={{ mb: 0.5 }}
            >
              News Feed
            </Link>
            <Link
              component={RouterLink}
              to="/graph"
              color="text.secondary"
              sx={{ mb: 0.5 }}
            >
              Global Graph
            </Link>
            <Link
              component={RouterLink}
              to="/about"
              color="text.secondary"
              sx={{ mb: 0.5 }}
            >
              About
            </Link>
          </Box>

          {/* Resources Section */}
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: { xs: 'center', sm: 'flex-start' },
            }}
          >
            <Typography variant="h6" color="text.primary" gutterBottom>
              Resources
            </Typography>
            <Link
              href="https://github.com/ramybaly/Article-Bias-Prediction"
              color="text.secondary"
              target="_blank"
              rel="noopener noreferrer"
              sx={{ mb: 0.5 }}
            >
              Dataset
            </Link>
            <Link
              href="https://arxiv.org/abs/2010.05338"
              color="text.secondary"
              target="_blank"
              rel="noopener noreferrer"
              sx={{ mb: 0.5 }}
            >
              Research Paper
            </Link>
            <Link
              href="https://github.com/yourusername/biaslens"
              color="text.secondary"
              target="_blank"
              rel="noopener noreferrer"
              sx={{ mb: 0.5 }}
            >
              GitHub
            </Link>
          </Box>

          {/* Contact Section */}
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: { xs: 'center', sm: 'flex-start' },
            }}
          >
            <Typography variant="h6" color="text.primary" gutterBottom>
              Contact
            </Typography>
            <Link
              href="mailto:contact@biaslens.com"
              color="text.secondary"
              sx={{ mb: 0.5 }}
            >
              Email Us
            </Link>
            <Link
              href="https://twitter.com/biaslens"
              color="text.secondary"
              target="_blank"
              rel="noopener noreferrer"
              sx={{ mb: 0.5 }}
            >
              Twitter
            </Link>
          </Box>
        </Box>

        {/* Copyright */}
        <Typography
          variant="body2"
          color="text.secondary"
          align="center"
          sx={{ mt: 3 }}
        >
          {'Copyright © '}
          <Link component={RouterLink} to="/" color="inherit">
            BiasLens
          </Link>{' '}
          {new Date().getFullYear()}
          {'. All rights reserved.'}
        </Typography>
      </Container>
    </Box>
  );
}

export default Footer;
