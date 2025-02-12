import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Articles API
export const articlesApi = {
  // Get latest articles with optional filters
  getArticles: async (params = {}) => {
    const response = await api.get('/articles', { params });
    return response.data;
  },

  // Get single article by ID with its subgraph
  getArticle: async (id) => {
    const response = await api.get(`/articles/${id}`);
    return response.data;
  },

  // Get article topics and entities
  getArticleAnalysis: async (id) => {
    const response = await api.get(`/articles/${id}/analysis`);
    return response.data;
  },

  // Get similar articles
  getSimilarArticles: async (id) => {
    const response = await api.get(`/articles/${id}/similar`);
    return response.data;
  },
};

// Graph API
export const graphApi = {
  // Get subgraph for an article
  getArticleSubgraph: async (id) => {
    const response = await api.get(`/graph/article/${id}`);
    return response.data;
  },

  // Get global graph data with optional filters
  getGlobalGraph: async (params = {}) => {
    const response = await api.get('/graph/global', { params });
    return response.data;
  },

  // Get graph data for a specific topic
  getTopicGraph: async (topic) => {
    const response = await api.get(`/graph/topic/${encodeURIComponent(topic)}`);
    return response.data;
  },

  // Get graph data for a specific source
  getSourceGraph: async (source) => {
    const response = await api.get(`/graph/source/${encodeURIComponent(source)}`);
    return response.data;
  },
};

// Analysis API
export const analysisApi = {
  // Get bias analysis for a source
  getSourceBiasAnalysis: async (source, params = {}) => {
    const response = await api.get(
      `/analysis/source/${encodeURIComponent(source)}`,
      { params }
    );
    return response.data;
  },

  // Get bias analysis for a topic
  getTopicBiasAnalysis: async (topic, params = {}) => {
    const response = await api.get(
      `/analysis/topic/${encodeURIComponent(topic)}`,
      { params }
    );
    return response.data;
  },

  // Get temporal bias analysis
  getTemporalAnalysis: async (params = {}) => {
    const response = await api.get('/analysis/temporal', { params });
    return response.data;
  },
};

// Topics API
export const topicsApi = {
  // Get all topics
  getTopics: async () => {
    const response = await api.get('/topics');
    return response.data;
  },

  // Get topic details with related articles
  getTopicDetails: async (topic) => {
    const response = await api.get(`/topics/${encodeURIComponent(topic)}`);
    return response.data;
  },
};

// Sources API
export const sourcesApi = {
  // Get all sources
  getSources: async () => {
    const response = await api.get('/sources');
    return response.data;
  },

  // Get source details with bias history
  getSourceDetails: async (source) => {
    const response = await api.get(`/sources/${encodeURIComponent(source)}`);
    return response.data;
  },
};

// Error handler middleware
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle different types of errors
    if (error.response) {
      // Server responded with error status
      console.error('API Error:', error.response.data);
      
      // Handle specific error codes
      switch (error.response.status) {
        case 401:
          // Handle unauthorized
          break;
        case 404:
          // Handle not found
          break;
        case 500:
          // Handle server error
          break;
        default:
          // Handle other errors
          break;
      }
    } else if (error.request) {
      // Request made but no response received
      console.error('Network Error:', error.request);
    } else {
      // Error in request configuration
      console.error('Request Error:', error.message);
    }

    return Promise.reject(error);
  }
);

export default api;
