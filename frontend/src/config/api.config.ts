/**
 * API Configuration
 * Controls whether to use live data or mock data
 */

export const API_CONFIG = {
    // Set to true to use live data from backend
    USE_LIVE_DATA: true,

    // API base URL
    API_BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',

    // WebSocket URL
    WS_BASE_URL: process.env.REACT_APP_WS_URL || 'ws://localhost:8000',

    // API endpoints
    ENDPOINTS: {
        // Market data
        MARKET_DATA: '/api/v1/market-data',
        MARKET_QUOTE: (symbol: string) => `/api/v1/market-data/${symbol}/quote`,
        MARKET_HISTORICAL: (symbol: string) => `/api/v1/market-data/${symbol}/historical`,
        MARKET_QUOTES: '/api/v1/market-data/quotes',
        MARKET_STATUS: '/api/v1/market-data/status/market',
        MARKET_SEARCH: (query: string) => `/api/v1/market-data/search/${query}`,
        MARKET_OPTIONS: (symbol: string) => `/api/v1/market-data/${symbol}/options`,
        MARKET_NEWS: (symbol: string) => `/api/v1/market-data/${symbol}/news`,
        INDICES_SUMMARY: '/api/v1/market-data/indices/summary',

        // Signals
        SIGNALS: '/api/v1/signals',
        SIGNAL_BY_ID: (id: string) => `/api/v1/signals/${id}`,
        SIGNALS_LATEST: '/api/v1/signals/latest',

        // Agents
        AGENTS: '/api/v1/agents',
        AGENT_PERFORMANCE: '/api/v1/agents/performance',

        // Portfolio
        PORTFOLIO: '/api/v1/portfolio',
        PORTFOLIO_PERFORMANCE: '/api/v1/portfolio/performance',

        // Analytics
        ANALYTICS: '/api/v1/analytics',
        SIGNAL_ANALYTICS: '/api/v1/analytics/signals',

        // System
        SYSTEM_HEALTH: '/api/v1/system/health',
        SYSTEM_METRICS: '/api/v1/system/metrics',
    },

    // Request timeouts
    TIMEOUTS: {
        DEFAULT: 30000,
        LONG_POLL: 60000,
        FILE_UPLOAD: 120000,
    },

    // Retry configuration
    RETRY: {
        MAX_ATTEMPTS: 3,
        DELAY: 1000,
        BACKOFF_MULTIPLIER: 2,
    },
};

// Helper to check if we should use live data
export const shouldUseLiveData = () => API_CONFIG.USE_LIVE_DATA;

// Helper to get full API URL
export const getApiUrl = (endpoint: string) => {
    return `${API_CONFIG.API_BASE_URL}${endpoint}`;
}; 