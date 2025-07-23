/**
 * Environment Configuration
 * Centralized configuration for all environment variables and feature flags
 */

export interface EnvironmentConfig {
    // API Configuration
    API_URL: string;
    API_TIMEOUT: number;

    // WebSocket Configuration
    WEBSOCKET_ENABLED: boolean;
    WEBSOCKET_URL: string;
    WEBSOCKET_RECONNECT: boolean;
    WEBSOCKET_RECONNECT_INTERVAL: number;

    // Feature Flags
    FEATURES: {
        AI_CHAT: boolean;
        ADVANCED_CHARTS: boolean;
        PAPER_TRADING: boolean;
        ADMIN_PANEL: boolean;
        BACKTESTING: boolean;
        PORTFOLIO_ANALYTICS: boolean;
        REAL_TIME_NOTIFICATIONS: boolean;
        MULTI_LANGUAGE: boolean;
    };

    // Logging Configuration
    LOG_LEVEL: 'debug' | 'info' | 'warn' | 'error';
    LOG_TO_FILE: boolean;
    LOG_BUFFER_SIZE: number;

    // Development Tools
    SHOW_DEV_TOOLS: boolean;
    MOCK_API: boolean;


    // Performance
    ENABLE_PERFORMANCE_MONITORING: boolean;
    SLOW_RENDER_THRESHOLD: number;

    // Environment Info
    IS_DEVELOPMENT: boolean;
    IS_PRODUCTION: boolean;
    IS_TEST: boolean;
}

function getEnvVar(key: string, defaultValue?: string): string {
    return import.meta.env[key] || defaultValue || '';
}

function getEnvBoolean(key: string, defaultValue: boolean = false): boolean {
    const value = getEnvVar(key, String(defaultValue));
    return value === 'true' || value === '1';
}

function getEnvNumber(key: string, defaultValue: number): number {
    const value = getEnvVar(key);
    const parsed = Number(value);
    return isNaN(parsed) ? defaultValue : parsed;
}

export const ENV: EnvironmentConfig = {
    // API Configuration
    API_URL: getEnvVar('VITE_API_BASE_URL', 'http://localhost:8000'),
    API_TIMEOUT: getEnvNumber('VITE_API_TIMEOUT', 30000),

    // WebSocket Configuration
    WEBSOCKET_ENABLED: getEnvBoolean('VITE_WEBSOCKET_ENABLED', true),
    WEBSOCKET_URL: getEnvVar('VITE_WS_BASE_URL', 'ws://localhost:8000') + '/ws',
    WEBSOCKET_RECONNECT: getEnvBoolean('VITE_WEBSOCKET_RECONNECT', true),
    WEBSOCKET_RECONNECT_INTERVAL: getEnvNumber('VITE_WEBSOCKET_RECONNECT_INTERVAL', 5000),

    // Feature Flags
    FEATURES: {
        AI_CHAT: getEnvBoolean('VITE_FEATURE_AI_CHAT', true),
        ADVANCED_CHARTS: getEnvBoolean('VITE_FEATURE_ADVANCED_CHARTS', false),
        PAPER_TRADING: getEnvBoolean('VITE_FEATURE_PAPER_TRADING', false),
        ADMIN_PANEL: getEnvBoolean('VITE_FEATURE_ADMIN_PANEL', false),
        BACKTESTING: getEnvBoolean('VITE_FEATURE_BACKTESTING', true),
        PORTFOLIO_ANALYTICS: getEnvBoolean('VITE_FEATURE_PORTFOLIO_ANALYTICS', true),
        REAL_TIME_NOTIFICATIONS: getEnvBoolean('VITE_FEATURE_REAL_TIME_NOTIFICATIONS', true),
        MULTI_LANGUAGE: getEnvBoolean('VITE_FEATURE_MULTI_LANGUAGE', false),
    },

    // Logging Configuration
    LOG_LEVEL: (getEnvVar('VITE_LOG_LEVEL', 'info') as any),
    LOG_TO_FILE: getEnvBoolean('VITE_LOG_TO_FILE', false),
    LOG_BUFFER_SIZE: getEnvNumber('VITE_LOG_BUFFER_SIZE', 100),

    // Development Tools
    SHOW_DEV_TOOLS: getEnvBoolean('VITE_SHOW_DEV_TOOLS', false),
    MOCK_API: getEnvBoolean('VITE_MOCK_API', false),


    // Performance
    ENABLE_PERFORMANCE_MONITORING: getEnvBoolean('VITE_ENABLE_PERFORMANCE_MONITORING', true),
    SLOW_RENDER_THRESHOLD: getEnvNumber('VITE_SLOW_RENDER_THRESHOLD', 16),

    // Environment Info
    IS_DEVELOPMENT: import.meta.env.DEV || process.env.NODE_ENV === 'development',
    IS_PRODUCTION: import.meta.env.PROD || process.env.NODE_ENV === 'production',
    IS_TEST: import.meta.env.MODE === 'test' || process.env.NODE_ENV === 'test',
};

// Helper function to check if a feature is enabled
export function isFeatureEnabled(feature: keyof typeof ENV.FEATURES): boolean {
    return ENV.FEATURES[feature];
}

// Helper function to get API endpoint
export function getApiEndpoint(path: string): string {
    const baseUrl = ENV.API_URL.replace(/\/$/, '');
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    return `${baseUrl}${cleanPath}`;
}

// Helper function to check if running in mock mode
export function isMockMode(): boolean {
    return ENV.MOCK_API;
}

// Export for debugging
if (ENV.IS_DEVELOPMENT) {
    (window as any).__ENV__ = ENV;
}
