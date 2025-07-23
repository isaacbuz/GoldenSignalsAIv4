/**
 * Comprehensive API Client for GoldenSignalsAI
 * Handles all backend communication with proper error handling and type safety
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { API_CONFIG } from '../../config/api.config';
import { createAPIInterceptor } from '../monitoring/performance';
import logger from '../logger';


// Types
export interface MarketData {
    symbol: string;
    price: number;
    change: number;
    change_percent: number;
    volume: number;
    high: number;
    low: number;
    timestamp: string;
    provider: string;
    market_cap?: number;
    pe_ratio?: number;
}

export interface HistoricalDataPoint {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

export interface Signal {
    signal_id: string;
    symbol: string;
    signal_type: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    strength: 'STRONG' | 'MODERATE' | 'WEAK';
    source: string;
    current_price?: number;
    target_price?: number;
    stop_loss?: number;
    take_profit?: number;
    risk_score?: number;
    reasoning?: string;
    created_at: string;
    expires_at?: string;
    indicators?: Record<string, number>;
    agent_breakdown?: Record<string, any>;
    metadata?: Record<string, any>;
}

export interface MarketStatus {
    isOpen: boolean;
    reason?: string;
    nextOpen?: string;
    nextClose?: string;
}

export interface SymbolSearchResult {
    symbol: string;
    name: string;
    type: string;
    exchange: string;
    currency: string;
}

export interface AIInsight {
    symbol: string;
    signal: string;
    confidence: number;
    reasoning: string;
    sentiment: 'bullish' | 'bearish' | 'neutral';
    summary: string;
    priority?: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
    type?: 'Bullish' | 'Bearish' | 'Warning' | 'Opportunity' | 'Neutral';
    timestamp?: string;
}

export interface PortfolioData {
    totalValue: number;
    dailyPnL: number;
    dailyPnLPercent: number;
    positions: Array<{
        symbol: string;
        quantity: number;
        avgPrice: number;
        currentPrice: number;
        marketValue: number;
        unrealizedPnL: number;
        unrealizedPnLPercent: number;
    }>;
}

export interface AgentPerformance {
    agent_name: string;
    total_signals: number;
    accuracy: number;
    avg_confidence: number;
    last_signal_time: string;
    performance_metrics: {
        sharpe_ratio?: number;
        win_rate?: number;
        avg_return?: number;
        max_drawdown?: number;
    };
}

export interface BacktestRequest {
    agent?: string;
    symbol: string;
    start_date: string;
    end_date: string;
    initial_capital?: number;
    strategy?: string;
}

export interface BacktestResult {
    parameters: {
        agent: string;
        symbol: string;
        period: string;
    };
    results: {
        total_return: number;
        annualized_return: number;
        sharpe_ratio: number;
        max_drawdown: number;
        win_rate: number;
        total_trades: number;
        profit_factor: number;
    };
    comparison: {
        buy_hold_return: number;
        outperformance: number;
        alpha: number;
        beta: number;
    };
}

export interface SystemHealth {
    status: string;
    timestamp: string;
    version: string;
    environment: string;
    database: string;
    redis: string;
    agents: string;
}

export interface LiveSignalUpdate {
    id: string;
    timestamp: string;
    symbol: string;
    consensus_action: string;
    confidence: number;
    agent_agreement: number;
    top_agents: Array<{
        name: string;
        action: string;
        confidence: number;
    }>;
}

class APIClient {
    private client: AxiosInstance;
    private requestQueue: Map<string, Promise<any>> = new Map();

    constructor() {
        this.client = axios.create({
            baseURL: API_CONFIG.API_BASE_URL,
            timeout: API_CONFIG.TIMEOUTS.DEFAULT,
            headers: {
                'Content-Type': 'application/json',
            },
        });

        // Add performance monitoring
        createAPIInterceptor(this.client);

        // Request interceptor for auth and request ID
        this.client.interceptors.request.use((config) => {
            config.headers['X-Request-ID'] = this.generateRequestId();

            // Add auth token if available
            const token = localStorage.getItem('auth_token');
            if (token) {
                config.headers.Authorization = `Bearer ${token}`;
            }

            return config;
        });

        // Response interceptor for error handling
        this.client.interceptors.response.use(
            (response) => response,
            async (error) => {
                if (error.response?.status === 401) {
                    // Handle unauthorized - redirect to login
                    localStorage.removeItem('auth_token');
                    window.location.href = '/login';
                }
                return Promise.reject(error);
            }
        );
    }

    private generateRequestId(): string {
        return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    private async makeRequest<T>(
        key: string,
        requestFn: () => Promise<AxiosResponse<T>>,
        options: { cache?: boolean; dedupe?: boolean } = {}
    ): Promise<T> {
        const { cache = true, dedupe = true } = options;

        // Request deduplication
        if (dedupe && this.requestQueue.has(key)) {
            return this.requestQueue.get(key);
        }

        const requestPromise = requestFn().then(response => response.data);

        if (dedupe) {
            this.requestQueue.set(key, requestPromise);
            requestPromise.finally(() => this.requestQueue.delete(key));
        }

        return requestPromise;
    }

    // System Health
    async getSystemHealth(): Promise<SystemHealth> {
        return this.makeRequest('system-health', () =>
            this.client.get('/api/v1/health')
        );
    }

    // Signals API - Enhanced with live backend integration
    async getSignals(filters?: {
        symbols?: string[];
        minConfidence?: number;
        signalType?: string;
        limit?: number;
    }): Promise<Signal[]> {
        const params = new URLSearchParams();
        if (filters?.symbols?.length) {
            params.append('symbols', filters.symbols.join(','));
        }
        if (filters?.minConfidence) {
            params.append('min_confidence', filters.minConfidence.toString());
        }
        if (filters?.signalType) {
            params.append('signal_type', filters.signalType);
        }
        if (filters?.limit) {
            params.append('limit', filters.limit.toString());
        }

        return this.makeRequest('signals', () =>
            this.client.get(`/api/v1/signals?${params.toString()}`)
        );
    }

    async getLatestSignals(limit: number = 10): Promise<Signal[]> {
        return this.makeRequest(`latest-signals-${limit}`, () =>
            this.client.get(`/api/v1/signals?limit=${limit}`)
        );
    }

    async getSignalById(id: string): Promise<Signal> {
        return this.makeRequest(`signal-${id}`, () =>
            this.client.get(`/api/v1/signals/${id}`)
        );
    }

    async getSignalsForSymbol(symbol: string, options?: {
        hours_back?: number;
        min_confidence?: number;
    }): Promise<Signal[]> {
        const params = new URLSearchParams();
        if (options?.hours_back) {
            params.append('hours_back', options.hours_back.toString());
        }
        if (options?.min_confidence) {
            params.append('min_confidence', options.min_confidence.toString());
        }

        return this.makeRequest(`signals-${symbol}`, () =>
            this.client.get(`/api/v1/signals/${symbol}?${params.toString()}`)
        );
    }

    // Live Signals from Performance Dashboard
    async getLiveSignals(limit: number = 10): Promise<LiveSignalUpdate[]> {
        return this.makeRequest(`live-signals-${limit}`, () =>
            this.client.get(`/api/v1/performance/live-signals?limit=${limit}`)
        );
    }

    // Agent Performance API
    async getAgentPerformance(): Promise<Record<string, AgentPerformance>> {
        return this.makeRequest('agent-performance', () =>
            this.client.get('/api/v1/agents/performance')
        );
    }

    async getAgentDetails(agentName: string): Promise<any> {
        return this.makeRequest(`agent-${agentName}`, () =>
            this.client.get(`/api/v1/performance/agent/${agentName}`)
        );
    }

    async getPerformanceOverview(): Promise<any> {
        return this.makeRequest('performance-overview', () =>
            this.client.get('/api/v1/performance/overview')
        );
    }

    // Market Data API
    async getMarketData(symbol: string): Promise<MarketData> {
        return this.makeRequest(`market-data-${symbol}`, () =>
            this.client.get(`/api/v1/market-data/${symbol}`)
        );
    }

    async getMarketQuote(symbol: string): Promise<MarketData> {
        return this.makeRequest(`quote-${symbol}`, () =>
            this.client.get(`/api/v1/market-data/${symbol}/quote`)
        );
    }

    async getHistoricalData(symbol: string, period: string = '1mo', barCount?: number): Promise<any[]> {
        const params = new URLSearchParams();
        params.append('period', period);
        if (barCount) {
            params.append('bar_count', barCount.toString());
        }

        return this.makeRequest(`historical-${symbol}-${period}-${barCount}`, () =>
            this.client.get(`/api/v1/market-data/${symbol}/historical?${params.toString()}`)
        );
    }

    async getMarketStatus(): Promise<any> {
        return this.makeRequest('market-status', () =>
            this.client.get('/api/v1/market-data/status/market')
        );
    }

    async searchSymbols(query: string): Promise<any[]> {
        return this.makeRequest(`search-${query}`, () =>
            this.client.get(`/api/v1/market-data/search/${query}`)
        );
    }

    // Backtesting API
    async runBacktest(request: BacktestRequest): Promise<BacktestResult> {
        return this.makeRequest(`backtest-${JSON.stringify(request)}`, () =>
            this.client.post('/api/v1/performance/backtest', request)
        );
    }

    async getBacktestHistory(): Promise<BacktestResult[]> {
        return this.makeRequest('backtest-history', () =>
            this.client.get('/api/v1/backtesting/history')
        );
    }

    // AI Insights API
    async getAIInsights(symbol?: string): Promise<AIInsight[]> {
        const endpoint = symbol
            ? `/api/v1/ai/insights/${symbol}`
            : '/api/v1/ai/insights';

        return this.makeRequest(`ai-insights-${symbol || 'all'}`, () =>
            this.client.get(endpoint)
        );
    }

    // Analytics API
    async getSignalAnalytics(timeframe: string = '1d'): Promise<any> {
        return this.makeRequest(`signal-analytics-${timeframe}`, () =>
            this.client.get(`/api/v1/analytics/signals?timeframe=${timeframe}`)
        );
    }

    async getAgentAnalytics(): Promise<any[]> {
        return this.makeRequest('agent-analytics', () =>
            this.client.get('/api/v1/analytics/agents')
        );
    }

    // Portfolio API
    async getPortfolio(): Promise<any> {
        return this.makeRequest('portfolio', () =>
            this.client.get('/api/v1/portfolio')
        );
    }

    async getPortfolioPerformance(): Promise<any> {
        return this.makeRequest('portfolio-performance', () =>
            this.client.get('/api/v1/portfolio/performance')
        );
    }

    // AI Chat Integration
    async sendChatMessage(message: string, context?: any): Promise<any> {
        return this.client.post('/api/v1/ai-chat/message', {
            message,
            context,
            timestamp: new Date().toISOString()
        });
    }

    async getChatHistory(sessionId?: string): Promise<any[]> {
        const params = sessionId ? `?session_id=${sessionId}` : '';
        return this.makeRequest(`chat-history-${sessionId || 'default'}`, () =>
            this.client.get(`/api/v1/ai-chat/history${params}`)
        );
    }

    // Options Analysis
    async getOptionsData(symbol: string): Promise<any> {
        return this.makeRequest(`options-${symbol}`, () =>
            this.client.get(`/api/v1/market-data/${symbol}/options`)
        );
    }

    // Precise Options Signals with timeframe parameter
    async getPreciseOptionsSignals(symbol: string, timeframe: string = '1d'): Promise<any[]> {
        return this.makeRequest(`precise-options-${symbol}-${timeframe}`, () =>
            this.client.get(`/api/v1/signals/options/${symbol}?timeframe=${timeframe}`)
        );
    }

    // Historical Market Data with multiple parameters
    async getHistoricalMarketData(symbol: string, period: string = '1mo', timeframe?: string): Promise<any[]> {
        const params = new URLSearchParams();
        params.append('period', period);
        if (timeframe) {
            params.append('timeframe', timeframe);
        }

        return this.makeRequest(`historical-market-${symbol}-${period}-${timeframe}`, () =>
            this.client.get(`/api/v1/market-data/${symbol}/historical?${params.toString()}`)
        );
    }

    // Integrated Signals (Hybrid System)
    async getIntegratedSignals(symbols: string[]): Promise<any> {
        return this.client.post('/api/v1/signals/scan', {
            symbols,
            include_options: true,
            include_arbitrage: true,
            min_confidence: 70.0
        });
    }

    // System Metrics
    async getSystemMetrics(): Promise<any> {
        return this.makeRequest('system-metrics', () =>
            this.client.get('/api/v1/system/metrics')
        );
    }

    // Generic request method for custom endpoints
    async request<T>(config: AxiosRequestConfig): Promise<T> {
        const response = await this.client.request<T>(config);
        return response.data;
    }

    // Batch operations
    async batchRequest<T>(requests: AxiosRequestConfig[]): Promise<T[]> {
        const promises = requests.map(config => this.client.request<T>(config));
        const responses = await Promise.allSettled(promises);

        return responses.map(result => {
            if (result.status === 'fulfilled') {
                return result.value.data;
            } else {
                logger.error('Batch request failed:', result.reason);
                return null;
            }
        }).filter(Boolean) as T[];
    }

    // News and Market Events API
    async getMarketNews(symbol?: string, limit: number = 10): Promise<any[]> {
        const params = new URLSearchParams();
        if (symbol) params.append('symbol', symbol);
        params.append('limit', limit.toString());

        return this.makeRequest(`market-news-${symbol || 'all'}-${limit}`, () =>
            this.client.get(`/api/v1/news?${params.toString()}`)
        );
    }

    async getNewsImpact(symbol: string): Promise<any> {
        return this.makeRequest(`news-impact-${symbol}`, () =>
            this.client.get(`/api/v1/news/impact/${symbol}`)
        );
    }

    // Alerts and Notifications API
    async getUserAlerts(): Promise<any[]> {
        return this.makeRequest('user-alerts', () =>
            this.client.get('/api/v1/alerts/user')
        );
    }

    async createAlert(alert: {
        symbol: string;
        condition: string;
        value: number;
        notification_type: 'email' | 'push' | 'sms';
    }): Promise<any> {
        return this.client.post('/api/v1/alerts', alert);
    }

    async deleteAlert(alertId: string): Promise<void> {
        await this.client.delete(`/api/v1/alerts/${alertId}`);
    }

    // Advanced Analytics API
    async getMarketSentiment(symbol?: string): Promise<any> {
        const endpoint = symbol
            ? `/api/v1/analytics/sentiment/${symbol}`
            : '/api/v1/analytics/sentiment';

        return this.makeRequest(`market-sentiment-${symbol || 'all'}`, () =>
            this.client.get(endpoint)
        );
    }

    async getVolatilityAnalysis(symbol: string, period: string = '1mo'): Promise<any> {
        return this.makeRequest(`volatility-${symbol}-${period}`, () =>
            this.client.get(`/api/v1/analytics/volatility/${symbol}?period=${period}`)
        );
    }

    async getCorrelationMatrix(symbols: string[]): Promise<any> {
        return this.makeRequest(`correlation-${symbols.join(',')}`, () =>
            this.client.post('/api/v1/analytics/correlation', { symbols })
        );
    }

    async getTechnicalIndicators(symbol: string, indicators: string[]): Promise<any> {
        return this.makeRequest(`technical-${symbol}-${indicators.join(',')}`, () =>
            this.client.post('/api/v1/analytics/technical', {
                symbol,
                indicators
            })
        );
    }

    // Risk Management API
    async getRiskMetrics(symbol: string): Promise<any> {
        return this.makeRequest(`risk-metrics-${symbol}`, () =>
            this.client.get(`/api/v1/risk/metrics/${symbol}`)
        );
    }

    async getPortfolioRisk(): Promise<any> {
        return this.makeRequest('portfolio-risk', () =>
            this.client.get('/api/v1/risk/portfolio')
        );
    }

    async getVaRAnalysis(confidence: number = 95): Promise<any> {
        return this.makeRequest(`var-analysis-${confidence}`, () =>
            this.client.get(`/api/v1/risk/var?confidence=${confidence}`)
        );
    }

    // Machine Learning Model API
    async getModelPerformance(): Promise<any> {
        return this.makeRequest('model-performance', () =>
            this.client.get('/api/v1/ml/models/performance')
        );
    }

    async getModelPredictions(symbol: string, model?: string): Promise<any> {
        const params = new URLSearchParams();
        if (model) params.append('model', model);

        return this.makeRequest(`model-predictions-${symbol}-${model || 'all'}`, () =>
            this.client.get(`/api/v1/ml/predictions/${symbol}?${params.toString()}`)
        );
    }

    async retrainModel(modelId: string): Promise<any> {
        return this.client.post(`/api/v1/ml/models/${modelId}/retrain`);
    }

    // Advanced Signal Processing API
    async getSignalConfidence(signalId: string): Promise<any> {
        return this.makeRequest(`signal-confidence-${signalId}`, () =>
            this.client.get(`/api/v1/signals/${signalId}/confidence`)
        );
    }

    async getSignalBacktest(signalId: string): Promise<any> {
        return this.makeRequest(`signal-backtest-${signalId}`, () =>
            this.client.get(`/api/v1/signals/${signalId}/backtest`)
        );
    }

    async getSignalCorrelation(signalId: string): Promise<any> {
        return this.makeRequest(`signal-correlation-${signalId}`, () =>
            this.client.get(`/api/v1/signals/${signalId}/correlation`)
        );
    }

    // Real-time Data Streaming API
    async getStreamingData(symbol: string, dataType: 'price' | 'volume' | 'trades'): Promise<any> {
        return this.makeRequest(`streaming-${symbol}-${dataType}`, () =>
            this.client.get(`/api/v1/streaming/${symbol}/${dataType}`)
        );
    }

    // User Preferences and Settings API
    async getUserPreferences(): Promise<any> {
        return this.makeRequest('user-preferences', () =>
            this.client.get('/api/v1/user/preferences')
        );
    }

    async updateUserPreferences(preferences: any): Promise<any> {
        return this.client.put('/api/v1/user/preferences', preferences);
    }

    async getUserWatchlist(): Promise<any[]> {
        return this.makeRequest('user-watchlist', () =>
            this.client.get('/api/v1/user/watchlist')
        );
    }

    async addToWatchlist(symbol: string): Promise<any> {
        return this.client.post('/api/v1/user/watchlist', { symbol });
    }

    async removeFromWatchlist(symbol: string): Promise<void> {
        await this.client.delete(`/api/v1/user/watchlist/${symbol}`);
    }

    // Market Data Enhancement API
    async getMarketOverview(): Promise<any> {
        return this.makeRequest('market-overview', () =>
            this.client.get('/api/v1/market-data/overview')
        );
    }

    async getMarketMovers(direction: 'gainers' | 'losers' | 'most_active' = 'gainers'): Promise<any[]> {
        return this.makeRequest(`market-movers-${direction}`, () =>
            this.client.get(`/api/v1/market-data/movers/${direction}`)
        );
    }

    async getEarningsCalendar(startDate?: string, endDate?: string): Promise<any[]> {
        const params = new URLSearchParams();
        if (startDate) params.append('start_date', startDate);
        if (endDate) params.append('end_date', endDate);

        return this.makeRequest(`earnings-calendar-${startDate}-${endDate}`, () =>
            this.client.get(`/api/v1/market-data/earnings?${params.toString()}`)
        );
    }

    async getEconomicEvents(): Promise<any[]> {
        return this.makeRequest('economic-events', () =>
            this.client.get('/api/v1/market-data/economic-events')
        );
    }

    // Enhanced Error Handling
    async withRetry<T>(
        operation: () => Promise<T>,
        maxRetries: number = 3,
        backoffMs: number = 1000
    ): Promise<T> {
        let lastError: Error;

        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                return await operation();
            } catch (error) {
                lastError = error as Error;

                if (attempt === maxRetries) {
                    throw lastError;
                }

                // Exponential backoff
                const delay = backoffMs * Math.pow(2, attempt);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }

        throw lastError!;
    }

    // Health Check and Monitoring
    async ping(): Promise<{ status: string; timestamp: string }> {
        return this.makeRequest('ping', () =>
            this.client.get('/api/v1/ping')
        );
    }

    async getAPIUsage(): Promise<any> {
        return this.makeRequest('api-usage', () =>
            this.client.get('/api/v1/usage/stats')
        );
    }

    async getRateLimitStatus(): Promise<any> {
        return this.makeRequest('rate-limit-status', () =>
            this.client.get('/api/v1/usage/rate-limit')
        );
    }
}

// Export singleton instance
export const apiClient = new APIClient();

// Legacy exports for backward compatibility
export const fetchMarketData = (symbol: string) => apiClient.getMarketData(symbol);
export const fetchAIInsights = (symbol: string) => apiClient.getAIInsights(symbol);
export const fetchSignals = (filters?: any) => apiClient.getSignals(filters);
