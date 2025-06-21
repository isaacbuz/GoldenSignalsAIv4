/**
 * API Service Layer for GoldenSignalsAI V3
 * 
 * Centralized API communication with the FastAPI backend
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import toast from 'react-hot-toast';
import {
  PreciseOptionsSignal,
  PerformanceMetrics,
  RiskMetrics,
  SignalFilters
} from '../types/signals';
import { API_CONFIG, shouldUseLiveData, getApiUrl } from '../config/api.config';

// API Configuration
const API_BASE_URL = API_CONFIG.API_BASE_URL;
const WS_BASE_URL = API_CONFIG.WS_BASE_URL;

// Types
export interface Signal {
  id: string;
  symbol: string;
  type: 'CALL' | 'PUT';
  strike: number;
  expiry: string;
  confidence: number;
  entryPrice: number;
  targetPrice: number;
  stopLoss: number;
  timeframe: string;
  reasoning: string;
  patterns: string[];
  urgency: 'HIGH' | 'MEDIUM' | 'LOW';
}

export interface MarketData {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  timestamp: number;
}

export interface AgentPerformance {
  agent_name: string;
  total_signals: number;
  correct_signals: number;
  accuracy: number;
  avg_confidence: number;
  last_updated: string;
}

export interface MarketStatus {
  is_open: boolean;
  current_time: string;
  market_hours: {
    open: string;
    close: string;
  };
  next_open: string;
  next_close: string;
}

export interface MarketDataParams {
  symbol: string;
  timeframe: string;
}

export interface SignalsParams {
  symbols: string[];
  signal_types?: string[];
  limit?: number;
  offset?: number;
}

export interface SignalsResponse {
  signals: Signal[];
  total: number;
}

export interface AIInsight {
  levels: Array<{
    price: number;
    type: 'SUPPORT' | 'RESISTANCE' | 'ENTRY' | 'TARGET' | 'STOP';
    confidence: number;
    label: string;
  }>;
  signals: Array<{
    price: number;
    type: 'ENTRY' | 'TARGET' | 'STOP';
    confidence: number;
    label: string;
    timestamp: number;
  }>;
  trendLines: Array<{
    time: number;
    value: number;
  }>;
  analysis: {
    sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
    confidence: number;
    summary: string;
    patterns: string[];
  };
}

export interface MarketOpportunity {
  id: string;
  symbol: string;
  name: string;
  type: 'CALL' | 'PUT';
  confidence: number;
  potentialReturn: number;
  timeframe: string;
  keyReason: string;
  momentum: 'strong' | 'moderate' | 'building';
  aiScore: number;
  sector?: string;
  volume?: number;
  volatility?: number;
}

// Market News interface
export interface MarketNews {
  id: string;
  title: string;
  source: string;
  url: string;
  timestamp: string;
  impact: 'HIGH' | 'MEDIUM' | 'LOW';
  symbols?: string[];
  sentiment?: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  summary?: string;
}

// API Client Class
class ApiClient {
  private instance: AxiosInstance;

  constructor() {
    this.instance = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.instance.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.instance.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Handle unauthorized access
          localStorage.removeItem('auth_token');
          toast.error('Session expired. Please login again.');
        } else if (error.response?.status >= 500) {
          toast.error('Server error. Please try again later.');
        } else if (error.code === 'ECONNABORTED') {
          toast.error('Request timeout. Please check your connection.');
        }
        return Promise.reject(error);
      }
    );
  }

  // Generic request method
  async request<T>(config: AxiosRequestConfig): Promise<T> {
    const response: AxiosResponse<T> = await this.instance.request(config);
    return response.data;
  }

  // Signal endpoints - Updated to match backend
  async getSignals(params: { symbol?: string; symbols?: string[] }): Promise<SignalsResponse> {
    // Accepts either a single symbol or an array of symbols
    let symbols: string[] = [];
    if (params.symbol) {
      symbols = [params.symbol];
    } else if (params.symbols) {
      symbols = params.symbols;
    }
    const response = await this.request<SignalsResponse>({
      method: 'GET',
      url: '/signals',
      params: {
        symbols,
        limit: 100
      }
    });
    return response;
  }

  async getSignal(signalId: string): Promise<Signal> {
    // For now, we'll use the symbol as the signal ID since backend doesn't have signal IDs
    return this.getSignalBySymbol(signalId);
  }

  async getSignalBySymbol(symbol: string): Promise<Signal> {
    try {
      const data = await this.request<any>({
        method: 'GET',
        url: `/signals/${symbol.toUpperCase()}`,
      });

      // Transform backend response to frontend format
      return {
        id: data.signal_id,
        symbol: data.symbol,
        type: data.signal_type as 'CALL' | 'PUT',
        strike: data.strike || 0,
        expiry: data.expiry || '',
        confidence: data.confidence,
        entryPrice: data.entry_price || 0,
        targetPrice: data.target_price || 0,
        stopLoss: data.stop_loss || 0,
        timeframe: data.timeframe || '',
        reasoning: data.reasoning || '',
        patterns: data.patterns || [],
        urgency: data.urgency as 'HIGH' | 'MEDIUM' | 'LOW' || 'LOW',
      };
    } catch (error) {
      console.error(`Error fetching signal for ${symbol}:`, error);
      throw error;
    }
  }

  async getSignalsBySymbol(symbol: string): Promise<Signal[]> {
    const signal = await this.getSignalBySymbol(symbol);
    return [signal];
  }

  async getLatestSignals(limit: number = 10): Promise<Signal[]> {
    const response = await this.request<Signal[]>({
      method: 'GET',
      url: '/signals/latest',
      params: { limit }
    });
    return response;
  }

  // Market data endpoints
  async getMarketData(symbol: string): Promise<MarketData> {
    if (shouldUseLiveData()) {
      try {
        const response = await this.request<MarketData>({
          method: 'GET',
          url: API_CONFIG.ENDPOINTS.MARKET_DATA + `/${symbol}`,
        });
        return response;
      } catch (error) {
        console.error(`Error fetching live market data for ${symbol}:`, error);
        // Fall back to mock data
      }
    }

    // Mock data fallback
    return {
      symbol: symbol,
      price: 150 + Math.random() * 50,
      change: (Math.random() - 0.5) * 10,
      change_percent: (Math.random() - 0.5) * 5,
      volume: Math.floor(Math.random() * 10000000),
      timestamp: Date.now()
    };
  }

  async getHistoricalMarketData(symbol: string, period: string = '1D'): Promise<any[]> {
    if (shouldUseLiveData()) {
      try {
        const response = await this.request<{ data: any[] }>({
          method: 'GET',
          url: API_CONFIG.ENDPOINTS.MARKET_HISTORICAL(symbol),
          params: { period }
        });
        return response.data;
      } catch (error) {
        console.error(`Error fetching live historical data for ${symbol}:`, error);
      }
    }

    // Return empty array as fallback
    return [];
  }

  async getMultipleMarketData(symbols: string[]): Promise<Record<string, MarketData>> {
    if (shouldUseLiveData()) {
      try {
        const response = await this.request<Record<string, MarketData>>({
          method: 'POST',
          url: API_CONFIG.ENDPOINTS.MARKET_QUOTES,
          data: symbols
        });
        return response;
      } catch (error) {
        console.error('Error fetching live market data for multiple symbols:', error);
      }
    }

    // Mock data fallback
    const marketData: Record<string, MarketData> = {};
    symbols.forEach(symbol => {
      marketData[symbol] = {
        symbol: symbol,
        price: 150 + Math.random() * 50,
        change: (Math.random() - 0.5) * 10,
        change_percent: (Math.random() - 0.5) * 5,
        volume: Math.floor(Math.random() * 10000000),
        timestamp: Date.now()
      };
    });
    return marketData;
  }

  async getHistoricalData(
    symbol: string,
    period: string = '1d',
    interval: string = '5m'
  ): Promise<any[]> {
    if (shouldUseLiveData()) {
      try {
        const response = await this.request<{ data: any[] }>({
          method: 'GET',
          url: API_CONFIG.ENDPOINTS.MARKET_HISTORICAL(symbol),
          params: { period, interval }
        });
        return response.data;
      } catch (error) {
        console.error(`Error fetching live historical data for ${symbol}:`, error);
      }
    }

    return [];
  }

  async getMarketStatus(): Promise<MarketStatus> {
    if (shouldUseLiveData()) {
      try {
        const response = await this.request<MarketStatus>({
          method: 'GET',
          url: API_CONFIG.ENDPOINTS.MARKET_STATUS,
        });
        return response;
      } catch (error) {
        console.error('Error fetching live market status:', error);
      }
    }

    // Mock data fallback
    const now = new Date();
    const isWeekday = now.getDay() >= 1 && now.getDay() <= 5;
    const hour = now.getHours();
    const isMarketHours = hour >= 9 && hour < 16;

    return {
      is_open: isWeekday && isMarketHours,
      current_time: now.toISOString(),
      market_hours: {
        open: '09:30',
        close: '16:00',
      },
      next_open: '',
      next_close: '',
    };
  }

  // Agent endpoints
  async getAgents(): Promise<any[]> {
    return this.request<any[]>({
      method: 'GET',
      url: '/agents',
    });
  }

  async getAgentPerformance(): Promise<any> {
    const response = await this.request<any>({
      method: 'GET',
      url: '/agents/performance',
    });
    return response;
  }

  // Orchestrator endpoints
  async getOrchestratorStatus(): Promise<any> {
    // This endpoint doesn't exist in backend yet
    return {
      status: 'HEALTHY',
      last_check_in: new Date().toISOString(),
      active_agents: 5,
      pending_signals: 2,
    };
  }

  // Portfolio endpoints
  async getPortfolio(): Promise<any> {
    // This endpoint doesn't exist in backend yet
    return {
      total_value: 125000.00,
      positions: [],
    };
  }

  async getPortfolioPerformance(): Promise<any> {
    // This endpoint doesn't exist in backend yet
    return {
      daily_change: 1200.50,
      total_change: 25000.00,
    };
  }

  // Analytics endpoints
  async getAnalytics(params?: any): Promise<any> {
    // This endpoint doesn't exist in backend yet
    return {};
  }

  async getSignalAnalytics(): Promise<any> {
    // This endpoint doesn't exist in backend yet
    return {};
  }

  // System endpoints
  async getSystemHealth(): Promise<any> {
    // This endpoint doesn't exist in backend yet
    return {
      api: 'HEALTHY',
      database: 'HEALTHY',
      websocket: 'HEALTHY',
    };
  }

  async getSystemMetrics(): Promise<any> {
    // This endpoint doesn't exist in backend yet
    return {};
  }

  async getAvailableSymbols(): Promise<string[]> {
    // This endpoint doesn't exist in backend yet
    return ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'NVDA', 'SPY', 'QQQ', 'META', 'AMZN', 'AMD'];
  }

  async getMarketSummary(): Promise<any> {
    // This endpoint doesn't exist in backend yet
    return {
      leading_index: {
        name: 'S&P 500',
        price: 4500.0,
        change: 25.5,
        change_percent: 0.57
      },
      top_gainer: { symbol: 'NVDA', change_percent: 4.5 },
      top_loser: { symbol: 'AMD', change_percent: -3.2 },
    };
  }

  // WebSocket URL helper
  getWebSocketUrl(): string {
    return WS_BASE_URL;
  }

  // Precise Options Signals endpoints
  async getActiveSignals(filters?: SignalFilters): Promise<PreciseOptionsSignal[]> {
    try {
      const response = await this.request<PreciseOptionsSignal[]>({
        method: 'GET',
        url: '/api/v1/signals/precise/active',
        params: filters,
      });
      return response;
    } catch (error) {
      // Fallback to mock data for development
      return this.getMockPreciseSignals();
    }
  }

  async getPreciseSignal(signalId: string): Promise<PreciseOptionsSignal> {
    return this.request<PreciseOptionsSignal>({
      method: 'GET',
      url: `/api/v1/signals/precise/${signalId}`,
    });
  }

  async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    try {
      const response = await this.request<PerformanceMetrics>({
        method: 'GET',
        url: '/api/v1/performance/metrics',
      });
      return response;
    } catch (error) {
      // Fallback to mock data
      return {
        winRate: 87,
        avgReturn: 24,
        totalSignals: 142,
        successfulSignals: 123,
        avgHoldTime: '2.5 days',
        bestPerformer: { symbol: 'NVDA', return: 145 },
        worstPerformer: { symbol: 'INTC', return: -12 },
      };
    }
  }

  async getRiskMetrics(): Promise<RiskMetrics> {
    try {
      const response = await this.request<RiskMetrics>({
        method: 'GET',
        url: '/api/v1/risk/metrics',
      });
      return response;
    } catch (error) {
      // Fallback to mock data
      return {
        activePositions: 5,
        totalExposure: 25000,
        maxDrawdown: 8.5,
        sharpeRatio: 1.85,
        currentRisk: 3500,
        riskLimit: 10000,
        utilizationPct: 35,
      };
    }
  }

  // Mock data generator for development
  private getMockPreciseSignals(): PreciseOptionsSignal[] {
    const symbols = ['AAPL', 'TSLA', 'NVDA', 'SPY', 'META'];
    const now = new Date();

    return symbols.map((symbol, index) => {
      const isCall = Math.random() > 0.5;
      const basePrice = 150 + index * 10;
      const entryPrice = basePrice + (Math.random() - 0.5) * 2;

      return {
        // Identification
        id: `${symbol}_${now.getTime()}_${index}`,
        symbol,
        signal_id: `${symbol}_${now.getTime()}_${index}`,
        generated_at: now.toISOString(),
        timestamp: new Date(now.getTime() + index * 3600000).toISOString(),

        // Trade Direction
        type: isCall ? 'CALL' : 'PUT',
        signal_type: isCall ? 'BUY_CALL' : 'BUY_PUT',
        confidence: 70 + Math.random() * 25,
        priority: index === 0 ? 'HIGH' : index < 3 ? 'MEDIUM' : 'LOW',

        entry_window: {
          date: index === 0 ? 'Today' : 'Tomorrow',
          start_time: '10:00 AM ET',
          end_time: '10:30 AM ET',
        },
        hold_duration: '2-3 days',
        expiration_warning: 'Exit by Friday 3:00 PM',

        strike_price: 150 + index * 5,
        expiration_date: new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000).toISOString(),
        contract_type: 'Weekly',
        max_premium: 3.50,

        current_price: basePrice,
        entry_price: entryPrice,
        entry_trigger: entryPrice,
        entry_zone: [entryPrice - 0.25, entryPrice + 0.25] as [number, number],

        stop_loss: isCall ? entryPrice * 0.98 : entryPrice * 1.02,
        stop_loss_pct: 2,
        position_size: 2,
        max_risk_dollars: 500,
        max_loss: 500,
        take_profit: isCall ? entryPrice * 1.05 : entryPrice * 0.95,

        targets: [
          { price: 152 + index * 5, exit_pct: 50 },
          { price: 155 + index * 5, exit_pct: 50 },
        ],
        risk_reward_ratio: 2.5,

        exit_rules: [
          'Exit 50% at first target',
          'Exit remaining at second target',
          'Hard stop if price crosses stop loss',
          'Time stop: Exit all by expiration minus 1 day',
        ],
        time_based_exits: {
          intraday: 'Exit if no movement by 2:00 PM',
          multi_day: 'Reduce position by 50% after 2 days',
          expiration: 'Exit all by expiration minus 1 day',
        },

        setup_name: ['Oversold Bounce', 'MACD Crossover', 'Breakout', 'Support Test'][index % 4],
        key_indicators: {
          RSI: (30 + Math.random() * 40).toFixed(1),
          MACD: Math.random() > 0.5 ? 'Bullish' : 'Bearish',
          Volume: `${(1 + Math.random() * 2).toFixed(1)}x avg`,
          ATR: (1.5 + Math.random()).toFixed(2),
        },
        chart_patterns: ['Bullish Flag', 'Volume Surge'],

        alerts_to_set: [
          `Price alert at entry: $${(148.50 + index * 5).toFixed(2)}`,
          `Stop loss alert: $${(146 + index * 5).toFixed(2)}`,
          `Target 1 alert: $${(152 + index * 5).toFixed(2)}`,
        ],
        pre_entry_checklist: [
          'Confirm market is open and liquid',
          'Check for pending news/earnings',
          'Verify option spread is reasonable',
          'Set all alerts before entry',
          'Have exit plan ready',
        ],
      };
    });
  }

  // Mock implementation for development
  private async getMockAIInsights(symbol: string): Promise<AIInsight> {
    const mockPrice = 150;
    return {
      levels: [
        { price: mockPrice * 0.95, type: 'SUPPORT', confidence: 0.85, label: 'Strong Support' },
        { price: mockPrice * 1.05, type: 'RESISTANCE', confidence: 0.82, label: 'Key Resistance' },
      ],
      signals: [
        {
          price: mockPrice,
          type: 'ENTRY',
          confidence: 0.94,
          label: 'Buy Signal',
          timestamp: Date.now() / 1000
        },
      ],
      trendLines: Array.from({ length: 20 }, (_, i) => ({
        time: (Date.now() / 1000) - (i * 300),
        value: mockPrice * (1 + (Math.sin(i / 10) * 0.05)),
      })).reverse(),
      analysis: {
        sentiment: 'BULLISH',
        confidence: 0.88,
        summary: 'Strong bullish momentum with increasing volume',
        patterns: ['Bullish Flag', 'Golden Cross', 'Volume Surge'],
      },
    };
  }

  // Precise Options Signals
  async getPreciseOptionsSignals(symbol?: string, timeframe?: string): Promise<PreciseOptionsSignal[]> {
    try {
      const params: any = {};
      if (symbol) params.symbol = symbol;
      if (timeframe) params.timeframe = timeframe;

      const response = await this.request<{ signals: PreciseOptionsSignal[] }>({
        method: 'GET',
        url: '/api/v1/signals/precise-options',
        params,
      });

      // Extract the signals array from the response object
      return response.signals || [];
    } catch (error) {
      console.error('Error fetching precise options signals:', error);
      // Return mock data for development
      return this.getMockPreciseOptionsSignals(symbol || 'SPY');
    }
  }

  // Generate signals for a specific symbol
  async generateSignalsForSymbol(symbol: string, timeframe: string): Promise<void> {
    try {
      await this.request({
        method: 'POST',
        url: '/api/v1/signals/generate',
        data: { symbol, timeframe },
      });
    } catch (error) {
      console.error('Error generating signals:', error);
      // Simulate signal generation in development
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  // Get AI insights for a signal
  async getAIInsights(signalId: string): Promise<any> {
    try {
      const response = await this.request<any>({
        method: 'GET',
        url: `/api/v1/signals/${signalId}/insights`,
      });
      return response;
    } catch (error) {
      console.error('Error fetching AI insights:', error);
      // Return mock insights
      return {
        technical: {
          rsi: { value: 45, signal: 'oversold' },
          macd: { signal: 'bullish_crossover' },
          support_resistance: { nearest_support: 440, nearest_resistance: 455 },
        },
        sentiment: {
          overall: 'bullish',
          social_score: 0.72,
          news_sentiment: 0.65,
        },
        volume: {
          unusual_activity: true,
          flow_bias: 'call',
          institutional_activity: 'accumulation',
        },
        risk: {
          volatility_rank: 0.45,
          max_drawdown: 0.02,
          sharpe_ratio: 1.8,
        },
      };
    }
  }

  // Mock data generator for precise options signals
  private getMockPreciseOptionsSignals(symbol: string): PreciseOptionsSignal[] {
    return this.getMockPreciseSignals().filter(s => s.symbol === symbol);
  }

  // Get market screener opportunities
  async getMarketOpportunities(): Promise<MarketOpportunity[]> {
    try {
      const response = await this.request<{ opportunities: MarketOpportunity[] }>({
        method: 'GET',
        url: '/api/v1/market/opportunities',
      });

      // Debug logging
      console.log('Market opportunities response:', response);
      console.log('Response type:', typeof response);
      console.log('Has opportunities property:', 'opportunities' in response);
      console.log('Opportunities type:', typeof response.opportunities);
      console.log('Is array:', Array.isArray(response.opportunities));

      // Extract the opportunities array from the response object
      return response.opportunities || [];
    } catch (error) {
      console.error('Error fetching market opportunities:', error);
      // Return mock data for development
      return this.getMockMarketOpportunities();
    }
  }

  // Mock market opportunities
  private getMockMarketOpportunities(): MarketOpportunity[] {
    const opportunities = [
      {
        id: '1',
        symbol: 'NVDA',
        name: 'NVIDIA Corporation',
        type: 'CALL' as const,
        confidence: 92,
        potentialReturn: 15.5,
        timeframe: '2-3 days',
        keyReason: 'AI sector momentum + breakout pattern',
        momentum: 'strong' as const,
        aiScore: 94,
        sector: 'Technology',
        volume: 125000000,
        volatility: 0.42,
      },
      {
        id: '2',
        symbol: 'TSLA',
        name: 'Tesla Inc',
        type: 'PUT' as const,
        confidence: 87,
        potentialReturn: 12.3,
        timeframe: '1-2 days',
        keyReason: 'Overbought + resistance at 250',
        momentum: 'moderate' as const,
        aiScore: 88,
        sector: 'Automotive',
        volume: 98000000,
        volatility: 0.58,
      },
      {
        id: '3',
        symbol: 'META',
        name: 'Meta Platforms Inc',
        type: 'CALL' as const,
        confidence: 85,
        potentialReturn: 10.8,
        timeframe: '3-5 days',
        keyReason: 'Bullish flag + positive sentiment',
        momentum: 'building' as const,
        aiScore: 86,
        sector: 'Technology',
        volume: 45000000,
        volatility: 0.35,
      },
      {
        id: '4',
        symbol: 'SPY',
        name: 'SPDR S&P 500 ETF',
        type: 'CALL' as const,
        confidence: 82,
        potentialReturn: 8.5,
        timeframe: '1 day',
        keyReason: 'Market breadth improving',
        momentum: 'moderate' as const,
        aiScore: 83,
        sector: 'ETF',
        volume: 78000000,
        volatility: 0.18,
      },
      {
        id: '5',
        symbol: 'AMD',
        name: 'Advanced Micro Devices',
        type: 'PUT' as const,
        confidence: 79,
        potentialReturn: 11.2,
        timeframe: '2-3 days',
        keyReason: 'Double top + sector rotation',
        momentum: 'building' as const,
        aiScore: 80,
        sector: 'Technology',
        volume: 65000000,
        volatility: 0.48,
      },
      {
        id: '6',
        symbol: 'AAPL',
        name: 'Apple Inc',
        type: 'CALL' as const,
        confidence: 78,
        potentialReturn: 7.5,
        timeframe: '5-7 days',
        keyReason: 'Support bounce + iPhone sales',
        momentum: 'building' as const,
        aiScore: 79,
        sector: 'Technology',
        volume: 52000000,
        volatility: 0.22,
      },
      {
        id: '7',
        symbol: 'QQQ',
        name: 'Invesco QQQ Trust',
        type: 'CALL' as const,
        confidence: 77,
        potentialReturn: 9.2,
        timeframe: '2-3 days',
        keyReason: 'Tech sector recovery',
        momentum: 'moderate' as const,
        aiScore: 78,
        sector: 'ETF',
        volume: 42000000,
        volatility: 0.24,
      },
      {
        id: '8',
        symbol: 'MSFT',
        name: 'Microsoft Corporation',
        type: 'CALL' as const,
        confidence: 76,
        potentialReturn: 6.8,
        timeframe: '3-4 days',
        keyReason: 'Cloud growth + AI integration',
        momentum: 'moderate' as const,
        aiScore: 77,
        sector: 'Technology',
        volume: 38000000,
        volatility: 0.19,
      },
    ];

    // Randomly shuffle and return top 5-8
    return opportunities
      .sort(() => Math.random() - 0.5)
      .slice(0, Math.floor(Math.random() * 3) + 5);
  }

  // Get market news
  async getMarketNews(): Promise<MarketNews[]> {
    // In production, this would fetch from your backend
    // For now, return mock financial news
    return [
      {
        id: '1',
        title: 'Fed Minutes Show Officials Divided on Rate Cut Timeline',
        source: 'Reuters',
        url: 'https://reuters.com',
        timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(), // 5 minutes ago
        impact: 'HIGH',
        symbols: ['SPY', 'QQQ'],
        sentiment: 'BEARISH',
      },
      {
        id: '2',
        title: 'NVIDIA Announces New AI Chip, Stock Surges 5% in Pre-Market',
        source: 'Bloomberg',
        url: 'https://bloomberg.com',
        timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(), // 15 minutes ago
        impact: 'HIGH',
        symbols: ['NVDA'],
        sentiment: 'BULLISH',
      },
      {
        id: '3',
        title: 'Oil Prices Rise on Middle East Tensions, Energy Sector Gains',
        source: 'CNBC',
        url: 'https://cnbc.com',
        timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(), // 30 minutes ago
        impact: 'MEDIUM',
        symbols: ['XLE', 'USO'],
        sentiment: 'BULLISH',
      },
      {
        id: '4',
        title: 'Apple Vision Pro Sales Below Expectations, Analysts Downgrade',
        source: 'WSJ',
        url: 'https://wsj.com',
        timestamp: new Date(Date.now() - 45 * 60 * 1000).toISOString(), // 45 minutes ago
        impact: 'MEDIUM',
        symbols: ['AAPL'],
        sentiment: 'BEARISH',
      },
      {
        id: '5',
        title: 'Bank Earnings Beat Estimates, Financial Sector Rallies',
        source: 'Financial Times',
        url: 'https://ft.com',
        timestamp: new Date(Date.now() - 60 * 60 * 1000).toISOString(), // 1 hour ago
        impact: 'MEDIUM',
        symbols: ['XLF', 'JPM', 'BAC'],
        sentiment: 'BULLISH',
      },
      {
        id: '6',
        title: 'Tesla Recalls 2M Vehicles Over Autopilot Concerns',
        source: 'AP News',
        url: 'https://apnews.com',
        timestamp: new Date(Date.now() - 90 * 60 * 1000).toISOString(), // 1.5 hours ago
        impact: 'HIGH',
        symbols: ['TSLA'],
        sentiment: 'BEARISH',
      },
    ];
  }
}

export const apiClient = new ApiClient();

// Utility Functions
export const formatPrice = (price: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(price);
};

export const formatPercentage = (percentage: number): string => {
  return `${percentage.toFixed(2)}%`;
};

export const formatVolume = (volume: number): string => {
  if (volume >= 1e9) return `${(volume / 1e9).toFixed(2)}B`;
  if (volume >= 1e6) return `${(volume / 1e6).toFixed(2)}M`;
  if (volume >= 1e3) return `${(volume / 1e3).toFixed(2)}K`;
  return volume.toString();
};

export const getSignalColor = (signalType: string): string => {
  switch (signalType) {
    case 'BUY':
      return '#4caf50'; // green
    case 'SELL':
      return '#f44336'; // red
    default:
      return '#ff9800'; // orange
  }
};

export const getSignalIcon = (signalType: string): string => {
  switch (signalType) {
    case 'BUY':
      return '▲';
    case 'SELL':
      return '▼';
    default:
      return '●';
  }
};

export const fetchMarketData = async (symbol: string): Promise<MarketData> => {
  const response = await apiClient.getMarketData(symbol);
  return response;
};

export const fetchActiveSignals = async (): Promise<PreciseOptionsSignal[]> => {
  const response = await apiClient.getActiveSignals();
  return response;
};

export const fetchAIInsights = async (symbol: string): Promise<AIInsight> => {
  const response = await apiClient.getAIInsights(symbol);
  return response;
};

export default apiClient; 