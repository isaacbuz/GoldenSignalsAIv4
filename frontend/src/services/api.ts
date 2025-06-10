/**
 * API Service Layer for GoldenSignalsAI V3
 * 
 * Centralized API communication with the FastAPI backend
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import toast from 'react-hot-toast';

// API Configuration
const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';
const WS_BASE_URL = (import.meta as any).env?.VITE_WS_BASE_URL || 'ws://localhost:8000';

// Types
export interface Signal {
  signal_id?: string;
  symbol: string;
  signal_type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  strength?: 'WEAK' | 'MODERATE' | 'STRONG';
  source?: string;
  current_price?: number;
  entry_price?: number;
  exit_price?: number;
  target_price?: number;
  price_target?: number;
  take_profit?: number;
  stop_loss?: number;
  risk_score?: number;
  reasoning?: string;
  features?: Record<string, any>;
  indicators?: Record<string, any>;
  created_at?: string;
  exit_timestamp?: string;
  expires_at?: string;
  timestamp?: string;
}

export interface MarketData {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  bid?: number;
  ask?: number;
  spread?: number;
  timestamp: string;
  provider?: string;
  market_cap?: number;
  pe_ratio?: number;
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
  private async request<T>(config: AxiosRequestConfig): Promise<T> {
    const response: AxiosResponse<T> = await this.instance.request(config);
    return response.data;
  }

  // Signal endpoints - Updated to match backend
  async getSignals(params?: {
    symbols?: string[];
    signal_types?: string[];
    limit?: number;
    offset?: number;
  }): Promise<{ signals: Signal[]; total: number }> {
    // Since backend only has individual signal endpoints, we'll fetch for multiple symbols
    const symbols = params?.symbols || ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'NVDA'];
    const signalPromises = symbols.map(symbol => this.getSignalBySymbol(symbol));
    
    try {
      const signals = await Promise.all(signalPromises);
      const validSignals = signals.filter(s => s !== null) as Signal[];
      return { signals: validSignals, total: validSignals.length };
    } catch (error) {
      console.error('Error fetching signals:', error);
      return { signals: [], total: 0 };
    }
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
        signal_id: `${symbol}-${Date.now()}`,
        symbol: data.symbol,
        signal_type: data.signal as 'BUY' | 'SELL' | 'HOLD',
        confidence: data.confidence,
        strength: data.confidence > 0.8 ? 'STRONG' : data.confidence > 0.6 ? 'MODERATE' : 'WEAK',
        source: 'ai_agent',
        current_price: data.indicators?.current_price || 0,
        price_target: data.price_target,
        stop_loss: data.stop_loss,
        risk_score: data.risk_score,
        reasoning: `Signal generated with ${Math.round(data.confidence * 100)}% confidence based on technical indicators`,
        indicators: data.indicators,
        timestamp: data.timestamp,
        created_at: data.timestamp,
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
    const symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'NVDA', 'SPY', 'QQQ', 'META', 'AMZN'];
    const promises = symbols.slice(0, limit).map(symbol => 
      this.getSignalBySymbol(symbol).catch(() => null)
    );
    
    const signals = await Promise.all(promises);
    return signals.filter(s => s !== null) as Signal[];
  }

  // Market data endpoints
  async getMarketData(symbol: string): Promise<MarketData> {
    return this.request({
      method: 'GET',
      url: `/market-data/${symbol.toUpperCase()}`,
    });
  }

  async getMultipleMarketData(symbols: string[]): Promise<Record<string, MarketData>> {
    const promises = symbols.map(async symbol => {
      try {
        const data = await this.getMarketData(symbol);
        return { symbol, data };
      } catch (error) {
        console.error(`Error fetching data for ${symbol}:`, error);
        return null;
      }
    });
    
    const results = await Promise.all(promises);
    const marketData: Record<string, MarketData> = {};
    
    results.forEach(result => {
      if (result) {
        marketData[result.symbol] = result.data;
      }
    });
    
    return marketData;
  }

  async getHistoricalData(
    symbol: string,
    period: string = '1d',
    interval: string = '5m'
  ): Promise<any[]> {
    // This endpoint doesn't exist in backend yet, return mock data
    console.warn('Historical data endpoint not implemented in backend');
    return [];
  }

  async getMarketStatus(): Promise<MarketStatus> {
    // This endpoint doesn't exist in backend yet, return mock data
    return {
      is_open: true,
      current_time: new Date().toISOString(),
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

  async getAgentPerformance(): Promise<{ agents: Record<string, AgentPerformance>; summary: any }> {
    return this.request({
      method: 'GET',
      url: '/agents/performance',
    });
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