import axios from 'axios';
import { signalWebSocketManager } from '../websocket/SignalWebSocketManager';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Types
export interface StockSearchResult {
  symbol: string;
  name: string;
  exchange: string;
  type: 'stock' | 'etf' | 'crypto';
  marketCap?: string;
  sector?: string;
}

export interface PriceData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Signal {
  id: string;
  symbol: string;
  action: 'buy' | 'sell' | 'hold';
  price: number;
  confidence: number;
  stopLoss?: number;
  takeProfit?: number[];
  timestamp: string;
  agents?: AgentVote[];
}

export interface AgentVote {
  agentId: string;
  agentName: string;
  vote: string;
  confidence: number;
  weight: number;
}

export interface NewsItem {
  id: string;
  title: string;
  source: string;
  time: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  impact: 'high' | 'medium' | 'low';
  url: string;
  summary?: string;
}

export interface OptionsFlow {
  strike: number;
  expiry: string;
  type: 'call' | 'put';
  volume: number;
  openInterest: number;
  premium: number;
  flow: 'bullish' | 'bearish';
}

// API Client
class TradingAPI {
  private axiosInstance = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Stock Search
  async searchStocks(query: string): Promise<StockSearchResult[]> {
    try {
      const response = await this.axiosInstance.get('/api/v1/stocks/search', {
        params: { q: query },
      });
      return response.data;
    } catch (error) {
      console.error('Error searching stocks:', error);
      // Fallback to mock data if API fails
      return this.getMockStockData().filter(
        stock =>
          stock.symbol.toLowerCase().includes(query.toLowerCase()) ||
          stock.name.toLowerCase().includes(query.toLowerCase())
      );
    }
  }

  // Price Data
  async getPriceData(symbol: string, timeframe: string): Promise<PriceData[]> {
    try {
      const response = await this.axiosInstance.get(`/api/v1/stocks/${symbol}/prices`, {
        params: { timeframe },
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching price data:', error);
      throw error;
    }
  }

  // Signal Generation
  async generateSignal(symbol: string, timeframe: string): Promise<Signal> {
    try {
      const response = await this.axiosInstance.post('/api/v1/signals/generate', {
        symbol,
        timeframe,
      });
      return response.data;
    } catch (error) {
      console.error('Error generating signal:', error);
      throw error;
    }
  }

  // Get Latest Signals
  async getLatestSignals(symbol?: string): Promise<Signal[]> {
    try {
      const response = await this.axiosInstance.get('/api/v1/signals', {
        params: symbol ? { symbol } : {},
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching signals:', error);
      throw error;
    }
  }

  // News Feed
  async getNews(symbol: string): Promise<NewsItem[]> {
    try {
      const response = await this.axiosInstance.get(`/api/v1/news/${symbol}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching news:', error);
      // Return mock data as fallback
      return this.getMockNews();
    }
  }

  // Options Flow
  async getOptionsFlow(symbol: string): Promise<OptionsFlow[]> {
    try {
      const response = await this.axiosInstance.get(`/api/v1/options/${symbol}/flow`);
      return response.data;
    } catch (error) {
      console.error('Error fetching options flow:', error);
      return this.getMockOptionsFlow();
    }
  }

  // Market Metrics
  async getMarketMetrics(): Promise<any> {
    try {
      const response = await this.axiosInstance.get('/api/v1/market/metrics');
      return response.data;
    } catch (error) {
      console.error('Error fetching market metrics:', error);
      throw error;
    }
  }

  // WebSocket Connection for Real-time Data
  connectToRealTimeData(symbol: string, onUpdate: (data: any) => void) {
    // Use the existing WebSocket manager
    signalWebSocketManager.connect();
    
    // Subscribe to price updates
    signalWebSocketManager.subscribe('price_update', (data) => {
      if (data.symbol === symbol) {
        onUpdate({ type: 'price', data });
      }
    });

    // Subscribe to signal updates
    signalWebSocketManager.subscribe('signal_update', (data) => {
      if (data.symbol === symbol) {
        onUpdate({ type: 'signal', data });
      }
    });

    // Return cleanup function
    return () => {
      signalWebSocketManager.unsubscribe('price_update');
      signalWebSocketManager.unsubscribe('signal_update');
    };
  }

  // Mock Data Methods (for fallback)
  private getMockStockData(): StockSearchResult[] {
    return [
      { symbol: 'AAPL', name: 'Apple Inc.', exchange: 'NASDAQ', type: 'stock', marketCap: '$2.95T', sector: 'Technology' },
      { symbol: 'MSFT', name: 'Microsoft Corporation', exchange: 'NASDAQ', type: 'stock', marketCap: '$2.85T', sector: 'Technology' },
      { symbol: 'GOOGL', name: 'Alphabet Inc.', exchange: 'NASDAQ', type: 'stock', marketCap: '$1.75T', sector: 'Technology' },
      { symbol: 'AMZN', name: 'Amazon.com Inc.', exchange: 'NASDAQ', type: 'stock', marketCap: '$1.56T', sector: 'Consumer Cyclical' },
      { symbol: 'NVDA', name: 'NVIDIA Corporation', exchange: 'NASDAQ', type: 'stock', marketCap: '$1.12T', sector: 'Technology' },
    ];
  }

  private getMockNews(): NewsItem[] {
    return [
      {
        id: '1',
        title: 'Market Update: Tech Stocks Rally',
        source: 'Reuters',
        time: '5 min ago',
        sentiment: 'positive',
        impact: 'high',
        url: '#',
        summary: 'Technology stocks lead market gains.',
      },
    ];
  }

  private getMockOptionsFlow(): OptionsFlow[] {
    return [
      {
        strike: 180,
        expiry: '12/15',
        type: 'call',
        volume: 25432,
        openInterest: 8123,
        premium: 2450000,
        flow: 'bullish',
      },
    ];
  }
}

// Export singleton instance
export const tradingAPI = new TradingAPI();