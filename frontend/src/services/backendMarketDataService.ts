/**
 * Backend Market Data Service
 * Enhanced service that connects to our Python backend for real-time market data
 */

import { Time } from 'lightweight-charts';
import logger from './logger';
import { marketDataNormalizer } from './marketDataNormalizer';


export interface MarketDataPoint {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
  isGapFilled?: boolean; // Mark gap-filled candles for different rendering
}

export interface MarketDataResponse {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume?: number;
  market_cap?: number;
  pe_ratio?: number;
  timestamp: string;
}

export interface HistoricalDataResponse {
  symbol: string;
  period: string;
  interval: string;
  data: MarketDataPoint[];
}

export interface SignalData {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  timestamp: string;
  time?: Time;
  agents_consensus?: {
    agentsInFavor: number;
    totalAgents: number;
  };
  stop_loss?: number;
  take_profit?: number[];
  reasoning?: string;
}

class BackendMarketDataService {
  private baseUrl: string;
  private cache: Map<string, { data: any; timestamp: number }>;
  private cacheTimeout: number = 30000; // 30 seconds cache
  private wsConnection: WebSocket | null = null;
  private wsCallbacks: Map<string, (data: any) => void> = new Map();

  constructor() {
    this.baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
    this.cache = new Map();
  }

  /**
   * Get current market data for a symbol
   */
  async getCurrentMarketData(symbol: string): Promise<MarketDataResponse> {
    const cacheKey = `market-${symbol}`;
    const cached = this.getFromCache(cacheKey);
    if (cached) return cached;

    try {
      const response = await fetch(`${this.baseUrl}/api/v1/market-data/${symbol}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        mode: 'cors',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.setCache(cacheKey, data);
      return data;
    } catch (error) {
      logger.error('Error fetching market data:', error);
      // Throw error instead of returning mock data
      throw new Error(`Failed to fetch market data for ${symbol}: ${error.message}`);
    }
  }

  /**
   * Get historical OHLC data for charting
   */
  async getHistoricalData(
    symbol: string,
    period: string = '30d',
    interval: string = '1d'
  ): Promise<HistoricalDataResponse> {
    const cacheKey = `history-${symbol}-${period}-${interval}`;
    const cached = this.getFromCache(cacheKey);
    if (cached) return cached;

    try {
      const params = new URLSearchParams({
        period,
        interval,
      });

      const response = await fetch(
        `${this.baseUrl}/api/v1/market-data/${symbol}/historical?${params}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
          mode: 'cors',
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Format and normalize the data with gap filling
      if (data.data && Array.isArray(data.data)) {
        data.data = this.fillDataGaps(data.data, interval);
      }

      this.setCache(cacheKey, data);
      return data;
    } catch (error) {
      logger.error('Error fetching historical data:', error);
      // Throw error instead of returning mock data
      throw new Error(`Failed to fetch historical data for ${symbol}: ${error.message}`);
    }
  }

  /**
   * Get AI trading signals for a symbol
   */
  async getSignals(symbol: string): Promise<SignalData[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/signals/symbol/${symbol}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        mode: 'cors',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Transform signals to include time field for chart
      const signals = (data.signals || []).map((signal: any) => ({
        ...signal,
        // Ensure timestamp is properly parsed (backend sends ISO strings)
        time: signal.timestamp ? Math.floor(new Date(signal.timestamp).getTime() / 1000) : Date.now() / 1000,
        agents_consensus: signal.agents_consensus || {
          agentsInFavor: signal.consensus_strength ? Math.floor(signal.consensus_strength * 30) : 20,
          totalAgents: 30,
        },
      }));

      return signals;
    } catch (error) {
      logger.error('Error fetching signals:', error);
      // Throw error instead of returning mock data
      throw new Error(`Failed to fetch signals for ${symbol}: ${error.message}`);
    }
  }

  /**
   * Trigger AI analysis for a symbol
   */
  async analyzeSymbol(symbol: string, timeframe: string = '5m'): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/workflow/analyze/${symbol}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ timeframe }),
        mode: 'cors',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      logger.error('Error analyzing symbol:', error);
      throw error;
    }
  }

  /**
   * Connect to WebSocket for real-time updates
   */
  async connectWebSocket(symbol: string, onMessage: (data: any) => void): Promise<void> {
    const wsUrl = this.baseUrl.replace('http', 'ws') + `/ws`;

    try {
      this.wsConnection = new WebSocket(wsUrl);

      this.wsConnection.onopen = () => {
        logger.info('WebSocket connected');
        // Subscribe to market data for the symbol
        this.wsConnection?.send(JSON.stringify({
          type: 'subscribe',
          data: {
            channel: 'market_data',
            symbol: symbol,
          }
        }));
      };

      this.wsConnection.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (error) {
          logger.error('Error parsing WebSocket message:', error);
        }
      };

      this.wsConnection.onerror = (error) => {
        logger.error('WebSocket error:', error);
      };

      this.wsConnection.onclose = () => {
        logger.info('WebSocket disconnected');
        // Attempt to reconnect after 5 seconds
        setTimeout(() => this.connectWebSocket(symbol, onMessage), 5000);
      };
    } catch (error) {
      logger.error('Error connecting to WebSocket:', error);
    }
  }

  /**
   * Disconnect WebSocket
   */
  disconnectWebSocket(): void {
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
  }

  /**
   * Map timeframe to period and interval for API
   * Matches how professional platforms fetch data
   */
  mapTimeframeToPeriodInterval(timeframe: string): { period: string; interval: string } {
    const mapping: Record<string, { period: string; interval: string }> = {
      // Intraday
      '1m': { period: '1d', interval: '1m' },     // 1 day of 1-minute data
      '5m': { period: '5d', interval: '5m' },     // 5 days of 5-minute data
      '15m': { period: '5d', interval: '15m' },   // 5 days of 15-minute data
      '30m': { period: '1mo', interval: '30m' },  // 1 month of 30-minute data
      '1h': { period: '1mo', interval: '1h' },    // 1 month of hourly data
      '4h': { period: '3mo', interval: '1h' },    // 3 months of hourly data (4h not supported by yfinance)

      // Daily & Above
      '1d': { period: '1y', interval: '1d' },     // 1 year of daily data
      '1w': { period: '5y', interval: '1wk' },    // 5 years of weekly data
      '1M': { period: '10y', interval: '1mo' },   // 10 years of monthly data

      // Long Term - Optimized for best data density
      '3M': { period: 'max', interval: '3mo' },   // All available quarterly data
      '6M': { period: 'max', interval: '3mo' },   // All available data (6M not supported, use 3M)
      '1y': { period: 'max', interval: '1mo' },   // All available monthly data
      '2y': { period: 'max', interval: '1mo' },   // All available monthly data
      '5y': { period: 'max', interval: '1mo' },   // All available monthly data
      '10y': { period: 'max', interval: '3mo' },  // All available quarterly data
      'max': { period: 'max', interval: '3mo' },  // All available quarterly data
    };

    return mapping[timeframe] || { period: '1mo', interval: '1d' };
  }

  /**
   * Fill gaps in historical data to prevent sparse candlesticks
   */
  private fillDataGaps(data: any[], interval: string): MarketDataPoint[] {
    if (!data || data.length < 2) return data;

    const intervalSeconds = this.getIntervalSeconds(interval);
    const filled: MarketDataPoint[] = [];

    // Sort data by time
    const sorted = [...data].sort((a, b) => a.time - b.time);

    for (let i = 0; i < sorted.length; i++) {
      const current = sorted[i];
      filled.push({
        time: current.time as Time,
        open: current.open,
        high: current.high,
        low: current.low,
        close: current.close,
        volume: current.volume || 0,
      });

      // Check if there's a gap to the next candle
      if (i < sorted.length - 1) {
        const next = sorted[i + 1];
        const currentTime = typeof current.time === 'string' ?
          new Date(current.time).getTime() / 1000 : current.time;
        const nextTime = typeof next.time === 'string' ?
          new Date(next.time).getTime() / 1000 : next.time;
        const gap = nextTime - currentTime;

        // If gap is larger than expected interval, fill it
        if (gap > intervalSeconds * 1.5) {
          const numCandles = Math.floor(gap / intervalSeconds) - 1;
          const closePrice = current.close;

          // Add flat candles for the gap with slight noise
          for (let j = 1; j <= numCandles && j < 100; j++) { // Limit to 100 to prevent infinite loops
            // Add small random noise (±0.1%) to make gaps more realistic
            const noise = 1 + (Math.random() - 0.5) * 0.001;
            const gapPrice = closePrice * noise;

            filled.push({
              time: (currentTime + (j * intervalSeconds)) as Time,
              open: gapPrice,
              high: gapPrice * (1 + Math.random() * 0.0005), // Slight high variation
              low: gapPrice * (1 - Math.random() * 0.0005),  // Slight low variation
              close: gapPrice,
              volume: 0,
              isGapFilled: true, // Mark as gap-filled
            });
          }
        }
      }
    }

    logger.debug(`Filled ${filled.length - data.length} gap candles (${data.length} -> ${filled.length})`);
    return filled;
  }

  /**
   * Convert backend data to chart format with normalization
   */
  formatChartData(data: any[], timeframe?: string): MarketDataPoint[] {
    const formatted = data.map(point => ({
      time: point.time as Time,
      open: point.open,
      high: point.high,
      low: point.low,
      close: point.close,
      volume: point.volume,
    }));

    // Validate data
    const validation = marketDataNormalizer.validateData(formatted);
    if (!validation.isValid) {
      logger.warn('Data validation issues:', validation.issues);
    }

    return formatted;
  }

  // Cache management
  private getFromCache(key: string): any | null {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }
    this.cache.delete(key);
    return null;
  }

  private setCache(key: string, data: any): void {
    this.cache.set(key, { data, timestamp: Date.now() });
  }

  clearCache(): void {
    this.cache.clear();
  }

  // Mock data generators for fallback
  // Removed mock data generation - using only real data

  // Removed mock signal generation - using only real data

  private getIntervalSeconds(interval: string): number {
    const mapping: Record<string, number> = {
      '1m': 60,
      '5m': 300,
      '15m': 900,
      '30m': 1800,
      '1h': 3600,
      '4h': 14400,
      '1d': 86400,
      '1wk': 604800,
      '1mo': 2592000,
    };
    return mapping[interval] || 86400;
  }

  private getNumPoints(period: string, interval: string): number {
    const periodDays = {
      '1d': 1,
      '5d': 5,
      '30d': 30,
      '1mo': 30,
      '3mo': 90,
      '6mo': 180,
      '1y': 365,
      '2y': 730,
      '5y': 1825,
    };

    const intervalMinutes = {
      '1m': 1,
      '5m': 5,
      '15m': 15,
      '30m': 30,
      '1h': 60,
      '4h': 240,
      '1d': 1440,
      '1wk': 10080,
      '1mo': 43200,
    };

    const days = periodDays[period] || 30;
    const minutesPerInterval = intervalMinutes[interval] || 1440;

    return Math.floor((days * 1440) / minutesPerInterval);
  }
}

// Singleton instance
export const backendMarketDataService = new BackendMarketDataService();

// Convenience functions
export const fetchMarketData = (symbol: string) => backendMarketDataService.getCurrentMarketData(symbol);
export const fetchHistoricalData = (symbol: string, timeframe: string) => {
  const { period, interval } = backendMarketDataService.mapTimeframeToPeriodInterval(timeframe);
  return backendMarketDataService.getHistoricalData(symbol, period, interval);
};
export const fetchSignals = (symbol: string) => backendMarketDataService.getSignals(symbol);
export const analyzeSymbol = (symbol: string, timeframe: string) =>
  backendMarketDataService.analyzeSymbol(symbol, timeframe);
