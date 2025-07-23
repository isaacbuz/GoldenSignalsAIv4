import logger from '../logger';

/**
 * WebSocket service for real-time market data streaming
 * Connects to Finnhub WebSocket API for live price updates
 */

export interface MarketDataUpdate {
  symbol: string;
  price: number;
  volume: number;
  timestamp: number;
  conditions?: string[];
}

export interface Trade {
  s: string;  // Symbol
  p: number;  // Price
  t: number;  // Timestamp in milliseconds
  v: number;  // Volume
  c?: string[]; // Conditions
}

class MarketDataWebSocket {
  private ws: WebSocket | null = null;
  private subscribers: Map<string, Set<(data: MarketDataUpdate) => void>> = new Map();
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private apiKey: string;
  private isConnecting = false;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private subscribedSymbols: Set<string> = new Set();

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  public connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      return;
    }

    this.isConnecting = true;

    try {
      // Connect to our backend WebSocket endpoint
      const wsUrl = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000';
      // Use a generic endpoint and subscribe to symbols via messages
      this.ws = new WebSocket(`${wsUrl}/ws/market-data`);

      this.ws.onopen = () => {
        logger.info('✅ WebSocket connected to market data stream');
        this.isConnecting = false;
        this.reconnectAttempts = 0;

        // Resubscribe to all symbols after reconnection
        this.subscribedSymbols.forEach(symbol => {
          this.sendSubscription(symbol);
        });

        // Start heartbeat
        this.startHeartbeat();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'trade' && data.data) {
            // Process trade data
            data.data.forEach((trade: Trade) => {
              const update: MarketDataUpdate = {
                symbol: trade.s,
                price: trade.p,
                volume: trade.v,
                timestamp: trade.t,
                conditions: trade.c
              };

              // Notify subscribers
              const symbolSubscribers = this.subscribers.get(trade.s);
              if (symbolSubscribers) {
                symbolSubscribers.forEach(callback => callback(update));
              }
            });
          }
        } catch (error) {
          logger.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onerror = (error) => {
        logger.error('WebSocket error:', error);
        this.isConnecting = false;
      };

      this.ws.onclose = (event) => {
        logger.info('WebSocket disconnected', { code: event.code, reason: event.reason });
        this.isConnecting = false;
        this.stopHeartbeat();

        // Only attempt reconnect if not a deliberate disconnect
        if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.attemptReconnect();
        }
      };
    } catch (error) {
      logger.error('Failed to create WebSocket:', error);
      this.isConnecting = false;
      this.attemptReconnect();
    }
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();

    // Send ping every 30 seconds to keep connection alive
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private attemptReconnect(): void {
    // Clear any existing reconnect timeout
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      logger.error('Max reconnection attempts reached');
      // Notify subscribers of connection failure
      this.notifyConnectionError();
      return;
    }

    this.reconnectAttempts++;
    // Exponential backoff with jitter
    const baseDelay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    const jitter = Math.random() * 1000; // Add up to 1 second of jitter
    const delay = Math.min(baseDelay + jitter, 30000); // Max 30 seconds

    logger.info(`Attempting to reconnect in ${Math.round(delay)}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectTimeout = null;
      this.connect();
    }, delay);
  }

  private notifyConnectionError(): void {
    // Notify all subscribers of connection failure
    this.subscribers.forEach((callbacks, symbol) => {
      callbacks.forEach(callback => {
        callback({
          symbol,
          price: 0,
          volume: 0,
          timestamp: Date.now(),
          conditions: ['CONNECTION_ERROR']
        });
      });
    });
  }

  private sendSubscription(symbol: string): void {
    // For backend WebSocket, send subscription message instead of reconnecting
    if (this.ws?.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({
        type: 'subscribe',
        symbol: symbol
      });
      this.ws.send(message);
      logger.info(`📊 Subscribed to ${symbol}`);
    }
  }

  public subscribe(symbol: string, callback: (data: MarketDataUpdate) => void): () => void {
    // Add to subscribers
    if (!this.subscribers.has(symbol)) {
      this.subscribers.set(symbol, new Set());
    }
    this.subscribers.get(symbol)!.add(callback);

    // Track subscribed symbols
    if (!this.subscribedSymbols.has(symbol)) {
      this.subscribedSymbols.add(symbol);
      this.sendSubscription(symbol);
    }

    // Connect if not connected
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.connect();
    }

    // Return unsubscribe function
    return () => {
      const symbolSubscribers = this.subscribers.get(symbol);
      if (symbolSubscribers) {
        symbolSubscribers.delete(callback);

        // If no more subscribers for this symbol, unsubscribe
        if (symbolSubscribers.size === 0) {
          this.subscribers.delete(symbol);
          this.subscribedSymbols.delete(symbol);

          if (this.ws?.readyState === WebSocket.OPEN) {
            const message = JSON.stringify({
              type: 'unsubscribe',
              symbol: symbol
            });
            this.ws.send(message);
            logger.info(`📊 Unsubscribed from ${symbol}`);
          }
        }
      }
    };
  }

  public disconnect(): void {
    // Prevent reconnection attempts
    this.reconnectAttempts = this.maxReconnectAttempts;

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    this.stopHeartbeat();

    if (this.ws) {
      // Remove all event handlers before closing
      this.ws.onopen = null;
      this.ws.onmessage = null;
      this.ws.onerror = null;
      this.ws.onclose = null;

      if (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING) {
        this.ws.close(1000, 'Client disconnect');
      }
      this.ws = null;
    }

    // Clear all subscribers and symbols
    this.subscribers.clear();
    this.subscribedSymbols.clear();
    this.isConnecting = false;
  }

  public isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Create singleton instance
let marketDataWS: MarketDataWebSocket | null = null;

export const getMarketDataWebSocket = (apiKey?: string): MarketDataWebSocket => {
  if (!marketDataWS && apiKey) {
    marketDataWS = new MarketDataWebSocket(apiKey);
  }

  if (!marketDataWS) {
    throw new Error('MarketDataWebSocket not initialized. Please provide API key.');
  }

  return marketDataWS;
};

export const disconnectMarketDataWebSocket = (): void => {
  if (marketDataWS) {
    marketDataWS.disconnect();
    marketDataWS = null;
  }
};

// Clean up on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    disconnectMarketDataWebSocket();
  });
}
