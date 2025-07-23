// WebSocket service for real-time chart data and AI predictions
import { Time } from 'lightweight-charts';

export interface MarketData {
  symbol: string;
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  bid?: number;
  ask?: number;
}

export interface PredictionData {
  symbol: string;
  time: Time;
  price: number;
  confidence: number;
  timeframe: number; // minutes ahead
  accuracy?: number;
}

export interface SignalData {
  id: string;
  symbol: string;
  type: 'buy' | 'sell';
  price: number;
  time: Time;
  confidence: number;
  stopLoss?: number;
  takeProfit?: number[];
  reasoning?: string;
  agents?: {
    total: number;
    agreeing: number;
    disagreeing: number;
  };
}

export interface PatternData {
  id: string;
  symbol: string;
  type: string;
  startTime: Time;
  endTime: Time;
  confidence: number;
  description: string;
  target?: number;
}

export interface WebSocketMessage {
  type: 'market' | 'prediction' | 'signal' | 'pattern' | 'indicator' | 'error';
  data: any;
  timestamp: number;
}

export class ChartWebSocketService {
  private ws: WebSocket | null = null;
  private reconnectInterval = 5000;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private subscribers: Map<string, Set<(data: any) => void>> = new Map();
  private messageQueue: WebSocketMessage[] = [];
  private isConnected = false;
  private url: string;

  constructor(url: string = 'ws://localhost:8000/ws/chart') {
    this.url = url;
  }

  // Connect to WebSocket
  connect(symbol: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      this.ws = new WebSocket(`${this.url}?symbol=${symbol}`);
      this.setupEventHandlers();
    } catch (error) {
      logger.error('WebSocket connection error:', error);
      this.scheduleReconnect(symbol);
    }
  }

  // Setup WebSocket event handlers
  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      logger.info('WebSocket connected');
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.startHeartbeat();
      this.flushMessageQueue();
      this.notifySubscribers('connection', { status: 'connected' });
    };

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        logger.error('Error parsing WebSocket message:', error);
      }
    };

    this.ws.onerror = (error) => {
      logger.error('WebSocket error:', error);
      this.notifySubscribers('error', { error });
    };

    this.ws.onclose = () => {
      logger.info('WebSocket disconnected');
      this.isConnected = false;
      this.stopHeartbeat();
      this.notifySubscribers('connection', { status: 'disconnected' });
      this.scheduleReconnect();
    };
  }

  // Handle incoming messages
  private handleMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case 'market':
        this.notifySubscribers('market', message.data as MarketData);
        break;
      case 'prediction':
        this.notifySubscribers('prediction', message.data as PredictionData);
        break;
      case 'signal':
        this.notifySubscribers('signal', message.data as SignalData);
        break;
      case 'pattern':
        this.notifySubscribers('pattern', message.data as PatternData);
        break;
      case 'indicator':
        this.notifySubscribers('indicator', message.data);
        break;
      case 'error':
        this.notifySubscribers('error', message.data);
        break;
      default:
        logger.warn('Unknown message type:', message.type);
    }
  }

  // Subscribe to specific events
  subscribe(event: string, callback: (data: any) => void): () => void {
    if (!this.subscribers.has(event)) {
      this.subscribers.set(event, new Set());
    }

    this.subscribers.get(event)!.add(callback);

    // Return unsubscribe function
    return () => {
      const callbacks = this.subscribers.get(event);
      if (callbacks) {
        callbacks.delete(callback);
        if (callbacks.size === 0) {
          this.subscribers.delete(event);
        }
      }
    };
  }

  // Notify all subscribers of an event
  private notifySubscribers(event: string, data: any): void {
    const callbacks = this.subscribers.get(event);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          logger.error(`Error in subscriber callback for ${event}:`, error);
        }
      });
    }
  }

  // Send message to server
  send(message: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      // Queue message if not connected
      this.messageQueue.push({
        type: 'market',
        data: message,
        timestamp: Date.now(),
      });
    }
  }

  // Request specific data
  requestPrediction(symbol: string, timeframe: string): void {
    this.send({
      action: 'request_prediction',
      symbol,
      timeframe,
    });
  }

  requestSignals(symbol: string): void {
    this.send({
      action: 'request_signals',
      symbol,
    });
  }

  requestPatterns(symbol: string, timeframe: string): void {
    this.send({
      action: 'request_patterns',
      symbol,
      timeframe,
    });
  }

  requestIndicator(symbol: string, indicator: string, params?: any): void {
    this.send({
      action: 'request_indicator',
      symbol,
      indicator,
      params,
    });
  }

  // Subscribe to real-time updates
  subscribeToSymbol(symbol: string, timeframe: string): void {
    this.send({
      action: 'subscribe',
      symbol,
      timeframe,
    });
  }

  unsubscribeFromSymbol(symbol: string): void {
    this.send({
      action: 'unsubscribe',
      symbol,
    });
  }

  // Heartbeat to keep connection alive
  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ action: 'ping' });
      }
    }, 30000); // Every 30 seconds
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  // Reconnection logic
  private scheduleReconnect(symbol?: string): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      logger.error('Max reconnection attempts reached');
      this.notifySubscribers('error', {
        error: 'Max reconnection attempts reached',
        canRetry: false,
      });
      return;
    }

    this.reconnectAttempts++;
    logger.info(`Reconnecting in ${this.reconnectInterval}ms... (Attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      if (symbol) {
        this.connect(symbol);
      }
    }, this.reconnectInterval);
  }

  // Flush queued messages
  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
      const message = this.messageQueue.shift();
      if (message) {
        this.ws.send(JSON.stringify(message.data));
      }
    }
  }

  // Disconnect WebSocket
  disconnect(): void {
    this.stopHeartbeat();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.subscribers.clear();
    this.messageQueue = [];
    this.isConnected = false;
  }

  // Get connection status
  getConnectionStatus(): boolean {
    return this.isConnected;
  }
}

// Singleton instance
let chartWebSocketInstance: ChartWebSocketService | null = null;

export function getChartWebSocket(): ChartWebSocketService {
  if (!chartWebSocketInstance) {
    chartWebSocketInstance = new ChartWebSocketService();
  }
  return chartWebSocketInstance;
}

// React Hook for WebSocket
import { useEffect, useRef, useState } from 'react';
import logger from '../logger';


export interface UseChartWebSocketOptions {
  symbol: string;
  timeframe: string;
  onMarketData?: (data: MarketData) => void;
  onPrediction?: (data: PredictionData) => void;
  onSignal?: (data: SignalData) => void;
  onPattern?: (data: PatternData) => void;
  onError?: (error: any) => void;
}

export function useChartWebSocket(options: UseChartWebSocketOptions) {
  const { symbol, timeframe, onMarketData, onPrediction, onSignal, onPattern, onError } = options;
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<ChartWebSocketService | null>(null);

  useEffect(() => {
    const ws = getChartWebSocket();
    wsRef.current = ws;

    // Connect to WebSocket
    ws.connect(symbol);

    // Subscribe to events
    const unsubscribers: (() => void)[] = [];

    unsubscribers.push(
      ws.subscribe('connection', (data) => {
        setIsConnected(data.status === 'connected');
      })
    );

    if (onMarketData) {
      unsubscribers.push(ws.subscribe('market', onMarketData));
    }

    if (onPrediction) {
      unsubscribers.push(ws.subscribe('prediction', onPrediction));
    }

    if (onSignal) {
      unsubscribers.push(ws.subscribe('signal', onSignal));
    }

    if (onPattern) {
      unsubscribers.push(ws.subscribe('pattern', onPattern));
    }

    if (onError) {
      unsubscribers.push(ws.subscribe('error', onError));
    }

    // Subscribe to symbol updates
    ws.subscribeToSymbol(symbol, timeframe);

    // Cleanup
    return () => {
      unsubscribers.forEach(unsubscribe => unsubscribe());
      ws.unsubscribeFromSymbol(symbol);
    };
  }, [symbol, timeframe]);

  return {
    isConnected,
    requestPrediction: () => wsRef.current?.requestPrediction(symbol, timeframe),
    requestSignals: () => wsRef.current?.requestSignals(symbol),
    requestPatterns: () => wsRef.current?.requestPatterns(symbol, timeframe),
    requestIndicator: (indicator: string, params?: any) =>
      wsRef.current?.requestIndicator(symbol, indicator, params),
  };
}
