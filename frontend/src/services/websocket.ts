/**
 * Enhanced WebSocket Service for GoldenSignalsAI V3
 * 
 * Real-time signal streaming with automatic reconnection and error handling
 */

import { useAppStore, createNotification } from '../store';
import { Signal, MarketData } from './api';

export interface WebSocketMessage {
  type: 'signal' | 'market_data' | 'market_update' | 'agent_status' | 'system_alert' | 'portfolio_update' | 'new_signal' | 'connection_established';
  data: any;
  timestamp: string;
}

interface WebSocketConfig {
  url: string;
  reconnectInterval: number;
  maxReconnectAttempts: number;
  heartbeatInterval: number;
}

const DEFAULT_CONFIG: WebSocketConfig = {
  url: import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws',
  reconnectInterval: 5000,
  maxReconnectAttempts: 10,
  heartbeatInterval: 30000,
};

class WebSocketService {
  private ws: WebSocket | null = null;
  private config: WebSocketConfig;
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private messageQueue: WebSocketMessage[] = [];
  private subscriptions = new Set<string>();
  private isIntentionallyClosed = false;
  private connectionPromise: Promise<void> | null = null;
  private connectionResolver: (() => void) | null = null;

  constructor(config: Partial<WebSocketConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.connect();
  }

  private connect(): Promise<void> {
    if (this.connectionPromise) {
      return this.connectionPromise;
    }

    this.connectionPromise = new Promise((resolve) => {
      this.connectionResolver = resolve;
    });

    try {
      console.log('🔌 Connecting to WebSocket:', this.config.url);
      this.ws = new WebSocket(this.config.url);
      this.setupEventHandlers();
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      this.scheduleReconnect();
    }

    return this.connectionPromise;
  }

  private setupEventHandlers() {
    if (!this.ws) return;

    this.ws.onopen = () => {
      console.log('✅ WebSocket connected');
      const { setWsConnected } = useAppStore.getState();
      setWsConnected(true);

      this.reconnectAttempts = 0;
      this.startHeartbeat();
      this.flushMessageQueue();
      this.resubscribeAll();

      // Resolve connection promise
      if (this.connectionResolver) {
        this.connectionResolver();
        this.connectionResolver = null;
        this.connectionPromise = null;
      }

      createNotification({
        id: `ws-connected-${Date.now()}`,
        type: 'success',
        title: 'Connected',
        message: 'Real-time signal updates active',
        timestamp: new Date(),
      });
    };

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);

      // Only show error notification if we've been connected before
      if (this.reconnectAttempts > 0) {
        createNotification({
          id: `ws-error-${Date.now()}`,
          type: 'warning',
          title: 'Connection Issue',
          message: 'Attempting to reconnect...',
          timestamp: new Date(),
        });
      }
    };

    this.ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      const { setWsConnected } = useAppStore.getState();
      setWsConnected(false);

      this.stopHeartbeat();

      // Clear connection promise
      this.connectionPromise = null;
      this.connectionResolver = null;

      if (!this.isIntentionallyClosed) {
        this.scheduleReconnect();
      }
    };
  }

  private handleMessage(message: WebSocketMessage) {
    const { addSignal, updateMarketData, setAgentStatus } = useAppStore.getState();

    switch (message.type) {
      case 'signal':
      case 'new_signal':
        if (message.data) {
          // Handle both signal formats
          const signalData = message.data.data || message.data;
          addSignal(signalData as Signal);

          // Show notification for high-confidence signals
          if (signalData.confidence >= 85) {
            createNotification({
              id: `signal-${Date.now()}`,
              type: 'info',
              title: 'New High-Confidence Signal',
              message: `${signalData.symbol} - ${signalData.type || signalData.signal_type} (${signalData.confidence}%)`,
              timestamp: new Date(),
            });
          }
        }
        break;

      case 'market_data':
      case 'market_update':
        if (message.data) {
          // Handle array of market updates
          if (Array.isArray(message.data)) {
            message.data.forEach((item: any) => {
              if (item.symbol) {
                updateMarketData(item.symbol, item as MarketData);
              }
            });
          } else if (message.data.symbol) {
            updateMarketData(message.data.symbol, message.data as MarketData);
          }
        }
        break;

      case 'agent_status':
        if (message.data) {
          setAgentStatus(message.data);
        }
        break;

      case 'system_alert':
        createNotification({
          id: `alert-${Date.now()}`,
          type: message.data.severity || 'info',
          title: message.data.title || 'System Alert',
          message: message.data.message,
          timestamp: new Date(message.timestamp),
        });
        break;

      case 'portfolio_update':
        // Handle portfolio updates
        const { updatePortfolio } = useAppStore.getState();
        if (message.data && updatePortfolio) {
          updatePortfolio(message.data);
        }
        break;

      case 'connection_established':
        console.log('Connection established with server');
        break;

      default:
        console.log('Unhandled message type:', message.type);
    }
  }

  private startHeartbeat() {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.sendMessage({
          type: 'heartbeat',
          data: { timestamp: new Date().toISOString() },
          timestamp: new Date().toISOString(),
        });
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private scheduleReconnect() {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      createNotification({
        id: `ws-max-reconnect-${Date.now()}`,
        type: 'error',
        title: 'Connection Failed',
        message: 'Unable to establish real-time connection. Please refresh the page.',
        timestamp: new Date(),
      });
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(
      this.config.reconnectInterval * Math.pow(1.5, this.reconnectAttempts - 1),
      30000
    );

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  private flushMessageQueue() {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message) {
        this.sendMessage(message);
      }
    }
  }

  private resubscribeAll() {
    this.subscriptions.forEach((subscription) => {
      const [type, ...params] = subscription.split(':');
      if (type === 'symbol') {
        this.subscribeToSymbol(params[0]);
      } else if (type === 'agent') {
        this.subscribeToAgent(params[0]);
      }
    });
  }

  // Public methods
  public async ensureConnected(): Promise<void> {
    if (this.isConnected()) {
      return Promise.resolve();
    }

    if (this.connectionPromise) {
      return this.connectionPromise;
    }

    return this.connect();
  }

  public sendMessage(message: WebSocketMessage) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(message));
      } catch (error) {
        console.error('Failed to send message:', error);
        this.messageQueue.push(message);
      }
    } else {
      // Queue message for later delivery
      this.messageQueue.push(message);
      console.warn('WebSocket not connected, message queued');

      // Try to reconnect if not already attempting
      if (this.reconnectAttempts === 0 && !this.isIntentionallyClosed) {
        this.connect();
      }
    }
  }

  public subscribeToSymbol(symbol: string) {
    const subscription = `symbol:${symbol}`;
    this.subscriptions.add(subscription);

    this.sendMessage({
      type: 'subscribe',
      data: { channel: 'market_data', symbol },
      timestamp: new Date().toISOString(),
    });

    this.sendMessage({
      type: 'subscribe',
      data: { channel: 'signals', symbol },
      timestamp: new Date().toISOString(),
    });
  }

  public unsubscribeFromSymbol(symbol: string) {
    const subscription = `symbol:${symbol}`;
    this.subscriptions.delete(subscription);

    this.sendMessage({
      type: 'unsubscribe',
      data: { channel: 'market_data', symbol },
      timestamp: new Date().toISOString(),
    });

    this.sendMessage({
      type: 'unsubscribe',
      data: { channel: 'signals', symbol },
      timestamp: new Date().toISOString(),
    });
  }

  public subscribeToAgent(agentName: string) {
    const subscription = `agent:${agentName}`;
    this.subscriptions.add(subscription);

    this.sendMessage({
      type: 'subscribe',
      data: { channel: 'agent_status', agent: agentName },
      timestamp: new Date().toISOString(),
    });
  }

  public unsubscribeFromAgent(agentName: string) {
    const subscription = `agent:${agentName}`;
    this.subscriptions.delete(subscription);

    this.sendMessage({
      type: 'unsubscribe',
      data: { channel: 'agent_status', agent: agentName },
      timestamp: new Date().toISOString(),
    });
  }

  public requestAgentStatus() {
    this.sendMessage({
      type: 'request',
      data: { action: 'agent_status' },
      timestamp: new Date().toISOString(),
    });
  }

  public requestMarketData(symbols: string[]) {
    this.sendMessage({
      type: 'request',
      data: { action: 'market_data', symbols },
      timestamp: new Date().toISOString(),
    });
  }

  public isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  public disconnect() {
    this.isIntentionallyClosed = true;

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    this.stopHeartbeat();

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }

    const { setWsConnected } = useAppStore.getState();
    setWsConnected(false);
  }

  public reconnect() {
    this.isIntentionallyClosed = false;
    this.disconnect();
    this.reconnectAttempts = 0;
    this.connect();
  }
}

// Create singleton instance
const webSocketService = new WebSocketService();

// React hook for WebSocket connection status
export const useWebSocketConnection = () => {
  const { wsConnected } = useAppStore();
  return wsConnected;
};

// React hook for WebSocket service
export const useWebSocket = () => {
  return {
    isConnected: webSocketService.isConnected(),
    sendMessage: webSocketService.sendMessage.bind(webSocketService),
    subscribeToSymbol: webSocketService.subscribeToSymbol.bind(webSocketService),
    unsubscribeFromSymbol: webSocketService.unsubscribeFromSymbol.bind(webSocketService),
    subscribeToAgent: webSocketService.subscribeToAgent.bind(webSocketService),
    unsubscribeFromAgent: webSocketService.unsubscribeFromAgent.bind(webSocketService),
    requestAgentStatus: webSocketService.requestAgentStatus.bind(webSocketService),
    requestMarketData: webSocketService.requestMarketData.bind(webSocketService),
    disconnect: webSocketService.disconnect.bind(webSocketService),
    reconnect: webSocketService.reconnect.bind(webSocketService),
    ensureConnected: webSocketService.ensureConnected.bind(webSocketService),
  };
};

export default webSocketService; 