/**
 * Consolidated WebSocket Adapters
 * Provides backward compatibility for existing code while using the unified WebSocket service
 */

import { getUnifiedWebSocket, WebSocketMessage } from '../UnifiedWebSocketService';
import logger from '../logger';

// Initialize the unified WebSocket
const wsUrl = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000';
const unifiedWS = getUnifiedWebSocket({
  url: `${wsUrl}/ws/v2/connect`,
  debug: import.meta.env.DEV
});

// Auto-connect on initialization
unifiedWS.connect();

/**
 * Market Data WebSocket Adapter
 * Provides compatibility for market data subscriptions
 */
export const marketDataWebSocket = {
  subscribe(symbol: string, callback: (data: any) => void): () => void {
    return unifiedWS.subscribe(
      `market:${symbol}`,
      'market_data',
      (message: WebSocketMessage) => {
        // Transform to expected format
        callback({
          type: 'price',
          symbol: message.data.symbol || symbol,
          price: message.data.price,
          volume: message.data.volume,
          timestamp: message.data.timestamp || Date.now(),
          bid: message.data.bid,
          ask: message.data.ask
        });
      }
    );
  },

  connect(): void {
    // Already connected via unified service
    logger.debug('Market data WebSocket adapter: using unified connection');
  },

  disconnect(): void {
    // Don't disconnect the shared connection
    logger.debug('Market data WebSocket adapter: keeping unified connection');
  },

  isConnected(): boolean {
    return unifiedWS.connected;
  }
};

/**
 * Signal WebSocket Adapter
 * Provides compatibility for signal subscriptions
 */
export const signalWebSocket = {
  subscribeToSignals(callback: (signal: any) => void): () => void {
    return unifiedWS.subscribe(
      'signals',
      'signal_update',
      (message: WebSocketMessage) => {
        callback(message.data);
      }
    );
  },

  subscribeToSymbol(symbol: string, callback: (signal: any) => void): () => void {
    return unifiedWS.subscribe(
      `signals:${symbol}`,
      'signal_update',
      (message: WebSocketMessage) => {
        if (message.data.symbol === symbol) {
          callback(message.data);
        }
      }
    );
  },

  connect(): void {
    logger.debug('Signal WebSocket adapter: using unified connection');
  },

  disconnect(): void {
    logger.debug('Signal WebSocket adapter: keeping unified connection');
  }
};

/**
 * Agent WebSocket Adapter
 * Provides compatibility for agent signal subscriptions
 */
export const agentWebSocket = {
  subscribeToAgentSignals(symbol: string, callback: (data: any) => void): () => void {
    return unifiedWS.subscribe(
      `agents:${symbol}`,
      'agent_signal',
      (message: WebSocketMessage) => {
        callback(message.data);
      }
    );
  },

  subscribeToWorkflow(symbol: string, callback: (data: any) => void): () => void {
    return unifiedWS.subscribe(
      `workflow:${symbol}`,
      'workflow_update',
      (message: WebSocketMessage) => {
        callback(message.data);
      }
    );
  },

  getConnectionStatus(): boolean {
    return unifiedWS.connected;
  }
};

/**
 * Alert WebSocket Adapter
 * Provides compatibility for alert subscriptions
 */
export const alertWebSocket = {
  subscribeToAlerts(callback: (alert: any) => void): () => void {
    return unifiedWS.subscribe(
      'alerts',
      'alert',
      (message: WebSocketMessage) => {
        callback(message.data);
      }
    );
  },

  subscribeToUserAlerts(userId: string, callback: (alert: any) => void): () => void {
    return unifiedWS.subscribe(
      `alerts:user:${userId}`,
      'alert',
      (message: WebSocketMessage) => {
        callback(message.data);
      }
    );
  }
};

/**
 * Connection status monitoring
 */
export const connectionMonitor = {
  onConnectionChange(callback: (connected: boolean) => void): () => void {
    return unifiedWS.onConnectionChange(callback);
  },

  isConnected(): boolean {
    return unifiedWS.connected;
  },

  reconnect(): void {
    unifiedWS.disconnect();
    setTimeout(() => unifiedWS.connect(), 100);
  }
};

/**
 * Generic WebSocket operations
 */
export const websocketOperations = {
  send(message: any): void {
    unifiedWS.send({
      type: message.type || 'message',
      channel: message.channel || 'default',
      data: message
    });
  },

  request<T = any>(message: any): Promise<T> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('WebSocket request timeout'));
      }, 10000);

      unifiedWS.send(
        {
          type: message.type || 'request',
          channel: message.channel || 'default',
          data: message
        },
        (response: T) => {
          clearTimeout(timeout);
          if (response === null) {
            reject(new Error('WebSocket request failed'));
          } else {
            resolve(response);
          }
        }
      );
    });
  }
};

// Export the unified instance for direct access if needed
export { unifiedWS };

// Auto-reconnect on window focus
if (typeof window !== 'undefined') {
  window.addEventListener('focus', () => {
    if (!unifiedWS.connected) {
      logger.info('Window focused, reconnecting WebSocket...');
      unifiedWS.connect();
    }
  });

  // Disconnect on page unload
  window.addEventListener('beforeunload', () => {
    unifiedWS.disconnect();
  });
}
