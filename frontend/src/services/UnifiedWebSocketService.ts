/**
 * Unified WebSocket Service
 * A single, consolidated WebSocket implementation for the entire frontend
 * Replaces all previous WebSocket implementations with a robust, feature-complete service
 */

import logger from './logger';

// Types
export type WebSocketMessageType =
  | 'market_data'
  | 'signal_update'
  | 'agent_signal'
  | 'workflow_update'
  | 'alert'
  | 'error'
  | 'heartbeat'
  | 'subscribe'
  | 'unsubscribe';

export interface WebSocketMessage<T = any> {
  type: WebSocketMessageType;
  channel?: string;
  data: T;
  timestamp?: number;
  id?: string;
}

export interface WebSocketConfig {
  url: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  messageTimeout?: number;
  debug?: boolean;
}

export type MessageHandler<T = any> = (message: WebSocketMessage<T>) => void;
export type ConnectionHandler = (connected: boolean) => void;

interface Subscription {
  channel: string;
  type: WebSocketMessageType;
  handler: MessageHandler;
}

/**
 * Unified WebSocket Service
 * Features:
 * - Single connection for all WebSocket needs
 * - Automatic reconnection with exponential backoff
 * - Message queuing during disconnection
 * - Channel-based subscriptions
 * - Type-safe message handling
 * - Heartbeat mechanism
 * - Connection state management
 * - Memory leak prevention
 */
export class UnifiedWebSocketService {
  private static instance: UnifiedWebSocketService | null = null;

  private ws: WebSocket | null = null;
  private config: Required<WebSocketConfig>;
  private subscriptions: Map<string, Subscription> = new Map();
  private messageQueue: WebSocketMessage[] = [];
  private connectionHandlers: Set<ConnectionHandler> = new Set();
  private reconnectAttempts = 0;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private isConnected = false;
  private isConnecting = false;
  private shouldReconnect = true;
  private lastPingTime = 0;
  private messageCallbacks: Map<string, (response: any) => void> = new Map();

  private constructor(config: WebSocketConfig) {
    this.config = {
      reconnectInterval: 1000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      messageTimeout: 10000,
      debug: false,
      ...config
    };
  }

  /**
   * Get singleton instance
   */
  static getInstance(config?: WebSocketConfig): UnifiedWebSocketService {
    if (!UnifiedWebSocketService.instance) {
      if (!config) {
        throw new Error('WebSocket config required for first initialization');
      }
      UnifiedWebSocketService.instance = new UnifiedWebSocketService(config);
    }
    return UnifiedWebSocketService.instance;
  }

  /**
   * Connect to WebSocket server
   */
  connect(): void {
    if (this.isConnecting || this.isConnected) {
      return;
    }

    this.isConnecting = true;
    this.shouldReconnect = true;

    try {
      this.ws = new WebSocket(this.config.url);

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onerror = this.handleError.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
    } catch (error) {
      logger.error('Failed to create WebSocket:', error);
      this.isConnecting = false;
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.shouldReconnect = false;
    this.cleanup();

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
    }
  }

  /**
   * Subscribe to a channel with a specific message type
   */
  subscribe<T = any>(
    channel: string,
    type: WebSocketMessageType,
    handler: MessageHandler<T>
  ): () => void {
    const id = `${channel}:${type}:${Math.random()}`;
    const subscription: Subscription = { channel, type, handler };

    this.subscriptions.set(id, subscription);

    // Send subscription message if connected
    if (this.isConnected) {
      this.send({
        type: 'subscribe',
        channel,
        data: { types: [type] }
      });
    }

    // Return unsubscribe function
    return () => {
      this.subscriptions.delete(id);

      // Check if this was the last subscription for this channel
      const hasMoreSubscriptions = Array.from(this.subscriptions.values())
        .some(sub => sub.channel === channel);

      if (!hasMoreSubscriptions && this.isConnected) {
        this.send({
          type: 'unsubscribe',
          channel,
          data: {}
        });
      }
    };
  }

  /**
   * Send a message with optional response callback
   */
  send<T = any>(message: WebSocketMessage, callback?: (response: T) => void): void {
    // Add message ID if callback provided
    if (callback) {
      message.id = `msg_${Date.now()}_${Math.random()}`;
      this.messageCallbacks.set(message.id, callback);

      // Set timeout for callback
      setTimeout(() => {
        if (this.messageCallbacks.has(message.id!)) {
          this.messageCallbacks.delete(message.id!);
          callback(null as any);
        }
      }, this.config.messageTimeout);
    }

    // Add timestamp
    message.timestamp = Date.now();

    if (this.isConnected && this.ws?.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(message));
        if (this.config.debug) {
          logger.debug('Sent WebSocket message:', message);
        }
      } catch (error) {
        logger.error('Failed to send WebSocket message:', error);
        this.queueMessage(message);
      }
    } else {
      this.queueMessage(message);
    }
  }

  /**
   * Add connection state change handler
   */
  onConnectionChange(handler: ConnectionHandler): () => void {
    this.connectionHandlers.add(handler);

    // Immediately call with current state
    handler(this.isConnected);

    // Return cleanup function
    return () => {
      this.connectionHandlers.delete(handler);
    };
  }

  /**
   * Get current connection state
   */
  get connected(): boolean {
    return this.isConnected;
  }

  // Private methods

  private handleOpen(): void {
    logger.info('WebSocket connected');
    this.isConnecting = false;
    this.isConnected = true;
    this.reconnectAttempts = 0;

    // Clear reconnect timeout
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    // Start heartbeat
    this.startHeartbeat();

    // Resubscribe to all channels
    const channels = new Set<string>();
    this.subscriptions.forEach(sub => channels.add(sub.channel));

    channels.forEach(channel => {
      const types = Array.from(this.subscriptions.values())
        .filter(sub => sub.channel === channel)
        .map(sub => sub.type);

      this.send({
        type: 'subscribe',
        channel,
        data: { types: [...new Set(types)] }
      });
    });

    // Process queued messages
    this.processMessageQueue();

    // Notify handlers
    this.notifyConnectionHandlers(true);
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);

      if (this.config.debug) {
        logger.debug('Received WebSocket message:', message);
      }

      // Handle heartbeat
      if (message.type === 'heartbeat') {
        this.lastPingTime = Date.now();
        return;
      }

      // Handle response callbacks
      if (message.id && this.messageCallbacks.has(message.id)) {
        const callback = this.messageCallbacks.get(message.id)!;
        this.messageCallbacks.delete(message.id);
        callback(message.data);
        return;
      }

      // Route to subscriptions
      this.subscriptions.forEach(subscription => {
        if (
          subscription.channel === message.channel &&
          subscription.type === message.type
        ) {
          try {
            subscription.handler(message);
          } catch (error) {
            logger.error('Error in subscription handler:', error);
          }
        }
      });
    } catch (error) {
      logger.error('Failed to parse WebSocket message:', error);
    }
  }

  private handleError(event: Event): void {
    logger.error('WebSocket error:', event);
    this.isConnecting = false;
  }

  private handleClose(event: CloseEvent): void {
    logger.info('WebSocket disconnected', {
      code: event.code,
      reason: event.reason
    });

    this.isConnecting = false;
    this.isConnected = false;

    // Stop heartbeat
    this.stopHeartbeat();

    // Notify handlers
    this.notifyConnectionHandlers(false);

    // Reconnect if not deliberate disconnect
    if (this.shouldReconnect && event.code !== 1000) {
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      logger.error('Max reconnection attempts reached');
      this.cleanup();
      return;
    }

    this.reconnectAttempts++;

    // Exponential backoff with jitter
    const baseDelay = this.config.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1);
    const jitter = Math.random() * 1000;
    const delay = Math.min(baseDelay + jitter, 30000);

    logger.info(`Reconnecting in ${Math.round(delay)}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectTimeout = null;
      this.connect();
    }, delay);
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();

    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected && this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'heartbeat', data: {} });

        // Check for missed heartbeats
        if (this.lastPingTime && Date.now() - this.lastPingTime > this.config.heartbeatInterval * 2) {
          logger.warn('Missed heartbeat, reconnecting...');
          this.ws.close();
        }
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private queueMessage(message: WebSocketMessage): void {
    // Limit queue size to prevent memory issues
    if (this.messageQueue.length >= 100) {
      this.messageQueue.shift();
    }

    this.messageQueue.push(message);
  }

  private processMessageQueue(): void {
    const queue = [...this.messageQueue];
    this.messageQueue = [];

    queue.forEach(message => {
      this.send(message);
    });
  }

  private notifyConnectionHandlers(connected: boolean): void {
    this.connectionHandlers.forEach(handler => {
      try {
        handler(connected);
      } catch (error) {
        logger.error('Error in connection handler:', error);
      }
    });
  }

  private cleanup(): void {
    // Clear timeouts
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    // Stop heartbeat
    this.stopHeartbeat();

    // Clear callbacks
    this.messageCallbacks.clear();

    // Clear message queue
    this.messageQueue = [];

    // Remove WebSocket handlers
    if (this.ws) {
      this.ws.onopen = null;
      this.ws.onmessage = null;
      this.ws.onerror = null;
      this.ws.onclose = null;
    }
  }
}

// Export singleton getter
export const getUnifiedWebSocket = (config?: WebSocketConfig) =>
  UnifiedWebSocketService.getInstance(config);
