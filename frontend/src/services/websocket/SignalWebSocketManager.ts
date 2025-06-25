/**
 * Unified WebSocket Manager for AI Signal Platform
 * Handles all real-time communication with a single connection
 */

import { EventEmitter } from 'events';

export enum WebSocketTopic {
  SIGNALS_LIVE = 'signals/live',
  AGENTS_STATUS = 'agents/status',
  CONSENSUS_UPDATES = 'consensus/updates',
  MODELS_PERFORMANCE = 'models/performance',
  ALERTS_USER = 'alerts/user',
}

export interface WebSocketMessage {
  type: 'signal' | 'agent' | 'consensus' | 'model' | 'alert';
  action: 'update' | 'create' | 'delete';
  timestamp: string;
  data: any;
  metadata?: {
    confidence?: number;
    agents?: string[];
    processing_time_ms?: number;
  };
}

export interface SignalUpdate {
  id: string;
  symbol: string;
  type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  agents: string[];
  reasoning: string;
  timestamp: string;
}

export interface AgentStatusUpdate {
  agentId: string;
  status: 'active' | 'idle' | 'processing' | 'error';
  accuracy: number;
  lastSignal: string;
}

export interface ConsensusUpdate {
  consensusId: string;
  signal: string;
  confidence: number;
  agentsAgreeing: number;
  totalAgents: number;
  reasoning: string;
}

class SignalWebSocketManager extends EventEmitter {
  private static instance: SignalWebSocketManager;
  private connection: WebSocket | null = null;
  private subscriptions: Map<string, Set<string>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private messageQueue: WebSocketMessage[] = [];
  private isConnected = false;

  private constructor() {
    super();
    this.setMaxListeners(100); // Increase for multiple component subscriptions
  }

  static getInstance(): SignalWebSocketManager {
    if (!SignalWebSocketManager.instance) {
      SignalWebSocketManager.instance = new SignalWebSocketManager();
    }
    return SignalWebSocketManager.instance;
  }

  connect(url?: string): void {
    if (this.connection?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }

    const wsUrl = url || process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws/signals';
    
    try {
      this.connection = new WebSocket(wsUrl);
      this.setupEventHandlers();
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.scheduleReconnect();
    }
  }

  private setupEventHandlers(): void {
    if (!this.connection) return;

    this.connection.onopen = () => {
      console.log('WebSocket connected');
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.emit('connected');
      
      // Send any queued messages
      this.flushMessageQueue();
      
      // Start heartbeat
      this.startHeartbeat();
      
      // Resubscribe to all topics
      this.resubscribeAll();
    };

    this.connection.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.connection.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.emit('error', error);
    };

    this.connection.onclose = () => {
      console.log('WebSocket disconnected');
      this.isConnected = false;
      this.stopHeartbeat();
      this.emit('disconnected');
      this.scheduleReconnect();
    };
  }

  private handleMessage(message: WebSocketMessage): void {
    // Emit to specific topic listeners
    const topic = `${message.type}/${message.action}`;
    this.emit(topic, message.data, message.metadata);

    // Emit to general type listeners
    this.emit(message.type, message);

    // Handle specific message types
    switch (message.type) {
      case 'signal':
        this.handleSignalMessage(message);
        break;
      case 'agent':
        this.handleAgentMessage(message);
        break;
      case 'consensus':
        this.handleConsensusMessage(message);
        break;
      case 'model':
        this.handleModelMessage(message);
        break;
      case 'alert':
        this.handleAlertMessage(message);
        break;
    }
  }

  private handleSignalMessage(message: WebSocketMessage): void {
    const signal = message.data as SignalUpdate;
    this.emit(WebSocketTopic.SIGNALS_LIVE, signal);
  }

  private handleAgentMessage(message: WebSocketMessage): void {
    const agentUpdate = message.data as AgentStatusUpdate;
    this.emit(WebSocketTopic.AGENTS_STATUS, agentUpdate);
  }

  private handleConsensusMessage(message: WebSocketMessage): void {
    const consensus = message.data as ConsensusUpdate;
    this.emit(WebSocketTopic.CONSENSUS_UPDATES, consensus);
  }

  private handleModelMessage(message: WebSocketMessage): void {
    this.emit(WebSocketTopic.MODELS_PERFORMANCE, message.data);
  }

  private handleAlertMessage(message: WebSocketMessage): void {
    this.emit(WebSocketTopic.ALERTS_USER, message.data);
  }

  subscribe(topic: WebSocketTopic, subscriberId: string): void {
    if (!this.subscriptions.has(topic)) {
      this.subscriptions.set(topic, new Set());
    }
    
    this.subscriptions.get(topic)!.add(subscriberId);
    
    // Send subscription message to server
    this.send({
      type: 'subscribe',
      topic,
      subscriberId,
    });
  }

  unsubscribe(topic: WebSocketTopic, subscriberId: string): void {
    const subscribers = this.subscriptions.get(topic);
    if (subscribers) {
      subscribers.delete(subscriberId);
      if (subscribers.size === 0) {
        this.subscriptions.delete(topic);
      }
    }

    // Send unsubscribe message to server
    this.send({
      type: 'unsubscribe',
      topic,
      subscriberId,
    });
  }

  send(data: any): void {
    if (this.connection?.readyState === WebSocket.OPEN) {
      this.connection.send(JSON.stringify(data));
    } else {
      // Queue message for later
      this.messageQueue.push(data);
    }
  }

  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message) {
        this.send(message);
      }
    }
  }

  private resubscribeAll(): void {
    this.subscriptions.forEach((subscribers, topic) => {
      subscribers.forEach((subscriberId) => {
        this.send({
          type: 'subscribe',
          topic,
          subscriberId,
        });
      });
    });
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.connection?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' });
      }
    }, 30000); // 30 seconds
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('reconnect_failed');
      return;
    }

    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
    console.log(`Reconnecting in ${delay}ms...`);
    
    setTimeout(() => {
      this.reconnectAttempts++;
      this.connect();
    }, delay);
  }

  disconnect(): void {
    this.stopHeartbeat();
    if (this.connection) {
      this.connection.close();
      this.connection = null;
    }
    this.isConnected = false;
    this.subscriptions.clear();
    this.messageQueue = [];
  }

  getConnectionStatus(): boolean {
    return this.isConnected;
  }

  getSubscriptionCount(): number {
    let count = 0;
    this.subscriptions.forEach((subscribers) => {
      count += subscribers.size;
    });
    return count;
  }
}

// Export singleton instance
export const wsManager = SignalWebSocketManager.getInstance();

// React Hook for WebSocket
import { useEffect, useCallback } from 'react';

export const useWebSocket = (
  topic: WebSocketTopic,
  handler: (data: any, metadata?: any) => void,
  componentId?: string
) => {
  const subscriberId = componentId || `component-${Math.random().toString(36).substr(2, 9)}`;

  useEffect(() => {
    // Connect if not connected
    if (!wsManager.getConnectionStatus()) {
      wsManager.connect();
    }

    // Subscribe to topic
    wsManager.subscribe(topic, subscriberId);
    
    // Add event listener
    const eventHandler = (data: any, metadata?: any) => {
      handler(data, metadata);
    };
    
    wsManager.on(topic, eventHandler);

    // Cleanup
    return () => {
      wsManager.off(topic, eventHandler);
      wsManager.unsubscribe(topic, subscriberId);
    };
  }, [topic, subscriberId]);

  const sendMessage = useCallback((data: any) => {
    wsManager.send(data);
  }, []);

  return {
    sendMessage,
    isConnected: wsManager.getConnectionStatus(),
  };
};

// Specific hooks for different data types
export const useSignalUpdates = (handler: (signal: SignalUpdate) => void, componentId?: string) => {
  return useWebSocket(WebSocketTopic.SIGNALS_LIVE, handler, componentId);
};

export const useAgentStatus = (handler: (status: AgentStatusUpdate) => void, componentId?: string) => {
  return useWebSocket(WebSocketTopic.AGENTS_STATUS, handler, componentId);
};

export const useConsensusUpdates = (handler: (consensus: ConsensusUpdate) => void, componentId?: string) => {
  return useWebSocket(WebSocketTopic.CONSENSUS_UPDATES, handler, componentId);
};

export default wsManager;
