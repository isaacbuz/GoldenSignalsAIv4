/**
 * Unified WebSocket Manager for AI Signal Platform
 * Handles all real-time communication with a single connection
 */

import { EventEmitter } from 'events';
import { useState, useEffect, useCallback } from 'react';
import { ENV } from '../../config/environment';

export enum WebSocketTopic {
  SIGNALS_LIVE = 'signals/live',
  AGENTS_STATUS = 'agents/status',
  CONSENSUS_UPDATES = 'consensus/updates',
  MODELS_PERFORMANCE = 'models/performance',
  ALERTS_USER = 'alerts/user',
}

export interface WebSocketMessage {
  type: 'signal' | 'agent' | 'consensus' | 'model' | 'alert' | 'market_data' | 'system' | 'heartbeat';
  action?: string;
  data?: any;
  metadata?: any;
  timestamp: string;
  source?: string;
  target?: string;
}

export interface SignalUpdate {
  signal_id: string;
  symbol: string;
  signal_type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  source: string;
  reasoning?: string;
  agent_breakdown?: Record<string, any>;
  timestamp: string;
  metadata?: any;
}

export interface AgentUpdate {
  agent_name: string;
  status: 'active' | 'inactive' | 'error';
  performance: {
    accuracy: number;
    total_signals: number;
    avg_confidence: number;
  };
  last_signal?: any;
  timestamp: string;
}

export interface MarketDataUpdate {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  timestamp: string;
  source: string;
}

export interface ConsensusUpdate {
  symbol: string;
  consensus_action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  agent_agreement: number;
  participating_agents: string[];
  top_agents: Array<{
    name: string;
    action: string;
    confidence: number;
  }>;
  timestamp: string;
}

export type MessageCallback = (message: WebSocketMessage) => void;
export type SignalCallback = (signal: SignalUpdate) => void;
export type AgentCallback = (agent: AgentUpdate) => void;
export type MarketDataCallback = (data: MarketDataUpdate) => void;
export type ConsensusCallback = (consensus: ConsensusUpdate) => void;

class SignalWebSocketManager {
  private static instance: SignalWebSocketManager;
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private connectionTimeout: NodeJS.Timeout | null = null;
  private isConnecting = false;
  private shouldReconnect = true;

  // Callback registries
  private messageCallbacks: Set<MessageCallback> = new Set();
  private signalCallbacks: Set<SignalCallback> = new Set();
  private agentCallbacks: Set<AgentCallback> = new Set();
  private marketDataCallbacks: Set<MarketDataCallback> = new Set();
  private consensusCallbacks: Set<ConsensusCallback> = new Set();

  // Subscription management
  private subscriptions: Set<string> = new Set();
  private subscribedSymbols: Set<string> = new Set();
  private subscribedAgents: Set<string> = new Set();

  // Connection state
  private connectionId: string | null = null;
  private lastHeartbeat: number = 0;
  private connectionStats = {
    messagesReceived: 0,
    messagesPerSecond: 0,
    lastMessageTime: 0,
    uptime: 0,
    reconnections: 0
  };

  // Add message queue property and processing
  private messageQueue: WebSocketMessage[] = [];

  private constructor() {
    this.connect();

    // Update connection stats every second
    setInterval(() => {
      this.updateConnectionStats();
    }, 1000);
  }

  static getInstance(): SignalWebSocketManager {
    if (!SignalWebSocketManager.instance) {
      SignalWebSocketManager.instance = new SignalWebSocketManager();
    }
    return SignalWebSocketManager.instance;
  }

  async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      return;
    }

    this.isConnecting = true;

    try {
      // Fix the URL construction - don't add /ws/signals if the URL already contains /ws
      const baseUrl = ENV.WEBSOCKET_URL.replace('http', 'ws');
      const wsUrl = baseUrl.endsWith('/ws') ? baseUrl + '/signals' : baseUrl + '/ws/signals';
      console.log('Connecting to WebSocket:', wsUrl);

      this.ws = new WebSocket(wsUrl);

      // Connection timeout
      this.connectionTimeout = setTimeout(() => {
        if (this.ws && this.ws.readyState !== WebSocket.OPEN) {
          console.warn('WebSocket connection timeout');
          this.ws.close();
          this.handleReconnect();
        }
      }, 10000);

      this.ws.onopen = (event) => {
        console.log('✅ WebSocket connected successfully');
        this.isConnecting = false;
        this.reconnectAttempts = 0;

        if (this.connectionTimeout) {
          clearTimeout(this.connectionTimeout);
          this.connectionTimeout = null;
        }

        // Send connection handshake
        this.sendMessage({
          type: 'system',
          action: 'handshake',
          data: {
            client_type: 'frontend',
            version: '1.0.0',
            capabilities: ['signals', 'agents', 'market_data', 'consensus']
          },
          timestamp: new Date().toISOString()
        });

        // Start heartbeat
        this.startHeartbeat();

        // Process any queued messages
        this.processQueuedMessages();

        // Resubscribe to previous subscriptions
        this.resubscribe();

        this.connectionStats.reconnections++;
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleMessage(message);

          // Update stats
          this.connectionStats.messagesReceived++;
          this.connectionStats.messagesPerSecond++;
          this.connectionStats.lastMessageTime = Date.now();

        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        this.isConnecting = false;
      };

      this.ws.onclose = (event) => {
        console.log('🔌 WebSocket disconnected:', event.code, event.reason);
        this.isConnecting = false;
        this.stopHeartbeat();

        if (this.connectionTimeout) {
          clearTimeout(this.connectionTimeout);
          this.connectionTimeout = null;
        }

        if (this.shouldReconnect && event.code !== 1000) {
          this.handleReconnect();
        }
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.isConnecting = false;
      this.handleReconnect();
    }
  }

  private handleMessage(message: WebSocketMessage) {
    // Handle special system messages
    if (message.type === 'system') {
      this.handleSystemMessage(message);
      return;
    }

    // Handle heartbeat responses
    if (message.type === 'heartbeat') {
      this.lastHeartbeat = Date.now();
      return;
    }

    // Dispatch to appropriate handlers
    switch (message.type) {
      case 'signal':
        this.handleSignalMessage(message);
        break;
      case 'agent':
        this.handleAgentMessage(message);
        break;
      case 'market_data':
        this.handleMarketDataMessage(message);
        break;
      case 'consensus':
        this.handleConsensusMessage(message);
        break;
    }

    // Notify general message callbacks
    this.messageCallbacks.forEach(callback => {
      try {
        callback(message);
      } catch (error) {
        console.error('Error in message callback:', error);
      }
    });
  }

  private handleSystemMessage(message: WebSocketMessage) {
    if (message.action === 'connection_established') {
      this.connectionId = message.data?.connection_id;
      console.log('✅ WebSocket connection established with ID:', this.connectionId);
    } else if (message.action === 'subscription_confirmed') {
      console.log('✅ Subscription confirmed:', message.data?.topic);
    } else if (message.action === 'error') {
      console.error('❌ WebSocket server error:', message.data?.error);
    }
  }

  private handleSignalMessage(message: WebSocketMessage) {
    if (message.action === 'create' || message.action === 'update') {
      const signalUpdate: SignalUpdate = {
        signal_id: message.data.signal_id || message.data.id,
        symbol: message.data.symbol,
        signal_type: message.data.signal_type || message.data.action,
        confidence: message.data.confidence,
        source: message.data.source || 'unknown',
        reasoning: message.data.reasoning,
        agent_breakdown: message.data.agent_breakdown,
        timestamp: message.timestamp,
        metadata: message.metadata
      };

      this.signalCallbacks.forEach(callback => {
        try {
          callback(signalUpdate);
        } catch (error) {
          console.error('Error in signal callback:', error);
        }
      });
    }
  }

  private handleAgentMessage(message: WebSocketMessage) {
    if (message.action === 'update' || message.action === 'status') {
      const agentUpdate: AgentUpdate = {
        agent_name: message.data.agent_name || message.data.name,
        status: message.data.status,
        performance: {
          accuracy: message.data.accuracy || message.data.performance?.accuracy || 0,
          total_signals: message.data.total_signals || message.data.performance?.total_signals || 0,
          avg_confidence: message.data.avg_confidence || message.data.performance?.avg_confidence || 0
        },
        last_signal: message.data.last_signal,
        timestamp: message.timestamp
      };

      this.agentCallbacks.forEach(callback => {
        try {
          callback(agentUpdate);
        } catch (error) {
          console.error('Error in agent callback:', error);
        }
      });
    }
  }

  private handleMarketDataMessage(message: WebSocketMessage) {
    const marketUpdate: MarketDataUpdate = {
      symbol: message.data.symbol,
      price: message.data.price || message.data.close,
      change: message.data.change || 0,
      change_percent: message.data.change_percent || message.data.changePercent || 0,
      volume: message.data.volume || 0,
      timestamp: message.timestamp,
      source: message.data.source || 'unknown'
    };

    this.marketDataCallbacks.forEach(callback => {
      try {
        callback(marketUpdate);
      } catch (error) {
        console.error('Error in market data callback:', error);
      }
    });
  }

  private handleConsensusMessage(message: WebSocketMessage) {
    const consensusUpdate: ConsensusUpdate = {
      symbol: message.data.symbol,
      consensus_action: message.data.consensus_action || message.data.action,
      confidence: message.data.confidence,
      agent_agreement: message.data.agent_agreement,
      participating_agents: message.data.participating_agents || [],
      top_agents: message.data.top_agents || [],
      timestamp: message.timestamp
    };

    this.consensusCallbacks.forEach(callback => {
      try {
        callback(consensusUpdate);
      } catch (error) {
        console.error('Error in consensus callback:', error);
      }
    });
  }

  private startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.sendMessage({
          type: 'heartbeat',
          action: 'ping',
          timestamp: new Date().toISOString()
        });
      }
    }, 30000); // Send heartbeat every 30 seconds
  }

  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private handleReconnect() {
    if (!this.shouldReconnect || this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('❌ Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);

    console.log(`🔄 Reconnecting WebSocket in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, delay);
  }

  private resubscribe() {
    // Resubscribe to all previous subscriptions
    this.subscriptions.forEach(topic => {
      this.sendMessage({
        type: 'system',
        action: 'subscribe',
        data: { topic },
        timestamp: new Date().toISOString()
      });
    });

    // Resubscribe to symbols
    this.subscribedSymbols.forEach(symbol => {
      this.subscribeToSymbol(symbol);
    });

    // Resubscribe to agents
    this.subscribedAgents.forEach(agent => {
      this.subscribeToAgent(agent);
    });
  }

  private sendMessage(message: WebSocketMessage) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      // Queue message if not connected instead of logging error
      if (!this.messageQueue) {
        this.messageQueue = [];
      }
      this.messageQueue.push(message);

      // Only log warning if we're not currently connecting
      if (!this.isConnecting) {
        console.warn('WebSocket not connected, message queued');
      }
    }
  }

  // Add message queue property and processing
  private processQueuedMessages() {
    if (this.messageQueue.length > 0 && this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.log(`Processing ${this.messageQueue.length} queued messages`);
      const messages = [...this.messageQueue];
      this.messageQueue = [];

      messages.forEach(message => {
        this.sendMessage(message);
      });
    }
  }

  // Public subscription methods
  subscribeToSignals(callback: SignalCallback) {
    this.signalCallbacks.add(callback);
    this.subscribe('signals.live');

    return () => {
      this.signalCallbacks.delete(callback);
    };
  }

  subscribeToAgents(callback: AgentCallback) {
    this.agentCallbacks.add(callback);
    this.subscribe('agents.status');

    return () => {
      this.agentCallbacks.delete(callback);
    };
  }

  subscribeToMarketData(callback: MarketDataCallback) {
    this.marketDataCallbacks.add(callback);
    this.subscribe('market_data.live');

    return () => {
      this.marketDataCallbacks.delete(callback);
    };
  }

  subscribeToConsensus(callback: ConsensusCallback) {
    this.consensusCallbacks.add(callback);
    this.subscribe('consensus.updates');

    return () => {
      this.consensusCallbacks.delete(callback);
    };
  }

  subscribeToSymbol(symbol: string) {
    this.subscribedSymbols.add(symbol);
    this.sendMessage({
      type: 'system',
      action: 'subscribe',
      data: { topic: `symbols.${symbol}` },
      timestamp: new Date().toISOString()
    });
  }

  unsubscribeFromSymbol(symbol: string) {
    this.subscribedSymbols.delete(symbol);
    this.sendMessage({
      type: 'system',
      action: 'unsubscribe',
      data: { topic: `symbols.${symbol}` },
      timestamp: new Date().toISOString()
    });
  }

  subscribeToAgent(agentName: string) {
    this.subscribedAgents.add(agentName);
    this.sendMessage({
      type: 'system',
      action: 'subscribe',
      data: { topic: `agents.${agentName}` },
      timestamp: new Date().toISOString()
    });
  }

  unsubscribeFromAgent(agentName: string) {
    this.subscribedAgents.delete(agentName);
    this.sendMessage({
      type: 'system',
      action: 'unsubscribe',
      data: { topic: `agents.${agentName}` },
      timestamp: new Date().toISOString()
    });
  }

  private subscribe(topic: string, subscriberId?: string) {
    this.subscriptions.add(topic);
    this.sendMessage({
      type: 'system',
      action: 'subscribe',
      data: { topic, subscriberId },
      timestamp: new Date().toISOString()
    });
  }

  private unsubscribe(topic: string, subscriberId?: string) {
    this.subscriptions.delete(topic);
    this.sendMessage({
      type: 'system',
      action: 'unsubscribe',
      data: { topic, subscriberId },
      timestamp: new Date().toISOString()
    });
  }

  // General message subscription
  onMessage(callback: MessageCallback) {
    this.messageCallbacks.add(callback);

    return () => {
      this.messageCallbacks.delete(callback);
    };
  }

  // Connection management
  disconnect() {
    this.shouldReconnect = false;
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
    }
    this.stopHeartbeat();
  }

  reconnect() {
    this.disconnect();
    this.shouldReconnect = true;
    this.reconnectAttempts = 0;
    setTimeout(() => this.connect(), 1000);
  }

  // Status methods
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  getConnectionStatus(): boolean {
    return this.isConnected();
  }

  getConnectionStats() {
    return {
      ...this.connectionStats,
      isConnected: this.isConnected(),
      connectionId: this.connectionId,
      subscriptions: Array.from(this.subscriptions),
      subscribedSymbols: Array.from(this.subscribedSymbols),
      subscribedAgents: Array.from(this.subscribedAgents),
      lastHeartbeat: this.lastHeartbeat
    };
  }

  // Send custom messages (for testing or special use cases)
  send(message: Partial<WebSocketMessage>) {
    const fullMessage: WebSocketMessage = {
      type: 'system',
      timestamp: new Date().toISOString(),
      ...message
    };
    this.sendMessage(fullMessage);
  }

  private updateConnectionStats() {
    const now = Date.now();
    if (this.connectionStats.lastMessageTime > 0) {
      this.connectionStats.uptime = now - this.connectionStats.lastMessageTime;
    }

    // Reset messages per second counter
    this.connectionStats.messagesPerSecond = 0;
  }

  // Event listener methods
  on(topic: string, handler: (data: any, metadata?: any) => void) {
    // For now, add to general message callbacks
    // In a full implementation, you'd maintain topic-specific callbacks
    this.messageCallbacks.add((message) => {
      if (message.type === topic || message.action === topic) {
        handler(message.data, message.metadata);
      }
    });
  }

  off(topic: string, handler: (data: any, metadata?: any) => void) {
    // Remove handler - simplified implementation
    // In a full implementation, you'd track handlers by topic
  }
}

// Export singleton instance
export const wsManager = SignalWebSocketManager.getInstance();
export const signalWebSocketManager = wsManager; // Alias for backward compatibility

// React Hook for WebSocket
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

export const useAgentStatus = (handler: (status: AgentUpdate) => void, componentId?: string) => {
  return useWebSocket(WebSocketTopic.AGENTS_STATUS, handler, componentId);
};

export const useConsensusUpdates = (handler: (consensus: ConsensusUpdate) => void, componentId?: string) => {
  return useWebSocket(WebSocketTopic.CONSENSUS_UPDATES, handler, componentId);
};

// Hook for general WebSocket connection status
export const useWebSocketConnection = () => {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Connect if not connected
    if (!wsManager.getConnectionStatus()) {
      wsManager.connect();
    }

    const handleConnect = () => setIsConnected(true);
    const handleDisconnect = () => setIsConnected(false);

    wsManager.on('connected', handleConnect);
    wsManager.on('disconnected', handleDisconnect);

    // Set initial state
    setIsConnected(wsManager.getConnectionStatus());

    return () => {
      wsManager.off('connected', handleConnect);
      wsManager.off('disconnected', handleDisconnect);
    };
  }, []);

  return isConnected;
};

export default wsManager;

