/**
 * Enhanced WebSocket Service for GoldenSignalsAI V3 - Simplified Version
 * 
 * Temporarily disabled to prevent connection errors
 */

import { useAppStore, createNotification } from '../store';
import { Signal, MarketData } from './api';

export interface WebSocketMessage {
  type: 'signal' | 'market_data' | 'agent_status' | 'system_alert' | 'portfolio_update';
  data: any;
  timestamp: string;
}

export interface AgentStatus {
  agent_name: string;
  status: 'active' | 'inactive' | 'error';
  last_signal: string;
  performance: {
    accuracy: number;
    total_signals: number;
    success_rate: number;
  };
}

class WebSocketService {
  private connected = false;

  constructor() {
    // Simulate connection for now
    setTimeout(() => {
      this.connected = true;
      const { setWsConnected } = useAppStore.getState();
      setWsConnected(true);
      console.log('WebSocket service initialized (mock mode)');
    }, 1000);
  }

  // Public methods (mock implementations)
  public sendMessage(message: WebSocketMessage) {
    console.log('Mock WebSocket message:', message);
  }

  public subscribeToSymbol(symbol: string) {
    console.log('Mock subscribe to symbol:', symbol);
  }

  public unsubscribeFromSymbol(symbol: string) {
    console.log('Mock unsubscribe from symbol:', symbol);
  }

  public subscribeToAgent(agentName: string) {
    console.log('Mock subscribe to agent:', agentName);
  }

  public unsubscribeFromAgent(agentName: string) {
    console.log('Mock unsubscribe from agent:', agentName);
  }

  public requestAgentStatus() {
    console.log('Mock request agent status');
  }

  public requestMarketData(symbols: string[]) {
    console.log('Mock request market data:', symbols);
  }

  public isConnected(): boolean {
    return this.connected;
  }

  public disconnect() {
    this.connected = false;
    const { setWsConnected } = useAppStore.getState();
    setWsConnected(false);
  }

  public reconnect() {
    this.connected = true;
    const { setWsConnected } = useAppStore.getState();
    setWsConnected(true);
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
  };
};

export default webSocketService; 