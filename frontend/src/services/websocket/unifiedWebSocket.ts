/**
 * Unified WebSocket Service
 * This consolidates all WebSocket functionality into a single, managed service
 * Deprecates: websocketService.ts and stableWebSocket.ts
 */

import { signalWebSocketManager } from './config';
import logger from '../logger';

// Also import our new consolidated adapters
import {
  marketDataWebSocket,
  signalWebSocket,
  agentWebSocket,
  alertWebSocket,
  connectionMonitor,
  websocketOperations,
  unifiedWS
} from './consolidatedWebSocket';


// Re-export the SignalWebSocketManager as the primary WebSocket service
export const unifiedWebSocketService = signalWebSocketManager;

// Helper function to migrate from old WebSocket services
export function migrateWebSocketConnection(oldServiceName: string) {
  logger.warn(`[WebSocket Migration] ${oldServiceName} is deprecated. Using unified SignalWebSocketManager.`);
  return signalWebSocketManager;
}

// Backward compatibility exports (will log deprecation warnings)
export const websocketService = new Proxy({}, {
  get(target, prop) {
    logger.warn('[Deprecation] websocketService is deprecated. Use unifiedWebSocketService instead.');
    return signalWebSocketManager[prop as keyof typeof signalWebSocketManager];
  }
});

export const stableWebSocket = new Proxy({}, {
  get(target, prop) {
    logger.warn('[Deprecation] stableWebSocket is deprecated. Use unifiedWebSocketService instead.');
    return signalWebSocketManager[prop as keyof typeof signalWebSocketManager];
  }
});

// Export all hooks from the unified service
export {
  useWebSocket,
  useWebSocketConnection,
  useSignalUpdates,
  useAgentStatus,
  useConsensusUpdates,
  useMarketData,
  useAlerts
} from './SignalWebSocketManager';

// Export types
export type {
  SignalData,
  AgentUpdate,
  ConsensusData,
  MarketData,
  AlertData,
  ConnectionStats,
  WebSocketTopic
} from './SignalWebSocketManager';

// Export the new consolidated adapters
export {
  marketDataWebSocket,
  signalWebSocket,
  agentWebSocket,
  alertWebSocket,
  connectionMonitor,
  websocketOperations,
  unifiedWS
};

// Helper to get the appropriate adapter
export function getWebSocketAdapter(type: 'market' | 'signal' | 'agent' | 'alert') {
  switch (type) {
    case 'market':
      return marketDataWebSocket;
    case 'signal':
      return signalWebSocket;
    case 'agent':
      return agentWebSocket;
    case 'alert':
      return alertWebSocket;
    default:
      throw new Error(`Unknown WebSocket adapter type: ${type}`);
  }
}
