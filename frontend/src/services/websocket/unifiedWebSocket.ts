/**
 * Unified WebSocket Service
 * This consolidates all WebSocket functionality into a single, managed service
 * Deprecates: websocketService.ts and stableWebSocket.ts
 */

import { signalWebSocketManager } from './config';

// Re-export the SignalWebSocketManager as the primary WebSocket service
export const unifiedWebSocketService = signalWebSocketManager;

// Helper function to migrate from old WebSocket services
export function migrateWebSocketConnection(oldServiceName: string) {
  console.warn(`[WebSocket Migration] ${oldServiceName} is deprecated. Using unified SignalWebSocketManager.`);
  return signalWebSocketManager;
}

// Backward compatibility exports (will log deprecation warnings)
export const websocketService = new Proxy({}, {
  get(target, prop) {
    console.warn('[Deprecation] websocketService is deprecated. Use unifiedWebSocketService instead.');
    return signalWebSocketManager[prop as keyof typeof signalWebSocketManager];
  }
});

export const stableWebSocket = new Proxy({}, {
  get(target, prop) {
    console.warn('[Deprecation] stableWebSocket is deprecated. Use unifiedWebSocketService instead.');
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