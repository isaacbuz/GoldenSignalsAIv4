// Unified WebSocket exports - SignalWebSocketManager is the single source of truth
export * from './unifiedWebSocket';

// Direct exports from SignalWebSocketManager for convenience
export {
    wsManager,
    WebSocketTopic,
    type WebSocketMessage,
    type SignalUpdate,
    type AgentStatusUpdate,
    type ConsensusUpdate
} from './SignalWebSocketManager';

// Export the unified service as the default
export { unifiedWebSocketService as default } from './unifiedWebSocket';
