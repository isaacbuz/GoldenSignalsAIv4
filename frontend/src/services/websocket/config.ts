/**
 * WebSocket Configuration
 */

export const WS_CONFIG = {
    // Base WebSocket URL
    url: process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws',

    // Reconnection settings
    reconnectInterval: 5000,
    maxReconnectAttempts: 10,

    // Heartbeat settings
    heartbeatInterval: 30000,

    // Message queue settings
    messageQueueSize: 100,
};

// Initialize WebSocket service on app start
import { getWebSocketService } from '../websocketService';
import { wsManager } from './SignalWebSocketManager';
import logger from '../logger';


export const initializeWebSocket = () => {
    // Initialize the robust WebSocket service
    try {
        const wsService = getWebSocketService({
            url: WS_CONFIG.url,
            ...WS_CONFIG
        });

        // Connect automatically
        wsService.connect().catch(error => {
            logger.error('Failed to connect WebSocket:', error);
        });
    } catch (error) {
        logger.error('Failed to initialize WebSocket service:', error);
    }

    // Also initialize the signal WebSocket manager
    wsManager.connect();
};
