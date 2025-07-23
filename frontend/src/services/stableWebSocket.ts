/**
 * Stable WebSocket Service
 *
 * Single, reliable WebSocket connection for the entire app.
 * Handles reconnection, heartbeat, and message routing automatically.
 */

import { useAppStore } from '../store/appStore';
import logger from './logger';


interface WebSocketMessage {
    type: string;
    data: any;
    timestamp?: number;
}

class StableWebSocket {
    private ws: WebSocket | null = null;
    private reconnectDelay = 1000;
    private maxReconnectDelay = 30000;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 10;
    private heartbeatInterval: NodeJS.Timeout | null = null;
    private reconnectTimeout: NodeJS.Timeout | null = null;
    private isIntentionallyClosed = false;
    private messageQueue: WebSocketMessage[] = [];

    // Event handlers
    private messageHandlers = new Map<string, ((data: any) => void)[]>();

    constructor(private url: string = 'ws://localhost:8000/ws') {
        this.url = url;
    }

    // === PUBLIC METHODS ===

    async connect(): Promise<void> {
        if (this.ws?.readyState === WebSocket.OPEN) {
            return;
        }

        if (this.ws?.readyState === WebSocket.CONNECTING) {
            return;
        }

        this.isIntentionallyClosed = false;
        useAppStore.getState().setWSReconnecting(true);

        try {
            logger.info('🔌 Connecting to WebSocket:', this.url);
            this.ws = new WebSocket(this.url);

            this.ws.onopen = this.handleOpen.bind(this);
            this.ws.onmessage = this.handleMessage.bind(this);
            this.ws.onclose = this.handleClose.bind(this);
            this.ws.onerror = this.handleError.bind(this);

        } catch (error) {
            logger.error('❌ Failed to create WebSocket:', error);
            this.handleError(error);
        }
    }

    disconnect(): void {
        this.isIntentionallyClosed = true;
        this.clearTimers();

        if (this.ws) {
            this.ws.close(1000, 'Intentional disconnect');
            this.ws = null;
        }

        useAppStore.getState().setWSConnected(false);
    }

    send(message: WebSocketMessage): void {
        const messageWithTimestamp = {
            ...message,
            timestamp: Date.now(),
        };

        if (this.ws?.readyState === WebSocket.OPEN) {
            try {
                this.ws.send(JSON.stringify(messageWithTimestamp));
            } catch (error) {
                logger.error('❌ Failed to send message:', error);
                this.queueMessage(messageWithTimestamp);
            }
        } else {
            // Queue message for when connection is restored
            this.queueMessage(messageWithTimestamp);
        }
    }

    // Subscribe to specific message types
    on(messageType: string, handler: (data: any) => void): () => void {
        if (!this.messageHandlers.has(messageType)) {
            this.messageHandlers.set(messageType, []);
        }

        this.messageHandlers.get(messageType)!.push(handler);

        // Return unsubscribe function
        return () => {
            const handlers = this.messageHandlers.get(messageType);
            if (handlers) {
                const index = handlers.indexOf(handler);
                if (index > -1) {
                    handlers.splice(index, 1);
                }
            }
        };
    }

    // Get connection state
    getState(): string {
        if (!this.ws) return 'DISCONNECTED';

        switch (this.ws.readyState) {
            case WebSocket.CONNECTING: return 'CONNECTING';
            case WebSocket.OPEN: return 'CONNECTED';
            case WebSocket.CLOSING: return 'CLOSING';
            case WebSocket.CLOSED: return 'DISCONNECTED';
            default: return 'UNKNOWN';
        }
    }

    isConnected(): boolean {
        return this.ws?.readyState === WebSocket.OPEN;
    }

    // === PRIVATE METHODS ===

    private handleOpen(): void {
        logger.info('✅ WebSocket connected successfully');
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;

        useAppStore.getState().setWSConnected(true);

        // Start heartbeat
        this.startHeartbeat();

        // Send queued messages
        this.flushMessageQueue();

        // Subscribe to signals
        this.send({
            type: 'subscribe',
            data: { channel: 'signals' }
        });
    }

    private handleMessage(event: MessageEvent): void {
        try {
            const message: WebSocketMessage = JSON.parse(event.data);
            useAppStore.getState().updateHeartbeat();

            // Route message to handlers
            this.routeMessage(message);

        } catch (error) {
            logger.error('❌ Failed to parse WebSocket message:', error);
        }
    }

    private handleClose(event: CloseEvent): void {
        logger.info('🔌 WebSocket disconnected:', event.code, event.reason);

        this.clearTimers();
        useAppStore.getState().setWSConnected(false);

        // Don't reconnect if intentionally closed
        if (this.isIntentionallyClosed) {
            return;
        }

        // Auto-reconnect with exponential backoff
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
        } else {
            logger.error('❌ Max reconnection attempts reached');
            useAppStore.getState().setWSReconnecting(false);
        }
    }

    private handleError(error: any): void {
        logger.error('❌ WebSocket error:', error);
    }

    private routeMessage(message: WebSocketMessage): void {
        const { type, data } = message;

        // Handle built-in message types
        switch (type) {
            case 'signal':
                this.handleSignalMessage(data);
                break;
            case 'market_data':
                this.handleMarketDataMessage(data);
                break;
            case 'heartbeat':
                // Heartbeat handled in handleMessage
                break;
            default:
                // Route to custom handlers
                const handlers = this.messageHandlers.get(type);
                if (handlers) {
                    handlers.forEach(handler => {
                        try {
                            handler(data);
                        } catch (error) {
                            logger.error(`❌ Error in message handler for ${type}:`, error);
                        }
                    });
                }
        }
    }

    private handleSignalMessage(data: any): void {
        const store = useAppStore.getState();

        if (Array.isArray(data)) {
            store.addSignals(data);
        } else if (data && typeof data === 'object') {
            store.addSignals([data]);
        }
    }

    private handleMarketDataMessage(data: any): void {
        const store = useAppStore.getState();

        if (data && data.symbol === store.selectedSymbol) {
            store.setMarketData(data);
        }
    }

    private startHeartbeat(): void {
        this.clearHeartbeat();

        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected()) {
                this.send({
                    type: 'heartbeat',
                    data: { timestamp: Date.now() }
                });
            }
        }, 30000); // Send heartbeat every 30 seconds
    }

    private clearHeartbeat(): void {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    private scheduleReconnect(): void {
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
        }

        this.reconnectAttempts++;
        const delay = Math.min(
            this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
            this.maxReconnectDelay
        );

        logger.info(`🔄 Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
        useAppStore.getState().setWSReconnecting(true);

        this.reconnectTimeout = setTimeout(() => {
            this.connect();
        }, delay);
    }

    private queueMessage(message: WebSocketMessage): void {
        this.messageQueue.push(message);

        // Limit queue size
        if (this.messageQueue.length > 100) {
            this.messageQueue = this.messageQueue.slice(-50);
        }
    }

    private flushMessageQueue(): void {
        while (this.messageQueue.length > 0 && this.isConnected()) {
            const message = this.messageQueue.shift();
            if (message) {
                this.send(message);
            }
        }
    }

    private clearTimers(): void {
        this.clearHeartbeat();

        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }
    }
}

// Singleton instance
let wsInstance: StableWebSocket | null = null;

export const getWebSocket = (): StableWebSocket => {
    if (!wsInstance) {
        const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';
        wsInstance = new StableWebSocket(wsUrl);
    }
    return wsInstance;
};

// React hook for using WebSocket
export const useStableWebSocket = () => {
    const ws = getWebSocket();
    const connected = useAppStore(state => state.wsConnected);
    const reconnecting = useAppStore(state => state.wsReconnecting);

    return {
        ws,
        connected,
        reconnecting,
        connect: () => ws.connect(),
        disconnect: () => ws.disconnect(),
        send: (message: WebSocketMessage) => ws.send(message),
        on: (type: string, handler: (data: any) => void) => ws.on(type, handler),
    };
};

export default StableWebSocket;
