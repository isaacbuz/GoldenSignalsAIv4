/**
 * Robust WebSocket Service with Automatic Reconnection
 * Handles connection failures, retries, and message queuing
 */

export interface WebSocketConfig {
    url: string;
    reconnectInterval?: number;
    maxReconnectAttempts?: number;
    heartbeatInterval?: number;
    messageQueueSize?: number;
}

export interface WebSocketMessage {
    type: string;
    data: any;
    timestamp?: string;
}

type MessageHandler = (message: WebSocketMessage) => void;
type ConnectionHandler = (event: Event) => void;
type ErrorHandler = (error: Event) => void;

export class RobustWebSocketService {
    private ws: WebSocket | null = null;
    private config: Required<WebSocketConfig>;
    private reconnectAttempts = 0;
    private messageHandlers: Map<string, Set<MessageHandler>> = new Map();
    private connectionHandlers: Set<ConnectionHandler> = new Set();
    private errorHandlers: Set<ErrorHandler> = new Set();
    private messageQueue: WebSocketMessage[] = [];
    private reconnectTimer: NodeJS.Timeout | null = null;
    private heartbeatTimer: NodeJS.Timeout | null = null;
    private isIntentionallyClosed = false;
    private connectionPromise: Promise<void> | null = null;

    constructor(config: WebSocketConfig) {
        this.config = {
            url: config.url,
            reconnectInterval: config.reconnectInterval ?? 5000,
            maxReconnectAttempts: config.maxReconnectAttempts ?? 10,
            heartbeatInterval: config.heartbeatInterval ?? 30000,
            messageQueueSize: config.messageQueueSize ?? 100,
        };
    }

    /**
     * Connect to WebSocket server
     */
    async connect(): Promise<void> {
        if (this.connectionPromise) {
            return this.connectionPromise;
        }

        this.connectionPromise = this._connect();
        return this.connectionPromise;
    }

    private async _connect(): Promise<void> {
        return new Promise((resolve, reject) => {
            try {
                this.isIntentionallyClosed = false;
                this.ws = new WebSocket(this.config.url);

                this.ws.onopen = (event) => {
                    console.log('WebSocket connected');
                    this.reconnectAttempts = 0;
                    this.connectionPromise = null;

                    // Start heartbeat
                    this.startHeartbeat();

                    // Flush message queue
                    this.flushMessageQueue();

                    // Notify handlers
                    this.connectionHandlers.forEach(handler => handler(event));

                    resolve();
                };

                this.ws.onmessage = (event) => {
                    try {
                        const message: WebSocketMessage = JSON.parse(event.data);
                        this.handleMessage(message);
                    } catch (error) {
                        console.error('Failed to parse WebSocket message:', error);
                    }
                };

                this.ws.onerror = (event) => {
                    console.error('WebSocket error:', event);
                    this.errorHandlers.forEach(handler => handler(event));
                    reject(event);
                };

                this.ws.onclose = (event) => {
                    console.log('WebSocket closed:', event.code, event.reason);
                    this.stopHeartbeat();
                    this.connectionPromise = null;

                    if (!this.isIntentionallyClosed && this.reconnectAttempts < this.config.maxReconnectAttempts) {
                        this.scheduleReconnect();
                    }
                };
            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Disconnect from WebSocket server
     */
    disconnect(): void {
        this.isIntentionallyClosed = true;
        this.stopHeartbeat();

        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    /**
     * Send a message through WebSocket
     */
    send(message: WebSocketMessage): void {
        if (!message.timestamp) {
            message.timestamp = new Date().toISOString();
        }

        if (this.isConnected()) {
            try {
                this.ws!.send(JSON.stringify(message));
            } catch (error) {
                console.error('Failed to send message:', error);
                this.queueMessage(message);
            }
        } else {
            this.queueMessage(message);
        }
    }

    /**
     * Subscribe to specific message types
     */
    subscribe(type: string, handler: MessageHandler): () => void {
        if (!this.messageHandlers.has(type)) {
            this.messageHandlers.set(type, new Set());
        }

        this.messageHandlers.get(type)!.add(handler);

        // Return unsubscribe function
        return () => {
            const handlers = this.messageHandlers.get(type);
            if (handlers) {
                handlers.delete(handler);
                if (handlers.size === 0) {
                    this.messageHandlers.delete(type);
                }
            }
        };
    }

    /**
     * Add connection event handler
     */
    onConnect(handler: ConnectionHandler): () => void {
        this.connectionHandlers.add(handler);
        return () => this.connectionHandlers.delete(handler);
    }

    /**
     * Add error event handler
     */
    onError(handler: ErrorHandler): () => void {
        this.errorHandlers.add(handler);
        return () => this.errorHandlers.delete(handler);
    }

    /**
     * Check if WebSocket is connected
     */
    isConnected(): boolean {
        return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
    }

    /**
     * Get connection state
     */
    getState(): string {
        if (!this.ws) return 'DISCONNECTED';

        switch (this.ws.readyState) {
            case WebSocket.CONNECTING:
                return 'CONNECTING';
            case WebSocket.OPEN:
                return 'CONNECTED';
            case WebSocket.CLOSING:
                return 'CLOSING';
            case WebSocket.CLOSED:
                return 'CLOSED';
            default:
                return 'UNKNOWN';
        }
    }

    /**
     * Get reconnection attempts
     */
    getReconnectAttempts(): number {
        return this.reconnectAttempts;
    }

    private handleMessage(message: WebSocketMessage): void {
        // Handle heartbeat
        if (message.type === 'pong') {
            return;
        }

        // Notify specific type handlers
        const handlers = this.messageHandlers.get(message.type);
        if (handlers) {
            handlers.forEach(handler => {
                try {
                    handler(message);
                } catch (error) {
                    console.error(`Error in message handler for type ${message.type}:`, error);
                }
            });
        }

        // Notify wildcard handlers
        const wildcardHandlers = this.messageHandlers.get('*');
        if (wildcardHandlers) {
            wildcardHandlers.forEach(handler => {
                try {
                    handler(message);
                } catch (error) {
                    console.error('Error in wildcard message handler:', error);
                }
            });
        }
    }

    private scheduleReconnect(): void {
        this.reconnectAttempts++;
        const delay = Math.min(
            this.config.reconnectInterval * Math.pow(1.5, this.reconnectAttempts - 1),
            30000 // Max 30 seconds
        );

        console.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);

        this.reconnectTimer = setTimeout(() => {
            this.connect().catch(error => {
                console.error('Reconnection failed:', error);
            });
        }, delay);
    }

    private startHeartbeat(): void {
        this.stopHeartbeat();

        this.heartbeatTimer = setInterval(() => {
            if (this.isConnected()) {
                this.send({ type: 'ping', data: {} });
            }
        }, this.config.heartbeatInterval);
    }

    private stopHeartbeat(): void {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }

    private queueMessage(message: WebSocketMessage): void {
        this.messageQueue.push(message);

        // Limit queue size
        if (this.messageQueue.length > this.config.messageQueueSize) {
            this.messageQueue.shift();
        }
    }

    private flushMessageQueue(): void {
        while (this.messageQueue.length > 0 && this.isConnected()) {
            const message = this.messageQueue.shift()!;
            try {
                this.ws!.send(JSON.stringify(message));
            } catch (error) {
                console.error('Failed to send queued message:', error);
                this.messageQueue.unshift(message);
                break;
            }
        }
    }
}

// Singleton instance
let wsService: RobustWebSocketService | null = null;

/**
 * Get or create WebSocket service instance
 */
export const getWebSocketService = (config?: WebSocketConfig): RobustWebSocketService => {
    if (!wsService && config) {
        wsService = new RobustWebSocketService(config);
    }

    if (!wsService) {
        throw new Error('WebSocket service not initialized. Please provide config on first call.');
    }

    return wsService;
};

/**
 * React hook for WebSocket connection
 */
export const useWebSocket = (messageType?: string) => {
    const [messages, setMessages] = React.useState<WebSocketMessage[]>([]);
    const [connectionState, setConnectionState] = React.useState<string>('DISCONNECTED');
    const [error, setError] = React.useState<Error | null>(null);

    React.useEffect(() => {
        const ws = getWebSocketService();

        // Subscribe to messages
        const unsubscribe = messageType
            ? ws.subscribe(messageType, (message) => {
                setMessages(prev => [...prev, message]);
            })
            : ws.subscribe('*', (message) => {
                setMessages(prev => [...prev, message]);
            });

        // Monitor connection state
        const interval = setInterval(() => {
            setConnectionState(ws.getState());
        }, 1000);

        // Handle errors
        const unsubscribeError = ws.onError((event) => {
            setError(new Error('WebSocket error'));
        });

        // Connect if not connected
        if (!ws.isConnected()) {
            ws.connect().catch(err => setError(err));
        }

        return () => {
            unsubscribe();
            unsubscribeError();
            clearInterval(interval);
        };
    }, [messageType]);

    return {
        messages,
        connectionState,
        error,
        send: (data: any) => {
            const ws = getWebSocketService();
            ws.send({ type: messageType || 'message', data });
        }
    };
}; 