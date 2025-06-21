import { useEffect, useRef, useState, useCallback } from 'react';
import { getWebSocketService, WebSocketMessage } from '../services/websocketService';

interface UseWebSocketOptions {
    url?: string;
    autoConnect?: boolean;
    reconnectInterval?: number;
    maxReconnectAttempts?: number;
    messageType?: string;
}

interface UseWebSocketReturn {
    messages: WebSocketMessage[];
    connectionState: string;
    error: Error | null;
    send: (data: any) => void;
    connect: () => Promise<void>;
    disconnect: () => void;
    clearMessages: () => void;
    isConnected: boolean;
    reconnectAttempts: number;
}

export const useWebSocket = (options: UseWebSocketOptions = {}): UseWebSocketReturn => {
    const {
        url = process.env.VITE_WS_URL || 'ws://localhost:8000/ws',
        autoConnect = true,
        reconnectInterval = 5000,
        maxReconnectAttempts = 10,
        messageType
    } = options;

    const [messages, setMessages] = useState<WebSocketMessage[]>([]);
    const [connectionState, setConnectionState] = useState<string>('DISCONNECTED');
    const [error, setError] = useState<Error | null>(null);
    const [reconnectAttempts, setReconnectAttempts] = useState(0);

    const wsServiceRef = useRef<ReturnType<typeof getWebSocketService> | null>(null);
    const unsubscribeRef = useRef<(() => void) | null>(null);
    const connectionMonitorRef = useRef<NodeJS.Timeout | null>(null);

    // Initialize WebSocket service
    useEffect(() => {
        try {
            wsServiceRef.current = getWebSocketService({
                url,
                reconnectInterval,
                maxReconnectAttempts
            });
        } catch (err) {
            // Service might already be initialized
            wsServiceRef.current = getWebSocketService();
        }
    }, [url, reconnectInterval, maxReconnectAttempts]);

    // Subscribe to messages
    useEffect(() => {
        if (!wsServiceRef.current) return;

        const ws = wsServiceRef.current;

        // Subscribe to messages
        unsubscribeRef.current = messageType
            ? ws.subscribe(messageType, (message) => {
                setMessages(prev => [...prev, message]);
            })
            : ws.subscribe('*', (message) => {
                setMessages(prev => [...prev, message]);
            });

        // Monitor connection state
        connectionMonitorRef.current = setInterval(() => {
            const state = ws.getState();
            setConnectionState(state);
            setReconnectAttempts(ws.getReconnectAttempts());
        }, 1000);

        // Handle errors
        const unsubscribeError = ws.onError((event) => {
            setError(new Error('WebSocket error'));
        });

        // Auto-connect if enabled
        if (autoConnect && !ws.isConnected()) {
            ws.connect().catch(err => setError(err));
        }

        // Cleanup
        return () => {
            if (unsubscribeRef.current) {
                unsubscribeRef.current();
            }
            if (connectionMonitorRef.current) {
                clearInterval(connectionMonitorRef.current);
            }
            unsubscribeError();
        };
    }, [messageType, autoConnect]);

    // Send message
    const send = useCallback((data: any) => {
        if (!wsServiceRef.current) {
            setError(new Error('WebSocket service not initialized'));
            return;
        }

        wsServiceRef.current.send({
            type: messageType || 'message',
            data
        });
    }, [messageType]);

    // Connect manually
    const connect = useCallback(async () => {
        if (!wsServiceRef.current) {
            setError(new Error('WebSocket service not initialized'));
            return;
        }

        try {
            await wsServiceRef.current.connect();
            setError(null);
        } catch (err) {
            setError(err as Error);
        }
    }, []);

    // Disconnect manually
    const disconnect = useCallback(() => {
        if (!wsServiceRef.current) return;
        wsServiceRef.current.disconnect();
    }, []);

    // Clear messages
    const clearMessages = useCallback(() => {
        setMessages([]);
    }, []);

    return {
        messages,
        connectionState,
        error,
        send,
        connect,
        disconnect,
        clearMessages,
        isConnected: connectionState === 'CONNECTED',
        reconnectAttempts
    };
};

// Typed message hooks for specific message types
export const useSignalMessages = () => {
    return useWebSocket({ messageType: 'signal' });
};

export const useMarketDataMessages = () => {
    return useWebSocket({ messageType: 'market_update' });
};

export const useAlertMessages = () => {
    return useWebSocket({ messageType: 'alert' });
};

export const useTradeMessages = () => {
    return useWebSocket({ messageType: 'trade' });
}; 