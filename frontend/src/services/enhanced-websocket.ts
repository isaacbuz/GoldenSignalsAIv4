/**
 * Enhanced WebSocket Service for GoldenSignalsAI
 * Features:
 * - Automatic reconnection with exponential backoff
 * - Message queuing and replay
 * - Connection health monitoring
 * - Data compression support
 * - Multi-channel subscriptions
 */

import React from 'react';
import { EventEmitter } from 'events';
import pako from 'pako';

interface WebSocketConfig {
    url: string;
    reconnectInterval: number;
    maxReconnectAttempts: number;
    heartbeatInterval: number;
    enableCompression: boolean;
    queueSize: number;
}

interface ConnectionHealth {
    status: 'connected' | 'connecting' | 'disconnected' | 'error';
    latency: number;
    lastHeartbeat: Date | null;
    reconnectAttempts: number;
    messagesQueued: number;
    bytesReceived: number;
    bytesSent: number;
}

export class EnhancedWebSocketService extends EventEmitter {
    private ws: WebSocket | null = null;
    private config: WebSocketConfig;
    private health: ConnectionHealth;
    private messageQueue: any[] = [];
    private subscriptions = new Map<string, Set<(data: any) => void>>();
    private reconnectTimer: NodeJS.Timeout | null = null;
    private heartbeatTimer: NodeJS.Timeout | null = null;
    private pingTimer: NodeJS.Timeout | null = null;
    private isIntentionallyClosed = false;
    private lastPingTime = 0;

    constructor(config: Partial<WebSocketConfig> = {}) {
        super();

        this.config = {
            url: import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws',
            reconnectInterval: 1000,
            maxReconnectAttempts: 10,
            heartbeatInterval: 30000,
            enableCompression: true,
            queueSize: 100,
            ...config,
        };

        this.health = {
            status: 'disconnected',
            latency: 0,
            lastHeartbeat: null,
            reconnectAttempts: 0,
            messagesQueued: 0,
            bytesReceived: 0,
            bytesSent: 0,
        };

        this.connect();
    }

    private connect() {
        if (this.health.status === 'connecting') return;

        this.health.status = 'connecting';
        this.emit('status', this.health.status);

        try {
            // Add connection parameters
            const url = new URL(this.config.url);
            if (this.config.enableCompression) {
                url.searchParams.append('compression', 'true');
            }

            this.ws = new WebSocket(url.toString());
            this.ws.binaryType = 'arraybuffer';
            this.setupEventHandlers();
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.scheduleReconnect();
        }
    }

    private setupEventHandlers() {
        if (!this.ws) return;

        this.ws.onopen = () => {
            console.log('✅ WebSocket connected');
            this.health.status = 'connected';
            this.health.reconnectAttempts = 0;
            this.emit('status', this.health.status);
            this.emit('connected');

            this.startHeartbeat();
            this.flushMessageQueue();
            this.resubscribeAll();
        };

        this.ws.onmessage = async (event) => {
            this.health.bytesReceived += event.data.byteLength || event.data.length;

            try {
                let data;

                // Handle binary data (compressed)
                if (event.data instanceof ArrayBuffer) {
                    const decompressed = pako.inflate(event.data, { to: 'string' });
                    data = JSON.parse(decompressed);
                } else {
                    data = JSON.parse(event.data);
                }

                // Handle different message types
                switch (data.type) {
                    case 'pong':
                        this.handlePong(data);
                        break;
                    case 'heartbeat':
                        this.health.lastHeartbeat = new Date();
                        break;
                    default:
                        this.handleMessage(data);
                }
            } catch (error) {
                console.error('Failed to process WebSocket message:', error);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.health.status = 'error';
            this.emit('error', error);
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            this.health.status = 'disconnected';
            this.emit('status', this.health.status);
            this.emit('disconnected', { code: event.code, reason: event.reason });

            this.stopHeartbeat();

            if (!this.isIntentionallyClosed && this.health.reconnectAttempts < this.config.maxReconnectAttempts) {
                this.scheduleReconnect();
            }
        };
    }

    private handleMessage(message: any) {
        // Emit to channel subscribers
        const channel = message.channel || message.type;
        const subscribers = this.subscriptions.get(channel);

        if (subscribers) {
            subscribers.forEach(callback => {
                try {
                    callback(message.data || message);
                } catch (error) {
                    console.error(`Error in subscriber callback for channel ${channel}:`, error);
                }
            });
        }

        // Emit global message event
        this.emit('message', message);
    }

    private handlePong(data: any) {
        const latency = Date.now() - this.lastPingTime;
        this.health.latency = latency;
        this.emit('latency', latency);
    }

    private startHeartbeat() {
        this.stopHeartbeat();

        // Send heartbeat
        this.heartbeatTimer = setInterval(() => {
            if (this.isConnected()) {
                this.send({
                    type: 'heartbeat',
                    timestamp: new Date().toISOString(),
                });
            }
        }, this.config.heartbeatInterval);

        // Send ping for latency measurement
        this.pingTimer = setInterval(() => {
            if (this.isConnected()) {
                this.lastPingTime = Date.now();
                this.send({
                    type: 'ping',
                    timestamp: this.lastPingTime,
                });
            }
        }, 5000); // Every 5 seconds
    }

    private stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
        if (this.pingTimer) {
            clearInterval(this.pingTimer);
            this.pingTimer = null;
        }
    }

    private scheduleReconnect() {
        if (this.reconnectTimer) return;

        this.health.reconnectAttempts++;
        const delay = Math.min(
            this.config.reconnectInterval * Math.pow(2, this.health.reconnectAttempts - 1),
            30000 // Max 30 seconds
        );

        console.log(`Reconnecting in ${delay}ms (attempt ${this.health.reconnectAttempts})`);
        this.emit('reconnecting', { attempt: this.health.reconnectAttempts, delay });

        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.connect();
        }, delay);
    }

    private flushMessageQueue() {
        while (this.messageQueue.length > 0 && this.isConnected()) {
            const message = this.messageQueue.shift();
            if (message) {
                this.send(message, false); // Don't re-queue on failure
            }
        }
        this.health.messagesQueued = this.messageQueue.length;
    }

    private resubscribeAll() {
        this.subscriptions.forEach((_, channel) => {
            this.send({
                type: 'subscribe',
                channel,
                timestamp: new Date().toISOString(),
            });
        });
    }

    // Public API
    public subscribe(channel: string, callback: (data: any) => void): () => void {
        if (!this.subscriptions.has(channel)) {
            this.subscriptions.set(channel, new Set());

            // Send subscription message if connected
            if (this.isConnected()) {
                this.send({
                    type: 'subscribe',
                    channel,
                    timestamp: new Date().toISOString(),
                });
            }
        }

        this.subscriptions.get(channel)!.add(callback);

        // Return unsubscribe function
        return () => {
            const subscribers = this.subscriptions.get(channel);
            if (subscribers) {
                subscribers.delete(callback);
                if (subscribers.size === 0) {
                    this.subscriptions.delete(channel);

                    // Send unsubscribe message if connected
                    if (this.isConnected()) {
                        this.send({
                            type: 'unsubscribe',
                            channel,
                            timestamp: new Date().toISOString(),
                        });
                    }
                }
            }
        };
    }

    public send(message: any, queue = true): boolean {
        if (!this.isConnected()) {
            if (queue && this.messageQueue.length < this.config.queueSize) {
                this.messageQueue.push(message);
                this.health.messagesQueued = this.messageQueue.length;
            }
            return false;
        }

        try {
            let data: string | ArrayBuffer;

            if (this.config.enableCompression && message.compress !== false) {
                // Compress large messages
                const json = JSON.stringify(message);
                if (json.length > 1024) { // Only compress if > 1KB
                    data = pako.deflate(json);
                } else {
                    data = json;
                }
            } else {
                data = JSON.stringify(message);
            }

            this.ws!.send(data);
            this.health.bytesSent += data.byteLength || data.length;
            return true;
        } catch (error) {
            console.error('Failed to send message:', error);
            if (queue && this.messageQueue.length < this.config.queueSize) {
                this.messageQueue.push(message);
                this.health.messagesQueued = this.messageQueue.length;
            }
            return false;
        }
    }

    public isConnected(): boolean {
        return this.ws?.readyState === WebSocket.OPEN;
    }

    public getHealth(): ConnectionHealth {
        return { ...this.health };
    }

    public disconnect() {
        this.isIntentionallyClosed = true;

        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        this.stopHeartbeat();

        if (this.ws) {
            this.ws.close(1000, 'Client disconnect');
            this.ws = null;
        }

        this.health.status = 'disconnected';
        this.emit('status', this.health.status);
    }

    public reconnect() {
        this.isIntentionallyClosed = false;
        this.disconnect();
        this.health.reconnectAttempts = 0;
        this.connect();
    }

    // Convenience methods for common channels
    public subscribeToMarketData(symbol: string, callback: (data: any) => void) {
        return this.subscribe(`market:${symbol}`, callback);
    }

    public subscribeToSignals(callback: (data: any) => void) {
        return this.subscribe('signals', callback);
    }

    public subscribeToAgentStatus(agentId: string, callback: (data: any) => void) {
        return this.subscribe(`agent:${agentId}`, callback);
    }

    public requestMarketSnapshot(symbols: string[]) {
        return this.send({
            type: 'request',
            action: 'market_snapshot',
            symbols,
            timestamp: new Date().toISOString(),
        });
    }

    public requestAgentPerformance() {
        return this.send({
            type: 'request',
            action: 'agent_performance',
            timestamp: new Date().toISOString(),
        });
    }
}

// Create singleton instance
export const enhancedWebSocket = new EnhancedWebSocketService();

// React hook for WebSocket health
export const useWebSocketHealth = () => {
    const [health, setHealth] = React.useState(enhancedWebSocket.getHealth());

    React.useEffect(() => {
        const updateHealth = () => setHealth(enhancedWebSocket.getHealth());

        enhancedWebSocket.on('status', updateHealth);
        enhancedWebSocket.on('latency', updateHealth);

        const interval = setInterval(updateHealth, 1000);

        return () => {
            enhancedWebSocket.off('status', updateHealth);
            enhancedWebSocket.off('latency', updateHealth);
            clearInterval(interval);
        };
    }, []);

    return health;
};

// React hook for WebSocket subscriptions
export const useWebSocketSubscription = (channel: string, callback: (data: any) => void) => {
    React.useEffect(() => {
        const unsubscribe = enhancedWebSocket.subscribe(channel, callback);
        return unsubscribe;
    }, [channel, callback]);
};

export default enhancedWebSocket; 