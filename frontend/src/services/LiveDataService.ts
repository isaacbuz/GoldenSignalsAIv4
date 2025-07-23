/**
 * Live Data Service - Unified Real-time Data Management
 *
 * Orchestrates all live data connections including:
 * - WebSocket signals
 * - Market data updates
 * - Agent performance
 * - AI insights
 * - System metrics
 */

import { EventEmitter } from 'events';
import { signalWebSocketManager } from './websocket/SignalWebSocketManager';
import { apiClient } from './api/apiClient';
import { MarketDataService } from './MarketDataService';

export interface LiveDataUpdate {
    type: 'signal' | 'market' | 'agent' | 'insight' | 'metric' | 'news';
    symbol?: string;
    data: any;
    timestamp: string;
    source: string;
}

export interface LiveSignalData {
    signal_id: string;
    symbol: string;
    signal_type: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    price: number;
    reasoning: string;
    created_at: string;
    agents: string[];
    metadata?: any;
}

export interface LiveMarketData {
    symbol: string;
    price: number;
    change: number;
    change_percent: number;
    volume: number;
    bid?: number;
    ask?: number;
    high?: number;
    low?: number;
    open?: number;
    timestamp: string;
}

export interface LiveAgentData {
    agent_name: string;
    status: 'active' | 'inactive' | 'error';
    performance: {
        accuracy: number;
        total_signals: number;
        win_rate: number;
        avg_return: number;
    };
    last_update: string;
}

export interface LiveMetrics {
    active_connections: number;
    signals_generated_today: number;
    system_health: 'healthy' | 'warning' | 'error';
    cpu_usage: number;
    memory_usage: number;
    api_response_time: number;
}

class LiveDataService extends EventEmitter {
    private static instance: LiveDataService;
    private isConnected = false;
    private subscriptions = new Map<string, Set<(data: any) => void>>();
    private marketDataService: MarketDataService;
    private updateIntervals = new Map<string, NodeJS.Timeout>();
    private subscribedSymbols = new Set<string>();

    // Data caches for immediate access
    private signalCache = new Map<string, LiveSignalData[]>();
    private marketCache = new Map<string, LiveMarketData>();
    private agentCache = new Map<string, LiveAgentData>();
    private metricsCache: LiveMetrics | null = null;

    // Connection statistics
    private stats = {
        totalUpdates: 0,
        signalUpdates: 0,
        marketUpdates: 0,
        agentUpdates: 0,
        lastUpdate: null as string | null,
        uptime: Date.now(),
    };

    private constructor() {
        super();
        this.marketDataService = MarketDataService.getInstance();
        this.initializeConnections();
    }

    public static getInstance(): LiveDataService {
        if (!LiveDataService.instance) {
            LiveDataService.instance = new LiveDataService();
        }
        return LiveDataService.instance;
    }

    /**
     * Initialize all data connections
     */
    private async initializeConnections() {
        try {
            // Initialize WebSocket connections
            await this.initializeWebSocket();

            // Start periodic updates
            this.startPeriodicUpdates();

            // Mark as connected
            this.isConnected = true;
            this.emit('connected');

            logger.info('✅ Live Data Service initialized successfully');
        } catch (error) {
            logger.error('❌ Failed to initialize Live Data Service:', error);
            this.emit('error', error);
        }
    }

    /**
     * Initialize WebSocket connections
     */
    private async initializeWebSocket() {
        // Connect to signal WebSocket
        await signalWebSocketManager.connect();

        // Subscribe to signal updates
        signalWebSocketManager.subscribeToSignals((signalUpdate) => {
            this.handleSignalUpdate(signalUpdate);
        });

        // Subscribe to agent updates
        signalWebSocketManager.subscribeToAgents((agentUpdate) => {
            this.handleAgentUpdate(agentUpdate);
        });

        // Subscribe to market data updates
        signalWebSocketManager.subscribeToMarketData((marketUpdate) => {
            this.handleMarketUpdate(marketUpdate);
        });

        // Monitor connection status
        setInterval(() => {
            const wsConnected = signalWebSocketManager.isConnected();
            if (wsConnected !== this.isConnected) {
                this.isConnected = wsConnected;
                this.emit(wsConnected ? 'connected' : 'disconnected');
            }
        }, 5000);
    }

    /**
     * Start periodic data updates
     */
    private startPeriodicUpdates() {
        // System metrics every 30 seconds
        this.updateIntervals.set('metrics', setInterval(async () => {
            try {
                const metrics = await apiClient.getSystemHealth();
                this.handleMetricsUpdate(metrics);
            } catch (error) {
                logger.warn('Failed to fetch system metrics:', error);
            }
        }, 30000));

        // Market data for subscribed symbols every 10 seconds
        this.updateIntervals.set('market', setInterval(async () => {
            if (this.subscribedSymbols.size === 0) return;

            logger.info(`Fetching live market data for ${this.subscribedSymbols.size} symbols:`, Array.from(this.subscribedSymbols));

            for (const symbol of this.subscribedSymbols) {
                try {
                    const marketData = await apiClient.getMarketData(symbol);
                    if (marketData) {
                        this.handleMarketUpdate({
                            symbol: marketData.symbol,
                            price: marketData.price,
                            change: marketData.change,
                            change_percent: marketData.change_percent,
                            volume: marketData.volume,
                            bid: marketData.bid,
                            ask: marketData.ask,
                            high: marketData.high,
                            low: marketData.low,
                            open: marketData.open,
                            timestamp: marketData.timestamp,
                        });

                        logger.info(`Updated market data for ${symbol}: $${marketData.price} (${marketData.change_percent > 0 ? '+' : ''}${marketData.change_percent.toFixed(2)}%)`);
                    }
                } catch (error) {
                    logger.warn(`Failed to fetch market data for ${symbol}:`, error);
                }
            }
        }, 10000));

        // Agent performance every 60 seconds
        this.updateIntervals.set('agents', setInterval(async () => {
            try {
                const agentPerformance = await apiClient.getAgentPerformance();
                Object.entries(agentPerformance).forEach(([agentName, performance]) => {
                    this.handleAgentUpdate({
                        agent_name: agentName,
                        status: 'active',
                        performance: performance as any,
                        last_update: new Date().toISOString(),
                    });
                });
            } catch (error) {
                logger.warn('Failed to fetch agent performance:', error);
            }
        }, 60000));
    }

    /**
     * Handle signal updates from WebSocket
     */
    private handleSignalUpdate(signalUpdate: any) {
        const signal: LiveSignalData = {
            signal_id: signalUpdate.signal_id || signalUpdate.id || `signal_${Date.now()}`,
            symbol: signalUpdate.symbol || 'UNKNOWN',
            signal_type: signalUpdate.signal_type || signalUpdate.type || 'HOLD',
            confidence: signalUpdate.confidence || 0,
            price: signalUpdate.price || 0,
            reasoning: signalUpdate.reasoning || '',
            created_at: signalUpdate.timestamp || signalUpdate.created_at || new Date().toISOString(),
            agents: signalUpdate.agents || [],
            metadata: signalUpdate.metadata,
        };

        // Update cache
        const symbolSignals = this.signalCache.get(signal.symbol) || [];
        symbolSignals.unshift(signal);
        this.signalCache.set(signal.symbol, symbolSignals.slice(0, 50)); // Keep last 50 signals

        // Update stats
        this.stats.signalUpdates++;
        this.stats.totalUpdates++;
        this.stats.lastUpdate = signal.created_at;

        // Emit update
        const update: LiveDataUpdate = {
            type: 'signal',
            symbol: signal.symbol,
            data: signal,
            timestamp: signal.created_at,
            source: 'websocket',
        };

        this.emit('update', update);
        this.emit('signal', signal);
        this.emit(`signal:${signal.symbol}`, signal);
    }

    /**
     * Handle market data updates
     */
    private handleMarketUpdate(marketUpdate: any) {
        const marketData: LiveMarketData = {
            symbol: marketUpdate.symbol,
            price: marketUpdate.price || 0,
            change: marketUpdate.change || 0,
            change_percent: marketUpdate.change_percent || 0,
            volume: marketUpdate.volume || 0,
            bid: marketUpdate.bid,
            ask: marketUpdate.ask,
            high: marketUpdate.high,
            low: marketUpdate.low,
            open: marketUpdate.open,
            timestamp: marketUpdate.timestamp || new Date().toISOString(),
        };

        // Check if data actually changed before updating
        const existingData = this.marketCache.get(marketData.symbol);
        if (existingData) {
            const hasChanged = (
                existingData.price !== marketData.price ||
                existingData.change !== marketData.change ||
                existingData.change_percent !== marketData.change_percent ||
                existingData.volume !== marketData.volume
            );

            if (!hasChanged) {
                // Data hasn't changed, skip update
                return;
            }
        }

        // Update cache
        this.marketCache.set(marketData.symbol, marketData);

        // Update stats
        this.stats.marketUpdates++;
        this.stats.totalUpdates++;
        this.stats.lastUpdate = marketData.timestamp;

        // Emit update
        const update: LiveDataUpdate = {
            type: 'market',
            symbol: marketData.symbol,
            data: marketData,
            timestamp: marketData.timestamp,
            source: 'api',
        };

        this.emit('update', update);
        this.emit('market', marketData);
        this.emit(`market:${marketData.symbol}`, marketData);
    }

    /**
     * Handle agent performance updates
     */
    private handleAgentUpdate(agentUpdate: any) {
        const agentData: LiveAgentData = {
            agent_name: agentUpdate.agent_name || agentUpdate.name || 'unknown',
            status: agentUpdate.status || 'active',
            performance: {
                accuracy: agentUpdate.performance?.accuracy || agentUpdate.accuracy || 0,
                total_signals: agentUpdate.performance?.total_signals || agentUpdate.total_signals || 0,
                win_rate: agentUpdate.performance?.win_rate || 0,
                avg_return: agentUpdate.performance?.avg_return || 0,
            },
            last_update: agentUpdate.timestamp || agentUpdate.last_update || new Date().toISOString(),
        };

        // Update cache
        this.agentCache.set(agentData.agent_name, agentData);

        // Update stats
        this.stats.agentUpdates++;
        this.stats.totalUpdates++;
        this.stats.lastUpdate = agentData.last_update;

        // Emit update
        const update: LiveDataUpdate = {
            type: 'agent',
            data: agentData,
            timestamp: agentData.last_update,
            source: 'websocket',
        };

        this.emit('update', update);
        this.emit('agent', agentData);
        this.emit(`agent:${agentData.agent_name}`, agentData);
    }

    /**
     * Handle system metrics updates
     */
    private handleMetricsUpdate(metricsUpdate: any) {
        const metrics: LiveMetrics = {
            active_connections: metricsUpdate.active_connections || 0,
            signals_generated_today: metricsUpdate.signals_generated_today || 0,
            system_health: metricsUpdate.system_health || 'healthy',
            cpu_usage: metricsUpdate.cpu_usage || 0,
            memory_usage: metricsUpdate.memory_usage || 0,
            api_response_time: metricsUpdate.api_response_time || 0,
        };

        // Update cache
        this.metricsCache = metrics;

        // Emit update
        const update: LiveDataUpdate = {
            type: 'metric',
            data: metrics,
            timestamp: new Date().toISOString(),
            source: 'api',
        };

        this.emit('update', update);
        this.emit('metrics', metrics);
    }

    /**
     * Subscribe to live data updates
     */
    public subscribe(type: string, callback: (data: any) => void): () => void {
        if (!this.subscriptions.has(type)) {
            this.subscriptions.set(type, new Set());
        }

        this.subscriptions.get(type)!.add(callback);
        this.on(type, callback);

        // Return unsubscribe function
        return () => {
            this.subscriptions.get(type)?.delete(callback);
            this.off(type, callback);
        };
    }

    /**
     * Subscribe to symbol-specific data
     */
    public subscribeToSymbol(symbol: string, callback: (data: any) => void): () => void {
        this.subscribedSymbols.add(symbol);

        // Subscribe to both signals and market data for this symbol
        const unsubSignal = this.subscribe(`signal:${symbol}`, callback);
        const unsubMarket = this.subscribe(`market:${symbol}`, callback);

        return () => {
            this.subscribedSymbols.delete(symbol);
            unsubSignal();
            unsubMarket();
        };
    }

    /**
     * Get cached data
     */
    public getCachedSignals(symbol: string): LiveSignalData[] {
        return this.signalCache.get(symbol) || [];
    }

    public getCachedMarketData(symbol: string): LiveMarketData | null {
        return this.marketCache.get(symbol) || null;
    }

    public getCachedAgentData(agentName: string): LiveAgentData | null {
        return this.agentCache.get(agentName) || null;
    }

    public getCachedMetrics(): LiveMetrics | null {
        return this.metricsCache;
    }

    /**
     * Get all cached data
     */
    public getAllCachedData() {
        return {
            signals: Object.fromEntries(this.signalCache),
            market: Object.fromEntries(this.marketCache),
            agents: Object.fromEntries(this.agentCache),
            metrics: this.metricsCache,
        };
    }

    /**
     * Get connection statistics
     */
    public getStats() {
        return {
            ...this.stats,
            isConnected: this.isConnected,
            subscribedSymbols: Array.from(this.subscribedSymbols),
            cacheSize: {
                signals: this.signalCache.size,
                market: this.marketCache.size,
                agents: this.agentCache.size,
            },
            uptime: Date.now() - this.stats.uptime,
        };
    }

    /**
     * Force refresh all data
     */
    public async refreshAll(): Promise<void> {
        try {
            // Refresh system metrics
            const metrics = await apiClient.getSystemHealth();
            this.handleMetricsUpdate(metrics);

            // Refresh market data for subscribed symbols
            for (const symbol of this.subscribedSymbols) {
                const marketData = await apiClient.getMarketData(symbol);
                if (marketData) {
                    this.handleMarketUpdate(marketData);
                }
            }

            // Refresh agent performance
            const agentPerformance = await apiClient.getAgentPerformance();
            Object.entries(agentPerformance).forEach(([agentName, performance]) => {
                this.handleAgentUpdate({
                    agent_name: agentName,
                    status: 'active',
                    performance: performance as any,
                    last_update: new Date().toISOString(),
                });
            });

            this.emit('refreshed');
        } catch (error) {
            logger.error('Failed to refresh live data:', error);
            this.emit('error', error);
        }
    }

    /**
     * Check connection status
     */
    public isConnectedToLiveData(): boolean {
        return this.isConnected && signalWebSocketManager.isConnected();
    }

    /**
     * Disconnect from all data sources
     */
    public disconnect() {
        // Clear intervals
        this.updateIntervals.forEach((interval) => {
            clearInterval(interval);
        });
        this.updateIntervals.clear();

        // Disconnect WebSocket
        signalWebSocketManager.disconnect();

        // Clear caches
        this.signalCache.clear();
        this.marketCache.clear();
        this.agentCache.clear();
        this.metricsCache = null;

        // Clear subscriptions
        this.subscriptions.clear();
        this.subscribedSymbols.clear();

        this.isConnected = false;
        this.emit('disconnected');
    }

    /**
     * Reconnect to all data sources
     */
    public async reconnect() {
        this.disconnect();
        await this.initializeConnections();
    }
}

// Export singleton instance
export const liveDataService = LiveDataService.getInstance();

// React hooks for live data
export const useLiveData = (type: string, callback: (data: any) => void) => {
    React.useEffect(() => {
        const unsubscribe = liveDataService.subscribe(type, callback);
        return unsubscribe;
    }, [type, callback]);
};

export const useLiveSignals = (symbol: string, callback: (signal: LiveSignalData) => void) => {
    React.useEffect(() => {
        const unsubscribe = liveDataService.subscribe(`signal:${symbol}`, callback);
        return unsubscribe;
    }, [symbol, callback]);
};

export const useLiveMarketData = (symbol: string, callback: (data: LiveMarketData) => void) => {
    React.useEffect(() => {
        const unsubscribe = liveDataService.subscribe(`market:${symbol}`, callback);
        return unsubscribe;
    }, [symbol, callback]);
};

export const useLiveMetrics = (callback: (metrics: LiveMetrics) => void) => {
    React.useEffect(() => {
        const unsubscribe = liveDataService.subscribe('metrics', callback);
        return unsubscribe;
    }, [callback]);
};

// Import React for hooks
import React from 'react';
import logger from './logger';
