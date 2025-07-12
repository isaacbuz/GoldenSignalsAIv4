/**
 * Enhanced Signal Processing Hook
 * Combines API calls, WebSocket updates, and intelligent caching
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../services/api/apiClient';
import { useSignalUpdates, SignalUpdate } from '../services/websocket/SignalWebSocketManager';
import { yieldToMain } from '../utils/performance';

export interface ProcessedSignal {
    signal_id: string;
    symbol: string;
    signal_type: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    strength: 'STRONG' | 'MODERATE' | 'WEAK';
    source: string;
    reasoning?: string;
    created_at: string;
    price?: number;
    target_price?: number;
    stop_loss?: number;
    risk_score?: number;
    agent_breakdown?: Record<string, any>;
    metadata?: Record<string, any>;

    // Enhanced fields
    isLive: boolean;
    timeToExpiry?: number;
    riskRewardRatio?: number;
    technicalScore?: number;
    sentimentScore?: number;
    consensusScore?: number;
}

export interface SignalFilters {
    symbols?: string[];
    minConfidence?: number;
    signalTypes?: string[];
    sources?: string[];
    maxAge?: number; // hours
}

export interface SignalProcessorConfig {
    enableLiveUpdates: boolean;
    maxSignals: number;
    autoRefreshInterval: number;
    confidenceThreshold: number;
}

const DEFAULT_CONFIG: SignalProcessorConfig = {
    enableLiveUpdates: true,
    maxSignals: 50,
    autoRefreshInterval: 30000, // 30 seconds
    confidenceThreshold: 70
};

export const useSignalProcessor = (
    filters: SignalFilters = {},
    config: Partial<SignalProcessorConfig> = {}
) => {
    const finalConfig = { ...DEFAULT_CONFIG, ...config };
    const queryClient = useQueryClient();

    // State management
    const [liveSignals, setLiveSignals] = useState<ProcessedSignal[]>([]);
    const [processingStats, setProcessingStats] = useState({
        totalProcessed: 0,
        averageConfidence: 0,
        strongSignals: 0,
        lastUpdate: null as Date | null
    });

    // Refs for performance
    const signalCache = useRef<Map<string, ProcessedSignal>>(new Map());
    const lastProcessTime = useRef<number>(0);

    // Fetch initial signals from API
    const {
        data: apiSignals,
        isLoading,
        error,
        refetch
    } = useQuery({
        queryKey: ['signals', filters],
        queryFn: async () => {
            try {
                const response = await apiClient.getSignals({
                    symbols: filters.symbols,
                    minConfidence: filters.minConfidence || finalConfig.confidenceThreshold,
                    limit: finalConfig.maxSignals
                });

                // Ensure we always return an array
                if (Array.isArray(response)) {
                    return response;
                } else if (response && Array.isArray(response.signals)) {
                    return response.signals;
                } else {
                    console.warn('API returned non-array signals data:', response);
                    return [];
                }
            } catch (error) {
                console.error('Error fetching signals:', error);
                return [];
            }
        },
        staleTime: finalConfig.autoRefreshInterval,
        refetchInterval: finalConfig.autoRefreshInterval,
        enabled: true
    });

    // Process raw signal into enhanced format
    const processSignal = useCallback((rawSignal: any, isLive: boolean = false): ProcessedSignal => {
        const now = Date.now();
        const signalTime = new Date(rawSignal.created_at || rawSignal.timestamp).getTime();
        const ageHours = (now - signalTime) / (1000 * 60 * 60);

        // Calculate enhanced metrics
        const riskRewardRatio = rawSignal.target_price && rawSignal.stop_loss && rawSignal.price
            ? Math.abs(rawSignal.target_price - rawSignal.price) / Math.abs(rawSignal.price - rawSignal.stop_loss)
            : undefined;

        const strength: 'STRONG' | 'MODERATE' | 'WEAK' =
            rawSignal.confidence >= 85 ? 'STRONG' :
                rawSignal.confidence >= 70 ? 'MODERATE' : 'WEAK';

        // Technical analysis score (mock for now - would integrate with real TA)
        const technicalScore = rawSignal.indicators
            ? Object.values(rawSignal.indicators).reduce((sum: number, val: any) => sum + (typeof val === 'number' ? val : 0), 0) / Object.keys(rawSignal.indicators).length
            : rawSignal.confidence;

        // Sentiment score from agent breakdown
        const sentimentScore = rawSignal.agent_breakdown?.sentiment_agent?.confidence || rawSignal.confidence;

        // Consensus score from multiple agents
        const consensusScore = rawSignal.agent_breakdown
            ? Object.keys(rawSignal.agent_breakdown).length * 20 // More agents = higher consensus
            : rawSignal.confidence;

        return {
            signal_id: rawSignal.signal_id || rawSignal.id,
            symbol: rawSignal.symbol,
            signal_type: rawSignal.signal_type || rawSignal.action,
            confidence: rawSignal.confidence,
            strength,
            source: rawSignal.source || 'unknown',
            reasoning: rawSignal.reasoning,
            created_at: rawSignal.created_at || rawSignal.timestamp,
            price: rawSignal.price || rawSignal.current_price,
            target_price: rawSignal.target_price,
            stop_loss: rawSignal.stop_loss,
            risk_score: rawSignal.risk_score,
            agent_breakdown: rawSignal.agent_breakdown,
            metadata: rawSignal.metadata,

            // Enhanced fields
            isLive,
            timeToExpiry: rawSignal.expires_at
                ? Math.max(0, new Date(rawSignal.expires_at).getTime() - now) / (1000 * 60 * 60)
                : undefined,
            riskRewardRatio,
            technicalScore,
            sentimentScore,
            consensusScore
        };
    }, []);

    // Optimized signal processing with task chunking
    const processSignalsInChunks = useCallback(async (rawSignals: any[]) => {
        const CHUNK_SIZE = 10; // Process 10 signals at a time
        const processedSignals: ProcessedSignal[] = [];

        for (let i = 0; i < rawSignals.length; i += CHUNK_SIZE) {
            const chunk = rawSignals.slice(i, i + CHUNK_SIZE);

            // Process chunk
            const chunkProcessed = chunk.map(signal => ({
                ...signal,
                // Add any processing here
                processed_at: new Date().toISOString(),
            }));

            processedSignals.push(...chunkProcessed);

            // Yield to main thread every chunk to prevent blocking
            if (i + CHUNK_SIZE < rawSignals.length) {
                await yieldToMain();
            }
        }

        return processedSignals;
    }, []);

    // Handle live WebSocket updates
    const handleLiveSignal = useCallback((signalUpdate: SignalUpdate) => {
        const processedSignal = processSignal(signalUpdate, true);

        // Update cache
        signalCache.current.set(processedSignal.signal_id, processedSignal);

        // Apply filters
        if (filters.symbols && !filters.symbols.includes(processedSignal.symbol)) return;
        if (filters.minConfidence && processedSignal.confidence < filters.minConfidence) return;
        if (filters.signalTypes && !filters.signalTypes.includes(processedSignal.signal_type)) return;

        setLiveSignals(prev => {
            // Remove duplicate if exists
            const filtered = prev.filter(s => s.signal_id !== processedSignal.signal_id);

            // Add new signal and sort by confidence
            const updated = [processedSignal, ...filtered]
                .sort((a, b) => b.confidence - a.confidence)
                .slice(0, finalConfig.maxSignals);

            return updated;
        });

        // Update processing stats
        setProcessingStats(prev => ({
            totalProcessed: prev.totalProcessed + 1,
            averageConfidence: (prev.averageConfidence * prev.totalProcessed + processedSignal.confidence) / (prev.totalProcessed + 1),
            strongSignals: prev.strongSignals + (processedSignal.strength === 'STRONG' ? 1 : 0),
            lastUpdate: new Date()
        }));

        // Invalidate related queries to trigger UI updates
        queryClient.invalidateQueries({ queryKey: ['signals'] });

    }, [filters, finalConfig.maxSignals, processSignal, queryClient]);

    // Subscribe to live updates
    useSignalUpdates(
        finalConfig.enableLiveUpdates ? handleLiveSignal : () => { },
        'signal-processor'
    );

    // Process API signals when they change
    useEffect(() => {
        if (apiSignals && Array.isArray(apiSignals)) {
            const processed = apiSignals.map(signal => processSignal(signal, false));

            // Merge with live signals, avoiding duplicates
            setLiveSignals(prev => {
                const liveIds = new Set(prev.filter(s => s.isLive).map(s => s.signal_id));
                const apiProcessed = processed.filter(s => !liveIds.has(s.signal_id));

                const merged = [...prev.filter(s => s.isLive), ...apiProcessed]
                    .sort((a, b) => {
                        // Prioritize live signals, then by confidence
                        if (a.isLive && !b.isLive) return -1;
                        if (!a.isLive && b.isLive) return 1;
                        return b.confidence - a.confidence;
                    })
                    .slice(0, finalConfig.maxSignals);

                return merged;
            });
        }
    }, [apiSignals, processSignal, finalConfig.maxSignals]);

    // Cleanup expired signals
    useEffect(() => {
        const cleanup = setInterval(() => {
            const now = Date.now();
            setLiveSignals(prev => prev.filter(signal => {
                if (!signal.timeToExpiry) return true;
                return signal.timeToExpiry > 0;
            }));
        }, 60000); // Check every minute

        return () => clearInterval(cleanup);
    }, []);

    // Performance tracking - only track when signals actually change
    useEffect(() => {
        const now = performance.now();
        if (lastProcessTime.current > 0) {
            const processingTime = now - lastProcessTime.current;
            if (processingTime > 200) { // Increased threshold to 200ms to reduce noise
                if (process.env.NODE_ENV === 'development') {
                    console.warn(`Signal processing took ${processingTime.toFixed(2)}ms`);
                }
            }
        }
        lastProcessTime.current = now;
    }, [liveSignals.length]); // Only track when the number of signals changes

    // Utility functions
    const getSignalsBySymbol = useCallback((symbol: string) => {
        return liveSignals.filter(s => s.symbol === symbol);
    }, [liveSignals]);

    const getSignalsByStrength = useCallback((strength: 'STRONG' | 'MODERATE' | 'WEAK') => {
        return liveSignals.filter(s => s.strength === strength);
    }, [liveSignals]);

    const getTopSignals = useCallback((count: number = 10) => {
        return liveSignals
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, count);
    }, [liveSignals]);

    const refreshSignals = useCallback(() => {
        refetch();
    }, [refetch]);

    return {
        // Data
        signals: liveSignals,
        apiSignals,
        processingStats,

        // State
        isLoading,
        error,

        // Utilities
        getSignalsBySymbol,
        getSignalsByStrength,
        getTopSignals,
        refreshSignals,

        // Config
        config: finalConfig
    };
}; 