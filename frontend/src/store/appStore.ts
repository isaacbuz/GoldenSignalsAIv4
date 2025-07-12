/**
 * Simplified App Store using Zustand
 * 
 * Single source of truth for the entire application.
 * Fast, simple, and TypeScript-friendly.
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

// Types
export interface Signal {
    id: string;
    symbol: string;
    type: 'BUY' | 'SELL' | 'HOLD';
    action: string;
    confidence: number;
    price: number;
    timestamp: number;
    reasoning: string;
    source: string;
    isLive?: boolean;
    timeframe?: string;
}

export interface MarketData {
    symbol: string;
    price: number;
    change: number;
    changePercent: number;
    volume: number;
    timestamp: number;
}

export interface AIMessage {
    id: string;
    type: 'user' | 'ai';
    content: string;
    timestamp: number;
    isTyping?: boolean;
}

export interface AppState {
    // === MARKET DATA ===
    selectedSymbol: string;
    marketData: MarketData | null;
    signals: Signal[];

    // === UI STATE ===
    aiChatOpen: boolean;
    selectedSignal: Signal | null;
    aiMessages: AIMessage[];
    isAITyping: boolean;

    // === CONNECTION STATE ===
    wsConnected: boolean;
    wsReconnecting: boolean;
    lastHeartbeat: number;

    // === PERFORMANCE ===
    signalCount: number;
    lastSignalTime: number;

    // === ACTIONS ===
    // Market actions
    setSymbol: (symbol: string) => void;
    setMarketData: (data: MarketData) => void;
    addSignals: (signals: Signal[]) => void;
    clearSignals: () => void;

    // UI actions
    toggleAIChat: () => void;
    selectSignal: (signal: Signal | null) => void;
    addAIMessage: (message: Omit<AIMessage, 'id' | 'timestamp'>) => void;
    setAITyping: (typing: boolean) => void;
    clearAIMessages: () => void;

    // Connection actions
    setWSConnected: (connected: boolean) => void;
    setWSReconnecting: (reconnecting: boolean) => void;
    updateHeartbeat: () => void;

    // Utility actions
    reset: () => void;
}

// Initial state
const initialState = {
    // Market data
    selectedSymbol: 'SPY',
    marketData: null,
    signals: [],

    // UI state
    aiChatOpen: false,
    selectedSignal: null,
    aiMessages: [
        {
            id: 'welcome',
            type: 'ai' as const,
            content: '🔮 Welcome, seeker of market wisdom. I am the Golden Eye AI Prophet. Ask me about any symbol or seek guidance on your trading journey...',
            timestamp: Date.now(),
        }
    ],
    isAITyping: false,

    // Connection state
    wsConnected: false,
    wsReconnecting: false,
    lastHeartbeat: 0,

    // Performance
    signalCount: 0,
    lastSignalTime: 0,
};

// Create store with Zustand
export const useAppStore = create<AppState>()(
    subscribeWithSelector((set, get) => ({
        ...initialState,

        // === MARKET ACTIONS ===
        setSymbol: (symbol: string) => {
            set({ selectedSymbol: symbol });
            // Clear signals when changing symbol
            set({ signals: [], signalCount: 0 });
        },

        setMarketData: (data: MarketData) => {
            set({ marketData: data });
        },

        addSignals: (newSignals: Signal[]) => {
            const state = get();
            const existingIds = new Set(state.signals.map(s => s.id));
            const uniqueSignals = newSignals.filter(s => !existingIds.has(s.id));

            if (uniqueSignals.length > 0) {
                set({
                    signals: [...state.signals, ...uniqueSignals]
                        .sort((a, b) => b.timestamp - a.timestamp) // Latest first
                        .slice(0, 100), // Keep only latest 100 signals
                    signalCount: state.signalCount + uniqueSignals.length,
                    lastSignalTime: Date.now(),
                });
            }
        },

        clearSignals: () => {
            set({ signals: [], signalCount: 0 });
        },

        // === UI ACTIONS ===
        toggleAIChat: () => {
            set(state => ({ aiChatOpen: !state.aiChatOpen }));
        },

        selectSignal: (signal: Signal | null) => {
            set({ selectedSignal: signal });
        },

        addAIMessage: (message: Omit<AIMessage, 'id' | 'timestamp'>) => {
            const newMessage: AIMessage = {
                ...message,
                id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                timestamp: Date.now(),
            };

            set(state => ({
                aiMessages: [...state.aiMessages, newMessage].slice(-50), // Keep only latest 50 messages
                isAITyping: false,
            }));
        },

        setAITyping: (typing: boolean) => {
            set({ isAITyping: typing });
        },

        clearAIMessages: () => {
            set({ aiMessages: [initialState.aiMessages[0]] }); // Keep welcome message
        },

        // === CONNECTION ACTIONS ===
        setWSConnected: (connected: boolean) => {
            set({ wsConnected: connected, wsReconnecting: false });
        },

        setWSReconnecting: (reconnecting: boolean) => {
            set({ wsReconnecting: reconnecting });
        },

        updateHeartbeat: () => {
            set({ lastHeartbeat: Date.now() });
        },

        // === UTILITY ACTIONS ===
        reset: () => {
            set(initialState);
        },
    }))
);

// Selectors for optimized subscriptions
export const useSignals = () => useAppStore(state => state.signals);
export const useSelectedSymbol = () => useAppStore(state => state.selectedSymbol);
export const useMarketData = () => useAppStore(state => state.marketData);
export const useWSConnected = () => useAppStore(state => state.wsConnected);
export const useAIChat = () => useAppStore(state => ({
    isOpen: state.aiChatOpen,
    messages: state.aiMessages,
    isTyping: state.isAITyping,
}));

// Performance monitoring
if (process.env.NODE_ENV === 'development') {
    useAppStore.subscribe(
        state => state.signals.length,
        (signalCount) => {
            if (signalCount % 10 === 0 && signalCount > 0) {
                console.log(`📊 Store performance: ${signalCount} signals in store`);
            }
        }
    );
} 