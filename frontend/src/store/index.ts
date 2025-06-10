/**
 * Centralized State Management with Zustand
 * 
 * Global state store for GoldenSignalsAI V3
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { persist } from 'zustand/middleware';
import { Signal, MarketData, AgentPerformance } from '../services/api';

// Types
export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'user' | 'viewer';
  preferences: UserPreferences;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  notifications: {
    signals: boolean;
    portfolio: boolean;
    system: boolean;
    email: boolean;
    push: boolean;
  };
  dashboard: {
    refreshInterval: number;
    defaultSymbols: string[];
    autoRefresh: boolean;
  };
  trading: {
    riskTolerance: 'low' | 'medium' | 'high';
    maxPositionSize: number;
    autoExecution: boolean;
  };
}

export interface AppState {
  // Auth
  user: User | null;
  isAuthenticated: boolean;
  
  // Market Data
  marketData: Record<string, MarketData>;
  isMarketOpen: boolean;
  
  // Signals
  signals: Signal[];
  activeSignalsCount: number;
  
  // Agents
  agents: any[];
  agentPerformance: Record<string, AgentPerformance>;
  
  // UI State
  isLoading: boolean;
  selectedSymbols: string[];
  notifications: Notification[];
  
  // WebSocket
  wsConnected: boolean;
  
  // Settings
  settings: {
    apiEndpoint: string;
    wsEndpoint: string;
    refreshInterval: number;
  };
}

export interface AppActions {
  // Auth Actions
  setUser: (user: User | null) => void;
  logout: () => void;
  
  // Market Data Actions
  setMarketData: (symbol: string, data: MarketData) => void;
  setMarketOpen: (isOpen: boolean) => void;
  
  // Signal Actions
  setSignals: (signals: Signal[]) => void;
  addSignal: (signal: Signal) => void;
  removeSignal: (signalId: string) => void;
  
  // Agent Actions
  setAgents: (agents: any[]) => void;
  setAgentPerformance: (performance: Record<string, AgentPerformance>) => void;
  
  // UI Actions
  setLoading: (loading: boolean) => void;
  setSelectedSymbols: (symbols: string[]) => void;
  addNotification: (notification: Notification) => void;
  removeNotification: (id: string) => void;
  
  // WebSocket Actions
  setWsConnected: (connected: boolean) => void;
  
  // Settings Actions
  updateSettings: (settings: Partial<AppState['settings']>) => void;
  updateUserPreferences: (preferences: Partial<UserPreferences>) => void;
}

export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  action?: {
    label: string;
    callback: () => void;
  };
}

type Store = AppState & AppActions;

// Default user preferences
const defaultPreferences: UserPreferences = {
  theme: 'dark',
  notifications: {
    signals: true,
    portfolio: true,
    system: true,
    email: false,
    push: true,
  },
  dashboard: {
    refreshInterval: 5000,
    defaultSymbols: ['SPY', 'QQQ', 'TSLA', 'AAPL'],
    autoRefresh: true,
  },
  trading: {
    riskTolerance: 'medium',
    maxPositionSize: 10000,
    autoExecution: false,
  },
};

// Create the store
export const useAppStore = create<Store>()(
  subscribeWithSelector(
    immer(
      persist(
        (set, get) => ({
          // Initial State
          user: null,
          isAuthenticated: false,
          marketData: {},
          isMarketOpen: false,
          signals: [
            // Sample signals for immediate display
            {
              signal_id: 'sample-1',
              symbol: 'AAPL',
              signal_type: 'BUY' as const,
              confidence: 0.85,
              strength: 'STRONG' as const,
              source: 'ai_agent',
              current_price: 203.93,
              entry_price: 203.93,
              exit_price: 215.00,
              stop_loss: 195.00,
              take_profit: 215.00,
              reasoning: 'Strong technical indicators with bullish momentum. RSI oversold and showing reversal patterns.',
              features: {},
              created_at: new Date().toISOString(),
            },
            {
              signal_id: 'sample-2',
              symbol: 'TSLA',
              signal_type: 'SELL' as const,
              confidence: 0.78,
              strength: 'MODERATE' as const,
              source: 'ai_agent',
              current_price: 248.50,
              entry_price: 248.50,
              exit_price: 230.00,
              stop_loss: 255.00,
              take_profit: 230.00,
              reasoning: 'Overbought conditions detected. Technical analysis suggests potential downward correction.',
              features: {},
              created_at: new Date(Date.now() - 300000).toISOString(), // 5 minutes ago
            },
            {
              signal_id: 'sample-3',
              symbol: 'GOOGL',
              signal_type: 'BUY' as const,
              confidence: 0.72,
              strength: 'MODERATE' as const,
              source: 'ai_agent',
              current_price: 138.21,
              entry_price: 138.21,
              exit_price: 145.00,
              stop_loss: 132.00,
              take_profit: 145.00,
              reasoning: 'Earnings momentum and positive market sentiment. Key support levels holding strong.',
              features: {},
              created_at: new Date(Date.now() - 600000).toISOString(), // 10 minutes ago
            },
          ],
          activeSignalsCount: 3,
          agents: [],
          agentPerformance: {},
          isLoading: false,
          selectedSymbols: ['SPY', 'QQQ', 'TSLA', 'AAPL'],
          notifications: [],
          wsConnected: false,
          settings: {
            apiEndpoint: 'http://localhost:8000/api/v1',
            wsEndpoint: 'ws://localhost:8000',
            refreshInterval: 5000,
          },

          // Auth Actions
          setUser: (user) =>
            set((state) => {
              state.user = user;
              state.isAuthenticated = !!user;
            }),

          logout: () =>
            set((state) => {
              state.user = null;
              state.isAuthenticated = false;
              state.notifications = [];
            }),

          // Market Data Actions
          setMarketData: (symbol, data) =>
            set((state) => {
              state.marketData[symbol] = data;
            }),

          setMarketOpen: (isOpen) =>
            set((state) => {
              state.isMarketOpen = isOpen;
            }),

          // Signal Actions
          setSignals: (signals) =>
            set((state) => {
              state.signals = signals;
              state.activeSignalsCount = signals.filter(
                (s) => s.signal_type !== 'HOLD'
              ).length;
            }),

          addSignal: (signal) =>
            set((state) => {
              const existingIndex = state.signals.findIndex(
                (s) => s.signal_id === signal.signal_id
              );
              if (existingIndex >= 0) {
                state.signals[existingIndex] = signal;
              } else {
                state.signals.unshift(signal);
              }
              state.activeSignalsCount = state.signals.filter(
                (s) => s.signal_type !== 'HOLD'
              ).length;
            }),

          removeSignal: (signalId) =>
            set((state) => {
              state.signals = state.signals.filter((s) => s.signal_id !== signalId);
              state.activeSignalsCount = state.signals.filter(
                (s) => s.signal_type !== 'HOLD'
              ).length;
            }),

          // Agent Actions
          setAgents: (agents) =>
            set((state) => {
              state.agents = agents;
            }),

          setAgentPerformance: (performance) =>
            set((state) => {
              state.agentPerformance = performance;
            }),

          // UI Actions
          setLoading: (loading) =>
            set((state) => {
              state.isLoading = loading;
            }),

          setSelectedSymbols: (symbols) =>
            set((state) => {
              state.selectedSymbols = symbols;
            }),

          addNotification: (notification) =>
            set((state) => {
              state.notifications.unshift(notification);
              // Keep only last 50 notifications
              if (state.notifications.length > 50) {
                state.notifications = state.notifications.slice(0, 50);
              }
            }),

          removeNotification: (id) =>
            set((state) => {
              state.notifications = state.notifications.filter((n) => n.id !== id);
            }),

          // WebSocket Actions
          setWsConnected: (connected) =>
            set((state) => {
              state.wsConnected = connected;
            }),

          // Settings Actions
          updateSettings: (newSettings) =>
            set((state) => {
              Object.assign(state.settings, newSettings);
            }),

          updateUserPreferences: (preferences) =>
            set((state) => {
              if (state.user) {
                Object.assign(state.user.preferences, preferences);
              }
            }),
        }),
        {
          name: 'goldensignals-store',
          partialize: (state) => ({
            user: state.user,
            selectedSymbols: state.selectedSymbols,
            settings: state.settings,
          }),
        }
      )
    )
  )
);

// Selectors for better performance
export const useAuth = () => useAppStore((state) => ({
  user: state.user,
  isAuthenticated: state.isAuthenticated,
  setUser: state.setUser,
  logout: state.logout,
}));

export const useMarketData = () => useAppStore((state) => ({
  marketData: state.marketData,
  isMarketOpen: state.isMarketOpen,
  setMarketData: state.setMarketData,
  setMarketOpen: state.setMarketOpen,
}));

export const useSignals = () => useAppStore((state) => ({
  signals: state.signals,
  activeSignalsCount: state.activeSignalsCount,
  setSignals: state.setSignals,
  addSignal: state.addSignal,
  removeSignal: state.removeSignal,
}));

export const useAgents = () => useAppStore((state) => ({
  agents: state.agents,
  agentPerformance: state.agentPerformance,
  setAgents: state.setAgents,
  setAgentPerformance: state.setAgentPerformance,
}));

export const useNotifications = () => useAppStore((state) => ({
  notifications: state.notifications,
  addNotification: state.addNotification,
  removeNotification: state.removeNotification,
}));

export const useWebSocket = () => useAppStore((state) => ({
  wsConnected: state.wsConnected,
  setWsConnected: state.setWsConnected,
}));

// Helper function to create notifications
export const createNotification = (
  type: Notification['type'],
  title: string,
  message: string,
  action?: Notification['action']
): Notification => ({
  id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
  type,
  title,
  message,
  timestamp: new Date(),
  read: false,
  action,
}); 