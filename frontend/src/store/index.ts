/**
 * Centralized State Management with Zustand
 * 
 * Global state store for GoldenSignalsAI V3
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { persist } from 'zustand/middleware';
import { MarketData, AgentPerformance } from '../services/api';
import { PreciseOptionsSignal } from '../types/signals';

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
  signals: PreciseOptionsSignal[];
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
  setSignals: (signals: PreciseOptionsSignal[]) => void;
  addSignal: (signal: PreciseOptionsSignal) => void;
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
          signals: [],
          activeSignalsCount: 0,
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
              state.activeSignalsCount = signals.length;
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
              state.activeSignalsCount = state.signals.length;
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