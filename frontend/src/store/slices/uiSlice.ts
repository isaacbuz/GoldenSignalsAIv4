import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface UIState {
    // Layout state
    sidebarOpen: boolean;
    sidebarCollapsed: boolean;

    // Modal and dialog state
    modals: {
        aiChat: boolean;
        signalDetails: boolean;
        settings: boolean;
        help: boolean;
        symbolSearch: boolean;
    };

    // Loading states
    loadingStates: {
        signals: boolean;
        marketData: boolean;
        aiAnalysis: boolean;
        chartData: boolean;
    };

    // WebSocket connection status
    connectionStatus: {
        websocket: 'connected' | 'disconnected' | 'connecting' | 'error';
        api: 'connected' | 'disconnected' | 'error';
    };

    // Current selections
    selectedSymbol: string;
    selectedTimeframe: string;
    selectedSignal: string | null;

    // Search and filters
    searchQuery: string;
    activeFilters: {
        signalType: string[];
        confidence: [number, number];
        timeframe: string[];
        agents: string[];
    };

    // Notifications and alerts
    notifications: Array<{
        id: string;
        type: 'success' | 'error' | 'warning' | 'info';
        title: string;
        message: string;
        timestamp: number;
        autoClose?: boolean;
        duration?: number;
    }>;

    // Performance metrics
    performanceMetrics: {
        renderTime: number;
        apiLatency: number;
        wsLatency: number;
        memoryUsage: number;
    };

    // Feature flags
    featureFlags: {
        aiChatEnabled: boolean;
        voiceCommandsEnabled: boolean;
        advancedChartsEnabled: boolean;
        realTimeAlertsEnabled: boolean;
    };
}

const initialState: UIState = {
    sidebarOpen: true,
    sidebarCollapsed: false,

    modals: {
        aiChat: false,
        signalDetails: false,
        settings: false,
        help: false,
        symbolSearch: false,
    },

    loadingStates: {
        signals: false,
        marketData: false,
        aiAnalysis: false,
        chartData: false,
    },

    connectionStatus: {
        websocket: 'disconnected',
        api: 'disconnected',
    },

    selectedSymbol: 'SPY',
    selectedTimeframe: '15m',
    selectedSignal: null,

    searchQuery: '',
    activeFilters: {
        signalType: [],
        confidence: [0, 100],
        timeframe: [],
        agents: [],
    },

    notifications: [],

    performanceMetrics: {
        renderTime: 0,
        apiLatency: 0,
        wsLatency: 0,
        memoryUsage: 0,
    },

    featureFlags: {
        aiChatEnabled: true,
        voiceCommandsEnabled: true,
        advancedChartsEnabled: true,
        realTimeAlertsEnabled: true,
    },
};

const uiSlice = createSlice({
    name: 'ui',
    initialState,
    reducers: {
        // Layout actions
        toggleSidebar: (state) => {
            state.sidebarOpen = !state.sidebarOpen;
        },

        setSidebarOpen: (state, action: PayloadAction<boolean>) => {
            state.sidebarOpen = action.payload;
        },

        toggleSidebarCollapsed: (state) => {
            state.sidebarCollapsed = !state.sidebarCollapsed;
        },

        // Modal actions
        openModal: (state, action: PayloadAction<keyof UIState['modals']>) => {
            state.modals[action.payload] = true;
        },

        closeModal: (state, action: PayloadAction<keyof UIState['modals']>) => {
            state.modals[action.payload] = false;
        },

        closeAllModals: (state) => {
            Object.keys(state.modals).forEach(key => {
                state.modals[key as keyof UIState['modals']] = false;
            });
        },

        // Loading state actions
        setLoadingState: (state, action: PayloadAction<{ key: keyof UIState['loadingStates']; loading: boolean }>) => {
            state.loadingStates[action.payload.key] = action.payload.loading;
        },

        // Connection status actions
        setConnectionStatus: (state, action: PayloadAction<{ type: keyof UIState['connectionStatus']; status: UIState['connectionStatus']['websocket'] }>) => {
            state.connectionStatus[action.payload.type] = action.payload.status;
        },

        // Selection actions
        setSelectedSymbol: (state, action: PayloadAction<string>) => {
            state.selectedSymbol = action.payload;
        },

        setSelectedTimeframe: (state, action: PayloadAction<string>) => {
            state.selectedTimeframe = action.payload;
        },

        setSelectedSignal: (state, action: PayloadAction<string | null>) => {
            state.selectedSignal = action.payload;
        },

        // Search and filter actions
        setSearchQuery: (state, action: PayloadAction<string>) => {
            state.searchQuery = action.payload;
        },

        updateFilters: (state, action: PayloadAction<Partial<UIState['activeFilters']>>) => {
            state.activeFilters = { ...state.activeFilters, ...action.payload };
        },

        clearFilters: (state) => {
            state.activeFilters = {
                signalType: [],
                confidence: [0, 100],
                timeframe: [],
                agents: [],
            };
        },

        // Notification actions
        addNotification: (state, action: PayloadAction<Omit<UIState['notifications'][0], 'id' | 'timestamp'>>) => {
            const notification = {
                ...action.payload,
                id: Date.now().toString(),
                timestamp: Date.now(),
            };
            state.notifications.push(notification);
        },

        removeNotification: (state, action: PayloadAction<string>) => {
            state.notifications = state.notifications.filter(n => n.id !== action.payload);
        },

        clearAllNotifications: (state) => {
            state.notifications = [];
        },

        // Performance metrics actions
        updatePerformanceMetrics: (state, action: PayloadAction<Partial<UIState['performanceMetrics']>>) => {
            state.performanceMetrics = { ...state.performanceMetrics, ...action.payload };
        },

        // Feature flags actions
        toggleFeatureFlag: (state, action: PayloadAction<keyof UIState['featureFlags']>) => {
            state.featureFlags[action.payload] = !state.featureFlags[action.payload];
        },

        setFeatureFlag: (state, action: PayloadAction<{ flag: keyof UIState['featureFlags']; enabled: boolean }>) => {
            state.featureFlags[action.payload.flag] = action.payload.enabled;
        },
    },
});

export const {
    toggleSidebar,
    setSidebarOpen,
    toggleSidebarCollapsed,
    openModal,
    closeModal,
    closeAllModals,
    setLoadingState,
    setConnectionStatus,
    setSelectedSymbol,
    setSelectedTimeframe,
    setSelectedSignal,
    setSearchQuery,
    updateFilters,
    clearFilters,
    addNotification,
    removeNotification,
    clearAllNotifications,
    updatePerformanceMetrics,
    toggleFeatureFlag,
    setFeatureFlag,
} = uiSlice.actions;

export default uiSlice.reducer;
