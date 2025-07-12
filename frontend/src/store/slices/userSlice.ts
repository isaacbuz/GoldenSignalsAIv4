import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface UserPreferences {
    theme: 'light' | 'dark' | 'auto';
    defaultSymbol: string;
    defaultTimeframe: string;
    dashboardLayout: 'standard' | 'advanced' | 'minimal' | 'institutional';
    dashboardMode: 'trading' | 'signals' | 'analytics' | 'ai' | 'models' | 'professional' | 'hybrid';
    aiChatSettings: {
        defaultModel: 'claude' | 'gpt4' | 'perplexity' | 'financeGPT';
        voiceEnabled: boolean;
        autoSuggestions: boolean;
    };
    chartSettings: {
        defaultType: 'candlestick' | 'line' | 'bar' | 'area';
        indicators: string[];
        timeframe: string;
    };
    notificationSettings: {
        signalAlerts: boolean;
        priceAlerts: boolean;
        emailNotifications: boolean;
        pushNotifications: boolean;
    };
}

interface UserState {
    isAuthenticated: boolean;
    user: {
        id: string | null;
        email: string | null;
        name: string | null;
        role: 'admin' | 'analyst' | 'viewer' | null;
    };
    preferences: UserPreferences;
    onboardingCompleted: boolean;
    lastActiveSymbol: string;
    recentSymbols: string[];
}

const initialState: UserState = {
    isAuthenticated: false,
    user: {
        id: null,
        email: null,
        name: null,
        role: null,
    },
    preferences: {
        theme: 'dark',
        defaultSymbol: 'SPY',
        defaultTimeframe: '15m',
        dashboardLayout: 'standard',
        dashboardMode: 'trading',
        aiChatSettings: {
            defaultModel: 'claude',
            voiceEnabled: true,
            autoSuggestions: true,
        },
        chartSettings: {
            defaultType: 'candlestick',
            indicators: ['EMA_20', 'RSI', 'MACD'],
            timeframe: '15m',
        },
        notificationSettings: {
            signalAlerts: true,
            priceAlerts: false,
            emailNotifications: true,
            pushNotifications: true,
        },
    },
    onboardingCompleted: false,
    lastActiveSymbol: 'SPY',
    recentSymbols: ['SPY', 'QQQ', 'AAPL'],
};

const userSlice = createSlice({
    name: 'user',
    initialState,
    reducers: {
        setUser: (state, action: PayloadAction<UserState['user']>) => {
            state.user = action.payload;
            state.isAuthenticated = !!action.payload.id;
        },

        updatePreferences: (state, action: PayloadAction<Partial<UserPreferences>>) => {
            state.preferences = { ...state.preferences, ...action.payload };
        },

        setTheme: (state, action: PayloadAction<UserPreferences['theme']>) => {
            state.preferences.theme = action.payload;
        },

        setDashboardMode: (state, action: PayloadAction<UserPreferences['dashboardMode']>) => {
            state.preferences.dashboardMode = action.payload;
        },

        setDashboardLayout: (state, action: PayloadAction<UserPreferences['dashboardLayout']>) => {
            state.preferences.dashboardLayout = action.payload;
        },

        setLastActiveSymbol: (state, action: PayloadAction<string>) => {
            state.lastActiveSymbol = action.payload;

            // Add to recent symbols if not already there
            if (!state.recentSymbols.includes(action.payload)) {
                state.recentSymbols = [action.payload, ...state.recentSymbols.slice(0, 9)];
            }
        },

        updateAISettings: (state, action: PayloadAction<Partial<UserPreferences['aiChatSettings']>>) => {
            state.preferences.aiChatSettings = {
                ...state.preferences.aiChatSettings,
                ...action.payload,
            };
        },

        updateChartSettings: (state, action: PayloadAction<Partial<UserPreferences['chartSettings']>>) => {
            state.preferences.chartSettings = {
                ...state.preferences.chartSettings,
                ...action.payload,
            };
        },

        updateNotificationSettings: (state, action: PayloadAction<Partial<UserPreferences['notificationSettings']>>) => {
            state.preferences.notificationSettings = {
                ...state.preferences.notificationSettings,
                ...action.payload,
            };
        },

        completeOnboarding: (state) => {
            state.onboardingCompleted = true;
        },

        logout: (state) => {
            state.isAuthenticated = false;
            state.user = {
                id: null,
                email: null,
                name: null,
                role: null,
            };
            // Keep preferences but reset user-specific data
        },
    },
});

export const {
    setUser,
    updatePreferences,
    setTheme,
    setDashboardMode,
    setDashboardLayout,
    setLastActiveSymbol,
    updateAISettings,
    updateChartSettings,
    updateNotificationSettings,
    completeOnboarding,
    logout,
} = userSlice.actions;

export default userSlice.reducer; 