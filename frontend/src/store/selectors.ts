import { createSelector } from '@reduxjs/toolkit';
import { RootState } from './store';

// Base selectors
export const selectApp = (state: RootState) => state.app;
export const selectUser = (state: RootState) => state.user;
export const selectUI = (state: RootState) => state.ui;

// App selectors
export const selectAppLoading = (state: RootState) => state.app.isLoading;
export const selectAppError = (state: RootState) => state.app.error;
export const selectAppNotifications = (state: RootState) => state.app.notifications;

// User selectors
export const selectUserProfile = (state: RootState) => state.user.user;
export const selectIsAuthenticated = (state: RootState) => state.user.isAuthenticated;
export const selectUserPreferences = (state: RootState) => state.user.preferences;
export const selectUserTheme = (state: RootState) => state.user.preferences.theme;
export const selectDashboardMode = (state: RootState) => state.user.preferences.dashboardMode;
export const selectDashboardLayout = (state: RootState) => state.user.preferences.dashboardLayout;
export const selectLastActiveSymbol = (state: RootState) => state.user.lastActiveSymbol;
export const selectRecentSymbols = (state: RootState) => state.user.recentSymbols;
export const selectAISettings = (state: RootState) => state.user.preferences.aiChatSettings;
export const selectChartSettings = (state: RootState) => state.user.preferences.chartSettings;
export const selectNotificationSettings = (state: RootState) => state.user.preferences.notificationSettings;

// UI selectors
export const selectSidebarOpen = (state: RootState) => state.ui.sidebarOpen;
export const selectSidebarCollapsed = (state: RootState) => state.ui.sidebarCollapsed;
export const selectModals = (state: RootState) => state.ui.modals;
export const selectLoadingStates = (state: RootState) => state.ui.loadingStates;
export const selectConnectionStatus = (state: RootState) => state.ui.connectionStatus;
export const selectSelectedSymbol = (state: RootState) => state.ui.selectedSymbol;
export const selectSelectedTimeframe = (state: RootState) => state.ui.selectedTimeframe;
export const selectSelectedSignal = (state: RootState) => state.ui.selectedSignal;
export const selectSearchQuery = (state: RootState) => state.ui.searchQuery;
export const selectActiveFilters = (state: RootState) => state.ui.activeFilters;
export const selectUINotifications = (state: RootState) => state.ui.notifications;
export const selectPerformanceMetrics = (state: RootState) => state.ui.performanceMetrics;
export const selectFeatureFlags = (state: RootState) => state.ui.featureFlags;

// Memoized selectors
export const selectIsAnyModalOpen = createSelector(
    [selectModals],
    (modals) => Object.values(modals).some(isOpen => isOpen)
);

export const selectIsAnyLoading = createSelector(
    [selectLoadingStates],
    (loadingStates) => Object.values(loadingStates).some(isLoading => isLoading)
);

export const selectIsConnected = createSelector(
    [selectConnectionStatus],
    (status) => status.websocket === 'connected' && status.api === 'connected'
);

export const selectUnreadNotifications = createSelector(
    [selectUINotifications],
    (notifications) => notifications.filter(n => !n.autoClose).length
);

export const selectActiveFiltersCount = createSelector(
    [selectActiveFilters],
    (filters) => {
        let count = 0;
        if (filters.signalType.length > 0) count++;
        if (filters.confidence[0] > 0 || filters.confidence[1] < 100) count++;
        if (filters.timeframe.length > 0) count++;
        if (filters.agents.length > 0) count++;
        return count;
    }
);

export const selectCurrentSymbolAndTimeframe = createSelector(
    [selectSelectedSymbol, selectSelectedTimeframe],
    (symbol, timeframe) => ({ symbol, timeframe })
);

export const selectDashboardConfiguration = createSelector(
    [selectDashboardMode, selectDashboardLayout, selectUserPreferences],
    (mode, layout, preferences) => ({
        mode,
        layout,
        theme: preferences.theme,
        aiSettings: preferences.aiChatSettings,
        chartSettings: preferences.chartSettings,
    })
);

export const selectAIConfiguration = createSelector(
    [selectAISettings, selectFeatureFlags],
    (aiSettings, featureFlags) => ({
        ...aiSettings,
        enabled: featureFlags.aiChatEnabled,
        voiceEnabled: featureFlags.voiceCommandsEnabled && aiSettings.voiceEnabled,
    })
);

export const selectChartConfiguration = createSelector(
    [selectChartSettings, selectFeatureFlags],
    (chartSettings, featureFlags) => ({
        ...chartSettings,
        advancedFeaturesEnabled: featureFlags.advancedChartsEnabled,
    })
);

export const selectApplicationState = createSelector(
    [selectIsAuthenticated, selectIsConnected, selectIsAnyLoading],
    (isAuthenticated, isConnected, isLoading) => ({
        isAuthenticated,
        isConnected,
        isLoading,
        isReady: isAuthenticated && isConnected && !isLoading,
    })
);

// Performance selectors
export const selectPerformanceStatus = createSelector(
    [selectPerformanceMetrics],
    (metrics) => ({
        ...metrics,
        status: metrics.apiLatency > 1000 ? 'slow' :
            metrics.apiLatency > 500 ? 'moderate' : 'fast',
        memoryStatus: metrics.memoryUsage > 80 ? 'high' :
            metrics.memoryUsage > 50 ? 'moderate' : 'low',
    })
); 