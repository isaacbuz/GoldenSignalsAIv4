import { useCallback } from 'react';
import { useAppDispatch, useAppSelector } from './store';
import {
    // User actions
    setUser,
    updatePreferences,
    setTheme,
    setDashboardMode,
    setDashboardLayout,
    setLastActiveSymbol,
    updateAISettings,
    updateChartSettings,
    updateNotificationSettings,
    logout,
} from './slices/userSlice';
import {
    // UI actions
    toggleSidebar,
    setSidebarOpen,
    openModal,
    closeModal,
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
    updatePerformanceMetrics,
    toggleFeatureFlag,
} from './slices/uiSlice';
import {
    // Selectors
    selectUserProfile,
    selectIsAuthenticated,
    selectUserPreferences,
    selectUserTheme,
    selectDashboardMode,
    selectDashboardLayout,
    selectLastActiveSymbol,
    selectRecentSymbols,
    selectAISettings,
    selectChartSettings,
    selectNotificationSettings,
    selectSidebarOpen,
    selectModals,
    selectLoadingStates,
    selectConnectionStatus,
    selectSelectedSymbol,
    selectSelectedTimeframe,
    selectSearchQuery,
    selectActiveFilters,
    selectUINotifications,
    selectFeatureFlags,
    selectDashboardConfiguration,
    selectAIConfiguration,
    selectApplicationState,
} from './selectors';

// User hooks
export const useUser = () => {
    const dispatch = useAppDispatch();
    const user = useAppSelector(selectUserProfile);
    const isAuthenticated = useAppSelector(selectIsAuthenticated);
    const preferences = useAppSelector(selectUserPreferences);

    const loginUser = useCallback((userData: any) => {
        dispatch(setUser(userData));
    }, [dispatch]);

    const logoutUser = useCallback(() => {
        dispatch(logout());
    }, [dispatch]);

    const updateUserPreferences = useCallback((newPreferences: any) => {
        dispatch(updatePreferences(newPreferences));
    }, [dispatch]);

    return {
        user,
        isAuthenticated,
        preferences,
        loginUser,
        logoutUser,
        updateUserPreferences,
    };
};

// Theme hooks
export const useTheme = () => {
    const dispatch = useAppDispatch();
    const theme = useAppSelector(selectUserTheme);

    const changeTheme = useCallback((newTheme: 'light' | 'dark' | 'auto') => {
        dispatch(setTheme(newTheme));
    }, [dispatch]);

    return {
        theme,
        changeTheme,
    };
};

// Dashboard hooks
export const useDashboard = () => {
    const dispatch = useAppDispatch();
    const configuration = useAppSelector(selectDashboardConfiguration);
    const mode = useAppSelector(selectDashboardMode);
    const layout = useAppSelector(selectDashboardLayout);

    const changeDashboardMode = useCallback((newMode: any) => {
        dispatch(setDashboardMode(newMode));
    }, [dispatch]);

    const changeDashboardLayout = useCallback((newLayout: any) => {
        dispatch(setDashboardLayout(newLayout));
    }, [dispatch]);

    return {
        configuration,
        mode,
        layout,
        changeDashboardMode,
        changeDashboardLayout,
    };
};

// Symbol selection hooks
export const useSymbolSelection = () => {
    const dispatch = useAppDispatch();
    const selectedSymbol = useAppSelector(selectSelectedSymbol);
    const lastActiveSymbol = useAppSelector(selectLastActiveSymbol);
    const recentSymbols = useAppSelector(selectRecentSymbols);

    const selectSymbol = useCallback((symbol: string) => {
        dispatch(setSelectedSymbol(symbol));
        dispatch(setLastActiveSymbol(symbol));
    }, [dispatch]);

    return {
        selectedSymbol,
        lastActiveSymbol,
        recentSymbols,
        selectSymbol,
    };
};

// UI state hooks
export const useUIState = () => {
    const dispatch = useAppDispatch();
    const sidebarOpen = useAppSelector(selectSidebarOpen);
    const modals = useAppSelector(selectModals);
    const loadingStates = useAppSelector(selectLoadingStates);
    const connectionStatus = useAppSelector(selectConnectionStatus);

    const toggleSidebarOpen = useCallback(() => {
        dispatch(toggleSidebar());
    }, [dispatch]);

    const setSidebarState = useCallback((open: boolean) => {
        dispatch(setSidebarOpen(open));
    }, [dispatch]);

    const openModalDialog = useCallback((modal: any) => {
        dispatch(openModal(modal));
    }, [dispatch]);

    const closeModalDialog = useCallback((modal: any) => {
        dispatch(closeModal(modal));
    }, [dispatch]);

    const setLoading = useCallback((key: any, loading: boolean) => {
        dispatch(setLoadingState({ key, loading }));
    }, [dispatch]);

    const updateConnectionStatus = useCallback((type: any, status: any) => {
        dispatch(setConnectionStatus({ type, status }));
    }, [dispatch]);

    return {
        sidebarOpen,
        modals,
        loadingStates,
        connectionStatus,
        toggleSidebarOpen,
        setSidebarState,
        openModalDialog,
        closeModalDialog,
        setLoading,
        updateConnectionStatus,
    };
};

// Search and filters hooks
export const useSearchAndFilters = () => {
    const dispatch = useAppDispatch();
    const searchQuery = useAppSelector(selectSearchQuery);
    const activeFilters = useAppSelector(selectActiveFilters);

    const setSearch = useCallback((query: string) => {
        dispatch(setSearchQuery(query));
    }, [dispatch]);

    const updateFilterSettings = useCallback((filters: any) => {
        dispatch(updateFilters(filters));
    }, [dispatch]);

    const clearAllFilters = useCallback(() => {
        dispatch(clearFilters());
    }, [dispatch]);

    return {
        searchQuery,
        activeFilters,
        setSearch,
        updateFilterSettings,
        clearAllFilters,
    };
};

// Notifications hooks
export const useNotifications = () => {
    const dispatch = useAppDispatch();
    const notifications = useAppSelector(selectUINotifications);

    const showNotification = useCallback((notification: any) => {
        dispatch(addNotification(notification));
    }, [dispatch]);

    const hideNotification = useCallback((id: string) => {
        dispatch(removeNotification(id));
    }, [dispatch]);

    const showSuccess = useCallback((message: string, title = 'Success') => {
        dispatch(addNotification({ type: 'success', title, message }));
    }, [dispatch]);

    const showError = useCallback((message: string, title = 'Error') => {
        dispatch(addNotification({ type: 'error', title, message }));
    }, [dispatch]);

    const showWarning = useCallback((message: string, title = 'Warning') => {
        dispatch(addNotification({ type: 'warning', title, message }));
    }, [dispatch]);

    const showInfo = useCallback((message: string, title = 'Info') => {
        dispatch(addNotification({ type: 'info', title, message }));
    }, [dispatch]);

    return {
        notifications,
        showNotification,
        hideNotification,
        showSuccess,
        showError,
        showWarning,
        showInfo,
    };
};

// AI configuration hooks
export const useAIConfiguration = () => {
    const dispatch = useAppDispatch();
    const aiConfiguration = useAppSelector(selectAIConfiguration);
    const aiSettings = useAppSelector(selectAISettings);

    const updateAIConfiguration = useCallback((settings: any) => {
        dispatch(updateAISettings(settings));
    }, [dispatch]);

    return {
        aiConfiguration,
        aiSettings,
        updateAIConfiguration,
    };
};

// Chart configuration hooks
export const useChartConfiguration = () => {
    const dispatch = useAppDispatch();
    const chartSettings = useAppSelector(selectChartSettings);
    const selectedTimeframe = useAppSelector(selectSelectedTimeframe);

    const updateChartConfiguration = useCallback((settings: any) => {
        dispatch(updateChartSettings(settings));
    }, [dispatch]);

    const changeTimeframe = useCallback((timeframe: string) => {
        dispatch(setSelectedTimeframe(timeframe));
    }, [dispatch]);

    return {
        chartSettings,
        selectedTimeframe,
        updateChartConfiguration,
        changeTimeframe,
    };
};

// Feature flags hooks
export const useFeatureFlags = () => {
    const dispatch = useAppDispatch();
    const featureFlags = useAppSelector(selectFeatureFlags);

    const toggleFeature = useCallback((flag: any) => {
        dispatch(toggleFeatureFlag(flag));
    }, [dispatch]);

    return {
        featureFlags,
        toggleFeature,
    };
};

// Application state hooks
export const useApplicationState = () => {
    const applicationState = useAppSelector(selectApplicationState);

    return applicationState;
};

// Performance monitoring hooks
export const usePerformanceMonitoring = () => {
    const dispatch = useAppDispatch();

    const recordPerformanceMetric = useCallback((metrics: any) => {
        dispatch(updatePerformanceMetrics(metrics));
    }, [dispatch]);

    return {
        recordPerformanceMetric,
    };
}; 