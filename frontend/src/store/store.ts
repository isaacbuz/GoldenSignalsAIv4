import { configureStore } from '@reduxjs/toolkit';
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';
import appReducer from './appSlice';
import userReducer from './slices/userSlice';
import uiReducer from './slices/uiSlice';

// Create the root reducer with actual reducers
const rootReducer = {
    app: appReducer,
    user: userReducer,
    ui: uiReducer,
};

export const store = configureStore({
    reducer: rootReducer,
    middleware: (getDefaultMiddleware) =>
        getDefaultMiddleware({
            serializableCheck: {
                // Ignore these action types for serialization checks
                ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
            },
        }),
});

// Infer the `RootState` and `AppDispatch` types from the store itself
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Use throughout your app instead of plain `useDispatch` and `useSelector`
export const useAppDispatch: () => AppDispatch = useDispatch;
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;

// Custom hook for settings page compatibility
export const useAppStore = () => {
    const dispatch = useAppDispatch();

    // Mock settings state - in a real app, this would come from a settings slice
    const settings = {
        apiEndpoint: 'http://localhost:8000/api/v1',
        refreshInterval: 5000,
    };

    // Mock updateSettings function
    const updateSettings = (newSettings: Partial<typeof settings>) => {
        // In a real app, this would dispatch an action to update the settings
        // Settings updated silently in production
        // For now, we'll just log it
    };

    return { settings, updateSettings };
};
