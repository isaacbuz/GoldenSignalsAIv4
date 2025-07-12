import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface AppState {
    isLoading: boolean;
    error: string | null;
    notifications: Array<{
        id: string;
        message: string;
        type: 'success' | 'error' | 'warning' | 'info';
    }>;
}

const initialState: AppState = {
    isLoading: false,
    error: null,
    notifications: [],
};

const appSlice = createSlice({
    name: 'app',
    initialState,
    reducers: {
        setLoading: (state, action: PayloadAction<boolean>) => {
            state.isLoading = action.payload;
        },
        setError: (state, action: PayloadAction<string | null>) => {
            state.error = action.payload;
        },
        addNotification: (state, action: PayloadAction<AppState['notifications'][0]>) => {
            state.notifications.push(action.payload);
        },
        removeNotification: (state, action: PayloadAction<string>) => {
            state.notifications = state.notifications.filter(n => n.id !== action.payload);
        },
        clearNotifications: (state) => {
            state.notifications = [];
        },
    },
});

export const {
    setLoading,
    setError,
    addNotification,
    removeNotification,
    clearNotifications
} = appSlice.actions;

export default appSlice.reducer; 