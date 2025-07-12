// Export from the main store configuration
export * from './store';

// Re-export the app slice actions
export * from './appSlice';

// Export the store as default
export { store as default } from './store';

// Add missing useSignals hook for backward compatibility
import { useAppSelector } from './store';

export const useSignals = () => {
    // Mock signals state for now - in a real app this would come from a signals slice
    const signals = useAppSelector((state) => state.app.notifications); // Use notifications as placeholder

    return {
        signals: signals || [],
        loading: false,
        error: null,
    };
}; 