import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import logger from '../services/logger';


export interface Notification {
    id: string;
    type: 'success' | 'error' | 'warning' | 'info';
    message: string;
    timestamp: Date;
}

interface NotificationsContextType {
    notifications: Notification[];
    addNotification: (type: Notification['type'], message: string) => void;
    removeNotification: (id: string) => void;
    clearNotifications: () => void;
}

const NotificationsContext = createContext<NotificationsContextType | undefined>(undefined);

export const useNotifications = () => {
    const context = useContext(NotificationsContext);
    if (!context) {
        // Return a mock implementation if no provider is found
        return {
            notifications: [],
            addNotification: (type: Notification['type'], message: string) => {
                logger.info(`${type.toUpperCase()}: ${message}`);
            },
            removeNotification: () => { },
            clearNotifications: () => { },
        };
    }
    return context;
};

interface NotificationsProviderProps {
    children: ReactNode;
}

export const NotificationsProvider: React.FC<NotificationsProviderProps> = ({ children }) => {
    const [notifications, setNotifications] = useState<Notification[]>([]);

    const addNotification = useCallback((type: Notification['type'], message: string) => {
        const id = Date.now().toString();
        const notification: Notification = {
            id,
            type,
            message,
            timestamp: new Date(),
        };

        setNotifications(prev => [...prev, notification]);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            setNotifications(prev => prev.filter(n => n.id !== id));
        }, 5000);
    }, []);

    const removeNotification = useCallback((id: string) => {
        setNotifications(prev => prev.filter(n => n.id !== id));
    }, []);

    const clearNotifications = useCallback(() => {
        setNotifications([]);
    }, []);

    const value: NotificationsContextType = {
        notifications,
        addNotification,
        removeNotification,
        clearNotifications,
    };

    return (
        <NotificationsContext.Provider value={value}>
            {children}
        </NotificationsContext.Provider>
    );
};
