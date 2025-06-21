/**
 * Smart Notifications Service
 * 
 * Features:
 * - Real-time signal notifications
 * - Priority-based alerts
 * - Sound notifications for high-confidence signals
 * - Browser notifications API
 * - Notification preferences
 */

import { Signal } from './api';
import { useAppStore } from '../store';

export interface NotificationPreferences {
    enabled: boolean;
    sound: boolean;
    desktop: boolean;
    minConfidence: number;
    signalTypes: string[];
    favoriteSymbolsOnly: boolean;
}

export interface SmartNotification {
    id: string;
    type: 'signal' | 'alert' | 'system' | 'agent';
    priority: 'critical' | 'high' | 'medium' | 'low';
    title: string;
    message: string;
    data?: any;
    timestamp: Date;
    read: boolean;
    actionUrl?: string;
}

class SmartNotificationService {
    private preferences: NotificationPreferences;
    private audioContext: AudioContext | null = null;
    private notificationSound: HTMLAudioElement | null = null;
    private criticalSound: HTMLAudioElement | null = null;

    constructor() {
        // Load preferences from localStorage
        this.preferences = this.loadPreferences();

        // Initialize audio
        this.initializeAudio();

        // Request notification permission
        this.requestPermission();
    }

    private loadPreferences(): NotificationPreferences {
        const saved = localStorage.getItem('notificationPreferences');
        if (saved) {
            return JSON.parse(saved);
        }

        return {
            enabled: true,
            sound: true,
            desktop: true,
            minConfidence: 80,
            signalTypes: ['CALL', 'PUT'],
            favoriteSymbolsOnly: false,
        };
    }

    private savePreferences() {
        localStorage.setItem('notificationPreferences', JSON.stringify(this.preferences));
    }

    private initializeAudio() {
        // Create audio elements for different notification types
        this.notificationSound = new Audio('/sounds/notification.mp3');
        this.criticalSound = new Audio('/sounds/critical.mp3');

        // Set volumes
        if (this.notificationSound) this.notificationSound.volume = 0.5;
        if (this.criticalSound) this.criticalSound.volume = 0.7;
    }

    private async requestPermission() {
        if ('Notification' in window && Notification.permission === 'default') {
            await Notification.requestPermission();
        }
    }

    public updatePreferences(preferences: Partial<NotificationPreferences>) {
        this.preferences = { ...this.preferences, ...preferences };
        this.savePreferences();
    }

    public getPreferences(): NotificationPreferences {
        return { ...this.preferences };
    }

    public async notifySignal(signal: Signal) {
        if (!this.preferences.enabled) return;

        // Check confidence threshold
        if (signal.confidence < this.preferences.minConfidence) return;

        // Check signal type filter
        if (this.preferences.signalTypes.length > 0 &&
            !this.preferences.signalTypes.includes(signal.type || signal.signal_type)) {
            return;
        }

        // Check favorite symbols filter
        if (this.preferences.favoriteSymbolsOnly) {
            const favorites = JSON.parse(localStorage.getItem('favoriteSymbols') || '[]');
            if (!favorites.includes(signal.symbol)) return;
        }

        // Determine priority based on confidence
        const priority = this.getSignalPriority(signal);

        // Create notification
        const notification: SmartNotification = {
            id: `signal-${Date.now()}`,
            type: 'signal',
            priority,
            title: `${signal.symbol} Signal Alert`,
            message: this.formatSignalMessage(signal),
            data: signal,
            timestamp: new Date(),
            read: false,
            actionUrl: `/signals/${signal.symbol}`,
        };

        // Store notification
        this.storeNotification(notification);

        // Play sound
        if (this.preferences.sound) {
            this.playNotificationSound(priority);
        }

        // Show desktop notification
        if (this.preferences.desktop && 'Notification' in window && Notification.permission === 'granted') {
            this.showDesktopNotification(notification);
        }

        // Update app store
        const { addNotification } = useAppStore.getState();
        addNotification({
            id: notification.id,
            type: priority === 'critical' ? 'error' : priority === 'high' ? 'warning' : 'info',
            title: notification.title,
            message: notification.message,
            timestamp: notification.timestamp,
        });
    }

    private getSignalPriority(signal: Signal): 'critical' | 'high' | 'medium' | 'low' {
        if (signal.confidence >= 95) return 'critical';
        if (signal.confidence >= 85) return 'high';
        if (signal.confidence >= 75) return 'medium';
        return 'low';
    }

    private formatSignalMessage(signal: Signal): string {
        const type = signal.type || signal.signal_type;
        const action = type === 'BUY_CALL' || type === 'CALL' ? '📈 CALL' : '📉 PUT';

        let message = `${action} opportunity detected\n`;
        message += `Strike: $${signal.strike_price}\n`;
        message += `Confidence: ${signal.confidence}%\n`;

        if (signal.expiration_date) {
            const days = Math.ceil((new Date(signal.expiration_date).getTime() - Date.now()) / (1000 * 60 * 60 * 24));
            message += `Expires in ${days} days`;
        }

        return message;
    }

    private playNotificationSound(priority: 'critical' | 'high' | 'medium' | 'low') {
        try {
            if (priority === 'critical' && this.criticalSound) {
                this.criticalSound.play();
            } else if (this.notificationSound) {
                this.notificationSound.play();
            }
        } catch (error) {
            console.error('Error playing notification sound:', error);
        }
    }

    private showDesktopNotification(notification: SmartNotification) {
        const icon = notification.priority === 'critical' ? '🚨' :
            notification.priority === 'high' ? '⚠️' :
                notification.priority === 'medium' ? '📊' : 'ℹ️';

        const desktopNotification = new Notification(notification.title, {
            body: notification.message,
            icon: '/icon-192x192.png',
            badge: '/icon-192x192.png',
            tag: notification.id,
            requireInteraction: notification.priority === 'critical',
            silent: !this.preferences.sound,
            data: notification.data,
        });

        desktopNotification.onclick = () => {
            window.focus();
            if (notification.actionUrl) {
                window.location.href = notification.actionUrl;
            }
            desktopNotification.close();
        };
    }

    private storeNotification(notification: SmartNotification) {
        // Get existing notifications
        const stored = localStorage.getItem('notifications');
        const notifications: SmartNotification[] = stored ? JSON.parse(stored) : [];

        // Add new notification
        notifications.unshift(notification);

        // Keep only last 100 notifications
        if (notifications.length > 100) {
            notifications.splice(100);
        }

        // Save back to localStorage
        localStorage.setItem('notifications', JSON.stringify(notifications));
    }

    public getNotifications(): SmartNotification[] {
        const stored = localStorage.getItem('notifications');
        return stored ? JSON.parse(stored) : [];
    }

    public markAsRead(notificationId: string) {
        const notifications = this.getNotifications();
        const notification = notifications.find(n => n.id === notificationId);
        if (notification) {
            notification.read = true;
            localStorage.setItem('notifications', JSON.stringify(notifications));
        }
    }

    public clearNotifications() {
        localStorage.removeItem('notifications');
    }

    // Agent status notifications
    public notifyAgentStatus(agentName: string, status: 'active' | 'error' | 'warning', message: string) {
        if (!this.preferences.enabled) return;

        const priority = status === 'error' ? 'high' : status === 'warning' ? 'medium' : 'low';

        const notification: SmartNotification = {
            id: `agent-${Date.now()}`,
            type: 'agent',
            priority,
            title: `Agent Alert: ${agentName}`,
            message,
            timestamp: new Date(),
            read: false,
        };

        this.storeNotification(notification);

        if (priority === 'high' && this.preferences.sound) {
            this.playNotificationSound(priority);
        }

        if (this.preferences.desktop && priority !== 'low') {
            this.showDesktopNotification(notification);
        }
    }

    // System alerts
    public notifySystem(title: string, message: string, priority: 'high' | 'medium' | 'low' = 'medium') {
        if (!this.preferences.enabled) return;

        const notification: SmartNotification = {
            id: `system-${Date.now()}`,
            type: 'system',
            priority,
            title,
            message,
            timestamp: new Date(),
            read: false,
        };

        this.storeNotification(notification);

        if (priority === 'high') {
            if (this.preferences.sound) {
                this.playNotificationSound(priority);
            }

            if (this.preferences.desktop) {
                this.showDesktopNotification(notification);
            }
        }
    }
}

// Create singleton instance
export const smartNotifications = new SmartNotificationService();

// React hook for notifications
export const useSmartNotifications = () => {
    const [notifications, setNotifications] = React.useState<SmartNotification[]>([]);
    const [preferences, setPreferences] = React.useState<NotificationPreferences>(
        smartNotifications.getPreferences()
    );

    React.useEffect(() => {
        // Load initial notifications
        setNotifications(smartNotifications.getNotifications());

        // Poll for updates
        const interval = setInterval(() => {
            setNotifications(smartNotifications.getNotifications());
        }, 1000);

        return () => clearInterval(interval);
    }, []);

    const updatePreferences = (prefs: Partial<NotificationPreferences>) => {
        smartNotifications.updatePreferences(prefs);
        setPreferences(smartNotifications.getPreferences());
    };

    const markAsRead = (notificationId: string) => {
        smartNotifications.markAsRead(notificationId);
        setNotifications(smartNotifications.getNotifications());
    };

    const clearAll = () => {
        smartNotifications.clearNotifications();
        setNotifications([]);
    };

    const unreadCount = notifications.filter(n => !n.read).length;

    return {
        notifications,
        preferences,
        unreadCount,
        updatePreferences,
        markAsRead,
        clearAll,
    };
};

export default smartNotifications; 