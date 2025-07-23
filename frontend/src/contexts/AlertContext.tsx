/**
 * Alert Context - Multi-channel alert management for AI signals
 * Handles sound alerts, push notifications, and in-app alerts
 */

import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { Howl } from 'howler';
import toast from 'react-hot-toast';
import logger from '../services/logger';


// Alert types
export interface Alert {
  id: string;
  type: 'CALL' | 'PUT';
  symbol: string;
  confidence: number;
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM';
  timestamp: Date;
  message: string;
  strike?: number;
  expiry?: string;
  expectedReturn?: string;
}

export interface AlertSettings {
  soundEnabled: boolean;
  pushEnabled: boolean;
  emailEnabled: boolean;
  minConfidence: number;
  criticalOnly: boolean;
  soundVolume: number;
}

interface AlertContextType {
  alerts: Alert[];
  activeAlerts: Alert[];
  alertHistory: Alert[];
  settings: AlertSettings;
  addAlert: (alert: Alert) => void;
  dismissAlert: (id: string) => void;
  clearAllAlerts: () => void;
  updateSettings: (settings: Partial<AlertSettings>) => void;
  playTestSound: () => void;
}

// Default settings
const defaultSettings: AlertSettings = {
  soundEnabled: true,
  pushEnabled: true,
  emailEnabled: false,
  minConfidence: 70,
  criticalOnly: false,
  soundVolume: 0.7
};

// Create sound instances but don't load them immediately
const createSound = (src: string) => new Howl({
  src: [src],
  preload: false,
  html5: true
});

const AlertContext = createContext<AlertContextType | null>(null);

export const useAlerts = () => {
  const context = useContext(AlertContext);
  if (!context) {
    throw new Error('useAlerts must be used within AlertProvider');
  }
  return context;
};

export const AlertProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [activeAlerts, setActiveAlerts] = useState<Alert[]>([]);
  const [alertHistory, setAlertHistory] = useState<Alert[]>([]);
  const [settings, setSettings] = useState<AlertSettings>(() => {
    // Load settings from localStorage
    const saved = localStorage.getItem('alertSettings');
    return saved ? JSON.parse(saved) : defaultSettings;
  });

  // Auto-dismiss timer refs
  const dismissTimers = useRef<Map<string, NodeJS.Timeout>>(new Map());

  // Load sounds after user interaction
  const [sounds, setSounds] = useState({
    critical: createSound('/sounds/critical.mp3'),
    high: createSound('/sounds/high.mp3'),
    medium: createSound('/sounds/medium.mp3')
  });

  // Request notification permission on mount
  useEffect(() => {
    if ('Notification' in window && settings.pushEnabled) {
      Notification.requestPermission();
    }
  }, [settings.pushEnabled]);

  // Save settings to localStorage
  useEffect(() => {
    localStorage.setItem('alertSettings', JSON.stringify(settings));
  }, [settings]);

  // Load sounds after user interaction
  useEffect(() => {
    const loadSounds = () => {
      Object.values(sounds).forEach(sound => {
        if (sound.state() !== 'loaded') {
          sound.load();
        }
      });
    };

    // Load sounds on first user interaction
    const handleInteraction = () => {
      loadSounds();
      document.removeEventListener('click', handleInteraction);
      document.removeEventListener('keydown', handleInteraction);
    };

    document.addEventListener('click', handleInteraction);
    document.addEventListener('keydown', handleInteraction);

    return () => {
      document.removeEventListener('click', handleInteraction);
      document.removeEventListener('keydown', handleInteraction);
    };
  }, []); // Empty dependency array since we only want this to run once

  // Play alert sound
  const playAlertSound = (severity: Alert['priority']) => {
    if (!settings.soundEnabled) return;

    try {
      const sound = sounds[severity];
      if (sound && sound.state() === 'loaded') {
        sound.volume(settings.soundVolume);
        sound.play();
      }
    } catch (error) {
      logger.error('Error playing alert sound:', error);
    }
  };

  // Show push notification
  const showPushNotification = useCallback((alert: Alert) => {
    if (!settings.pushEnabled || !('Notification' in window)) return;

    if (Notification.permission === 'granted') {
      const notification = new Notification(`🎯 AI Signal: ${alert.symbol} ${alert.type}`, {
        body: `${alert.confidence}% confidence - ${alert.message}`,
        icon: '/icon-192.png',
        badge: '/icon-192.png',
        tag: alert.id,
        requireInteraction: alert.priority === 'CRITICAL'
      });

      notification.onclick = () => {
        window.focus();
        notification.close();
      };
    }
  }, [settings.pushEnabled]);

  // Add alert
  const addAlert = useCallback((alert: Alert) => {
    // Check if alert meets minimum confidence threshold
    if (settings.criticalOnly && alert.priority !== 'CRITICAL') return;
    if (alert.confidence < settings.minConfidence) return;

    // Add to alerts
    setAlerts(prev => [alert, ...prev]);
    setActiveAlerts(prev => [alert, ...prev]);
    setAlertHistory(prev => [alert, ...prev].slice(0, 100)); // Keep last 100

    // Play sound
    playAlertSound(alert.priority);

    // Show push notification
    showPushNotification(alert);

    // Show toast
    const toastId = toast.custom((t) => (
      <div className={`${t.visible ? 'animate-enter' : 'animate-leave'} w-full`}>
        <div
          className={`
            bg-gray-900 border-2 rounded-lg p-4 shadow-xl
            ${alert.type === 'CALL' ? 'border-green-500' : 'border-red-500'}
          `}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <span className="text-2xl">
                {alert.priority === 'CRITICAL' ? '🚨' : alert.priority === 'HIGH' ? '⚡' : '💡'}
              </span>
              <div>
                <h4 className="font-bold text-white">
                  {alert.symbol} {alert.type} - {alert.confidence}%
                </h4>
                <p className="text-gray-300 text-sm">{alert.message}</p>
              </div>
            </div>
            <button
              onClick={() => toast.dismiss(toastId)}
              className="text-gray-400 hover:text-white"
            >
              ✕
            </button>
          </div>
        </div>
      </div>
    ), {
      duration: alert.priority === 'CRITICAL' ? 30000 : 10000,
      position: 'top-right'
    });

    // Auto-dismiss non-critical alerts
    if (alert.priority !== 'CRITICAL') {
      const timerId = setTimeout(() => {
        dismissAlert(alert.id);
      }, 30000); // 30 seconds

      dismissTimers.current.set(alert.id, timerId);
    }
  }, [settings, playAlertSound, showPushNotification]);

  // Dismiss alert
  const dismissAlert = useCallback((id: string) => {
    setActiveAlerts(prev => prev.filter(a => a.id !== id));

    // Clear auto-dismiss timer if exists
    const timerId = dismissTimers.current.get(id);
    if (timerId) {
      clearTimeout(timerId);
      dismissTimers.current.delete(id);
    }
  }, []);

  // Clear all alerts
  const clearAllAlerts = useCallback(() => {
    setActiveAlerts([]);

    // Clear all timers
    dismissTimers.current.forEach(timerId => clearTimeout(timerId));
    dismissTimers.current.clear();
  }, []);

  // Update settings
  const updateSettings = useCallback((newSettings: Partial<AlertSettings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  }, []);

  // Play test sound
  const playTestSound = useCallback(() => {
    playAlertSound('HIGH');
  }, [playAlertSound]);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      dismissTimers.current.forEach(timerId => clearTimeout(timerId));
    };
  }, []);

  const value: AlertContextType = {
    alerts,
    activeAlerts,
    alertHistory,
    settings,
    addAlert,
    dismissAlert,
    clearAllAlerts,
    updateSettings,
    playTestSound
  };

  return (
    <AlertContext.Provider value={value}>
      {children}
    </AlertContext.Provider>
  );
};
