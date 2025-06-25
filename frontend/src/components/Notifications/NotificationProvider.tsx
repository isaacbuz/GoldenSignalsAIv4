import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { Snackbar, Alert, AlertTitle, Slide, IconButton, Box, Typography } from '@mui/material';
import { Close, CheckCircle, Error, Warning, Info } from '@mui/icons-material';
import { TransitionProps } from '@mui/material/transitions';
import { useWebSocket, WebSocketTopic } from '../../services/websocket/SignalWebSocketManager';

interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info' | 'signal';
  title?: string;
  message: string;
  duration?: number;
  timestamp: Date;
  data?: any;
}

interface NotificationContextType {
  notifications: Notification[];
  showNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
  clearAll: () => void;
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within NotificationProvider');
  }
  return context;
};

function SlideTransition(props: TransitionProps) {
  return <Slide {...props} direction="up" />;
}

export const NotificationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [openNotifications, setOpenNotifications] = useState<Set<string>>(new Set());

  // Subscribe to WebSocket alerts
  useWebSocket(WebSocketTopic.ALERTS_USER, (alert) => {
    showNotification({
      type: alert.severity || 'info',
      title: alert.title,
      message: alert.message,
      data: alert,
    });
  }, 'notification-provider');

  const showNotification = useCallback((notification: Omit<Notification, 'id' | 'timestamp'>) => {
    const id = `notif-${Date.now()}-${Math.random()}`;
    const newNotification: Notification = {
      ...notification,
      id,
      timestamp: new Date(),
      duration: notification.duration || 5000,
    };

    setNotifications(prev => [...prev, newNotification]);
    setOpenNotifications(prev => new Set(prev).add(id));

    // Auto-hide after duration
    if (newNotification.duration && newNotification.duration > 0) {
      setTimeout(() => {
        setOpenNotifications(prev => {
          const newSet = new Set(prev);
          newSet.delete(id);
          return newSet;
        });
      }, newNotification.duration);
    }

    // Request browser notification permission
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification(notification.title || 'GoldenSignals AI', {
        body: notification.message,
        icon: '/favicon.ico',
        tag: id,
      });
    }
  }, []);

  const removeNotification = useCallback((id: string) => {
    setOpenNotifications(prev => {
      const newSet = new Set(prev);
      newSet.delete(id);
      return newSet;
    });
    
    // Remove from list after animation
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 300);
  }, []);

  const clearAll = useCallback(() => {
    setOpenNotifications(new Set());
    setTimeout(() => {
      setNotifications([]);
    }, 300);
  }, []);

  // Request notification permission on mount
  useEffect(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);

  const getIcon = (type: Notification['type']) => {
    switch (type) {
      case 'success':
        return <CheckCircle />;
      case 'error':
        return <Error />;
      case 'warning':
        return <Warning />;
      default:
        return <Info />;
    }
  };

  return (
    <NotificationContext.Provider value={{ notifications, showNotification, removeNotification, clearAll }}>
      {children}
      
      {/* Render notifications */}
      {notifications.map((notification, index) => (
        <Snackbar
          key={notification.id}
          open={openNotifications.has(notification.id)}
          autoHideDuration={notification.duration}
          onClose={() => removeNotification(notification.id)}
          TransitionComponent={SlideTransition}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
          sx={{ 
            bottom: (theme) => theme.spacing(8 + index * 10),
            zIndex: 1400 + index,
          }}
        >
          <Alert
            severity={notification.type === 'signal' ? 'info' : notification.type}
            icon={getIcon(notification.type)}
            action={
              <IconButton
                size="small"
                aria-label="close"
                color="inherit"
                onClick={() => removeNotification(notification.id)}
              >
                <Close fontSize="small" />
              </IconButton>
            }
            sx={{
              minWidth: 300,
              backgroundColor: (theme) => {
                switch (notification.type) {
                  case 'success':
                    return 'rgba(76, 175, 80, 0.1)';
                  case 'error':
                    return 'rgba(244, 67, 54, 0.1)';
                  case 'warning':
                    return 'rgba(255, 165, 0, 0.1)';
                  case 'signal':
                    return 'rgba(255, 215, 0, 0.1)';
                  default:
                    return 'rgba(33, 150, 243, 0.1)';
                }
              },
              border: '1px solid',
              borderColor: (theme) => {
                switch (notification.type) {
                  case 'success':
                    return 'rgba(76, 175, 80, 0.3)';
                  case 'error':
                    return 'rgba(244, 67, 54, 0.3)';
                  case 'warning':
                    return 'rgba(255, 165, 0, 0.3)';
                  case 'signal':
                    return 'rgba(255, 215, 0, 0.3)';
                  default:
                    return 'rgba(33, 150, 243, 0.3)';
                }
              },
            }}
          >
            {notification.title && <AlertTitle>{notification.title}</AlertTitle>}
            <Box>
              <Typography variant="body2">{notification.message}</Typography>
              {notification.data?.symbol && (
                <Typography variant="caption" sx={{ display: 'block', mt: 0.5, opacity: 0.8 }}>
                  Symbol: {notification.data.symbol}
                </Typography>
              )}
            </Box>
          </Alert>
        </Snackbar>
      ))}
    </NotificationContext.Provider>
  );
};
