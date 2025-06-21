/**
 * Notification Center Component
 * 
 * Comprehensive notification management for institutional trading platform
 */

import React, { useState } from 'react';
import {
  Box,
  Drawer,
  Typography,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  Chip,
  Button,
  Divider,
  Badge,
  Tabs,
  Tab,
  Card,
  Stack,
  Avatar,
  useTheme,
  alpha,
  Popover,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Notifications,
  NotificationsActive,
  Close,
  TrendingUp,
  TrendingDown,
  Warning,
  Info,
  Error,
  CheckCircle,
  SmartToy,
  AccountBalance,
  Settings,
  Delete,
  MarkEmailRead,
  FilterList,
} from '@mui/icons-material';
import { formatDistanceToNow } from 'date-fns';
import { useNotifications, Notification } from '../../store';
import { motion, AnimatePresence } from 'framer-motion';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`notification-tabpanel-${index}`}
      aria-labelledby={`notification-tab-${index}`}
      {...other}
    >
      {value === index && <Box>{children}</Box>}
    </div>
  );
}

export default function NotificationCenter() {
  const theme = useTheme();
  const { notifications, removeNotification } = useNotifications();
  const [tabValue, setTabValue] = useState(0);
  const [filter, setFilter] = useState<'all' | 'unread' | 'signals' | 'system'>('all');
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [notificationSettings, setNotificationSettings] = useState({
    signals: true,
    alerts: true,
    trades: true,
    news: true,
    sound: true,
    desktop: false,
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const getNotificationIcon = (type: Notification['type']) => {
    const iconProps = { fontSize: 'small' as const };
    switch (type) {
      case 'success':
        return <CheckCircle {...iconProps} color="success" />;
      case 'error':
        return <Error {...iconProps} color="error" />;
      case 'warning':
        return <Warning {...iconProps} color="warning" />;
      case 'info':
      default:
        return <Info {...iconProps} color="info" />;
    }
  };

  const getNotificationColor = (type: Notification['type']) => {
    switch (type) {
      case 'success':
        return theme.palette.success.main;
      case 'error':
        return theme.palette.error.main;
      case 'warning':
        return theme.palette.warning.main;
      case 'info':
      default:
        return theme.palette.info.main;
    }
  };

  const getCategoryIcon = (title: string) => {
    if (title.includes('Signal')) return <TrendingUp />;
    if (title.includes('Agent')) return <SmartToy />;
    if (title.includes('Portfolio')) return <AccountBalance />;
    if (title.includes('System')) return <Settings />;
    return <Info />;
  };

  const filterNotifications = (notifications: Notification[]) => {
    let filtered = notifications;

    switch (filter) {
      case 'unread':
        filtered = notifications.filter(n => !n.read);
        break;
      case 'signals':
        filtered = notifications.filter(n => n.title.toLowerCase().includes('signal'));
        break;
      case 'system':
        filtered = notifications.filter(n =>
          n.title.toLowerCase().includes('system') ||
          n.title.toLowerCase().includes('connection') ||
          n.title.toLowerCase().includes('error')
        );
        break;
      default:
        break;
    }

    return filtered;
  };

  const groupNotificationsByDate = (notifications: Notification[]) => {
    const groups: Record<string, Notification[]> = {};

    notifications.forEach(notification => {
      const date = new Date(notification.timestamp);
      const today = new Date();
      const yesterday = new Date(today);
      yesterday.setDate(yesterday.getDate() - 1);

      let groupKey: string;
      if (date.toDateString() === today.toDateString()) {
        groupKey = 'Today';
      } else if (date.toDateString() === yesterday.toDateString()) {
        groupKey = 'Yesterday';
      } else {
        groupKey = date.toLocaleDateString();
      }

      if (!groups[groupKey]) {
        groups[groupKey] = [];
      }
      groups[groupKey].push(notification);
    });

    return groups;
  };

  const handleMarkAllRead = () => {
    // Implementation would mark all notifications as read
    console.log('Mark all as read');
  };

  const handleClearAll = () => {
    notifications.forEach(notification => {
      removeNotification(notification.id);
    });
  };

  const unreadCount = notifications.filter(n => !n.read).length;

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
    setShowSettings(false);
  };

  const handleMarkAsRead = (id: string) => {
    // Implementation would mark a specific notification as read
    console.log(`Marking notification ${id} as read`);
  };

  const handleMarkAllAsRead = () => {
    // Implementation would mark all notifications as read
    console.log('Mark all as read');
  };

  const handleDelete = (id: string) => {
    removeNotification(id);
  };

  const filteredNotifications = filterNotifications(notifications);
  const groupedNotifications = groupNotificationsByDate(filteredNotifications);

  return (
    <>
      <IconButton
        onClick={handleClick}
        sx={{
          animation: unreadCount > 0 ? 'pulse 2s infinite' : 'none',
          '@keyframes pulse': {
            '0%': { transform: 'scale(1)' },
            '50%': { transform: 'scale(1.1)' },
            '100%': { transform: 'scale(1)' },
          },
        }}
      >
        <Badge badgeContent={unreadCount} color="error">
          {unreadCount > 0 ? (
            <NotificationsActive />
          ) : (
            <Notifications />
          )}
        </Badge>
      </IconButton>

      <Popover
        open={Boolean(anchorEl)}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        PaperProps={{
          sx: {
            width: 420,
            maxHeight: 600,
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
          },
        }}
      >
        <AnimatePresence mode="wait">
          {showSettings ? (
            <motion.div
              key="settings"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
            >
              <Box sx={{ p: 2 }}>
                <Stack direction="row" alignItems="center" justifyContent="space-between" mb={2}>
                  <Typography variant="h6">Notification Settings</Typography>
                  <IconButton size="small" onClick={() => setShowSettings(false)}>
                    <Close />
                  </IconButton>
                </Stack>

                <Stack spacing={1}>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    Notification Types
                  </Typography>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={notificationSettings.signals}
                        onChange={(e) => setNotificationSettings(prev => ({ ...prev, signals: e.target.checked }))}
                      />
                    }
                    label="Trading Signals"
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={notificationSettings.alerts}
                        onChange={(e) => setNotificationSettings(prev => ({ ...prev, alerts: e.target.checked }))}
                      />
                    }
                    label="Risk Alerts"
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={notificationSettings.trades}
                        onChange={(e) => setNotificationSettings(prev => ({ ...prev, trades: e.target.checked }))}
                      />
                    }
                    label="Trade Executions"
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={notificationSettings.news}
                        onChange={(e) => setNotificationSettings(prev => ({ ...prev, news: e.target.checked }))}
                      />
                    }
                    label="Market News"
                  />

                  <Divider sx={{ my: 2 }} />

                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    Delivery Methods
                  </Typography>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={notificationSettings.sound}
                        onChange={(e) => setNotificationSettings(prev => ({ ...prev, sound: e.target.checked }))}
                      />
                    }
                    label="Sound Notifications"
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={notificationSettings.desktop}
                        onChange={(e) => setNotificationSettings(prev => ({ ...prev, desktop: e.target.checked }))}
                      />
                    }
                    label="Desktop Notifications"
                  />
                </Stack>
              </Box>
            </motion.div>
          ) : (
            <motion.div
              key="notifications"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.2 }}
              style={{ display: 'flex', flexDirection: 'column', height: '100%' }}
            >
              <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                <Stack direction="row" alignItems="center" justifyContent="space-between">
                  <Typography variant="h6">Notifications</Typography>
                  <Stack direction="row" spacing={1}>
                    <IconButton size="small" onClick={() => setShowSettings(true)}>
                      <Settings fontSize="small" />
                    </IconButton>
                    {unreadCount > 0 && (
                      <Button
                        size="small"
                        startIcon={<MarkEmailRead />}
                        onClick={handleMarkAllAsRead}
                      >
                        Mark all read
                      </Button>
                    )}
                  </Stack>
                </Stack>
              </Box>

              <Tabs
                value={tabValue}
                onChange={handleTabChange}
                variant="fullWidth"
                sx={{ borderBottom: 1, borderColor: 'divider' }}
              >
                <Tab label={`All (${notifications.length})`} />
                <Tab label={`Unread (${unreadCount})`} />
                <Tab label="Signals" />
                <Tab label="System" />
              </Tabs>

              <List sx={{ flexGrow: 1, overflow: 'auto', p: 0 }}>
                {filteredNotifications.length === 0 ? (
                  <Box sx={{ p: 4, textAlign: 'center' }}>
                    <Typography variant="body2" color="text.secondary">
                      No notifications
                    </Typography>
                  </Box>
                ) : (
                  filteredNotifications.map((notification, index) => (
                    <motion.div
                      key={notification.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      <ListItem
                        sx={{
                          backgroundColor: notification.read ? 'transparent' : 'action.hover',
                          '&:hover': { backgroundColor: 'action.hover' },
                        }}
                      >
                        <ListItemIcon>{getNotificationIcon(notification.type)}</ListItemIcon>
                        <ListItemText
                          primary={
                            <Stack direction="row" alignItems="center" spacing={1}>
                              <Typography variant="subtitle2">
                                {notification.title}
                              </Typography>
                              <Chip
                                label={notification.priority || 'normal'}
                                size="small"
                                color={getNotificationColor(notification.type) as any}
                                sx={{ height: 20, fontSize: '0.7rem' }}
                              />
                            </Stack>
                          }
                          secondary={
                            <>
                              <Typography variant="body2" color="text.secondary">
                                {notification.message}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {formatDistanceToNow(notification.timestamp, { addSuffix: true })}
                              </Typography>
                            </>
                          }
                        />
                        <ListItemSecondaryAction>
                          <Stack direction="row" spacing={0.5}>
                            {!notification.read && (
                              <IconButton
                                size="small"
                                onClick={() => handleMarkAsRead(notification.id)}
                              >
                                <CheckCircle fontSize="small" />
                              </IconButton>
                            )}
                            <IconButton
                              size="small"
                              onClick={() => handleDelete(notification.id)}
                            >
                              <Delete fontSize="small" />
                            </IconButton>
                          </Stack>
                        </ListItemSecondaryAction>
                      </ListItem>
                      {notification.action && (
                        <Box sx={{ px: 2, pb: 1 }}>
                          <Button
                            size="small"
                            variant="outlined"
                            onClick={notification.action.callback}
                            fullWidth
                          >
                            {notification.action.label || 'View Details'}
                          </Button>
                        </Box>
                      )}
                      {index < filteredNotifications.length - 1 && <Divider />}
                    </motion.div>
                  ))
                )}
              </List>

              {notifications.length > 0 && (
                <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
                  <Button
                    fullWidth
                    color="error"
                    startIcon={<Delete />}
                    onClick={handleClearAll}
                  >
                    Clear All Notifications
                  </Button>
                </Box>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </Popover>
    </>
  );
} 