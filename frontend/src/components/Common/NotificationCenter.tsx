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
} from '@mui/material';
import {
  Notifications,
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

interface NotificationCenterProps {
  open: boolean;
  onClose: () => void;
}

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

export default function NotificationCenter({ open, onClose }: NotificationCenterProps) {
  const theme = useTheme();
  const { notifications, removeNotification } = useNotifications();
  const [tabValue, setTabValue] = useState(0);
  const [filter, setFilter] = useState<'all' | 'unread' | 'signals' | 'system'>('all');

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

  const filteredNotifications = filterNotifications(notifications);
  const groupedNotifications = groupNotificationsByDate(filteredNotifications);
  const unreadCount = notifications.filter(n => !n.read).length;

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      PaperProps={{
        sx: {
          width: 400,
          maxWidth: '90vw',
        },
      }}
    >
      <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" component="div">
              Notifications
              {unreadCount > 0 && (
                <Badge badgeContent={unreadCount} color="error" sx={{ ml: 1 }}>
                  <Box />
                </Badge>
              )}
            </Typography>
            <IconButton onClick={onClose} size="small">
              <Close />
            </IconButton>
          </Box>

          {/* Action Buttons */}
          <Stack direction="row" spacing={1}>
            <Button
              size="small"
              startIcon={<MarkEmailRead />}
              onClick={handleMarkAllRead}
              disabled={unreadCount === 0}
            >
              Mark All Read
            </Button>
            <Button
              size="small"
              startIcon={<Delete />}
              onClick={handleClearAll}
              disabled={notifications.length === 0}
              color="error"
            >
              Clear All
            </Button>
          </Stack>
        </Box>

        {/* Filter Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab label="All" />
            <Tab label={`Unread (${unreadCount})`} />
            <Tab label="Signals" />
            <Tab label="System" />
          </Tabs>
        </Box>

        {/* Notifications List */}
        <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
          <TabPanel value={tabValue} index={0}>
            {Object.keys(groupedNotifications).length === 0 ? (
              <Box sx={{ p: 3, textAlign: 'center' }}>
                <Notifications sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                <Typography variant="body1" color="text.secondary">
                  No notifications
                </Typography>
              </Box>
            ) : (
              Object.entries(groupedNotifications).map(([date, dateNotifications]) => (
                <Box key={date}>
                  <Typography
                    variant="subtitle2"
                    sx={{
                      p: 2,
                      pb: 1,
                      color: 'text.secondary',
                      fontWeight: 600,
                      fontSize: '0.75rem',
                      textTransform: 'uppercase',
                    }}
                  >
                    {date}
                  </Typography>
                  <List disablePadding>
                    {dateNotifications.map((notification) => (
                      <ListItem
                        key={notification.id}
                        sx={{
                          borderLeft: 4,
                          borderLeftColor: getNotificationColor(notification.type),
                          bgcolor: notification.read ? 'transparent' : alpha(getNotificationColor(notification.type), 0.05),
                          '&:hover': {
                            bgcolor: alpha(theme.palette.action.hover, 0.5),
                          },
                        }}
                      >
                        <ListItemIcon>
                          <Avatar
                            sx={{
                              width: 32,
                              height: 32,
                              bgcolor: alpha(getNotificationColor(notification.type), 0.1),
                              color: getNotificationColor(notification.type),
                            }}
                          >
                            {getCategoryIcon(notification.title)}
                          </Avatar>
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography
                                variant="body2"
                                sx={{
                                  fontWeight: notification.read ? 400 : 600,
                                  flexGrow: 1,
                                }}
                              >
                                {notification.title}
                              </Typography>
                              {!notification.read && (
                                <Box
                                  sx={{
                                    width: 8,
                                    height: 8,
                                    borderRadius: '50%',
                                    bgcolor: getNotificationColor(notification.type),
                                  }}
                                />
                              )}
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                                {notification.message}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {formatDistanceToNow(notification.timestamp, { addSuffix: true })}
                              </Typography>
                            </Box>
                          }
                        />
                        <ListItemSecondaryAction>
                          <Stack direction="column" spacing={1} alignItems="flex-end">
                            <IconButton
                              size="small"
                              onClick={() => removeNotification(notification.id)}
                            >
                              <Close fontSize="small" />
                            </IconButton>
                            {notification.action && (
                              <Button
                                size="small"
                                variant="outlined"
                                onClick={notification.action.callback}
                                sx={{ fontSize: '0.75rem' }}
                              >
                                {notification.action.label}
                              </Button>
                            )}
                          </Stack>
                        </ListItemSecondaryAction>
                      </ListItem>
                    ))}
                  </List>
                  <Divider />
                </Box>
              ))
            )}
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            {filterNotifications(notifications.filter(n => !n.read)).length === 0 ? (
              <Box sx={{ p: 3, textAlign: 'center' }}>
                <CheckCircle sx={{ fontSize: 48, color: 'success.main', mb: 2 }} />
                <Typography variant="body1" color="text.secondary">
                  All caught up!
                </Typography>
              </Box>
            ) : (
              <List>
                {filterNotifications(notifications.filter(n => !n.read)).map((notification) => (
                  <ListItem key={notification.id}>
                    {/* Similar structure as above */}
                  </ListItem>
                ))}
              </List>
            )}
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            {filterNotifications(notifications.filter(n => n.title.toLowerCase().includes('signal'))).length === 0 ? (
              <Box sx={{ p: 3, textAlign: 'center' }}>
                <TrendingUp sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                <Typography variant="body1" color="text.secondary">
                  No signal notifications
                </Typography>
              </Box>
            ) : (
              <List>
                {filterNotifications(notifications.filter(n => n.title.toLowerCase().includes('signal'))).map((notification) => (
                  <ListItem key={notification.id}>
                    {/* Similar structure as above */}
                  </ListItem>
                ))}
              </List>
            )}
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            {filterNotifications(notifications.filter(n => 
              n.title.toLowerCase().includes('system') || 
              n.title.toLowerCase().includes('connection') ||
              n.title.toLowerCase().includes('error')
            )).length === 0 ? (
              <Box sx={{ p: 3, textAlign: 'center' }}>
                <Settings sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                <Typography variant="body1" color="text.secondary">
                  No system notifications
                </Typography>
              </Box>
            ) : (
              <List>
                {filterNotifications(notifications.filter(n => 
                  n.title.toLowerCase().includes('system') || 
                  n.title.toLowerCase().includes('connection') ||
                  n.title.toLowerCase().includes('error')
                )).map((notification) => (
                  <ListItem key={notification.id}>
                    {/* Similar structure as above */}
                  </ListItem>
                ))}
              </List>
            )}
          </TabPanel>
        </Box>
      </Box>
    </Drawer>
  );
} 