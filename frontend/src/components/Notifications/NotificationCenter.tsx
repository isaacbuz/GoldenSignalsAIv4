import React, { useState } from 'react';
import {
  Box,
  IconButton,
  Badge,
  Popover,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Button,
  Divider,
  Chip,
  Tab,
  Tabs,
} from '@mui/material';
import {
  Notifications,
  CheckCircle,
  Error,
  Warning,
  Info,
  Clear,
  Settings,
  AutoAwesome,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { useNotifications } from './NotificationProvider';
import { formatDistanceToNow } from 'date-fns';

const NotificationPopover = styled(Popover)(({ theme }) => ({
  '& .MuiPaper-root': {
    width: 400,
    maxHeight: 600,
    backgroundColor: '#0A0E27',
    border: '1px solid rgba(255, 215, 0, 0.2)',
    backgroundImage: 'none',
  },
}));

const NotificationCenter: React.FC = () => {
  const { notifications, removeNotification, clearAll } = useNotifications();
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [tab, setTab] = useState(0);

  const unreadCount = notifications.filter(n => !n.read).length;

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const open = Boolean(anchorEl);

  const getIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle sx={{ color: '#4CAF50' }} />;
      case 'error':
        return <Error sx={{ color: '#F44336' }} />;
      case 'warning':
        return <Warning sx={{ color: '#FFA500' }} />;
      case 'signal':
        return <AutoAwesome sx={{ color: '#FFD700' }} />;
      default:
        return <Info sx={{ color: '#2196F3' }} />;
    }
  };

  const filteredNotifications = tab === 0 
    ? notifications 
    : notifications.filter(n => n.type === 'signal');

  return (
    <>
      <IconButton color="inherit" onClick={handleClick}>
        <Badge badgeContent={unreadCount} color="error">
          <Notifications />
        </Badge>
      </IconButton>

      <NotificationPopover
        open={open}
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
      >
        <Box>
          <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h6">Notifications</Typography>
            <Box>
              <IconButton size="small" onClick={clearAll}>
                <Clear />
              </IconButton>
              <IconButton size="small">
                <Settings />
              </IconButton>
            </Box>
          </Box>

          <Divider />

          <Tabs
            value={tab}
            onChange={(_, newValue) => setTab(newValue)}
            sx={{ borderBottom: 1, borderColor: 'divider' }}
          >
            <Tab label="All" />
            <Tab label="Signals" />
          </Tabs>

          <List sx={{ maxHeight: 400, overflow: 'auto' }}>
            {filteredNotifications.length === 0 ? (
              <Box sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  No notifications
                </Typography>
              </Box>
            ) : (
              filteredNotifications.map((notification) => (
                <ListItem
                  key={notification.id}
                  sx={{
                    '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.05)' },
                  }}
                >
                  <ListItemIcon>
                    {getIcon(notification.type)}
                  </ListItemIcon>
                  <ListItemText
                    primary={notification.title || notification.message}
                    secondary={
                      <Box>
                        {notification.title && (
                          <Typography variant="caption" component="p">
                            {notification.message}
                          </Typography>
                        )}
                        <Typography variant="caption" color="text.secondary">
                          {formatDistanceToNow(notification.timestamp, { addSuffix: true })}
                        </Typography>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton
                      edge="end"
                      size="small"
                      onClick={() => removeNotification(notification.id)}
                    >
                      <Clear fontSize="small" />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))
            )}
          </List>

          {filteredNotifications.length > 0 && (
            <>
              <Divider />
              <Box sx={{ p: 1, textAlign: 'center' }}>
                <Button size="small" onClick={clearAll}>
                  Clear All
                </Button>
              </Box>
            </>
          )}
        </Box>
      </NotificationPopover>
    </>
  );
};

export default NotificationCenter;
