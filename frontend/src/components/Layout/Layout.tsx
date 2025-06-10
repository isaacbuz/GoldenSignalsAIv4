/**
 * Premium Layout Component - My Vision for Modern Trading Platform
 * 
 * Design Philosophy:
 * - Floating navigation that doesn't obstruct data
 * - Command center approach with quick access controls
 * - Contextual information display
 * - Elegant animations and micro-interactions
 * - Professional aesthetics with subtle depth
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Button,
  IconButton,
  Badge,
  Chip,
  Menu,
  MenuItem,
  Avatar,
  Divider,
  useTheme,
  alpha,
  Stack,
  Tooltip,
  ButtonGroup,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Analytics as AnalyticsIcon,
  SmartToy as AgentsIcon,
  TrendingUp as SignalsIcon,
  Search as SearchIcon,
  Notifications as NotificationsIcon,
  Settings as SettingsIcon,
  Refresh,
  Fullscreen,
  TrendingUp,
} from '@mui/icons-material';
import { useLocation, useNavigate, Outlet } from 'react-router-dom';

export default function Layout() {
  const theme = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'disconnected'>('connected');
  const [marketStatus, setMarketStatus] = useState<'open' | 'closed' | 'pre-market' | 'after-hours'>('open');
  const [notificationMenuAnchor, setNotificationMenuAnchor] = useState<null | HTMLElement>(null);

  const navigationItems = [
    { path: '/', label: 'Dashboard', icon: <DashboardIcon /> },
    { path: '/signals', label: 'Signals', icon: <SignalsIcon />, badge: 3 },
    { path: '/analytics', label: 'Analytics', icon: <AnalyticsIcon /> },
    { path: '/agents', label: 'Agents', icon: <AgentsIcon /> },
    { path: '/settings', label: 'Settings', icon: <SettingsIcon /> },
  ];

  const handleNotificationClick = (event: React.MouseEvent<HTMLElement>) => {
    setNotificationMenuAnchor(event.currentTarget);
  };

  const handleNotificationClose = () => {
    setNotificationMenuAnchor(null);
  };
  
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <Box sx={{ 
      display: 'flex',
      flexDirection: 'column',
      minHeight: '100vh',
      background: theme.palette.background.default,
    }}>
      <AppBar 
        position="fixed" 
        elevation={0}
        sx={{
          background: alpha(theme.palette.background.default, 0.85),
          backdropFilter: 'blur(20px)',
          borderBottom: `1px solid ${theme.palette.divider}`,
        }}
      >
        <Toolbar sx={{ justifyContent: 'space-between', px: 3 }}>
          <Stack direction="row" alignItems="center" spacing={4}>
            <Stack direction="row" alignItems="center" spacing={1.5}
              onClick={() => navigate('/')}
              sx={{ cursor: 'pointer' }}
            >
              <Box
                sx={{
                  width: 32,
                  height: 32,
                  borderRadius: '8px',
                  background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main} 100%)`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <TrendingUp sx={{ color: 'white', fontSize: 18 }} />
              </Box>
              <Typography 
                variant="h6" 
                sx={{ 
                  fontWeight: 700,
                  color: 'text.primary',
                }}
              >
                GoldenSignals
              </Typography>
            </Stack>

            <ButtonGroup variant="text" size="small">
              {navigationItems.map((item) => {
                const isActive = location.pathname === item.path;
                return (
                  <Button
                    key={item.path}
                    onClick={() => navigate(item.path)}
                    sx={{
                      color: isActive ? 'primary.main' : 'text.secondary',
                      fontWeight: isActive ? 600 : 500,
                    }}
                  >
                    {item.label}
                  </Button>
                );
              })}
            </ButtonGroup>
          </Stack>

          <Stack direction="row" alignItems="center" spacing={2}>
            <Chip
              icon={<TrendingUp sx={{ fontSize: 12, color: marketStatus === 'open' ? 'success.main' : 'warning.main' }} />}
              label={marketStatus === 'open' ? 'Market Open' : 'Market Closed'}
              size="small"
              variant="outlined"
            />
            
            <Divider orientation="vertical" flexItem />

            <IconButton size="small"><SearchIcon fontSize="small" /></IconButton>
            <IconButton size="small"><Refresh fontSize="small" /></IconButton>
            <IconButton size="small"><Fullscreen fontSize="small" /></IconButton>

            <Divider orientation="vertical" flexItem />

            <IconButton onClick={handleNotificationClick}>
              <Badge badgeContent={3} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>

            <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
              <Typography variant="caption" fontWeight={600}>IA</Typography>
            </Avatar>
          </Stack>
        </Toolbar>
      </AppBar>

      <Menu
        anchorEl={notificationMenuAnchor}
        open={Boolean(notificationMenuAnchor)}
        onClose={handleNotificationClose}
      >
        <MenuItem onClick={handleNotificationClose}>New BUY Signal for AAPL</MenuItem>
        <MenuItem onClick={handleNotificationClose}>Stop Loss Triggered for TSLA</MenuItem>
        <MenuItem onClick={handleNotificationClose}>Target Reached for MSFT</MenuItem>
      </Menu>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          pt: '80px', 
          pb: 3,
          px: 4,
        }}
      >
        <Outlet />
      </Box>
    </Box>
  );
} 