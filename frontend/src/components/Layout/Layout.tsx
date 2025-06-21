/**
 * Premium Layout Component - Enhanced with Professional UI/UX
 * 
 * Design Philosophy:
 * - Clean top navigation for simplicity
 * - Breadcrumb navigation for better context
 * - Advanced notification center
 * - Command palette for power users
 * - Professional aesthetics with smooth animations
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
  Psychology,
  AccountBalance,
  StarBorder,
} from '@mui/icons-material';
import { useLocation, useNavigate, Outlet } from 'react-router-dom';
import { useSignals } from '../../store';
import { Breadcrumbs } from '../Common/Breadcrumbs';
import NotificationCenter from '../Common/NotificationCenter';
import { CommandPalette } from '../Common/CommandPalette';
import { SymbolSearchBar } from '../Common/SymbolSearchBar';
import { WebSocketStatus } from '../Common/WebSocketStatus';

export default function Layout() {
  const theme = useTheme();
  const location = useLocation();
  const navigate = useNavigate();

  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'disconnected'>('connected');
  const [selectedSymbol, setSelectedSymbol] = useState('SPY');
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);

  // Essential symbols only
  const ESSENTIAL_SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'TSLA'];

  // User favorites from localStorage
  const [favoriteSymbols, setFavoriteSymbols] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem('favoriteSymbols') || '[]');
    } catch {
      return [];
    }
  });

  // Combine essentials and favorites, remove duplicates
  const quickSymbols = [...new Set([...ESSENTIAL_SYMBOLS, ...favoriteSymbols])].slice(0, 6);

  const navigationItems = [
    { path: '/dashboard', label: 'Dashboard', icon: <DashboardIcon /> },
    { path: '/signals', label: 'Signals', icon: <SignalsIcon />, badge: 3 },
    { path: '/ai-command', label: 'AI Command', icon: <Psychology />, badge: 19 },
    { path: '/analytics', label: 'Analytics', icon: <AnalyticsIcon /> },
    { path: '/agents', label: 'Agents', icon: <AgentsIcon /> },
    { path: '/settings', label: 'Settings', icon: <SettingsIcon /> },
  ];

  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Keyboard shortcut for command palette
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setCommandPaletteOpen(true);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* Top AppBar with Two-Tier Layout */}
      <AppBar
        position="fixed"
        elevation={0}
        sx={{
          background: alpha(theme.palette.background.paper, 0.95),
          backdropFilter: 'blur(20px)',
          borderBottom: `1px solid ${theme.palette.divider}`,
        }}
      >
        {/* First Toolbar: Navigation and Actions */}
        <Toolbar sx={{ justifyContent: 'space-between', px: 3, minHeight: '56px !important' }}>
          {/* Left side: Logo and Navigation */}
          <Stack direction="row" alignItems="center" spacing={4}>
            {/* Logo */}
            <Stack
              direction="row"
              alignItems="center"
              spacing={1.5}
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
                  fontSize: '1.1rem',
                }}
              >
                GoldenSignals
              </Typography>
            </Stack>

            {/* Navigation Buttons */}
            <ButtonGroup variant="text" size="small">
              {navigationItems.map((item) => {
                const isActive = location.pathname === item.path ||
                  (item.path !== '/' && location.pathname.startsWith(item.path));
                return (
                  <Button
                    key={item.path}
                    onClick={() => navigate(item.path)}
                    startIcon={item.icon}
                    sx={{
                      color: isActive ? 'primary.main' : 'text.secondary',
                      fontWeight: isActive ? 600 : 500,
                      position: 'relative',
                      '&::after': {
                        content: '""',
                        position: 'absolute',
                        bottom: 0,
                        left: 0,
                        right: 0,
                        height: 2,
                        backgroundColor: 'primary.main',
                        transform: isActive ? 'scaleX(1)' : 'scaleX(0)',
                        transition: 'transform 0.2s',
                      },
                    }}
                  >
                    {item.label}
                    {item.badge && (
                      <Badge
                        badgeContent={item.badge}
                        color="error"
                        sx={{ ml: 1 }}
                      />
                    )}
                  </Button>
                );
              })}
            </ButtonGroup>
          </Stack>

          {/* Right side: Actions */}
          <Stack direction="row" alignItems="center" spacing={2}>
            {/* WebSocket Status */}
            <WebSocketStatus />

            {/* Action Buttons */}
            <Tooltip title="Refresh Data">
              <IconButton size="small">
                <Refresh fontSize="small" />
              </IconButton>
            </Tooltip>

            <Tooltip title="Fullscreen">
              <IconButton size="small">
                <Fullscreen fontSize="small" />
              </IconButton>
            </Tooltip>

            <Divider orientation="vertical" flexItem />

            {/* Notification Center */}
            <NotificationCenter />

            {/* User Avatar */}
            <Avatar
              sx={{
                width: 32,
                height: 32,
                bgcolor: 'primary.main',
                cursor: 'pointer',
                fontSize: '0.875rem',
                '&:hover': {
                  transform: 'scale(1.05)',
                },
                transition: 'transform 0.2s',
              }}
              onClick={() => navigate('/settings')}
            >
              IA
            </Avatar>
          </Stack>
        </Toolbar>

        {/* Second Toolbar: Search Bar */}
        <Box
          sx={{
            backgroundColor: alpha(theme.palette.background.default, 0.5),
            borderTop: `1px solid ${theme.palette.divider}`,
            px: 3,
            py: 1.5,
          }}
        >
          <Stack direction="row" alignItems="center" justifyContent="center" spacing={3}>
            {/* Symbol Search Bar - Centered */}
            <Box sx={{ maxWidth: 600, width: '100%' }}>
              <SymbolSearchBar
                currentSymbol={selectedSymbol}
                onSymbolChange={(symbol) => {
                  setSelectedSymbol(symbol);
                  // Dispatch event for all components to update
                  window.dispatchEvent(new CustomEvent('symbol-change', { detail: { symbol } }));
                }}
                onAddToFavorites={(symbol) => {
                  if (!favoriteSymbols.includes(symbol) && !ESSENTIAL_SYMBOLS.includes(symbol)) {
                    const newFavorites = [...favoriteSymbols, symbol];
                    setFavoriteSymbols(newFavorites);
                    localStorage.setItem('favoriteSymbols', JSON.stringify(newFavorites));
                  }
                }}
                favorites={favoriteSymbols}
              />
            </Box>

            {/* Quick Symbol Switcher with Favorites */}
            <Stack direction="row" spacing={0.5} alignItems="center">
              {quickSymbols.map((symbol, index) => (
                <React.Fragment key={symbol}>
                  {/* Add separator after essential symbols */}
                  {index === ESSENTIAL_SYMBOLS.length && favoriteSymbols.length > 0 && (
                    <Divider orientation="vertical" flexItem sx={{ mx: 0.5, height: 20, alignSelf: 'center' }} />
                  )}
                  <Chip
                    label={symbol}
                    size="small"
                    clickable
                    variant={favoriteSymbols.includes(symbol) && !ESSENTIAL_SYMBOLS.includes(symbol) ? "outlined" : "filled"}
                    onDelete={favoriteSymbols.includes(symbol) && !ESSENTIAL_SYMBOLS.includes(symbol) ? () => {
                      const newFavorites = favoriteSymbols.filter(s => s !== symbol);
                      setFavoriteSymbols(newFavorites);
                      localStorage.setItem('favoriteSymbols', JSON.stringify(newFavorites));
                    } : undefined}
                    onClick={() => {
                      setSelectedSymbol(symbol);
                      window.dispatchEvent(new CustomEvent('symbol-change', { detail: { symbol } }));
                    }}
                    sx={{
                      backgroundColor: selectedSymbol === symbol
                        ? theme.palette.primary.main
                        : favoriteSymbols.includes(symbol) && !ESSENTIAL_SYMBOLS.includes(symbol)
                          ? 'transparent'
                          : alpha(theme.palette.action.hover, 0.1),
                      borderColor: favoriteSymbols.includes(symbol) && !ESSENTIAL_SYMBOLS.includes(symbol)
                        ? alpha(theme.palette.warning.main, 0.5)
                        : undefined,
                      color: selectedSymbol === symbol
                        ? 'primary.contrastText'
                        : 'text.primary',
                      fontWeight: selectedSymbol === symbol ? 'bold' : 'normal',
                      transition: 'all 0.2s',
                      '&:hover': {
                        backgroundColor: selectedSymbol === symbol
                          ? theme.palette.primary.dark
                          : alpha(theme.palette.action.hover, 0.2),
                        transform: 'translateY(-1px)',
                      },
                      '& .MuiChip-deleteIcon': {
                        fontSize: '1rem',
                        color: 'inherit',
                        opacity: 0.7,
                        '&:hover': {
                          opacity: 1,
                        },
                      },
                    }}
                  />
                </React.Fragment>
              ))}

              {/* Add to Favorites Button */}
              {selectedSymbol && !favoriteSymbols.includes(selectedSymbol) && !ESSENTIAL_SYMBOLS.includes(selectedSymbol) && (
                <Tooltip title="Add to favorites">
                  <IconButton
                    size="small"
                    onClick={() => {
                      const newFavorites = [...favoriteSymbols, selectedSymbol];
                      setFavoriteSymbols(newFavorites);
                      localStorage.setItem('favoriteSymbols', JSON.stringify(newFavorites));
                    }}
                    sx={{
                      ml: 0.5,
                      color: theme.palette.text.secondary,
                      '&:hover': {
                        color: theme.palette.warning.main,
                      },
                    }}
                  >
                    <StarBorder sx={{ fontSize: 18 }} />
                  </IconButton>
                </Tooltip>
              )}</Stack>
          </Stack>
        </Box>
      </AppBar>

      {/* Breadcrumbs */}
      <Box sx={{ mt: 12 }}>
        <Breadcrumbs />
      </Box>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: 'background.default',
        }}
      >
        <Box
          sx={{
            flexGrow: 1,
            p: 3,
          }}
        >
          <Outlet context={{ selectedSymbol }} />
        </Box>
      </Box>

      {/* Command Palette */}
      <CommandPalette
        open={commandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
      />
    </Box>
  );
} 