/**
 * Professional Trading Platform Layout
 * 
 * Design Features:
 * - Sleek, minimal navigation bar inspired by Bloomberg Terminal
 * - Enhanced typography with larger, more readable fonts
 * - Professional color scheme with subtle animations
 * - Integrated symbol search and user controls
 */

import React, { useState } from 'react';
import { Link, useLocation, Outlet, useNavigate } from 'react-router-dom';
import {
  AppBar,
  Box,
  Toolbar,
  Typography,
  Container,
  Button,
  alpha,
  useTheme,
  IconButton,
  Stack,
  Divider,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  ShowChart,
  Assessment,
  AccountBalance,
  Settings,
  Notifications,
  AccountCircle
} from '@mui/icons-material';
import { UnifiedSearchBar } from '../Common/UnifiedSearchBar';
import { SimpleSearchBar } from '../Common/SimpleSearchBar';

const StyledAppBar = styled(AppBar)(({ theme }) => ({
  background: 'rgba(10, 13, 20, 0.95)',
  backdropFilter: 'blur(20px)',
  borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  boxShadow: 'none',
  height: '64px',
}));

const LogoContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1.5),
}));

const LogoText = styled(Typography)(({ theme }) => ({
  fontSize: '1.5rem',
  fontWeight: 800,
  letterSpacing: '-0.02em',
  background: 'linear-gradient(135deg, #FFD700 0%, #FFA500 100%)',
  WebkitBackgroundClip: 'text',
  WebkitTextFillColor: 'transparent',
  display: 'flex',
  alignItems: 'center',
}));

const NavSection = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(0.5),
  marginLeft: theme.spacing(6),
}));

const NavButton = styled(Button)(({ theme }) => ({
  color: theme.palette.text.secondary,
  fontSize: '0.95rem',
  fontWeight: 500,
  letterSpacing: '0.01em',
  padding: theme.spacing(1, 2),
  borderRadius: theme.shape.borderRadius,
  textTransform: 'none',
  minWidth: 'auto',
  position: 'relative',
  transition: 'all 0.2s ease',
  '&:hover': {
    color: theme.palette.text.primary,
    backgroundColor: alpha(theme.palette.primary.main, 0.08),
  },
  '&.active': {
    color: theme.palette.primary.main,
    fontWeight: 600,
    '&::after': {
      content: '""',
      position: 'absolute',
      bottom: -1,
      left: '50%',
      transform: 'translateX(-50%)',
      width: '70%',
      height: '2px',
      backgroundColor: theme.palette.primary.main,
      borderRadius: '2px 2px 0 0',
    },
  },
}));

const UserControls = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  marginLeft: 'auto',
}));

const StyledIconButton = styled(IconButton)(({ theme }) => ({
  color: theme.palette.text.secondary,
  transition: 'all 0.2s ease',
  '&:hover': {
    color: theme.palette.text.primary,
    backgroundColor: alpha(theme.palette.primary.main, 0.08),
  },
}));

interface LayoutProps {
  children?: React.ReactNode;
  onSymbolSelect?: (symbol: string) => void;
}

export const Layout: React.FC<LayoutProps> = ({ children, onSymbolSelect }) => {
  const location = useLocation();
  const theme = useTheme();
  const [currentSymbol, setCurrentSymbol] = useState('SPY');
  const navigate = useNavigate();

  const navItems = [
    { path: '/dashboard', label: 'Dashboard', icon: ShowChart },
    { path: '/signals', label: 'Signals', icon: ShowChart },
    { path: '/analytics', label: 'Analytics', icon: Assessment },
    { path: '/ai-center', label: 'AI Center', icon: Assessment },
    { path: '/professional', label: 'Professional', icon: Assessment },
    { path: '/portfolio', label: 'Portfolio', icon: AccountBalance },
  ];

  const isActive = (path: string) => {
    if (path === '/signals' && location.pathname === '/dashboard') return true;
    return location.pathname === path;
  };

  const handleSymbolChange = (symbol: string) => {
    setCurrentSymbol(symbol);
    // You can also emit an event or update a global state here
    // to notify other components about the symbol change
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <StyledAppBar position="fixed">
        <Toolbar sx={{ height: '64px' }}>
          <LogoContainer>
            <Link to="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center' }}>
              <LogoText variant="h6">
                GoldenSignals
                <Typography
                  component="span"
                  sx={{
                    fontSize: '0.7em',
                    fontWeight: 300,
                    color: theme.palette.text.secondary,
                    ml: 0.5,
                    WebkitTextFillColor: 'unset',
                  }}
                >
                  AI
                </Typography>
              </LogoText>
            </Link>
          </LogoContainer>

          <Divider
            orientation="vertical"
            flexItem
            sx={{
              mx: 3,
              height: '60%',
              alignSelf: 'center',
              borderColor: alpha(theme.palette.divider, 0.2),
            }}
          />

          <NavSection>
            {navItems.map(({ path, label, icon: Icon }) => (
              <Link key={path} to={path} style={{ textDecoration: 'none' }}>
                <NavButton
                  className={isActive(path) ? 'active' : ''}
                  startIcon={<Icon sx={{ fontSize: '1.1rem' }} />}
                >
                  {label}
                </NavButton>
              </Link>
            ))}
          </NavSection>

          <UserControls>
            <StyledIconButton size="medium">
              <Notifications sx={{ fontSize: '1.3rem' }} />
            </StyledIconButton>

            <StyledIconButton size="medium" onClick={() => navigate('/settings')}>
              <Settings sx={{ fontSize: '1.3rem' }} />
            </StyledIconButton>

            <Divider
              orientation="vertical"
              flexItem
              sx={{
                mx: 1,
                height: '60%',
                alignSelf: 'center',
                borderColor: alpha(theme.palette.divider, 0.2),
              }}
            />

            <StyledIconButton size="medium">
              <AccountCircle sx={{ fontSize: '1.5rem' }} />
            </StyledIconButton>
          </UserControls>
        </Toolbar>
      </StyledAppBar>

      {/* Enhanced Search Bar - Fixed Below Navigation */}
      <Box sx={{ height: '64px' }} /> {/* Spacer for fixed nav */}
      <Box sx={{
        position: 'fixed',
        top: '64px',
        left: 0,
        right: 0,
        zIndex: 1100,
        bgcolor: alpha(theme.palette.background.default, 0.95),
        backdropFilter: 'blur(20px)',
        borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
      }}>
        <Container maxWidth={false} sx={{ py: 1 }}>
          <SimpleSearchBar
            onSymbolChange={onSymbolSelect}
            placeholder="Search symbols, ask AI, or analyze markets..."
          />
        </Container>
      </Box>
      <Box sx={{ height: '48px' }} /> {/* Further reduced spacer for fixed search bar */}

      <Container
        component="main"
        sx={{
          flexGrow: 1,
          pt: 0, // Removed padding to eliminate gap
          pb: 3,
          px: { xs: 2, sm: 3, md: 4 },
        }}
        maxWidth={false}
      >
        <Outlet context={{ currentSymbol, onSymbolChange: handleSymbolChange }} />
      </Container>

    </Box>
  );
};

export default Layout; 