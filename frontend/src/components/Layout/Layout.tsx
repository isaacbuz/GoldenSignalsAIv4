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
  background: 'linear-gradient(180deg, rgba(10, 13, 20, 0.98) 0%, rgba(10, 13, 20, 0.95) 100%)',
  backdropFilter: 'blur(24px)',
  borderBottom: `1px solid ${alpha('#FFD700', 0.1)}`,
  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
  height: '56px',
  transition: 'all 0.3s ease',
}));

const LogoContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1.5),
}));

const LogoText = styled(Typography)(({ theme }) => ({
  fontSize: '1.4rem',
  fontWeight: 700,
  letterSpacing: '-0.02em',
  background: 'linear-gradient(135deg, #FFD700 0%, #FFC107 50%, #FF9800 100%)',
  WebkitBackgroundClip: 'text',
  WebkitTextFillColor: 'transparent',
  display: 'flex',
  alignItems: 'center',
  textShadow: '0 0 30px rgba(255, 215, 0, 0.3)',
}));

const NavSection = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(0.5),
  marginLeft: theme.spacing(6),
}));

const NavButton = styled(Button)(({ theme }) => ({
  color: alpha(theme.palette.text.secondary, 0.8),
  fontSize: '0.875rem',
  fontWeight: 500,
  letterSpacing: '0.02em',
  padding: theme.spacing(0.75, 1.5),
  borderRadius: '8px',
  textTransform: 'none',
  minWidth: 'auto',
  position: 'relative',
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  '&:hover': {
    color: theme.palette.primary.main,
    backgroundColor: alpha(theme.palette.primary.main, 0.12),
    transform: 'translateY(-1px)',
    boxShadow: `0 4px 12px ${alpha(theme.palette.primary.main, 0.25)}`,
  },
  '&.active': {
    color: theme.palette.primary.main,
    fontWeight: 600,
    backgroundColor: alpha(theme.palette.primary.main, 0.15),
    boxShadow: `0 0 20px ${alpha(theme.palette.primary.main, 0.3)}`,
    '&::after': {
      content: '""',
      position: 'absolute',
      bottom: -8,
      left: '50%',
      transform: 'translateX(-50%)',
      width: '60%',
      height: '3px',
      backgroundColor: theme.palette.primary.main,
      borderRadius: '3px 3px 0 0',
      boxShadow: `0 -2px 8px ${alpha(theme.palette.primary.main, 0.5)}`,
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
  color: alpha(theme.palette.text.secondary, 0.7),
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  padding: theme.spacing(1),
  '&:hover': {
    color: theme.palette.primary.main,
    backgroundColor: alpha(theme.palette.primary.main, 0.1),
    transform: 'scale(1.05)',
    boxShadow: `0 0 16px ${alpha(theme.palette.primary.main, 0.3)}`,
  },
  '& .MuiBadge-badge': {
    backgroundColor: theme.palette.primary.main,
    color: theme.palette.background.default,
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
        <Toolbar sx={{ height: '56px', px: 3 }}>
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
      <Box sx={{ height: '56px' }} /> {/* Spacer for fixed nav */}
      <Box sx={{
        position: 'fixed',
        top: '56px',
        left: 0,
        right: 0,
        zIndex: 1100,
        bgcolor: alpha(theme.palette.background.default, 0.98),
        backdropFilter: 'blur(24px)',
        borderBottom: `1px solid ${alpha('#FFD700', 0.08)}`,
        boxShadow: '0 1px 4px rgba(0, 0, 0, 0.2)',
      }}>
        <Container maxWidth={false} sx={{ py: 0.75 }}>
          <SimpleSearchBar
            onSymbolChange={onSymbolSelect}
            placeholder="Search symbols, ask AI, or analyze markets..."
          />
        </Container>
      </Box>
      <Box sx={{ height: '52px' }} /> {/* Spacer for fixed search bar */}

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          pt: 0,
          pb: 0,
          px: 0,
          height: 'calc(100vh - 108px)', // Full height minus navbar (56px) and search bar (52px)
        }}
      >
        <Outlet context={{ currentSymbol, onSymbolChange: handleSymbolChange }} />
      </Box>

    </Box>
  );
};

export default Layout;
