# GoldenSignalsAI Frontend Implementation Guide

## Overview

This guide provides step-by-step instructions for completing the implementation of the redesigned GoldenSignalsAI frontend. The foundation has been laid with a modern, scalable architecture using React 18, TypeScript, Material-UI, and TradingView Lightweight Charts.

## Current Status

### ✅ Completed
1. **Project Structure**: Feature-based architecture with atomic design
2. **Core Configuration**: TypeScript, Vite, ESLint, Prettier
3. **Type System**: Comprehensive type definitions for all domains
4. **State Management**: Zustand with slices for modular state
5. **Theme System**: Complete theme with light/dark modes
6. **API Architecture**: Robust API client with interceptors
7. **Core Components Started**: Button, Input, Card, TradingChart, SignalCard
8. **Routing Setup**: Lazy-loaded routes with error boundaries

### 🚧 To Be Implemented
1. Complete component library
2. Feature modules (Dashboard, Signals, Analytics, Portfolio)
3. WebSocket integration
4. Testing setup
5. Performance optimizations

## Implementation Steps

### Step 1: Install Dependencies

```bash
cd frontend-v2
npm install
```

### Step 2: Complete Component Library

#### 2.1 Remaining Atomic Components

**Badge Component** (`src/components/atoms/Badge/Badge.tsx`):
```typescript
import React from 'react';
import { Badge as MuiBadge, BadgeProps as MuiBadgeProps } from '@mui/material';

export interface BadgeProps extends MuiBadgeProps {
  variant?: 'dot' | 'standard';
  pulse?: boolean;
}

export const Badge: React.FC<BadgeProps> = ({ 
  children, 
  variant = 'standard',
  pulse = false,
  ...props 
}) => {
  return (
    <MuiBadge
      variant={variant}
      classes={{
        badge: pulse ? 'pulse-animation' : undefined
      }}
      {...props}
    >
      {children}
    </MuiBadge>
  );
};
```

**Spinner Component** (`src/components/atoms/Spinner/Spinner.tsx`):
```typescript
import React from 'react';
import { CircularProgress, Box } from '@mui/material';

export interface SpinnerProps {
  size?: number;
  fullScreen?: boolean;
  message?: string;
}

export const Spinner: React.FC<SpinnerProps> = ({ 
  size = 40, 
  fullScreen = false,
  message 
}) => {
  const content = (
    <Box textAlign="center">
      <CircularProgress size={size} />
      {message && (
        <Typography variant="body2" sx={{ mt: 2 }}>
          {message}
        </Typography>
      )}
    </Box>
  );

  if (fullScreen) {
    return (
      <Box
        display="flex"
        alignItems="center"
        justifyContent="center"
        minHeight="100vh"
      >
        {content}
      </Box>
    );
  }

  return content;
};
```

#### 2.2 Molecule Components

**SearchBar Component** (`src/components/molecules/SearchBar/SearchBar.tsx`):
```typescript
import React, { useState } from 'react';
import { TextField, InputAdornment, IconButton } from '@mui/material';
import { Search, Clear } from '@mui/icons-material';

export interface SearchBarProps {
  placeholder?: string;
  onSearch: (query: string) => void;
  defaultValue?: string;
  autoFocus?: boolean;
}

export const SearchBar: React.FC<SearchBarProps> = ({
  placeholder = 'Search...',
  onSearch,
  defaultValue = '',
  autoFocus = false,
}) => {
  const [query, setQuery] = useState(defaultValue);

  const handleSearch = () => {
    onSearch(query);
  };

  const handleClear = () => {
    setQuery('');
    onSearch('');
  };

  return (
    <TextField
      fullWidth
      value={query}
      onChange={(e) => setQuery(e.target.value)}
      onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
      placeholder={placeholder}
      autoFocus={autoFocus}
      InputProps={{
        startAdornment: (
          <InputAdornment position="start">
            <Search />
          </InputAdornment>
        ),
        endAdornment: query && (
          <InputAdornment position="end">
            <IconButton size="small" onClick={handleClear}>
              <Clear />
            </IconButton>
          </InputAdornment>
        ),
      }}
    />
  );
};
```

#### 2.3 Organism Components

**Header Component** (`src/components/organisms/Header/Header.tsx`):
```typescript
import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Badge,
  Avatar,
  Menu,
  MenuItem,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Notifications,
  DarkMode,
  LightMode,
} from '@mui/icons-material';
import { useUIStore, useAuthStore } from '@/store';

export const Header: React.FC = () => {
  const { theme, toggleTheme, toggleSidebar, notifications } = useUIStore();
  const { user, logout } = useAuthStore();
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const unreadCount = notifications.filter(n => !n.read).length;

  return (
    <AppBar
      position="fixed"
      sx={{
        zIndex: (theme) => theme.zIndex.drawer + 1,
        backdropFilter: 'blur(10px)',
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
      }}
    >
      <Toolbar>
        <IconButton
          edge="start"
          color="inherit"
          onClick={toggleSidebar}
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>

        <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
          GoldenSignalsAI
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <IconButton color="inherit" onClick={toggleTheme}>
            {theme === 'dark' ? <LightMode /> : <DarkMode />}
          </IconButton>

          <IconButton color="inherit">
            <Badge badgeContent={unreadCount} color="error">
              <Notifications />
            </Badge>
          </IconButton>

          <IconButton
            onClick={(e) => setAnchorEl(e.currentTarget)}
            sx={{ ml: 1 }}
          >
            <Avatar src={user?.avatar} alt={user?.name}>
              {user?.name?.[0]}
            </Avatar>
          </IconButton>
        </Box>

        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={() => setAnchorEl(null)}
        >
          <MenuItem onClick={() => navigate('/settings')}>Settings</MenuItem>
          <MenuItem onClick={logout}>Logout</MenuItem>
        </Menu>
      </Toolbar>
    </AppBar>
  );
};
```

**Sidebar Component** (`src/components/organisms/Sidebar/Sidebar.tsx`):
```typescript
import React from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Box,
} from '@mui/material';
import {
  Dashboard,
  TrendingUp,
  Analytics,
  AccountBalance,
  Settings,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useUIStore } from '@/store';

const DRAWER_WIDTH = 240;

const menuItems = [
  { text: 'Dashboard', icon: Dashboard, path: '/' },
  { text: 'Signals', icon: TrendingUp, path: '/signals' },
  { text: 'Analytics', icon: Analytics, path: '/analytics' },
  { text: 'Portfolio', icon: AccountBalance, path: '/portfolio' },
  { text: 'Settings', icon: Settings, path: '/settings' },
];

export const Sidebar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { sidebarOpen } = useUIStore();

  return (
    <Drawer
      variant="persistent"
      anchor="left"
      open={sidebarOpen}
      sx={{
        width: DRAWER_WIDTH,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: DRAWER_WIDTH,
          boxSizing: 'border-box',
          top: 64, // Header height
          height: 'calc(100% - 64px)',
        },
      }}
    >
      <Box sx={{ overflow: 'auto' }}>
        <List>
          {menuItems.map((item) => (
            <ListItem key={item.text} disablePadding>
              <ListItemButton
                selected={location.pathname === item.path}
                onClick={() => navigate(item.path)}
              >
                <ListItemIcon>
                  <item.icon />
                </ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>
    </Drawer>
  );
};
```

### Step 3: Complete Feature Modules

#### 3.1 Dashboard Components

Create `src/features/dashboard/components/MarketOverview.tsx`:
```typescript
import React from 'react';
import { Grid, Card, CardContent, Typography, Box } from '@mui/material';
import { TrendingUp, TrendingDown } from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/services/api/client';
import { formatCurrency, formatPercentage } from '@/utils/formatters';

export const MarketOverview: React.FC = () => {
  const { data: marketData } = useQuery({
    queryKey: ['market-overview'],
    queryFn: async () => {
      const response = await api.get('/market/overview');
      return response.data;
    },
    refetchInterval: 30000,
  });

  const indices = [
    { symbol: 'SPY', name: 'S&P 500', ...marketData?.spy },
    { symbol: 'QQQ', name: 'NASDAQ', ...marketData?.qqq },
    { symbol: 'DIA', name: 'Dow Jones', ...marketData?.dia },
  ];

  return (
    <Grid container spacing={2}>
      {indices.map((index) => (
        <Grid item xs={12} sm={4} key={index.symbol}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary">
                {index.name}
              </Typography>
              <Typography variant="h5" fontWeight="bold">
                {formatCurrency(index.price || 0)}
              </Typography>
              <Box display="flex" alignItems="center" gap={1}>
                {index.change >= 0 ? (
                  <TrendingUp color="success" />
                ) : (
                  <TrendingDown color="error" />
                )}
                <Typography
                  variant="body2"
                  color={index.change >= 0 ? 'success.main' : 'error.main'}
                >
                  {formatCurrency(Math.abs(index.change || 0))} (
                  {formatPercentage(Math.abs(index.changePercent || 0))})
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
};
```

### Step 4: WebSocket Integration

Create `src/services/websocket/client.ts`:
```typescript
import { io, Socket } from 'socket.io-client';
import { useStore } from '@/store';

class WebSocketClient {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  connect() {
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
    
    this.socket = io(wsUrl, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: 1000,
    });

    this.setupEventHandlers();
  }

  private setupEventHandlers() {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      useStore.getState().setWsConnected(true);
      this.reconnectAttempts = 0;
    });

    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      useStore.getState().setWsConnected(false);
    });

    this.socket.on('market_update', (data) => {
      useStore.getState().updateMarketData(data.symbol, data);
    });

    this.socket.on('signal_update', (signal) => {
      useStore.getState().addSignal(signal);
    });

    this.socket.on('error', (error) => {
      console.error('WebSocket error:', error);
    });
  }

  subscribe(channel: string, symbols?: string[]) {
    if (!this.socket) return;
    
    this.socket.emit('subscribe', {
      channel,
      symbols,
    });
  }

  unsubscribe(channel: string, symbols?: string[]) {
    if (!this.socket) return;
    
    this.socket.emit('unsubscribe', {
      channel,
      symbols,
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
}

export const wsClient = new WebSocketClient();
```

### Step 5: Testing Setup

Create `src/tests/setup.ts`:
```typescript
import '@testing-library/jest-dom';
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

afterEach(() => {
  cleanup();
});
```

Create `vitest.config.ts`:
```typescript
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: './src/tests/setup.ts',
    globals: true,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

### Step 6: Environment Variables

Create `.env.example`:
```
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000
VITE_APP_NAME=GoldenSignalsAI
```

### Step 7: Running the Application

1. **Start the backend** (in the main project directory):
   ```bash
   cd src && uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start the frontend** (in a new terminal):
   ```bash
   cd frontend-v2
   npm install
   npm run dev
   ```

3. **Access the application** at `http://localhost:3000`

## Next Steps

1. **Complete remaining components** following the patterns established
2. **Add comprehensive tests** for all components and features
3. **Implement performance optimizations** (memoization, virtualization)
4. **Add PWA support** for offline functionality
5. **Set up CI/CD pipeline** for automated testing and deployment

## Best Practices

1. **Always use TypeScript** - No `any` types allowed
2. **Follow atomic design** - Build from atoms up
3. **Use React Query** for all server state
4. **Use Zustand** for client state only
5. **Memoize expensive computations**
6. **Lazy load routes and heavy components**
7. **Write tests for critical paths**
8. **Document complex logic**

## Troubleshooting

### Common Issues

1. **Module not found errors**: Run `npm install` and check import paths
2. **Type errors**: Ensure all types are properly imported from `@/types`
3. **API connection issues**: Check backend is running and CORS is configured
4. **WebSocket issues**: Ensure WebSocket endpoint is correct in `.env`

### Performance Tips

1. Use `React.memo` for components that receive stable props
2. Use `useMemo` and `useCallback` appropriately
3. Implement virtual scrolling for long lists
4. Use code splitting for large features
5. Optimize bundle size with tree shaking

## Conclusion

This implementation guide provides the foundation for completing the GoldenSignalsAI frontend. Follow the established patterns, maintain consistency, and focus on creating a professional, performant trading platform that users can trust with their financial decisions. 