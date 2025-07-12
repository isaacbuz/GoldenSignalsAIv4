import React, { useState, useEffect } from 'react';
import {
    AppBar,
    Toolbar,
    Box,
    Typography,
    TextField,
    InputAdornment,
    IconButton,
    Chip,
    Stack,
    Badge,
    Avatar,
    Menu,
    MenuItem,
    Divider,
    useTheme,
    alpha,
} from '@mui/material';
import {
    Search,
    NotificationsOutlined,
    SmartToy,
    TrendingUp,
    TrendingDown,
    Circle,
    AccountCircle,
    Settings,
    Logout,
    ShowChart,
} from '@mui/icons-material';
// import { colors, getMarketColor, formatPercent } from '../../theme/modernTradingTheme';

interface MarketTicker {
    symbol: string;
    price: number;
    change: number;
    volume: number;
}

interface ModernTopBarProps {
    onAIOpen: () => void;
    currentSymbol: string;
    onSymbolChange: (symbol: string) => void;
}

export const ModernTopBar: React.FC<ModernTopBarProps> = ({
    onAIOpen,
    currentSymbol,
    onSymbolChange,
}) => {
    const theme = useTheme();

    // Helper functions
    const getMarketColor = (change: number) => {
        return change > 0 ? theme.palette.success.main : theme.palette.error.main;
    };

    const formatPercent = (value: number) => {
        return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
    };
    // State
    const [searchValue, setSearchValue] = useState(currentSymbol);
    const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);
    const [currentTime, setCurrentTime] = useState(new Date());

    // Market data (in real app, this would be from WebSocket)
    const [marketData, setMarketData] = useState<MarketTicker[]>([
        { symbol: 'SPY', price: 456.78, change: 0.85, volume: 45720000 },
        { symbol: 'QQQ', price: 378.23, change: -0.42, volume: 32180000 },
        { symbol: 'DIA', price: 340.91, change: 0.23, volume: 12450000 },
        { symbol: 'IWM', price: 198.45, change: 1.12, volume: 18920000 },
        { symbol: 'VIX', price: 18.42, change: -2.85, volume: 8750000 },
    ]);

    // Update time and simulate market data
    useEffect(() => {
        const timer = setInterval(() => {
            setCurrentTime(new Date());

            // Simulate real-time price updates
            setMarketData(prev => prev.map(ticker => ({
                ...ticker,
                price: ticker.price + (Math.random() - 0.5) * 1,
                change: ticker.change + (Math.random() - 0.5) * 0.2,
            })));
        }, 3000);

        return () => clearInterval(timer);
    }, []);

    // Market status
    const isMarketOpen = () => {
        const now = new Date();
        const hour = now.getHours();
        const day = now.getDay();
        return day >= 1 && day <= 5 && hour >= 9 && hour < 16; // Simplified market hours
    };

    const handleSearchSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (searchValue.trim()) {
            onSymbolChange(searchValue.toUpperCase());
        }
    };

    const handleUserMenuClick = (event: React.MouseEvent<HTMLElement>) => {
        setUserMenuAnchor(event.currentTarget);
    };

    const handleUserMenuClose = () => {
        setUserMenuAnchor(null);
    };

    return (
        <AppBar
            position="static"
            elevation={0}
            sx={{
                bgcolor: theme.palette.background.paper,
                borderBottom: `1px solid ${theme.palette.divider}`,
                zIndex: 1200,
            }}
        >
            <Toolbar sx={{ px: 3, py: 1, minHeight: '72px !important' }}>
                {/* Brand & Logo */}
                <Box sx={{ display: 'flex', alignItems: 'center', mr: 4 }}>
                    <ShowChart sx={{ color: theme.palette.secondary.main, fontSize: 28, mr: 1 }} />
                    <Typography
                        variant="h6"
                        sx={{
                            fontWeight: 700,
                            background: `linear-gradient(45deg, ${theme.palette.secondary.main}, ${theme.palette.primary.main})`,
                            backgroundClip: 'text',
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                        }}
                    >
                        GoldenSignals AI
                    </Typography>
                </Box>

                {/* Market Status & Time */}
                <Box sx={{ display: 'flex', alignItems: 'center', mr: 3 }}>
                    <Circle
                        sx={{
                            color: isMarketOpen() ? theme.palette.success.main : theme.palette.error.main,
                            fontSize: 12,
                            mr: 1,
                            animation: isMarketOpen() ? 'pulse 2s infinite' : 'none',
                        }}
                    />
                    <Typography variant="caption" sx={{ color: theme.palette.text.secondary, mr: 2 }}>
                        {isMarketOpen() ? 'MARKET OPEN' : 'MARKET CLOSED'}
                    </Typography>
                    <Typography variant="caption" sx={{ color: theme.palette.text.secondary }}>
                        {currentTime.toLocaleTimeString('en-US', {
                            hour12: false,
                            hour: '2-digit',
                            minute: '2-digit'
                        })} ET
                    </Typography>
                </Box>

                {/* Market Tickers */}
                <Stack direction="row" spacing={3} sx={{ flex: 1, mx: 2 }}>
                    {marketData.slice(0, 4).map((ticker) => (
                        <Box
                            key={ticker.symbol}
                            sx={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 1,
                                cursor: 'pointer',
                                p: 0.5,
                                borderRadius: 1,
                                transition: 'all 0.2s ease',
                                '&:hover': {
                                    bgcolor: colors.hover,
                                }
                            }}
                            onClick={() => onSymbolChange(ticker.symbol)}
                        >
                            <Typography
                                variant="body2"
                                sx={{
                                    color: currentSymbol === ticker.symbol ? colors.primary : colors.text.primary,
                                    fontWeight: currentSymbol === ticker.symbol ? 'bold' : 'medium',
                                    minWidth: 40
                                }}
                            >
                                {ticker.symbol}
                            </Typography>
                            <Typography variant="body2" sx={{ color: colors.text.secondary, minWidth: 60 }}>
                                ${ticker.price.toFixed(2)}
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                {ticker.change > 0 ? (
                                    <TrendingUp sx={{ fontSize: 16, color: colors.bullish }} />
                                ) : (
                                    <TrendingDown sx={{ fontSize: 16, color: colors.bearish }} />
                                )}
                                <Typography
                                    variant="body2"
                                    sx={{
                                        color: getMarketColor(ticker.change),
                                        fontWeight: 'medium',
                                        minWidth: 50
                                    }}
                                >
                                    {formatPercent(ticker.change)}
                                </Typography>
                            </Box>
                        </Box>
                    ))}
                </Stack>

                {/* Search Bar */}
                <Box component="form" onSubmit={handleSearchSubmit} sx={{ mr: 2 }}>
                    <TextField
                        size="small"
                        value={searchValue}
                        onChange={(e) => setSearchValue(e.target.value.toUpperCase())}
                        placeholder="Search symbols..."
                        InputProps={{
                            startAdornment: (
                                <InputAdornment position="start">
                                    <Search sx={{ fontSize: 18, color: colors.text.muted }} />
                                </InputAdornment>
                            ),
                        }}
                        sx={{
                            width: 200,
                            '& .MuiOutlinedInput-root': {
                                bgcolor: colors.bg.elevated,
                                '& input': {
                                    color: colors.text.primary,
                                    '&::placeholder': {
                                        color: colors.text.muted,
                                        opacity: 1
                                    }
                                },
                            },
                        }}
                    />
                </Box>

                {/* Actions */}
                <Stack direction="row" spacing={1} alignItems="center">
                    {/* AI Assistant */}
                    <IconButton
                        onClick={onAIOpen}
                        sx={{
                            bgcolor: 'rgba(139, 92, 246, 0.1)',
                            border: '1px solid rgba(139, 92, 246, 0.3)',
                            '&:hover': {
                                bgcolor: 'rgba(139, 92, 246, 0.2)',
                                animation: 'pulse 1s ease-in-out',
                            }
                        }}
                    >
                        <SmartToy sx={{ color: colors.secondary }} />
                    </IconButton>

                    {/* Notifications */}
                    <Badge
                        badgeContent={3}
                        sx={{ '& .MuiBadge-badge': { bgcolor: colors.bearish, color: 'white' } }}
                    >
                        <IconButton sx={{ color: colors.text.muted }}>
                            <NotificationsOutlined />
                        </IconButton>
                    </Badge>

                    {/* User Menu */}
                    <IconButton
                        onClick={handleUserMenuClick}
                        sx={{ color: colors.text.muted }}
                    >
                        <AccountCircle />
                    </IconButton>
                    <Menu
                        anchorEl={userMenuAnchor}
                        open={Boolean(userMenuAnchor)}
                        onClose={handleUserMenuClose}
                        PaperProps={{
                            sx: {
                                bgcolor: colors.bg.elevated,
                                border: `1px solid ${colors.border}`,
                                mt: 1,
                            }
                        }}
                    >
                        <MenuItem onClick={handleUserMenuClose} sx={{ color: colors.text.primary }}>
                            <Settings sx={{ mr: 2, fontSize: 20 }} />
                            Settings
                        </MenuItem>
                        <Divider sx={{ borderColor: colors.border }} />
                        <MenuItem onClick={handleUserMenuClose} sx={{ color: colors.text.primary }}>
                            <Logout sx={{ mr: 2, fontSize: 20 }} />
                            Logout
                        </MenuItem>
                    </Menu>
                </Stack>
            </Toolbar>
        </AppBar>
    );
}; 