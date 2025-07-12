import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
    Box,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Typography,
    IconButton,
    Chip,
    Divider,
    useTheme,
    Stack,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    LinearProgress,
} from '@mui/material';
import {
    Dashboard,
    AutoAwesome,
    Analytics,
    School,
    Settings,
    ChevronLeft,
    ChevronRight,
    TrendingUp,
    TrendingDown,
    Circle,
    Star,
    StarBorder,
} from '@mui/icons-material';
import { getMarketColor, formatMarketValue } from '../../theme/tradingTheme';

interface SidebarProps {
    collapsed?: boolean;
    onToggle?: () => void;
}

interface WatchlistItem {
    symbol: string;
    price: number;
    change: number;
    changePercent: number;
    volume: string;
    favorite: boolean;
}

interface AISignal {
    symbol: string;
    action: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    time: string;
    price: number;
}

export const TradingSidebar: React.FC<SidebarProps> = ({
    collapsed = false,
    onToggle,
}) => {
    const theme = useTheme();

    // Navigation items
    const navItems = [
        { icon: Dashboard, label: 'Dashboard', path: '/', badge: null },
        { icon: AutoAwesome, label: 'AI Signals', path: '/signals', badge: '8' },
        { icon: Analytics, label: 'Analytics', path: '/analytics', badge: null },
        { icon: School, label: 'Learn', path: '/education', badge: 'NEW' },
        { icon: Settings, label: 'Settings', path: '/settings', badge: null },
    ];

    // Mock watchlist data
    const [watchlist] = useState<WatchlistItem[]>([
        { symbol: 'AAPL', price: 195.89, change: 2.47, changePercent: 1.28, volume: '54.7M', favorite: true },
        { symbol: 'MSFT', price: 378.85, change: -1.23, changePercent: -0.32, volume: '28.3M', favorite: true },
        { symbol: 'GOOGL', price: 142.56, change: 0.89, changePercent: 0.63, volume: '31.2M', favorite: false },
        { symbol: 'TSLA', price: 248.50, change: -5.67, changePercent: -2.23, volume: '89.1M', favorite: true },
        { symbol: 'NVDA', price: 485.32, change: 12.45, changePercent: 2.63, volume: '41.8M', favorite: false },
        { symbol: 'AMZN', price: 155.73, change: 1.89, changePercent: 1.23, volume: '25.4M', favorite: false },
    ]);

    // Mock AI signals
    const [signals] = useState<AISignal[]>([
        { symbol: 'AAPL', action: 'BUY', confidence: 92, time: '2m', price: 195.89 },
        { symbol: 'MSFT', action: 'BUY', confidence: 88, time: '5m', price: 378.85 },
        { symbol: 'TSLA', action: 'SELL', confidence: 85, time: '8m', price: 248.50 },
        { symbol: 'GOOGL', action: 'HOLD', confidence: 75, time: '12m', price: 142.56 },
    ]);

    const sidebarWidth = collapsed ? 60 : 280;

    return (
        <Box
            sx={{
                width: sidebarWidth,
                height: '100vh',
                bgcolor: 'background.paper',
                borderRight: `1px solid ${theme.palette.divider}`,
                display: 'flex',
                flexDirection: 'column',
                transition: 'width 0.2s ease',
                overflow: 'hidden',
            }}
        >
            {/* Header */}
            <Box sx={{ p: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                {!collapsed && (
                    <Typography variant="caption" sx={{ fontWeight: 700, color: 'text.secondary' }}>
                        NAVIGATION
                    </Typography>
                )}
                <IconButton onClick={onToggle} size="small">
                    {collapsed ? <ChevronRight /> : <ChevronLeft />}
                </IconButton>
            </Box>

            {/* Navigation */}
            <Box sx={{ px: 1 }}>
                <List dense>
                    {navItems.map((item) => (
                        <ListItem key={item.path} disablePadding sx={{ mb: 0.5 }}>
                            <ListItemButton
                                component={NavLink}
                                to={item.path}
                                sx={{
                                    borderRadius: 1,
                                    minHeight: 36,
                                    px: 1,
                                    '&.active': {
                                        bgcolor: theme.palette.primary.main,
                                        color: 'white',
                                        '& .MuiListItemIcon-root': { color: 'white' },
                                    },
                                }}
                            >
                                <ListItemIcon sx={{ minWidth: collapsed ? 'auto' : 36 }}>
                                    <item.icon sx={{ fontSize: 18 }} />
                                </ListItemIcon>
                                {!collapsed && (
                                    <>
                                        <ListItemText
                                            primary={item.label}
                                            primaryTypographyProps={{
                                                fontSize: '0.75rem',
                                                fontWeight: 600
                                            }}
                                        />
                                        {item.badge && (
                                            <Chip
                                                label={item.badge}
                                                size="small"
                                                color={item.badge === 'NEW' ? 'secondary' : 'primary'}
                                                sx={{
                                                    fontSize: '0.625rem',
                                                    height: 16,
                                                }}
                                            />
                                        )}
                                    </>
                                )}
                            </ListItemButton>
                        </ListItem>
                    ))}
                </List>
            </Box>

            {!collapsed && (
                <>
                    <Divider sx={{ my: 1 }} />

                    {/* AI Signals Section */}
                    <Box sx={{ px: 1, mb: 2 }}>
                        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
                            <Typography variant="caption" sx={{ fontWeight: 700, color: 'text.secondary' }}>
                                LIVE AI SIGNALS
                            </Typography>
                            <Circle sx={{ fontSize: 8, color: 'success.main' }} />
                        </Stack>

                        <Paper sx={{ bgcolor: 'background.default', p: 1 }}>
                            {signals.map((signal, idx) => (
                                <Box key={idx} sx={{ mb: 1, '&:last-child': { mb: 0 } }}>
                                    <Stack direction="row" alignItems="center" justifyContent="space-between">
                                        <Typography variant="body2" sx={{ fontWeight: 700, fontSize: '0.75rem' }}>
                                            {signal.symbol}
                                        </Typography>
                                        <Chip
                                            label={signal.action}
                                            size="small"
                                            sx={{
                                                fontSize: '0.625rem',
                                                height: 18,
                                                bgcolor: signal.action === 'BUY' ? 'success.main' :
                                                    signal.action === 'SELL' ? 'error.main' : 'warning.main',
                                                color: 'white',
                                            }}
                                        />
                                    </Stack>
                                    <Stack direction="row" alignItems="center" justifyContent="space-between">
                                        <Typography variant="caption" color="text.secondary">
                                            {signal.confidence}% conf.
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary">
                                            {signal.time} ago
                                        </Typography>
                                    </Stack>
                                    <LinearProgress
                                        variant="determinate"
                                        value={signal.confidence}
                                        sx={{
                                            height: 3,
                                            borderRadius: 1,
                                            bgcolor: 'divider',
                                            '& .MuiLinearProgress-bar': {
                                                bgcolor: signal.action === 'BUY' ? 'success.main' :
                                                    signal.action === 'SELL' ? 'error.main' : 'warning.main',
                                            }
                                        }}
                                    />
                                </Box>
                            ))}
                        </Paper>
                    </Box>

                    <Divider />

                    {/* Watchlist Section */}
                    <Box sx={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                        <Box sx={{ px: 1, py: 1 }}>
                            <Typography variant="caption" sx={{ fontWeight: 700, color: 'text.secondary' }}>
                                WATCHLIST
                            </Typography>
                        </Box>

                        <Box sx={{ flex: 1, overflow: 'auto' }}>
                            <Table size="small">
                                <TableHead>
                                    <TableRow>
                                        <TableCell sx={{ py: 0.5, fontSize: '0.625rem' }}>Symbol</TableCell>
                                        <TableCell align="right" sx={{ py: 0.5, fontSize: '0.625rem' }}>Price</TableCell>
                                        <TableCell align="right" sx={{ py: 0.5, fontSize: '0.625rem' }}>Chg%</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {watchlist.map((item) => (
                                        <TableRow
                                            key={item.symbol}
                                            sx={{
                                                cursor: 'pointer',
                                                '&:hover': { bgcolor: 'action.hover' }
                                            }}
                                        >
                                            <TableCell sx={{ py: 0.5 }}>
                                                <Stack direction="row" alignItems="center" spacing={0.5}>
                                                    <IconButton size="small" sx={{ p: 0 }}>
                                                        {item.favorite ?
                                                            <Star sx={{ fontSize: 12, color: 'warning.main' }} /> :
                                                            <StarBorder sx={{ fontSize: 12 }} />
                                                        }
                                                    </IconButton>
                                                    <Typography variant="body2" sx={{ fontWeight: 700, fontSize: '0.75rem' }}>
                                                        {item.symbol}
                                                    </Typography>
                                                </Stack>
                                            </TableCell>
                                            <TableCell align="right" sx={{ py: 0.5 }}>
                                                <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                                                    ${item.price.toFixed(2)}
                                                </Typography>
                                            </TableCell>
                                            <TableCell align="right" sx={{ py: 0.5 }}>
                                                <Stack alignItems="flex-end">
                                                    <Typography
                                                        variant="body2"
                                                        sx={{
                                                            fontSize: '0.75rem',
                                                            color: getMarketColor(item.change),
                                                            fontWeight: 600,
                                                        }}
                                                    >
                                                        {formatMarketValue(item.changePercent, '', '%')}
                                                    </Typography>
                                                    <Typography variant="caption" color="text.secondary">
                                                        {item.volume}
                                                    </Typography>
                                                </Stack>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </Box>
                    </Box>
                </>
            )}
        </Box>
    );
}; 