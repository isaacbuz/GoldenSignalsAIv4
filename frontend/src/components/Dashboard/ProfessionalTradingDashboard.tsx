/**
 * Professional Trading Dashboard
 *
 * A reimagined professional trading interface inspired by Bloomberg Terminal
 * Features the AI Prophet with golden eye dock and professional layout
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
    Box,
    Grid,
    Card,
    CardContent,
    Typography,
    Stack,
    Chip,
    IconButton,
    Button,
    Divider,
    Paper,
    Alert,
    useTheme,
    alpha,
    Skeleton,
    Badge,
    Avatar,
    List,
    ListItem,
    ListItemText,
    ListItemIcon,
    ListItemButton,
    Tabs,
    Tab,
    Fade,
    Zoom,
    LinearProgress,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    ShowChart,
    Article,
    Bookmark,
    Star,
    StarBorder,
    Refresh,
    Settings,
    Timeline,
    Analytics,
    Speed,
    AccessTime,
    AttachMoney,
    BarChart,
    Insights,
    Visibility,
    Remove,
    Add,
    MoreVert,
    Launch,
    Notifications,
    Psychology,
    AutoAwesome,
    CandlestickChart,
    QueryStats,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../services/api/apiClient';
import { UnifiedChart } from '../Chart/UnifiedChart';
import { UnifiedAIChat } from '../AI/UnifiedAIChat';
import { FloatingAIProphetWidget } from '../AI/FloatingAIProphetWidget';
import { AIInsightsPanel } from '../AI/AIInsightsPanel';
import { utilityClasses } from '../../theme/goldenTheme';
import { liveDataService, LiveSignalData, LiveMarketData, LiveMetrics } from '../../services/LiveDataService';
import logger from '../../services/logger';


// Professional color scheme
const PROFESSIONAL_COLORS = {
    background: '#0A0E1A',
    surface: '#131A2A',
    card: '#1E293B',
    accent: '#FFD700',
    bullish: '#00D4AA',
    bearish: '#FF4757',
    text: '#E2E8F0',
    textSecondary: '#94A3B8',
    border: 'rgba(255, 215, 0, 0.1)',
};

// Sample data - replace with real API calls
const TRENDING_STOCKS = [
    { symbol: 'AAPL', name: 'Apple Inc.', price: 175.43, change: +2.31, changePercent: +1.33, volume: '45.2M' },
    { symbol: 'NVDA', name: 'NVIDIA Corp.', price: 431.20, change: +8.75, changePercent: +2.07, volume: '32.1M' },
    { symbol: 'TSLA', name: 'Tesla Inc.', price: 248.87, change: -3.42, changePercent: -1.36, volume: '28.9M' },
    { symbol: 'MSFT', name: 'Microsoft Corp.', price: 378.91, change: +1.23, changePercent: +0.33, volume: '21.7M' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 134.56, change: +0.87, changePercent: +0.65, volume: '18.4M' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.', price: 143.21, change: -1.45, changePercent: -1.00, volume: '25.3M' },
    { symbol: 'META', name: 'Meta Platforms', price: 298.74, change: +4.12, changePercent: +1.40, volume: '19.8M' },
    { symbol: 'AMD', name: 'Advanced Micro Devices', price: 142.33, change: +2.87, changePercent: +2.06, volume: '31.2M' },
];

const NEWS_ITEMS = [
    {
        id: 1,
        title: 'Federal Reserve Holds Interest Rates Steady',
        summary: 'The Fed maintains current rates while signaling potential changes ahead...',
        source: 'Reuters',
        timestamp: '2 hours ago',
        impact: 'high',
        sentiment: 'neutral',
    },
    {
        id: 2,
        title: 'Tech Giants Report Strong Q4 Earnings',
        summary: 'Major technology companies exceed analyst expectations...',
        source: 'Bloomberg',
        timestamp: '4 hours ago',
        impact: 'high',
        sentiment: 'positive',
    },
    {
        id: 3,
        title: 'Oil Prices Surge on Supply Concerns',
        summary: 'Crude oil futures jump 3% amid geopolitical tensions...',
        source: 'MarketWatch',
        timestamp: '6 hours ago',
        impact: 'medium',
        sentiment: 'positive',
    },
    {
        id: 4,
        title: 'AI Chip Demand Continues to Accelerate',
        summary: 'Semiconductor companies see unprecedented demand...',
        source: 'CNBC',
        timestamp: '8 hours ago',
        impact: 'high',
        sentiment: 'positive',
    },
];

interface ProfessionalTradingDashboardProps {
    selectedSymbol?: string;
    onSymbolChange?: (symbol: string) => void;
}

export const ProfessionalTradingDashboard: React.FC<ProfessionalTradingDashboardProps> = ({
    selectedSymbol = 'SPY',
    onSymbolChange,
}) => {
    const theme = useTheme();
    const [aiChatOpen, setAiChatOpen] = useState(false);
    const [watchlist, setWatchlist] = useState<string[]>(['SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA']);
    const [newsTab, setNewsTab] = useState(0);
    const [refreshing, setRefreshing] = useState(false);

    // Add CSS animations
    useEffect(() => {
        const style = document.createElement('style');
        style.textContent = `
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.7; }
                100% { opacity: 1; }
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            @keyframes glow {
                0% { box-shadow: 0 0 5px rgba(255, 215, 0, 0.3); }
                50% { box-shadow: 0 0 20px rgba(255, 215, 0, 0.6); }
                100% { box-shadow: 0 0 5px rgba(255, 215, 0, 0.3); }
            }
        `;
        document.head.appendChild(style);

        return () => {
            document.head.removeChild(style);
        };
    }, []);

    // Live data state
    const [liveSignals, setLiveSignals] = useState<LiveSignalData[]>([]);
    const [liveMarketData, setLiveMarketData] = useState<Map<string, LiveMarketData>>(new Map());
    const [liveMetrics, setLiveMetrics] = useState<LiveMetrics | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [updatingSymbols, setUpdatingSymbols] = useState<Set<string>>(new Set());

    // Fetch market data
    const { data: marketData, isLoading: marketLoading } = useQuery({
        queryKey: ['market-data', selectedSymbol],
        queryFn: () => apiClient.getMarketData(selectedSymbol),
        refetchInterval: 30000,
    });

    // Fetch signals
    const { data: signals, isLoading: signalsLoading } = useQuery({
        queryKey: ['signals', selectedSymbol],
        queryFn: () => apiClient.getSignalsForSymbol(selectedSymbol, { hours_back: 24 }),
        refetchInterval: 15000,
    });

    const handleSymbolSelect = useCallback((symbol: string) => {
        onSymbolChange?.(symbol);
    }, [onSymbolChange]);

    const handleAddToWatchlist = useCallback((symbol: string) => {
        setWatchlist(prev => prev.includes(symbol) ? prev : [...prev, symbol]);
    }, []);

    const handleRemoveFromWatchlist = useCallback((symbol: string) => {
        setWatchlist(prev => prev.filter(s => s !== symbol));
    }, []);

    // Live data subscriptions
    useEffect(() => {
        // Subscribe to live data service
        const unsubscribeConnection = liveDataService.subscribe('connected', () => {
            setIsConnected(true);
        });

        const unsubscribeDisconnection = liveDataService.subscribe('disconnected', () => {
            setIsConnected(false);
        });

        const unsubscribeSignals = liveDataService.subscribe('signal', (signal: LiveSignalData) => {
            setLiveSignals(prev => [signal, ...prev.slice(0, 49)]);
        });

        const unsubscribeMarket = liveDataService.subscribe('market', (marketData: LiveMarketData) => {
            setLiveMarketData(prev => {
                const newMap = new Map(prev);
                newMap.set(marketData.symbol, marketData);
                return newMap;
            });

            // Show update indicator
            setUpdatingSymbols(prev => {
                const newSet = new Set(prev);
                newSet.add(marketData.symbol);
                return newSet;
            });

            // Remove update indicator after 2 seconds
            setTimeout(() => {
                setUpdatingSymbols(prev => {
                    const newSet = new Set(prev);
                    newSet.delete(marketData.symbol);
                    return newSet;
                });
            }, 2000);
        });

        const unsubscribeMetrics = liveDataService.subscribe('metrics', (metrics: LiveMetrics) => {
            setLiveMetrics(metrics);
        });

        // Check initial connection status
        setIsConnected(liveDataService.isConnectedToLiveData());

        // Load cached data
        const cachedData = liveDataService.getAllCachedData();
        if (cachedData.signals[selectedSymbol]) {
            setLiveSignals(cachedData.signals[selectedSymbol]);
        }
        if (cachedData.market) {
            setLiveMarketData(new Map(Object.entries(cachedData.market)));
        }
        if (cachedData.metrics) {
            setLiveMetrics(cachedData.metrics);
        }

        return () => {
            unsubscribeConnection();
            unsubscribeDisconnection();
            unsubscribeSignals();
            unsubscribeMarket();
            unsubscribeMetrics();
        };
    }, [selectedSymbol]);

    // Separate effect for symbol subscriptions to prevent unnecessary re-subscriptions
    useEffect(() => {
        const symbolUnsubscribers: Array<() => void> = [];

        // Subscribe to selected symbol
        symbolUnsubscribers.push(
            liveDataService.subscribeToSymbol(selectedSymbol, (data) => {
                logger.info('Symbol update:', data);
            })
        );

        // Subscribe to all trending stocks for real-time updates
        TRENDING_STOCKS.forEach(stock => {
            symbolUnsubscribers.push(
                liveDataService.subscribeToSymbol(stock.symbol, (data) => {
                    logger.info(`Trending stock update for ${stock.symbol}:`, data);
                })
            );
        });

        return () => {
            symbolUnsubscribers.forEach(unsub => unsub());
        };
    }, [selectedSymbol]);

    // Separate effect for watchlist subscriptions
    useEffect(() => {
        const watchlistUnsubscribers: Array<() => void> = [];

        // Subscribe to all watchlist symbols for real-time updates
        watchlist.forEach(symbol => {
            watchlistUnsubscribers.push(
                liveDataService.subscribeToSymbol(symbol, (data) => {
                    logger.info(`Watchlist update for ${symbol}:`, data);
                })
            );
        });

        return () => {
            watchlistUnsubscribers.forEach(unsub => unsub());
        };
    }, [watchlist]);

    const handleRefresh = useCallback(async () => {
        setRefreshing(true);
        try {
            await liveDataService.refreshAll();
        } catch (error) {
            logger.error('Failed to refresh data:', error);
        }
        setRefreshing(false);
    }, []);

    // Get live market data for trending stocks
    const getTrendingStocksWithLiveData = useMemo(() => {
        return TRENDING_STOCKS.map(stock => {
            const liveData = liveMarketData.get(stock.symbol);
            if (liveData) {
                return {
                    ...stock,
                    price: liveData.price,
                    change: liveData.change,
                    changePercent: liveData.change_percent,
                    volume: liveData.volume.toString(),
                };
            }
            return stock;
        });
    }, [liveMarketData]);

    // Trending Stocks Component
    const TrendingStocks = () => (
        <Card sx={{
            height: '100%',
            background: PROFESSIONAL_COLORS.card,
            border: `1px solid ${PROFESSIONAL_COLORS.border}`,
            ...utilityClasses.glassmorphism
        }}>
            <CardContent sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                    <Typography variant="h6" sx={{
                        color: PROFESSIONAL_COLORS.text,
                        fontWeight: 700,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1
                    }}>
                        <TrendingUp sx={{ color: PROFESSIONAL_COLORS.accent }} />
                        Trending Stocks
                    </Typography>
                    <Stack direction="row" alignItems="center" spacing={1}>
                        {isConnected && (
                            <Chip
                                label="LIVE"
                                size="small"
                                sx={{
                                    height: 20,
                                    fontSize: '0.6rem',
                                    bgcolor: alpha(PROFESSIONAL_COLORS.bullish, 0.2),
                                    color: PROFESSIONAL_COLORS.bullish,
                                    fontWeight: 700,
                                    animation: 'pulse 2s infinite'
                                }}
                            />
                        )}
                        <IconButton size="small" onClick={handleRefresh} disabled={refreshing}>
                            <Refresh sx={{
                                color: PROFESSIONAL_COLORS.textSecondary,
                                animation: refreshing ? 'spin 1s linear infinite' : 'none'
                            }} />
                        </IconButton>
                    </Stack>
                </Box>

                <List dense sx={{ maxHeight: 400, overflow: 'auto' }}>
                    {getTrendingStocksWithLiveData.map((stock, index) => (
                        <motion.div
                            key={stock.symbol}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.1 }}
                        >
                            <ListItemButton
                                onClick={() => handleSymbolSelect(stock.symbol)}
                                sx={{
                                    borderRadius: 1,
                                    mb: 0.5,
                                    border: selectedSymbol === stock.symbol ? `1px solid ${PROFESSIONAL_COLORS.accent}` : 'none',
                                    background: selectedSymbol === stock.symbol ? alpha(PROFESSIONAL_COLORS.accent, 0.1) : 'transparent',
                                    '&:hover': {
                                        background: alpha(PROFESSIONAL_COLORS.accent, 0.05),
                                    }
                                }}
                            >
                                <ListItemIcon sx={{ minWidth: 40 }}>
                                    <Avatar sx={{
                                        width: 32,
                                        height: 32,
                                        bgcolor: stock.change > 0 ? PROFESSIONAL_COLORS.bullish : PROFESSIONAL_COLORS.bearish,
                                        fontSize: '0.75rem',
                                        fontWeight: 700,
                                        animation: updatingSymbols.has(stock.symbol) ? 'pulse 1s ease-in-out' : 'none',
                                        boxShadow: updatingSymbols.has(stock.symbol) ? `0 0 10px ${PROFESSIONAL_COLORS.accent}` : 'none',
                                    }}>
                                        {stock.symbol.slice(0, 2)}
                                    </Avatar>
                                </ListItemIcon>
                                <ListItemText
                                    primary={
                                        <Stack direction="row" justifyContent="space-between" alignItems="center">
                                            <Typography variant="body2" sx={{ color: PROFESSIONAL_COLORS.text, fontWeight: 600 }}>
                                                {stock.symbol}
                                            </Typography>
                                            <Typography variant="body2" sx={{ color: PROFESSIONAL_COLORS.text, fontWeight: 700 }}>
                                                ${stock.price.toFixed(2)}
                                            </Typography>
                                        </Stack>
                                    }
                                    secondary={
                                        <Stack direction="row" justifyContent="space-between" alignItems="center">
                                            <Typography variant="caption" sx={{ color: PROFESSIONAL_COLORS.textSecondary }}>
                                                {stock.name}
                                            </Typography>
                                            <Stack direction="row" alignItems="center" spacing={0.5}>
                                                <Typography
                                                    variant="caption"
                                                    sx={{
                                                        color: stock.change > 0 ? PROFESSIONAL_COLORS.bullish : PROFESSIONAL_COLORS.bearish,
                                                        fontWeight: 600
                                                    }}
                                                >
                                                    {stock.change > 0 ? '+' : ''}{stock.change.toFixed(2)}
                                                </Typography>
                                                <Typography
                                                    variant="caption"
                                                    sx={{
                                                        color: stock.change > 0 ? PROFESSIONAL_COLORS.bullish : PROFESSIONAL_COLORS.bearish,
                                                        fontWeight: 600
                                                    }}
                                                >
                                                    ({stock.changePercent > 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%)
                                                </Typography>
                                            </Stack>
                                        </Stack>
                                    }
                                />
                                <IconButton
                                    size="small"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        watchlist.includes(stock.symbol)
                                            ? handleRemoveFromWatchlist(stock.symbol)
                                            : handleAddToWatchlist(stock.symbol);
                                    }}
                                    sx={{ ml: 1 }}
                                >
                                    {watchlist.includes(stock.symbol) ?
                                        <Star sx={{ color: PROFESSIONAL_COLORS.accent, fontSize: 16 }} /> :
                                        <StarBorder sx={{ color: PROFESSIONAL_COLORS.textSecondary, fontSize: 16 }} />
                                    }
                                </IconButton>
                            </ListItemButton>
                        </motion.div>
                    ))}
                </List>
            </CardContent>
        </Card>
    );

    // News Component
    const NewsComponent = () => (
        <Card sx={{
            height: '100%',
            background: PROFESSIONAL_COLORS.card,
            border: `1px solid ${PROFESSIONAL_COLORS.border}`,
            ...utilityClasses.glassmorphism
        }}>
            <CardContent sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                    <Typography variant="h6" sx={{
                        color: PROFESSIONAL_COLORS.text,
                        fontWeight: 700,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1
                    }}>
                        <Article sx={{ color: PROFESSIONAL_COLORS.accent }} />
                        Market News
                    </Typography>
                    <Tabs
                        value={newsTab}
                        onChange={(_, newValue) => setNewsTab(newValue)}
                        sx={{ minHeight: 32 }}
                    >
                        <Tab label="All" sx={{ minHeight: 32, py: 0.5, color: PROFESSIONAL_COLORS.textSecondary }} />
                        <Tab label="High Impact" sx={{ minHeight: 32, py: 0.5, color: PROFESSIONAL_COLORS.textSecondary }} />
                    </Tabs>
                </Box>

                <List dense sx={{ maxHeight: 400, overflow: 'auto' }}>
                    {NEWS_ITEMS
                        .filter(item => newsTab === 0 || item.impact === 'high')
                        .map((item, index) => (
                            <motion.div
                                key={item.id}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: index * 0.1 }}
                            >
                                <ListItemButton sx={{
                                    borderRadius: 1,
                                    mb: 1,
                                    border: `1px solid ${alpha(PROFESSIONAL_COLORS.border, 0.5)}`,
                                    '&:hover': {
                                        background: alpha(PROFESSIONAL_COLORS.accent, 0.05),
                                    }
                                }}>
                                    <ListItemIcon sx={{ minWidth: 40 }}>
                                        <Box sx={{
                                            width: 32,
                                            height: 32,
                                            borderRadius: 1,
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            bgcolor: item.impact === 'high' ? alpha(PROFESSIONAL_COLORS.bearish, 0.2) : alpha(PROFESSIONAL_COLORS.bullish, 0.2),
                                            border: `1px solid ${item.impact === 'high' ? PROFESSIONAL_COLORS.bearish : PROFESSIONAL_COLORS.bullish}`,
                                        }}>
                                            <Article sx={{
                                                fontSize: 16,
                                                color: item.impact === 'high' ? PROFESSIONAL_COLORS.bearish : PROFESSIONAL_COLORS.bullish
                                            }} />
                                        </Box>
                                    </ListItemIcon>
                                    <ListItemText
                                        primary={
                                            <Typography variant="body2" sx={{
                                                color: PROFESSIONAL_COLORS.text,
                                                fontWeight: 600,
                                                mb: 0.5
                                            }}>
                                                {item.title}
                                            </Typography>
                                        }
                                        secondary={
                                            <Box>
                                                <Typography variant="caption" sx={{
                                                    color: PROFESSIONAL_COLORS.textSecondary,
                                                    display: 'block',
                                                    mb: 0.5
                                                }}>
                                                    {item.summary}
                                                </Typography>
                                                <Stack direction="row" justifyContent="space-between" alignItems="center">
                                                    <Typography variant="caption" sx={{ color: PROFESSIONAL_COLORS.textSecondary }}>
                                                        {item.source} • {item.timestamp}
                                                    </Typography>
                                                    <Stack direction="row" spacing={0.5}>
                                                        <Chip
                                                            label={item.impact.toUpperCase()}
                                                            size="small"
                                                            sx={{
                                                                height: 18,
                                                                fontSize: '0.6rem',
                                                                bgcolor: item.impact === 'high' ? alpha(PROFESSIONAL_COLORS.bearish, 0.2) : alpha(PROFESSIONAL_COLORS.bullish, 0.2),
                                                                color: item.impact === 'high' ? PROFESSIONAL_COLORS.bearish : PROFESSIONAL_COLORS.bullish,
                                                            }}
                                                        />
                                                        <Chip
                                                            label={item.sentiment.toUpperCase()}
                                                            size="small"
                                                            sx={{
                                                                height: 18,
                                                                fontSize: '0.6rem',
                                                                bgcolor: alpha(PROFESSIONAL_COLORS.accent, 0.2),
                                                                color: PROFESSIONAL_COLORS.accent,
                                                            }}
                                                        />
                                                    </Stack>
                                                </Stack>
                                            </Box>
                                        }
                                    />
                                    <IconButton size="small">
                                        <Launch sx={{ color: PROFESSIONAL_COLORS.textSecondary, fontSize: 16 }} />
                                    </IconButton>
                                </ListItemButton>
                            </motion.div>
                        ))}
                </List>
            </CardContent>
        </Card>
    );

    // Get watchlist with live data
    const getWatchlistWithLiveData = useMemo(() => {
        return watchlist.map(symbol => {
            const liveData = liveMarketData.get(symbol);
            const baseStock = TRENDING_STOCKS.find(s => s.symbol === symbol);

            if (liveData) {
                return {
                    symbol,
                    name: baseStock?.name || symbol,
                    price: liveData.price,
                    change: liveData.change,
                    changePercent: liveData.change_percent,
                    volume: liveData.volume.toString(),
                };
            }

            return baseStock || {
                symbol,
                name: symbol,
                price: 0,
                change: 0,
                changePercent: 0,
                volume: '0',
            };
        });
    }, [watchlist, liveMarketData]);

    // Watchlist Component
    const WatchlistComponent = () => (
        <Card sx={{
            height: '100%',
            background: PROFESSIONAL_COLORS.card,
            border: `1px solid ${PROFESSIONAL_COLORS.border}`,
            ...utilityClasses.glassmorphism
        }}>
            <CardContent sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                    <Typography variant="h6" sx={{
                        color: PROFESSIONAL_COLORS.text,
                        fontWeight: 700,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1
                    }}>
                        <Bookmark sx={{ color: PROFESSIONAL_COLORS.accent }} />
                        Watchlist
                    </Typography>
                    <Stack direction="row" alignItems="center" spacing={1}>
                        {isConnected && (
                            <Chip
                                label="LIVE"
                                size="small"
                                sx={{
                                    height: 20,
                                    fontSize: '0.6rem',
                                    bgcolor: alpha(PROFESSIONAL_COLORS.bullish, 0.2),
                                    color: PROFESSIONAL_COLORS.bullish,
                                    fontWeight: 700,
                                    animation: 'pulse 2s infinite'
                                }}
                            />
                        )}
                        <Badge badgeContent={watchlist.length} color="primary">
                            <Visibility sx={{ color: PROFESSIONAL_COLORS.textSecondary }} />
                        </Badge>
                    </Stack>
                </Box>

                <List dense sx={{ maxHeight: 300, overflow: 'auto' }}>
                    {getWatchlistWithLiveData.map((stock, index) => (
                        <motion.div
                            key={stock.symbol}
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: index * 0.1 }}
                        >
                            <ListItemButton
                                onClick={() => handleSymbolSelect(stock.symbol)}
                                sx={{
                                    borderRadius: 1,
                                    mb: 0.5,
                                    border: selectedSymbol === stock.symbol ? `1px solid ${PROFESSIONAL_COLORS.accent}` : 'none',
                                    background: selectedSymbol === stock.symbol ? alpha(PROFESSIONAL_COLORS.accent, 0.1) : 'transparent',
                                    '&:hover': {
                                        background: alpha(PROFESSIONAL_COLORS.accent, 0.05),
                                    }
                                }}
                            >
                                <ListItemIcon sx={{ minWidth: 40 }}>
                                    <ShowChart sx={{
                                        color: PROFESSIONAL_COLORS.accent,
                                        animation: updatingSymbols.has(stock.symbol) ? 'pulse 1s ease-in-out' : 'none',
                                        filter: updatingSymbols.has(stock.symbol) ? `drop-shadow(0 0 5px ${PROFESSIONAL_COLORS.accent})` : 'none',
                                    }} />
                                </ListItemIcon>
                                <ListItemText
                                    primary={
                                        <Stack direction="row" justifyContent="space-between" alignItems="center">
                                            <Typography variant="body2" sx={{ color: PROFESSIONAL_COLORS.text, fontWeight: 600 }}>
                                                {stock.symbol}
                                            </Typography>
                                            <Typography variant="body2" sx={{ color: PROFESSIONAL_COLORS.text, fontWeight: 700 }}>
                                                ${stock.price.toFixed(2)}
                                            </Typography>
                                        </Stack>
                                    }
                                    secondary={
                                        <Stack direction="row" justifyContent="space-between" alignItems="center">
                                            <Typography variant="caption" sx={{ color: PROFESSIONAL_COLORS.textSecondary }}>
                                                {stock.name}
                                            </Typography>
                                            <Stack direction="row" alignItems="center" spacing={0.5}>
                                                <Typography
                                                    variant="caption"
                                                    sx={{
                                                        color: stock.change > 0 ? PROFESSIONAL_COLORS.bullish : PROFESSIONAL_COLORS.bearish,
                                                        fontWeight: 600
                                                    }}
                                                >
                                                    {stock.change > 0 ? '+' : ''}{stock.change.toFixed(2)}
                                                </Typography>
                                                <Typography
                                                    variant="caption"
                                                    sx={{
                                                        color: stock.change > 0 ? PROFESSIONAL_COLORS.bullish : PROFESSIONAL_COLORS.bearish,
                                                        fontWeight: 600
                                                    }}
                                                >
                                                    ({stock.changePercent > 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%)
                                                </Typography>
                                            </Stack>
                                        </Stack>
                                    }
                                />
                                <IconButton
                                    size="small"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        handleRemoveFromWatchlist(stock.symbol);
                                    }}
                                    sx={{ ml: 1 }}
                                >
                                    <Remove sx={{ color: PROFESSIONAL_COLORS.bearish, fontSize: 16 }} />
                                </IconButton>
                            </ListItemButton>
                        </motion.div>
                    ))}
                </List>
            </CardContent>
        </Card>
    );

    return (
        <Box sx={{
            height: '100vh',
            background: PROFESSIONAL_COLORS.background,
            overflow: 'hidden',
            '& @keyframes pulse': {
                '0%': { opacity: 1, transform: 'scale(1)' },
                '50%': { opacity: 0.7, transform: 'scale(1.05)' },
                '100%': { opacity: 1, transform: 'scale(1)' }
            }
        }}>
            {/* Professional Header */}
            <Paper sx={{
                background: PROFESSIONAL_COLORS.surface,
                borderBottom: `1px solid ${PROFESSIONAL_COLORS.border}`,
                px: 3,
                py: 1.5,
                ...utilityClasses.glassmorphism
            }}>
                <Stack direction="row" alignItems="center" justifyContent="space-between">
                    <Stack direction="row" alignItems="center" spacing={2}>
                        <Typography variant="h5" sx={{
                            color: PROFESSIONAL_COLORS.text,
                            fontWeight: 700,
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1
                        }}>
                            <AutoAwesome sx={{ color: PROFESSIONAL_COLORS.accent }} />
                            GoldenSignals AI
                        </Typography>
                        <Chip
                            label="PROPHET MODE"
                            size="small"
                            sx={{
                                bgcolor: alpha(PROFESSIONAL_COLORS.accent, 0.2),
                                color: PROFESSIONAL_COLORS.accent,
                                fontWeight: 700
                            }}
                        />
                    </Stack>

                    <Stack direction="row" alignItems="center" spacing={2}>
                        <Typography variant="h6" sx={{ color: PROFESSIONAL_COLORS.text }}>
                            {selectedSymbol}
                        </Typography>

                        {/* Live Connection Status */}
                        <Stack direction="row" alignItems="center" spacing={1}>
                            <Box sx={{
                                width: 8,
                                height: 8,
                                borderRadius: '50%',
                                bgcolor: isConnected ? PROFESSIONAL_COLORS.bullish : PROFESSIONAL_COLORS.bearish,
                                animation: isConnected ? 'pulse 2s infinite' : 'none',
                            }} />
                            <Typography variant="caption" sx={{
                                color: isConnected ? PROFESSIONAL_COLORS.bullish : PROFESSIONAL_COLORS.bearish,
                                fontWeight: 600
                            }}>
                                {isConnected ? 'LIVE' : 'OFFLINE'}
                            </Typography>
                        </Stack>

                        {/* Live Metrics */}
                        {liveMetrics && (
                            <Stack direction="row" spacing={2}>
                                <Typography variant="caption" sx={{ color: PROFESSIONAL_COLORS.textSecondary }}>
                                    Signals Today: {liveMetrics.signals_generated_today}
                                </Typography>
                                <Typography variant="caption" sx={{ color: PROFESSIONAL_COLORS.textSecondary }}>
                                    Health: {liveMetrics.system_health}
                                </Typography>
                            </Stack>
                        )}

                        <Typography variant="body2" sx={{ color: PROFESSIONAL_COLORS.textSecondary }}>
                            {new Date().toLocaleTimeString()}
                        </Typography>
                        <IconButton onClick={() => setAiChatOpen(true)}>
                            <Psychology sx={{ color: PROFESSIONAL_COLORS.accent }} />
                        </IconButton>
                    </Stack>
                </Stack>
            </Paper>

            {/* Main Dashboard Grid */}
            <Grid container sx={{ height: 'calc(100vh - 80px)' }}>
                {/* Left Sidebar - Trending Stocks */}
                <Grid item xs={12} md={3} sx={{ height: '100%' }}>
                    <Box sx={{ p: 2, height: '100%' }}>
                        <TrendingStocks />
                    </Box>
                </Grid>

                {/* Central Chart */}
                <Grid item xs={12} md={6} sx={{ height: '100%' }}>
                    <Box sx={{ p: 2, height: '100%' }}>
                        <Card sx={{
                            height: '100%',
                            background: PROFESSIONAL_COLORS.card,
                            border: `1px solid ${PROFESSIONAL_COLORS.border}`,
                            ...utilityClasses.glassmorphism
                        }}>
                            <CardContent sx={{ p: 2, height: '100%' }}>
                                <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2 }}>
                                    <Typography variant="h6" sx={{
                                        color: PROFESSIONAL_COLORS.text,
                                        fontWeight: 700,
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 1
                                    }}>
                                        <CandlestickChart sx={{ color: PROFESSIONAL_COLORS.accent }} />
                                        {selectedSymbol} Chart
                                    </Typography>
                                    <Stack direction="row" spacing={1}>
                                        <Chip
                                            label="1D"
                                            size="small"
                                            variant="outlined"
                                            sx={{ color: PROFESSIONAL_COLORS.textSecondary }}
                                        />
                                        <Chip
                                            label="LIVE"
                                            size="small"
                                            sx={{
                                                bgcolor: alpha(PROFESSIONAL_COLORS.bullish, 0.2),
                                                color: PROFESSIONAL_COLORS.bullish,
                                                fontWeight: 700
                                            }}
                                        />
                                    </Stack>
                                </Stack>

                                <Box sx={{ height: 'calc(100% - 60px)' }}>
                                    <UnifiedChart
                                        symbol={selectedSymbol}
                                        timeframe="1d"
                                        height="100%"
                                        showAdvancedFeatures={true}
                                        signals={liveSignals.filter(s => s.symbol === selectedSymbol).map(signal => ({
                                            id: signal.signal_id,
                                            type: signal.signal_type,
                                            symbol: signal.symbol,
                                            timestamp: new Date(signal.created_at).getTime(),
                                            price: signal.price,
                                            confidence: signal.confidence,
                                            entry: signal.price,
                                            reasoning: signal.reasoning,
                                        }))}
                                        theme="dark"
                                    />
                                </Box>
                            </CardContent>
                        </Card>
                    </Box>
                </Grid>

                {/* Right Sidebar - News and Watchlist */}
                <Grid item xs={12} md={3} sx={{ height: '100%' }}>
                    <Box sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column', gap: 2 }}>
                        {/* News Component */}
                        <Box sx={{ flex: 1 }}>
                            <NewsComponent />
                        </Box>

                        {/* Watchlist Component */}
                        <Box sx={{ flex: 0.6 }}>
                            <WatchlistComponent />
                        </Box>
                    </Box>
                </Grid>
            </Grid>

            {/* AI Insights Panel */}
            <Paper sx={{
                position: 'fixed',
                bottom: 0,
                left: 0,
                right: 0,
                height: 120,
                background: PROFESSIONAL_COLORS.surface,
                borderTop: `1px solid ${PROFESSIONAL_COLORS.border}`,
                ...utilityClasses.glassmorphism,
                zIndex: 1200
            }}>
                <Box sx={{ p: 2, height: '100%' }}>
                    <AIInsightsPanel
                        symbol={selectedSymbol}
                        signals={liveSignals.filter(s => s.symbol === selectedSymbol).map(signal => ({
                            id: signal.signal_id,
                            type: signal.signal_type,
                            symbol: signal.symbol,
                            timestamp: new Date(signal.created_at).getTime(),
                            price: signal.price,
                            confidence: signal.confidence,
                            entry: signal.price,
                            reasoning: signal.reasoning,
                        }))}
                        maxInsights={5}
                        enableLiveUpdates={true}
                        showFilters={false}
                        showPriorityBadges={true}
                        autoRefresh={true}
                        refreshInterval={30000}
                        height={80}
                        compact={true}
                    />
                </Box>
            </Paper>

            {/* AI Prophet Widget */}
            <FloatingAIProphetWidget
                onClick={() => setAiChatOpen(true)}
                isVisible={true}
            />

            {/* AI Chat Modal */}
            {aiChatOpen && (
                <UnifiedAIChat
                    mode="floating"
                    isOpen={aiChatOpen}
                    onClose={() => setAiChatOpen(false)}
                    symbol={selectedSymbol}
                    signals={liveSignals}
                />
            )}
        </Box>
    );
};

export default ProfessionalTradingDashboard;
