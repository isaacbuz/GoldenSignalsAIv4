/**
 * GoldenSignalsAI - Professional Options Signal Dashboard
 * 
 * A reimagined, signal-first interface for professional options traders
 * Inspired by Bloomberg Terminal, ThinkorSwim, and modern fintech apps
 */

import React, { useState, useEffect, useMemo } from 'react';
import { useOutletContext } from 'react-router-dom';
import {
    Box,
    Container,
    Card,
    Typography,
    Chip,
    IconButton,
    Button,
    Stack,
    Badge,
    Tooltip,
    useTheme,
    alpha,
    Tabs,
    Tab,
    Divider,
    LinearProgress,
    Skeleton,
    Autocomplete,
    TextField,
    InputAdornment,
    Paper,
    Fade,
    Zoom,
    Avatar,
    ButtonGroup,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    Timer,
    Speed,
    Warning,
    CheckCircle,
    Notifications,
    FilterList,
    ExpandMore,
    Psychology,
    ShowChart,
    AttachMoney,
    Schedule,
    PriorityHigh,
    Visibility,
    AccessTime,
    BarChart,
    Search,
    Refresh,
    AutoAwesome,
    Insights,
    QueryStats,
    CandlestickChart,
    Analytics,
    TimerOff,
    ContentCopy,
    Share,
    InfoOutlined,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../../services/api';
import SignalCard from '../../components/Signals/SignalCard';
import SignalDetailsModal from '../../components/Signals/SignalDetailsModal';
import TradingChart from '../../components/Chart/TradingChart';
import AIExplanationPanel from '../../components/AI/AIExplanationPanel';
import MarketScreener from '../../components/Signals/MarketScreener';
import MiniHeatMap from '../../components/Market/MiniHeatMap';
import { PreciseOptionsSignal, SignalFilters } from '../../types/signals';
import { useWebSocketConnection } from '../../services/websocket';

// Popular tickers for quick access
const POPULAR_TICKERS = [
    { symbol: 'SPY', name: 'SPDR S&P 500 ETF' },
    { symbol: 'QQQ', name: 'Invesco QQQ Trust' },
    { symbol: 'AAPL', name: 'Apple Inc.' },
    { symbol: 'MSFT', name: 'Microsoft Corporation' },
    { symbol: 'NVDA', name: 'NVIDIA Corporation' },
    { symbol: 'TSLA', name: 'Tesla, Inc.' },
    { symbol: 'AMD', name: 'Advanced Micro Devices' },
    { symbol: 'META', name: 'Meta Platforms' },
    { symbol: 'AMZN', name: 'Amazon.com' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.' },
];

// Timeframe options
const TIMEFRAMES = [
    { value: '1m', label: '1M', minutes: 1 },
    { value: '5m', label: '5M', minutes: 5 },
    { value: '15m', label: '15M', minutes: 15 },
    { value: '30m', label: '30M', minutes: 30 },
    { value: '1h', label: '1H', minutes: 60 },
    { value: '4h', label: '4H', minutes: 240 },
    { value: '1d', label: '1D', minutes: 1440 },
];

const SignalsDashboard: React.FC = () => {
    const theme = useTheme();
    const queryClient = useQueryClient();
    const { selectedSymbol: contextSymbol } = useOutletContext<{ selectedSymbol: string }>();

    // Enable WebSocket for real-time updates
    const isConnected = useWebSocketConnection();

    // Enhanced Paper styles for consistent design
    const enhancedPaperStyles = {
        background: alpha(theme.palette.background.paper, 0.8),
        backdropFilter: 'blur(10px)',
        border: `1px solid ${alpha(theme.palette.divider, 0.08)}`,
        borderRadius: 3,
        boxShadow: `0 4px 24px ${alpha(theme.palette.common.black, 0.06)}`,
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        '&:hover': {
            boxShadow: `0 8px 32px ${alpha(theme.palette.common.black, 0.08)}`,
            transform: 'translateY(-2px)',
        },
    };

    // Custom scrollbar styles
    const scrollbarStyles = {
        '&::-webkit-scrollbar': {
            width: '8px',
            height: '8px',
        },
        '&::-webkit-scrollbar-track': {
            background: alpha(theme.palette.background.paper, 0.1),
            borderRadius: '4px',
        },
        '&::-webkit-scrollbar-thumb': {
            background: alpha(theme.palette.primary.main, 0.3),
            borderRadius: '4px',
            '&:hover': {
                background: alpha(theme.palette.primary.main, 0.5),
            },
        },
    };

    // State
    const [selectedSymbol, setSelectedSymbol] = useState(contextSymbol || 'SPY');
    const [searchInput, setSearchInput] = useState('');
    const [selectedTimeframe, setSelectedTimeframe] = useState('15m');
    const [selectedSignal, setSelectedSignal] = useState<PreciseOptionsSignal | null>(null);
    const [detailsModalOpen, setDetailsModalOpen] = useState(false);
    const [filters, setFilters] = useState<SignalFilters>({
        signalType: 'all',
        minConfidence: 0,
        timeframe: 'all',
    });
    const [isGeneratingSignals, setIsGeneratingSignals] = useState(false);

    // Listen for symbol changes from context and custom events
    useEffect(() => {
        if (contextSymbol && contextSymbol !== selectedSymbol) {
            handleSymbolChange(contextSymbol);
        }
    }, [contextSymbol]);

    useEffect(() => {
        const handleSymbolChangeEvent = (event: CustomEvent) => {
            if (event.detail?.symbol) {
                handleSymbolChange(event.detail.symbol);
            }
        };

        window.addEventListener('symbol-change', handleSymbolChangeEvent as EventListener);
        return () => {
            window.removeEventListener('symbol-change', handleSymbolChangeEvent as EventListener);
        };
    }, []);

    // Fetch signals with real-time updates
    const { data: signals = [], isLoading: signalsLoading, refetch: refetchSignals } = useQuery({
        queryKey: ['preciseOptionsSignals', selectedSymbol, selectedTimeframe],
        queryFn: () => apiClient.getPreciseOptionsSignals(selectedSymbol, selectedTimeframe),
        refetchInterval: selectedTimeframe === '1m' ? 60000 :
            selectedTimeframe === '5m' ? 300000 :
                selectedTimeframe === '15m' ? 900000 : 1800000,
    });

    // Fetch market data
    const { data: marketData } = useQuery({
        queryKey: ['marketData', selectedSymbol, selectedTimeframe],
        queryFn: () => apiClient.getMarketData(selectedSymbol),
        refetchInterval: 30000,
    });

    // Fetch market news
    const { data: marketNews = [], isLoading: newsLoading, refetch: refetchNews } = useQuery({
        queryKey: ['marketNews'],
        queryFn: () => apiClient.getMarketNews(),
        refetchInterval: 30000, // Refresh every 30 seconds
    });

    // Fetch AI insights for selected signal
    const { data: aiInsights } = useQuery({
        queryKey: ['aiInsights', selectedSignal?.signal_id || selectedSignal?.id],
        queryFn: () => selectedSignal ? apiClient.getAIInsights(selectedSignal.signal_id || selectedSignal.id) : null,
        enabled: !!selectedSignal && !!(selectedSignal.signal_id || selectedSignal.id),
    });

    // Handle symbol search and selection
    const handleSymbolChange = async (newSymbol: string) => {
        if (!newSymbol) return;

        setSelectedSymbol(newSymbol.toUpperCase());
        setIsGeneratingSignals(true);

        // Trigger backend signal generation
        try {
            await apiClient.generateSignalsForSymbol(newSymbol, selectedTimeframe);
            await refetchSignals();
        } catch (error) {
            console.error('Error generating signals:', error);
        } finally {
            setIsGeneratingSignals(false);
        }
    };

    // Filter and group signals
    const groupedSignals = useMemo(() => {
        const filtered = signals.filter(signal => {
            if (filters.signalType !== 'all' && signal.type !== filters.signalType) return false;
            if (filters.minConfidence && signal.confidence < filters.minConfidence) return false;
            return true;
        });

        const now = new Date();
        const urgent: PreciseOptionsSignal[] = [];
        const today: PreciseOptionsSignal[] = [];
        const upcoming: PreciseOptionsSignal[] = [];

        filtered.forEach(signal => {
            const signalTime = new Date(signal.timestamp);
            const hoursUntil = (signalTime.getTime() - now.getTime()) / (1000 * 60 * 60);

            if (hoursUntil <= 1) urgent.push(signal);
            else if (hoursUntil <= 24) today.push(signal);
            else upcoming.push(signal);
        });

        return { urgent, today, upcoming };
    }, [signals, filters]);

    // Auto-select first signal when new signals arrive
    useEffect(() => {
        if (signals.length > 0 && !selectedSignal) {
            setSelectedSignal(signals[0]);
        }
    }, [signals, selectedSignal]);

    // Helper function to get timeframe label
    const getTimeframeLabel = (tf: string) => {
        const timeframe = TIMEFRAMES.find(t => t.value === tf);
        return timeframe ? `${timeframe.label} Chart` : tf;
    };

    // Helper function to format time ago
    const formatTimeAgo = (timestamp: string) => {
        const now = new Date();
        const time = new Date(timestamp);
        const diff = Math.floor((now.getTime() - time.getTime()) / 1000); // seconds

        if (diff < 60) return 'just now';
        if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
        if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
        return `${Math.floor(diff / 86400)}d ago`;
    };

    return (
        <Container maxWidth={false} sx={{
            p: { xs: 2, sm: 3, lg: 4 },
            minHeight: '100vh',
            display: 'flex',
            flexDirection: 'column',
            background: `linear-gradient(180deg, ${alpha(theme.palette.primary.main, 0.02)} 0%, transparent 100%)`,
        }}>
            {/* Main Content Area */}
            <Box sx={{
                flex: 1,
                display: 'flex',
                gap: 3,
                minHeight: 0,
                height: '100%',
                overflow: 'hidden'
            }}>
                {/* Left Sidebar - Signals & Market Screener */}
                <Box sx={{
                    width: '320px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 2,
                    minHeight: 0,
                    height: '100%'
                }}>
                    {/* Active Signals Panel */}
                    <Paper
                        elevation={0}
                        sx={{
                            flex: '1 1 60%',
                            p: 2.5,
                            background: alpha(theme.palette.background.paper, 0.8),
                            backdropFilter: 'blur(10px)',
                            border: `1px solid ${alpha(theme.palette.divider, 0.08)}`,
                            borderRadius: 3,
                            display: 'flex',
                            flexDirection: 'column',
                            minHeight: 0,
                            boxShadow: `0 4px 24px ${alpha(theme.palette.common.black, 0.06)}`,
                            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                            '&:hover': {
                                boxShadow: `0 8px 32px ${alpha(theme.palette.common.black, 0.08)}`,
                                transform: 'translateY(-2px)',
                            },
                        }}
                    >
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2, flexShrink: 0 }}>
                            <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Insights sx={{ color: theme.palette.primary.main }} />
                                {selectedSymbol} Analysis
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                {isGeneratingSignals && (
                                    <Chip
                                        icon={<Psychology />}
                                        label="Analyzing..."
                                        color="primary"
                                        size="small"
                                        sx={{
                                            background: alpha(theme.palette.primary.main, 0.1),
                                            animation: 'pulse 2s infinite',
                                            '@keyframes pulse': {
                                                '0%': { opacity: 1 },
                                                '50%': { opacity: 0.5 },
                                                '100%': { opacity: 1 },
                                            },
                                        }}
                                    />
                                )}
                                <Tooltip title="Refresh analysis">
                                    <IconButton
                                        onClick={() => refetchSignals()}
                                        disabled={isGeneratingSignals}
                                        size="small"
                                        sx={{ ml: 0.5 }}
                                    >
                                        <Refresh fontSize="small" className={isGeneratingSignals ? 'rotating' : ''} />
                                    </IconButton>
                                </Tooltip>
                            </Box>
                        </Box>

                        {isGeneratingSignals && (
                            <Box sx={{ mb: 2 }}>
                                <LinearProgress sx={{ height: 2 }} />
                                <Typography variant="caption" color="primary" sx={{ mt: 0.5 }}>
                                    AI agents analyzing {selectedSymbol}...
                                </Typography>
                            </Box>
                        )}

                        <Box sx={{ flex: 1, overflow: 'auto', minHeight: 0, ...scrollbarStyles }}>
                            {signalsLoading ? (
                                <Box sx={{ p: 1 }}>
                                    {[1, 2, 3].map(i => (
                                        <Skeleton
                                            key={i}
                                            variant="rectangular"
                                            height={80}
                                            sx={{
                                                mb: 2,
                                                borderRadius: 2,
                                                '&::after': {
                                                    background: `linear-gradient(90deg, transparent, ${alpha(theme.palette.primary.main, 0.1)}, transparent)`,
                                                }
                                            }}
                                        />
                                    ))}
                                </Box>
                            ) : (
                                <Stack spacing={2.5}>
                                    {/* Current Signal Summary */}
                                    {signals.length > 0 && signals[0] && (
                                        <>
                                            <Box>
                                                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                                    Latest Signal
                                                </Typography>
                                                <Box sx={{
                                                    p: 2.5,
                                                    borderRadius: 2,
                                                    background: `linear-gradient(135deg, ${alpha(signals[0].signal_type === 'BUY_CALL' ? theme.palette.success.main : theme.palette.error.main, 0.15)} 0%, ${alpha(signals[0].signal_type === 'BUY_CALL' ? theme.palette.success.main : theme.palette.error.main, 0.05)} 100%)`,
                                                    border: `2px solid ${alpha(signals[0].signal_type === 'BUY_CALL' ? theme.palette.success.main : theme.palette.error.main, 0.3)}`,
                                                    boxShadow: `0 4px 16px ${alpha(signals[0].signal_type === 'BUY_CALL' ? theme.palette.success.main : theme.palette.error.main, 0.15)}`,
                                                    position: 'relative',
                                                    overflow: 'hidden',
                                                    '&::before': {
                                                        content: '""',
                                                        position: 'absolute',
                                                        top: -2,
                                                        left: -2,
                                                        right: -2,
                                                        bottom: -2,
                                                        background: `linear-gradient(45deg, ${signals[0].signal_type === 'BUY_CALL' ? theme.palette.success.main : theme.palette.error.main}, transparent)`,
                                                        opacity: 0.1,
                                                        zIndex: 0,
                                                    },
                                                    '& > *': {
                                                        position: 'relative',
                                                        zIndex: 1,
                                                    },
                                                }}>
                                                    <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                        {signals[0].signal_type === 'BUY_CALL' ? <TrendingUp /> : <TrendingDown />}
                                                        {signals[0].signal_type === 'BUY_CALL' ? 'CALL' : 'PUT'}
                                                    </Typography>
                                                    <Stack spacing={0.5} sx={{ mt: 1 }}>
                                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                            <Typography variant="body2" color="text.secondary">
                                                                Strike Price:
                                                            </Typography>
                                                            <Typography variant="body1" fontWeight="bold" color="text.primary">
                                                                ${signals[0].strike_price}
                                                            </Typography>
                                                            {marketData?.price && (
                                                                <Chip
                                                                    label={
                                                                        signals[0].signal_type === 'BUY_CALL'
                                                                            ? signals[0].strike_price > marketData.price
                                                                                ? `OTM +${((signals[0].strike_price - marketData.price) / marketData.price * 100).toFixed(1)}%`
                                                                                : signals[0].strike_price < marketData.price
                                                                                    ? `ITM -${((marketData.price - signals[0].strike_price) / marketData.price * 100).toFixed(1)}%`
                                                                                    : 'ATM'
                                                                            : signals[0].strike_price < marketData.price
                                                                                ? `OTM -${((marketData.price - signals[0].strike_price) / marketData.price * 100).toFixed(1)}%`
                                                                                : signals[0].strike_price > marketData.price
                                                                                    ? `ITM +${((signals[0].strike_price - marketData.price) / marketData.price * 100).toFixed(1)}%`
                                                                                    : 'ATM'
                                                                    }
                                                                    size="small"
                                                                    color={
                                                                        signals[0].signal_type === 'BUY_CALL'
                                                                            ? signals[0].strike_price <= marketData.price ? "success" : "default"
                                                                            : signals[0].strike_price >= marketData.price ? "success" : "default"
                                                                    }
                                                                    sx={{ fontSize: '0.7rem', height: 20 }}
                                                                />
                                                            )}
                                                        </Box>
                                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                            <Typography variant="body2" color="text.secondary">
                                                                Expiration:
                                                            </Typography>
                                                            <Typography variant="body2">
                                                                {new Date(signals[0].expiration_date).toLocaleDateString('en-US', {
                                                                    month: 'short',
                                                                    day: 'numeric',
                                                                    year: 'numeric'
                                                                })}
                                                            </Typography>
                                                            <Typography variant="caption" color="text.secondary">
                                                                ({Math.ceil((new Date(signals[0].expiration_date).getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24))} days)
                                                            </Typography>
                                                        </Box>
                                                    </Stack>
                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                                                        <Chip
                                                            label={`${signals[0].confidence}% confidence`}
                                                            size="small"
                                                            color={signals[0].confidence >= 85 ? "success" : "default"}
                                                        />
                                                        <Chip
                                                            label={getTimeframeLabel(selectedTimeframe)}
                                                            size="small"
                                                            variant="outlined"
                                                        />
                                                    </Box>
                                                </Box>
                                            </Box>

                                            <Divider />
                                        </>
                                    )}

                                    {/* Key Metrics */}
                                    <Box>
                                        <Typography variant="subtitle2" color="text.secondary" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
                                            Key Metrics
                                        </Typography>
                                        <Stack spacing={2}>
                                            <Box sx={{
                                                display: 'flex',
                                                justifyContent: 'space-between',
                                                alignItems: 'center',
                                                p: 1.5,
                                                borderRadius: 1.5,
                                                backgroundColor: alpha(theme.palette.background.default, 0.5),
                                                transition: 'all 0.2s',
                                                '&:hover': {
                                                    backgroundColor: alpha(theme.palette.primary.main, 0.05),
                                                }
                                            }}>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <AttachMoney sx={{ fontSize: 18, color: theme.palette.primary.main }} />
                                                    <Typography variant="body2">Current Price</Typography>
                                                </Box>
                                                <Typography
                                                    variant="body1"
                                                    fontWeight="bold"
                                                    sx={{
                                                        transition: 'all 0.3s ease',
                                                        '&:hover': {
                                                            transform: 'scale(1.05)',
                                                        }
                                                    }}
                                                >
                                                    ${marketData?.price?.toFixed(2) || '--'}
                                                </Typography>
                                            </Box>
                                            <Box sx={{
                                                display: 'flex',
                                                justifyContent: 'space-between',
                                                alignItems: 'center',
                                                p: 1.5,
                                                borderRadius: 1.5,
                                                backgroundColor: alpha(theme.palette.background.default, 0.5),
                                                transition: 'all 0.2s',
                                                '&:hover': {
                                                    backgroundColor: alpha(
                                                        marketData?.change_percent && marketData.change_percent >= 0
                                                            ? theme.palette.success.main
                                                            : theme.palette.error.main,
                                                        0.05
                                                    ),
                                                }
                                            }}>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    {marketData?.change_percent && marketData.change_percent >= 0 ? (
                                                        <TrendingUp sx={{ fontSize: 18, color: theme.palette.success.main }} />
                                                    ) : (
                                                        <TrendingDown sx={{ fontSize: 18, color: theme.palette.error.main }} />
                                                    )}
                                                    <Typography variant="body2">Day Change</Typography>
                                                </Box>
                                                <Typography
                                                    variant="body1"
                                                    fontWeight="bold"
                                                    color={marketData?.change_percent && marketData.change_percent >= 0 ? 'success.main' : 'error.main'}
                                                >
                                                    {marketData?.change_percent ? `${marketData.change_percent >= 0 ? '+' : ''}${marketData.change_percent.toFixed(2)}%` : '--'}
                                                </Typography>
                                            </Box>
                                            <Box sx={{
                                                display: 'flex',
                                                justifyContent: 'space-between',
                                                alignItems: 'center',
                                                p: 1.5,
                                                borderRadius: 1.5,
                                                backgroundColor: alpha(theme.palette.background.default, 0.5),
                                                transition: 'all 0.2s',
                                                '&:hover': {
                                                    backgroundColor: alpha(theme.palette.info.main, 0.05),
                                                }
                                            }}>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <BarChart sx={{ fontSize: 18, color: theme.palette.info.main }} />
                                                    <Typography variant="body2">Volume</Typography>
                                                </Box>
                                                <Typography variant="body1" fontWeight="bold">
                                                    {marketData?.volume ? `${(marketData.volume / 1000000).toFixed(1)}M` : '--'}
                                                </Typography>
                                            </Box>
                                            <Box sx={{
                                                display: 'flex',
                                                justifyContent: 'space-between',
                                                alignItems: 'center',
                                                p: 1.5,
                                                borderRadius: 1.5,
                                                backgroundColor: alpha(theme.palette.background.default, 0.5),
                                                transition: 'all 0.2s',
                                                '&:hover': {
                                                    backgroundColor: alpha(theme.palette.warning.main, 0.05),
                                                }
                                            }}>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <ShowChart sx={{ fontSize: 18, color: theme.palette.warning.main }} />
                                                    <Typography variant="body2">Day Range</Typography>
                                                </Box>
                                                <Typography variant="body1" fontWeight="bold" sx={{ fontSize: '0.875rem' }}>
                                                    {marketData ? `$${(marketData.price - Math.abs(marketData.change)).toFixed(2)} - $${(marketData.price + Math.abs(marketData.change)).toFixed(2)}` : '--'}
                                                </Typography>
                                            </Box>
                                        </Stack>
                                    </Box>

                                    <Divider />

                                    {/* AI Insights Summary */}
                                    <Box>
                                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                            AI Insights
                                        </Typography>
                                        {aiInsights ? (
                                            <Stack spacing={1}>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <Box sx={{
                                                        width: 8,
                                                        height: 8,
                                                        borderRadius: '50%',
                                                        backgroundColor: aiInsights.sentiment === 'bullish' ? theme.palette.success.main :
                                                            aiInsights.sentiment === 'bearish' ? theme.palette.error.main :
                                                                theme.palette.warning.main
                                                    }} />
                                                    <Typography variant="body2" textTransform="capitalize">
                                                        {typeof aiInsights.sentiment === 'string'
                                                            ? aiInsights.sentiment
                                                            : aiInsights.sentiment?.overall || 'Neutral'} Sentiment
                                                    </Typography>
                                                </Box>
                                                <Typography variant="caption" color="text.secondary">
                                                    {aiInsights.summary}
                                                </Typography>
                                            </Stack>
                                        ) : (
                                            <Typography variant="body2" color="text.secondary">
                                                Loading insights...
                                            </Typography>
                                        )}
                                    </Box>

                                    {/* Quick Actions */}
                                    {selectedSignal && (
                                        <>
                                            <Divider />
                                            <Box>
                                                <Button
                                                    fullWidth
                                                    variant="contained"
                                                    size="small"
                                                    onClick={() => setDetailsModalOpen(true)}
                                                    startIcon={<InfoOutlined />}
                                                >
                                                    View Full Analysis
                                                </Button>
                                            </Box>
                                        </>
                                    )}

                                    {signals.length === 0 && (
                                        <Box textAlign="center" py={4}>
                                            <Typography variant="body2" color="text.secondary">
                                                No signals generated yet
                                            </Typography>
                                            <Typography variant="caption" color="text.secondary">
                                                AI is analyzing {selectedSymbol}...
                                            </Typography>
                                        </Box>
                                    )}
                                </Stack>
                            )}
                        </Box>
                    </Paper>

                    {/* Market Screener - Top Opportunities */}
                    <Paper
                        elevation={0}
                        sx={{
                            flex: '1 1 40%',
                            p: 2.5,
                            background: alpha(theme.palette.background.paper, 0.8),
                            backdropFilter: 'blur(10px)',
                            border: `1px solid ${alpha(theme.palette.divider, 0.08)}`,
                            borderRadius: 3,
                            display: 'flex',
                            flexDirection: 'column',
                            minHeight: 0,
                            boxShadow: `0 4px 24px ${alpha(theme.palette.common.black, 0.06)}`,
                            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                            '&:hover': {
                                boxShadow: `0 8px 32px ${alpha(theme.palette.common.black, 0.08)}`,
                                transform: 'translateY(-2px)',
                            },
                        }}
                    >
                        <MarketScreener
                            onSymbolSelect={(symbol: string) => {
                                handleSymbolChange(symbol);
                                // Optionally scroll to top to see the new signals
                                window.scrollTo({ top: 0, behavior: 'smooth' });
                            }}
                        />
                    </Paper>
                </Box>

                {/* Main Content - Chart & Analysis */}
                <Box sx={{
                    flex: 1,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 2,
                    minHeight: 0,
                    height: '100%',
                    overflow: 'hidden'
                }}>
                    {/* Chart with integrated header */}
                    <Paper
                        elevation={0}
                        sx={{
                            flex: 1,
                            ...enhancedPaperStyles,
                            p: 0,
                            overflow: 'hidden',
                            display: 'flex',
                            flexDirection: 'column',
                            minHeight: 0,
                        }}
                    >
                        <TradingChart
                            defaultSymbol={selectedSymbol}
                            height={700}
                            showAIInsights={true}
                            onSymbolChange={(symbol: string) => {
                                handleSymbolChange(symbol);
                            }}
                            onSelectSignal={(signal: any) => {
                                setSelectedSignal(signal);
                            }}
                        />
                    </Paper>

                    {/* AI Analysis Panel */}
                    <Paper
                        elevation={0}
                        sx={{
                            flex: '0 0 35%',
                            background: alpha(theme.palette.background.paper, 0.5),
                            border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                            borderRadius: 2,
                            overflow: 'hidden',
                            display: 'flex',
                            flexDirection: 'column',
                        }}
                    >
                        <AIExplanationPanel
                            signal={selectedSignal}
                            insights={aiInsights}
                            symbol={selectedSymbol}
                            timeframe={selectedTimeframe}
                            expanded={true}
                        />
                    </Paper>
                </Box>

                {/* Right Sidebar - Stats & Actions */}
                <Box sx={{
                    width: '280px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 2,
                    minHeight: 0,
                    height: '100%',
                    overflowY: 'auto',
                    overflowX: 'hidden',
                    pr: 1, // Add padding for scrollbar
                    ...scrollbarStyles,
                }}>
                    {/* Top Financial News */}
                    <Paper
                        elevation={0}
                        sx={{
                            p: 2,
                            background: alpha(theme.palette.background.paper, 0.5),
                            border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                            borderRadius: 2,
                        }}
                    >
                        <Typography variant="subtitle2" gutterBottom fontWeight="bold" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <TrendingUp sx={{ fontSize: 18, color: theme.palette.primary.main }} />
                            Market News
                            <Chip
                                label="LIVE"
                                size="small"
                                sx={{
                                    ml: 'auto',
                                    backgroundColor: theme.palette.error.main,
                                    color: 'white',
                                    fontSize: '0.7rem',
                                    height: 20,
                                    animation: 'pulse 2s infinite',
                                    '@keyframes pulse': {
                                        '0%': { opacity: 1 },
                                        '50%': { opacity: 0.7 },
                                        '100%': { opacity: 1 },
                                    },
                                }}
                            />
                        </Typography>
                        <Stack spacing={2} sx={{ mt: 2 }}>
                            {newsLoading ? (
                                <>
                                    {[1, 2, 3].map(i => (
                                        <Skeleton key={i} variant="rectangular" height={60} sx={{ borderRadius: 1 }} />
                                    ))}
                                </>
                            ) : marketNews.length > 0 ? (
                                marketNews.slice(0, 5).map((news, index) => (
                                    <Box
                                        key={news.id || index}
                                        sx={{
                                            pb: 1.5,
                                            borderBottom: index < 4 ? `1px solid ${alpha(theme.palette.divider, 0.1)}` : 'none',
                                            cursor: 'pointer',
                                            '&:hover': {
                                                '& .news-title': {
                                                    color: theme.palette.primary.main,
                                                },
                                            },
                                        }}
                                        onClick={() => news.url && window.open(news.url, '_blank')}
                                    >
                                        <Typography
                                            variant="body2"
                                            className="news-title"
                                            sx={{
                                                fontWeight: 500,
                                                mb: 0.5,
                                                display: '-webkit-box',
                                                WebkitLineClamp: 2,
                                                WebkitBoxOrient: 'vertical',
                                                overflow: 'hidden',
                                                transition: 'color 0.2s',
                                            }}
                                        >
                                            {news.title}
                                        </Typography>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            <Typography variant="caption" color="text.secondary">
                                                {news.source}
                                            </Typography>
                                            <Typography variant="caption" color="text.secondary">
                                                •
                                            </Typography>
                                            <Typography variant="caption" color="text.secondary">
                                                {formatTimeAgo(news.timestamp)}
                                            </Typography>
                                            {news.impact && (
                                                <>
                                                    <Typography variant="caption" color="text.secondary">
                                                        •
                                                    </Typography>
                                                    <Chip
                                                        label={news.impact}
                                                        size="small"
                                                        sx={{
                                                            height: 16,
                                                            fontSize: '0.65rem',
                                                            backgroundColor:
                                                                news.impact === 'HIGH' ? alpha(theme.palette.error.main, 0.1) :
                                                                    news.impact === 'MEDIUM' ? alpha(theme.palette.warning.main, 0.1) :
                                                                        alpha(theme.palette.info.main, 0.1),
                                                            color:
                                                                news.impact === 'HIGH' ? theme.palette.error.main :
                                                                    news.impact === 'MEDIUM' ? theme.palette.warning.main :
                                                                        theme.palette.info.main,
                                                        }}
                                                    />
                                                </>
                                            )}
                                        </Box>
                                    </Box>
                                ))
                            ) : (
                                <Typography variant="body2" color="text.secondary" textAlign="center">
                                    No market news available
                                </Typography>
                            )}
                        </Stack>
                        <Box sx={{ mt: 2, pt: 1, borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}` }}>
                            <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                <span>Updates every 30s</span>
                                <IconButton size="small" onClick={() => refetchNews()}>
                                    <Refresh sx={{ fontSize: 16 }} />
                                </IconButton>
                            </Typography>
                        </Box>
                    </Paper>

                    {/* Today's Performance */}
                    <Paper
                        elevation={0}
                        sx={{
                            p: 2,
                            background: alpha(theme.palette.background.paper, 0.5),
                            border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                            borderRadius: 2,
                        }}
                    >
                        <Typography variant="subtitle2" gutterBottom fontWeight="bold">
                            Today's Performance
                        </Typography>
                        <Stack spacing={2} sx={{ mt: 2 }}>
                            <Box>
                                <Typography variant="body2" color="text.secondary">
                                    Signals Generated
                                </Typography>
                                <Typography variant="h6">
                                    {signals.filter(s => {
                                        const today = new Date();
                                        const signalDate = new Date(s.timestamp);
                                        return signalDate.toDateString() === today.toDateString();
                                    }).length}
                                </Typography>
                            </Box>
                            <Box>
                                <Typography variant="body2" color="text.secondary">
                                    Avg Confidence
                                </Typography>
                                <Typography variant="h6" color="primary.main">
                                    {signals.length > 0
                                        ? `${Math.round(signals.reduce((sum, s) => sum + s.confidence, 0) / signals.length)}%`
                                        : '0%'
                                    }
                                </Typography>
                            </Box>
                            <Box>
                                <Typography variant="body2" color="text.secondary">
                                    High Priority
                                </Typography>
                                <Typography variant="h6" color="warning.main">
                                    {signals.filter(s => s.confidence >= 85).length}
                                </Typography>
                            </Box>
                        </Stack>
                    </Paper>

                    {/* Market Overview */}
                    <MiniHeatMap onSymbolSelect={handleSymbolChange} />
                </Box>
            </Box>

            {/* Signal Details Modal */}
            <SignalDetailsModal
                open={detailsModalOpen && !!selectedSignal}
                onClose={() => setDetailsModalOpen(false)}
                signal={selectedSignal || {} as PreciseOptionsSignal}
            />

            {/* CSS for rotating animation */}
            <style>
                {`
                    @keyframes rotate {
                        from { transform: rotate(0deg); }
                        to { transform: rotate(360deg); }
                    }
                    .rotating {
                        animation: rotate 1s linear infinite;
                    }
                `}
            </style>
        </Container>
    );
};

export default SignalsDashboard; 