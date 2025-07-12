import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
    Box,
    Grid,
    Card,
    CardContent,
    Typography,
    Chip,
    IconButton,
    Divider,
    LinearProgress,
    Fab,
    Drawer,
    AppBar,
    Toolbar,
    Badge,
    Tooltip,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Stack,
} from '@mui/material';
import {
    Psychology as PsychologyIcon,
    TrendingUp as TrendingUpIcon,
    TrendingDown as TrendingDownIcon,
    Analytics as AnalyticsIcon,
    Notifications as NotificationsIcon,
    Settings as SettingsIcon,
    Refresh as RefreshIcon,
    Fullscreen as FullscreenIcon,
    Close as CloseIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { useTheme } from '@mui/material/styles';
import { motion, AnimatePresence } from 'framer-motion';

import ProfessionalSearchBar from '../../components/Common/ProfessionalSearchBar';
import { getMarketColor, getConfidenceColor, formatCurrency, formatPercentage } from '../../theme/professional';

// Import the proper Golden Eye AI Prophet Widget
import { FloatingAIProphetWidget } from '../../components/AI/FloatingAIProphetWidget';
import { EnhancedAIProphetChat } from '../../components/AI/EnhancedAIProphetChat';
import { MassiveOptionsChart } from '../../components/Chart/MassiveOptionsChart';

// Professional styled components
const DashboardContainer = styled(Box)(({ theme }) => ({
    minHeight: '100vh',
    backgroundColor: theme.palette.background.default,
    position: 'relative',
}));

const MarketTicker = styled(Paper)(({ theme }) => ({
    backgroundColor: theme.palette.background.paper,
    borderBottom: `1px solid ${theme.palette.divider}`,
    padding: '8px 16px',
    borderRadius: 0,
    position: 'sticky',
    top: 0,
    zIndex: 1000,
}));

const TickerItem = styled(Box)(({ theme }) => ({
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '4px 12px',
    borderRadius: 4,
    transition: 'all 0.2s ease-in-out',
    cursor: 'pointer',
    '&:hover': {
        backgroundColor: theme.palette.action.hover,
    },
}));

const DashboardCard = styled(Card)(({ theme }) => ({
    backgroundColor: theme.palette.background.paper,
    border: `1px solid ${theme.palette.divider}`,
    borderRadius: 8,
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    transition: 'all 0.2s ease-in-out',
    '&:hover': {
        borderColor: theme.palette.primary.main,
        boxShadow: '0 4px 12px rgba(255, 215, 0, 0.1)',
    },
}));

const SignalCard = styled(Card)(({ theme }) => ({
    backgroundColor: theme.palette.background.paper,
    border: `1px solid ${theme.palette.divider}`,
    borderRadius: 6,
    padding: '12px',
    marginBottom: '8px',
    transition: 'all 0.2s ease-in-out',
    cursor: 'pointer',
    '&:hover': {
        borderColor: theme.palette.primary.main,
        transform: 'translateY(-1px)',
    },
}));

const GoldenEyeFab = styled(Fab)(({ theme }) => ({
    position: 'fixed',
    bottom: 24,
    right: 24,
    backgroundColor: theme.palette.primary.main,
    color: theme.palette.primary.contrastText,
    zIndex: 1300,
    '&:hover': {
        backgroundColor: theme.palette.primary.dark,
        transform: 'scale(1.1)',
    },
}));

const AIDrawer = styled(Drawer)(({ theme }) => ({
    '& .MuiDrawer-paper': {
        width: 400,
        backgroundColor: theme.palette.background.paper,
        borderLeft: `1px solid ${theme.palette.divider}`,
    },
}));

// Mock data - in real app, this would come from API
const MARKET_DATA = [
    { symbol: 'SPY', price: 445.67, change: 2.34, changePercent: 0.53 },
    { symbol: 'AAPL', price: 189.45, change: -1.23, changePercent: -0.65 },
    { symbol: 'MSFT', price: 378.92, change: 4.56, changePercent: 1.22 },
    { symbol: 'GOOGL', price: 142.78, change: 0.89, changePercent: 0.63 },
    { symbol: 'AMZN', price: 156.23, change: -2.45, changePercent: -1.54 },
    { symbol: 'TSLA', price: 267.89, change: 8.45, changePercent: 3.26 },
];

const LATEST_SIGNALS = [
    {
        id: '1',
        symbol: 'SPY',
        action: 'BUY',
        confidence: 94,
        price: 445.67,
        target: 452.00,
        stopLoss: 440.00,
        timestamp: new Date(),
        strategy: 'AI Prophet Momentum',
    },
    {
        id: '2',
        symbol: 'AAPL',
        action: 'HOLD',
        confidence: 78,
        price: 189.45,
        target: 195.00,
        stopLoss: 185.00,
        timestamp: new Date(),
        strategy: 'Technical Analysis',
    },
    {
        id: '3',
        symbol: 'MSFT',
        action: 'BUY',
        confidence: 87,
        price: 378.92,
        target: 390.00,
        stopLoss: 370.00,
        timestamp: new Date(),
        strategy: 'Earnings Momentum',
    },
];

const PERFORMANCE_METRICS = {
    totalReturn: 12.45,
    todayReturn: 2.34,
    winRate: 73.2,
    sharpeRatio: 1.85,
    maxDrawdown: -8.9,
    totalTrades: 156,
};

interface ProfessionalDashboardProps {
    // Props can be added here for customization
}

export const ProfessionalDashboard: React.FC<ProfessionalDashboardProps> = () => {
    const theme = useTheme();
    const [aiDrawerOpen, setAiDrawerOpen] = useState(false);
    const [selectedSymbol, setSelectedSymbol] = useState('SPY');
    const [refreshing, setRefreshing] = useState(false);
    const [notifications, setNotifications] = useState(3);

    // Memoized market ticker to prevent re-renders
    const marketTicker = useMemo(() => (
        <MarketTicker elevation={0}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, overflowX: 'auto' }}>
                <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.secondary', minWidth: 'fit-content' }}>
                    MARKET
                </Typography>
                <Divider orientation="vertical" flexItem />
                {MARKET_DATA.map((item) => (
                    <TickerItem key={item.symbol} onClick={() => setSelectedSymbol(item.symbol)}>
                        <Typography variant="body2" sx={{ fontWeight: 600, minWidth: 'fit-content' }}>
                            {item.symbol}
                        </Typography>
                        <Typography variant="body2" sx={{ minWidth: 'fit-content' }}>
                            {formatCurrency(item.price)}
                        </Typography>
                        <Typography
                            variant="body2"
                            sx={{
                                color: getMarketColor(item.change),
                                fontWeight: 500,
                                minWidth: 'fit-content',
                            }}
                        >
                            {item.change > 0 ? '+' : ''}{item.change.toFixed(2)} ({formatPercentage(item.changePercent)})
                        </Typography>
                    </TickerItem>
                ))}
            </Box>
        </MarketTicker>
    ), []);

    const handleRefresh = useCallback(async () => {
        setRefreshing(true);
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));
        setRefreshing(false);
    }, []);

    const handleSearchSelect = useCallback((suggestion: any) => {
        if (suggestion.category === 'symbols') {
            setSelectedSymbol(suggestion.label);
        } else if (suggestion.category === 'ai') {
            setAiDrawerOpen(true);
        }
    }, []);

    return (
        <DashboardContainer>
            {/* Market Ticker */}
            {marketTicker}

            {/* Main Content */}
            <Box sx={{ p: 3 }}>
                {/* Header with Search */}
                <Box sx={{ mb: 3 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                        <Typography variant="h4" sx={{ fontWeight: 700, color: 'text.primary' }}>
                            Professional Dashboard
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Tooltip title="Refresh Data">
                                <IconButton onClick={handleRefresh} disabled={refreshing}>
                                    <RefreshIcon />
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="Notifications">
                                <IconButton>
                                    <Badge badgeContent={notifications} color="error">
                                        <NotificationsIcon />
                                    </Badge>
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="Settings">
                                <IconButton>
                                    <SettingsIcon />
                                </IconButton>
                            </Tooltip>
                        </Box>
                    </Box>

                    <ProfessionalSearchBar
                        onSelect={handleSearchSelect}
                        placeholder="Search symbols, signals, or ask Golden Eye AI..."
                    />
                </Box>

                {/* Main Grid */}
                <Grid container spacing={3}>
                    {/* Live Signals Feed */}
                    <Grid item xs={12} md={4}>
                        <DashboardCard>
                            <CardContent sx={{ flex: 1 }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                                        Live Signals
                                    </Typography>
                                    <Chip
                                        label={`${LATEST_SIGNALS.length} Active`}
                                        color="success"
                                        size="small"
                                        variant="outlined"
                                    />
                                </Box>

                                <Box sx={{ maxHeight: 400, overflowY: 'auto' }}>
                                    {LATEST_SIGNALS.map((signal) => (
                                        <motion.div
                                            key={signal.id}
                                            initial={{ opacity: 0, y: 20 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ duration: 0.3 }}
                                        >
                                            <SignalCard>
                                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                                                    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                                                        {signal.symbol}
                                                    </Typography>
                                                    <Chip
                                                        label={signal.action}
                                                        color={signal.action === 'BUY' ? 'success' : signal.action === 'SELL' ? 'error' : 'warning'}
                                                        size="small"
                                                    />
                                                </Box>

                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
                                                    <Typography variant="body2" color="text.secondary">
                                                        Confidence:
                                                    </Typography>
                                                    <Box sx={{ flex: 1 }}>
                                                        <LinearProgress
                                                            variant="determinate"
                                                            value={signal.confidence}
                                                            sx={{
                                                                height: 6,
                                                                borderRadius: 3,
                                                                backgroundColor: 'action.hover',
                                                                '& .MuiLinearProgress-bar': {
                                                                    backgroundColor: getConfidenceColor(signal.confidence),
                                                                },
                                                            }}
                                                        />
                                                    </Box>
                                                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                                        {signal.confidence}%
                                                    </Typography>
                                                </Box>

                                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                    <Box>
                                                        <Typography variant="caption" color="text.secondary">
                                                            Target: {formatCurrency(signal.target)}
                                                        </Typography>
                                                    </Box>
                                                    <Box>
                                                        <Typography variant="caption" color="text.secondary">
                                                            Stop: {formatCurrency(signal.stopLoss)}
                                                        </Typography>
                                                    </Box>
                                                </Box>
                                            </SignalCard>
                                        </motion.div>
                                    ))}
                                </Box>
                            </CardContent>
                        </DashboardCard>
                    </Grid>

                    {/* Massive Options Chart */}
                    <Grid item xs={12}>
                        <MassiveOptionsChart
                            symbol="SPY"
                            height={800}
                            showOptionsFlow={true}
                            showGammaExposure={true}
                            showDarkPool={true}
                        />
                    </Grid>

                    {/* Market Overview */}
                    <Grid item xs={12} md={6}>
                        <DashboardCard>
                            <CardContent>
                                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                    <TrendingUpIcon sx={{ mr: 1, color: 'success.main' }} />
                                    <Typography variant="h6" fontWeight="bold">Market Overview</Typography>
                                </Box>
                                <Stack spacing={2}>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <Typography>S&P 500</Typography>
                                        <Typography color="success.main" fontWeight="bold">+1.24%</Typography>
                                    </Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <Typography>NASDAQ</Typography>
                                        <Typography color="success.main" fontWeight="bold">+0.87%</Typography>
                                    </Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <Typography>VIX</Typography>
                                        <Typography color="error.main" fontWeight="bold">-2.15%</Typography>
                                    </Box>
                                </Stack>
                            </CardContent>
                        </DashboardCard>
                    </Grid>

                    {/* Market Intel */}
                    <Grid item xs={12} md={3}>
                        <DashboardCard>
                            <CardContent sx={{ flex: 1 }}>
                                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                                    Market Intel
                                </Typography>

                                {/* Performance Metrics */}
                                <Box sx={{ mb: 3 }}>
                                    <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                                        Performance
                                    </Typography>
                                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <Typography variant="body2">Total Return</Typography>
                                            <Typography variant="body2" sx={{ color: getMarketColor(PERFORMANCE_METRICS.totalReturn), fontWeight: 500 }}>
                                                {formatPercentage(PERFORMANCE_METRICS.totalReturn)}
                                            </Typography>
                                        </Box>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <Typography variant="body2">Today</Typography>
                                            <Typography variant="body2" sx={{ color: getMarketColor(PERFORMANCE_METRICS.todayReturn), fontWeight: 500 }}>
                                                {formatPercentage(PERFORMANCE_METRICS.todayReturn)}
                                            </Typography>
                                        </Box>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <Typography variant="body2">Win Rate</Typography>
                                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                                {PERFORMANCE_METRICS.winRate}%
                                            </Typography>
                                        </Box>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <Typography variant="body2">Sharpe Ratio</Typography>
                                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                                {PERFORMANCE_METRICS.sharpeRatio}
                                            </Typography>
                                        </Box>
                                    </Box>
                                </Box>

                                <Divider sx={{ my: 2 }} />

                                {/* Quick Stats */}
                                <Box>
                                    <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                                        Quick Stats
                                    </Typography>
                                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <Typography variant="body2">Active Signals</Typography>
                                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                                {LATEST_SIGNALS.length}
                                            </Typography>
                                        </Box>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <Typography variant="body2">Total Trades</Typography>
                                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                                {PERFORMANCE_METRICS.totalTrades}
                                            </Typography>
                                        </Box>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <Typography variant="body2">Max Drawdown</Typography>
                                            <Typography variant="body2" sx={{ color: getMarketColor(PERFORMANCE_METRICS.maxDrawdown), fontWeight: 500 }}>
                                                {formatPercentage(PERFORMANCE_METRICS.maxDrawdown)}
                                            </Typography>
                                        </Box>
                                    </Box>
                                </Box>
                            </CardContent>
                        </DashboardCard>
                    </Grid>
                </Grid>

                {/* AI Insights Bar */}
                <Box sx={{ mt: 3 }}>
                    <DashboardCard>
                        <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                <PsychologyIcon sx={{ color: 'primary.main' }} />
                                <Box sx={{ flex: 1 }}>
                                    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                                        Golden Eye AI Prophet Insights
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        Market sentiment is bullish with strong momentum in tech sector. SPY showing breakout pattern with 94% confidence.
                                    </Typography>
                                </Box>
                                <Chip
                                    label="Ask AI"
                                    color="primary"
                                    onClick={() => setAiDrawerOpen(true)}
                                    sx={{ cursor: 'pointer' }}
                                />
                            </Box>
                        </CardContent>
                    </DashboardCard>
                </Box>
            </Box>

            {/* Golden Eye AI Floating Action Button */}
            <FloatingAIProphetWidget
                onClick={() => setAiDrawerOpen(true)}
                isVisible={true}
            />

            {/* AI Drawer */}
            <AIDrawer
                anchor="right"
                open={aiDrawerOpen}
                onClose={() => setAiDrawerOpen(false)}
            >
                <Box sx={{ p: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                            Golden Eye AI Prophet
                        </Typography>
                        <IconButton onClick={() => setAiDrawerOpen(false)}>
                            <CloseIcon />
                        </IconButton>
                    </Box>

                    <EnhancedAIProphetChat />
                </Box>
            </AIDrawer>

            {/* Loading indicator */}
            {refreshing && (
                <LinearProgress
                    sx={{
                        position: 'fixed',
                        top: 0,
                        left: 0,
                        right: 0,
                        zIndex: 1400,
                    }}
                />
            )}
        </DashboardContainer>
    );
};

export default ProfessionalDashboard; 