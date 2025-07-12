import React, { useState, useEffect, useCallback } from 'react';
import {
    Box,
    Grid,
    Typography,
    TextField,
    Autocomplete,
    IconButton,
    Fab,
    useTheme,
    alpha,
    Container,
    AppBar,
    Toolbar,
    Switch,
    FormControlLabel,
    Chip,
    Badge,
} from '@mui/material';
import {
    Search as SearchIcon,
    DarkMode,
    LightMode,
    Notifications,
    Settings,
    TrendingUp,
    Psychology,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';

// Import our new trading components
import RealTimeChart from '../../components/TradingSignals/RealTimeChart';
import SignalAlerts from '../../components/TradingSignals/SignalAlerts';
import TradeGuidancePanel from '../../components/TradingSignals/TradeGuidancePanel';

// Professional styled components
const TradingContainer = styled(Box)(({ theme }) => ({
    minHeight: '100vh',
    backgroundColor: theme.palette.background.default,
    background: `linear-gradient(135deg, ${theme.palette.background.default} 0%, ${alpha(theme.palette.primary.main, 0.02)} 100%)`,
}));

const MainContent = styled(Container)(({ theme }) => ({
    paddingTop: theme.spacing(2),
    paddingBottom: theme.spacing(2),
    height: 'calc(100vh - 80px)',
    maxWidth: '100% !important',
}));

const StockSearchBar = styled(Autocomplete)(({ theme }) => ({
    '& .MuiOutlinedInput-root': {
        backgroundColor: alpha(theme.palette.background.paper, 0.8),
        backdropFilter: 'blur(10px)',
        borderRadius: '12px',
        border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
        '&:hover': {
            border: `1px solid ${alpha(theme.palette.primary.main, 0.4)}`,
        },
        '&.Mui-focused': {
            border: `1px solid ${theme.palette.primary.main}`,
            boxShadow: `0 0 0 2px ${alpha(theme.palette.primary.main, 0.2)}`,
        },
    },
}));

const StatusIndicator = styled(Box)(({ theme, status }) => ({
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(1),
    padding: theme.spacing(0.5, 1),
    borderRadius: '20px',
    backgroundColor: alpha(
        status === 'live' ? theme.palette.success.main :
            status === 'analyzing' ? theme.palette.warning.main :
                theme.palette.error.main,
        0.1
    ),
    border: `1px solid ${alpha(
        status === 'live' ? theme.palette.success.main :
            status === 'analyzing' ? theme.palette.warning.main :
                theme.palette.error.main,
        0.3
    )}`,
}));

const PulsingDot = styled(motion.div)(({ theme, color }) => ({
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    backgroundColor: color,
}));

// Mock data for demonstration
const POPULAR_STOCKS = [
    { label: 'Apple Inc.', symbol: 'AAPL', price: 157.85 },
    { label: 'Microsoft Corporation', symbol: 'MSFT', price: 342.45 },
    { label: 'NVIDIA Corporation', symbol: 'NVDA', price: 445.67 },
    { label: 'Tesla, Inc.', symbol: 'TSLA', price: 245.32 },
    { label: 'Amazon.com, Inc.', symbol: 'AMZN', price: 134.56 },
    { label: 'Alphabet Inc.', symbol: 'GOOGL', price: 142.78 },
    { label: 'Meta Platforms, Inc.', symbol: 'META', price: 298.45 },
    { label: 'Berkshire Hathaway Inc.', symbol: 'BRK.B', price: 367.89 },
];

interface Signal {
    id: string;
    symbol: string;
    type: 'buy' | 'sell';
    action: string;
    price: number;
    confidence: number;
    timestamp: number;
    reason: string;
    timeframe: string;
    status: 'active' | 'executed' | 'expired';
    x: number;
    y: number;
    targetPrice?: number;
    stopLoss?: number;
    riskReward?: number;
    agentAnalysis?: {
        technical: string;
        sentiment: string;
        aiPredictor: string;
    };
    historicalPerformance?: {
        successRate: number;
        todayTrades: number;
        weeklyPerformance: number;
    };
}

const TradingSignalsApp: React.FC = () => {
    const theme = useTheme();
    const [selectedStock, setSelectedStock] = useState(POPULAR_STOCKS[0]);
    const [darkMode, setDarkMode] = useState(true);
    const [signals, setSignals] = useState<Signal[]>([]);
    const [selectedSignal, setSelectedSignal] = useState<Signal | null>(null);
    const [marketStatus, setMarketStatus] = useState<'live' | 'analyzing' | 'closed'>('analyzing');
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    // Simulate real-time signal generation
    useEffect(() => {
        const generateSignal = () => {
            const signal: Signal = {
                id: `signal-${Date.now()}`,
                symbol: selectedStock.symbol,
                type: Math.random() > 0.5 ? 'buy' : 'sell',
                action: Math.random() > 0.5 ? 'Consider CALL option' : 'Buy shares now',
                price: selectedStock.price + (Math.random() - 0.5) * 10,
                confidence: Math.floor(Math.random() * 30) + 70, // 70-100%
                timestamp: Date.now(),
                reason: 'Our analysis detects a bullish breakout above resistance, coupled with rising volume and a bullish MACD crossover. This suggests strong upward momentum with multiple confirmation signals.',
                timeframe: ['1m', '5m', '15m', '1h'][Math.floor(Math.random() * 4)],
                status: 'active',
                x: Math.random() * 80 + 10, // 10-90% from left
                y: Math.random() * 60 + 20, // 20-80% from top
                targetPrice: selectedStock.price * (1 + Math.random() * 0.1),
                stopLoss: selectedStock.price * (1 - Math.random() * 0.05),
                riskReward: 2 + Math.random() * 2,
                agentAnalysis: {
                    technical: 'Bullish breakout above $155 resistance with high volume confirmation',
                    sentiment: 'Positive news sentiment with 85% bullish social media mentions',
                    aiPredictor: '95% probability of 5% upside movement in next 3-5 trading days',
                },
                historicalPerformance: {
                    successRate: Math.floor(Math.random() * 20) + 80,
                    todayTrades: Math.floor(Math.random() * 10) + 5,
                    weeklyPerformance: Math.random() * 15 + 5,
                },
            };

            setSignals(prev => [signal, ...prev.slice(0, 19)]); // Keep last 20 signals

            // Show toast notification
            toast.success(
                `${signal.type === 'buy' ? '🚀' : '📉'} ${signal.symbol} ${signal.type.toUpperCase()} Signal - ${signal.confidence}% Confidence`,
                {
                    duration: 5000,
                    position: 'top-right',
                }
            );
        };

        // Generate initial signal after stock selection
        const initialTimer = setTimeout(() => {
            setIsAnalyzing(true);
            setMarketStatus('analyzing');

            setTimeout(() => {
                generateSignal();
                setMarketStatus('live');
                setIsAnalyzing(false);
            }, 2000);
        }, 1000);

        // Generate signals periodically
        const interval = setInterval(() => {
            if (Math.random() > 0.7) { // 30% chance every interval
                generateSignal();
            }
        }, 15000); // Every 15 seconds

        return () => {
            clearTimeout(initialTimer);
            clearInterval(interval);
        };
    }, [selectedStock]);

    const handleStockChange = useCallback((event: any, newValue: any) => {
        if (newValue) {
            setSelectedStock(newValue);
            setSignals([]); // Clear signals when switching stocks
            setSelectedSignal(null);
            setIsAnalyzing(true);
            setMarketStatus('analyzing');

            toast(`Analyzing ${newValue.symbol}...`, {
                icon: '🔍',
                duration: 2000,
            });
        }
    }, []);

    const handleSignalClick = useCallback((signal: Signal) => {
        setSelectedSignal(signal);
    }, []);

    const handleDismissAlert = useCallback((signalId: string) => {
        // Mark signal as dismissed or remove from active alerts
        console.log('Dismissed signal:', signalId);
    }, []);

    const handleCopyTradeInfo = useCallback((signal: Signal) => {
        const tradeInfo = `
${signal.symbol} ${signal.type.toUpperCase()} Signal
Entry: $${signal.price.toFixed(2)}
Target: $${signal.targetPrice?.toFixed(2) || 'N/A'}
Stop Loss: $${signal.stopLoss?.toFixed(2) || 'N/A'}
Confidence: ${signal.confidence}%
Timeframe: ${signal.timeframe}

${signal.reason}
        `.trim();

        navigator.clipboard.writeText(tradeInfo);
        toast.success('Trade info copied to clipboard!');
    }, []);

    const handleSimulateTrade = useCallback((signal: Signal) => {
        toast.success('Trade simulation started! 📊');
        // Implement paper trading simulation
    }, []);

    return (
        <TradingContainer>
            {/* Top Navigation */}
            <AppBar
                position="static"
                elevation={0}
                sx={{
                    backgroundColor: alpha(theme.palette.background.paper, 0.8),
                    backdropFilter: 'blur(10px)',
                    borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                }}
            >
                <Toolbar>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flex: 1 }}>
                        <Psychology sx={{ color: theme.palette.primary.main, fontSize: 32 }} />
                        <Typography variant="h6" fontWeight="bold" color="text.primary">
                            GoldenSignals AI
                        </Typography>

                        {/* Market Status */}
                        <StatusIndicator status={marketStatus}>
                            <PulsingDot
                                color={
                                    marketStatus === 'live' ? theme.palette.success.main :
                                        marketStatus === 'analyzing' ? theme.palette.warning.main :
                                            theme.palette.error.main
                                }
                                animate={{
                                    scale: marketStatus === 'live' ? [1, 1.2, 1] : 1,
                                    opacity: marketStatus === 'analyzing' ? [1, 0.5, 1] : 1,
                                }}
                                transition={{
                                    duration: 1.5,
                                    repeat: marketStatus !== 'closed' ? Infinity : 0
                                }}
                            />
                            <Typography variant="caption" fontWeight="medium">
                                {marketStatus === 'live' ? 'Live Analysis' :
                                    marketStatus === 'analyzing' ? 'Analyzing...' :
                                        'Market Closed'}
                            </Typography>
                        </StatusIndicator>
                    </Box>

                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Badge badgeContent={signals.filter(s => s.status === 'active').length} color="error">
                            <IconButton color="inherit">
                                <Notifications />
                            </IconButton>
                        </Badge>

                        <FormControlLabel
                            control={
                                <Switch
                                    checked={darkMode}
                                    onChange={(e) => setDarkMode(e.target.checked)}
                                    size="small"
                                />
                            }
                            label={darkMode ? <DarkMode /> : <LightMode />}
                            sx={{ ml: 1 }}
                        />

                        <IconButton color="inherit">
                            <Settings />
                        </IconButton>
                    </Box>
                </Toolbar>
            </AppBar>

            <MainContent>
                {/* Stock Selection */}
                <Box sx={{ mb: 3 }}>
                    <StockSearchBar
                        options={POPULAR_STOCKS}
                        getOptionLabel={(option) => `${option.symbol} - ${option.label}`}
                        value={selectedStock}
                        onChange={handleStockChange}
                        renderInput={(params) => (
                            <TextField
                                {...params}
                                placeholder="Search for a stock symbol (e.g., AAPL, TSLA, NVDA)"
                                InputProps={{
                                    ...params.InputProps,
                                    startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
                                }}
                            />
                        )}
                        renderOption={(props, option) => (
                            <Box component="li" {...props} sx={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                                <Box>
                                    <Typography variant="body2" fontWeight="bold">
                                        {option.symbol}
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary">
                                        {option.label}
                                    </Typography>
                                </Box>
                                <Typography variant="body2" color="primary" fontWeight="bold">
                                    ${option.price.toFixed(2)}
                                </Typography>
                            </Box>
                        )}
                        sx={{ maxWidth: 600, mx: 'auto' }}
                    />
                </Box>

                {/* Main Dashboard Grid */}
                <Grid container spacing={3} sx={{ height: 'calc(100% - 100px)' }}>
                    {/* Chart Section */}
                    <Grid item xs={12} lg={8}>
                        <Box sx={{ height: '100%', position: 'relative' }}>
                            <RealTimeChart
                                symbol={selectedStock.symbol}
                                data={[]}
                                signals={signals}
                                onSignalClick={handleSignalClick}
                                height={600}
                            />

                            {/* Analysis Loading Overlay */}
                            <AnimatePresence>
                                {isAnalyzing && (
                                    <motion.div
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                        style={{
                                            position: 'absolute',
                                            top: 0,
                                            left: 0,
                                            right: 0,
                                            bottom: 0,
                                            backgroundColor: alpha(theme.palette.background.default, 0.8),
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            zIndex: 10,
                                            borderRadius: '12px',
                                        }}
                                    >
                                        <Box sx={{ textAlign: 'center' }}>
                                            <motion.div
                                                animate={{ rotate: 360 }}
                                                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                                            >
                                                <Psychology sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                                            </motion.div>
                                            <Typography variant="h6" gutterBottom>
                                                Analyzing {selectedStock.symbol}
                                            </Typography>
                                            <Typography variant="body2" color="text.secondary">
                                                AI agents are processing market data and technical indicators...
                                            </Typography>
                                        </Box>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </Box>
                    </Grid>

                    {/* Sidebar Section */}
                    <Grid item xs={12} lg={4}>
                        <Grid container spacing={2} sx={{ height: '100%' }}>
                            {/* Signals Panel */}
                            <Grid item xs={12} md={6} lg={12}>
                                <Box sx={{ height: selectedSignal ? '40%' : '100%' }}>
                                    <SignalAlerts
                                        signals={signals}
                                        onSignalClick={handleSignalClick}
                                        onDismissAlert={handleDismissAlert}
                                    />
                                </Box>
                            </Grid>

                            {/* Trade Guidance Panel */}
                            {selectedSignal && (
                                <Grid item xs={12} md={6} lg={12}>
                                    <Box sx={{ height: '60%' }}>
                                        <TradeGuidancePanel
                                            signal={selectedSignal}
                                            onCopyTradeInfo={handleCopyTradeInfo}
                                            onSimulateTrade={handleSimulateTrade}
                                            onDismiss={() => setSelectedSignal(null)}
                                        />
                                    </Box>
                                </Grid>
                            )}
                        </Grid>
                    </Grid>
                </Grid>
            </MainContent>
        </TradingContainer>
    );
};

export default TradingSignalsApp; 