import React, { useState, useEffect } from 'react';
import {
    Box,
    Card,
    CardContent,
    Typography,
    Chip,
    Stack,
    IconButton,
    Tooltip,
    useTheme,
    alpha,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    ShowChart,
    Timeline,
    Refresh,
    Fullscreen,
    Settings,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

const ChartContainer = styled(Box)(({ theme }) => ({
    width: '100%',
    height: 400,
    position: 'relative',
    background: `linear-gradient(135deg, ${alpha(theme.palette.background.paper, 0.8)} 0%, ${alpha(theme.palette.background.default, 0.9)} 100%)`,
    borderRadius: theme.spacing(2),
    border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
    overflow: 'hidden',
}));

const SignalIndicator = styled(Box)<{ signal: 'buy' | 'sell' | 'neutral' }>(({ theme, signal }) => ({
    position: 'absolute',
    top: theme.spacing(2),
    right: theme.spacing(2),
    padding: theme.spacing(0.5, 1),
    borderRadius: theme.spacing(1),
    background: signal === 'buy'
        ? alpha(theme.palette.success.main, 0.2)
        : signal === 'sell'
            ? alpha(theme.palette.error.main, 0.2)
            : alpha(theme.palette.warning.main, 0.2),
    border: `1px solid ${signal === 'buy'
            ? theme.palette.success.main
            : signal === 'sell'
                ? theme.palette.error.main
                : theme.palette.warning.main
        }`,
    color: signal === 'buy'
        ? theme.palette.success.main
        : signal === 'sell'
            ? theme.palette.error.main
            : theme.palette.warning.main,
    fontWeight: 600,
    fontSize: '0.75rem',
    textTransform: 'uppercase',
}));

interface AdvancedSignalChartProps {
    symbol?: string;
    height?: number;
    showControls?: boolean;
    showSignals?: boolean;
    timeframe?: '1d' | '1w' | '1m' | '3m' | '1y';
    onSymbolChange?: (symbol: string) => void;
}

export const AdvancedSignalChart: React.FC<AdvancedSignalChartProps> = ({
    symbol = 'SPY',
    height = 400,
    showControls = true,
    showSignals = true,
    timeframe = '1d',
    onSymbolChange,
}) => {
    const theme = useTheme();
    const [currentSignal, setCurrentSignal] = useState<'buy' | 'sell' | 'neutral'>('buy');
    const [isLoading, setIsLoading] = useState(false);
    const [price, setPrice] = useState(458.32);
    const [change, setChange] = useState(5.68);
    const [changePercent, setChangePercent] = useState(1.24);

    // Simulate real-time data updates
    useEffect(() => {
        const interval = setInterval(() => {
            const randomChange = (Math.random() - 0.5) * 2;
            const newPrice = price + randomChange;
            const newChange = newPrice - price;
            const newChangePercent = (newChange / price) * 100;

            setPrice(newPrice);
            setChange(newChange);
            setChangePercent(newChangePercent);

            // Update signal based on trend
            if (newChangePercent > 0.5) {
                setCurrentSignal('buy');
            } else if (newChangePercent < -0.5) {
                setCurrentSignal('sell');
            } else {
                setCurrentSignal('neutral');
            }
        }, 5000);

        return () => clearInterval(interval);
    }, [price]);

    const handleRefresh = () => {
        setIsLoading(true);
        setTimeout(() => setIsLoading(false), 1000);
    };

    return (
        <Card
            elevation={2}
            sx={{
                background: 'linear-gradient(135deg, rgba(255, 215, 0, 0.05) 0%, rgba(255, 165, 0, 0.02) 100%)',
                border: '1px solid rgba(255, 215, 0, 0.2)',
                borderRadius: 3,
                overflow: 'visible',
            }}
        >
            <CardContent sx={{ p: 3 }}>
                {/* Header */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Box>
                        <Typography variant="h6" fontWeight="bold" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <ShowChart sx={{ color: 'primary.main' }} />
                            {symbol} Advanced Chart
                        </Typography>
                        <Stack direction="row" spacing={2} sx={{ mt: 1 }}>
                            <Typography variant="h4" fontWeight="bold">
                                ${price.toFixed(2)}
                            </Typography>
                            <Chip
                                icon={change >= 0 ? <TrendingUp /> : <TrendingDown />}
                                label={`${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePercent.toFixed(2)}%)`}
                                color={change >= 0 ? 'success' : 'error'}
                                size="small"
                            />
                        </Stack>
                    </Box>

                    {showControls && (
                        <Stack direction="row" spacing={1}>
                            <Tooltip title="Refresh Data">
                                <IconButton size="small" onClick={handleRefresh} disabled={isLoading}>
                                    <Refresh sx={{ fontSize: 20 }} />
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="Chart Settings">
                                <IconButton size="small">
                                    <Settings sx={{ fontSize: 20 }} />
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="Fullscreen">
                                <IconButton size="small">
                                    <Fullscreen sx={{ fontSize: 20 }} />
                                </IconButton>
                            </Tooltip>
                        </Stack>
                    )}
                </Box>

                {/* Chart Container */}
                <ChartContainer sx={{ height }}>
                    {/* Simulated Chart Area */}
                    <Box
                        sx={{
                            width: '100%',
                            height: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            background: `linear-gradient(45deg, ${alpha(theme.palette.primary.main, 0.1)} 0%, ${alpha(theme.palette.secondary.main, 0.1)} 100%)`,
                            position: 'relative',
                        }}
                    >
                        {/* Placeholder Chart Visualization */}
                        <Box
                            sx={{
                                width: '90%',
                                height: '80%',
                                background: `linear-gradient(to right, ${alpha(theme.palette.success.main, 0.3)}, ${alpha(theme.palette.primary.main, 0.3)}, ${alpha(theme.palette.warning.main, 0.3)})`,
                                borderRadius: 1,
                                position: 'relative',
                                overflow: 'hidden',
                                '&::before': {
                                    content: '""',
                                    position: 'absolute',
                                    top: '20%',
                                    left: 0,
                                    right: 0,
                                    height: '60%',
                                    background: `repeating-linear-gradient(
                                        90deg,
                                        transparent,
                                        transparent 20px,
                                        ${alpha(theme.palette.divider, 0.3)} 20px,
                                        ${alpha(theme.palette.divider, 0.3)} 21px
                                    )`,
                                },
                                '&::after': {
                                    content: '""',
                                    position: 'absolute',
                                    top: 0,
                                    bottom: 0,
                                    left: '10%',
                                    right: '10%',
                                    background: `repeating-linear-gradient(
                                        0deg,
                                        transparent,
                                        transparent 30px,
                                        ${alpha(theme.palette.divider, 0.2)} 30px,
                                        ${alpha(theme.palette.divider, 0.2)} 31px
                                    )`,
                                },
                            }}
                        />

                        {/* Chart Loading Overlay */}
                        {isLoading && (
                            <Box
                                sx={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    right: 0,
                                    bottom: 0,
                                    background: alpha(theme.palette.background.paper, 0.8),
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    borderRadius: 1,
                                }}
                            >
                                <Typography variant="body2" color="text.secondary">
                                    Loading chart data...
                                </Typography>
                            </Box>
                        )}

                        {/* Chart Label */}
                        <Typography
                            variant="body2"
                            color="text.secondary"
                            sx={{
                                position: 'absolute',
                                bottom: 16,
                                left: 16,
                                fontFamily: 'monospace',
                            }}
                        >
                            {timeframe.toUpperCase()} • Real-time signals enabled
                        </Typography>
                    </Box>

                    {/* Signal Indicator */}
                    {showSignals && (
                        <SignalIndicator signal={currentSignal}>
                            {currentSignal === 'buy' ? 'Strong Buy' : currentSignal === 'sell' ? 'Strong Sell' : 'Hold'}
                        </SignalIndicator>
                    )}
                </ChartContainer>

                {/* Chart Controls */}
                <Stack direction="row" spacing={1} sx={{ mt: 2, justifyContent: 'center' }}>
                    {['1d', '1w', '1m', '3m', '1y'].map((tf) => (
                        <Chip
                            key={tf}
                            label={tf.toUpperCase()}
                            size="small"
                            variant={timeframe === tf ? 'filled' : 'outlined'}
                            color={timeframe === tf ? 'primary' : 'default'}
                            clickable
                            sx={{ minWidth: 40 }}
                        />
                    ))}
                </Stack>

                {/* Technical Indicators */}
                <Stack direction="row" spacing={2} sx={{ mt: 2, justifyContent: 'space-around' }}>
                    <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="caption" color="text.secondary">RSI</Typography>
                        <Typography variant="body2" fontWeight="bold" color="success.main">67.8</Typography>
                    </Box>
                    <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="caption" color="text.secondary">MACD</Typography>
                        <Typography variant="body2" fontWeight="bold" color="primary.main">+2.45</Typography>
                    </Box>
                    <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="caption" color="text.secondary">Volume</Typography>
                        <Typography variant="body2" fontWeight="bold">125.4M</Typography>
                    </Box>
                    <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="caption" color="text.secondary">Support</Typography>
                        <Typography variant="body2" fontWeight="bold" color="info.main">${(price - 5).toFixed(2)}</Typography>
                    </Box>
                </Stack>
            </CardContent>
        </Card>
    );
}; 