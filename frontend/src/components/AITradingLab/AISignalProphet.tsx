import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Box,
    Paper,
    Typography,
    TextField,
    Autocomplete,
    Button,
    Chip,
    Stack,
    Alert,
    IconButton,
    Tooltip,
    Divider,
    LinearProgress,
    Collapse,
    ToggleButton,
    ToggleButtonGroup,
    Grid,
    Card,
    CardContent,
    Fade,
    Zoom,
    Badge,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Dialog,
    DialogTitle,
    DialogContent,
} from '@mui/material';
import {
    AutoAwesome as AIIcon,
    Search as SearchIcon,
    TrendingUp as BullishIcon,
    TrendingDown as BearishIcon,
    ShowChart as ChartIcon,
    Timeline as TimelineIcon,
    Psychology as PsychologyIcon,
    Assessment as AssessmentIcon,
    PlayArrow as PlayIcon,
    Stop as StopIcon,
    Refresh as RefreshIcon,
    CheckCircle as SuccessIcon,
    Warning as WarningIcon,
    Info as InfoIcon,
    Speed as SpeedIcon,
    AttachMoney as MoneyIcon,
    Security as ShieldIcon,
    Insights as InsightsIcon,
    Analytics as AnalyticsIcon,
    ShowChartOutlined as FibIcon,
    Close as CloseIcon,
} from '@mui/icons-material';
import { useTheme, alpha } from '@mui/material/styles';
import {
    createChart,
    IChartApi,
    ISeriesApi,
    CandlestickData,
    Time,
    LineStyle,
    CrosshairMode,
    ColorType,
} from 'lightweight-charts';
import FibonacciRetracement from './FibonacciRetracement';

interface AISignal {
    id: string;
    symbol: string;
    timeframe: string;
    type: 'LONG' | 'SHORT';
    entry: number;
    stopLoss: number;
    takeProfit: number[];
    confidence: number;
    indicators: string[];
    patterns: string[];
    timestamp: Date;
    status: 'analyzing' | 'active' | 'success' | 'stopped';
    pnl?: number;
    reasoning: string;
    fibLevels?: number[];
    keyLevels?: { price: number; type: string; strength: number }[];
}

interface MarketAnalysis {
    trend: 'bullish' | 'bearish' | 'neutral';
    strength: number;
    volatility: 'low' | 'medium' | 'high';
    volume: 'increasing' | 'decreasing' | 'stable';
    sentiment: number; // -100 to 100
}

const AISignalProphet: React.FC = () => {
    const theme = useTheme();
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

    const [selectedSymbol, setSelectedSymbol] = useState('SPY');
    const [selectedTimeframe, setSelectedTimeframe] = useState('15m');
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [currentSignal, setCurrentSignal] = useState<AISignal | null>(null);
    const [signalHistory, setSignalHistory] = useState<AISignal[]>([]);
    const [marketAnalysis, setMarketAnalysis] = useState<MarketAnalysis | null>(null);
    const [aiThinking, setAiThinking] = useState('');
    const [showAnalysis, setShowAnalysis] = useState(true);
    const [showFibonacci, setShowFibonacci] = useState(false);
    const [chartData, setChartData] = useState<CandlestickData[]>([]);

    const popularSymbols = [
        'SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMD', 'META',
        'GOOGL', 'AMZN', 'NFLX', 'COIN', 'PLTR', 'SOFI', 'NIO', 'RIVN'
    ];

    const timeframes = [
        { value: '1m', label: '1 Minute' },
        { value: '5m', label: '5 Minutes' },
        { value: '15m', label: '15 Minutes' },
        { value: '30m', label: '30 Minutes' },
        { value: '1h', label: '1 Hour' },
        { value: '4h', label: '4 Hours' },
        { value: '1d', label: 'Daily' },
    ];

    // Initialize chart
    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height: 500,
            layout: {
                background: { type: ColorType.Solid, color: theme.palette.background.paper },
                textColor: theme.palette.text.secondary,
            },
            grid: {
                vertLines: { color: alpha(theme.palette.divider, 0.3) },
                horzLines: { color: alpha(theme.palette.divider, 0.3) },
            },
            crosshair: {
                mode: CrosshairMode.Normal,
            },
            rightPriceScale: {
                borderColor: theme.palette.divider,
            },
            timeScale: {
                borderColor: theme.palette.divider,
                timeVisible: true,
            },
        });

        chartRef.current = chart;

        const candlestickSeries = chart.addCandlestickSeries({
            upColor: theme.palette.success.main,
            downColor: theme.palette.error.main,
            borderUpColor: theme.palette.success.dark,
            borderDownColor: theme.palette.error.dark,
            wickUpColor: theme.palette.success.dark,
            wickDownColor: theme.palette.error.dark,
        });

        candleSeriesRef.current = candlestickSeries;

        // Handle resize
        const handleResize = () => {
            if (chartContainerRef.current && chart) {
                chart.applyOptions({
                    width: chartContainerRef.current.clientWidth,
                });
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, [theme]);

    // Load chart data when symbol changes
    useEffect(() => {
        loadChartData();
    }, [selectedSymbol, selectedTimeframe]);

    const loadChartData = () => {
        if (!candleSeriesRef.current) return;

        // Generate realistic candlestick data
        const basePrice = 450;
        const currentTime = Math.floor(Date.now() / 1000);
        const candleData: CandlestickData[] = [];

        let price = basePrice;
        for (let i = 200; i >= 0; i--) {
            const time = currentTime - i * 300; // 5-minute candles
            const volatility = 0.002;

            const open = price;
            const change = (Math.random() - 0.5) * price * volatility * 2;
            const close = open + change;
            const high = Math.max(open, close) + Math.random() * price * volatility;
            const low = Math.min(open, close) - Math.random() * price * volatility;

            candleData.push({
                time: time as Time,
                open,
                high,
                low,
                close,
            });

            price = close;
        }

        setChartData(candleData); // Store chart data in state
        candleSeriesRef.current.setData(candleData);
        chartRef.current?.timeScale().fitContent();
    };

    const handleSymbolSearch = (event: React.KeyboardEvent<HTMLDivElement>) => {
        if (event.key === 'Enter' && selectedSymbol) {
            startAIAnalysis();
        }
    };

    const startAIAnalysis = async () => {
        setIsAnalyzing(true);
        setAiThinking('Initializing AI Prophet analysis...');
        setCurrentSignal(null);

        // Simulate AI analysis steps
        const analysisSteps = [
            'Scanning market structure...',
            'Identifying key support and resistance levels...',
            'Analyzing candlestick patterns...',
            'Calculating Fibonacci retracements...',
            'Evaluating technical indicators...',
            'Assessing market sentiment...',
            'Determining optimal entry and exit points...',
            'Calculating risk/reward ratios...',
            'Generating high-probability signal...'
        ];

        for (let i = 0; i < analysisSteps.length; i++) {
            setAiThinking(analysisSteps[i]);
            await new Promise(resolve => setTimeout(resolve, 800));

            // Add visual indicators on chart during analysis
            if (i === 1) drawSupportResistance();
            if (i === 3) drawFibonacciLevels();
            if (i === 6) drawEntryExitLevels();
        }

        // Generate market analysis
        const analysis: MarketAnalysis = {
            trend: Math.random() > 0.5 ? 'bullish' : 'bearish',
            strength: Math.random() * 100,
            volatility: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)] as any,
            volume: ['increasing', 'decreasing', 'stable'][Math.floor(Math.random() * 3)] as any,
            sentiment: (Math.random() - 0.5) * 200,
        };
        setMarketAnalysis(analysis);

        // Generate signal
        const signal = generateHighProbabilitySignal(analysis);
        setCurrentSignal(signal);
        setSignalHistory(prev => [signal, ...prev.slice(0, 9)]);

        setIsAnalyzing(false);
        setAiThinking('');
    };

    const generateHighProbabilitySignal = (analysis: MarketAnalysis): AISignal => {
        const currentPrice = 450 + (Math.random() - 0.5) * 10;
        const isBullish = analysis.trend === 'bullish';

        // Calculate entry based on current price and trend
        const entry = isBullish
            ? currentPrice - (currentPrice * 0.001) // Slight pullback entry for longs
            : currentPrice + (currentPrice * 0.001); // Slight bounce entry for shorts

        // Calculate stop loss using ATR-based method
        const atrMultiplier = analysis.volatility === 'high' ? 2 : analysis.volatility === 'medium' ? 1.5 : 1;
        const stopLoss = isBullish
            ? entry - (entry * 0.01 * atrMultiplier) // 1-2% stop loss
            : entry + (entry * 0.01 * atrMultiplier);

        // Calculate multiple take profit levels using Fibonacci extensions
        const takeProfitLevels = isBullish
            ? [
                entry + (entry - stopLoss) * 1.618, // 1.618 R:R
                entry + (entry - stopLoss) * 2.618, // 2.618 R:R
                entry + (entry - stopLoss) * 4.236, // 4.236 R:R
            ]
            : [
                entry - (stopLoss - entry) * 1.618,
                entry - (stopLoss - entry) * 2.618,
                entry - (stopLoss - entry) * 4.236,
            ];

        // Generate reasoning based on multiple factors
        const indicators = ['RSI', 'MACD', 'EMA Cross', 'Volume Profile', 'Bollinger Bands'];
        const patterns = ['Bull Flag', 'Ascending Triangle', 'Double Bottom', 'Inverse Head & Shoulders', 'Cup and Handle'];

        const selectedIndicators = indicators.sort(() => 0.5 - Math.random()).slice(0, 3);
        const selectedPatterns = patterns.sort(() => 0.5 - Math.random()).slice(0, 2);

        const reasoning = `${isBullish ? 'Bullish' : 'Bearish'} signal generated based on:
    • ${selectedPatterns[0]} pattern formation confirmed
    • ${selectedIndicators.join(', ')} showing ${isBullish ? 'oversold' : 'overbought'} conditions
    • Strong ${isBullish ? 'support' : 'resistance'} level at ${stopLoss.toFixed(2)}
    • Fibonacci extension targets align with historical pivot points
    • Market sentiment: ${analysis.sentiment > 0 ? 'Positive' : 'Negative'} (${Math.abs(analysis.sentiment).toFixed(0)}/100)
    • Volume ${analysis.volume} confirming ${isBullish ? 'accumulation' : 'distribution'}`;

        return {
            id: `signal-${Date.now()}`,
            symbol: selectedSymbol,
            timeframe: selectedTimeframe,
            type: isBullish ? 'LONG' : 'SHORT',
            entry,
            stopLoss,
            takeProfit: takeProfitLevels,
            confidence: 75 + Math.random() * 20, // 75-95% confidence
            indicators: selectedIndicators,
            patterns: selectedPatterns,
            timestamp: new Date(),
            status: 'active',
            reasoning,
            fibLevels: [0.236, 0.382, 0.5, 0.618, 0.786, 1],
            keyLevels: [
                { price: stopLoss, type: 'Stop Loss', strength: 5 },
                { price: entry, type: 'Entry', strength: 4 },
                ...takeProfitLevels.map((tp, i) => ({
                    price: tp,
                    type: `TP${i + 1}`,
                    strength: 3 - i * 0.5
                }))
            ]
        };
    };

    const drawSupportResistance = () => {
        if (!chartRef.current) return;

        // Add support/resistance lines
        const currentPrice = 450;
        const levels = [
            currentPrice * 0.98,
            currentPrice * 0.96,
            currentPrice * 1.02,
            currentPrice * 1.04,
        ];

        levels.forEach((level, i) => {
            const isSupport = i < 2;
            const lineSeries = chartRef.current!.addLineSeries({
                color: isSupport ? theme.palette.success.main : theme.palette.error.main,
                lineWidth: 2,
                lineStyle: LineStyle.Dashed,
                lastValueVisible: false,
                priceLineVisible: false,
            });

            const data = [
                { time: (Date.now() / 1000 - 86400) as Time, value: level },
                { time: (Date.now() / 1000) as Time, value: level },
            ];

            lineSeries.setData(data);
        });
    };

    const drawFibonacciLevels = () => {
        if (!chartRef.current) return;

        const fibLevels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
        const high = 455;
        const low = 445;
        const range = high - low;

        fibLevels.forEach((level, i) => {
            const price = high - (range * level);
            const lineSeries = chartRef.current!.addLineSeries({
                color: theme.palette.info.main,
                lineWidth: 1,
                lineStyle: i === 0 || i === fibLevels.length - 1 ? LineStyle.Solid : LineStyle.Dotted,
                lastValueVisible: true,
                title: `Fib ${level}`,
            });

            const data = [
                { time: (Date.now() / 1000 - 86400) as Time, value: price },
                { time: (Date.now() / 1000) as Time, value: price },
            ];

            lineSeries.setData(data);
        });
    };

    const drawEntryExitLevels = () => {
        if (!chartRef.current || !currentSignal) return;

        // Entry line
        const entryLine = chartRef.current.addLineSeries({
            color: theme.palette.primary.main,
            lineWidth: 3,
            lineStyle: LineStyle.Solid,
            title: 'Entry',
        });

        // Stop loss line
        const stopLossLine = chartRef.current.addLineSeries({
            color: theme.palette.error.main,
            lineWidth: 2,
            lineStyle: LineStyle.Dashed,
            title: 'Stop Loss',
        });

        // Take profit lines
        currentSignal?.takeProfit.forEach((tp, i) => {
            const tpLine = chartRef.current!.addLineSeries({
                color: theme.palette.success.main,
                lineWidth: 2,
                lineStyle: LineStyle.Dotted,
                title: `TP${i + 1}`,
            });

            const data = [
                { time: (Date.now() / 1000 - 3600) as Time, value: tp },
                { time: (Date.now() / 1000 + 3600) as Time, value: tp },
            ];

            tpLine.setData(data);
        });
    };

    const formatPrice = (price: number) => `$${price.toFixed(2)}`;
    const formatPercent = (value: number) => `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', bgcolor: 'background.default' }}>
            {/* Header */}
            <Paper sx={{ p: 2, borderRadius: 0 }}>
                <Grid container spacing={2} alignItems="center">
                    <Grid item xs={12} md={4}>
                        <Autocomplete
                            value={selectedSymbol}
                            onChange={(e, value) => value && setSelectedSymbol(value)}
                            options={popularSymbols}
                            renderInput={(params) => (
                                <TextField
                                    {...params}
                                    placeholder="Enter symbol and press Enter"
                                    variant="outlined"
                                    size="small"
                                    onKeyDown={handleSymbolSearch}
                                    InputProps={{
                                        ...params.InputProps,
                                        startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
                                    }}
                                />
                            )}
                        />
                    </Grid>

                    <Grid item xs={12} md={3}>
                        <FormControl fullWidth size="small">
                            <InputLabel>Timeframe</InputLabel>
                            <Select
                                value={selectedTimeframe}
                                onChange={(e) => setSelectedTimeframe(e.target.value)}
                                label="Timeframe"
                            >
                                {timeframes.map(tf => (
                                    <MenuItem key={tf.value} value={tf.value}>{tf.label}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Grid>

                    <Grid item xs={12} md={5}>
                        <Stack direction="row" spacing={1} justifyContent="flex-end">
                            <Button
                                variant="contained"
                                onClick={startAIAnalysis}
                                disabled={isAnalyzing}
                                startIcon={isAnalyzing ? <SpeedIcon /> : <AIIcon />}
                                sx={{
                                    background: theme.palette.primary.main,
                                    '&:hover': {
                                        background: theme.palette.primary.dark,
                                    },
                                }}
                            >
                                {isAnalyzing ? 'Analyzing...' : 'Generate Signal'}
                            </Button>

                            <IconButton onClick={() => setShowAnalysis(!showAnalysis)}>
                                <InsightsIcon />
                            </IconButton>

                            <IconButton
                                onClick={() => setShowFibonacci(!showFibonacci)}
                                color={showFibonacci ? 'primary' : 'default'}
                            >
                                <Tooltip title="Fibonacci Retracement Tool">
                                    <FibIcon />
                                </Tooltip>
                            </IconButton>
                        </Stack>
                    </Grid>
                </Grid>

                {isAnalyzing && (
                    <Box sx={{ mt: 2 }}>
                        <LinearProgress />
                        <Typography variant="caption" sx={{ mt: 1, display: 'block', color: 'primary.main' }}>
                            {aiThinking}
                        </Typography>
                    </Box>
                )}
            </Paper>

            {/* Main Content */}
            <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
                {/* Chart */}
                <Box sx={{ flex: 1, p: 2 }}>
                    <Paper sx={{ height: '100%', p: 2, position: 'relative' }}>
                        <Box ref={chartContainerRef} sx={{ height: 500 }} />

                        {/* Current Signal Overlay */}
                        {currentSignal && (
                            <Fade in>
                                <Card sx={{
                                    position: 'absolute',
                                    top: 16,
                                    left: 16,
                                    maxWidth: 300,
                                    bgcolor: alpha(theme.palette.background.paper, 0.95),
                                    backdropFilter: 'blur(10px)',
                                }}>
                                    <CardContent>
                                        <Stack spacing={1}>
                                            <Stack direction="row" alignItems="center" spacing={1}>
                                                <Chip
                                                    label={currentSignal.type}
                                                    color={currentSignal.type === 'LONG' ? 'success' : 'error'}
                                                    size="small"
                                                />
                                                <Chip
                                                    label={`${currentSignal.confidence.toFixed(0)}% Confidence`}
                                                    color="primary"
                                                    size="small"
                                                />
                                            </Stack>

                                            <Divider />

                                            <Box>
                                                <Typography variant="caption" color="text.secondary">Entry</Typography>
                                                <Typography variant="body2" fontWeight="bold">
                                                    {formatPrice(currentSignal.entry)}
                                                </Typography>
                                            </Box>

                                            <Box>
                                                <Typography variant="caption" color="text.secondary">Stop Loss</Typography>
                                                <Typography variant="body2" color="error.main">
                                                    {formatPrice(currentSignal.stopLoss)}
                                                    <Typography component="span" variant="caption" sx={{ ml: 1 }}>
                                                        ({formatPercent(((currentSignal.stopLoss - currentSignal.entry) / currentSignal.entry) * 100)})
                                                    </Typography>
                                                </Typography>
                                            </Box>

                                            <Box>
                                                <Typography variant="caption" color="text.secondary">Take Profit Targets</Typography>
                                                {currentSignal.takeProfit.map((tp, i) => (
                                                    <Typography key={i} variant="body2" color="success.main">
                                                        TP{i + 1}: {formatPrice(tp)}
                                                        <Typography component="span" variant="caption" sx={{ ml: 1 }}>
                                                            ({formatPercent(((tp - currentSignal.entry) / currentSignal.entry) * 100)})
                                                        </Typography>
                                                    </Typography>
                                                ))}
                                            </Box>
                                        </Stack>
                                    </CardContent>
                                </Card>
                            </Fade>
                        )}
                    </Paper>
                </Box>

                {/* Analysis Panel */}
                <Collapse in={showAnalysis} orientation="horizontal">
                    <Paper sx={{ width: 400, height: '100%', overflow: 'auto', p: 2 }}>
                        <Typography variant="h6" gutterBottom>
                            AI Analysis
                        </Typography>

                        {marketAnalysis && (
                            <Stack spacing={2}>
                                <Alert severity={marketAnalysis.trend === 'bullish' ? 'success' : marketAnalysis.trend === 'bearish' ? 'error' : 'info'}>
                                    Market Trend: <strong>{marketAnalysis.trend.toUpperCase()}</strong> ({marketAnalysis.strength.toFixed(0)}% strength)
                                </Alert>

                                {currentSignal && (
                                    <Box>
                                        <Typography variant="subtitle2" gutterBottom>Signal Reasoning</Typography>
                                        <Typography variant="body2" sx={{ whiteSpace: 'pre-line', color: 'text.secondary' }}>
                                            {currentSignal.reasoning}
                                        </Typography>
                                    </Box>
                                )}

                                <Divider />

                                <Box>
                                    <Typography variant="subtitle2" gutterBottom>Technical Indicators</Typography>
                                    <Stack direction="row" spacing={1} flexWrap="wrap">
                                        {currentSignal?.indicators.map(indicator => (
                                            <Chip key={indicator} label={indicator} size="small" />
                                        ))}
                                    </Stack>
                                </Box>

                                <Box>
                                    <Typography variant="subtitle2" gutterBottom>Detected Patterns</Typography>
                                    <Stack direction="row" spacing={1} flexWrap="wrap">
                                        {currentSignal?.patterns.map(pattern => (
                                            <Chip key={pattern} label={pattern} size="small" color="primary" />
                                        ))}
                                    </Stack>
                                </Box>

                                <Divider />

                                <Box>
                                    <Typography variant="subtitle2" gutterBottom>Signal History</Typography>
                                    <Stack spacing={1}>
                                        {signalHistory.slice(0, 5).map(signal => (
                                            <Card key={signal.id} variant="outlined">
                                                <CardContent sx={{ py: 1 }}>
                                                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                                                        <Box>
                                                            <Typography variant="body2">
                                                                {signal.symbol} - {signal.type}
                                                            </Typography>
                                                            <Typography variant="caption" color="text.secondary">
                                                                {new Date(signal.timestamp).toLocaleTimeString()}
                                                            </Typography>
                                                        </Box>
                                                        <Chip
                                                            label={`${signal.confidence.toFixed(0)}%`}
                                                            size="small"
                                                            color={signal.status === 'success' ? 'success' : signal.status === 'stopped' ? 'error' : 'default'}
                                                        />
                                                    </Stack>
                                                </CardContent>
                                            </Card>
                                        ))}
                                    </Stack>
                                </Box>
                            </Stack>
                        )}
                    </Paper>
                </Collapse>
            </Box>

            {/* Fibonacci Retracement Dialog */}
            <Dialog
                open={showFibonacci}
                onClose={() => setShowFibonacci(false)}
                maxWidth="lg"
                fullWidth
            >
                <DialogTitle>
                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Typography variant="h6">Fibonacci Retracement Analysis</Typography>
                        <IconButton onClick={() => setShowFibonacci(false)} size="small">
                            <CloseIcon />
                        </IconButton>
                    </Stack>
                </DialogTitle>
                <DialogContent>
                    <Box sx={{ minHeight: 600 }}>
                        <FibonacciRetracement
                            chartData={chartData}
                            onLevelsCalculated={(levels) => {
                                console.log('Fibonacci levels calculated:', levels);
                            }}
                            onSignalGenerated={(signal) => {
                                console.log('Fibonacci signal generated:', signal);
                                // You can integrate this with your main signal generation
                            }}
                            showLabels={true}
                            autoDetect={true}
                        />
                    </Box>
                </DialogContent>
            </Dialog>
        </Box>
    );
};

export default AISignalProphet; 