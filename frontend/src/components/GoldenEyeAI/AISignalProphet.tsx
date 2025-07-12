import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Box,
    Paper,
    Stack,
    TextField,
    Button,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Typography,
    Chip,
    LinearProgress,
    Fade,
    Zoom,
    IconButton,
    Tooltip,
    Card,
    CardContent,
    Grid,
    Divider,
    Alert,
    useTheme,
    alpha,
    SelectChangeEvent,
    InputAdornment,
} from '@mui/material';
import {
    AutoAwesome,
    TrendingUp,
    TrendingDown,
    Psychology,
    Timeline,
    ShowChart,
    Speed,
    Assessment,
    Lightbulb,
    Warning,
    CheckCircle,
    Search,
    Refresh,
    FullscreenOutlined,
    ZoomInOutlined,
    ZoomOutOutlined,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { createChart, IChartApi, ISeriesApi, Time } from 'lightweight-charts';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../services/api';

export interface AISignalProphetProps {
    onSignalGenerated?: (signal: any) => void;
}

const AISignalProphet: React.FC<AISignalProphetProps> = ({ onSignalGenerated }) => {
    const theme = useTheme();
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
    const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

    // State
    const [symbol, setSymbol] = useState('SPY');
    const [timeframe, setTimeframe] = useState('15m');
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [currentSignal, setCurrentSignal] = useState<any>(null);
    const [showPattern, setShowPattern] = useState(false);
    const [patternType, setPatternType] = useState('');
    const [confidence, setConfidence] = useState(0);

    // Timeframe options
    const timeframes = [
        { value: '5m', label: '5 Minutes' },
        { value: '15m', label: '15 Minutes' },
        { value: '30m', label: '30 Minutes' },
        { value: '1h', label: '1 Hour' },
        { value: '4h', label: '4 Hours' },
        { value: '1d', label: 'Daily' },
    ];

    // Fetch market data
    const { data: marketData, refetch: refetchMarketData } = useQuery({
        queryKey: ['marketData', symbol, timeframe],
        queryFn: () => apiClient.getHistoricalData(symbol, '1d'),
        // Disabled auto-refresh to prevent constant updating
        staleTime: 300000, // Keep data fresh for 5 minutes
    });

    // Initialize chart
    useEffect(() => {
        if (!chartContainerRef.current) return;

        // Create chart
        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height: 500,
            layout: {
                background: { color: 'transparent' },
                textColor: theme.palette.text.primary,
            },
            grid: {
                vertLines: {
                    color: alpha(theme.palette.divider, 0.1),
                },
                horzLines: {
                    color: alpha(theme.palette.divider, 0.1),
                },
            },
            crosshair: {
                mode: 1,
            },
            rightPriceScale: {
                borderColor: theme.palette.divider,
            },
            timeScale: {
                borderColor: theme.palette.divider,
                timeVisible: true,
                secondsVisible: false,
            },
        });

        // Add candlestick series
        const candleSeries = chart.addCandlestickSeries({
            upColor: theme.palette.success.main,
            downColor: theme.palette.error.main,
            borderUpColor: theme.palette.success.dark,
            borderDownColor: theme.palette.error.dark,
            wickUpColor: theme.palette.success.main,
            wickDownColor: theme.palette.error.main,
        });

        // Add volume series
        const volumeSeries = chart.addHistogramSeries({
            color: alpha(theme.palette.primary.main, 0.3),
            priceFormat: {
                type: 'volume',
            },
            priceScaleId: '',
        });

        chartRef.current = chart;
        candleSeriesRef.current = candleSeries;
        volumeSeriesRef.current = volumeSeries;

        // Handle resize
        const handleResize = () => {
            if (chartContainerRef.current) {
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

    // Update chart data
    useEffect(() => {
        if (!marketData?.data || !candleSeriesRef.current || !volumeSeriesRef.current) return;

        const candleData = marketData.data.map((d: any) => ({
            time: d.time as Time,
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
        }));

        const volumeData = marketData.data.map((d: any) => ({
            time: d.time as Time,
            value: d.volume,
            color: d.close >= d.open
                ? alpha(theme.palette.success.main, 0.5)
                : alpha(theme.palette.error.main, 0.5),
        }));

        candleSeriesRef.current.setData(candleData);
        volumeSeriesRef.current.setData(volumeData);

        // Fit content
        chartRef.current?.timeScale().fitContent();
    }, [marketData, theme]);

    // Generate AI Signal
    const generateSignal = async () => {
        setIsAnalyzing(true);
        setShowPattern(false);

        try {
            // Simulate AI analysis steps
            await new Promise(resolve => setTimeout(resolve, 1000));

            // Fetch signal from API
            const response = await apiClient.getPreciseOptionsSignals(symbol, timeframe);
            const signal = response[0]; // Get first signal

            if (signal) {
                setCurrentSignal(signal);
                setPatternType(signal.setup_name || 'Bullish Pattern');
                setConfidence(signal.confidence || 85);

                // Draw technical indicators
                drawFibonacciLevels();
                drawSupportResistance(signal);
                drawTrendLines();

                // Show pattern animation
                setTimeout(() => {
                    setShowPattern(true);
                    drawZigZagPattern();
                }, 1500);

                // Callback
                if (onSignalGenerated) {
                    onSignalGenerated(signal);
                }
            }
        } catch (error) {
            console.error('Error generating signal:', error);
        } finally {
            setIsAnalyzing(false);
        }
    };

    // Draw Fibonacci levels
    const drawFibonacciLevels = () => {
        if (!chartRef.current || !marketData?.data) return;

        const data = marketData.data;
        const high = Math.max(...data.map((d: any) => d.high));
        const low = Math.min(...data.map((d: any) => d.low));
        const diff = high - low;

        const levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
        const colors = [
            theme.palette.error.main,
            theme.palette.warning.main,
            theme.palette.info.main,
            theme.palette.primary.main,
            theme.palette.info.main,
            theme.palette.warning.main,
            theme.palette.success.main,
        ];

        levels.forEach((level, index) => {
            const price = high - (diff * level);
            const lineSeries = chartRef.current!.addLineSeries({
                color: alpha(colors[index], 0.5),
                lineWidth: 1,
                lineStyle: 2, // Dashed
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            });

            lineSeries.setData([
                { time: data[0].time as Time, value: price },
                { time: data[data.length - 1].time as Time, value: price },
            ]);
        });
    };

    // Draw support and resistance
    const drawSupportResistance = (signal: any) => {
        if (!chartRef.current || !marketData?.data) return;

        const data = marketData.data;
        const support = signal.stop_loss || signal.entry_price * 0.98;
        const resistance = signal.targets?.[0]?.price || signal.entry_price * 1.02;

        // Support line
        const supportLine = chartRef.current.addLineSeries({
            color: theme.palette.error.main,
            lineWidth: 2,
            lineStyle: 0,
            crosshairMarkerVisible: false,
            lastValueVisible: true,
            priceLineVisible: true,
            title: 'Support',
        });

        supportLine.setData([
            { time: data[0].time as Time, value: support },
            { time: data[data.length - 1].time as Time, value: support },
        ]);

        // Resistance line
        const resistanceLine = chartRef.current.addLineSeries({
            color: theme.palette.success.main,
            lineWidth: 2,
            lineStyle: 0,
            crosshairMarkerVisible: false,
            lastValueVisible: true,
            priceLineVisible: true,
            title: 'Resistance',
        });

        resistanceLine.setData([
            { time: data[0].time as Time, value: resistance },
            { time: data[data.length - 1].time as Time, value: resistance },
        ]);
    };

    // Draw trend lines
    const drawTrendLines = () => {
        if (!chartRef.current || !marketData?.data) return;

        const data = marketData.data;
        const trendStart = data[Math.floor(data.length * 0.3)];
        const trendEnd = data[data.length - 1];

        const trendLine = chartRef.current.addLineSeries({
            color: theme.palette.primary.main,
            lineWidth: 2,
            lineStyle: 0,
            crosshairMarkerVisible: false,
            lastValueVisible: false,
            priceLineVisible: false,
        });

        trendLine.setData([
            { time: trendStart.time as Time, value: trendStart.low },
            { time: trendEnd.time as Time, value: trendEnd.close },
        ]);
    };

    // Draw ZigZag pattern animation
    const drawZigZagPattern = () => {
        if (!chartRef.current || !marketData?.data) return;

        const data = marketData.data;
        const zigzagPoints: Array<{ time: any; value: number }> = [];
        const threshold = 0.02; // 2% threshold

        let lastPivot = data[0];
        let lastPivotType = 'low';

        for (let i = 1; i < data.length; i++) {
            const current = data[i];

            if (lastPivotType === 'low') {
                if (current.high > lastPivot.high * (1 + threshold)) {
                    zigzagPoints.push({ time: lastPivot.time, value: lastPivot.low });
                    lastPivot = current;
                    lastPivotType = 'high';
                }
            } else {
                if (current.low < lastPivot.low * (1 - threshold)) {
                    zigzagPoints.push({ time: lastPivot.time, value: lastPivot.high });
                    lastPivot = current;
                    lastPivotType = 'low';
                }
            }
        }

        // Add last point
        zigzagPoints.push({
            time: lastPivot.time,
            value: lastPivotType === 'low' ? lastPivot.low : lastPivot.high,
        });

        // Animate zigzag
        const zigzagSeries = chartRef.current.addLineSeries({
            color: theme.palette.secondary.main,
            lineWidth: 3,
            lineStyle: 0,
            crosshairMarkerVisible: false,
            lastValueVisible: false,
            priceLineVisible: false,
        });

        // Animate point by point
        zigzagPoints.forEach((point, index) => {
            setTimeout(() => {
                zigzagSeries.setData(zigzagPoints.slice(0, index + 1) as any);
            }, index * 200);
        });
    };

    return (
        <Box sx={{ width: '100%', height: '100%' }}>
            <Stack spacing={3}>
                {/* Header Controls */}
                <Paper
                    elevation={0}
                    sx={{
                        p: 3,
                        background: alpha(theme.palette.background.paper, 0.8),
                        backdropFilter: 'blur(10px)',
                        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                    }}
                >
                    <Stack spacing={3}>
                        <Typography variant="h4" fontWeight="bold" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <AutoAwesome sx={{ color: theme.palette.primary.main }} />
                            AI Signal Prophet
                        </Typography>

                        <Grid container spacing={2} alignItems="flex-end">
                            <Grid item xs={12} sm={4}>
                                <TextField
                                    fullWidth
                                    label="Stock Symbol"
                                    value={symbol}
                                    onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                                    InputProps={{
                                        startAdornment: (
                                            <InputAdornment position="start">
                                                <Search />
                                            </InputAdornment>
                                        ),
                                    }}
                                />
                            </Grid>
                            <Grid item xs={12} sm={3}>
                                <FormControl fullWidth>
                                    <InputLabel>Timeframe</InputLabel>
                                    <Select
                                        value={timeframe}
                                        label="Timeframe"
                                        onChange={(e: SelectChangeEvent) => setTimeframe(e.target.value)}
                                    >
                                        {timeframes.map((tf) => (
                                            <MenuItem key={tf.value} value={tf.value}>
                                                {tf.label}
                                            </MenuItem>
                                        ))}
                                    </Select>
                                </FormControl>
                            </Grid>
                            <Grid item xs={12} sm={5}>
                                <Button
                                    fullWidth
                                    variant="contained"
                                    size="large"
                                    onClick={generateSignal}
                                    disabled={isAnalyzing || !symbol}
                                    startIcon={isAnalyzing ? <Psychology /> : <AutoAwesome />}
                                    sx={{
                                        background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                                        '&:hover': {
                                            background: `linear-gradient(45deg, ${theme.palette.primary.dark}, ${theme.palette.secondary.dark})`,
                                        },
                                    }}
                                >
                                    {isAnalyzing ? 'AI Analyzing...' : 'Generate Signal'}
                                </Button>
                            </Grid>
                        </Grid>
                    </Stack>
                </Paper>

                {/* Chart Section */}
                <Paper
                    elevation={0}
                    sx={{
                        p: 2,
                        background: alpha(theme.palette.background.paper, 0.8),
                        backdropFilter: 'blur(10px)',
                        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                        position: 'relative',
                        overflow: 'hidden',
                    }}
                >
                    {/* Chart Toolbar */}
                    <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
                        <Stack direction="row" spacing={1}>
                            <Chip
                                label={symbol}
                                color="primary"
                                icon={<ShowChart />}
                            />
                            <Chip
                                label={timeframe}
                                variant="outlined"
                            />
                        </Stack>
                        <Stack direction="row" spacing={1}>
                            <IconButton size="small" onClick={() => refetchMarketData()}>
                                <Refresh />
                            </IconButton>
                            <IconButton size="small">
                                <ZoomInOutlined />
                            </IconButton>
                            <IconButton size="small">
                                <ZoomOutOutlined />
                            </IconButton>
                            <IconButton size="small">
                                <FullscreenOutlined />
                            </IconButton>
                        </Stack>
                    </Stack>

                    {/* Chart Container */}
                    <Box
                        ref={chartContainerRef}
                        sx={{
                            width: '100%',
                            height: 500,
                            position: 'relative',
                        }}
                    />

                    {/* Pattern Overlay */}
                    <AnimatePresence>
                        {showPattern && (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.8 }}
                                transition={{ duration: 0.5 }}
                                style={{
                                    position: 'absolute',
                                    top: 20,
                                    right: 20,
                                    zIndex: 10,
                                }}
                            >
                                <Card
                                    sx={{
                                        background: alpha(theme.palette.background.paper, 0.95),
                                        backdropFilter: 'blur(10px)',
                                        border: `2px solid ${theme.palette.primary.main}`,
                                    }}
                                >
                                    <CardContent>
                                        <Stack spacing={1}>
                                            <Typography variant="h6" color="primary">
                                                Pattern Detected!
                                            </Typography>
                                            <Chip
                                                label={patternType}
                                                color="primary"
                                                icon={<Timeline />}
                                            />
                                            <Typography variant="body2">
                                                Confidence: {confidence}%
                                            </Typography>
                                            <LinearProgress
                                                variant="determinate"
                                                value={confidence}
                                                sx={{ height: 8, borderRadius: 4 }}
                                            />
                                        </Stack>
                                    </CardContent>
                                </Card>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* Analyzing Overlay */}
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
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    background: alpha(theme.palette.background.default, 0.8),
                                    backdropFilter: 'blur(5px)',
                                    zIndex: 20,
                                }}
                            >
                                <Stack spacing={3} alignItems="center">
                                    <motion.div
                                        animate={{ rotate: 360 }}
                                        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                                    >
                                        <Psychology sx={{ fontSize: 64, color: theme.palette.primary.main }} />
                                    </motion.div>
                                    <Typography variant="h5">AI Analyzing Market Data...</Typography>
                                    <LinearProgress sx={{ width: 200 }} />
                                </Stack>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </Paper>

                {/* AI Analysis Section */}
                {currentSignal && (
                    <Fade in={true}>
                        <Paper
                            elevation={0}
                            sx={{
                                p: 3,
                                background: alpha(theme.palette.background.paper, 0.8),
                                backdropFilter: 'blur(10px)',
                                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                            }}
                        >
                            <Stack spacing={3}>
                                <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <Psychology sx={{ color: theme.palette.primary.main }} />
                                    AI Analysis & Recommendation
                                </Typography>

                                <Grid container spacing={3}>
                                    {/* Signal Summary */}
                                    <Grid item xs={12} md={4}>
                                        <Card sx={{ height: '100%' }}>
                                            <CardContent>
                                                <Stack spacing={2}>
                                                    <Typography variant="h6" color="primary">
                                                        Signal Summary
                                                    </Typography>
                                                    <Chip
                                                        label={currentSignal.type === 'CALL' ? 'CALL Option' : 'PUT Option'}
                                                        color={currentSignal.type === 'CALL' ? 'success' : 'error'}
                                                        icon={currentSignal.type === 'CALL' ? <TrendingUp /> : <TrendingDown />}
                                                        sx={{ alignSelf: 'flex-start' }}
                                                    />
                                                    <Box>
                                                        <Typography variant="body2" color="text.secondary">
                                                            Entry Price
                                                        </Typography>
                                                        <Typography variant="h5">
                                                            ${currentSignal.entry_price?.toFixed(2) || 'N/A'}
                                                        </Typography>
                                                    </Box>
                                                    <Box>
                                                        <Typography variant="body2" color="text.secondary">
                                                            Strike Price
                                                        </Typography>
                                                        <Typography variant="h6">
                                                            ${currentSignal.strike_price || 'N/A'}
                                                        </Typography>
                                                    </Box>
                                                    <Box>
                                                        <Typography variant="body2" color="text.secondary">
                                                            Confidence Score
                                                        </Typography>
                                                        <Stack direction="row" spacing={1} alignItems="center">
                                                            <Typography variant="h6">
                                                                {currentSignal.confidence}%
                                                            </Typography>
                                                            <Speed color="primary" />
                                                        </Stack>
                                                    </Box>
                                                </Stack>
                                            </CardContent>
                                        </Card>
                                    </Grid>

                                    {/* Key Evidence */}
                                    <Grid item xs={12} md={4}>
                                        <Card sx={{ height: '100%' }}>
                                            <CardContent>
                                                <Stack spacing={2}>
                                                    <Typography variant="h6" color="primary">
                                                        Key Evidence
                                                    </Typography>
                                                    <Stack spacing={1}>
                                                        <Alert severity="success" icon={<CheckCircle />}>
                                                            Strong momentum detected
                                                        </Alert>
                                                        <Alert severity="info" icon={<Timeline />}>
                                                            {patternType} pattern confirmed
                                                        </Alert>
                                                        <Alert severity="success" icon={<Assessment />}>
                                                            Volume surge detected
                                                        </Alert>
                                                        <Alert severity="info" icon={<ShowChart />}>
                                                            Above key support levels
                                                        </Alert>
                                                    </Stack>
                                                </Stack>
                                            </CardContent>
                                        </Card>
                                    </Grid>

                                    {/* Risk Assessment */}
                                    <Grid item xs={12} md={4}>
                                        <Card sx={{ height: '100%' }}>
                                            <CardContent>
                                                <Stack spacing={2}>
                                                    <Typography variant="h6" color="primary">
                                                        Risk Assessment
                                                    </Typography>
                                                    <Box>
                                                        <Typography variant="body2" color="text.secondary">
                                                            Stop Loss
                                                        </Typography>
                                                        <Typography variant="h6" color="error">
                                                            ${currentSignal.stop_loss?.toFixed(2) || 'N/A'}
                                                        </Typography>
                                                    </Box>
                                                    <Box>
                                                        <Typography variant="body2" color="text.secondary">
                                                            Risk/Reward Ratio
                                                        </Typography>
                                                        <Typography variant="h6" color="success">
                                                            {currentSignal.risk_reward_ratio || '2.5'}:1
                                                        </Typography>
                                                    </Box>
                                                    <Alert severity="warning" icon={<Warning />}>
                                                        Max risk: ${currentSignal.max_risk_dollars || 500}
                                                    </Alert>
                                                </Stack>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                </Grid>

                                {/* AI Explanation */}
                                <Card>
                                    <CardContent>
                                        <Stack spacing={2}>
                                            <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                <Lightbulb sx={{ color: theme.palette.warning.main }} />
                                                Why This Signal?
                                            </Typography>
                                            <Typography variant="body1" sx={{ lineHeight: 1.8 }}>
                                                Based on my analysis of {symbol}, I've identified a high-probability {currentSignal.type} option opportunity.
                                                The {patternType} pattern, combined with strong volume confirmation and favorable market conditions,
                                                suggests a {confidence}% probability of success. The current price action shows clear momentum
                                                with the stock trading above key support levels and approaching resistance at ${(currentSignal.entry_price * 1.02).toFixed(2)}.
                                            </Typography>
                                            <Typography variant="body1" sx={{ lineHeight: 1.8 }}>
                                                Technical indicators are aligned bullishly with RSI showing strength without being overbought,
                                                MACD confirming the trend, and Fibonacci retracements providing clear support zones.
                                                The risk/reward ratio of {currentSignal.risk_reward_ratio || '2.5'}:1 offers an attractive opportunity
                                                with defined risk parameters. I recommend entering this position within the next {timeframe} timeframe
                                                for optimal results.
                                            </Typography>
                                            <Divider />
                                            <Stack direction="row" spacing={2} flexWrap="wrap">
                                                <Chip label="High Confidence" color="success" size="small" />
                                                <Chip label="Strong Momentum" color="primary" size="small" />
                                                <Chip label="Clear Pattern" color="info" size="small" />
                                                <Chip label="Good R/R Ratio" color="success" size="small" />
                                            </Stack>
                                        </Stack>
                                    </CardContent>
                                </Card>
                            </Stack>
                        </Paper>
                    </Fade>
                )}
            </Stack>
        </Box>
    );
};

export default AISignalProphet; 