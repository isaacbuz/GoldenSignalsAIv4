import React, { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, Time } from 'lightweight-charts';
import {
    Box,
    Paper,
    Typography,
    IconButton,
    ToggleButton,
    ToggleButtonGroup,
    Chip,
    Stack,
    TextField,
    Autocomplete,
    Menu,
    MenuItem,
    Tooltip,
    Divider,
    alpha,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    ShowChart,
    Timeline,
    GridOn,
    Fullscreen,
    Settings,
    VolumeUp,
    AutoGraph,
    Psychology,
    CheckCircle,
    Cancel,
    Schedule,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface SignalData {
    id: string;
    timestamp: Time;
    type: 'BUY' | 'SELL';
    price: number;
    pattern: string;
    confidence: number;
    status: 'active' | 'success' | 'failed';
}

const TrendSpiderChart: React.FC = () => {
    const theme = useTheme();
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
    const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

    const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
    const [selectedTimeframe, setSelectedTimeframe] = useState('5m');
    const [showVolume, setShowVolume] = useState(true);
    const [showGrid, setShowGrid] = useState(true);
    const [showTrendLines, setShowTrendLines] = useState(false);
    const [showFibonacci, setShowFibonacci] = useState(false);
    const [showPatterns, setShowPatterns] = useState(true);
    const [isAIActive, setIsAIActive] = useState(true);
    const [detectedPatterns, setDetectedPatterns] = useState<string[]>([]);
    const [currentPrice, setCurrentPrice] = useState(0);
    const [priceChange, setPriceChange] = useState(0);
    const [signalHistory, setSignalHistory] = useState<SignalData[]>([]);

    const popularSymbols = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
        'V', 'JNJ', 'WMT', 'PG', 'DIS', 'MA', 'HD', 'PYPL', 'BAC', 'NFLX',
        'ADBE', 'CRM', 'PFE', 'TMO', 'CSCO', 'PEP'
    ];

    const timeframes = [
        { value: '1m', label: '1M' },
        { value: '5m', label: '5M' },
        { value: '15m', label: '15M' },
        { value: '1h', label: '1H' },
        { value: '4h', label: '4H' },
        { value: 'd', label: 'D' },
        { value: 'w', label: 'W' },
    ];

    useEffect(() => {
        if (!chartContainerRef.current) return;

        // Create chart with TrendSpider-style dark theme
        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
            layout: {
                background: { color: '#0a0e1a' },
                textColor: '#d1d4dc',
            },
            grid: {
                vertLines: {
                    visible: showGrid,
                    color: '#1e222d',
                },
                horzLines: {
                    visible: showGrid,
                    color: '#1e222d',
                },
            },
            crosshair: {
                mode: 1,
                vertLine: {
                    width: 1,
                    color: '#787b86',
                    style: 0,
                },
                horzLine: {
                    width: 1,
                    color: '#787b86',
                    style: 0,
                },
            },
            rightPriceScale: {
                borderColor: '#1e222d',
                scaleMargins: {
                    top: 0.1,
                    bottom: showVolume ? 0.2 : 0.1,
                },
            },
            timeScale: {
                borderColor: '#1e222d',
                timeVisible: true,
                secondsVisible: false,
            },
            watermark: {
                visible: true,
                fontSize: 48,
                horzAlign: 'center',
                vertAlign: 'center',
                color: 'rgba(255, 255, 255, 0.02)',
                text: selectedSymbol,
            },
        });

        // Add candlestick series
        const candlestickSeries = chart.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderUpColor: '#26a69a',
            borderDownColor: '#ef5350',
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        });

        // Add volume series
        const volumeSeries = chart.addHistogramSeries({
            color: '#26a69a',
            priceFormat: {
                type: 'volume',
            },
            priceScaleId: '',
            scaleMargins: {
                top: 0.85,
                bottom: 0,
            },
        });

        chartRef.current = chart;
        candlestickSeriesRef.current = candlestickSeries;
        volumeSeriesRef.current = volumeSeries;

        // Load initial data
        loadChartData();

        // Handle resize
        const handleResize = () => {
            if (chartContainerRef.current && chart) {
                chart.applyOptions({
                    width: chartContainerRef.current.clientWidth,
                    height: chartContainerRef.current.clientHeight,
                });
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, []);

    const loadChartData = () => {
        // Generate mock data
        const data = [];
        const volumeData = [];
        const basePrice = 180;
        const baseTime = Math.floor(Date.now() / 1000) - 100 * 300;

        for (let i = 0; i < 100; i++) {
            const time = baseTime + i * 300;
            const open = basePrice + Math.random() * 10 - 5;
            const close = open + Math.random() * 4 - 2;
            const high = Math.max(open, close) + Math.random() * 2;
            const low = Math.min(open, close) - Math.random() * 2;
            const volume = Math.floor(Math.random() * 1000000) + 500000;

            data.push({ time: time as Time, open, high, low, close });
            volumeData.push({
                time: time as Time,
                value: volume,
                color: close > open ? '#26a69a' : '#ef5350',
            });
        }

        candlestickSeriesRef.current?.setData(data);
        volumeSeriesRef.current?.setData(volumeData);

        // Update current price
        const lastCandle = data[data.length - 1];
        setCurrentPrice(lastCandle.close);
        setPriceChange(((lastCandle.close - data[0].open) / data[0].open) * 100);
    };

    // AI Pattern Detection
    useEffect(() => {
        if (!isAIActive) return;

        const detectPatterns = () => {
            const patterns = [
                'Ascending Triangle',
                'Bull Flag',
                'Cup and Handle',
                'Double Bottom',
            ];

            const detected = patterns.filter(() => Math.random() > 0.6);
            setDetectedPatterns(detected);

            // Generate signal if pattern detected
            if (detected.length > 0 && Math.random() > 0.7) {
                const newSignal: SignalData = {
                    id: Date.now().toString(),
                    timestamp: Math.floor(Date.now() / 1000) as Time,
                    type: Math.random() > 0.5 ? 'BUY' : 'SELL',
                    price: currentPrice,
                    pattern: detected[0],
                    confidence: Math.floor(Math.random() * 20) + 80,
                    status: 'active',
                };
                setSignalHistory(prev => [newSignal, ...prev.slice(0, 49)]);
            }
        };

        const interval = setInterval(detectPatterns, 10000);
        return () => clearInterval(interval);
    }, [isAIActive, currentPrice]);

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            {/* Header Bar */}
            <Box sx={{
                height: 48,
                display: 'flex',
                alignItems: 'center',
                px: 2,
                borderBottom: '1px solid #1e222d',
                backgroundColor: '#131722',
            }}>
                <Stack direction="row" spacing={2} alignItems="center" sx={{ flex: 1 }}>
                    {/* Symbol Search */}
                    <Autocomplete
                        value={selectedSymbol}
                        onChange={(event, newValue) => {
                            if (newValue) setSelectedSymbol(newValue);
                        }}
                        options={popularSymbols}
                        sx={{ width: 150 }}
                        renderInput={(params) => (
                            <TextField
                                {...params}
                                size="small"
                                placeholder="Symbol"
                                sx={{
                                    '& .MuiOutlinedInput-root': {
                                        color: '#d1d4dc',
                                        '& fieldset': {
                                            borderColor: '#1e222d',
                                        },
                                        '&:hover fieldset': {
                                            borderColor: '#787b86',
                                        },
                                    },
                                }}
                            />
                        )}
                    />

                    <Divider orientation="vertical" flexItem sx={{ borderColor: '#1e222d' }} />

                    {/* Timeframe Selector */}
                    <ToggleButtonGroup
                        value={selectedTimeframe}
                        exclusive
                        onChange={(e, value) => value && setSelectedTimeframe(value)}
                        size="small"
                        sx={{
                            '& .MuiToggleButton-root': {
                                color: '#787b86',
                                borderColor: '#1e222d',
                                '&.Mui-selected': {
                                    color: '#2962ff',
                                    backgroundColor: alpha('#2962ff', 0.1),
                                },
                            },
                        }}
                    >
                        {timeframes.map(tf => (
                            <ToggleButton key={tf.value} value={tf.value}>
                                {tf.label}
                            </ToggleButton>
                        ))}
                    </ToggleButtonGroup>

                    <Divider orientation="vertical" flexItem sx={{ borderColor: '#1e222d' }} />

                    {/* Technical Tools */}
                    <Stack direction="row" spacing={1}>
                        <Tooltip title="Trend Lines">
                            <IconButton
                                size="small"
                                onClick={() => setShowTrendLines(!showTrendLines)}
                                sx={{ color: showTrendLines ? '#2962ff' : '#787b86' }}
                            >
                                <ShowChart />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Fibonacci">
                            <IconButton
                                size="small"
                                onClick={() => setShowFibonacci(!showFibonacci)}
                                sx={{ color: showFibonacci ? '#2962ff' : '#787b86' }}
                            >
                                <Timeline />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Pattern Detection">
                            <IconButton
                                size="small"
                                onClick={() => setShowPatterns(!showPatterns)}
                                sx={{ color: showPatterns ? '#2962ff' : '#787b86' }}
                            >
                                <AutoGraph />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Volume">
                            <IconButton
                                size="small"
                                onClick={() => setShowVolume(!showVolume)}
                                sx={{ color: showVolume ? '#2962ff' : '#787b86' }}
                            >
                                <VolumeUp />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Grid">
                            <IconButton
                                size="small"
                                onClick={() => setShowGrid(!showGrid)}
                                sx={{ color: showGrid ? '#2962ff' : '#787b86' }}
                            >
                                <GridOn />
                            </IconButton>
                        </Tooltip>
                    </Stack>

                    <Box sx={{ flexGrow: 1 }} />

                    {/* AI Control */}
                    <Chip
                        icon={<Psychology />}
                        label={isAIActive ? "AI Active" : "AI Inactive"}
                        color={isAIActive ? "success" : "default"}
                        onClick={() => setIsAIActive(!isAIActive)}
                        size="small"
                        sx={{
                            backgroundColor: isAIActive ? alpha(theme.palette.success.main, 0.1) : 'transparent',
                            border: `1px solid ${isAIActive ? theme.palette.success.main : '#1e222d'}`,
                        }}
                    />

                    <IconButton size="small" sx={{ color: '#787b86' }}>
                        <Fullscreen />
                    </IconButton>
                </Stack>
            </Box>

            {/* Main Chart Area */}
            <Box sx={{ flex: 1, display: 'flex', position: 'relative' }}>
                {/* Chart Container */}
                <Box
                    ref={chartContainerRef}
                    sx={{
                        flex: 1,
                        position: 'relative',
                        backgroundColor: '#0a0e1a',
                    }}
                >
                    {/* Price Overlay */}
                    <Box sx={{
                        position: 'absolute',
                        top: 16,
                        left: 16,
                        zIndex: 1,
                        backgroundColor: alpha('#131722', 0.9),
                        p: 2,
                        borderRadius: 1,
                        border: '1px solid #1e222d',
                    }}>
                        <Typography variant="h5" sx={{ fontWeight: 700, mb: 0.5 }}>
                            {selectedSymbol}
                        </Typography>
                        <Typography variant="h4" sx={{
                            fontWeight: 700,
                            color: priceChange >= 0 ? '#26a69a' : '#ef5350',
                        }}>
                            ${currentPrice.toFixed(2)}
                        </Typography>
                        <Typography variant="body2" sx={{
                            color: priceChange >= 0 ? '#26a69a' : '#ef5350',
                        }}>
                            {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
                        </Typography>
                    </Box>

                    {/* AI Status Overlay */}
                    {isAIActive && detectedPatterns.length > 0 && (
                        <Box sx={{
                            position: 'absolute',
                            top: 16,
                            right: 16,
                            zIndex: 1,
                            backgroundColor: alpha('#131722', 0.9),
                            p: 2,
                            borderRadius: 1,
                            border: '1px solid #1e222d',
                            maxWidth: 250,
                        }}>
                            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                                <Psychology sx={{ fontSize: 16, color: '#2962ff' }} />
                                <Typography variant="caption" sx={{ color: '#787b86' }}>
                                    AI Analysis Active
                                </Typography>
                            </Stack>
                            <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                                Detected Patterns:
                            </Typography>
                            {detectedPatterns.map((pattern, index) => (
                                <Chip
                                    key={index}
                                    label={pattern}
                                    size="small"
                                    icon={<TrendingUp sx={{ fontSize: 14 }} />}
                                    sx={{
                                        mr: 0.5,
                                        mb: 0.5,
                                        backgroundColor: alpha('#26a69a', 0.1),
                                        border: '1px solid #26a69a',
                                        color: '#26a69a',
                                    }}
                                />
                            ))}
                        </Box>
                    )}
                </Box>

                {/* Signal History Sidebar */}
                <Paper sx={{
                    width: 300,
                    backgroundColor: '#131722',
                    borderLeft: '1px solid #1e222d',
                    display: 'flex',
                    flexDirection: 'column',
                    overflow: 'hidden',
                }}>
                    <Box sx={{ p: 2, borderBottom: '1px solid #1e222d' }}>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                            Signal History
                        </Typography>
                    </Box>
                    <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
                        <Stack spacing={1}>
                            {signalHistory.map((signal) => (
                                <Paper
                                    key={signal.id}
                                    sx={{
                                        p: 1.5,
                                        backgroundColor: '#0a0e1a',
                                        border: '1px solid #1e222d',
                                        cursor: 'pointer',
                                        transition: 'all 0.2s',
                                        '&:hover': {
                                            borderColor: '#2962ff',
                                            backgroundColor: alpha('#2962ff', 0.05),
                                        },
                                    }}
                                >
                                    <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                                        {signal.type === 'BUY' ? (
                                            <TrendingUp sx={{ fontSize: 16, color: '#26a69a' }} />
                                        ) : (
                                            <TrendingDown sx={{ fontSize: 16, color: '#ef5350' }} />
                                        )}
                                        <Typography variant="subtitle2" sx={{
                                            fontWeight: 600,
                                            color: signal.type === 'BUY' ? '#26a69a' : '#ef5350',
                                        }}>
                                            {signal.type} Signal
                                        </Typography>
                                        <Box sx={{ flexGrow: 1 }} />
                                        <Chip
                                            label={signal.status}
                                            size="small"
                                            icon={signal.status === 'success' ? <CheckCircle /> : signal.status === 'failed' ? <Cancel /> : <Schedule />}
                                            sx={{
                                                height: 20,
                                                '& .MuiChip-label': { px: 1 },
                                                backgroundColor:
                                                    signal.status === 'success' ? alpha('#26a69a', 0.1) :
                                                        signal.status === 'failed' ? alpha('#ef5350', 0.1) :
                                                            alpha('#ffa726', 0.1),
                                                color:
                                                    signal.status === 'success' ? '#26a69a' :
                                                        signal.status === 'failed' ? '#ef5350' :
                                                            '#ffa726',
                                            }}
                                        />
                                    </Stack>
                                    <Typography variant="body2" sx={{ color: '#d1d4dc' }}>
                                        ${signal.price.toFixed(2)}
                                    </Typography>
                                    <Typography variant="caption" sx={{ color: '#787b86' }}>
                                        {signal.pattern} • {signal.confidence}% confidence
                                    </Typography>
                                </Paper>
                            ))}
                        </Stack>
                    </Box>
                </Paper>
            </Box>

            {/* Status Bar */}
            <Box sx={{
                height: 32,
                display: 'flex',
                alignItems: 'center',
                px: 2,
                borderTop: '1px solid #1e222d',
                backgroundColor: '#131722',
                fontSize: 12,
                color: '#787b86',
            }}>
                <Stack direction="row" spacing={2} alignItems="center">
                    <Stack direction="row" spacing={1} alignItems="center">
                        <Box sx={{
                            width: 8,
                            height: 8,
                            borderRadius: '50%',
                            backgroundColor: '#26a69a',
                        }} />
                        <Typography variant="caption">Connected</Typography>
                    </Stack>
                    <Divider orientation="vertical" flexItem sx={{ borderColor: '#1e222d' }} />
                    <Typography variant="caption">{selectedTimeframe.toUpperCase()}</Typography>
                    <Divider orientation="vertical" flexItem sx={{ borderColor: '#1e222d' }} />
                    <Typography variant="caption">{selectedSymbol}</Typography>
                </Stack>
                <Box sx={{ flexGrow: 1 }} />
                <Typography variant="caption">
                    {new Date().toLocaleString()}
                </Typography>
            </Box>
        </Box>
    );
};

export default TrendSpiderChart; 