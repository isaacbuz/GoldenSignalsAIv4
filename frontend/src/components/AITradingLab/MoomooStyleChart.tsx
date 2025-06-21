import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, ColorType, CrosshairMode } from 'lightweight-charts';
import { Box, Paper, Typography, IconButton, ToggleButton, ToggleButtonGroup, Chip, Stack, Button, TextField, Autocomplete, LinearProgress, Tooltip } from '@mui/material';
import {
    TrendingUp,
    ShowChart,
    CandlestickChart,
    Timeline,
    Speed,
    Visibility,
    VisibilityOff,
    ZoomIn,
    ZoomOut,
    Fullscreen,
    Settings,
    Analytics,
    AutoGraph,
    Psychology as PsychologyIcon,
    TrendingDown,
    Warning,
    CheckCircle
} from '@mui/icons-material';
import { alpha } from '@mui/material/styles';

interface MoomooStyleChartProps {
    symbol?: string;
    onSymbolChange?: (symbol: string) => void;
}

// Moomoo-inspired color scheme
const moomooTheme = {
    background: '#0A0E1A',
    card: '#141823',
    border: '#1E2330',
    text: {
        primary: '#FFFFFF',
        secondary: '#8B92A9',
        muted: '#5A6376'
    },
    accent: {
        primary: '#FF6B3B', // Moomoo orange
        secondary: '#4285F4', // Blue
        success: '#00D37F',
        danger: '#FF4757',
        warning: '#FFA502'
    },
    chart: {
        grid: '#1A1E2E',
        crosshair: '#434651',
        volume: '#2D3142'
    }
};

// Popular symbols for quick access
const POPULAR_SYMBOLS = [
    'SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD', 'META',
    'GOOGL', 'AMZN', 'NFLX', 'JPM', 'BAC', 'XLF', 'IWM', 'DIA'
];

export default function MoomooStyleChart({ symbol = 'SPY', onSymbolChange }: MoomooStyleChartProps) {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
    const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

    const [chartType, setChartType] = useState<'candle' | 'line'>('candle');
    const [timeframe, setTimeframe] = useState('15m');
    const [indicators, setIndicators] = useState({
        ma: true,
        ema: true,
        volume: true,
        rsi: false,
        macd: false,
        bollinger: false
    });
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [currentSignal, setCurrentSignal] = useState<any>(null);
    const [selectedSymbol, setSelectedSymbol] = useState(symbol);

    // Initialize chart with moomoo styling
    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height: 600,
            layout: {
                background: { type: ColorType.Solid, color: moomooTheme.background },
                textColor: moomooTheme.text.secondary,
                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                fontSize: 12
            },
            grid: {
                vertLines: { color: moomooTheme.chart.grid, style: 1 },
                horzLines: { color: moomooTheme.chart.grid, style: 1 }
            },
            crosshair: {
                mode: CrosshairMode.Normal,
                vertLine: {
                    color: moomooTheme.chart.crosshair,
                    width: 1,
                    style: 2,
                    labelBackgroundColor: moomooTheme.card
                },
                horzLine: {
                    color: moomooTheme.chart.crosshair,
                    width: 1,
                    style: 2,
                    labelBackgroundColor: moomooTheme.card
                }
            },
            rightPriceScale: {
                borderColor: moomooTheme.border,
                scaleMargins: {
                    top: 0.1,
                    bottom: 0.2
                }
            },
            timeScale: {
                borderColor: moomooTheme.border,
                timeVisible: true,
                secondsVisible: false,
                tickMarkFormatter: (time: any) => {
                    const date = new Date(time * 1000);
                    const hours = date.getHours().toString().padStart(2, '0');
                    const minutes = date.getMinutes().toString().padStart(2, '0');
                    return `${hours}:${minutes}`;
                }
            }
        });

        // Add candlestick series
        const candlestickSeries = chart.addCandlestickSeries({
            upColor: moomooTheme.accent.success,
            downColor: moomooTheme.accent.danger,
            borderUpColor: moomooTheme.accent.success,
            borderDownColor: moomooTheme.accent.danger,
            wickUpColor: moomooTheme.accent.success,
            wickDownColor: moomooTheme.accent.danger
        });

        // Add volume series
        const volumeSeries = chart.addHistogramSeries({
            color: moomooTheme.chart.volume,
            priceFormat: {
                type: 'volume'
            },
            priceScaleId: '',
            scaleMargins: {
                top: 0.8,
                bottom: 0
            }
        });

        // Generate sample data
        const data = generateSampleData();
        candlestickSeries.setData(data);

        const volumeData = data.map(d => ({
            time: d.time,
            value: d.volume,
            color: d.close >= d.open ? alpha(moomooTheme.accent.success, 0.5) : alpha(moomooTheme.accent.danger, 0.5)
        }));
        volumeSeries.setData(volumeData);

        chartRef.current = chart;
        candlestickSeriesRef.current = candlestickSeries;
        volumeSeriesRef.current = volumeSeries;

        // Handle resize
        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({
                    width: chartContainerRef.current.clientWidth
                });
            }
        };
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, []);

    // Generate sample data
    const generateSampleData = () => {
        const data = [];
        const basePrice = 450;
        let lastClose = basePrice;
        const now = Math.floor(Date.now() / 1000);

        for (let i = 100; i >= 0; i--) {
            const time = now - i * 300; // 5-minute intervals
            const volatility = 0.02;
            const trend = Math.sin(i / 20) * 2;

            const open = lastClose;
            const change = (Math.random() - 0.5) * volatility * lastClose + trend * 0.1;
            const high = Math.max(open, open + Math.random() * volatility * lastClose);
            const low = Math.min(open, open - Math.random() * volatility * lastClose);
            const close = open + change;
            const volume = Math.floor(Math.random() * 1000000) + 500000;

            lastClose = close;

            data.push({
                time,
                open,
                high,
                low,
                close,
                volume
            });
        }

        return data;
    };

    // Handle AI Analysis
    const handleAIAnalysis = useCallback(async () => {
        setIsAnalyzing(true);

        // Simulate AI analysis
        setTimeout(() => {
            setCurrentSignal({
                type: 'BUY',
                confidence: 87,
                entry: 452.50,
                stopLoss: 448.20,
                takeProfit1: 456.80,
                takeProfit2: 461.40,
                indicators: {
                    rsi: { value: 42, signal: 'Oversold' },
                    macd: { value: 0.85, signal: 'Bullish Crossover' },
                    ma20: { value: 451.30, signal: 'Above Support' },
                    volume: { value: 'Above Average', signal: 'Strong Interest' }
                },
                reasoning: 'Strong bullish setup with RSI oversold bounce, MACD crossover, and volume confirmation. Price holding above 20MA support.'
            });

            // Add visual indicators to chart
            if (chartRef.current && candlestickSeriesRef.current) {
                // Add markers for entry, stop loss, and take profit
                const markers = [
                    {
                        time: Math.floor(Date.now() / 1000),
                        position: 'belowBar' as const,
                        color: moomooTheme.accent.success,
                        shape: 'arrowUp' as const,
                        text: 'BUY'
                    }
                ];
                candlestickSeriesRef.current.setMarkers(markers);
            }

            setIsAnalyzing(false);
        }, 2000);
    }, []);

    return (
        <Box sx={{
            bgcolor: moomooTheme.background,
            height: '100vh',
            display: 'flex',
            flexDirection: 'column',
            color: moomooTheme.text.primary
        }}>
            {/* Header Bar */}
            <Paper sx={{
                bgcolor: moomooTheme.card,
                borderRadius: 0,
                borderBottom: `1px solid ${moomooTheme.border}`,
                p: 1.5
            }}>
                <Stack direction="row" alignItems="center" spacing={2}>
                    {/* Symbol Search */}
                    <Autocomplete
                        value={selectedSymbol}
                        onChange={(event, newValue) => {
                            if (newValue) {
                                setSelectedSymbol(newValue);
                                onSymbolChange?.(newValue);
                            }
                        }}
                        options={POPULAR_SYMBOLS}
                        sx={{ width: 150 }}
                        renderInput={(params) => (
                            <TextField
                                {...params}
                                size="small"
                                placeholder="Symbol"
                                sx={{
                                    '& .MuiOutlinedInput-root': {
                                        bgcolor: moomooTheme.background,
                                        color: moomooTheme.text.primary,
                                        '& fieldset': {
                                            borderColor: moomooTheme.border
                                        }
                                    }
                                }}
                            />
                        )}
                    />

                    {/* Price Info */}
                    <Box>
                        <Typography variant="h6" sx={{ color: moomooTheme.accent.success, fontWeight: 600 }}>
                            $452.38
                        </Typography>
                        <Typography variant="caption" sx={{ color: moomooTheme.accent.success }}>
                            +2.45 (+0.54%)
                        </Typography>
                    </Box>

                    {/* Timeframe Selector */}
                    <ToggleButtonGroup
                        value={timeframe}
                        exclusive
                        onChange={(e, value) => value && setTimeframe(value)}
                        size="small"
                        sx={{
                            '& .MuiToggleButton-root': {
                                color: moomooTheme.text.secondary,
                                borderColor: moomooTheme.border,
                                '&.Mui-selected': {
                                    bgcolor: moomooTheme.accent.primary,
                                    color: 'white',
                                    '&:hover': {
                                        bgcolor: alpha(moomooTheme.accent.primary, 0.8)
                                    }
                                }
                            }
                        }}
                    >
                        <ToggleButton value="1m">1m</ToggleButton>
                        <ToggleButton value="5m">5m</ToggleButton>
                        <ToggleButton value="15m">15m</ToggleButton>
                        <ToggleButton value="1h">1h</ToggleButton>
                        <ToggleButton value="1d">1D</ToggleButton>
                    </ToggleButtonGroup>

                    {/* Chart Type */}
                    <ToggleButtonGroup
                        value={chartType}
                        exclusive
                        onChange={(e, value) => value && setChartType(value)}
                        size="small"
                    >
                        <ToggleButton value="candle">
                            <CandlestickChart fontSize="small" />
                        </ToggleButton>
                        <ToggleButton value="line">
                            <ShowChart fontSize="small" />
                        </ToggleButton>
                    </ToggleButtonGroup>

                    <Box sx={{ flexGrow: 1 }} />

                    {/* AI Analysis Button */}
                    <Button
                        variant="contained"
                        startIcon={<AutoGraph />}
                        onClick={handleAIAnalysis}
                        disabled={isAnalyzing}
                        sx={{
                            bgcolor: moomooTheme.accent.primary,
                            '&:hover': {
                                bgcolor: alpha(moomooTheme.accent.primary, 0.8)
                            }
                        }}
                    >
                        {isAnalyzing ? 'Analyzing...' : 'AI Analysis'}
                    </Button>

                    {/* Tools */}
                    <IconButton size="small" sx={{ color: moomooTheme.text.secondary }}>
                        <Settings />
                    </IconButton>
                    <IconButton size="small" sx={{ color: moomooTheme.text.secondary }}>
                        <Fullscreen />
                    </IconButton>
                </Stack>
            </Paper>

            {/* Main Content */}
            <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
                {/* Left Sidebar - Indicators */}
                <Paper sx={{
                    width: 200,
                    bgcolor: moomooTheme.card,
                    borderRadius: 0,
                    borderRight: `1px solid ${moomooTheme.border}`,
                    p: 2,
                    overflow: 'auto'
                }}>
                    <Typography variant="subtitle2" sx={{ mb: 2, color: moomooTheme.text.secondary }}>
                        Indicators
                    </Typography>
                    <Stack spacing={1}>
                        {Object.entries({
                            ma: 'Moving Average',
                            ema: 'EMA',
                            volume: 'Volume',
                            rsi: 'RSI',
                            macd: 'MACD',
                            bollinger: 'Bollinger Bands'
                        }).map(([key, label]) => (
                            <Box
                                key={key}
                                sx={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between',
                                    p: 1,
                                    borderRadius: 1,
                                    bgcolor: indicators[key as keyof typeof indicators] ? alpha(moomooTheme.accent.primary, 0.1) : 'transparent',
                                    cursor: 'pointer',
                                    '&:hover': {
                                        bgcolor: alpha(moomooTheme.accent.primary, 0.05)
                                    }
                                }}
                                onClick={() => setIndicators(prev => ({ ...prev, [key]: !prev[key as keyof typeof indicators] }))}
                            >
                                <Typography variant="body2">{label}</Typography>
                                <IconButton size="small">
                                    {indicators[key as keyof typeof indicators] ? <Visibility /> : <VisibilityOff />}
                                </IconButton>
                            </Box>
                        ))}
                    </Stack>
                </Paper>

                {/* Chart Area */}
                <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                    {isAnalyzing && <LinearProgress sx={{ bgcolor: moomooTheme.border }} />}

                    <Box ref={chartContainerRef} sx={{ flex: 1, bgcolor: moomooTheme.background }} />

                    {/* Signal Panel */}
                    {currentSignal && (
                        <Paper sx={{
                            bgcolor: moomooTheme.card,
                            borderTop: `1px solid ${moomooTheme.border}`,
                            p: 2
                        }}>
                            <Stack direction="row" spacing={3} alignItems="center">
                                <Chip
                                    icon={currentSignal.type === 'BUY' ? <TrendingUp /> : <TrendingDown />}
                                    label={`${currentSignal.type} Signal`}
                                    sx={{
                                        bgcolor: currentSignal.type === 'BUY' ? moomooTheme.accent.success : moomooTheme.accent.danger,
                                        color: 'white',
                                        fontWeight: 600
                                    }}
                                />

                                <Box>
                                    <Typography variant="caption" sx={{ color: moomooTheme.text.secondary }}>
                                        Confidence
                                    </Typography>
                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                        {currentSignal.confidence}%
                                    </Typography>
                                </Box>

                                <Box>
                                    <Typography variant="caption" sx={{ color: moomooTheme.text.secondary }}>
                                        Entry
                                    </Typography>
                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                        ${currentSignal.entry}
                                    </Typography>
                                </Box>

                                <Box>
                                    <Typography variant="caption" sx={{ color: moomooTheme.text.secondary }}>
                                        Stop Loss
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: moomooTheme.accent.danger, fontWeight: 600 }}>
                                        ${currentSignal.stopLoss}
                                    </Typography>
                                </Box>

                                <Box>
                                    <Typography variant="caption" sx={{ color: moomooTheme.text.secondary }}>
                                        Target 1
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: moomooTheme.accent.success, fontWeight: 600 }}>
                                        ${currentSignal.takeProfit1}
                                    </Typography>
                                </Box>

                                <Box>
                                    <Typography variant="caption" sx={{ color: moomooTheme.text.secondary }}>
                                        Target 2
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: moomooTheme.accent.success, fontWeight: 600 }}>
                                        ${currentSignal.takeProfit2}
                                    </Typography>
                                </Box>

                                <Box sx={{ flexGrow: 1 }} />

                                <Tooltip title={currentSignal.reasoning}>
                                    <IconButton size="small" sx={{ color: moomooTheme.accent.primary }}>
                                        <PsychologyIcon />
                                    </IconButton>
                                </Tooltip>
                            </Stack>
                        </Paper>
                    )}
                </Box>

                {/* Right Sidebar - Signal Details */}
                {currentSignal && (
                    <Paper sx={{
                        width: 300,
                        bgcolor: moomooTheme.card,
                        borderRadius: 0,
                        borderLeft: `1px solid ${moomooTheme.border}`,
                        p: 2,
                        overflow: 'auto'
                    }}>
                        <Typography variant="subtitle2" sx={{ mb: 2, color: moomooTheme.text.secondary }}>
                            Signal Analysis
                        </Typography>

                        <Stack spacing={2}>
                            {/* Indicators Status */}
                            <Box>
                                <Typography variant="caption" sx={{ color: moomooTheme.text.muted, mb: 1 }}>
                                    INDICATORS
                                </Typography>
                                {Object.entries(currentSignal.indicators).map(([key, data]: [string, any]) => (
                                    <Box key={key} sx={{
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        alignItems: 'center',
                                        py: 0.5
                                    }}>
                                        <Typography variant="body2" sx={{ textTransform: 'uppercase' }}>
                                            {key}
                                        </Typography>
                                        <Stack direction="row" spacing={1} alignItems="center">
                                            <Typography variant="body2" sx={{ color: moomooTheme.text.secondary }}>
                                                {data.value}
                                            </Typography>
                                            <Chip
                                                size="small"
                                                label={data.signal}
                                                sx={{
                                                    bgcolor: data.signal.includes('Bullish') || data.signal.includes('Above')
                                                        ? alpha(moomooTheme.accent.success, 0.2)
                                                        : alpha(moomooTheme.accent.warning, 0.2),
                                                    color: data.signal.includes('Bullish') || data.signal.includes('Above')
                                                        ? moomooTheme.accent.success
                                                        : moomooTheme.accent.warning,
                                                    fontSize: '0.7rem'
                                                }}
                                            />
                                        </Stack>
                                    </Box>
                                ))}
                            </Box>

                            {/* Risk/Reward */}
                            <Box>
                                <Typography variant="caption" sx={{ color: moomooTheme.text.muted }}>
                                    RISK/REWARD
                                </Typography>
                                <Box sx={{ mt: 1 }}>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                        <Typography variant="body2">Risk</Typography>
                                        <Typography variant="body2" sx={{ color: moomooTheme.accent.danger }}>
                                            -0.95%
                                        </Typography>
                                    </Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                        <Typography variant="body2">Reward T1</Typography>
                                        <Typography variant="body2" sx={{ color: moomooTheme.accent.success }}>
                                            +0.95%
                                        </Typography>
                                    </Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <Typography variant="body2">Reward T2</Typography>
                                        <Typography variant="body2" sx={{ color: moomooTheme.accent.success }}>
                                            +1.96%
                                        </Typography>
                                    </Box>
                                </Box>
                            </Box>

                            {/* AI Reasoning */}
                            <Box>
                                <Typography variant="caption" sx={{ color: moomooTheme.text.muted }}>
                                    AI REASONING
                                </Typography>
                                <Typography variant="body2" sx={{ mt: 1, color: moomooTheme.text.secondary }}>
                                    {currentSignal.reasoning}
                                </Typography>
                            </Box>

                            {/* Action Buttons */}
                            <Stack spacing={1} sx={{ mt: 3 }}>
                                <Button
                                    fullWidth
                                    variant="contained"
                                    sx={{
                                        bgcolor: moomooTheme.accent.success,
                                        '&:hover': {
                                            bgcolor: alpha(moomooTheme.accent.success, 0.8)
                                        }
                                    }}
                                >
                                    Execute Trade
                                </Button>
                                <Button
                                    fullWidth
                                    variant="outlined"
                                    sx={{
                                        borderColor: moomooTheme.border,
                                        color: moomooTheme.text.secondary
                                    }}
                                >
                                    Save Signal
                                </Button>
                            </Stack>
                        </Stack>
                    </Paper>
                )}
            </Box>
        </Box>
    );
} 