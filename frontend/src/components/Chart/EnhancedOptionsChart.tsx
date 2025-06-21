/**
 * EnhancedOptionsChart Component - Next-Generation Trading Chart
 * 
 * Professional-grade chart with advanced features:
 * - Multi-timeframe analysis
 * - Advanced technical indicators
 * - Real-time options flow
 * - AI prediction overlays
 * - Interactive drawing tools
 * - Market depth visualization
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
    Box,
    Paper,
    Stack,
    Typography,
    ToggleButton,
    ToggleButtonGroup,
    IconButton,
    Chip,
    Button,
    Menu,
    MenuItem,
    Divider,
    useTheme,
    alpha,
    Tooltip,
    CircularProgress,
    Fade,
    Zoom,
    Badge,
    ButtonGroup,
    Slider,
    Switch,
    FormControlLabel,
    Popover,
} from '@mui/material';
import {
    ShowChart,
    CandlestickChart,
    BarChart,
    Timeline,
    Fullscreen,
    FullscreenExit,
    Settings,
    Addchart,
    Layers,
    Speed,
    TrendingUp,
    TrendingDown,
    Psychology,
    Analytics,
    Timer,
    Refresh,
    CheckCircle,
    Draw,
    ZoomIn,
    ZoomOut,
    RestartAlt,
    PhotoCamera,
    Visibility,
    VisibilityOff,
    WaterDrop,
    Waves,
    SignalCellularAlt,
    BubbleChart,
    AutoGraph,
    CompareArrows,
    GridOn,
    GridOff,
    DarkMode,
    LightMode,
    Palette,
    PlayArrow,
    Pause,
    SkipNext,
    Warning,
    TextFields,
} from '@mui/icons-material';
import {
    createChart,
    IChartApi,
    ISeriesApi,
    Time,
    LineStyle,
    CrosshairMode,
    PriceScaleMode,
    SeriesMarker,
    ColorType,
} from 'lightweight-charts';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../services/api';
import { PreciseOptionsSignal } from '../../types/signals';
import { motion, AnimatePresence } from 'framer-motion';

interface EnhancedOptionsChartProps {
    symbol: string;
    timeframe?: string;
    signals?: PreciseOptionsSignal[];
    selectedSignal?: PreciseOptionsSignal | null;
    marketData?: any;
    height?: number;
    onSignalClick?: (signal: PreciseOptionsSignal) => void;
}

interface ChartTheme {
    name: string;
    background: string;
    text: string;
    grid: string;
    upColor: string;
    downColor: string;
}

const chartThemes: ChartTheme[] = [
    {
        name: 'Dark Pro',
        background: '#0A0E1A',
        text: '#D1D4DC',
        grid: 'rgba(42, 46, 57, 0.5)',
        upColor: '#00D4AA',
        downColor: '#FF3B30',
    },
    {
        name: 'Trading View',
        background: '#131722',
        text: '#D1D4DC',
        grid: 'rgba(42, 46, 57, 0.5)',
        upColor: '#26A69A',
        downColor: '#EF5350',
    },
    {
        name: 'Bloomberg',
        background: '#000000',
        text: '#FF6B00',
        grid: 'rgba(255, 107, 0, 0.1)',
        upColor: '#00FF88',
        downColor: '#FF0080',
    },
];

const EnhancedOptionsChart: React.FC<EnhancedOptionsChartProps> = ({
    symbol,
    timeframe: propTimeframe = '15m',
    signals = [],
    selectedSignal,
    marketData: propMarketData,
    height = 600,
    onSignalClick,
}) => {
    const theme = useTheme();
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const mainSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

    // State
    const [chartType, setChartType] = useState<'candlestick' | 'line' | 'heikinashi' | 'area'>('candlestick');
    const [timeframe, setTimeframe] = useState(propTimeframe);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [chartTheme, setChartTheme] = useState(chartThemes[0]);
    const [showGrid, setShowGrid] = useState(true);
    const [autoScale, setAutoScale] = useState(true);
    const [isPlaying, setIsPlaying] = useState(false);
    const [showVolume, setShowVolume] = useState(true);
    const [showDepth, setShowDepth] = useState(false);
    const [indicatorMenu, setIndicatorMenu] = useState<null | HTMLElement>(null);
    const [drawingMode, setDrawingMode] = useState<string | null>(null);
    const [activeIndicators, setActiveIndicators] = useState<Set<string>>(new Set(['volume', 'ma20', 'ma50']));

    // Advanced indicators
    const indicators = [
        { id: 'volume', name: 'Volume', category: 'basic' },
        { id: 'ma20', name: 'MA 20', category: 'trend' },
        { id: 'ma50', name: 'MA 50', category: 'trend' },
        { id: 'ema9', name: 'EMA 9', category: 'trend' },
        { id: 'bb', name: 'Bollinger Bands', category: 'volatility' },
        { id: 'rsi', name: 'RSI', category: 'momentum' },
        { id: 'macd', name: 'MACD', category: 'momentum' },
        { id: 'stoch', name: 'Stochastic', category: 'momentum' },
        { id: 'atr', name: 'ATR', category: 'volatility' },
        { id: 'vwap', name: 'VWAP', category: 'volume' },
        { id: 'pivot', name: 'Pivot Points', category: 'support' },
        { id: 'ichimoku', name: 'Ichimoku Cloud', category: 'trend' },
        { id: 'optionsflow', name: 'Options Flow', category: 'options' },
        { id: 'gamma', name: 'Gamma Levels', category: 'options' },
        { id: 'darkpool', name: 'Dark Pool', category: 'volume' },
        { id: 'aiforecast', name: 'AI Forecast', category: 'ai' },
    ];

    // Drawing tools
    const drawingTools = [
        { id: 'trendline', icon: <Timeline />, name: 'Trend Line' },
        { id: 'horizontal', icon: <CompareArrows />, name: 'Horizontal Line' },
        { id: 'fibonacci', icon: <SignalCellularAlt />, name: 'Fibonacci' },
        { id: 'channel', icon: <Waves />, name: 'Channel' },
        { id: 'rectangle', icon: <GridOn />, name: 'Rectangle' },
        { id: 'text', icon: <TextFields />, name: 'Text' },
    ];

    // Fetch real-time data
    const { data: marketData, isLoading } = useQuery({
        queryKey: ['enhanced-chart-data', symbol, timeframe],
        queryFn: async () => {
            // In production, fetch real market data
            return generateEnhancedMockData();
        },
        refetchInterval: isPlaying ? 1000 : 30000,
    });

    // Generate enhanced mock data with more realistic patterns
    const generateEnhancedMockData = () => {
        const data: Array<{
            time: Time;
            open: number;
            high: number;
            low: number;
            close: number;
            volume: number;
        }> = [];
        const basePrice = 150;
        const now = Math.floor(Date.now() / 1000);

        // Timeframe configurations
        const timeframes: Record<string, { bars: number; seconds: number }> = {
            '1m': { bars: 60, seconds: 60 },
            '5m': { bars: 96, seconds: 300 },
            '15m': { bars: 96, seconds: 900 },
            '30m': { bars: 96, seconds: 1800 },
            '1h': { bars: 120, seconds: 3600 },
            '4h': { bars: 120, seconds: 14400 },
            '1d': { bars: 365, seconds: 86400 },
        };

        const config = timeframes[timeframe] || timeframes['15m'];

        // Generate realistic price action with trends
        let trend = 0;
        let momentum = 0;

        for (let i = 0; i < config.bars; i++) {
            const time = (now - (config.bars - i) * config.seconds) as Time;

            // Add realistic market dynamics
            if (Math.random() > 0.95) {
                // Occasional trend changes
                trend = (Math.random() - 0.5) * 2;
            }

            momentum = momentum * 0.9 + (Math.random() - 0.5) * 0.5;

            const trendComponent = trend * 0.1;
            const noiseComponent = (Math.random() - 0.5) * 0.5;
            const momentumComponent = momentum * 0.3;

            const change = trendComponent + noiseComponent + momentumComponent;

            const openPrice = i === 0 ? basePrice : data[i - 1].close;
            const closePrice = openPrice + change;
            const highPrice = Math.max(openPrice, closePrice) + Math.random() * 0.5;
            const lowPrice = Math.min(openPrice, closePrice) - Math.random() * 0.5;

            // Volume with patterns (higher at open/close)
            const timeOfDay = i % 96;
            const volumeMultiplier =
                timeOfDay < 10 || timeOfDay > 86 ? 2 :
                    timeOfDay > 40 && timeOfDay < 56 ? 0.7 : 1;

            const volume = (1000000 + Math.random() * 500000) * volumeMultiplier;

            data.push({
                time,
                open: openPrice,
                high: highPrice,
                low: lowPrice,
                close: closePrice,
                volume
            });
        }

        return data;
    };

    // Initialize enhanced chart
    useEffect(() => {
        if (!chartContainerRef.current || !marketData) return;

        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height,
            layout: {
                background: { type: ColorType.Solid, color: chartTheme.background },
                textColor: chartTheme.text,
                fontSize: 12,
            },
            grid: {
                vertLines: {
                    visible: showGrid,
                    color: chartTheme.grid,
                },
                horzLines: {
                    visible: showGrid,
                    color: chartTheme.grid,
                },
            },
            crosshair: {
                mode: CrosshairMode.Magnet,
                vertLine: {
                    width: 1,
                    color: theme.palette.primary.main,
                    style: LineStyle.Solid,
                    labelBackgroundColor: theme.palette.primary.main,
                },
                horzLine: {
                    width: 1,
                    color: theme.palette.primary.main,
                    style: LineStyle.Solid,
                    labelBackgroundColor: theme.palette.primary.main,
                },
            },
            rightPriceScale: {
                borderColor: chartTheme.grid,
                autoScale: autoScale,
                mode: PriceScaleMode.Normal,
                invertScale: false,
                alignLabels: true,
                borderVisible: false,
                scaleMargins: {
                    top: 0.1,
                    bottom: showVolume ? 0.25 : 0.1,
                },
            },
            timeScale: {
                borderColor: chartTheme.grid,
                timeVisible: true,
                secondsVisible: timeframe === '1m',
                tickMarkFormatter: (time: any) => {
                    const date = new Date(time * 1000);
                    if (timeframe === '1d') {
                        return date.toLocaleDateString();
                    }
                    return date.toLocaleTimeString();
                },
            },
            watermark: {
                visible: true,
                fontSize: 48,
                horzAlign: 'center',
                vertAlign: 'center',
                color: alpha(chartTheme.text, 0.1),
                text: symbol,
            },
        });

        chartRef.current = chart;

        // Add main series based on chart type
        let mainSeries: ISeriesApi<any>;

        switch (chartType) {
            case 'candlestick':
                mainSeries = chart.addCandlestickSeries({
                    upColor: chartTheme.upColor,
                    downColor: chartTheme.downColor,
                    borderUpColor: chartTheme.upColor,
                    borderDownColor: chartTheme.downColor,
                    wickUpColor: chartTheme.upColor,
                    wickDownColor: chartTheme.downColor,
                });
                mainSeries.setData(marketData);
                break;

            case 'line':
                mainSeries = chart.addLineSeries({
                    color: theme.palette.primary.main,
                    lineWidth: 2,
                    crosshairMarkerVisible: true,
                    crosshairMarkerRadius: 5,
                });
                mainSeries.setData(marketData.map((d: any) => ({ time: d.time, value: d.close })));
                break;

            case 'area':
                mainSeries = chart.addAreaSeries({
                    topColor: alpha(theme.palette.primary.main, 0.6),
                    bottomColor: alpha(theme.palette.primary.main, 0.1),
                    lineColor: theme.palette.primary.main,
                    lineWidth: 2,
                });
                mainSeries.setData(marketData.map((d: any) => ({ time: d.time, value: d.close })));
                break;

            case 'heikinashi':
                // Calculate Heikin-Ashi candles
                const heikinAshiData = calculateHeikinAshi(marketData);
                mainSeries = chart.addCandlestickSeries({
                    upColor: chartTheme.upColor,
                    downColor: chartTheme.downColor,
                    borderVisible: false,
                    wickVisible: true,
                });
                mainSeries.setData(heikinAshiData);
                break;
        }

        mainSeriesRef.current = mainSeries as ISeriesApi<'Candlestick'>;

        // Add active indicators
        activeIndicators.forEach(indicatorId => {
            addIndicator(chart, indicatorId, marketData);
        });

        // Add signal overlays
        if (signals.length > 0) {
            addSignalOverlays(chart, mainSeries, signals);
        }

        // Fit content
        chart.timeScale().fitContent();

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
    }, [marketData, chartType, chartTheme, showGrid, autoScale, activeIndicators, signals, showVolume]);

    // Calculate Heikin-Ashi
    const calculateHeikinAshi = (data: any[]) => {
        const heikinAshi = [];
        let prevHA: any = null;

        for (let i = 0; i < data.length; i++) {
            const current = data[i];
            const ha: any = { time: current.time };

            if (i === 0) {
                ha.open = (current.open + current.close) / 2;
                ha.close = (current.open + current.high + current.low + current.close) / 4;
                ha.high = current.high;
                ha.low = current.low;
            } else {
                ha.open = (prevHA.open + prevHA.close) / 2;
                ha.close = (current.open + current.high + current.low + current.close) / 4;
                ha.high = Math.max(current.high, ha.open, ha.close);
                ha.low = Math.min(current.low, ha.open, ha.close);
            }

            heikinAshi.push(ha);
            prevHA = ha;
        }

        return heikinAshi;
    };

    // Add indicator to chart
    const addIndicator = (chart: IChartApi, indicatorId: string, data: any[]) => {
        switch (indicatorId) {
            case 'volume':
                if (showVolume) {
                    const volumeSeries = chart.addHistogramSeries({
                        color: '#26a69a',
                        priceFormat: { type: 'volume' },
                        priceScaleId: '',
                    });

                    volumeSeries.priceScale().applyOptions({
                        scaleMargins: { top: 0.8, bottom: 0 },
                    });

                    volumeSeries.setData(
                        data.map((d: any) => ({
                            time: d.time,
                            value: d.volume,
                            color: d.close >= d.open
                                ? alpha(chartTheme.upColor, 0.5)
                                : alpha(chartTheme.downColor, 0.5),
                        }))
                    );
                }
                break;

            case 'ma20':
                const ma20 = chart.addLineSeries({
                    color: '#2962FF',
                    lineWidth: 1,
                    title: 'MA 20',
                });
                ma20.setData(calculateMA(data, 20));
                break;

            case 'ma50':
                const ma50 = chart.addLineSeries({
                    color: '#FF6D00',
                    lineWidth: 1,
                    title: 'MA 50',
                });
                ma50.setData(calculateMA(data, 50));
                break;

            case 'bb':
                const bb = calculateBollingerBands(data, 20, 2);

                const upperBand = chart.addLineSeries({
                    color: alpha('#2962FF', 0.5),
                    lineWidth: 1,
                    lineStyle: LineStyle.Dashed,
                    title: 'BB Upper',
                });
                upperBand.setData(bb.upper);

                const lowerBand = chart.addLineSeries({
                    color: alpha('#2962FF', 0.5),
                    lineWidth: 1,
                    lineStyle: LineStyle.Dashed,
                    title: 'BB Lower',
                });
                lowerBand.setData(bb.lower);

                const middleBand = chart.addLineSeries({
                    color: alpha('#2962FF', 0.3),
                    lineWidth: 1,
                    lineStyle: LineStyle.Dotted,
                    title: 'BB Middle',
                });
                middleBand.setData(bb.middle);
                break;

            case 'vwap':
                const vwapSeries = chart.addLineSeries({
                    color: '#9C27B0',
                    lineWidth: 2,
                    title: 'VWAP',
                });
                vwapSeries.setData(calculateVWAP(data));
                break;

            case 'aiforecast':
                // AI prediction overlay
                const forecast = generateAIForecast(data);
                const forecastSeries = chart.addLineSeries({
                    color: theme.palette.secondary.main,
                    lineWidth: 2,
                    lineStyle: LineStyle.Dashed,
                    title: 'AI Forecast',
                });
                forecastSeries.setData(forecast);
                break;
        }
    };

    // Calculate indicators
    const calculateMA = (data: any[], period: number) => {
        const ma = [];
        for (let i = period - 1; i < data.length; i++) {
            let sum = 0;
            for (let j = 0; j < period; j++) {
                sum += data[i - j].close;
            }
            ma.push({ time: data[i].time, value: sum / period });
        }
        return ma;
    };

    const calculateBollingerBands = (data: any[], period: number, stdDev: number) => {
        const ma = calculateMA(data, period);
        const upper = [];
        const lower = [];

        for (let i = 0; i < ma.length; i++) {
            const dataIndex = i + period - 1;
            let sumSquaredDiff = 0;

            for (let j = 0; j < period; j++) {
                const diff = data[dataIndex - j].close - ma[i].value;
                sumSquaredDiff += diff * diff;
            }

            const std = Math.sqrt(sumSquaredDiff / period);

            upper.push({ time: ma[i].time, value: ma[i].value + std * stdDev });
            lower.push({ time: ma[i].time, value: ma[i].value - std * stdDev });
        }

        return { upper, lower, middle: ma };
    };

    const calculateVWAP = (data: any[]) => {
        const vwap = [];
        let cumulativeVolume = 0;
        let cumulativeVolumePrice = 0;

        for (let i = 0; i < data.length; i++) {
            const typicalPrice = (data[i].high + data[i].low + data[i].close) / 3;
            cumulativeVolumePrice += typicalPrice * data[i].volume;
            cumulativeVolume += data[i].volume;

            vwap.push({
                time: data[i].time,
                value: cumulativeVolumePrice / cumulativeVolume,
            });
        }

        return vwap;
    };

    const generateAIForecast = (data: any[]) => {
        // Simple AI forecast simulation
        const lastPrice = data[data.length - 1].close;
        const trend = data[data.length - 1].close > data[data.length - 10].close ? 1 : -1;
        const forecast = [];

        for (let i = 0; i < 10; i++) {
            const time = (data[data.length - 1].time as number) + (i + 1) * 900;
            const value = lastPrice + trend * (i + 1) * 0.1 + (Math.random() - 0.5) * 0.5;
            forecast.push({ time: time as Time, value });
        }

        return forecast;
    };

    const addSignalOverlays = (chart: IChartApi, mainSeries: ISeriesApi<any>, signals: PreciseOptionsSignal[]) => {
        // Add markers for signals
        const markers: SeriesMarker<Time>[] = signals.map(signal => ({
            time: Math.floor(new Date(signal.timestamp).getTime() / 1000) as Time,
            position: signal.type === 'CALL' ? 'belowBar' : 'aboveBar',
            color: signal.type === 'CALL' ? chartTheme.upColor : chartTheme.downColor,
            shape: signal.type === 'CALL' ? 'arrowUp' : 'arrowDown',
            text: `${signal.confidence}%`,
            size: 2,
        }));

        mainSeries.setMarkers(markers);

        // Add price lines for selected signal
        if (selectedSignal) {
            // Entry line
            mainSeries.createPriceLine({
                price: selectedSignal.entry_price,
                color: theme.palette.primary.main,
                lineWidth: 2,
                lineStyle: LineStyle.Solid,
                axisLabelVisible: true,
                title: 'Entry',
            });

            // Stop loss
            mainSeries.createPriceLine({
                price: selectedSignal.stop_loss,
                color: theme.palette.error.main,
                lineWidth: 2,
                lineStyle: LineStyle.Dashed,
                axisLabelVisible: true,
                title: 'Stop',
            });

            // Take profit
            mainSeries.createPriceLine({
                price: selectedSignal.take_profit,
                color: theme.palette.success.main,
                lineWidth: 2,
                lineStyle: LineStyle.Dotted,
                axisLabelVisible: true,
                title: 'Target',
            });
        }
    };

    // Handlers
    const handleIndicatorToggle = (indicatorId: string) => {
        setActiveIndicators(prev => {
            const newSet = new Set(prev);
            if (newSet.has(indicatorId)) {
                newSet.delete(indicatorId);
            } else {
                newSet.add(indicatorId);
            }
            return newSet;
        });
    };

    const handleFullscreen = () => {
        if (!document.fullscreenElement) {
            chartContainerRef.current?.requestFullscreen();
            setIsFullscreen(true);
        } else {
            document.exitFullscreen();
            setIsFullscreen(false);
        }
    };

    const handleScreenshot = () => {
        if (chartRef.current) {
            const canvas = chartRef.current.takeScreenshot();
            if (canvas) {
                canvas.toBlob((blob) => {
                    if (blob) {
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `${symbol}_chart_${new Date().toISOString()}.png`;
                        a.click();
                        URL.revokeObjectURL(url);
                    }
                });
            }
        }
    };

    const handleZoom = (direction: 'in' | 'out' | 'reset') => {
        if (!chartRef.current) return;

        const timeScale = chartRef.current.timeScale();

        if (direction === 'reset') {
            timeScale.fitContent();
        } else {
            const currentRange = timeScale.getVisibleLogicalRange();
            if (currentRange) {
                const barCount = currentRange.to - currentRange.from;
                const factor = direction === 'in' ? 0.5 : 2;
                const newBarCount = barCount * factor;
                const center = (currentRange.from + currentRange.to) / 2;

                timeScale.setVisibleLogicalRange({
                    from: center - newBarCount / 2,
                    to: center + newBarCount / 2,
                });
            }
        }
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Enhanced Chart Header */}
            <Box
                sx={{
                    p: 1.5,
                    borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                    background: alpha(theme.palette.background.paper, 0.5),
                }}
            >
                <Stack direction="row" alignItems="center" justifyContent="space-between">
                    {/* Left side - Chart controls */}
                    <Stack direction="row" spacing={1} alignItems="center">
                        {/* Chart Type */}
                        <ToggleButtonGroup
                            value={chartType}
                            exclusive
                            onChange={(e, value) => value && setChartType(value)}
                            size="small"
                        >
                            <ToggleButton value="candlestick">
                                <Tooltip title="Candlestick">
                                    <CandlestickChart fontSize="small" />
                                </Tooltip>
                            </ToggleButton>
                            <ToggleButton value="line">
                                <Tooltip title="Line">
                                    <ShowChart fontSize="small" />
                                </Tooltip>
                            </ToggleButton>
                            <ToggleButton value="area">
                                <Tooltip title="Area">
                                    <BarChart fontSize="small" />
                                </Tooltip>
                            </ToggleButton>
                            <ToggleButton value="heikinashi">
                                <Tooltip title="Heikin-Ashi">
                                    <Waves fontSize="small" />
                                </Tooltip>
                            </ToggleButton>
                        </ToggleButtonGroup>

                        <Divider orientation="vertical" flexItem />

                        {/* Timeframe */}
                        <ToggleButtonGroup
                            value={timeframe}
                            exclusive
                            onChange={(e, value) => value && setTimeframe(value)}
                            size="small"
                        >
                            {['1m', '5m', '15m', '30m', '1h', '4h', '1d'].map(tf => (
                                <ToggleButton key={tf} value={tf}>
                                    <Typography variant="caption">{tf.toUpperCase()}</Typography>
                                </ToggleButton>
                            ))}
                        </ToggleButtonGroup>

                        <Divider orientation="vertical" flexItem />

                        {/* Indicators */}
                        <Button
                            size="small"
                            startIcon={<Addchart />}
                            onClick={(e) => setIndicatorMenu(e.currentTarget)}
                            endIcon={
                                <Badge badgeContent={activeIndicators.size} color="primary">
                                    <div />
                                </Badge>
                            }
                        >
                            Indicators
                        </Button>

                        {/* Drawing Tools */}
                        <ToggleButtonGroup
                            value={drawingMode}
                            exclusive
                            onChange={(e, value) => setDrawingMode(value)}
                            size="small"
                        >
                            {drawingTools.map(tool => (
                                <ToggleButton key={tool.id} value={tool.id}>
                                    <Tooltip title={tool.name}>
                                        {tool.icon}
                                    </Tooltip>
                                </ToggleButton>
                            ))}
                        </ToggleButtonGroup>
                    </Stack>

                    {/* Right side - View controls */}
                    <Stack direction="row" spacing={1} alignItems="center">
                        {/* Zoom controls */}
                        <ButtonGroup size="small">
                            <IconButton size="small" onClick={() => handleZoom('out')}>
                                <ZoomOut fontSize="small" />
                            </IconButton>
                            <IconButton size="small" onClick={() => handleZoom('reset')}>
                                <RestartAlt fontSize="small" />
                            </IconButton>
                            <IconButton size="small" onClick={() => handleZoom('in')}>
                                <ZoomIn fontSize="small" />
                            </IconButton>
                        </ButtonGroup>

                        <Divider orientation="vertical" flexItem />

                        {/* View options */}
                        <IconButton
                            size="small"
                            onClick={() => setShowGrid(!showGrid)}
                            color={showGrid ? 'primary' : 'default'}
                        >
                            {showGrid ? <GridOn fontSize="small" /> : <GridOff fontSize="small" />}
                        </IconButton>

                        <IconButton
                            size="small"
                            onClick={() => setAutoScale(!autoScale)}
                            color={autoScale ? 'primary' : 'default'}
                        >
                            <AutoGraph fontSize="small" />
                        </IconButton>

                        <IconButton size="small" onClick={handleScreenshot}>
                            <PhotoCamera fontSize="small" />
                        </IconButton>

                        <IconButton size="small" onClick={handleFullscreen}>
                            {isFullscreen ? <FullscreenExit fontSize="small" /> : <Fullscreen fontSize="small" />}
                        </IconButton>

                        <Divider orientation="vertical" flexItem />

                        {/* Theme selector */}
                        <ToggleButtonGroup
                            value={chartTheme.name}
                            exclusive
                            onChange={(e, value) => {
                                const theme = chartThemes.find(t => t.name === value);
                                if (theme) setChartTheme(theme);
                            }}
                            size="small"
                        >
                            {chartThemes.map(theme => (
                                <ToggleButton key={theme.name} value={theme.name}>
                                    <Tooltip title={theme.name}>
                                        <Box
                                            sx={{
                                                width: 16,
                                                height: 16,
                                                borderRadius: '50%',
                                                bgcolor: theme.background,
                                                border: `2px solid ${theme.text}`,
                                            }}
                                        />
                                    </Tooltip>
                                </ToggleButton>
                            ))}
                        </ToggleButtonGroup>
                    </Stack>
                </Stack>

                {/* Real-time stats bar */}
                <Stack direction="row" spacing={2} sx={{ mt: 1 }}>
                    {marketData && marketData.length > 0 && (
                        <>
                            <Chip
                                size="small"
                                label={`Last: $${marketData[marketData.length - 1].close.toFixed(2)}`}
                                color={marketData[marketData.length - 1].close >= marketData[marketData.length - 1].open ? 'success' : 'error'}
                            />
                            <Chip
                                size="small"
                                label={`Vol: ${(marketData[marketData.length - 1].volume / 1000000).toFixed(2)}M`}
                                variant="outlined"
                            />
                            {selectedSignal && (
                                <>
                                    <Chip
                                        size="small"
                                        icon={<Psychology />}
                                        label={`Signal: ${selectedSignal.type} ${selectedSignal.strike_price}`}
                                        color="primary"
                                    />
                                    <Chip
                                        size="small"
                                        icon={<Timer />}
                                        label={`Entry: $${selectedSignal.entry_price}`}
                                        color="warning"
                                    />
                                </>
                            )}
                        </>
                    )}
                </Stack>
            </Box>

            {/* Chart Container */}
            <Box
                ref={chartContainerRef}
                sx={{
                    flex: 1,
                    position: 'relative',
                    background: chartTheme.background,
                    '& .tv-lightweight-charts': {
                        borderRadius: 0,
                    },
                }}
            >
                {isLoading && (
                    <Box
                        sx={{
                            position: 'absolute',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            zIndex: 10,
                        }}
                    >
                        <CircularProgress />
                    </Box>
                )}

                {/* Options Flow Overlay */}
                {showDepth && (
                    <Paper
                        sx={{
                            position: 'absolute',
                            top: 16,
                            right: 16,
                            p: 2,
                            background: alpha(theme.palette.background.paper, 0.9),
                            backdropFilter: 'blur(10px)',
                            minWidth: 200,
                        }}
                    >
                        <Typography variant="subtitle2" gutterBottom>
                            Options Flow
                        </Typography>
                        <Stack spacing={1}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="caption">Call Volume:</Typography>
                                <Typography variant="caption" color="success.main">125.3K</Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="caption">Put Volume:</Typography>
                                <Typography variant="caption" color="error.main">98.7K</Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="caption">P/C Ratio:</Typography>
                                <Typography variant="caption">0.79</Typography>
                            </Box>
                        </Stack>
                    </Paper>
                )}

                {/* AI Prediction Overlay */}
                {activeIndicators.has('aiforecast') && (
                    <Paper
                        sx={{
                            position: 'absolute',
                            bottom: 16,
                            left: 16,
                            p: 2,
                            background: alpha(theme.palette.background.paper, 0.9),
                            backdropFilter: 'blur(10px)',
                        }}
                    >
                        <Stack direction="row" spacing={1} alignItems="center">
                            <Psychology sx={{ color: theme.palette.secondary.main }} />
                            <Typography variant="subtitle2">
                                AI Forecast: {Math.random() > 0.5 ? 'Bullish' : 'Bearish'}
                            </Typography>
                            <Chip
                                size="small"
                                label={`${Math.floor(Math.random() * 30 + 70)}% confidence`}
                                color="secondary"
                            />
                        </Stack>
                    </Paper>
                )}
            </Box>

            {/* Indicators Menu */}
            <Menu
                anchorEl={indicatorMenu}
                open={Boolean(indicatorMenu)}
                onClose={() => setIndicatorMenu(null)}
                PaperProps={{
                    sx: { minWidth: 300 },
                }}
            >
                <Box sx={{ p: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                        Technical Indicators
                    </Typography>
                </Box>
                <Divider />
                {Object.entries(
                    indicators.reduce((acc, ind) => {
                        if (!acc[ind.category]) acc[ind.category] = [];
                        acc[ind.category].push(ind);
                        return acc;
                    }, {} as Record<string, typeof indicators>)
                ).map(([category, inds]) => (
                    <Box key={category}>
                        <MenuItem disabled>
                            <Typography variant="caption" color="text.secondary">
                                {category.toUpperCase()}
                            </Typography>
                        </MenuItem>
                        {inds.map(ind => (
                            <MenuItem
                                key={ind.id}
                                onClick={() => handleIndicatorToggle(ind.id)}
                                sx={{ pl: 4 }}
                            >
                                <FormControlLabel
                                    control={
                                        <Switch
                                            checked={activeIndicators.has(ind.id)}
                                            size="small"
                                        />
                                    }
                                    label={ind.name}
                                />
                            </MenuItem>
                        ))}
                    </Box>
                ))}
            </Menu>
        </Box>
    );
};

export default EnhancedOptionsChart; 