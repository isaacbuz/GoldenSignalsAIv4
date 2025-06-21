import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Box,
    Paper,
    Stack,
    IconButton,
    Tooltip,
    Button,
    Chip,
    Typography,
    ToggleButton,
    ToggleButtonGroup,
    Divider,
    FormControl,
    Select,
    MenuItem,
    Switch,
    FormControlLabel,
    Collapse,
    LinearProgress,
    Alert,
    Fade,
    alpha,
} from '@mui/material';
import {
    AutoAwesome as AIIcon,
    ShowChart as TrendLineIcon,
    Timeline as TimelineIcon,
    Layers as LayersIcon,
    Psychology as PsychologyIcon,
    PlayArrow as PlayIcon,
    Pause as PauseIcon,
    Settings as SettingsIcon,
    Refresh as RefreshIcon,
    ZoomIn as ZoomInIcon,
    ZoomOut as ZoomOutIcon,
    Fullscreen as FullscreenIcon,
    FullscreenExit as FullscreenExitIcon,
    TrendingUp,
    TrendingDown,
    Analytics as AnalyticsIcon,
    Speed as SpeedIcon,
    Assessment as PatternIcon,
    Security as ShieldIcon,
    Insights,
    CandlestickChart as CandleIcon,
    AreaChart,
    ShowChartOutlined as FibIcon,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
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
import { motion, AnimatePresence } from 'framer-motion';

interface EnhancedTradingChartProps {
    symbol: string;
    height?: number;
    onSymbolChange?: (symbol: string) => void;
    onNewSignal?: (signal: any) => void;
    showAIFeatures?: boolean;
}

interface AISignal {
    id: string;
    type: 'BUY' | 'SELL';
    entry: number;
    stopLoss: number;
    takeProfit: number[];
    confidence: number;
    patterns: string[];
    indicators: string[];
    confluenceScore: number;
    reasoning: string;
    timestamp: Time;
}

interface AIDrawing {
    id: string;
    type: 'trendline' | 'support' | 'resistance' | 'pattern' | 'fibonacci' | 'entry' | 'exit';
    points: { time: Time; price: number }[];
    label: string;
    confidence: number;
    color: string;
}

const EnhancedTradingChart: React.FC<EnhancedTradingChartProps> = ({
    symbol,
    height = 600,
    onSymbolChange,
    onNewSignal,
    showAIFeatures = true,
}) => {
    const theme = useTheme();
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

    // State
    const [isAIActive, setIsAIActive] = useState(false);
    const [aiMode, setAIMode] = useState<'auto' | 'manual' | 'scheduled'>('auto');
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [aiThinking, setAiThinking] = useState('');
    const [showAIPanel, setShowAIPanel] = useState(false);
    const [chartType, setChartType] = useState<'candlestick' | 'line' | 'area'>('candlestick');
    const [timeframe, setTimeframe] = useState('15m');
    const [currentSignal, setCurrentSignal] = useState<AISignal | null>(null);
    const [aiDrawings, setAIDrawings] = useState<AIDrawing[]>([]);
    const [detectedPatterns, setDetectedPatterns] = useState<string[]>([]);

    // Indicators & Tools
    const [showIndicators, setShowIndicators] = useState({
        ma: true,
        ema: false,
        bollinger: false,
        volume: true,
        fibonacci: false,
        supportResistance: false,
        patterns: false,
    });

    // Initialize chart
    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height: height - 120, // Account for toolbar
            layout: {
                background: { type: ColorType.Solid, color: theme.palette.background.paper },
                textColor: theme.palette.text.secondary,
            },
            grid: {
                vertLines: { color: alpha(theme.palette.divider, 0.1) },
                horzLines: { color: alpha(theme.palette.divider, 0.1) },
            },
            crosshair: {
                mode: CrosshairMode.Normal,
            },
            rightPriceScale: {
                borderColor: theme.palette.divider,
                scaleMargins: { top: 0.1, bottom: 0.2 },
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

        // Load initial data
        loadChartData();

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
    }, [theme, height]);

    // AI Analysis Effect
    useEffect(() => {
        if (!isAIActive || !chartRef.current) return;

        let interval: NodeJS.Timeout;

        if (aiMode === 'auto') {
            // Run AI analysis every 30 seconds in auto mode
            interval = setInterval(() => {
                runAIAnalysis();
            }, 30000);

            // Run immediately
            runAIAnalysis();
        }

        return () => {
            if (interval) clearInterval(interval);
        };
    }, [isAIActive, aiMode, symbol]);

    const loadChartData = () => {
        // Generate mock data for demonstration
        const basePrice = 450;
        const data: CandlestickData[] = [];
        const currentTime = Math.floor(Date.now() / 1000);

        for (let i = 200; i >= 0; i--) {
            const time = currentTime - i * 300; // 5-minute candles
            const volatility = 0.002;
            const trend = Math.sin(i / 20) * 5;

            const open = basePrice + trend + (Math.random() - 0.5) * basePrice * volatility;
            const close = open + (Math.random() - 0.5) * basePrice * volatility;
            const high = Math.max(open, close) + Math.random() * basePrice * volatility;
            const low = Math.min(open, close) - Math.random() * basePrice * volatility;

            data.push({
                time: time as Time,
                open,
                high,
                low,
                close,
            });
        }

        candleSeriesRef.current?.setData(data);
        chartRef.current?.timeScale().fitContent();
    };

    const runAIAnalysis = async () => {
        setIsAnalyzing(true);
        setShowAIPanel(true);
        setAiThinking('Initializing AI analysis...');

        // Simulate AI analysis steps
        const steps = [
            { message: 'Scanning market structure...', action: detectMarketStructure },
            { message: 'Identifying key levels...', action: drawSupportResistance },
            { message: 'Detecting chart patterns...', action: detectPatterns },
            { message: 'Calculating Fibonacci levels...', action: drawFibonacci },
            { message: 'Analyzing indicators...', action: analyzeIndicators },
            { message: 'Generating high-probability signal...', action: generateSignal },
        ];

        for (const step of steps) {
            setAiThinking(step.message);
            await new Promise(resolve => setTimeout(resolve, 1000));
            await step.action();
        }

        setIsAnalyzing(false);
        setAiThinking('');
    };

    const detectMarketStructure = async () => {
        // Simulate market structure detection
        const structures = ['Uptrend', 'Downtrend', 'Range-bound', 'Breakout'];
        const detected = structures[Math.floor(Math.random() * structures.length)];
        console.log('Detected market structure:', detected);
    };

    const drawSupportResistance = async () => {
        if (!chartRef.current) return;

        const currentPrice = 450;
        const levels = [
            { price: currentPrice * 0.98, type: 'support' },
            { price: currentPrice * 0.96, type: 'support' },
            { price: currentPrice * 1.02, type: 'resistance' },
            { price: currentPrice * 1.04, type: 'resistance' },
        ];

        levels.forEach(level => {
            const drawing: AIDrawing = {
                id: `sr-${Date.now()}-${Math.random()}`,
                type: level.type as 'support' | 'resistance',
                points: [
                    { time: (Date.now() / 1000 - 86400) as Time, price: level.price },
                    { time: (Date.now() / 1000) as Time, price: level.price },
                ],
                label: `${level.type} @ ${level.price.toFixed(2)}`,
                confidence: 85 + Math.random() * 15,
                color: level.type === 'support' ? theme.palette.success.main : theme.palette.error.main,
            };

            setAIDrawings(prev => [...prev, drawing]);
            drawLineOnChart(drawing);
        });
    };

    const detectPatterns = async () => {
        const patterns = [
            'Bull Flag',
            'Ascending Triangle',
            'Double Bottom',
            'Cup and Handle',
            'Head and Shoulders',
            'Wedge',
        ];

        const detected = patterns
            .sort(() => 0.5 - Math.random())
            .slice(0, 2);

        setDetectedPatterns(detected);
    };

    const drawFibonacci = async () => {
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

        setShowIndicators(prev => ({ ...prev, fibonacci: true }));
    };

    const analyzeIndicators = async () => {
        // Simulate indicator analysis
        console.log('Analyzing RSI, MACD, Bollinger Bands...');
    };

    const generateSignal = async () => {
        const currentPrice = 450;
        const isBullish = Math.random() > 0.5;

        const signal: AISignal = {
            id: `signal-${Date.now()}`,
            type: isBullish ? 'BUY' : 'SELL',
            entry: currentPrice + (isBullish ? -0.5 : 0.5),
            stopLoss: currentPrice + (isBullish ? -2 : 2),
            takeProfit: [
                currentPrice + (isBullish ? 2 : -2),
                currentPrice + (isBullish ? 4 : -4),
                currentPrice + (isBullish ? 6 : -6),
            ],
            confidence: 75 + Math.random() * 20,
            patterns: detectedPatterns,
            indicators: ['RSI Oversold', 'MACD Bullish Cross', 'Volume Spike'],
            confluenceScore: 85 + Math.random() * 10,
            reasoning: `Strong ${isBullish ? 'bullish' : 'bearish'} setup with multiple confirmations`,
            timestamp: (Date.now() / 1000) as Time,
        };

        setCurrentSignal(signal);
        onNewSignal?.(signal);
        drawSignalOnChart(signal);
    };

    const drawLineOnChart = (drawing: AIDrawing) => {
        if (!chartRef.current) return;

        const lineSeries = chartRef.current.addLineSeries({
            color: drawing.color,
            lineWidth: 2,
            lineStyle: LineStyle.Dashed,
            lastValueVisible: false,
            priceLineVisible: false,
        });

        const data = drawing.points.map(p => ({ time: p.time, value: p.price }));
        lineSeries.setData(data);
    };

    const drawSignalOnChart = (signal: AISignal) => {
        if (!chartRef.current) return;

        // Entry line
        const entryLine = chartRef.current.addLineSeries({
            color: theme.palette.primary.main,
            lineWidth: 3,
            lineStyle: LineStyle.Solid,
            title: 'Entry',
        });

        entryLine.setData([
            { time: (Date.now() / 1000 - 3600) as Time, value: signal.entry },
            { time: (Date.now() / 1000 + 3600) as Time, value: signal.entry },
        ]);

        // Stop loss line
        const stopLossLine = chartRef.current.addLineSeries({
            color: theme.palette.error.main,
            lineWidth: 2,
            lineStyle: LineStyle.Dashed,
            title: 'Stop Loss',
        });

        stopLossLine.setData([
            { time: (Date.now() / 1000 - 3600) as Time, value: signal.stopLoss },
            { time: (Date.now() / 1000 + 3600) as Time, value: signal.stopLoss },
        ]);

        // Take profit lines
        signal.takeProfit.forEach((tp, i) => {
            const tpLine = chartRef.current!.addLineSeries({
                color: theme.palette.success.main,
                lineWidth: 2,
                lineStyle: LineStyle.Dotted,
                title: `TP${i + 1}`,
            });

            tpLine.setData([
                { time: (Date.now() / 1000 - 3600) as Time, value: tp },
                { time: (Date.now() / 1000 + 3600) as Time, value: tp },
            ]);
        });
    };

    const handleAIToggle = () => {
        setIsAIActive(!isAIActive);
        if (!isAIActive) {
            setShowAIPanel(true);
        } else {
            // Clear AI drawings when turning off
            setAIDrawings([]);
            setDetectedPatterns([]);
            setCurrentSignal(null);
        }
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Enhanced Toolbar */}
            <Paper sx={{ p: 1.5, borderRadius: 0 }}>
                <Stack direction="row" spacing={2} alignItems="center">
                    {/* Chart Type Selector */}
                    <ToggleButtonGroup
                        value={chartType}
                        exclusive
                        onChange={(e, newType) => newType && setChartType(newType)}
                        size="small"
                    >
                        <ToggleButton value="candlestick">
                            <Tooltip title="Candlestick">
                                <CandleIcon fontSize="small" />
                            </Tooltip>
                        </ToggleButton>
                        <ToggleButton value="line">
                            <Tooltip title="Line">
                                <TrendLineIcon fontSize="small" />
                            </Tooltip>
                        </ToggleButton>
                        <ToggleButton value="area">
                            <Tooltip title="Area">
                                <AreaChart fontSize="small" />
                            </Tooltip>
                        </ToggleButton>
                    </ToggleButtonGroup>

                    <Divider orientation="vertical" flexItem />

                    {/* Timeframe Selector */}
                    <FormControl size="small" sx={{ minWidth: 80 }}>
                        <Select
                            value={timeframe}
                            onChange={(e) => setTimeframe(e.target.value)}
                        >
                            <MenuItem value="1m">1m</MenuItem>
                            <MenuItem value="5m">5m</MenuItem>
                            <MenuItem value="15m">15m</MenuItem>
                            <MenuItem value="30m">30m</MenuItem>
                            <MenuItem value="1h">1h</MenuItem>
                            <MenuItem value="4h">4h</MenuItem>
                            <MenuItem value="1d">1D</MenuItem>
                        </Select>
                    </FormControl>

                    <Divider orientation="vertical" flexItem />

                    {/* Indicator Toggles */}
                    <Stack direction="row" spacing={0.5}>
                        <Tooltip title="Moving Averages">
                            <IconButton
                                size="small"
                                onClick={() => setShowIndicators(prev => ({ ...prev, ma: !prev.ma }))}
                                color={showIndicators.ma ? 'primary' : 'default'}
                            >
                                <TimelineIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Fibonacci">
                            <IconButton
                                size="small"
                                onClick={() => setShowIndicators(prev => ({ ...prev, fibonacci: !prev.fibonacci }))}
                                color={showIndicators.fibonacci ? 'primary' : 'default'}
                            >
                                <FibIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Support/Resistance">
                            <IconButton
                                size="small"
                                onClick={() => setShowIndicators(prev => ({ ...prev, supportResistance: !prev.supportResistance }))}
                                color={showIndicators.supportResistance ? 'primary' : 'default'}
                            >
                                <LayersIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Pattern Detection">
                            <IconButton
                                size="small"
                                onClick={() => setShowIndicators(prev => ({ ...prev, patterns: !prev.patterns }))}
                                color={showIndicators.patterns ? 'primary' : 'default'}
                            >
                                <PatternIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                    </Stack>

                    <Box sx={{ flexGrow: 1 }} />

                    {/* AI Controls */}
                    {showAIFeatures && (
                        <>
                            <Divider orientation="vertical" flexItem />
                            <Stack direction="row" spacing={1} alignItems="center">
                                <FormControlLabel
                                    control={
                                        <Switch
                                            checked={isAIActive}
                                            onChange={handleAIToggle}
                                            color="primary"
                                        />
                                    }
                                    label={
                                        <Stack direction="row" spacing={0.5} alignItems="center">
                                            <AIIcon fontSize="small" />
                                            <Typography variant="body2">AI Mode</Typography>
                                        </Stack>
                                    }
                                />

                                {isAIActive && (
                                    <FormControl size="small" sx={{ minWidth: 100 }}>
                                        <Select
                                            value={aiMode}
                                            onChange={(e) => setAIMode(e.target.value as any)}
                                        >
                                            <MenuItem value="auto">Auto</MenuItem>
                                            <MenuItem value="manual">Manual</MenuItem>
                                            <MenuItem value="scheduled">Scheduled</MenuItem>
                                        </Select>
                                    </FormControl>
                                )}

                                {isAIActive && aiMode === 'manual' && (
                                    <Button
                                        variant="contained"
                                        size="small"
                                        onClick={runAIAnalysis}
                                        disabled={isAnalyzing}
                                        startIcon={isAnalyzing ? <SpeedIcon /> : <AIIcon />}
                                    >
                                        {isAnalyzing ? 'Analyzing...' : 'Analyze'}
                                    </Button>
                                )}
                            </Stack>
                        </>
                    )}

                    {/* Chart Actions */}
                    <Stack direction="row" spacing={0.5}>
                        <Tooltip title="Zoom In">
                            <IconButton size="small">
                                <ZoomInIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Zoom Out">
                            <IconButton size="small">
                                <ZoomOutIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Refresh">
                            <IconButton size="small" onClick={loadChartData}>
                                <RefreshIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Fullscreen">
                            <IconButton size="small">
                                <FullscreenIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                    </Stack>
                </Stack>
            </Paper>

            {/* AI Thinking Panel */}
            <AnimatePresence>
                {showAIPanel && isAIActive && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                    >
                        <Paper sx={{ p: 2, borderRadius: 0, bgcolor: alpha(theme.palette.primary.main, 0.05) }}>
                            <Stack spacing={2}>
                                {isAnalyzing && (
                                    <>
                                        <Stack direction="row" spacing={2} alignItems="center">
                                            <PsychologyIcon color="primary" />
                                            <Typography variant="body2" color="primary">
                                                {aiThinking}
                                            </Typography>
                                        </Stack>
                                        <LinearProgress sx={{ height: 2 }} />
                                    </>
                                )}

                                {currentSignal && !isAnalyzing && (
                                    <Alert
                                        severity="success"
                                        icon={<Insights />}
                                        action={
                                            <Button size="small" onClick={() => setCurrentSignal(null)}>
                                                Dismiss
                                            </Button>
                                        }
                                    >
                                        <Stack spacing={1}>
                                            <Typography variant="subtitle2">
                                                {currentSignal.type} Signal Generated - {currentSignal.confidence.toFixed(1)}% Confidence
                                            </Typography>
                                            <Typography variant="body2">
                                                Entry: ${currentSignal.entry.toFixed(2)} |
                                                Stop: ${currentSignal.stopLoss.toFixed(2)} |
                                                Targets: {currentSignal.takeProfit.map(tp => `$${tp.toFixed(2)}`).join(', ')}
                                            </Typography>
                                            <Typography variant="caption" color="text.secondary">
                                                {currentSignal.reasoning}
                                            </Typography>
                                        </Stack>
                                    </Alert>
                                )}

                                {detectedPatterns.length > 0 && (
                                    <Stack direction="row" spacing={1} alignItems="center">
                                        <Typography variant="caption" color="text.secondary">
                                            Detected Patterns:
                                        </Typography>
                                        {detectedPatterns.map(pattern => (
                                            <Chip
                                                key={pattern}
                                                label={pattern}
                                                size="small"
                                                color="primary"
                                                variant="outlined"
                                            />
                                        ))}
                                    </Stack>
                                )}
                            </Stack>
                        </Paper>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Chart Container */}
            <Box
                ref={chartContainerRef}
                sx={{
                    flex: 1,
                    position: 'relative',
                    bgcolor: theme.palette.background.paper,
                }}
            />
        </Box>
    );
};

export default EnhancedTradingChart; 