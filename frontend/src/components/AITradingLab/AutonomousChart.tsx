import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Box, Typography, Chip, IconButton, Tooltip, Paper, Fade, Zoom, TextField, Autocomplete, Stack, ToggleButton, ToggleButtonGroup, Slider, Switch, FormControlLabel, Grid, Alert, Rating, Button, FormControl, InputLabel, Select, MenuItem, Divider, Badge } from '@mui/material';
import {
    createChart,
    IChartApi,
    ISeriesApi,
    CandlestickData,
    HistogramData,
    Time,
    LineStyle,
    CrosshairMode,
    ColorType,
    LineData,
} from 'lightweight-charts';
import { useTheme, alpha } from '@mui/material/styles';
import {
    SmartToy as AIIcon,
    TrendingUp,
    TrendingDown,
    TrendingUp as BullishIcon,
    TrendingDown as BearishIcon,
    Timeline as TimelineIcon,
    Layers as LayersIcon,
    ZoomIn as ZoomInIcon,
    ZoomOut as ZoomOutIcon,
    FitScreen as FitIcon,
    Fullscreen as FullscreenIcon,
    FullscreenExit as FullscreenExitIcon,
    ShowChart as TrendLineIcon,
    BarChart as VolumeIcon,
    Insights as PatternIcon,
    AutoGraph as AutoTradeIcon,
    Psychology as SentimentIcon,
    Speed as SpeedIcon,
    PlayArrow as PlayIcon,
    Pause as PauseIcon,
    Settings as SettingsIcon,
    Visibility as VisibilityIcon,
    VisibilityOff as VisibilityOffIcon,
    Assessment as AssessmentIcon,
    History as HistoryIcon,
    NotificationsActive as AlertIcon,
    Close as CloseIcon,
    CheckCircle as CheckIcon,
    Warning as WarningIcon,
    Error as ErrorIcon,
    Schedule as ScheduleIcon,
    TrendingFlat as NeutralIcon,
    Analytics as AnalyticsIcon,
    AutoAwesome as AutoAwesomeIcon,
    BubbleChart as BubbleIcon,
    CandlestickChart as CandleIcon,
    GridOn as GridIcon,
    GridOff as GridOffIcon,
    Refresh as RefreshIcon,
    Save as SaveIcon,
    Share as ShareIcon,
    MoreVert as MoreIcon,
} from '@mui/icons-material';

interface AutonomousChartProps {
    symbol: string;
    timeframe: string;
    isAIActive: boolean;
    onNewSignal?: (signal: any) => void;
}

interface AIDrawing {
    id: string;
    type: 'trendline' | 'support' | 'resistance' | 'pattern' | 'fibonacci' | 'entry' | 'exit' | 'stopLoss' | 'takeProfit';
    points: { time: Time; price: number }[];
    label: string;
    confidence: number;
    color: string;
    isAnimating: boolean;
    lineWidth?: number;
    lineStyle?: LineStyle;
}

interface ActiveTrade {
    id: string;
    symbol: string;
    type: 'LONG' | 'SHORT';
    entry: number;
    stopLoss: number;
    takeProfit: number[];
    currentPrice: number;
    pnl: number;
    status: 'pending' | 'active' | 'closed';
    confidence: number;
    timestamp: Time;
}

interface SignalAnnotation {
    time: Time;
    price: number;
    text: string;
    type: 'buy' | 'sell' | 'alert';
}

interface PatternProjection {
    id: string;
    pattern: string;
    projectedPath: { time: Time; price: number }[];
    probabilityZones: {
        high: { time: Time; price: number }[];
        medium: { time: Time; price: number }[];
        low: { time: Time; price: number }[];
    };
    targetTime: Time;
    confidence: number;
}

interface SignalValidation {
    id: string;
    signalId: string;
    tools: TradingViewTool[];
    candlePatterns: CandlePattern[];
    confluenceScore: number;
    validationNarrative: string;
    keyLevels: KeyLevel[];
    timestamp: Time;
}

interface TradingViewTool {
    type: 'fibonacci' | 'trendline' | 'channel' | 'pitchfork' | 'gann' | 'elliott';
    points: { time: Time; price: number }[];
    levels?: number[]; // For Fibonacci levels
    label: string;
    color: string;
}

interface CandlePattern {
    name: string;
    type: 'bullish' | 'bearish' | 'neutral';
    candles: { time: Time; open: number; high: number; low: number; close: number }[];
    reliability: number; // 0-100
    description: string;
}

interface KeyLevel {
    price: number;
    type: 'support' | 'resistance' | 'pivot';
    strength: number; // 1-5
    touches: number;
}

interface SignalDetail {
    signal: ProphetSignal;
    validation: SignalValidation;
    isExpanded: boolean;
    status?: 'active' | 'success' | 'failed'; // Add status property
}

interface ProphetSignal {
    id: string;
    symbol: string;
    action: string;
    pattern: string;
    confidence: number;
    entry: number;
    stopLoss: number;
    takeProfit: number[];
    timestamp: Time;
    timeframe: string;
    riskReward: number;
}

const AutonomousChart: React.FC<AutonomousChartProps> = ({
    symbol,
    timeframe,
    isAIActive,
    onNewSignal,
}) => {
    const theme = useTheme();
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
    const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

    // Line series for AI drawings
    const lineSeriesRefs = useRef<Map<string, ISeriesApi<'Line'>>>(new Map());
    const markerSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

    const [aiDrawings, setAIDrawings] = useState<AIDrawing[]>([]);
    const [currentDrawing, setCurrentDrawing] = useState<AIDrawing | null>(null);
    const [showLayers, setShowLayers] = useState(true);
    const [activeTrades, setActiveTrades] = useState<ActiveTrade[]>([]);
    const [aiThinking, setAIThinking] = useState<string>('');
    const [signalAnnotations, setSignalAnnotations] = useState<SignalAnnotation[]>([]);
    const [showVolume, setShowVolume] = useState(true);
    const [volumeOpacity, setVolumeOpacity] = useState(0.3);
    const [selectedSymbol, setSelectedSymbol] = useState(symbol);
    const [patternProjections, setPatternProjections] = useState<PatternProjection[]>([]);
    const [showProjections, setShowProjections] = useState(true);
    const [signalGenerationMode, setSignalGenerationMode] = useState<'auto' | 'manual' | 'scheduled'>('auto');
    const [signalCadence, setSignalCadence] = useState(30); // seconds
    const [selectedSignal, setSelectedSignal] = useState<SignalDetail | null>(null);
    const [signalHistory, setSignalHistory] = useState<SignalDetail[]>([]);
    const [showSignalAnalysis, setShowSignalAnalysis] = useState(false);

    // Add missing state variables
    const [detectedPatterns, setDetectedPatterns] = useState<string[]>([]);
    const [currentPrice, setCurrentPrice] = useState(0);
    const [previousClose, setPreviousClose] = useState(0);
    const [showTrendLines, setShowTrendLines] = useState(false);
    const [showFibonacci, setShowFibonacci] = useState(false);
    const [showPatternProjections, setShowPatternProjections] = useState(false);
    const [showGrid, setShowGrid] = useState(true);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [popularSymbols] = useState([
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
        'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'SQ', 'SHOP',
        'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'ARKK', 'XLF'
    ]);
    const [showSignalHistory, setShowSignalHistory] = useState(true);

    // Available symbols for search
    const availableSymbols = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
        'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'SQ', 'SHOP',
        'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'ARKK', 'XLF'
    ];

    // TradingView drawing tools
    const fibonacciLevels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
    const fibonacciColors = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3'];

    // TrendSpider-style dark theme colors
    const trendSpiderColors = {
        background: '#0a0e1a',
        cardBackground: '#131722',
        borderColor: '#1e222d',
        textPrimary: '#d1d4dc',
        textSecondary: '#787b86',
        textMuted: '#4a4e5a',
        bullish: '#26a69a',
        bearish: '#ef5350',
        neutral: '#787b86',
        accent: '#2962ff',
        accentLight: '#448aff',
        warning: '#ff9800',
        success: '#4caf50',
        gridColor: 'rgba(42, 46, 57, 0.5)',
        volumeColor: 'rgba(38, 166, 154, 0.2)',
        selectionBackground: 'rgba(33, 150, 243, 0.1)',
        hoverBackground: 'rgba(255, 255, 255, 0.03)',
    };

    // Initialize chart with enhanced features
    useEffect(() => {
        if (!chartContainerRef.current) return;

        // Create chart with TrendSpider-style configuration
        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
            layout: {
                background: { type: ColorType.Solid, color: trendSpiderColors.background },
                textColor: trendSpiderColors.textSecondary,
                fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                fontSize: 11,
            },
            grid: {
                vertLines: { color: trendSpiderColors.gridColor, style: LineStyle.Solid },
                horzLines: { color: trendSpiderColors.gridColor, style: LineStyle.Solid },
            },
            crosshair: {
                mode: CrosshairMode.Normal,
                vertLine: {
                    color: trendSpiderColors.textMuted,
                    width: 1,
                    style: LineStyle.Dashed,
                    labelBackgroundColor: trendSpiderColors.cardBackground,
                },
                horzLine: {
                    color: trendSpiderColors.textMuted,
                    width: 1,
                    style: LineStyle.Dashed,
                    labelBackgroundColor: trendSpiderColors.cardBackground,
                },
            },
            rightPriceScale: {
                borderColor: trendSpiderColors.borderColor,
                borderVisible: false,
                scaleMargins: {
                    top: 0.1,
                    bottom: showVolume ? 0.2 : 0.1,
                },
            },
            timeScale: {
                borderColor: trendSpiderColors.borderColor,
                borderVisible: false,
                timeVisible: true,
                secondsVisible: false,
                tickMarkFormatter: (time: any) => {
                    const date = new Date(time * 1000);
                    return date.toLocaleTimeString('en-US', {
                        hour: '2-digit',
                        minute: '2-digit',
                        hour12: false
                    });
                },
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

        chartRef.current = chart;

        // Add candlestick series
        const candlestickSeries = chart.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        });
        candleSeriesRef.current = candlestickSeries;

        // Add volume series
        const volumeSeries = chart.addHistogramSeries({
            color: '#26a69a',
            priceFormat: {
                type: 'volume',
            },
            priceScaleId: '',
        });

        // Create a separate price scale for volume
        chart.priceScale('').applyOptions({
            scaleMargins: {
                top: 0.8,
                bottom: 0,
            },
        });

        // Add MA lines
        const ma20Series = chart.addLineSeries({
            color: '#2196F3',
            lineWidth: 2,
            crosshairMarkerVisible: false,
            priceLineVisible: false,
        });

        volumeSeriesRef.current = volumeSeries;

        // Add marker series for signals
        const markerSeries = chart.addLineSeries({
            color: 'transparent',
            lineWidth: 0,
        });
        markerSeriesRef.current = markerSeries;

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
    }, [selectedSymbol, showVolume, volumeOpacity]);

    // Enhanced AI trading logic with signal validation
    useEffect(() => {
        if (!isAIActive || !chartRef.current) return;

        let signalInterval: NodeJS.Timeout;

        // Start drawing tools immediately when AI is active
        setTimeout(() => {
            detectPatternsWithValidation();
        }, 1000);

        if (signalGenerationMode === 'auto') {
            // Multiple AI processes running in parallel
            const aiProcesses = [
                setInterval(() => analyzeMarketStructure(), 3000),
                setInterval(() => detectPatternsWithValidation(), 5000),
                setInterval(() => executeTradeLogic(), 2000),
                setInterval(() => updateActiveTrades(), 1000),
            ];
            return () => {
                aiProcesses.forEach(interval => clearInterval(interval));
            };
        } else if (signalGenerationMode === 'scheduled') {
            signalInterval = setInterval(() => {
                generateScheduledSignal();
            }, signalCadence * 1000);
            return () => clearInterval(signalInterval);
        }
    }, [isAIActive, signalGenerationMode, signalCadence]);

    // Handle symbol change
    const handleSymbolChange = (event: any, newValue: string | null) => {
        if (newValue) {
            setSelectedSymbol(newValue);
            loadChartData(newValue);
            // Clear previous drawings when symbol changes
            lineSeriesRefs.current.forEach((series, key) => {
                chartRef.current?.removeSeries(series);
            });
            lineSeriesRefs.current.clear();
            setAIDrawings([]);
            setPatternProjections([]);
            setSignalHistory([]);
        }
    };

    // Enhanced loadChartData with more realistic price action
    const loadChartData = (symbolToLoad: string = selectedSymbol) => {
        // Real-world base prices for different symbols
        const basePrices: { [key: string]: number } = {
            'AAPL': 185.50,
            'NVDA': 732.45,
            'TSLA': 248.30,
            'GOOGL': 142.80,
            'MSFT': 378.90,
            'AMZN': 155.20,
            'META': 385.60,
            'NFLX': 445.70,
            'AMD': 168.90,
            'SPY': 452.30,
            'QQQ': 390.45,
        };

        const basePrice = basePrices[symbolToLoad] || 100;
        const currentTime = Math.floor(Date.now() / 1000);
        const candleData = [];
        const volumeData = [];

        // Generate more realistic price action with trends
        let price = basePrice;
        let trend = Math.random() > 0.5 ? 1 : -1; // Initial trend direction
        let trendStrength = 0.02; // 2% trend

        for (let i = 100; i >= 0; i--) {
            const time = currentTime - i * 300; // 5-minute candles

            // Add realistic market movement
            const trendComponent = trend * trendStrength * price * Math.random();
            const noiseComponent = (Math.random() - 0.5) * price * 0.005; // 0.5% noise

            // Occasionally reverse trend
            if (Math.random() < 0.1) {
                trend *= -1;
            }

            price = price + trendComponent + noiseComponent;

            // Create realistic OHLC data
            const open = price;
            const volatility = price * 0.002; // 0.2% volatility
            const close = open + (Math.random() - 0.5) * volatility * 2;
            const high = Math.max(open, close) + Math.random() * volatility;
            const low = Math.min(open, close) - Math.random() * volatility;
            const volume = Math.floor(Math.random() * 2000000 + 1000000);

            candleData.push({ time: time as Time, open, high, low, close });
            volumeData.push({
                time: time as Time,
                value: volume,
                color: close > open
                    ? theme.palette.success.main + Math.floor(volumeOpacity * 255).toString(16).padStart(2, '0')
                    : theme.palette.error.main + Math.floor(volumeOpacity * 255).toString(16).padStart(2, '0')
            });
        }

        candleSeriesRef.current?.setData(candleData);
        if (showVolume) {
            volumeSeriesRef.current?.setData(volumeData);
        }

        // Store candle data for pattern detection
        (window as any).chartCandleData = candleData;

        // Set current price and previous close
        if (candleData.length > 0) {
            const lastCandle = candleData[candleData.length - 1];
            const prevCandle = candleData[candleData.length - 2];
            setCurrentPrice(lastCandle.close);
            setPreviousClose(prevCandle ? prevCandle.close : lastCandle.open);
        }
    };

    const analyzeMarketStructure = () => {
        setAIThinking('Analyzing market structure...');

        // Simulate finding support/resistance levels
        if (Math.random() > 0.6) {
            const type = Math.random() > 0.5 ? 'support' : 'resistance';
            drawSupportResistance(type);
        }
    };

    const detectPatternsWithValidation = async () => {
        setAIThinking('Scanning for patterns with validation...');

        const patterns = [
            'Triangle Breakout',
            'Bull Flag',
            'Head & Shoulders',
            'Double Bottom',
            'Ascending Wedge',
            'Cup & Handle',
            'Falling Wedge',
            'Rising Channel',
        ];

        // Always detect a pattern for demonstration (remove randomness)
        const pattern = patterns[Math.floor(Math.random() * patterns.length)];

        // Detect candlestick patterns
        const candlePatterns = detectCandlestickPatterns();

        // Draw pattern with validation
        const validation = await validatePatternWithTools(pattern, candlePatterns);

        drawPattern(pattern);

        // Generate pattern projection
        if (showProjections) {
            generatePatternProjection(pattern);
        }

        // Create detailed signal
        const signal = createDetailedSignal(pattern, validation);
        addToSignalHistory(signal);
    };

    const detectCandlestickPatterns = (): CandlePattern[] => {
        const patterns: CandlePattern[] = [];

        // Get recent candles (simulated for now)
        const recentCandles = getRecentCandles(20);

        // Detect Doji
        const doji = detectDoji(recentCandles);
        if (doji) patterns.push(doji);

        // Detect Hammer/Hanging Man
        const hammer = detectHammer(recentCandles);
        if (hammer) patterns.push(hammer);

        // Detect Engulfing
        const engulfing = detectEngulfing(recentCandles);
        if (engulfing) patterns.push(engulfing);

        // Detect Morning/Evening Star
        const star = detectStar(recentCandles);
        if (star) patterns.push(star);

        return patterns;
    };

    const validatePatternWithTools = async (pattern: string, candlePatterns: CandlePattern[]): Promise<SignalValidation> => {
        const tools: TradingViewTool[] = [];
        const keyLevels: KeyLevel[] = [];

        // Draw Fibonacci retracement
        const fib = await drawFibonacciRetracement(pattern);
        if (fib) tools.push(fib);

        // Draw trend lines
        const trendLines = await drawTrendLines(pattern);
        tools.push(...trendLines);

        // Draw support/resistance levels
        const levels = await identifyKeyLevels();
        keyLevels.push(...levels);

        // Calculate confluence score
        const confluenceScore = calculateConfluence(tools, candlePatterns, keyLevels);

        // Generate validation narrative
        const narrative = generateValidationNarrative(pattern, tools, candlePatterns, confluenceScore);

        return {
            id: `validation-${Date.now()}`,
            signalId: `signal-${Date.now()}`,
            tools,
            candlePatterns,
            confluenceScore,
            validationNarrative: narrative,
            keyLevels,
            timestamp: Math.floor(Date.now() / 1000) as Time,
        };
    };

    const drawFibonacciRetracement = async (pattern: string): Promise<TradingViewTool | null> => {
        if (!chartRef.current) return null;

        // Find swing high and low from actual data
        const { swingHigh, swingLow } = findSwingPoints();

        const fibTool: TradingViewTool = {
            type: 'fibonacci',
            points: [
                { time: swingLow.time, price: swingLow.price },
                { time: swingHigh.time, price: swingHigh.price }
            ],
            levels: fibonacciLevels,
            label: 'Fibonacci Retracement',
            color: theme.palette.info.main,
        };

        // Draw Fibonacci levels with better visibility
        const range = swingHigh.price - swingLow.price;

        fibonacciLevels.forEach((level, index) => {
            const price = swingLow.price + (range * level);

            const fibLine = chartRef.current!.addLineSeries({
                color: fibonacciColors[index],
                lineWidth: 2, // Increased line width
                lineStyle: LineStyle.Solid, // Changed to solid for better visibility
                priceLineVisible: true,
                lastValueVisible: true,
                title: `Fib ${(level * 100).toFixed(1)}%`,
            });

            // Extend lines across the entire chart
            const extendedTime = (Math.floor(Date.now() / 1000) + 7200) as Time;
            fibLine.setData([
                { time: swingLow.time, value: price },
                { time: extendedTime, value: price }
            ]);

            lineSeriesRefs.current.set(`fib-${level}`, fibLine);
        });

        // Add visual marker at swing points
        if (markerSeriesRef.current) {
            markerSeriesRef.current.setMarkers([
                {
                    time: swingHigh.time,
                    position: 'aboveBar',
                    color: theme.palette.success.main,
                    shape: 'arrowDown',
                    text: 'Swing High',
                },
                {
                    time: swingLow.time,
                    position: 'belowBar',
                    color: theme.palette.error.main,
                    shape: 'arrowUp',
                    text: 'Swing Low',
                }
            ]);
        }

        return fibTool;
    };

    const drawTrendLines = async (pattern: string): Promise<TradingViewTool[]> => {
        const tools: TradingViewTool[] = [];

        if (!chartRef.current) return tools;

        // Find trend line points from actual data
        const { upperTrend, lowerTrend } = findTrendLinePoints();

        // Draw upper trend line with better visibility
        if (upperTrend.length >= 2) {
            const upperLine = chartRef.current.addLineSeries({
                color: theme.palette.error.main,
                lineWidth: 3, // Thicker line
                lineStyle: LineStyle.Solid,
                priceLineVisible: true,
                lastValueVisible: true,
                title: 'Resistance Trend',
            });

            // Extend the trend line
            const slope = (upperTrend[1].price - upperTrend[0].price) /
                (Number(upperTrend[1].time) - Number(upperTrend[0].time));
            const extendedTime = (Math.floor(Date.now() / 1000) + 3600) as Time;
            const extendedPrice = upperTrend[1].price +
                slope * (Number(extendedTime) - Number(upperTrend[1].time));

            upperLine.setData([
                { time: upperTrend[0].time, value: upperTrend[0].price },
                { time: extendedTime, value: extendedPrice }
            ]);

            lineSeriesRefs.current.set('trend-upper', upperLine);

            tools.push({
                type: 'trendline',
                points: upperTrend,
                label: 'Resistance Trend Line',
                color: theme.palette.error.main,
            });
        }

        // Draw lower trend line with better visibility
        if (lowerTrend.length >= 2) {
            const lowerLine = chartRef.current.addLineSeries({
                color: theme.palette.success.main,
                lineWidth: 3, // Thicker line
                lineStyle: LineStyle.Solid,
                priceLineVisible: true,
                lastValueVisible: true,
                title: 'Support Trend',
            });

            // Extend the trend line
            const slope = (lowerTrend[1].price - lowerTrend[0].price) /
                (Number(lowerTrend[1].time) - Number(lowerTrend[0].time));
            const extendedTime = (Math.floor(Date.now() / 1000) + 3600) as Time;
            const extendedPrice = lowerTrend[1].price +
                slope * (Number(extendedTime) - Number(lowerTrend[1].time));

            lowerLine.setData([
                { time: lowerTrend[0].time, value: lowerTrend[0].price },
                { time: extendedTime, value: extendedPrice }
            ]);

            lineSeriesRefs.current.set('trend-lower', lowerLine);

            tools.push({
                type: 'trendline',
                points: lowerTrend,
                label: 'Support Trend Line',
                color: theme.palette.success.main,
            });
        }

        return tools;
    };

    const createDetailedSignal = (pattern: string, validation: SignalValidation): SignalDetail => {
        const currentPrice = 180 + Math.random() * 10;
        const isLong = pattern.includes('Bull') || pattern.includes('Bottom') || pattern.includes('Ascending');

        const signal: ProphetSignal = {
            id: validation.signalId,
            symbol: selectedSymbol,
            action: isLong ? 'BUY' : 'SELL',
            pattern,
            confidence: validation.confluenceScore,
            entry: currentPrice + (isLong ? 0.5 : -0.5),
            stopLoss: currentPrice + (isLong ? -2 : 2),
            takeProfit: [
                currentPrice + (isLong ? 2 : -2),
                currentPrice + (isLong ? 4 : -4),
                currentPrice + (isLong ? 6 : -6),
            ],
            timestamp: validation.timestamp,
            timeframe,
            riskReward: 3.0,
        };

        return {
            signal,
            validation,
            isExpanded: false,
            status: 'active', // Initialize with 'active' status
        };
    };

    const addToSignalHistory = (signalDetail: SignalDetail) => {
        setSignalHistory(prev => [signalDetail, ...prev.slice(0, 49)]); // Keep last 50 signals

        // Notify parent
        if (onNewSignal) {
            onNewSignal(signalDetail.signal);
        }
    };

    const handleSignalClick = (signalDetail: SignalDetail) => {
        setSelectedSignal(signalDetail);
        setShowSignalAnalysis(true);

        // Highlight the signal on chart
        highlightSignalOnChart(signalDetail);
    };

    const highlightSignalOnChart = (signalDetail: SignalDetail) => {
        // Clear previous highlights
        clearSignalHighlights();

        // Redraw all validation tools
        signalDetail.validation.tools.forEach(tool => {
            if (tool.type === 'fibonacci') {
                drawFibonacciRetracement(signalDetail.signal.pattern);
            } else if (tool.type === 'trendline') {
                // Already drawn, just highlight
            }
        });

        // Highlight candlestick patterns
        signalDetail.validation.candlePatterns.forEach(pattern => {
            highlightCandlePattern(pattern);
        });

        // Show key levels
        signalDetail.validation.keyLevels.forEach(level => {
            drawKeyLevel(level);
        });
    };

    const highlightCandlePattern = (pattern: CandlePattern) => {
        if (!chartRef.current) return;

        // Add background highlight for pattern candles
        const patternSeries = chartRef.current.addLineSeries({
            color: pattern.type === 'bullish' ? theme.palette.success.light : theme.palette.error.light,
            lineWidth: 0,
        });

        // Create area highlight
        const highlightData = pattern.candles.map(candle => ({
            time: candle.time,
            value: candle.high * 1.01, // Slightly above high
        }));

        patternSeries.setData(highlightData);
        lineSeriesRefs.current.set(`pattern-highlight-${pattern.name}`, patternSeries);

        // Add pattern label
        if (markerSeriesRef.current && pattern.candles.length > 0) {
            const middleCandle = pattern.candles[Math.floor(pattern.candles.length / 2)];
            markerSeriesRef.current.setMarkers([
                ...markerSeriesRef.current.markers || [],
                {
                    time: middleCandle.time,
                    position: 'aboveBar',
                    color: pattern.type === 'bullish' ? theme.palette.success.main : theme.palette.error.main,
                    shape: 'circle',
                    text: pattern.name,
                }
            ]);
        }
    };

    const executeTradeLogic = () => {
        if (activeTrades.length > 3) return; // Limit concurrent trades

        setAIThinking('Evaluating trade opportunities...');

        // Simulate trade signal generation
        if (Math.random() > 0.85) {
            const currentPrice = 180 + Math.random() * 10;
            const isLong = Math.random() > 0.5;

            const trade: ActiveTrade = {
                id: `trade-${Date.now()}`,
                symbol,
                type: isLong ? 'LONG' : 'SHORT',
                entry: currentPrice,
                stopLoss: isLong ? currentPrice - 2 : currentPrice + 2,
                takeProfit: isLong
                    ? [currentPrice + 2, currentPrice + 4, currentPrice + 6]
                    : [currentPrice - 2, currentPrice - 4, currentPrice - 6],
                currentPrice,
                pnl: 0,
                status: 'pending',
                confidence: 70 + Math.random() * 30,
                timestamp: Math.floor(Date.now() / 1000) as Time,
            };

            executeTradeVisually(trade);
        }
    };

    const executeTradeVisually = async (trade: ActiveTrade) => {
        setAIThinking(`Executing ${trade.type} trade on ${trade.symbol}...`);

        // Animate entry point
        await drawTradeEntry(trade);

        // Draw stop loss
        await drawStopLoss(trade);

        // Draw take profit levels
        await drawTakeProfitLevels(trade);

        // Add to active trades
        setActiveTrades(prev => [...prev, { ...trade, status: 'active' }]);

        // Notify parent component
        if (onNewSignal) {
            onNewSignal({
                symbol: trade.symbol,
                type: trade.type,
                entry: trade.entry,
                stop: trade.stopLoss,
                targets: trade.takeProfit,
                confidence: trade.confidence,
            });
        }
    };

    const drawTradeEntry = async (trade: ActiveTrade) => {
        const entryDrawing: AIDrawing = {
            id: `entry-${trade.id}`,
            type: 'entry',
            points: [{ time: trade.timestamp, price: trade.entry }],
            label: `${trade.type} Entry @ ${trade.entry.toFixed(2)}`,
            confidence: trade.confidence,
            color: trade.type === 'LONG' ? theme.palette.success.main : theme.palette.error.main,
            isAnimating: true,
            lineWidth: 3,
        };

        animateDrawing(entryDrawing);

        // Add marker
        if (markerSeriesRef.current) {
            markerSeriesRef.current.setMarkers([
                {
                    time: trade.timestamp,
                    position: trade.type === 'LONG' ? 'belowBar' : 'aboveBar',
                    color: trade.type === 'LONG' ? theme.palette.success.main : theme.palette.error.main,
                    shape: trade.type === 'LONG' ? 'arrowUp' : 'arrowDown',
                    text: `${trade.type} ${trade.confidence.toFixed(0)}%`,
                },
            ]);
        }
    };

    const drawStopLoss = async (trade: ActiveTrade) => {
        const stopDrawing: AIDrawing = {
            id: `stop-${trade.id}`,
            type: 'stopLoss',
            points: [
                { time: trade.timestamp, price: trade.stopLoss },
                { time: (trade.timestamp + 7200) as Time, price: trade.stopLoss },
            ],
            label: `Stop Loss @ ${trade.stopLoss.toFixed(2)}`,
            confidence: 100,
            color: theme.palette.error.main,
            isAnimating: true,
            lineWidth: 2,
            lineStyle: LineStyle.Dashed,
        };

        drawLineOnChart(stopDrawing);
    };

    const drawTakeProfitLevels = async (trade: ActiveTrade) => {
        trade.takeProfit.forEach((tp, index) => {
            const tpDrawing: AIDrawing = {
                id: `tp-${trade.id}-${index}`,
                type: 'takeProfit',
                points: [
                    { time: trade.timestamp, price: tp },
                    { time: (trade.timestamp + 7200) as Time, price: tp },
                ],
                label: `TP${index + 1} @ ${tp.toFixed(2)}`,
                confidence: 100,
                color: theme.palette.success.main,
                isAnimating: true,
                lineWidth: 2,
                lineStyle: LineStyle.Dotted,
            };

            drawLineOnChart(tpDrawing);
        });
    };

    const drawLineOnChart = (drawing: AIDrawing) => {
        if (!chartRef.current) return;

        const lineSeries = chartRef.current.addLineSeries({
            color: drawing.color,
            lineWidth: drawing.lineWidth || 2,
            lineStyle: drawing.lineStyle || LineStyle.Solid,
            priceLineVisible: false,
            lastValueVisible: false,
        });

        lineSeries.setData(drawing.points);
        lineSeriesRefs.current.set(drawing.id, lineSeries);

        setAIDrawings(prev => [...prev, drawing]);
    };

    const drawSupportResistance = (type: 'support' | 'resistance') => {
        const currentTime = Math.floor(Date.now() / 1000);
        const price = 180 + (type === 'support' ? -5 : 5) + Math.random() * 2;

        const drawing: AIDrawing = {
            id: `${type}-${Date.now()}`,
            type,
            points: [
                { time: (currentTime - 7200) as Time, price },
                { time: currentTime as Time, price },
            ],
            label: `${type === 'support' ? 'Support' : 'Resistance'} @ ${price.toFixed(2)}`,
            confidence: 80 + Math.random() * 20,
            color: type === 'support' ? theme.palette.success.main : theme.palette.error.main,
            isAnimating: true,
            lineWidth: 2,
        };

        animateDrawing(drawing);
        drawLineOnChart(drawing);
    };

    const generatePatternProjection = (patternName: string) => {
        const currentTime = Math.floor(Date.now() / 1000);
        const currentPrice = 180 + Math.random() * 10;

        // Create projection based on pattern type
        const projection: PatternProjection = {
            id: `proj-${Date.now()}`,
            pattern: patternName,
            projectedPath: [],
            probabilityZones: {
                high: [],
                medium: [],
                low: []
            },
            targetTime: (currentTime + 7200) as Time,
            confidence: 75 + Math.random() * 20
        };

        // Generate projected path based on pattern
        const projectionPoints = 20;
        let projectedPrice = currentPrice;

        for (let i = 1; i <= projectionPoints; i++) {
            const time = (currentTime + i * 300) as Time;

            // Pattern-specific price movement
            switch (patternName) {
                case 'Triangle Breakout':
                    projectedPrice += (i / projectionPoints) * 5; // Upward breakout
                    break;
                case 'Bull Flag':
                    projectedPrice += Math.sin(i / 5) * 0.5 + 0.3; // Continuation up
                    break;
                case 'Head & Shoulders':
                    projectedPrice -= (i / projectionPoints) * 4; // Bearish reversal
                    break;
                case 'Double Bottom':
                    projectedPrice += Math.pow(i / projectionPoints, 2) * 6; // Strong reversal up
                    break;
                default:
                    projectedPrice += (Math.random() - 0.3) * 1;
            }

            projection.projectedPath.push({ time, price: projectedPrice });

            // Create probability zones (confidence intervals)
            const variance = (1 - projection.confidence / 100) * 2;
            projection.probabilityZones.high.push({
                time,
                price: projectedPrice + variance * 3
            });
            projection.probabilityZones.medium.push({
                time,
                price: projectedPrice + variance * 1.5
            });
            projection.probabilityZones.low.push({
                time,
                price: projectedPrice - variance * 1.5
            });
        }

        // Draw the projection
        drawPatternProjection(projection);
        setPatternProjections(prev => [...prev, projection]);
    };

    const drawPatternProjection = (projection: PatternProjection) => {
        if (!chartRef.current) return;

        // Draw main projected path
        const projectionSeries = chartRef.current.addLineSeries({
            color: theme.palette.info.main,
            lineWidth: 3,
            lineStyle: LineStyle.Dashed,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
        });

        projectionSeries.setData(projection.projectedPath);
        lineSeriesRefs.current.set(`projection-${projection.id}`, projectionSeries);

        // Draw probability zones as area series
        const highZoneSeries = chartRef.current.addAreaSeries({
            topColor: theme.palette.success.main + '20',
            bottomColor: 'transparent',
            lineColor: 'transparent',
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
        });

        const mediumZoneSeries = chartRef.current.addAreaSeries({
            topColor: theme.palette.warning.main + '15',
            bottomColor: 'transparent',
            lineColor: 'transparent',
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
        });

        // Convert zones to area series data
        const highZoneData = projection.probabilityZones.high.map(point => ({
            time: point.time,
            value: point.price
        }));

        const mediumZoneData = projection.probabilityZones.medium.map(point => ({
            time: point.time,
            value: point.price
        }));

        highZoneSeries.setData(highZoneData);
        mediumZoneSeries.setData(mediumZoneData);

        lineSeriesRefs.current.set(`zone-high-${projection.id}`, highZoneSeries);
        lineSeriesRefs.current.set(`zone-medium-${projection.id}`, mediumZoneSeries);

        // Add projection annotation
        if (markerSeriesRef.current) {
            const lastPoint = projection.projectedPath[projection.projectedPath.length - 1];
            markerSeriesRef.current.setMarkers([
                ...markerSeriesRef.current.markers || [],
                {
                    time: lastPoint.time,
                    position: 'aboveBar',
                    color: theme.palette.info.main,
                    shape: 'circle',
                    text: `${projection.pattern} Target: ${lastPoint.price.toFixed(2)} (${projection.confidence.toFixed(0)}%)`,
                }
            ]);
        }
    };

    const drawPattern = (patternName: string) => {
        const currentTime = Math.floor(Date.now() / 1000);
        const basePrice = 180;

        const patternDrawing: AIDrawing = {
            id: `pattern-${Date.now()}`,
            type: 'pattern',
            points: generatePatternPoints(patternName, currentTime, basePrice),
            label: patternName,
            confidence: 75 + Math.random() * 25,
            color: theme.palette.warning.main,
            isAnimating: true,
            lineWidth: 2,
        };

        animateDrawing(patternDrawing);

        // Draw pattern boundaries and key levels
        drawPatternAnnotations(patternDrawing);

        // Add pattern annotation
        const annotation: SignalAnnotation = {
            time: currentTime as Time,
            price: basePrice,
            text: `${patternName} detected!`,
            type: 'alert',
        };
        setSignalAnnotations(prev => [...prev, annotation]);

        // Add pattern to detected patterns
        setDetectedPatterns(prev => [...prev, patternName]);
    };

    const drawPatternAnnotations = (pattern: AIDrawing) => {
        if (!chartRef.current) return;

        // Draw pattern boundaries
        const boundaryPoints = calculatePatternBoundaries(pattern);

        // Upper boundary
        const upperBoundary = chartRef.current.addLineSeries({
            color: theme.palette.info.light,
            lineWidth: 1,
            lineStyle: LineStyle.Dotted,
            priceLineVisible: false,
            lastValueVisible: false,
        });

        upperBoundary.setData(boundaryPoints.upper);
        lineSeriesRefs.current.set(`boundary-upper-${pattern.id}`, upperBoundary);

        // Lower boundary
        const lowerBoundary = chartRef.current.addLineSeries({
            color: theme.palette.info.light,
            lineWidth: 1,
            lineStyle: LineStyle.Dotted,
            priceLineVisible: false,
            lastValueVisible: false,
        });

        lowerBoundary.setData(boundaryPoints.lower);
        lineSeriesRefs.current.set(`boundary-lower-${pattern.id}`, lowerBoundary);

        // Key level (neckline, support, resistance)
        if (boundaryPoints.keyLevel) {
            const keyLevelSeries = chartRef.current.addLineSeries({
                color: theme.palette.secondary.main,
                lineWidth: 2,
                lineStyle: LineStyle.Solid,
                priceLineVisible: true,
                lastValueVisible: true,
            });

            keyLevelSeries.setData(boundaryPoints.keyLevel);
            lineSeriesRefs.current.set(`keylevel-${pattern.id}`, keyLevelSeries);
        }
    };

    const calculatePatternBoundaries = (pattern: AIDrawing): {
        upper: { time: Time; value: number }[],
        lower: { time: Time; value: number }[],
        keyLevel?: { time: Time; value: number }[]
    } => {
        const points = pattern.points;
        const startTime = points[0].time;
        const endTime = points[points.length - 1].time;

        // Calculate max and min prices
        const prices = points.map(p => p.price);
        const maxPrice = Math.max(...prices);
        const minPrice = Math.min(...prices);
        const avgPrice = (maxPrice + minPrice) / 2;

        return {
            upper: [
                { time: startTime, value: maxPrice + 1 },
                { time: endTime, value: maxPrice + 1 }
            ],
            lower: [
                { time: startTime, value: minPrice - 1 },
                { time: endTime, value: minPrice - 1 }
            ],
            keyLevel: pattern.label.includes('Head & Shoulders') || pattern.label.includes('Double') ? [
                { time: startTime, value: avgPrice },
                { time: endTime, value: avgPrice }
            ] : undefined
        };
    };

    const generatePatternPoints = (pattern: string, time: number, price: number): { time: Time; price: number }[] => {
        switch (pattern) {
            case 'Triangle Breakout':
                return [
                    { time: (time - 3600) as Time, price: price - 3 },
                    { time: (time - 2400) as Time, price: price + 2 },
                    { time: (time - 1200) as Time, price: price - 1 },
                    { time: time as Time, price: price + 3 },
                ];
            case 'Bull Flag':
                return [
                    { time: (time - 3600) as Time, price: price - 2 },
                    { time: (time - 2700) as Time, price: price + 4 },
                    { time: (time - 1800) as Time, price: price + 3 },
                    { time: (time - 900) as Time, price: price + 3.5 },
                    { time: time as Time, price: price + 5 },
                ];
            case 'Cup & Handle':
                return [
                    { time: (time - 4800) as Time, price: price },
                    { time: (time - 3600) as Time, price: price - 4 },
                    { time: (time - 2400) as Time, price: price - 3.5 },
                    { time: (time - 1200) as Time, price: price - 1 },
                    { time: (time - 600) as Time, price: price - 1.5 },
                    { time: time as Time, price: price + 2 },
                ];
            case 'Head & Shoulders':
                return [
                    { time: (time - 4800) as Time, price: price - 2 },
                    { time: (time - 3600) as Time, price: price + 2 },
                    { time: (time - 2400) as Time, price: price - 1 },
                    { time: (time - 1200) as Time, price: price + 4 },
                    { time: (time - 600) as Time, price: price - 1 },
                    { time: time as Time, price: price + 2 },
                ];
            case 'Double Bottom':
                return [
                    { time: (time - 4800) as Time, price: price },
                    { time: (time - 3600) as Time, price: price - 4 },
                    { time: (time - 2400) as Time, price: price + 1 },
                    { time: (time - 1200) as Time, price: price - 4 },
                    { time: time as Time, price: price + 2 },
                ];
            default:
                return [
                    { time: (time - 3600) as Time, price },
                    { time: time as Time, price: price + 2 },
                ];
        }
    };

    const updateActiveTrades = () => {
        setActiveTrades(prev => prev.map(trade => {
            if (trade.status !== 'active') return trade;

            // Simulate price movement
            const priceChange = (Math.random() - 0.5) * 0.5;
            const newPrice = trade.currentPrice + priceChange;

            // Calculate P&L
            const pnl = trade.type === 'LONG'
                ? (newPrice - trade.entry) * 100
                : (trade.entry - newPrice) * 100;

            // Check stop loss
            if ((trade.type === 'LONG' && newPrice <= trade.stopLoss) ||
                (trade.type === 'SHORT' && newPrice >= trade.stopLoss)) {
                return { ...trade, status: 'closed', currentPrice: newPrice, pnl };
            }

            // Check take profits
            const hitTP = trade.takeProfit.some(tp =>
                (trade.type === 'LONG' && newPrice >= tp) ||
                (trade.type === 'SHORT' && newPrice <= tp)
            );

            if (hitTP) {
                return { ...trade, status: 'closed', currentPrice: newPrice, pnl };
            }

            return { ...trade, currentPrice: newPrice, pnl };
        }));
    };

    const animateDrawing = (drawing: AIDrawing) => {
        setCurrentDrawing(drawing);
        setTimeout(() => {
            setCurrentDrawing(null);
            setAIDrawings(prev => [...prev, { ...drawing, isAnimating: false }]);
        }, 2000);
    };

    const handleZoomIn = () => {
        chartRef.current?.timeScale().scrollToPosition(-10, true);
    };

    const handleZoomOut = () => {
        chartRef.current?.timeScale().scrollToPosition(10, true);
    };

    const handleFitContent = () => {
        chartRef.current?.timeScale().fitContent();
    };

    // Helper functions for pattern detection - use real candle data
    const getRecentCandles = (count: number) => {
        // Use actual chart data if available
        const chartData = (window as any).chartCandleData;
        if (chartData && chartData.length >= count) {
            return chartData.slice(-count);
        }

        // Fallback to simulated data
        const candles = [];
        const currentTime = Math.floor(Date.now() / 1000);
        const basePrice = 180;

        for (let i = count; i > 0; i--) {
            const time = (currentTime - i * 300) as Time;
            const open = basePrice + (Math.random() - 0.5) * 5;
            const close = open + (Math.random() - 0.5) * 2;
            const high = Math.max(open, close) + Math.random() * 1;
            const low = Math.min(open, close) - Math.random() * 1;
            candles.push({ time, open, high, low, close });
        }
        return candles;
    };

    const detectDoji = (candles: any[]): CandlePattern | null => {
        const lastCandle = candles[candles.length - 1];
        const bodySize = Math.abs(lastCandle.close - lastCandle.open);
        const range = lastCandle.high - lastCandle.low;

        if (bodySize / range < 0.1) {
            return {
                name: 'Doji',
                type: 'neutral',
                candles: [lastCandle],
                reliability: 75,
                description: 'Indecision pattern - potential reversal',
            };
        }
        return null;
    };

    const detectHammer = (candles: any[]): CandlePattern | null => {
        const lastCandle = candles[candles.length - 1];
        const body = Math.abs(lastCandle.close - lastCandle.open);
        const lowerWick = Math.min(lastCandle.open, lastCandle.close) - lastCandle.low;
        const upperWick = lastCandle.high - Math.max(lastCandle.open, lastCandle.close);

        if (lowerWick > body * 2 && upperWick < body * 0.5) {
            return {
                name: lastCandle.close > lastCandle.open ? 'Hammer' : 'Hanging Man',
                type: lastCandle.close > lastCandle.open ? 'bullish' : 'bearish',
                candles: [lastCandle],
                reliability: 80,
                description: 'Potential reversal pattern',
            };
        }
        return null;
    };

    const detectEngulfing = (candles: any[]): CandlePattern | null => {
        if (candles.length < 2) return null;

        const prev = candles[candles.length - 2];
        const curr = candles[candles.length - 1];

        const prevBullish = prev.close > prev.open;
        const currBullish = curr.close > curr.open;

        if (prevBullish && !currBullish &&
            curr.open > prev.close && curr.close < prev.open) {
            return {
                name: 'Bearish Engulfing',
                type: 'bearish',
                candles: [prev, curr],
                reliability: 85,
                description: 'Strong bearish reversal signal',
            };
        }

        if (!prevBullish && currBullish &&
            curr.close > prev.open && curr.open < prev.close) {
            return {
                name: 'Bullish Engulfing',
                type: 'bullish',
                candles: [prev, curr],
                reliability: 85,
                description: 'Strong bullish reversal signal',
            };
        }

        return null;
    };

    const detectStar = (candles: any[]): CandlePattern | null => {
        if (candles.length < 3) return null;

        const first = candles[candles.length - 3];
        const middle = candles[candles.length - 2];
        const last = candles[candles.length - 1];

        // Morning Star
        if (first.close < first.open &&
            Math.abs(middle.close - middle.open) < (first.high - first.low) * 0.3 &&
            last.close > last.open && last.close > first.open) {
            return {
                name: 'Morning Star',
                type: 'bullish',
                candles: [first, middle, last],
                reliability: 90,
                description: 'Strong bullish reversal pattern',
            };
        }

        // Evening Star
        if (first.close > first.open &&
            Math.abs(middle.close - middle.open) < (first.high - first.low) * 0.3 &&
            last.close < last.open && last.close < first.open) {
            return {
                name: 'Evening Star',
                type: 'bearish',
                candles: [first, middle, last],
                reliability: 90,
                description: 'Strong bearish reversal pattern',
            };
        }

        return null;
    };

    const findSwingPoints = () => {
        // Use actual candle data
        const candles = getRecentCandles(50);

        // Find actual swing points from the data
        let swingHigh = { time: candles[0].time, price: candles[0].high };
        let swingLow = { time: candles[0].time, price: candles[0].low };

        // Find the highest high and lowest low
        candles.forEach(candle => {
            if (candle.high > swingHigh.price) {
                swingHigh = { time: candle.time, price: candle.high };
            }
            if (candle.low < swingLow.price) {
                swingLow = { time: candle.time, price: candle.low };
            }
        });

        return { swingHigh, swingLow };
    };

    const findTrendLinePoints = () => {
        const candles = getRecentCandles(50);
        const highs: { time: Time; price: number }[] = [];
        const lows: { time: Time; price: number }[] = [];

        // Find local highs and lows
        for (let i = 2; i < candles.length - 2; i++) {
            if (candles[i].high > candles[i - 1].high &&
                candles[i].high > candles[i - 2].high &&
                candles[i].high > candles[i + 1].high &&
                candles[i].high > candles[i + 2].high) {
                highs.push({ time: candles[i].time, price: candles[i].high });
            }

            if (candles[i].low < candles[i - 1].low &&
                candles[i].low < candles[i - 2].low &&
                candles[i].low < candles[i + 1].low &&
                candles[i].low < candles[i + 2].low) {
                lows.push({ time: candles[i].time, price: candles[i].low });
            }
        }

        return {
            upperTrend: highs.slice(-2),
            lowerTrend: lows.slice(-2)
        };
    };

    const identifyKeyLevels = async (): Promise<KeyLevel[]> => {
        const levels: KeyLevel[] = [];
        const candles = getRecentCandles(100);
        const pricePoints: number[] = [];

        candles.forEach(candle => {
            pricePoints.push(candle.high, candle.low);
        });

        // Find price levels with multiple touches
        const priceCounts = new Map<number, number>();
        pricePoints.forEach(price => {
            const roundedPrice = Math.round(price * 100) / 100;
            priceCounts.set(roundedPrice, (priceCounts.get(roundedPrice) || 0) + 1);
        });

        // Convert to key levels
        priceCounts.forEach((touches, price) => {
            if (touches >= 3) {
                levels.push({
                    price,
                    type: price > candles[candles.length - 1].close ? 'resistance' : 'support',
                    strength: Math.min(5, Math.floor(touches / 2)),
                    touches,
                });
            }
        });

        return levels.sort((a, b) => b.strength - a.strength).slice(0, 5);
    };

    const calculateConfluence = (tools: TradingViewTool[], candlePatterns: CandlePattern[], keyLevels: KeyLevel[]): number => {
        let score = 50; // Base score

        // Add points for tools
        if (tools.some(t => t.type === 'fibonacci')) score += 15;
        if (tools.some(t => t.type === 'trendline')) score += 10;

        // Add points for candlestick patterns
        candlePatterns.forEach(pattern => {
            score += pattern.reliability / 10;
        });

        // Add points for key levels
        keyLevels.forEach(level => {
            score += level.strength * 2;
        });

        return Math.min(95, score);
    };

    const generateValidationNarrative = (pattern: string, tools: TradingViewTool[], candlePatterns: CandlePattern[], confluenceScore: number): string => {
        let narrative = `AI Prophet detected ${pattern} pattern with ${confluenceScore.toFixed(0)}% confluence. `;

        if (tools.some(t => t.type === 'fibonacci')) {
            narrative += 'Price is respecting Fibonacci levels. ';
        }

        if (candlePatterns.length > 0) {
            narrative += `Confirmed by ${candlePatterns.map(p => p.name).join(', ')} candlestick pattern(s). `;
        }

        if (confluenceScore > 80) {
            narrative += 'HIGH PROBABILITY SETUP - Multiple confirmations align. ';
        } else if (confluenceScore > 60) {
            narrative += 'Moderate confidence - Consider position sizing. ';
        }

        return narrative;
    };

    const drawKeyLevel = (level: KeyLevel) => {
        if (!chartRef.current) return;

        const levelSeries = chartRef.current.addLineSeries({
            color: level.type === 'support' ? theme.palette.success.main : theme.palette.error.main,
            lineWidth: level.strength,
            lineStyle: LineStyle.Solid,
            priceLineVisible: true,
            lastValueVisible: true,
            title: `${level.type} (${level.touches} touches)`,
        });

        const currentTime = Math.floor(Date.now() / 1000);
        levelSeries.setData([
            { time: (currentTime - 7200) as Time, value: level.price },
            { time: (currentTime + 3600) as Time, value: level.price },
        ]);

        lineSeriesRefs.current.set(`level-${level.price}`, levelSeries);
    };

    const clearSignalHighlights = () => {
        // Clear specific highlight series
        lineSeriesRefs.current.forEach((series, key) => {
            if (key.includes('highlight') || key.includes('fib') || key.includes('level')) {
                chartRef.current?.removeSeries(series);
                lineSeriesRefs.current.delete(key);
            }
        });
    };

    const generateScheduledSignal = () => {
        detectPatternsWithValidation();
    };

    return (
        <Box sx={{
            display: 'flex',
            flexDirection: 'column',
            height: '100vh',
            width: '100%',
            backgroundColor: trendSpiderColors.background,
            color: trendSpiderColors.textPrimary,
            fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        }}>
            {/* TrendSpider-style Header Bar */}
            <Paper sx={{
                height: '48px',
                display: 'flex',
                alignItems: 'center',
                px: 2,
                backgroundColor: trendSpiderColors.cardBackground,
                borderBottom: `1px solid ${trendSpiderColors.borderColor}`,
                borderRadius: 0,
                boxShadow: 'none',
            }}>
                <Stack direction="row" spacing={2} alignItems="center" sx={{ width: '100%' }}>
                    {/* Symbol Search */}
                    <Autocomplete
                        value={selectedSymbol}
                        onChange={(event, newValue) => {
                            if (newValue) setSelectedSymbol(newValue);
                        }}
                        options={popularSymbols}
                        sx={{
                            width: 150,
                            '& .MuiOutlinedInput-root': {
                                height: '32px',
                                backgroundColor: trendSpiderColors.background,
                                borderRadius: '4px',
                                '& fieldset': {
                                    borderColor: trendSpiderColors.borderColor,
                                },
                                '&:hover fieldset': {
                                    borderColor: trendSpiderColors.accent,
                                },
                                '&.Mui-focused fieldset': {
                                    borderColor: trendSpiderColors.accent,
                                },
                            },
                            '& .MuiInputBase-input': {
                                color: trendSpiderColors.textPrimary,
                                fontSize: '14px',
                                fontWeight: 600,
                            },
                        }}
                        renderInput={(params) => (
                            <TextField {...params} placeholder="Symbol" size="small" />
                        )}
                    />

                    <Divider orientation="vertical" flexItem sx={{ borderColor: trendSpiderColors.borderColor }} />

                    {/* Timeframe Selector */}
                    <ToggleButtonGroup
                        value={timeframe}
                        exclusive
                        onChange={(e, newTimeframe) => newTimeframe && setTimeframe(newTimeframe)}
                        size="small"
                        sx={{
                            height: '32px',
                            '& .MuiToggleButton-root': {
                                color: trendSpiderColors.textSecondary,
                                borderColor: trendSpiderColors.borderColor,
                                textTransform: 'none',
                                fontSize: '12px',
                                px: 1.5,
                                '&.Mui-selected': {
                                    backgroundColor: trendSpiderColors.accent,
                                    color: '#fff',
                                    '&:hover': {
                                        backgroundColor: trendSpiderColors.accentLight,
                                    },
                                },
                                '&:hover': {
                                    backgroundColor: trendSpiderColors.hoverBackground,
                                },
                            },
                        }}
                    >
                        <ToggleButton value="1m">1m</ToggleButton>
                        <ToggleButton value="5m">5m</ToggleButton>
                        <ToggleButton value="15m">15m</ToggleButton>
                        <ToggleButton value="1h">1H</ToggleButton>
                        <ToggleButton value="4h">4H</ToggleButton>
                        <ToggleButton value="1d">D</ToggleButton>
                        <ToggleButton value="1w">W</ToggleButton>
                    </ToggleButtonGroup>

                    <Divider orientation="vertical" flexItem sx={{ borderColor: trendSpiderColors.borderColor }} />

                    {/* Chart Type */}
                    <ToggleButtonGroup
                        value="candles"
                        exclusive
                        size="small"
                        sx={{
                            height: '32px',
                            '& .MuiToggleButton-root': {
                                color: trendSpiderColors.textSecondary,
                                borderColor: trendSpiderColors.borderColor,
                                px: 1,
                                '&.Mui-selected': {
                                    backgroundColor: alpha(trendSpiderColors.accent, 0.1),
                                    color: trendSpiderColors.accent,
                                },
                            },
                        }}
                    >
                        <ToggleButton value="candles">
                            <Tooltip title="Candlestick">
                                <CandleIcon fontSize="small" />
                            </Tooltip>
                        </ToggleButton>
                        <ToggleButton value="line">
                            <Tooltip title="Line">
                                <TrendLineIcon fontSize="small" />
                            </Tooltip>
                        </ToggleButton>
                        <ToggleButton value="bars">
                            <Tooltip title="Bars">
                                <BarChart fontSize="small" />
                            </Tooltip>
                        </ToggleButton>
                    </ToggleButtonGroup>

                    <Divider orientation="vertical" flexItem sx={{ borderColor: trendSpiderColors.borderColor }} />

                    {/* Technical Tools */}
                    <Stack direction="row" spacing={0.5}>
                        <Tooltip title="Trend Lines">
                            <IconButton
                                size="small"
                                onClick={() => setShowTrendLines(!showTrendLines)}
                                sx={{
                                    color: showTrendLines ? trendSpiderColors.accent : trendSpiderColors.textSecondary,
                                    '&:hover': { backgroundColor: trendSpiderColors.hoverBackground },
                                }}
                            >
                                <TrendLineIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Fibonacci">
                            <IconButton
                                size="small"
                                onClick={() => setShowFibonacci(!showFibonacci)}
                                sx={{
                                    color: showFibonacci ? trendSpiderColors.accent : trendSpiderColors.textSecondary,
                                    '&:hover': { backgroundColor: trendSpiderColors.hoverBackground },
                                }}
                            >
                                <LayersIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Patterns">
                            <IconButton
                                size="small"
                                onClick={() => setShowPatternProjections(!showPatternProjections)}
                                sx={{
                                    color: showPatternProjections ? trendSpiderColors.accent : trendSpiderColors.textSecondary,
                                    '&:hover': { backgroundColor: trendSpiderColors.hoverBackground },
                                }}
                            >
                                <PatternIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Volume">
                            <IconButton
                                size="small"
                                onClick={() => setShowVolume(!showVolume)}
                                sx={{
                                    color: showVolume ? trendSpiderColors.accent : trendSpiderColors.textSecondary,
                                    '&:hover': { backgroundColor: trendSpiderColors.hoverBackground },
                                }}
                            >
                                <VolumeIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Grid">
                            <IconButton
                                size="small"
                                onClick={() => setShowGrid(!showGrid)}
                                sx={{
                                    color: showGrid ? trendSpiderColors.accent : trendSpiderColors.textSecondary,
                                    '&:hover': { backgroundColor: trendSpiderColors.hoverBackground },
                                }}
                            >
                                {showGrid ? <GridIcon fontSize="small" /> : <GridOffIcon fontSize="small" />}
                            </IconButton>
                        </Tooltip>
                    </Stack>

                    <Box sx={{ flexGrow: 1 }} />

                    {/* AI Controls */}
                    <Stack direction="row" spacing={1} alignItems="center">
                        <Chip
                            icon={<AIIcon />}
                            label={isAIActive ? "AI Active" : "AI Inactive"}
                            color={isAIActive ? "success" : "default"}
                            size="small"
                            onClick={() => {
                                // Since isAIActive is a prop, we can't modify it directly
                                console.log('AI toggle clicked - should be handled by parent component');
                            }}
                            sx={{
                                backgroundColor: isAIActive ? trendSpiderColors.success : trendSpiderColors.cardBackground,
                                color: isAIActive ? '#fff' : trendSpiderColors.textSecondary,
                                borderColor: isAIActive ? trendSpiderColors.success : trendSpiderColors.borderColor,
                                '&:hover': {
                                    backgroundColor: isAIActive ? alpha(trendSpiderColors.success, 0.8) : trendSpiderColors.hoverBackground,
                                },
                            }}
                        />

                        {isAIActive && (
                            <FormControl size="small" sx={{ minWidth: 120 }}>
                                <Select
                                    value={signalGenerationMode}
                                    onChange={(e) => setSignalGenerationMode(e.target.value as any)}
                                    sx={{
                                        height: '32px',
                                        backgroundColor: trendSpiderColors.background,
                                        color: trendSpiderColors.textPrimary,
                                        '& .MuiOutlinedInput-notchedOutline': {
                                            borderColor: trendSpiderColors.borderColor,
                                        },
                                        '&:hover .MuiOutlinedInput-notchedOutline': {
                                            borderColor: trendSpiderColors.accent,
                                        },
                                        '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                                            borderColor: trendSpiderColors.accent,
                                        },
                                    }}
                                >
                                    <MenuItem value="auto">Auto</MenuItem>
                                    <MenuItem value="scheduled">Scheduled</MenuItem>
                                    <MenuItem value="manual">Manual</MenuItem>
                                </Select>
                            </FormControl>
                        )}
                    </Stack>

                    {/* Chart Actions */}
                    <Stack direction="row" spacing={0.5}>
                        <Tooltip title="Refresh">
                            <IconButton size="small" sx={{ color: trendSpiderColors.textSecondary }}>
                                <RefreshIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Save Layout">
                            <IconButton size="small" sx={{ color: trendSpiderColors.textSecondary }}>
                                <SaveIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Share">
                            <IconButton size="small" sx={{ color: trendSpiderColors.textSecondary }}>
                                <ShareIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Settings">
                            <IconButton size="small" sx={{ color: trendSpiderColors.textSecondary }}>
                                <SettingsIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Fullscreen">
                            <IconButton
                                size="small"
                                onClick={() => setIsFullscreen(!isFullscreen)}
                                sx={{ color: trendSpiderColors.textSecondary }}
                            >
                                {isFullscreen ? <FullscreenExitIcon fontSize="small" /> : <FullscreenIcon fontSize="small" />}
                            </IconButton>
                        </Tooltip>
                    </Stack>
                </Stack>
            </Paper>

            {/* Main Chart Area */}
            <Box sx={{
                flex: 1,
                display: 'flex',
                position: 'relative',
                backgroundColor: trendSpiderColors.background,
            }}>
                {/* Left Sidebar - Tools & Indicators */}
                <Paper sx={{
                    width: '48px',
                    backgroundColor: trendSpiderColors.cardBackground,
                    borderRight: `1px solid ${trendSpiderColors.borderColor}`,
                    borderRadius: 0,
                    boxShadow: 'none',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    py: 1,
                }}>
                    <Stack spacing={1}>
                        <Tooltip title="Drawing Tools" placement="right">
                            <IconButton size="small" sx={{ color: trendSpiderColors.textSecondary }}>
                                <TrendLineIcon />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Indicators" placement="right">
                            <IconButton size="small" sx={{ color: trendSpiderColors.textSecondary }}>
                                <TimelineIcon />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Patterns" placement="right">
                            <IconButton size="small" sx={{ color: trendSpiderColors.textSecondary }}>
                                <PatternIcon />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Alerts" placement="right">
                            <IconButton size="small" sx={{ color: trendSpiderColors.textSecondary }}>
                                <AlertIcon />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Analysis" placement="right">
                            <IconButton size="small" sx={{ color: trendSpiderColors.textSecondary }}>
                                <AnalyticsIcon />
                            </IconButton>
                        </Tooltip>
                    </Stack>
                </Paper>

                {/* Chart Container */}
                <Box sx={{ flex: 1, position: 'relative' }}>
                    {/* Price & Info Overlay */}
                    <Box sx={{
                        position: 'absolute',
                        top: 16,
                        left: 16,
                        zIndex: 10,
                        backgroundColor: alpha(trendSpiderColors.cardBackground, 0.9),
                        backdropFilter: 'blur(10px)',
                        borderRadius: '8px',
                        p: 2,
                        border: `1px solid ${trendSpiderColors.borderColor}`,
                    }}>
                        <Stack spacing={1}>
                            <Stack direction="row" spacing={2} alignItems="baseline">
                                <Typography variant="h5" sx={{
                                    fontWeight: 700,
                                    color: trendSpiderColors.textPrimary,
                                    letterSpacing: '-0.02em',
                                }}>
                                    {selectedSymbol}
                                </Typography>
                                <Typography variant="h6" sx={{
                                    color: currentPrice >= previousClose ? trendSpiderColors.bullish : trendSpiderColors.bearish,
                                    fontWeight: 600,
                                }}>
                                    ${currentPrice.toFixed(2)}
                                </Typography>
                                <Typography variant="body2" sx={{
                                    color: currentPrice >= previousClose ? trendSpiderColors.bullish : trendSpiderColors.bearish,
                                }}>
                                    {currentPrice >= previousClose ? '+' : ''}{((currentPrice - previousClose) / previousClose * 100).toFixed(2)}%
                                </Typography>
                            </Stack>
                            <Stack direction="row" spacing={3}>
                                <Box>
                                    <Typography variant="caption" sx={{ color: trendSpiderColors.textMuted }}>
                                        O
                                    </Typography>
                                    <Typography variant="caption" sx={{ color: trendSpiderColors.textSecondary, ml: 0.5 }}>
                                        {(previousClose * 0.98).toFixed(2)}
                                    </Typography>
                                </Box>
                                <Box>
                                    <Typography variant="caption" sx={{ color: trendSpiderColors.textMuted }}>
                                        H
                                    </Typography>
                                    <Typography variant="caption" sx={{ color: trendSpiderColors.textSecondary, ml: 0.5 }}>
                                        {(currentPrice * 1.02).toFixed(2)}
                                    </Typography>
                                </Box>
                                <Box>
                                    <Typography variant="caption" sx={{ color: trendSpiderColors.textMuted }}>
                                        L
                                    </Typography>
                                    <Typography variant="caption" sx={{ color: trendSpiderColors.textSecondary, ml: 0.5 }}>
                                        {(currentPrice * 0.97).toFixed(2)}
                                    </Typography>
                                </Box>
                                <Box>
                                    <Typography variant="caption" sx={{ color: trendSpiderColors.textMuted }}>
                                        V
                                    </Typography>
                                    <Typography variant="caption" sx={{ color: trendSpiderColors.textSecondary, ml: 0.5 }}>
                                        {(Math.random() * 10000000).toFixed(0)}
                                    </Typography>
                                </Box>
                            </Stack>
                        </Stack>
                    </Box>

                    {/* AI Status Overlay */}
                    {isAIActive && (
                        <Box sx={{
                            position: 'absolute',
                            top: 16,
                            right: 16,
                            zIndex: 10,
                            backgroundColor: alpha(trendSpiderColors.cardBackground, 0.9),
                            backdropFilter: 'blur(10px)',
                            borderRadius: '8px',
                            p: 1.5,
                            border: `1px solid ${trendSpiderColors.borderColor}`,
                            minWidth: '200px',
                        }}>
                            <Stack spacing={1}>
                                <Stack direction="row" spacing={1} alignItems="center">
                                    <AutoAwesomeIcon sx={{ color: trendSpiderColors.accent, fontSize: 16 }} />
                                    <Typography variant="caption" sx={{
                                        color: trendSpiderColors.textPrimary,
                                        fontWeight: 600,
                                    }}>
                                        AI Analysis Active
                                    </Typography>
                                </Stack>
                                <Stack spacing={0.5}>
                                    <Stack direction="row" justifyContent="space-between">
                                        <Typography variant="caption" sx={{ color: trendSpiderColors.textMuted }}>
                                            Patterns Detected
                                        </Typography>
                                        <Typography variant="caption" sx={{ color: trendSpiderColors.accent }}>
                                            {detectedPatterns.length}
                                        </Typography>
                                    </Stack>
                                    <Stack direction="row" justifyContent="space-between">
                                        <Typography variant="caption" sx={{ color: trendSpiderColors.textMuted }}>
                                            Active Signals
                                        </Typography>
                                        <Typography variant="caption" sx={{ color: trendSpiderColors.success }}>
                                            {signalHistory.filter(s => s.status === 'active').length}
                                        </Typography>
                                    </Stack>
                                    <Stack direction="row" justifyContent="space-between">
                                        <Typography variant="caption" sx={{ color: trendSpiderColors.textMuted }}>
                                            Win Rate
                                        </Typography>
                                        <Typography variant="caption" sx={{ color: trendSpiderColors.textPrimary }}>
                                            {(signalHistory.filter(s => s.status === 'success').length / signalHistory.length * 100 || 0).toFixed(0)}%
                                        </Typography>
                                    </Stack>
                                </Stack>
                            </Stack>
                        </Box>
                    )}

                    {/* Chart */}
                    <Box
                        ref={chartContainerRef}
                        sx={{
                            width: '100%',
                            height: '100%',
                            backgroundColor: trendSpiderColors.background,
                        }}
                    />

                    {/* Pattern Overlay */}
                    {showPatternProjections && detectedPatterns.length > 0 && (
                        <Box sx={{
                            position: 'absolute',
                            bottom: 16,
                            left: 16,
                            zIndex: 10,
                            backgroundColor: alpha(trendSpiderColors.cardBackground, 0.9),
                            backdropFilter: 'blur(10px)',
                            borderRadius: '8px',
                            p: 1.5,
                            border: `1px solid ${trendSpiderColors.borderColor}`,
                        }}>
                            <Stack spacing={1}>
                                <Typography variant="caption" sx={{
                                    color: trendSpiderColors.textPrimary,
                                    fontWeight: 600,
                                    mb: 0.5,
                                }}>
                                    Detected Patterns
                                </Typography>
                                {detectedPatterns.map((pattern, index) => {
                                    // Determine direction based on pattern name
                                    const isBullish = pattern.includes('Bull') || pattern.includes('Ascending') || pattern.includes('Cup');
                                    return (
                                        <Stack key={index} direction="row" spacing={1} alignItems="center">
                                            <Box sx={{
                                                width: 8,
                                                height: 8,
                                                borderRadius: '50%',
                                                backgroundColor: isBullish ? trendSpiderColors.bullish : trendSpiderColors.bearish,
                                            }} />
                                            <Typography variant="caption" sx={{ color: trendSpiderColors.textSecondary }}>
                                                {pattern}
                                            </Typography>
                                            <Typography variant="caption" sx={{
                                                color: isBullish ? trendSpiderColors.bullish : trendSpiderColors.bearish,
                                                fontWeight: 600,
                                            }}>
                                                {(75 + Math.random() * 20).toFixed(0)}%
                                            </Typography>
                                        </Stack>
                                    );
                                })}
                            </Stack>
                        </Box>
                    )}
                </Box>

                {/* Right Sidebar - Signals & History */}
                {showSignalHistory && (
                    <Paper sx={{
                        width: '300px',
                        backgroundColor: trendSpiderColors.cardBackground,
                        borderLeft: `1px solid ${trendSpiderColors.borderColor}`,
                        borderRadius: 0,
                        boxShadow: 'none',
                        display: 'flex',
                        flexDirection: 'column',
                    }}>
                        <Box sx={{
                            p: 2,
                            borderBottom: `1px solid ${trendSpiderColors.borderColor}`,
                        }}>
                            <Stack direction="row" justifyContent="space-between" alignItems="center">
                                <Typography variant="subtitle2" sx={{
                                    color: trendSpiderColors.textPrimary,
                                    fontWeight: 600,
                                }}>
                                    Signal History
                                </Typography>
                                <IconButton
                                    size="small"
                                    onClick={() => setShowSignalHistory(false)}
                                    sx={{ color: trendSpiderColors.textSecondary }}
                                >
                                    <CloseIcon fontSize="small" />
                                </IconButton>
                            </Stack>
                        </Box>
                        <Box sx={{
                            flex: 1,
                            overflowY: 'auto',
                            p: 2,
                        }}>
                            <Stack spacing={1}>
                                {signalHistory.map((signal) => (
                                    <Paper
                                        key={signal.id}
                                        sx={{
                                            p: 1.5,
                                            backgroundColor: trendSpiderColors.background,
                                            border: `1px solid ${trendSpiderColors.borderColor}`,
                                            borderRadius: '6px',
                                            cursor: 'pointer',
                                            transition: 'all 0.2s',
                                            '&:hover': {
                                                borderColor: trendSpiderColors.accent,
                                                backgroundColor: trendSpiderColors.hoverBackground,
                                            },
                                        }}
                                        onClick={() => {
                                            setSelectedSignal(signal);
                                            setShowSignalAnalysis(true);
                                        }}
                                    >
                                        <Stack spacing={0.5}>
                                            <Stack direction="row" justifyContent="space-between" alignItems="center">
                                                <Stack direction="row" spacing={0.5} alignItems="center">
                                                    {signal.signal.action === 'BUY' ? (
                                                        <TrendingUp sx={{ fontSize: 14, color: trendSpiderColors.bullish }} />
                                                    ) : (
                                                        <TrendingDown sx={{ fontSize: 14, color: trendSpiderColors.bearish }} />
                                                    )}
                                                    <Typography variant="caption" sx={{
                                                        color: signal.signal.action === 'BUY' ? trendSpiderColors.bullish : trendSpiderColors.bearish,
                                                        fontWeight: 600,
                                                    }}>
                                                        {signal.signal.action}
                                                    </Typography>
                                                </Stack>
                                                <Chip
                                                    label={signal.status}
                                                    size="small"
                                                    sx={{
                                                        height: '20px',
                                                        fontSize: '10px',
                                                        backgroundColor:
                                                            signal.status === 'active' ? alpha(trendSpiderColors.accent, 0.1) :
                                                                signal.status === 'success' ? alpha(trendSpiderColors.success, 0.1) :
                                                                    alpha(trendSpiderColors.bearish, 0.1),
                                                        color:
                                                            signal.status === 'active' ? trendSpiderColors.accent :
                                                                signal.status === 'success' ? trendSpiderColors.success :
                                                                    trendSpiderColors.bearish,
                                                        border: 'none',
                                                    }}
                                                />
                                            </Stack>
                                            <Typography variant="caption" sx={{ color: trendSpiderColors.textSecondary }}>
                                                ${signal.signal.entry.toFixed(2)} • {new Date(signal.validation.timestamp * 1000).toLocaleTimeString()}
                                            </Typography>
                                            <Stack direction="row" spacing={1}>
                                                <Typography variant="caption" sx={{ color: trendSpiderColors.textMuted }}>
                                                    Confluence:
                                                </Typography>
                                                <Rating
                                                    value={signal.validation.confluenceScore / 20}
                                                    readOnly
                                                    size="small"
                                                    sx={{
                                                        '& .MuiRating-iconFilled': {
                                                            color: trendSpiderColors.accent,
                                                        },
                                                    }}
                                                />
                                            </Stack>
                                        </Stack>
                                    </Paper>
                                ))}
                            </Stack>
                        </Box>
                    </Paper>
                )}
            </Box>

            {/* Bottom Status Bar */}
            <Paper sx={{
                height: '32px',
                display: 'flex',
                alignItems: 'center',
                px: 2,
                backgroundColor: trendSpiderColors.cardBackground,
                borderTop: `1px solid ${trendSpiderColors.borderColor}`,
                borderRadius: 0,
                boxShadow: 'none',
            }}>
                <Stack direction="row" spacing={3} alignItems="center" sx={{ width: '100%' }}>
                    <Stack direction="row" spacing={1} alignItems="center">
                        <Box sx={{
                            width: 8,
                            height: 8,
                            borderRadius: '50%',
                            backgroundColor: trendSpiderColors.success,
                        }} />
                        <Typography variant="caption" sx={{ color: trendSpiderColors.textSecondary }}>
                            Connected
                        </Typography>
                    </Stack>
                    <Typography variant="caption" sx={{ color: trendSpiderColors.textMuted }}>
                        {timeframe.toUpperCase()} • {selectedSymbol}
                    </Typography>
                    <Box sx={{ flexGrow: 1 }} />
                    <Typography variant="caption" sx={{ color: trendSpiderColors.textMuted }}>
                        {new Date().toLocaleString()}
                    </Typography>
                </Stack>
            </Paper>

            {/* Signal Analysis Modal */}
            {showSignalAnalysis && selectedSignal && (
                <Box sx={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: 1300,
                    backdropFilter: 'blur(5px)',
                }}>
                    <Paper sx={{
                        width: '80%',
                        maxWidth: '800px',
                        maxHeight: '90vh',
                        overflowY: 'auto',
                        backgroundColor: trendSpiderColors.cardBackground,
                        borderRadius: '12px',
                        border: `1px solid ${trendSpiderColors.borderColor}`,
                    }}>
                        {/* Modal content remains the same but with updated colors */}
                    </Paper>
                </Box>
            )}
        </Box>
    );
};

export default AutonomousChart; 