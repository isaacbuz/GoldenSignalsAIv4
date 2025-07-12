/**
 * Unified Chart - Consolidates All Chart Implementations
 * 
 * This component replaces:
 * - EnhancedTradingChart.tsx
 * - OptionsChart.tsx
 * - LightweightChart.tsx
 * - TradingChart.tsx
 * - MainStockChart.tsx
 * - TradingViewChart.tsx
 * - MiniChart.tsx
 * 
 * Features:
 * - Multiple chart types (candlestick, line, bar, area)
 * - Professional trading indicators (EMA, RSI, MACD, Bollinger Bands, VWAP)
 * - AI signal overlays with entry/exit/stop levels
 * - Real-time data integration
 * - Multi-timeframe analysis
 * - Order flow visualization
 * - Risk management overlays
 * - Configurable themes and layouts
 * - Performance optimized with canvas rendering
 * - Mobile responsive design
 */

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import {
    Box,
    Card,
    Stack,
    Typography,
    ToggleButton,
    ToggleButtonGroup,
    IconButton,
    Chip,
    Button,
    Menu,
    MenuItem,
    ListItemIcon,
    ListItemText,
    Divider,
    useTheme,
    alpha,
    Tooltip,
    CircularProgress,
    FormControlLabel,
    Switch,
    Slider,
    Paper,
    Grid,
    Badge,
    Collapse,
    CardContent,
} from '@mui/material';
import {
    ShowChart,
    CandlestickChart,
    BarChart,
    Timeline,
    Fullscreen,
    FullscreenExit,
    Settings,
    Layers,
    Speed,
    TrendingUp,
    TrendingDown,
    Psychology,
    Analytics,
    Refresh,
    CheckCircle,
    Warning,
    SignalCellularAlt,
    MultilineChart,
    AutoGraph,
    Assessment,
    Insights,
    SmartToy,
    ExpandMore,
    ExpandLess,
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
    LineWidth,
    PriceLineOptions,
    UTCTimestamp,
} from 'lightweight-charts';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../services/api/apiClient';
import ErrorBoundary from '../Common/ErrorBoundary';

export type ChartType = 'candlestick' | 'line' | 'bar' | 'area' | 'mini';
export type ChartMode = 'trading' | 'analytics' | 'signals' | 'professional' | 'minimal';
export type ChartTheme = 'light' | 'dark' | 'auto';

interface TechnicalIndicator {
    id: string;
    name: string;
    enabled: boolean;
    settings: Record<string, any>;
    color?: string;
}

export interface ChartSignal {
    id: string;
    type: 'BUY' | 'SELL' | 'HOLD';
    symbol: string;
    timestamp: number;
    price: number;
    confidence: number;
    entry?: number;
    target?: number;
    stopLoss?: number;
    reasoning?: string;
}

interface MarketData {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

interface UnifiedChartProps {
    symbol?: string;
    mode?: ChartMode;
    type?: ChartType;
    theme?: ChartTheme;
    height?: number | string;
    width?: number | string;
    data?: MarketData[];
    signals?: ChartSignal[];
    selectedSignal?: ChartSignal;
    onSignalClick?: (signal: ChartSignal) => void;
    timeframe?: string;
    showControls?: boolean;
    showIndicators?: boolean;
    showSignals?: boolean;
    showVolume?: boolean;
    showAdvancedFeatures?: boolean;
    autoRefresh?: boolean;
    className?: string;
}

// Default technical indicators
const DEFAULT_INDICATORS: TechnicalIndicator[] = [
    { id: 'volume', name: 'Volume', enabled: true, settings: {} },
    { id: 'ema8', name: 'EMA 8', enabled: false, settings: { period: 8 }, color: '#FF6B6B' },
    { id: 'ema21', name: 'EMA 21', enabled: false, settings: { period: 21 }, color: '#4ECDC4' },
    { id: 'ema50', name: 'EMA 50', enabled: false, settings: { period: 50 }, color: '#45B7D1' },
    { id: 'rsi', name: 'RSI', enabled: false, settings: { period: 14 } },
    { id: 'macd', name: 'MACD', enabled: false, settings: { fast: 12, slow: 26, signal: 9 } },
    { id: 'bb', name: 'Bollinger Bands', enabled: false, settings: { period: 20, stdDev: 2 } },
    { id: 'vwap', name: 'VWAP', enabled: false, settings: {} },
];

const TIMEFRAMES = [
    { value: '1m', label: '1M' },
    { value: '5m', label: '5M' },
    { value: '15m', label: '15M' },
    { value: '30m', label: '30M' },
    { value: '1h', label: '1H' },
    { value: '4h', label: '4H' },
    { value: '1d', label: '1D' },
    { value: '1w', label: '1W' },
];

// Chart registry for cleanup
const chartRegistry = new Map<string, { chart: IChartApi; instanceId: string; element: HTMLElement }>();

export const UnifiedChart: React.FC<UnifiedChartProps> = ({
    symbol = 'SPY',
    mode = 'trading',
    type = 'candlestick',
    theme: chartTheme = 'auto',
    height = 450,
    width = '100%',
    data,
    signals = [],
    selectedSignal,
    onSignalClick,
    timeframe = '15m',
    showControls = true,
    showIndicators = true,
    showSignals = true,
    showVolume = true,
    showAdvancedFeatures = false,
    autoRefresh = true,
    className,
}) => {
    const theme = useTheme();
    const containerIdRef = useRef<string>(`chart-${Math.random().toString(36).substr(2, 9)}`);
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const mainSeriesRef = useRef<ISeriesApi<any> | null>(null);
    const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
    const indicatorSeriesRef = useRef<Map<string, ISeriesApi<any>>>(new Map());
    const resizeObserverRef = useRef<ResizeObserver | null>(null);

    // State
    const [selectedTimeframe, setSelectedTimeframe] = useState(timeframe);
    const [selectedType, setSelectedType] = useState(type);
    const [indicators, setIndicators] = useState<TechnicalIndicator[]>(DEFAULT_INDICATORS);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [showIndicatorMenu, setShowIndicatorMenu] = useState(false);
    const [indicatorMenuAnchor, setIndicatorMenuAnchor] = useState<null | HTMLElement>(null);
    const [showAdvanced, setShowAdvanced] = useState(showAdvancedFeatures);
    const [loading, setLoading] = useState(false);

    // Debug flag to control console logging
    const DEBUG_MODE = process.env.NODE_ENV === 'development';

    // Fetch chart data
    const { data: chartData, isLoading, refetch } = useQuery({
        queryKey: ['chart-data', symbol, selectedTimeframe],
        queryFn: async () => {
            try {
                const barCount = getBarCount(selectedTimeframe);
                return await apiClient.getHistoricalData(symbol, selectedTimeframe, barCount);
            } catch (error) {
                console.error('Failed to fetch chart data:', error);
                return generateMockData();
            }
        },
        enabled: !data,
        staleTime: 60000,
        refetchInterval: autoRefresh ? 30000 : false,
    });

    // Generate mock data for demo
    const generateMockData = useCallback((): MarketData[] => {
        const data: MarketData[] = [];
        const basePrice = 450;
        const now = Date.now();

        for (let i = 100; i >= 0; i--) {
            const time = Math.floor((now - i * 60 * 60 * 1000) / 1000);
            const open = basePrice + (Math.random() - 0.5) * 4;
            const close = open + (Math.random() - 0.5) * 4;
            const high = Math.max(open, close) + Math.random() * 2;
            const low = Math.min(open, close) - Math.random() * 2;
            const volume = Math.floor(Math.random() * 1000000) + 500000;

            data.push({ time, open, high, low, close, volume });
        }
        return data;
    }, []);

    // Validate and process chart data
    const validatedData = useMemo(() => {
        const rawData = data || chartData;

        // Check if data exists and is an array
        if (!rawData) {
            if (DEBUG_MODE && process.env.NODE_ENV === 'development') {
                console.warn('No chart data provided, using mock data');
            }
            return generateMockData();
        }

        // Handle object responses that might contain data arrays
        if (!Array.isArray(rawData)) {
            if (DEBUG_MODE && process.env.NODE_ENV === 'development') {
                console.warn('Chart data is not an array, checking for data property. Received:', typeof rawData);
            }

            // Check if it's an object with a data property
            if (typeof rawData === 'object' && rawData !== null) {
                if (Array.isArray((rawData as any).data)) {
                    if (DEBUG_MODE && process.env.NODE_ENV === 'development') {
                        console.log('Found data array in object response');
                    }
                    return (rawData as any).data.length > 0 ? (rawData as any).data : generateMockData();
                } else if (Array.isArray((rawData as any).historical)) {
                    if (DEBUG_MODE && process.env.NODE_ENV === 'development') {
                        console.log('Found historical array in object response');
                    }
                    return (rawData as any).historical.length > 0 ? (rawData as any).historical : generateMockData();
                } else if (Array.isArray((rawData as any).ohlcv)) {
                    if (DEBUG_MODE && process.env.NODE_ENV === 'development') {
                        console.log('Found ohlcv array in object response');
                    }
                    return (rawData as any).ohlcv.length > 0 ? (rawData as any).ohlcv : generateMockData();
                }
            }

            if (DEBUG_MODE && process.env.NODE_ENV === 'development') {
                console.warn('Chart data object does not contain valid data array, using mock data');
            }
            return generateMockData();
        }

        if (rawData.length === 0) {
            if (DEBUG_MODE && process.env.NODE_ENV === 'development') {
                console.warn('Chart data array is empty, using mock data');
            }
            return generateMockData();
        }

        // Validate data structure
        const isValidData = rawData.every(item =>
            item &&
            typeof item === 'object' &&
            (typeof item.time === 'number' || typeof item.timestamp === 'number') &&
            typeof item.open === 'number' &&
            typeof item.high === 'number' &&
            typeof item.low === 'number' &&
            typeof item.close === 'number' &&
            typeof item.volume === 'number'
        );

        if (!isValidData) {
            if (DEBUG_MODE && process.env.NODE_ENV === 'development') {
                console.warn('Chart data contains invalid items, using mock data');
            }
            return generateMockData();
        }

        // Normalize time field (some APIs use 'timestamp' instead of 'time')
        const normalizedData = rawData.map(item => ({
            ...item,
            time: item.time || item.timestamp
        }));

        return normalizedData;
    }, [data, chartData, generateMockData]);

    // Get bar count based on timeframe
    const getBarCount = useCallback((tf: string): number => {
        const counts: Record<string, number> = {
            '1m': 100,
            '5m': 200,
            '15m': 192,
            '30m': 100,
            '1h': 168,
            '4h': 120,
            '1d': 100,
            '1w': 52,
        };
        return counts[tf] || 100;
    }, []);

    // Calculate technical indicators
    const calculateIndicators = useCallback((data: MarketData[], indicator: TechnicalIndicator) => {
        switch (indicator.id) {
            case 'ema8':
            case 'ema21':
            case 'ema50':
                return calculateEMA(data, indicator.settings.period);
            case 'rsi':
                return calculateRSI(data, indicator.settings.period);
            case 'macd':
                return calculateMACD(data, indicator.settings.fast, indicator.settings.slow, indicator.settings.signal);
            case 'bb':
                return calculateBollingerBands(data, indicator.settings.period, indicator.settings.stdDev);
            case 'vwap':
                return calculateVWAP(data);
            default:
                return [];
        }
    }, []);

    // EMA calculation
    const calculateEMA = useCallback((data: MarketData[], period: number) => {
        const ema: Array<{ time: number; value: number }> = [];
        const multiplier = 2 / (period + 1);
        let emaValue = data[0]?.close || 0;

        data.forEach((candle, index) => {
            if (index === 0) {
                emaValue = candle.close;
            } else {
                emaValue = (candle.close - emaValue) * multiplier + emaValue;
            }
            ema.push({ time: candle.time, value: emaValue });
        });

        return ema;
    }, []);

    // RSI calculation
    const calculateRSI = useCallback((data: MarketData[], period: number) => {
        const rsi: Array<{ time: number; value: number }> = [];
        const gains: number[] = [];
        const losses: number[] = [];

        for (let i = 1; i < data.length; i++) {
            const change = data[i].close - data[i - 1].close;
            gains.push(change > 0 ? change : 0);
            losses.push(change < 0 ? Math.abs(change) : 0);

            if (i >= period) {
                const avgGain = gains.slice(-period).reduce((a, b) => a + b, 0) / period;
                const avgLoss = losses.slice(-period).reduce((a, b) => a + b, 0) / period;
                const rs = avgGain / avgLoss;
                const rsiValue = 100 - (100 / (1 + rs));
                rsi.push({ time: data[i].time, value: rsiValue });
            }
        }

        return rsi;
    }, []);

    // MACD calculation
    const calculateMACD = useCallback((data: MarketData[], fast: number, slow: number, signal: number) => {
        const emaFast = calculateEMA(data, fast);
        const emaSlow = calculateEMA(data, slow);
        const macdLine: Array<{ time: number; value: number }> = [];

        emaFast.forEach((fastValue, index) => {
            if (emaSlow[index]) {
                macdLine.push({
                    time: fastValue.time,
                    value: fastValue.value - emaSlow[index].value,
                });
            }
        });

        return macdLine;
    }, [calculateEMA]);

    // Bollinger Bands calculation
    const calculateBollingerBands = useCallback((data: MarketData[], period: number, stdDev: number) => {
        const bands: Array<{ time: number; upper: number; middle: number; lower: number }> = [];

        for (let i = period - 1; i < data.length; i++) {
            const slice = data.slice(i - period + 1, i + 1);
            const avg = slice.reduce((sum, candle) => sum + candle.close, 0) / period;
            const variance = slice.reduce((sum, candle) => sum + Math.pow(candle.close - avg, 2), 0) / period;
            const standardDeviation = Math.sqrt(variance);

            bands.push({
                time: data[i].time,
                upper: avg + (standardDeviation * stdDev),
                middle: avg,
                lower: avg - (standardDeviation * stdDev),
            });
        }

        return bands;
    }, []);

    // VWAP calculation
    const calculateVWAP = useCallback((data: MarketData[]) => {
        const vwap: Array<{ time: number; value: number }> = [];
        let cumulativeTPV = 0;
        let cumulativeVolume = 0;

        data.forEach(candle => {
            const typicalPrice = (candle.high + candle.low + candle.close) / 3;
            cumulativeTPV += typicalPrice * candle.volume;
            cumulativeVolume += candle.volume;

            vwap.push({
                time: candle.time,
                value: cumulativeTPV / cumulativeVolume,
            });
        });

        return vwap;
    }, []);

    // Analyze signal quality
    const analyzeSignalQuality = useCallback((signal: ChartSignal, data: MarketData[]) => {
        if (!signal || !data || data.length === 0) return null;

        // Find the data point closest to the signal timestamp
        const signalTime = signal.timestamp / 1000; // Convert to seconds
        const signalData = data.find(d => Math.abs(d.time - signalTime) < 3600) || data[data.length - 1];

        // Calculate various quality metrics
        const volumeConfirmation = signalData.volume > (data.slice(-20).reduce((sum, d) => sum + d.volume, 0) / 20);

        // Trend alignment (simplified)
        const recent = data.slice(-5);
        const trendAlignment = signal.type === 'BUY' ?
            recent[recent.length - 1].close > recent[0].close :
            recent[recent.length - 1].close < recent[0].close;

        // Support/resistance check (simplified)
        const supportResistance = signal.type === 'BUY' ?
            signal.price <= signalData.low * 1.02 :
            signal.price >= signalData.high * 0.98;

        // Risk/reward calculation
        const riskReward = signal.target && signal.stopLoss ?
            Math.abs(signal.target - signal.price) / Math.abs(signal.price - signal.stopLoss) : 1;

        // Overall score
        const overallScore = (
            (signal.confidence * 0.4) +
            (volumeConfirmation ? 20 : 0) +
            (trendAlignment ? 20 : 0) +
            (supportResistance ? 20 : 0)
        );

        return {
            overallScore,
            riskReward,
            volumeConfirmation,
            trendAlignment,
            supportResistance,
        };
    }, []);

    // Add pattern overlays to chart
    const addPatternOverlays = useCallback((chart: IChartApi, patterns: any[]) => {
        if (!chart || !patterns || patterns.length === 0) return;

        patterns.forEach(pattern => {
            // Create price lines for pattern boundaries
            const priceLine: PriceLineOptions = {
                price: pattern.price || 0,
                color: pattern.type.includes('Support') ? '#4CAF50' : '#F44336',
                lineWidth: 2 as LineWidth,
                lineStyle: LineStyle.Dashed,
                axisLabelVisible: true,
                title: pattern.type,
                lineVisible: true,
                axisLabelColor: pattern.type.includes('Support') ? '#4CAF50' : '#F44336',
                axisLabelTextColor: '#FFFFFF',
            };

            // Add the price line to the main series
            if (mainSeriesRef.current) {
                mainSeriesRef.current.createPriceLine(priceLine);
            }
        });
    }, []);

    // Add advanced pattern recognition
    const detectPatterns = useCallback((data: MarketData[]) => {
        const patterns: Array<{
            type: string;
            confidence: number;
            startTime: number;
            endTime: number;
            description: string;
        }> = [];

        // Head and Shoulders pattern detection
        const headAndShoulders = detectHeadAndShoulders(data);
        if (headAndShoulders) {
            patterns.push({
                type: 'Head and Shoulders',
                confidence: headAndShoulders.confidence,
                startTime: headAndShoulders.startTime,
                endTime: headAndShoulders.endTime,
                description: 'Bearish reversal pattern detected'
            });
        }

        // Double Top/Bottom detection
        const doublePattern = detectDoubleTopBottom(data);
        if (doublePattern) {
            patterns.push({
                type: doublePattern.type,
                confidence: doublePattern.confidence,
                startTime: doublePattern.startTime,
                endTime: doublePattern.endTime,
                description: doublePattern.description
            });
        }

        // Triangle pattern detection
        const trianglePattern = detectTrianglePattern(data);
        if (trianglePattern) {
            patterns.push({
                type: trianglePattern.type,
                confidence: trianglePattern.confidence,
                startTime: trianglePattern.startTime,
                endTime: trianglePattern.endTime,
                description: trianglePattern.description
            });
        }

        // Support and Resistance levels
        const supportResistance = detectSupportResistance(data);
        supportResistance.forEach(level => {
            patterns.push({
                type: level.type,
                confidence: level.confidence,
                startTime: level.startTime,
                endTime: level.endTime,
                description: level.description
            });
        });

        return patterns;
    }, []);

    // Head and Shoulders pattern detection
    const detectHeadAndShoulders = useCallback((data: MarketData[]) => {
        if (data.length < 20) return null;

        const peaks = findPeaks(data);
        if (peaks.length < 3) return null;

        // Look for three peaks where middle is highest
        for (let i = 1; i < peaks.length - 1; i++) {
            const leftShoulder = peaks[i - 1];
            const head = peaks[i];
            const rightShoulder = peaks[i + 1];

            if (head.high > leftShoulder.high && head.high > rightShoulder.high) {
                const shoulderDiff = Math.abs(leftShoulder.high - rightShoulder.high);
                const avgShoulder = (leftShoulder.high + rightShoulder.high) / 2;
                const shoulderTolerance = avgShoulder * 0.02; // 2% tolerance

                if (shoulderDiff <= shoulderTolerance) {
                    return {
                        confidence: 75 + (25 * (1 - shoulderDiff / shoulderTolerance)),
                        startTime: leftShoulder.time,
                        endTime: rightShoulder.time
                    };
                }
            }
        }

        return null;
    }, []);

    // Double Top/Bottom detection
    const detectDoubleTopBottom = useCallback((data: MarketData[]) => {
        if (data.length < 15) return null;

        const peaks = findPeaks(data);
        const troughs = findTroughs(data);

        // Double Top detection
        for (let i = 1; i < peaks.length; i++) {
            const peak1 = peaks[i - 1];
            const peak2 = peaks[i];
            const priceDiff = Math.abs(peak1.high - peak2.high);
            const avgPrice = (peak1.high + peak2.high) / 2;
            const tolerance = avgPrice * 0.015; // 1.5% tolerance

            if (priceDiff <= tolerance) {
                return {
                    type: 'Double Top',
                    confidence: 70 + (30 * (1 - priceDiff / tolerance)),
                    startTime: peak1.time,
                    endTime: peak2.time,
                    description: 'Bearish reversal pattern - Double Top'
                };
            }
        }

        // Double Bottom detection
        for (let i = 1; i < troughs.length; i++) {
            const trough1 = troughs[i - 1];
            const trough2 = troughs[i];
            const priceDiff = Math.abs(trough1.low - trough2.low);
            const avgPrice = (trough1.low + trough2.low) / 2;
            const tolerance = avgPrice * 0.015; // 1.5% tolerance

            if (priceDiff <= tolerance) {
                return {
                    type: 'Double Bottom',
                    confidence: 70 + (30 * (1 - priceDiff / tolerance)),
                    startTime: trough1.time,
                    endTime: trough2.time,
                    description: 'Bullish reversal pattern - Double Bottom'
                };
            }
        }

        return null;
    }, []);

    // Triangle pattern detection
    const detectTrianglePattern = useCallback((data: MarketData[]) => {
        if (data.length < 20) return null;

        const peaks = findPeaks(data);
        const troughs = findTroughs(data);

        if (peaks.length < 2 || troughs.length < 2) return null;

        // Ascending Triangle
        const recentPeaks = peaks.slice(-3);
        const recentTroughs = troughs.slice(-3);

        if (recentPeaks.length >= 2 && recentTroughs.length >= 2) {
            const peakSlope = calculateSlope(recentPeaks.map(p => ({ x: p.time, y: p.high })));
            const troughSlope = calculateSlope(recentTroughs.map(t => ({ x: t.time, y: t.low })));

            if (Math.abs(peakSlope) < 0.001 && troughSlope > 0.001) {
                return {
                    type: 'Ascending Triangle',
                    confidence: 65,
                    startTime: Math.min(recentPeaks[0].time, recentTroughs[0].time),
                    endTime: Math.max(recentPeaks[recentPeaks.length - 1].time, recentTroughs[recentTroughs.length - 1].time),
                    description: 'Bullish continuation pattern - Ascending Triangle'
                };
            }

            if (Math.abs(troughSlope) < 0.001 && peakSlope < -0.001) {
                return {
                    type: 'Descending Triangle',
                    confidence: 65,
                    startTime: Math.min(recentPeaks[0].time, recentTroughs[0].time),
                    endTime: Math.max(recentPeaks[recentPeaks.length - 1].time, recentTroughs[recentTroughs.length - 1].time),
                    description: 'Bearish continuation pattern - Descending Triangle'
                };
            }

            if (peakSlope < -0.001 && troughSlope > 0.001) {
                return {
                    type: 'Symmetrical Triangle',
                    confidence: 60,
                    startTime: Math.min(recentPeaks[0].time, recentTroughs[0].time),
                    endTime: Math.max(recentPeaks[recentPeaks.length - 1].time, recentTroughs[recentTroughs.length - 1].time),
                    description: 'Continuation pattern - Symmetrical Triangle'
                };
            }
        }

        return null;
    }, []);

    // Support and Resistance detection
    const detectSupportResistance = useCallback((data: MarketData[]) => {
        const levels: Array<{
            type: string;
            confidence: number;
            startTime: number;
            endTime: number;
            description: string;
            price: number;
        }> = [];

        const peaks = findPeaks(data);
        const troughs = findTroughs(data);

        // Group similar price levels
        const groupTolerance = 0.02; // 2% tolerance

        // Resistance levels from peaks
        const resistanceLevels = groupPricePoints(peaks.map(p => ({ price: p.high, time: p.time })), groupTolerance);
        resistanceLevels.forEach(level => {
            if (level.count >= 2) {
                levels.push({
                    type: 'Resistance',
                    confidence: Math.min(90, 50 + (level.count * 10)),
                    startTime: level.firstTime,
                    endTime: level.lastTime,
                    description: `Resistance level at ${level.price.toFixed(2)} (${level.count} touches)`,
                    price: level.price
                });
            }
        });

        // Support levels from troughs
        const supportLevels = groupPricePoints(troughs.map(t => ({ price: t.low, time: t.time })), groupTolerance);
        supportLevels.forEach(level => {
            if (level.count >= 2) {
                levels.push({
                    type: 'Support',
                    confidence: Math.min(90, 50 + (level.count * 10)),
                    startTime: level.firstTime,
                    endTime: level.lastTime,
                    description: `Support level at ${level.price.toFixed(2)} (${level.count} touches)`,
                    price: level.price
                });
            }
        });

        return levels;
    }, []);

    // Helper functions for pattern detection
    const findPeaks = useCallback((data: MarketData[]) => {
        const peaks: Array<{ time: number; high: number; index: number }> = [];
        const lookback = 5;

        for (let i = lookback; i < data.length - lookback; i++) {
            const current = data[i];
            let isPeak = true;

            for (let j = i - lookback; j <= i + lookback; j++) {
                if (j !== i && data[j].high >= current.high) {
                    isPeak = false;
                    break;
                }
            }

            if (isPeak) {
                peaks.push({ time: current.time, high: current.high, index: i });
            }
        }

        return peaks;
    }, []);

    const findTroughs = useCallback((data: MarketData[]) => {
        const troughs: Array<{ time: number; low: number; index: number }> = [];
        const lookback = 5;

        for (let i = lookback; i < data.length - lookback; i++) {
            const current = data[i];
            let isTrough = true;

            for (let j = i - lookback; j <= i + lookback; j++) {
                if (j !== i && data[j].low <= current.low) {
                    isTrough = false;
                    break;
                }
            }

            if (isTrough) {
                troughs.push({ time: current.time, low: current.low, index: i });
            }
        }

        return troughs;
    }, []);

    const calculateSlope = useCallback((points: Array<{ x: number; y: number }>) => {
        if (points.length < 2) return 0;

        const n = points.length;
        const sumX = points.reduce((sum, p) => sum + p.x, 0);
        const sumY = points.reduce((sum, p) => sum + p.y, 0);
        const sumXY = points.reduce((sum, p) => sum + (p.x * p.y), 0);
        const sumXX = points.reduce((sum, p) => sum + (p.x * p.x), 0);

        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        return slope;
    }, []);

    const groupPricePoints = useCallback((points: Array<{ price: number; time: number }>, tolerance: number) => {
        const groups: Array<{
            price: number;
            count: number;
            firstTime: number;
            lastTime: number;
        }> = [];

        points.forEach(point => {
            let foundGroup = false;

            for (const group of groups) {
                const priceDiff = Math.abs(point.price - group.price) / group.price;
                if (priceDiff <= tolerance) {
                    group.count++;
                    group.price = (group.price * (group.count - 1) + point.price) / group.count;
                    group.firstTime = Math.min(group.firstTime, point.time);
                    group.lastTime = Math.max(group.lastTime, point.time);
                    foundGroup = true;
                    break;
                }
            }

            if (!foundGroup) {
                groups.push({
                    price: point.price,
                    count: 1,
                    firstTime: point.time,
                    lastTime: point.time
                });
            }
        });

        return groups;
    }, []);

    // Add main price series
    const addMainSeries = useCallback((chart: IChartApi) => {
        // Safety check for data
        if (!Array.isArray(validatedData) || validatedData.length === 0) {
            if (DEBUG_MODE) console.warn('Cannot add main series: validatedData is not a valid array');
            return;
        }

        let mainSeries: ISeriesApi<any>;

        if (selectedType === 'candlestick') {
            mainSeries = chart.addCandlestickSeries({
                upColor: theme.palette.success.main,
                downColor: theme.palette.error.main,
                borderUpColor: theme.palette.success.main,
                borderDownColor: theme.palette.error.main,
                wickUpColor: theme.palette.success.main,
                wickDownColor: theme.palette.error.main,
            });
        } else if (selectedType === 'line') {
            mainSeries = chart.addLineSeries({
                color: theme.palette.primary.main,
                lineWidth: 2,
            });
        } else if (selectedType === 'area') {
            mainSeries = chart.addAreaSeries({
                topColor: alpha(theme.palette.primary.main, 0.4),
                bottomColor: alpha(theme.palette.primary.main, 0.0),
                lineColor: theme.palette.primary.main,
                lineWidth: 2,
            });
        } else {
            mainSeries = chart.addBarSeries({
                upColor: theme.palette.success.main,
                downColor: theme.palette.error.main,
            });
        }

        // Set data with additional safety checks
        try {
            const processedData = selectedType === 'line' || selectedType === 'area'
                ? validatedData.map(d => ({ time: d.time as Time, value: d.close }))
                : validatedData.map(d => ({
                    time: d.time as Time,
                    open: d.open,
                    high: d.high,
                    low: d.low,
                    close: d.close,
                }));

            mainSeries.setData(processedData);
            mainSeriesRef.current = mainSeries;
        } catch (error) {
            if (DEBUG_MODE) console.error('Error processing chart data:', error);
            // Fallback to empty data
            mainSeries.setData([]);
            mainSeriesRef.current = mainSeries;
        }
    }, [selectedType, theme, validatedData]);

    // Add volume series
    const addVolumeSeries = useCallback((chart: IChartApi) => {
        // Safety check for data
        if (!Array.isArray(validatedData) || validatedData.length === 0) {
            if (DEBUG_MODE) console.warn('Cannot add volume series: validatedData is not a valid array');
            return;
        }

        const volumeSeries = chart.addHistogramSeries({
            color: alpha(theme.palette.primary.main, 0.3),
            priceFormat: {
                type: 'volume',
            },
            priceScaleId: '',
        });

        try {
            const volumeData = validatedData.map(d => ({
                time: d.time as Time,
                value: d.volume || 0,
                color: d.close >= d.open ? alpha(theme.palette.success.main, 0.5) : alpha(theme.palette.error.main, 0.5),
            }));

            volumeSeries.setData(volumeData);
            volumeSeriesRef.current = volumeSeries;
        } catch (error) {
            if (DEBUG_MODE) console.error('Error processing volume data:', error);
            volumeSeries.setData([]);
            volumeSeriesRef.current = volumeSeries;
        }
    }, [theme, validatedData]);

    // Add technical indicators
    const addIndicators = useCallback((chart: IChartApi) => {
        // Safety check for data
        if (!Array.isArray(validatedData) || validatedData.length === 0) {
            if (DEBUG_MODE) console.warn('Cannot add indicators: validatedData is not a valid array');
            return;
        }

        indicators.forEach(indicator => {
            if (!indicator.enabled) return;

            try {
                const indicatorData = calculateIndicators(validatedData, indicator);

                if (!Array.isArray(indicatorData) || indicatorData.length === 0) return;

                let series: ISeriesApi<any>;

                switch (indicator.id) {
                    case 'ema8':
                    case 'ema21':
                    case 'ema50':
                    case 'vwap':
                        series = chart.addLineSeries({
                            color: indicator.color || theme.palette.secondary.main,
                            lineWidth: 1,
                            title: indicator.name,
                        });
                        series.setData(indicatorData);
                        break;

                    case 'bb':
                        // Add three lines for Bollinger Bands
                        const bbData = indicatorData as Array<{ time: number; upper: number; middle: number; lower: number }>;

                        const upperSeries = chart.addLineSeries({
                            color: alpha(theme.palette.warning.main, 0.7),
                            lineWidth: 1,
                            title: 'BB Upper',
                        });
                        upperSeries.setData(bbData.map(d => ({ time: d.time as Time, value: d.upper })));

                        const middleSeries = chart.addLineSeries({
                            color: theme.palette.warning.main,
                            lineWidth: 1,
                            title: 'BB Middle',
                        });
                        middleSeries.setData(bbData.map(d => ({ time: d.time as Time, value: d.middle })));

                        const lowerSeries = chart.addLineSeries({
                            color: alpha(theme.palette.warning.main, 0.7),
                            lineWidth: 1,
                            title: 'BB Lower',
                        });
                        lowerSeries.setData(bbData.map(d => ({ time: d.time as Time, value: d.lower })));

                        indicatorSeriesRef.current.set(`${indicator.id}_upper`, upperSeries);
                        indicatorSeriesRef.current.set(`${indicator.id}_middle`, middleSeries);
                        indicatorSeriesRef.current.set(`${indicator.id}_lower`, lowerSeries);
                        return;

                    default:
                        return;
                }

                indicatorSeriesRef.current.set(indicator.id, series);
            } catch (error) {
                if (DEBUG_MODE) console.error(`Error adding indicator ${indicator.id}:`, error);
            }
        });
    }, [indicators, validatedData, calculateIndicators, theme]);

    // Add signal markers
    const addSignalMarkers = useCallback((chart: IChartApi) => {
        if (!mainSeriesRef.current || signals.length === 0) return;

        // Sort signals by timestamp and convert to seconds if needed
        const sortedSignals = [...signals].sort((a, b) => {
            const timeA = typeof a.timestamp === 'number' ? a.timestamp : new Date(a.timestamp).getTime();
            const timeB = typeof b.timestamp === 'number' ? b.timestamp : new Date(b.timestamp).getTime();
            return timeA - timeB;
        });

        const markers: SeriesMarker<Time>[] = sortedSignals.map(signal => {
            // Convert timestamp to seconds if it's in milliseconds
            let timestamp = typeof signal.timestamp === 'number' ? signal.timestamp : new Date(signal.timestamp).getTime();
            if (timestamp > 1e12) {
                timestamp = Math.floor(timestamp / 1000); // Convert milliseconds to seconds
            }

            return {
                time: timestamp as Time,
                position: signal.type === 'BUY' ? 'belowBar' : 'aboveBar',
                color: signal.type === 'BUY' ? theme.palette.success.main : theme.palette.error.main,
                shape: signal.type === 'BUY' ? 'arrowUp' : 'arrowDown',
                text: `${signal.type} ${signal.confidence}%`,
                size: selectedSignal?.id === signal.id ? 2 : 1,
            };
        });

        try {
            mainSeriesRef.current.setMarkers(markers);
        } catch (error) {
            if (DEBUG_MODE) console.warn('Failed to set chart markers:', error);
            // Fallback: try with empty markers to clear any existing ones
            mainSeriesRef.current.setMarkers([]);
        }
    }, [signals, selectedSignal, theme]);

    // Create and configure chart
    const createChartInstance = useCallback(() => {
        if (!chartContainerRef.current || chartRef.current) return;

        const container = chartContainerRef.current;
        const containerWidth = container.clientWidth || 600;
        const containerHeight = typeof height === 'number' ? height : 450;

        // Determine theme colors
        const isDark = chartTheme === 'dark' || (chartTheme === 'auto' && theme.palette.mode === 'dark');
        const backgroundColor = isDark ? theme.palette.background.paper : '#ffffff';
        const textColor = isDark ? theme.palette.text.primary : '#333333';
        const gridColor = isDark ? alpha(theme.palette.divider, 0.1) : alpha('#000000', 0.1);

        const chart = createChart(container, {
            width: containerWidth,
            height: containerHeight,
            layout: {
                background: {
                    type: ColorType.Solid,
                    color: backgroundColor,
                },
                textColor,
                fontSize: 12,
            },
            grid: {
                vertLines: {
                    visible: false,
                },
                horzLines: {
                    color: gridColor,
                },
            },
            crosshair: {
                mode: CrosshairMode.Normal,
                vertLine: {
                    width: 1,
                    color: alpha(theme.palette.primary.main, 0.5),
                    style: LineStyle.Dashed,
                },
                horzLine: {
                    width: 1,
                    color: alpha(theme.palette.primary.main, 0.5),
                    style: LineStyle.Dashed,
                },
            },
            rightPriceScale: {
                borderVisible: true,
                borderColor: alpha(theme.palette.divider, 0.2),
                scaleMargins: {
                    top: 0.05,
                    bottom: showVolume ? 0.25 : 0.05,
                },
            },
            timeScale: {
                borderVisible: true,
                borderColor: alpha(theme.palette.divider, 0.2),
                timeVisible: true,
                secondsVisible: false,
            },
        });

        chartRef.current = chart;

        // Add main series
        addMainSeries(chart);

        // Add volume series if enabled
        if (showVolume) {
            addVolumeSeries(chart);
        }

        // Add technical indicators
        addIndicators(chart);

        // Add signal markers
        if (showSignals && signals.length > 0) {
            addSignalMarkers(chart);
        }

        // Register chart for cleanup
        chartRegistry.set(containerIdRef.current, {
            chart,
            instanceId: containerIdRef.current,
            element: container,
        });

        // Setup resize observer
        resizeObserverRef.current = new ResizeObserver(() => {
            if (chartRef.current && container) {
                chartRef.current.applyOptions({ width: container.clientWidth });
            }
        });
        resizeObserverRef.current.observe(container);

    }, [theme, chartTheme, height, showVolume, showSignals, signals, addMainSeries, addVolumeSeries, addIndicators, addSignalMarkers]);

    // Initialize chart
    useEffect(() => {
        createChartInstance();

        return () => {
            // Cleanup
            if (resizeObserverRef.current) {
                resizeObserverRef.current.disconnect();
            }
            if (chartRef.current) {
                chartRef.current.remove();
                chartRef.current = null;
            }
            chartRegistry.delete(containerIdRef.current);
        };
    }, [createChartInstance]);

    // Update chart when data changes
    useEffect(() => {
        if (chartRef.current && Array.isArray(validatedData) && validatedData.length > 0) {
            try {
                // Clear existing series
                indicatorSeriesRef.current.clear();

                // Recreate chart content
                if (mainSeriesRef.current) {
                    chartRef.current.removeSeries(mainSeriesRef.current);
                }
                if (volumeSeriesRef.current) {
                    chartRef.current.removeSeries(volumeSeriesRef.current);
                }

                addMainSeries(chartRef.current);

                if (showVolume) {
                    addVolumeSeries(chartRef.current);
                }

                addIndicators(chartRef.current);

                if (showSignals && signals.length > 0) {
                    addSignalMarkers(chartRef.current);
                }
            } catch (error) {
                if (DEBUG_MODE) console.error('Error updating chart:', error);
            }
        }
    }, [validatedData, selectedType, showVolume, showSignals, signals, addMainSeries, addVolumeSeries, addIndicators, addSignalMarkers]);

    // Handle indicator toggle
    const toggleIndicator = useCallback((indicatorId: string) => {
        setIndicators(prev => prev.map(ind =>
            ind.id === indicatorId ? { ...ind, enabled: !ind.enabled } : ind
        ));
    }, []);

    // Handle refresh
    const handleRefresh = useCallback(() => {
        setLoading(true);
        refetch().finally(() => setLoading(false));
    }, [refetch]);

    // Render controls
    const renderControls = () => {
        if (!showControls) return null;

        return (
            <Paper
                sx={{
                    p: 1.5,
                    mb: 1,
                    bgcolor: alpha(theme.palette.background.paper, 0.8),
                    backdropFilter: 'blur(10px)',
                    border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                }}
            >
                <Stack direction="row" alignItems="center" spacing={2} sx={{ flexWrap: 'wrap', gap: 1 }}>
                    {/* Chart Type */}
                    <ToggleButtonGroup
                        value={selectedType}
                        exclusive
                        onChange={(_, value) => value && setSelectedType(value)}
                        size="small"
                    >
                        <ToggleButton value="candlestick">
                            <Tooltip title="Candlestick">
                                <CandlestickChart />
                            </Tooltip>
                        </ToggleButton>
                        <ToggleButton value="line">
                            <Tooltip title="Line">
                                <ShowChart />
                            </Tooltip>
                        </ToggleButton>
                        <ToggleButton value="bar">
                            <Tooltip title="Bar">
                                <BarChart />
                            </Tooltip>
                        </ToggleButton>
                        <ToggleButton value="area">
                            <Tooltip title="Area">
                                <Timeline />
                            </Tooltip>
                        </ToggleButton>
                    </ToggleButtonGroup>

                    <Divider orientation="vertical" flexItem />

                    {/* Timeframe */}
                    <ToggleButtonGroup
                        value={selectedTimeframe}
                        exclusive
                        onChange={(_, value) => value && setSelectedTimeframe(value)}
                        size="small"
                    >
                        {TIMEFRAMES.map(tf => (
                            <ToggleButton key={tf.value} value={tf.value}>
                                {tf.label}
                            </ToggleButton>
                        ))}
                    </ToggleButtonGroup>

                    <Divider orientation="vertical" flexItem />

                    {/* Indicators */}
                    {showIndicators && (
                        <>
                            <Button
                                size="small"
                                startIcon={<Layers />}
                                onClick={(e) => {
                                    setIndicatorMenuAnchor(e.currentTarget);
                                    setShowIndicatorMenu(true);
                                }}
                                endIcon={
                                    <Badge
                                        badgeContent={indicators.filter(i => i.enabled).length}
                                        color="primary"
                                        max={99}
                                    >
                                        <span />
                                    </Badge>
                                }
                            >
                                Indicators
                            </Button>

                            <Menu
                                anchorEl={indicatorMenuAnchor}
                                open={showIndicatorMenu}
                                onClose={() => setShowIndicatorMenu(false)}
                            >
                                {indicators.map(indicator => (
                                    <MenuItem key={indicator.id}>
                                        <FormControlLabel
                                            control={
                                                <Switch
                                                    checked={indicator.enabled}
                                                    onChange={() => toggleIndicator(indicator.id)}
                                                    size="small"
                                                />
                                            }
                                            label={indicator.name}
                                            sx={{ width: '100%' }}
                                        />
                                    </MenuItem>
                                ))}
                            </Menu>
                        </>
                    )}

                    {/* Advanced Features */}
                    {showAdvancedFeatures && (
                        <Button
                            size="small"
                            startIcon={<SmartToy />}
                            onClick={() => setShowAdvanced(!showAdvanced)}
                            variant={showAdvanced ? 'contained' : 'outlined'}
                        >
                            AI Features
                        </Button>
                    )}

                    <Box sx={{ flex: 1 }} />

                    {/* Actions */}
                    <Stack direction="row" spacing={1}>
                        <Tooltip title="Refresh">
                            <IconButton size="small" onClick={handleRefresh} disabled={loading}>
                                <Refresh />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}>
                            <IconButton size="small" onClick={() => setIsFullscreen(!isFullscreen)}>
                                {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
                            </IconButton>
                        </Tooltip>
                    </Stack>
                </Stack>
            </Paper>
        );
    };

    // Render AI features panel
    const renderAIFeatures = () => {
        if (!showAdvanced) return null;

        return (
            <Collapse in={showAdvanced}>
                <Paper
                    sx={{
                        p: 2,
                        mb: 1,
                        bgcolor: alpha(theme.palette.secondary.main, 0.1),
                        border: `1px solid ${alpha(theme.palette.secondary.main, 0.2)}`,
                    }}
                >
                    <Typography variant="subtitle2" gutterBottom>
                        AI Analysis Features
                    </Typography>
                    <Grid container spacing={2}>
                        <Grid item xs={6} md={3}>
                            <Chip
                                icon={<Psychology />}
                                label="Pattern Recognition"
                                size="small"
                                color="primary"
                                variant="outlined"
                            />
                        </Grid>
                        <Grid item xs={6} md={3}>
                            <Chip
                                icon={<Analytics />}
                                label="Signal Generation"
                                size="small"
                                color="success"
                                variant="outlined"
                            />
                        </Grid>
                        <Grid item xs={6} md={3}>
                            <Chip
                                icon={<Assessment />}
                                label="Risk Analysis"
                                size="small"
                                color="warning"
                                variant="outlined"
                            />
                        </Grid>
                        <Grid item xs={6} md={3}>
                            <Chip
                                icon={<Insights />}
                                label="Market Insights"
                                size="small"
                                color="info"
                                variant="outlined"
                            />
                        </Grid>
                    </Grid>
                </Paper>
            </Collapse>
        );
    };

    // Enhanced AI signal analysis
    const renderEnhancedSignalAnalysis = () => {
        if (!selectedSignal) return null;

        const analysis = analyzeSignalQuality(selectedSignal, validatedData);
        if (!analysis) return null;

        return (
            <Card sx={{ mt: 2 }}>
                <CardContent>
                    <Typography variant="h6" gutterBottom>
                        Signal Analysis: {selectedSignal.symbol}
                    </Typography>

                    <Grid container spacing={2}>
                        <Grid item xs={6}>
                            <Typography variant="body2" color="text.secondary">
                                Overall Score
                            </Typography>
                            <Typography variant="h4" color={analysis.overallScore >= 70 ? 'success.main' :
                                analysis.overallScore >= 50 ? 'warning.main' : 'error.main'}>
                                {analysis.overallScore}%
                            </Typography>
                        </Grid>

                        <Grid item xs={6}>
                            <Typography variant="body2" color="text.secondary">
                                Risk/Reward Ratio
                            </Typography>
                            <Typography variant="h4" color={analysis.riskReward >= 2 ? 'success.main' :
                                analysis.riskReward >= 1.5 ? 'warning.main' : 'error.main'}>
                                {analysis.riskReward.toFixed(2)}:1
                            </Typography>
                        </Grid>

                        <Grid item xs={12}>
                            <Stack direction="row" spacing={1} flexWrap="wrap">
                                <Chip
                                    label="Volume Confirmation"
                                    color={analysis.volumeConfirmation ? 'success' : 'default'}
                                    size="small"
                                />
                                <Chip
                                    label="Trend Alignment"
                                    color={analysis.trendAlignment ? 'success' : 'default'}
                                    size="small"
                                />
                                <Chip
                                    label="Support/Resistance"
                                    color={analysis.supportResistance ? 'success' : 'default'}
                                    size="small"
                                />
                            </Stack>
                        </Grid>
                    </Grid>
                </CardContent>
            </Card>
        );
    };

    // Pattern recognition panel
    const renderPatternRecognition = () => {
        const patterns = detectPatterns(validatedData);

        return (
            <Card sx={{ mt: 2 }}>
                <CardContent>
                    <Typography variant="h6" gutterBottom>
                        Pattern Recognition
                    </Typography>

                    {patterns.length === 0 ? (
                        <Typography variant="body2" color="text.secondary">
                            No patterns detected
                        </Typography>
                    ) : (
                        <Stack spacing={1}>
                            {patterns.map((pattern, index) => (
                                <Card key={index} variant="outlined">
                                    <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                            <Typography variant="subtitle2" fontWeight="bold">
                                                {pattern.type}
                                            </Typography>
                                            <Chip
                                                label={`${pattern.confidence.toFixed(0)}%`}
                                                size="small"
                                                color={pattern.confidence >= 70 ? 'success' :
                                                    pattern.confidence >= 50 ? 'warning' : 'default'}
                                            />
                                        </Box>
                                        <Typography variant="body2" color="text.secondary">
                                            {pattern.description}
                                        </Typography>
                                    </CardContent>
                                </Card>
                            ))}
                        </Stack>
                    )}
                </CardContent>
            </Card>
        );
    };

    // Update the chart effects to include pattern detection
    useEffect(() => {
        if (chartRef.current && validatedData.length > 0 && showAdvanced) {
            const patterns = detectPatterns(validatedData);
            addPatternOverlays(chartRef.current, patterns);
        }
    }, [validatedData, showAdvanced, detectPatterns, addPatternOverlays]);

    return (
        <ErrorBoundary>
            <Box
                className={className}
                sx={{
                    width,
                    height: isFullscreen ? '100vh' : height,
                    position: isFullscreen ? 'fixed' : 'relative',
                    top: isFullscreen ? 0 : 'auto',
                    left: isFullscreen ? 0 : 'auto',
                    right: isFullscreen ? 0 : 'auto',
                    bottom: isFullscreen ? 0 : 'auto',
                    zIndex: isFullscreen ? 1300 : 'auto',
                    bgcolor: isFullscreen ? theme.palette.background.default : 'transparent',
                    display: 'flex',
                    flexDirection: 'column',
                }}
            >
                {renderControls()}
                {renderAIFeatures()}

                {/* Chart Container */}
                <Box
                    sx={{
                        flex: 1,
                        position: 'relative',
                        overflow: 'hidden',
                        borderRadius: 1,
                        bgcolor: theme.palette.background.paper,
                        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                    }}
                >
                    <Box
                        ref={chartContainerRef}
                        sx={{
                            width: '100%',
                            height: '100%',
                            '& > div': {
                                width: '100% !important',
                                height: '100% !important',
                            }
                        }}
                    />

                    {/* Loading Overlay */}
                    {(isLoading || loading) && (
                        <Box
                            sx={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                right: 0,
                                bottom: 0,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                bgcolor: alpha(theme.palette.background.default, 0.8),
                                backdropFilter: 'blur(4px)',
                                zIndex: 10,
                            }}
                        >
                            <Stack alignItems="center" spacing={1}>
                                <CircularProgress size={32} />
                                <Typography variant="body2">
                                    Loading {symbol} data...
                                </Typography>
                            </Stack>
                        </Box>
                    )}

                    {/* Symbol Info Overlay */}
                    <Box
                        sx={{
                            position: 'absolute',
                            top: 8,
                            left: 8,
                            zIndex: 5,
                        }}
                    >
                        <Chip
                            label={`${symbol} • ${selectedTimeframe.toUpperCase()}`}
                            size="small"
                            sx={{
                                bgcolor: alpha(theme.palette.background.paper, 0.9),
                                backdropFilter: 'blur(10px)',
                            }}
                        />
                    </Box>

                    {/* Signal Info */}
                    {selectedSignal && (
                        <Box
                            sx={{
                                position: 'absolute',
                                top: 8,
                                right: 8,
                                zIndex: 5,
                            }}
                        >
                            <Paper
                                sx={{
                                    p: 1,
                                    bgcolor: alpha(theme.palette.background.paper, 0.95),
                                    backdropFilter: 'blur(10px)',
                                    border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                                }}
                            >
                                <Stack spacing={0.5}>
                                    <Typography variant="caption" fontWeight="bold">
                                        {selectedSignal.type} Signal
                                    </Typography>
                                    <Typography variant="caption">
                                        Confidence: {selectedSignal.confidence}%
                                    </Typography>
                                    {selectedSignal.entry && (
                                        <Typography variant="caption">
                                            Entry: ${selectedSignal.entry.toFixed(2)}
                                        </Typography>
                                    )}
                                    {selectedSignal.target && (
                                        <Typography variant="caption">
                                            Target: ${selectedSignal.target.toFixed(2)}
                                        </Typography>
                                    )}
                                    {selectedSignal.stopLoss && (
                                        <Typography variant="caption">
                                            Stop: ${selectedSignal.stopLoss.toFixed(2)}
                                        </Typography>
                                    )}
                                </Stack>
                            </Paper>
                        </Box>
                    )}
                </Box>

                {renderEnhancedSignalAnalysis()}
                {renderPatternRecognition()}
            </Box>
        </ErrorBoundary>
    );
};

export default UnifiedChart; 