/**
 * Predictive Trading Chart with Advanced Visualization
 * 
 * Enhanced charting component with price predictions, candlestick patterns,
 * and trend visualization
 */

import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import {
    Box,
    IconButton,
    Chip,
    Typography,
    Stack,
    Button,
    useTheme,
    CircularProgress,
    Tooltip,
    Paper,
    Divider,
    ToggleButton,
    ToggleButtonGroup,
    Menu,
    MenuItem,
    ListItemIcon,
    ListItemText,
    Switch,
    FormControlLabel,
    alpha,
    Card,
    Badge,
    LinearProgress,
    Alert,
    Collapse,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    ShowChart,
    Timeline,
    Insights,
    Pattern as PatternIcon,
    AutoAwesome as AIIcon,
    Close as CloseIcon,
    Analytics,
    Assessment,
    PredictiveAnalytics,
    CandlestickChart,
    Visibility,
    VisibilityOff,
    ZoomIn,
    ZoomOut,
    RestartAlt,
} from '@mui/icons-material';
import {
    createChart,
    IChartApi,
    ISeriesApi,
    CandlestickData,
    Time,
    LineStyle,
    CrosshairMode,
    SeriesMarker,
    LineData,
    AreaData,
} from 'lightweight-charts';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../services/api';
import { motion, AnimatePresence } from 'framer-motion';

interface PredictiveTradingChartProps {
    symbol: string;
    height?: number;
    onPatternDetected?: (pattern: any) => void;
    onPredictionUpdate?: (prediction: any) => void;
}

interface PredictionData {
    timestamp: string;
    price: number;
    confidence: number;
    upperBound: number;
    lowerBound: number;
}

interface CandlestickPattern {
    type: string;
    timestamp: string;
    price: number;
    direction: string;
    strength: number;
    confidence: number;
    successRate: number;
    description: string;
    targets: {
        priceTarget: number | null;
        stopLoss: number | null;
    };
}

const PredictiveTradingChart: React.FC<PredictiveTradingChartProps> = ({
    symbol,
    height = 600,
    onPatternDetected,
    onPredictionUpdate,
}) => {
    const theme = useTheme();
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
    const predictionSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
    const upperBoundSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
    const lowerBoundSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
    const confidenceAreaSeriesRef = useRef<ISeriesApi<'Area'> | null>(null);

    const [showPredictions, setShowPredictions] = useState(true);
    const [showPatterns, setShowPatterns] = useState(true);
    const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
    const [predictionPeriods, setPredictionPeriods] = useState(20);
    const [detectedPatterns, setDetectedPatterns] = useState<CandlestickPattern[]>([]);
    const [currentPrediction, setCurrentPrediction] = useState<any>(null);

    // Chart theme
    const chartColors = useMemo(() => ({
        background: theme.palette.mode === 'dark' ? '#0A0A0A' : '#FFFFFF',
        text: theme.palette.mode === 'dark' ? '#E2E8F0' : '#1F2937',
        grid: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.04)' : 'rgba(0, 0, 0, 0.06)',
        crosshair: '#2563EB',
        upColor: '#10B981',
        downColor: '#EF4444',
        predictionColor: '#8B5CF6',
        confidenceColor: 'rgba(139, 92, 246, 0.2)',
        patternBullish: '#10B981',
        patternBearish: '#EF4444',
        patternNeutral: '#6B7280',
    }), [theme.palette.mode]);

    // Fetch historical data
    const { data: historicalData, isLoading: historicalLoading } = useQuery({
        queryKey: ['historical-data', symbol, '1d', '5m'],
        queryFn: async () => {
            const response = await fetch(
                `http://localhost:8000/api/v1/market-data/${symbol}/historical?period=1d&interval=5m`
            );
            const data = await response.json();
            return data.data;
        },
        refetchInterval: 60000,
    });

    // Fetch predictions
    const { data: predictions, isLoading: predictionsLoading } = useQuery({
        queryKey: ['predictions', symbol, selectedTimeframe, predictionPeriods],
        queryFn: async () => {
            const response = await fetch(
                `http://localhost:8000/api/v1/predictions/${symbol}?timeframe=${selectedTimeframe}&periods=${predictionPeriods}`
            );
            return response.json();
        },
        refetchInterval: 300000, // Refresh every 5 minutes
        onSuccess: (data) => {
            setCurrentPrediction(data);
            onPredictionUpdate?.(data);
        },
    });

    // Fetch candlestick patterns
    const { data: patterns, isLoading: patternsLoading } = useQuery({
        queryKey: ['patterns', symbol],
        queryFn: async () => {
            const response = await fetch(
                `http://localhost:8000/api/v1/patterns/${symbol}?lookback=100`
            );
            return response.json();
        },
        refetchInterval: 60000,
        onSuccess: (data) => {
            setDetectedPatterns(data.patterns || []);
            if (data.patterns?.length > 0) {
                onPatternDetected?.(data.patterns[0]);
            }
        },
    });

    // Initialize chart
    useEffect(() => {
        if (!chartContainerRef.current || chartRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height: height,
            layout: {
                background: { type: 'solid', color: chartColors.background },
                textColor: chartColors.text,
            },
            grid: {
                vertLines: { color: chartColors.grid },
                horzLines: { color: chartColors.grid },
            },
            crosshair: {
                mode: CrosshairMode.Normal,
                vertLine: {
                    color: chartColors.crosshair,
                    width: 1,
                    style: LineStyle.Dashed,
                },
                horzLine: {
                    color: chartColors.crosshair,
                    width: 1,
                    style: LineStyle.Dashed,
                },
            },
            rightPriceScale: {
                borderColor: chartColors.grid,
            },
            timeScale: {
                borderColor: chartColors.grid,
                timeVisible: true,
                secondsVisible: false,
            },
        });

        // Create candlestick series
        const candlestickSeries = chart.addCandlestickSeries({
            upColor: chartColors.upColor,
            downColor: chartColors.downColor,
            borderUpColor: chartColors.upColor,
            borderDownColor: chartColors.downColor,
            wickUpColor: chartColors.upColor,
            wickDownColor: chartColors.downColor,
        });

        // Create prediction line series
        const predictionSeries = chart.addLineSeries({
            color: chartColors.predictionColor,
            lineWidth: 3,
            lineStyle: LineStyle.Solid,
            crosshairMarkerVisible: true,
            crosshairMarkerRadius: 5,
            title: 'Prediction',
        });

        // Create confidence bounds
        const upperBoundSeries = chart.addLineSeries({
            color: alpha(chartColors.predictionColor, 0.5),
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            crosshairMarkerVisible: false,
            title: 'Upper Bound',
        });

        const lowerBoundSeries = chart.addLineSeries({
            color: alpha(chartColors.predictionColor, 0.5),
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            crosshairMarkerVisible: false,
            title: 'Lower Bound',
        });

        // Create confidence area
        const confidenceAreaSeries = chart.addAreaSeries({
            topColor: alpha(chartColors.predictionColor, 0.2),
            bottomColor: alpha(chartColors.predictionColor, 0.05),
            lineColor: 'transparent',
            crosshairMarkerVisible: false,
        });

        chartRef.current = chart;
        candlestickSeriesRef.current = candlestickSeries;
        predictionSeriesRef.current = predictionSeries;
        upperBoundSeriesRef.current = upperBoundSeries;
        lowerBoundSeriesRef.current = lowerBoundSeries;
        confidenceAreaSeriesRef.current = confidenceAreaSeries;

        // Handle resize
        const handleResize = () => {
            if (chartRef.current && chartContainerRef.current) {
                chartRef.current.applyOptions({
                    width: chartContainerRef.current.clientWidth,
                });
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            if (chartRef.current) {
                chartRef.current.remove();
                chartRef.current = null;
            }
        };
    }, [height, chartColors]);

    // Update historical data
    useEffect(() => {
        if (!candlestickSeriesRef.current || !historicalData) return;

        const formattedData: CandlestickData[] = historicalData.map((d: any) => ({
            time: d.time as Time,
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
        }));

        candlestickSeriesRef.current.setData(formattedData);
        chartRef.current?.timeScale().fitContent();
    }, [historicalData]);

    // Update predictions
    useEffect(() => {
        if (!predictions || !showPredictions) {
            predictionSeriesRef.current?.setData([]);
            upperBoundSeriesRef.current?.setData([]);
            lowerBoundSeriesRef.current?.setData([]);
            confidenceAreaSeriesRef.current?.setData([]);
            return;
        }

        if (predictions.predictions && predictions.predictions.length > 0) {
            // Get the last historical data point time
            const lastHistoricalTime = historicalData?.[historicalData.length - 1]?.time || Math.floor(Date.now() / 1000);

            // Prepare prediction data
            const predictionData: LineData[] = predictions.predictions.map((p: PredictionData, index: number) => ({
                time: (lastHistoricalTime + (index + 1) * 3600) as Time, // Add hours
                value: p.price,
            }));

            // Add current price as starting point
            predictionData.unshift({
                time: lastHistoricalTime as Time,
                value: predictions.currentPrice,
            });

            // Prepare confidence bounds
            const upperBoundData: LineData[] = predictions.predictions.map((p: PredictionData, index: number) => ({
                time: (lastHistoricalTime + (index + 1) * 3600) as Time,
                value: p.upperBound,
            }));

            upperBoundData.unshift({
                time: lastHistoricalTime as Time,
                value: predictions.currentPrice,
            });

            const lowerBoundData: LineData[] = predictions.predictions.map((p: PredictionData, index: number) => ({
                time: (lastHistoricalTime + (index + 1) * 3600) as Time,
                value: p.lowerBound,
            }));

            lowerBoundData.unshift({
                time: lastHistoricalTime as Time,
                value: predictions.currentPrice,
            });

            // Create area data for confidence interval
            const areaData: AreaData[] = predictions.predictions.map((p: PredictionData, index: number) => ({
                time: (lastHistoricalTime + (index + 1) * 3600) as Time,
                value: (p.upperBound + p.lowerBound) / 2,
            }));

            areaData.unshift({
                time: lastHistoricalTime as Time,
                value: predictions.currentPrice,
            });

            // Set data
            predictionSeriesRef.current?.setData(predictionData);
            upperBoundSeriesRef.current?.setData(upperBoundData);
            lowerBoundSeriesRef.current?.setData(lowerBoundData);
            confidenceAreaSeriesRef.current?.setData(areaData);
        }
    }, [predictions, historicalData, showPredictions]);

    // Add pattern markers
    useEffect(() => {
        if (!candlestickSeriesRef.current || !showPatterns || !detectedPatterns.length) {
            candlestickSeriesRef.current?.setMarkers([]);
            return;
        }

        const markers: SeriesMarker<Time>[] = detectedPatterns.map((pattern) => {
            const timestamp = new Date(pattern.timestamp).getTime() / 1000;
            return {
                time: timestamp as Time,
                position: pattern.direction === 'bullish' ? 'belowBar' : 'aboveBar',
                color: pattern.direction === 'bullish' ? chartColors.patternBullish :
                    pattern.direction === 'bearish' ? chartColors.patternBearish :
                        chartColors.patternNeutral,
                shape: pattern.direction === 'bullish' ? 'arrowUp' :
                    pattern.direction === 'bearish' ? 'arrowDown' : 'circle',
                text: pattern.type.replace(/_/g, ' '),
            };
        });

        candlestickSeriesRef.current.setMarkers(markers);
    }, [detectedPatterns, showPatterns, chartColors]);

    // Add support/resistance levels
    useEffect(() => {
        if (!chartRef.current || !candlestickSeriesRef.current || !predictions) return;

        // Clear existing price lines
        // Note: LightweightCharts doesn't have a clear method for price lines,
        // so we need to track them separately if needed

        // Add support levels
        predictions.levels?.support?.forEach((level: number, index: number) => {
            if (index < 3) { // Show top 3 levels
                candlestickSeriesRef.current?.createPriceLine({
                    price: level,
                    color: alpha(chartColors.upColor, 0.5),
                    lineWidth: 1,
                    lineStyle: LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: `Support ${index + 1}`,
                });
            }
        });

        // Add resistance levels
        predictions.levels?.resistance?.forEach((level: number, index: number) => {
            if (index < 3) { // Show top 3 levels
                candlestickSeriesRef.current?.createPriceLine({
                    price: level,
                    color: alpha(chartColors.downColor, 0.5),
                    lineWidth: 1,
                    lineStyle: LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: `Resistance ${index + 1}`,
                });
            }
        });
    }, [predictions, chartColors]);

    const handleZoomIn = () => {
        if (chartRef.current) {
            const timeScale = chartRef.current.timeScale();
            const currentRange = timeScale.getVisibleRange();
            if (currentRange) {
                const center = (currentRange.from + currentRange.to) / 2;
                const newRange = (currentRange.to - currentRange.from) * 0.5;
                timeScale.setVisibleRange({
                    from: (center - newRange / 2) as Time,
                    to: (center + newRange / 2) as Time,
                });
            }
        }
    };

    const handleZoomOut = () => {
        if (chartRef.current) {
            const timeScale = chartRef.current.timeScale();
            const currentRange = timeScale.getVisibleRange();
            if (currentRange) {
                const center = (currentRange.from + currentRange.to) / 2;
                const newRange = (currentRange.to - currentRange.from) * 2;
                timeScale.setVisibleRange({
                    from: (center - newRange / 2) as Time,
                    to: (center + newRange / 2) as Time,
                });
            }
        }
    };

    const handleResetZoom = () => {
        chartRef.current?.timeScale().fitContent();
    };

    return (
        <Card sx={{
            p: 2,
            backgroundColor: chartColors.background,
            height: height + 200,
            display: 'flex',
            flexDirection: 'column',
        }}>
            {/* Header */}
            <Box sx={{ mb: 2 }}>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                    <Stack direction="row" spacing={2} alignItems="center">
                        <Typography variant="h6" sx={{ color: chartColors.text }}>
                            {symbol} - Predictive Analysis
                        </Typography>

                        {currentPrediction && (
                            <Chip
                                icon={currentPrediction.trend.direction === 'bullish' ? <TrendingUp /> : <TrendingDown />}
                                label={`${currentPrediction.trend.direction.toUpperCase()} (${currentPrediction.trend.confidence.toFixed(0)}%)`}
                                color={currentPrediction.trend.direction === 'bullish' ? 'success' : 'error'}
                                size="small"
                            />
                        )}
                    </Stack>

                    <Stack direction="row" spacing={1}>
                        <ToggleButtonGroup size="small" exclusive>
                            <ToggleButton
                                value="predictions"
                                selected={showPredictions}
                                onClick={() => setShowPredictions(!showPredictions)}
                            >
                                <Tooltip title="Toggle Predictions">
                                    <Timeline fontSize="small" />
                                </Tooltip>
                            </ToggleButton>
                            <ToggleButton
                                value="patterns"
                                selected={showPatterns}
                                onClick={() => setShowPatterns(!showPatterns)}
                            >
                                <Tooltip title="Toggle Patterns">
                                    <PatternIcon fontSize="small" />
                                </Tooltip>
                            </ToggleButton>
                        </ToggleButtonGroup>

                        <IconButton size="small" onClick={handleZoomIn}>
                            <ZoomIn />
                        </IconButton>
                        <IconButton size="small" onClick={handleZoomOut}>
                            <ZoomOut />
                        </IconButton>
                        <IconButton size="small" onClick={handleResetZoom}>
                            <RestartAlt />
                        </IconButton>
                    </Stack>
                </Stack>
            </Box>

            {/* Loading State */}
            {(historicalLoading || predictionsLoading || patternsLoading) && (
                <LinearProgress sx={{ mb: 1 }} />
            )}

            {/* Chart */}
            <Box ref={chartContainerRef} sx={{ flex: 1, position: 'relative' }} />

            {/* Info Panels */}
            <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
                {/* Prediction Info */}
                {currentPrediction && showPredictions && (
                    <Paper sx={{ p: 2, flex: 1, backgroundColor: alpha(chartColors.background, 0.5) }}>
                        <Typography variant="subtitle2" sx={{ mb: 1, color: chartColors.text }}>
                            Price Prediction
                        </Typography>
                        <Stack spacing={1}>
                            <Typography variant="body2" sx={{ color: chartColors.text }}>
                                Current: ${currentPrediction.currentPrice.toFixed(2)}
                            </Typography>
                            {currentPrediction.predictions?.[currentPrediction.predictions.length - 1] && (
                                <>
                                    <Typography variant="body2" sx={{ color: chartColors.text }}>
                                        Target: ${currentPrediction.predictions[currentPrediction.predictions.length - 1].price.toFixed(2)}
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: chartColors.text }}>
                                        Range: ${currentPrediction.predictions[currentPrediction.predictions.length - 1].lowerBound.toFixed(2)} -
                                        ${currentPrediction.predictions[currentPrediction.predictions.length - 1].upperBound.toFixed(2)}
                                    </Typography>
                                </>
                            )}
                            <Stack direction="row" spacing={1}>
                                <Chip
                                    label={`Momentum: ${currentPrediction.metrics.momentum.toFixed(0)}`}
                                    size="small"
                                    color={currentPrediction.metrics.momentum > 0 ? 'success' : 'error'}
                                />
                                <Chip
                                    label={`Volatility: ${currentPrediction.metrics.volatility.toFixed(0)}%`}
                                    size="small"
                                />
                            </Stack>
                        </Stack>
                    </Paper>
                )}

                {/* Pattern Info */}
                {detectedPatterns.length > 0 && showPatterns && (
                    <Paper sx={{ p: 2, flex: 1, backgroundColor: alpha(chartColors.background, 0.5) }}>
                        <Typography variant="subtitle2" sx={{ mb: 1, color: chartColors.text }}>
                            Recent Patterns
                        </Typography>
                        <Stack spacing={1}>
                            {detectedPatterns.slice(0, 3).map((pattern, index) => (
                                <Box key={index}>
                                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                                        <Typography variant="body2" sx={{ color: chartColors.text }}>
                                            {pattern.type.replace(/_/g, ' ')}
                                        </Typography>
                                        <Chip
                                            label={`${pattern.confidence.toFixed(0)}%`}
                                            size="small"
                                            color={pattern.direction === 'bullish' ? 'success' :
                                                pattern.direction === 'bearish' ? 'error' : 'default'}
                                        />
                                    </Stack>
                                    <Typography variant="caption" sx={{ color: alpha(chartColors.text, 0.7) }}>
                                        {pattern.description}
                                    </Typography>
                                </Box>
                            ))}
                        </Stack>
                    </Paper>
                )}
            </Stack>
        </Card>
    );
};

export default PredictiveTradingChart; 