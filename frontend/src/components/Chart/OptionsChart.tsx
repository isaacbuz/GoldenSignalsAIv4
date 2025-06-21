/**
 * OptionsChart Component - Professional Options Trading Chart
 * 
 * Central chart component with:
 * - Real-time price action
 * - Options signal overlays
 * - Technical indicators
 * - Entry/exit zones
 * - Options Greeks visualization
 * - Volume profile
 */

import React, { useEffect, useRef, useState, useMemo } from 'react';
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
    Divider,
    useTheme,
    alpha,
    Tooltip,
    CircularProgress,
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
} from 'lightweight-charts';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../services/api';
import { PreciseOptionsSignal } from '../../types/signals';

interface OptionsChartProps {
    symbol: string;
    timeframe?: string;
    signals?: PreciseOptionsSignal[];
    selectedSignal?: PreciseOptionsSignal | null;
    marketData?: any;
    height?: number;
    onSignalClick?: (signal: PreciseOptionsSignal) => void;
}

interface Indicator {
    id: string;
    name: string;
    active: boolean;
    series?: ISeriesApi<any>;
}

const OptionsChart: React.FC<OptionsChartProps> = ({
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
    const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
    const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
    const signalMarkersRef = useRef<SeriesMarker<Time>[]>([]);

    const [chartType, setChartType] = useState<'candlestick' | 'line' | 'bar'>('candlestick');
    const [timeframe, setTimeframe] = useState(propTimeframe || '1D');
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [indicators, setIndicators] = useState<Indicator[]>([
        { id: 'sma20', name: 'SMA 20', active: true },
        { id: 'sma50', name: 'SMA 50', active: true },
        { id: 'ema9', name: 'EMA 9', active: false },
        { id: 'bb', name: 'Bollinger Bands', active: true },
        { id: 'volume', name: 'Volume', active: true },
    ]);
    const [showSignalOverlays, setShowSignalOverlays] = useState(true);
    const [showOptionsFlow, setShowOptionsFlow] = useState(true);
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

    // Fetch market data
    const { data: fetchedMarketData, isLoading, refetch } = useQuery({
        queryKey: ['options-chart-data', symbol, timeframe],
        queryFn: async () => {
            // In production, this would fetch real data
            return generateMockData();
        },
        refetchInterval: 30000,
    });

    const marketData = propMarketData || fetchedMarketData;

    // Fetch options flow data
    const { data: optionsFlow } = useQuery({
        queryKey: ['options-flow', symbol],
        queryFn: async () => {
            // Mock options flow data
            return generateMockOptionsFlow();
        },
        refetchInterval: 60000,
    });

    // Generate mock data
    const generateMockData = () => {
        const data = [];
        const basePrice = 150;
        const now = Math.floor(Date.now() / 1000);
        const intervals = {
            '1D': { count: 96, interval: 900 }, // 15-min bars
            '5D': { count: 120, interval: 3600 }, // 1-hour bars
            '1M': { count: 22, interval: 86400 }, // Daily bars
            '3M': { count: 66, interval: 86400 },
            '1Y': { count: 252, interval: 86400 },
        };

        const config = intervals[timeframe as keyof typeof intervals] || intervals['1D'];

        for (let i = 0; i < config.count; i++) {
            const time = now - (config.count - i) * config.interval;
            const trend = Math.sin(i / 20) * 5;
            const volatility = 2;

            const open = basePrice + trend + (Math.random() - 0.5) * volatility;
            const close = basePrice + trend + (Math.random() - 0.5) * volatility;
            const high = Math.max(open, close) + Math.random() * volatility;
            const low = Math.min(open, close) - Math.random() * volatility;
            const volume = 1000000 + Math.random() * 500000;

            data.push({ time: time as Time, open, high, low, close, volume });
        }

        return data;
    };

    const generateMockOptionsFlow = () => {
        return {
            callVolume: 125000,
            putVolume: 98000,
            putCallRatio: 0.78,
            unusualActivity: [
                { time: Date.now() - 3600000, type: 'CALL', strike: 155, volume: 5000, premium: 125000 },
                { time: Date.now() - 7200000, type: 'PUT', strike: 145, volume: 3000, premium: 75000 },
            ],
        };
    };

    // Initialize chart
    useEffect(() => {
        if (!chartContainerRef.current || !marketData) return;

        // Create chart
        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height,
            layout: {
                background: { color: 'transparent' },
                textColor: theme.palette.text.primary,
                fontSize: 12,
            },
            grid: {
                vertLines: { color: alpha(theme.palette.divider, 0.1) },
                horzLines: { color: alpha(theme.palette.divider, 0.1) },
            },
            crosshair: {
                mode: CrosshairMode.Normal,
                vertLine: {
                    color: theme.palette.primary.main,
                    width: 1,
                    style: LineStyle.Dashed,
                },
                horzLine: {
                    color: theme.palette.primary.main,
                    width: 1,
                    style: LineStyle.Dashed,
                },
            },
            rightPriceScale: {
                borderColor: alpha(theme.palette.divider, 0.1),
                scaleMargins: {
                    top: 0.1,
                    bottom: 0.25,
                },
            },
            timeScale: {
                borderColor: alpha(theme.palette.divider, 0.1),
                timeVisible: true,
                secondsVisible: false,
            },
        });

        chartRef.current = chart;

        // Add main price series
        let mainSeries: ISeriesApi<any>;
        if (chartType === 'candlestick') {
            mainSeries = chart.addCandlestickSeries({
                upColor: theme.palette.success.main,
                downColor: theme.palette.error.main,
                borderUpColor: theme.palette.success.main,
                borderDownColor: theme.palette.error.main,
                wickUpColor: theme.palette.success.main,
                wickDownColor: theme.palette.error.main,
            });
            seriesRef.current = mainSeries;
        } else if (chartType === 'line') {
            mainSeries = chart.addLineSeries({
                color: theme.palette.primary.main,
                lineWidth: 2,
            });
        } else {
            mainSeries = chart.addBarSeries({
                upColor: theme.palette.success.main,
                downColor: theme.palette.error.main,
            });
        }

        // Set data
        const priceData = chartType === 'line'
            ? marketData.map((d: any) => ({ time: d.time, value: d.close }))
            : marketData;
        mainSeries.setData(priceData);

        // Add volume
        if (indicators.find(i => i.id === 'volume' && i.active)) {
            const volumeSeries = chart.addHistogramSeries({
                color: '#26a69a',
                priceFormat: { type: 'volume' },
                priceScaleId: '',
            });

            volumeSeries.priceScale().applyOptions({
                scaleMargins: {
                    top: 0.8,
                    bottom: 0,
                },
            });

            volumeSeries.setData(
                marketData.map((d: any) => ({
                    time: d.time,
                    value: d.volume,
                    color: d.close >= d.open ? alpha(theme.palette.success.main, 0.5) : alpha(theme.palette.error.main, 0.5),
                }))
            );
            volumeSeriesRef.current = volumeSeries;
        }

        // Add indicators
        if (indicators.find(i => i.id === 'sma20' && i.active)) {
            const sma20Series = chart.addLineSeries({
                color: '#2962FF',
                lineWidth: 1,
                title: 'SMA 20',
            });
            sma20Series.setData(calculateSMA(marketData, 20));
        }

        if (indicators.find(i => i.id === 'sma50' && i.active)) {
            const sma50Series = chart.addLineSeries({
                color: '#FF6D00',
                lineWidth: 1,
                title: 'SMA 50',
            });
            sma50Series.setData(calculateSMA(marketData, 50));
        }

        // Add Bollinger Bands
        if (indicators.find(i => i.id === 'bb' && i.active)) {
            const bbData = calculateBollingerBands(marketData, 20, 2);

            const upperBand = chart.addLineSeries({
                color: alpha(theme.palette.info.main, 0.5),
                lineWidth: 1,
                lineStyle: LineStyle.Dashed,
                title: 'BB Upper',
            });
            upperBand.setData(bbData.upper);

            const lowerBand = chart.addLineSeries({
                color: alpha(theme.palette.info.main, 0.5),
                lineWidth: 1,
                lineStyle: LineStyle.Dashed,
                title: 'BB Lower',
            });
            lowerBand.setData(bbData.lower);
        }

        // Add signal overlays
        if (showSignalOverlays && signals.length > 0) {
            signals.forEach((signal) => {
                // Entry line
                mainSeries.createPriceLine({
                    price: signal.entry_trigger,
                    color: theme.palette.primary.main,
                    lineWidth: 2,
                    lineStyle: LineStyle.Solid,
                    axisLabelVisible: true,
                    title: `Entry ${signal.signal_type}`,
                });

                // Stop loss line
                mainSeries.createPriceLine({
                    price: signal.stop_loss,
                    color: theme.palette.error.main,
                    lineWidth: 2,
                    lineStyle: LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: 'Stop Loss',
                });

                // Target lines
                signal.targets.forEach((target, idx) => {
                    mainSeries.createPriceLine({
                        price: target.price,
                        color: theme.palette.success.main,
                        lineWidth: 2,
                        lineStyle: LineStyle.Dotted,
                        axisLabelVisible: true,
                        title: `Target ${idx + 1}`,
                    });
                });

                // Entry zone shading
                const entryZoneSeries = chart.addLineSeries({
                    color: alpha(theme.palette.primary.main, 0.2),
                    lineVisible: false,
                    priceLineVisible: false,
                });

                // Add markers for signals
                const lastBar = marketData[marketData.length - 1];
                if (lastBar) {
                    mainSeries.setMarkers([{
                        time: lastBar.time,
                        position: signal.signal_type === 'BUY_CALL' ? 'belowBar' : 'aboveBar',
                        color: signal.signal_type === 'BUY_CALL' ? theme.palette.success.main : theme.palette.error.main,
                        shape: signal.signal_type === 'BUY_CALL' ? 'arrowUp' : 'arrowDown',
                        text: `${signal.confidence}%`,
                        size: 2,
                    }]);
                }
            });
        }

        // Fit content
        chart.timeScale().fitContent();

        // Handle resize
        const handleResize = () => {
            if (chartContainerRef.current && chartRef.current) {
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
    }, [theme, marketData, chartType, indicators, showSignalOverlays, signals, height]);

    // Calculate SMA
    const calculateSMA = (data: any[], period: number) => {
        const sma = [];
        for (let i = period - 1; i < data.length; i++) {
            let sum = 0;
            for (let j = 0; j < period; j++) {
                sum += data[i - j].close;
            }
            sma.push({
                time: data[i].time,
                value: sum / period,
            });
        }
        return sma;
    };

    // Calculate Bollinger Bands
    const calculateBollingerBands = (data: any[], period: number, stdDev: number) => {
        const sma = calculateSMA(data, period);
        const upper = [];
        const lower = [];

        for (let i = 0; i < sma.length; i++) {
            const dataIndex = i + period - 1;
            let sumSquaredDiff = 0;

            for (let j = 0; j < period; j++) {
                const diff = data[dataIndex - j].close - sma[i].value;
                sumSquaredDiff += diff * diff;
            }

            const std = Math.sqrt(sumSquaredDiff / period);

            upper.push({
                time: sma[i].time,
                value: sma[i].value + std * stdDev,
            });

            lower.push({
                time: sma[i].time,
                value: sma[i].value - std * stdDev,
            });
        }

        return { upper, lower, middle: sma };
    };

    const handleIndicatorToggle = (indicatorId: string) => {
        setIndicators(prev =>
            prev.map(ind =>
                ind.id === indicatorId ? { ...ind, active: !ind.active } : ind
            )
        );
    };

    // Update chart when signals or selectedSignal changes
    useEffect(() => {
        if (!seriesRef.current || !signals.length) return;

        // Clear existing markers
        seriesRef.current.setMarkers([]);

        // Create markers for all signals
        const markers: SeriesMarker<Time>[] = signals.map(signal => {
            const isSelected = selectedSignal?.id === signal.id;
            return {
                time: Math.floor(new Date(signal.timestamp).getTime() / 1000) as Time,
                position: signal.type === 'CALL' ? 'belowBar' : 'aboveBar',
                color: signal.type === 'CALL' ? '#00D4AA' : '#FF3B30',
                shape: signal.type === 'CALL' ? 'arrowUp' : 'arrowDown',
                text: `${signal.type} ${signal.strike_price}`,
                size: isSelected ? 3 : 2,
            };
        });

        seriesRef.current.setMarkers(markers);
        signalMarkersRef.current = markers;

        // Add price lines for selected signal
        if (selectedSignal && chartRef.current) {
            // Remove existing price lines
            chartRef.current.removeAllPricelines?.();

            // Add entry price line
            const entryPriceLine = {
                price: selectedSignal.entry_price,
                color: '#00D4AA',
                lineWidth: 2,
                lineStyle: LineStyle.Solid,
                axisLabelVisible: true,
                title: 'Entry',
            };

            // Add stop loss line
            const stopLossPriceLine = {
                price: selectedSignal.stop_loss,
                color: '#FF3B30',
                lineWidth: 2,
                lineStyle: LineStyle.Dashed,
                axisLabelVisible: true,
                title: 'Stop Loss',
            };

            // Add take profit line
            const takeProfitPriceLine = {
                price: selectedSignal.take_profit,
                color: '#00D4AA',
                lineWidth: 2,
                lineStyle: LineStyle.Dashed,
                axisLabelVisible: true,
                title: 'Take Profit',
            };

            // Note: createPriceLine requires proper type casting
            if (seriesRef.current) {
                seriesRef.current.createPriceLine({
                    ...entryPriceLine,
                    lineWidth: 2 as any,
                });
                seriesRef.current.createPriceLine({
                    ...stopLossPriceLine,
                    lineWidth: 2 as any,
                });
                seriesRef.current.createPriceLine({
                    ...takeProfitPriceLine,
                    lineWidth: 2 as any,
                });
            }
        }
    }, [signals, selectedSignal]);

    // Update timeframe display
    const getTimeframeLabel = (tf: string) => {
        const labels: Record<string, string> = {
            '1m': '1 Minute',
            '5m': '5 Minutes',
            '15m': '15 Minutes',
            '30m': '30 Minutes',
            '1h': '1 Hour',
            '4h': '4 Hours',
            '1d': 'Daily',
        };
        return labels[tf] || tf;
    };

    return (
        <Card sx={{
            height: isFullscreen ? '100vh' : 'auto',
            position: isFullscreen ? 'fixed' : 'relative',
            top: isFullscreen ? 0 : 'auto',
            left: isFullscreen ? 0 : 'auto',
            right: isFullscreen ? 0 : 'auto',
            bottom: isFullscreen ? 0 : 'auto',
            zIndex: isFullscreen ? 9999 : 'auto',
            background: alpha(theme.palette.background.paper, 0.95),
            backdropFilter: 'blur(10px)',
        }}>
            {/* Chart Header */}
            <Stack
                direction="row"
                alignItems="center"
                justifyContent="space-between"
                sx={{
                    p: 2,
                    borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                }}
            >
                <Stack direction="row" alignItems="center" spacing={2}>
                    <Typography variant="h6" fontWeight="bold">
                        {symbol}
                    </Typography>

                    {/* Timeframe selector */}
                    <ToggleButtonGroup
                        value={timeframe}
                        exclusive
                        onChange={(_, value) => value && setTimeframe(value)}
                        size="small"
                    >
                        {['1D', '5D', '1M', '3M', '1Y'].map((tf) => (
                            <ToggleButton key={tf} value={tf}>
                                {tf}
                            </ToggleButton>
                        ))}
                    </ToggleButtonGroup>

                    {/* Chart type selector */}
                    <ToggleButtonGroup
                        value={chartType}
                        exclusive
                        onChange={(_, value) => value && setChartType(value)}
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
                    </ToggleButtonGroup>
                </Stack>

                <Stack direction="row" spacing={1}>
                    {/* Options flow indicator */}
                    {optionsFlow && (
                        <Chip
                            icon={<Speed />}
                            label={`P/C: ${optionsFlow.putCallRatio.toFixed(2)}`}
                            color={optionsFlow.putCallRatio < 0.7 ? 'success' : optionsFlow.putCallRatio > 1.3 ? 'error' : 'default'}
                            size="small"
                        />
                    )}

                    {/* Signal overlay toggle */}
                    <Tooltip title="Toggle signal overlays">
                        <IconButton
                            size="small"
                            onClick={() => setShowSignalOverlays(!showSignalOverlays)}
                            color={showSignalOverlays ? 'primary' : 'default'}
                        >
                            <Layers />
                        </IconButton>
                    </Tooltip>

                    {/* Indicators menu */}
                    <Tooltip title="Indicators">
                        <IconButton
                            size="small"
                            onClick={(e) => setAnchorEl(e.currentTarget)}
                        >
                            <Addchart />
                        </IconButton>
                    </Tooltip>

                    {/* Refresh */}
                    <Tooltip title="Refresh">
                        <IconButton size="small" onClick={() => refetch()}>
                            <Refresh />
                        </IconButton>
                    </Tooltip>

                    {/* Fullscreen */}
                    <Tooltip title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}>
                        <IconButton
                            size="small"
                            onClick={() => setIsFullscreen(!isFullscreen)}
                        >
                            {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
                        </IconButton>
                    </Tooltip>
                </Stack>
            </Stack>

            {/* Chart Container */}
            <Box sx={{ position: 'relative' }}>
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

                <Box
                    ref={chartContainerRef}
                    sx={{
                        width: '100%',
                        height: isFullscreen ? 'calc(100vh - 140px)' : height,
                        opacity: isLoading ? 0.5 : 1,
                    }}
                />

                {/* Options Flow Overlay */}
                {showOptionsFlow && optionsFlow && (
                    <Box
                        sx={{
                            position: 'absolute',
                            top: 16,
                            right: 16,
                            background: alpha(theme.palette.background.paper, 0.9),
                            backdropFilter: 'blur(10px)',
                            border: `1px solid ${alpha(theme.palette.divider, 0.2)}`,
                            borderRadius: 1,
                            p: 2,
                            minWidth: 200,
                        }}
                    >
                        <Typography variant="subtitle2" gutterBottom>
                            Options Flow
                        </Typography>
                        <Stack spacing={1}>
                            <Stack direction="row" justifyContent="space-between">
                                <Stack direction="row" spacing={0.5} alignItems="center">
                                    <TrendingUp sx={{ fontSize: 16, color: theme.palette.success.main }} />
                                    <Typography variant="body2">Calls</Typography>
                                </Stack>
                                <Typography variant="body2" fontWeight="bold">
                                    {(optionsFlow.callVolume / 1000).toFixed(0)}K
                                </Typography>
                            </Stack>
                            <Stack direction="row" justifyContent="space-between">
                                <Stack direction="row" spacing={0.5} alignItems="center">
                                    <TrendingDown sx={{ fontSize: 16, color: theme.palette.error.main }} />
                                    <Typography variant="body2">Puts</Typography>
                                </Stack>
                                <Typography variant="body2" fontWeight="bold">
                                    {(optionsFlow.putVolume / 1000).toFixed(0)}K
                                </Typography>
                            </Stack>
                            <Divider />
                            <Typography variant="caption" color="text.secondary">
                                Unusual Activity: {optionsFlow.unusualActivity.length} trades
                            </Typography>
                        </Stack>
                    </Box>
                )}

                {/* Active Signals Overlay */}
                {showSignalOverlays && signals.length > 0 && (
                    <Box
                        sx={{
                            position: 'absolute',
                            top: 16,
                            left: 16,
                            background: alpha(theme.palette.background.paper, 0.9),
                            backdropFilter: 'blur(10px)',
                            border: `1px solid ${alpha(theme.palette.divider, 0.2)}`,
                            borderRadius: 1,
                            p: 2,
                            maxWidth: 300,
                        }}
                    >
                        <Stack direction="row" alignItems="center" spacing={1} mb={1}>
                            <Psychology sx={{ fontSize: 20 }} />
                            <Typography variant="subtitle2">
                                Active Signals ({signals.length})
                            </Typography>
                        </Stack>
                        <Stack spacing={1}>
                            {signals.slice(0, 3).map((signal) => (
                                <Chip
                                    key={signal.signal_id}
                                    label={`${signal.signal_type.replace('BUY_', '')} ${signal.strike_price} - ${signal.confidence}%`}
                                    size="small"
                                    color={signal.signal_type === 'BUY_CALL' ? 'success' : 'error'}
                                    onClick={() => onSignalClick?.(signal)}
                                    sx={{ cursor: 'pointer' }}
                                />
                            ))}
                        </Stack>
                    </Box>
                )}
            </Box>

            {/* Chart Footer */}
            <Stack
                direction="row"
                alignItems="center"
                justifyContent="space-between"
                sx={{
                    p: 1,
                    borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                }}
            >
                <Stack direction="row" spacing={2}>
                    <Typography variant="caption" color="text.secondary">
                        Last Update: {new Date().toLocaleTimeString()}
                    </Typography>
                </Stack>

                <Stack direction="row" spacing={1}>
                    {indicators.filter(i => i.active).map((ind) => (
                        <Chip
                            key={ind.id}
                            label={ind.name}
                            size="small"
                            variant="outlined"
                            onDelete={() => handleIndicatorToggle(ind.id)}
                        />
                    ))}
                </Stack>
            </Stack>

            {/* Indicators Menu */}
            <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={() => setAnchorEl(null)}
            >
                {indicators.map((indicator) => (
                    <MenuItem
                        key={indicator.id}
                        onClick={() => {
                            handleIndicatorToggle(indicator.id);
                            setAnchorEl(null);
                        }}
                    >
                        <Stack direction="row" spacing={2} alignItems="center" width="100%">
                            <Typography>{indicator.name}</Typography>
                            {indicator.active && <CheckCircle sx={{ fontSize: 16, color: theme.palette.success.main }} />}
                        </Stack>
                    </MenuItem>
                ))}
            </Menu>
        </Card>
    );
};

export default OptionsChart; 