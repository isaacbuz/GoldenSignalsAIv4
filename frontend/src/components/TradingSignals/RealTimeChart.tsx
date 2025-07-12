import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
    Box,
    Card,
    CardContent,
    IconButton,
    Tooltip,
    ButtonGroup,
    Button,
    Switch,
    FormControlLabel,
    Typography,
    Chip,
    useTheme,
    alpha,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    ZoomIn,
    ZoomOut,
    Settings,
    Fullscreen,
    Timeline,
    ShowChart,
    CandlestickChart,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { motion, AnimatePresence } from 'framer-motion';

// Professional styled components
const ChartContainer = styled(Box)(({ theme }) => ({
    position: 'relative',
    width: '100%',
    height: '600px',
    backgroundColor: theme.palette.background.paper,
    borderRadius: '12px',
    overflow: 'hidden',
    border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
}));

const ChartCanvas = styled(Box)(({ theme }) => ({
    width: '100%',
    height: '100%',
    background: `linear-gradient(135deg, ${alpha(theme.palette.background.default, 0.95)} 0%, ${alpha(theme.palette.background.paper, 0.98)} 100%)`,
    position: 'relative',
    overflow: 'hidden',
}));

const SignalAnnotation = styled(motion.div)(({ theme, signalType }) => ({
    position: 'absolute',
    zIndex: 10,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '40px',
    height: '40px',
    borderRadius: '50%',
    backgroundColor: signalType === 'buy' ? theme.palette.success.main : theme.palette.error.main,
    color: theme.palette.common.white,
    fontSize: '20px',
    fontWeight: 'bold',
    boxShadow: `0 4px 20px ${alpha(signalType === 'buy' ? theme.palette.success.main : theme.palette.error.main, 0.4)}`,
    cursor: 'pointer',
    '&::after': {
        content: '""',
        position: 'absolute',
        top: '100%',
        left: '50%',
        transform: 'translateX(-50%)',
        width: 0,
        height: 0,
        borderLeft: '8px solid transparent',
        borderRight: '8px solid transparent',
        borderTop: `8px solid ${signalType === 'buy' ? theme.palette.success.main : theme.palette.error.main}`,
    },
}));

const TechnicalIndicator = styled(motion.div)(({ theme }) => ({
    position: 'absolute',
    height: '2px',
    backgroundColor: theme.palette.primary.main,
    opacity: 0.8,
    '&.moving-average': {
        backgroundColor: '#FFD700',
    },
    '&.support': {
        backgroundColor: theme.palette.success.main,
    },
    '&.resistance': {
        backgroundColor: theme.palette.error.main,
    },
}));

const ChartControls = styled(Box)(({ theme }) => ({
    position: 'absolute',
    top: '16px',
    right: '16px',
    display: 'flex',
    gap: '8px',
    zIndex: 5,
}));

const TimeframeControls = styled(Box)(({ theme }) => ({
    position: 'absolute',
    bottom: '16px',
    left: '16px',
    display: 'flex',
    gap: '4px',
    zIndex: 5,
}));

const PriceTooltip = styled(motion.div)(({ theme }) => ({
    position: 'absolute',
    backgroundColor: alpha(theme.palette.background.paper, 0.95),
    border: `1px solid ${alpha(theme.palette.primary.main, 0.3)}`,
    borderRadius: '8px',
    padding: '12px',
    fontSize: '12px',
    color: theme.palette.text.primary,
    zIndex: 15,
    backdropFilter: 'blur(10px)',
    boxShadow: `0 8px 32px ${alpha(theme.palette.common.black, 0.2)}`,
}));

interface Signal {
    id: string;
    type: 'buy' | 'sell';
    price: number;
    time: number;
    confidence: number;
    reason: string;
    x: number;
    y: number;
}

interface TechnicalLine {
    id: string;
    type: 'moving-average' | 'support' | 'resistance' | 'trend';
    points: { x: number; y: number }[];
    color?: string;
}

interface RealTimeChartProps {
    symbol: string;
    data: any[];
    signals: Signal[];
    onSignalClick: (signal: Signal) => void;
    height?: number;
}

const RealTimeChart: React.FC<RealTimeChartProps> = ({
    symbol,
    data,
    signals,
    onSignalClick,
    height = 600,
}) => {
    const theme = useTheme();
    const chartRef = useRef<HTMLDivElement>(null);
    const [timeframe, setTimeframe] = useState('5m');
    const [chartType, setChartType] = useState('candlestick');
    const [showIndicators, setShowIndicators] = useState(true);
    const [hoveredPoint, setHoveredPoint] = useState<any>(null);
    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
    const [technicalLines, setTechnicalLines] = useState<TechnicalLine[]>([]);
    const [animationPhase, setAnimationPhase] = useState(0);

    // Simulate real-time candlestick data
    const [candlestickData, setCandlestickData] = useState([
        { time: Date.now() - 300000, open: 150.25, high: 152.80, low: 149.90, close: 151.45, volume: 2500000 },
        { time: Date.now() - 240000, open: 151.45, high: 153.20, low: 150.85, close: 152.10, volume: 1800000 },
        { time: Date.now() - 180000, open: 152.10, high: 154.75, low: 151.60, close: 154.20, volume: 3200000 },
        { time: Date.now() - 120000, open: 154.20, high: 155.90, low: 153.40, close: 155.35, volume: 2100000 },
        { time: Date.now() - 60000, open: 155.35, high: 157.25, low: 154.80, close: 156.90, volume: 2800000 },
        { time: Date.now(), open: 156.90, high: 158.45, low: 156.20, close: 157.85, volume: 1900000 },
    ]);

    // Animate technical analysis drawing
    useEffect(() => {
        const phases = [
            () => {
                // Phase 1: Draw moving average
                setTechnicalLines(prev => [...prev, {
                    id: 'ma-20',
                    type: 'moving-average',
                    points: candlestickData.map((candle, index) => ({
                        x: (index / (candlestickData.length - 1)) * 100,
                        y: 50 + (Math.sin(index * 0.5) * 20),
                    })),
                }]);
            },
            () => {
                // Phase 2: Draw support line
                setTechnicalLines(prev => [...prev, {
                    id: 'support',
                    type: 'support',
                    points: [
                        { x: 0, y: 75 },
                        { x: 100, y: 72 },
                    ],
                }]);
            },
            () => {
                // Phase 3: Draw resistance line
                setTechnicalLines(prev => [...prev, {
                    id: 'resistance',
                    type: 'resistance',
                    points: [
                        { x: 20, y: 25 },
                        { x: 100, y: 30 },
                    ],
                }]);
            },
        ];

        const timer = setTimeout(() => {
            if (animationPhase < phases.length) {
                phases[animationPhase]();
                setAnimationPhase(prev => prev + 1);
            }
        }, 1000 + animationPhase * 800);

        return () => clearTimeout(timer);
    }, [animationPhase, candlestickData]);

    // Real-time data updates
    useEffect(() => {
        const interval = setInterval(() => {
            setCandlestickData(prev => {
                const lastCandle = prev[prev.length - 1];
                const newPrice = lastCandle.close + (Math.random() - 0.5) * 2;
                const newCandle = {
                    time: Date.now(),
                    open: lastCandle.close,
                    high: Math.max(lastCandle.close, newPrice) + Math.random() * 0.5,
                    low: Math.min(lastCandle.close, newPrice) - Math.random() * 0.5,
                    close: newPrice,
                    volume: Math.floor(Math.random() * 2000000) + 1000000,
                };
                return [...prev.slice(-20), newCandle];
            });
        }, 3000);

        return () => clearInterval(interval);
    }, []);

    const handleMouseMove = useCallback((event: React.MouseEvent) => {
        if (!chartRef.current) return;

        const rect = chartRef.current.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        setMousePosition({ x, y });

        // Simulate finding nearest data point
        const dataIndex = Math.floor((x / rect.width) * candlestickData.length);
        const candle = candlestickData[dataIndex];

        if (candle) {
            setHoveredPoint({
                ...candle,
                x: event.clientX,
                y: event.clientY,
            });
        }
    }, [candlestickData]);

    const handleMouseLeave = () => {
        setHoveredPoint(null);
    };

    const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];

    return (
        <Card sx={{ height: height + 100, position: 'relative' }}>
            <CardContent sx={{ p: 0, height: '100%' }}>
                <ChartContainer>
                    <ChartCanvas
                        ref={chartRef}
                        onMouseMove={handleMouseMove}
                        onMouseLeave={handleMouseLeave}
                    >
                        {/* Chart Controls */}
                        <ChartControls>
                            <Tooltip title="Chart Settings">
                                <IconButton size="small" sx={{ bgcolor: alpha(theme.palette.background.paper, 0.8) }}>
                                    <Settings />
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="Fullscreen">
                                <IconButton size="small" sx={{ bgcolor: alpha(theme.palette.background.paper, 0.8) }}>
                                    <Fullscreen />
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="Chart Type">
                                <IconButton
                                    size="small"
                                    sx={{ bgcolor: alpha(theme.palette.background.paper, 0.8) }}
                                    onClick={() => setChartType(prev => prev === 'candlestick' ? 'line' : 'candlestick')}
                                >
                                    {chartType === 'candlestick' ? <CandlestickChart /> : <ShowChart />}
                                </IconButton>
                            </Tooltip>
                        </ChartControls>

                        {/* Timeframe Controls */}
                        <TimeframeControls>
                            <ButtonGroup size="small" variant="contained">
                                {timeframes.map((tf) => (
                                    <Button
                                        key={tf}
                                        variant={timeframe === tf ? 'contained' : 'outlined'}
                                        onClick={() => setTimeframe(tf)}
                                        sx={{ minWidth: '40px' }}
                                    >
                                        {tf}
                                    </Button>
                                ))}
                            </ButtonGroup>
                        </TimeframeControls>

                        {/* Candlestick Chart Visualization */}
                        <Box sx={{ position: 'relative', width: '100%', height: '100%', p: 2 }}>
                            {/* Simulated candlesticks */}
                            {candlestickData.map((candle, index) => {
                                const x = (index / (candlestickData.length - 1)) * 90 + 5;
                                const bodyHeight = Math.abs(candle.close - candle.open) / 2;
                                const bodyTop = 50 - bodyHeight / 2;
                                const isGreen = candle.close > candle.open;

                                return (
                                    <motion.div
                                        key={candle.time}
                                        initial={{ opacity: 0, scaleY: 0 }}
                                        animate={{ opacity: 1, scaleY: 1 }}
                                        transition={{ duration: 0.5, delay: index * 0.1 }}
                                        style={{
                                            position: 'absolute',
                                            left: `${x}%`,
                                            top: `${bodyTop}%`,
                                            width: '8px',
                                            height: `${bodyHeight}%`,
                                            backgroundColor: isGreen ? theme.palette.success.main : theme.palette.error.main,
                                            border: `1px solid ${isGreen ? theme.palette.success.dark : theme.palette.error.dark}`,
                                        }}
                                    />
                                );
                            })}

                            {/* Technical Indicators */}
                            <AnimatePresence>
                                {showIndicators && technicalLines.map((line) => (
                                    <TechnicalIndicator
                                        key={line.id}
                                        className={line.type}
                                        initial={{ pathLength: 0, opacity: 0 }}
                                        animate={{ pathLength: 1, opacity: 0.8 }}
                                        exit={{ opacity: 0 }}
                                        transition={{ duration: 1.5, ease: "easeInOut" }}
                                        style={{
                                            left: `${line.points[0]?.x || 0}%`,
                                            top: `${line.points[0]?.y || 50}%`,
                                            width: `${Math.abs((line.points[1]?.x || 0) - (line.points[0]?.x || 0))}%`,
                                            transform: `rotate(${Math.atan2(
                                                (line.points[1]?.y || 0) - (line.points[0]?.y || 0),
                                                (line.points[1]?.x || 0) - (line.points[0]?.x || 0)
                                            )}rad)`,
                                        }}
                                    />
                                ))}
                            </AnimatePresence>

                            {/* Signal Annotations */}
                            <AnimatePresence>
                                {signals.map((signal) => (
                                    <SignalAnnotation
                                        key={signal.id}
                                        signalType={signal.type}
                                        initial={{ scale: 0, opacity: 0 }}
                                        animate={{
                                            scale: [0, 1.2, 1],
                                            opacity: 1,
                                            boxShadow: [
                                                `0 4px 20px ${alpha(signal.type === 'buy' ? theme.palette.success.main : theme.palette.error.main, 0.4)}`,
                                                `0 8px 40px ${alpha(signal.type === 'buy' ? theme.palette.success.main : theme.palette.error.main, 0.8)}`,
                                                `0 4px 20px ${alpha(signal.type === 'buy' ? theme.palette.success.main : theme.palette.error.main, 0.4)}`,
                                            ]
                                        }}
                                        exit={{ scale: 0, opacity: 0 }}
                                        transition={{ duration: 0.6, times: [0, 0.6, 1] }}
                                        style={{
                                            left: `${signal.x}%`,
                                            top: `${signal.y}%`,
                                        }}
                                        onClick={() => onSignalClick(signal)}
                                        whileHover={{ scale: 1.1 }}
                                        whileTap={{ scale: 0.95 }}
                                    >
                                        {signal.type === 'buy' ? '↑' : '↓'}
                                    </SignalAnnotation>
                                ))}
                            </AnimatePresence>

                            {/* Live Price Line */}
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                style={{
                                    position: 'absolute',
                                    right: '5%',
                                    top: '45%',
                                    width: '2px',
                                    height: '10%',
                                    backgroundColor: theme.palette.primary.main,
                                    zIndex: 5,
                                }}
                            />

                            {/* Current Price Label */}
                            <motion.div
                                initial={{ x: 20, opacity: 0 }}
                                animate={{ x: 0, opacity: 1 }}
                                style={{
                                    position: 'absolute',
                                    right: '2%',
                                    top: '42%',
                                    backgroundColor: theme.palette.primary.main,
                                    color: theme.palette.primary.contrastText,
                                    padding: '4px 8px',
                                    borderRadius: '4px',
                                    fontSize: '12px',
                                    fontWeight: 'bold',
                                }}
                            >
                                ${candlestickData[candlestickData.length - 1]?.close.toFixed(2)}
                            </motion.div>
                        </Box>

                        {/* Price Tooltip */}
                        <AnimatePresence>
                            {hoveredPoint && (
                                <PriceTooltip
                                    initial={{ opacity: 0, scale: 0.8 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    exit={{ opacity: 0, scale: 0.8 }}
                                    style={{
                                        left: hoveredPoint.x + 10,
                                        top: hoveredPoint.y - 60,
                                    }}
                                >
                                    <Typography variant="caption" display="block">
                                        <strong>Open:</strong> ${hoveredPoint.open?.toFixed(2)}
                                    </Typography>
                                    <Typography variant="caption" display="block">
                                        <strong>High:</strong> ${hoveredPoint.high?.toFixed(2)}
                                    </Typography>
                                    <Typography variant="caption" display="block">
                                        <strong>Low:</strong> ${hoveredPoint.low?.toFixed(2)}
                                    </Typography>
                                    <Typography variant="caption" display="block">
                                        <strong>Close:</strong> ${hoveredPoint.close?.toFixed(2)}
                                    </Typography>
                                    <Typography variant="caption" display="block">
                                        <strong>Volume:</strong> {(hoveredPoint.volume / 1000000).toFixed(1)}M
                                    </Typography>
                                </PriceTooltip>
                            )}
                        </AnimatePresence>
                    </ChartCanvas>

                    {/* Chart Footer with Indicators Toggle */}
                    <Box sx={{
                        position: 'absolute',
                        bottom: '60px',
                        right: '16px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1,
                    }}>
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={showIndicators}
                                    onChange={(e) => setShowIndicators(e.target.checked)}
                                    size="small"
                                />
                            }
                            label="Technical Indicators"
                            sx={{ fontSize: '12px' }}
                        />
                    </Box>
                </ChartContainer>
            </CardContent>
        </Card>
    );
};

export default RealTimeChart; 