import React, { useEffect, useRef, useState } from 'react';
import { Box, Paper, Typography, Chip, Stack, Tooltip, IconButton, Badge } from '@mui/material';
import {
    Timeline as TimelineIcon,
    TrendingUp as TrendingUpIcon,
    TrendingDown as TrendingDownIcon,
    CheckCircle as CheckIcon,
    Warning as WarningIcon,
    Info as InfoIcon
} from '@mui/icons-material';

interface FibonacciLevel {
    level: number;
    price: number;
    label: string;
    color: string;
    strength: 'strong' | 'medium' | 'weak';
}

interface SwingPoint {
    price: number;
    time: number;
    type: 'high' | 'low';
}

interface FibonacciRetracementProps {
    chartData: any[];
    onLevelsCalculated?: (levels: FibonacciLevel[]) => void;
    onSignalGenerated?: (signal: any) => void;
    showLabels?: boolean;
    autoDetect?: boolean;
}

const FibonacciRetracement: React.FC<FibonacciRetracementProps> = ({
    chartData,
    onLevelsCalculated,
    onSignalGenerated,
    showLabels = true,
    autoDetect = true
}) => {
    const [fibLevels, setFibLevels] = useState<FibonacciLevel[]>([]);
    const [swingPoints, setSwingPoints] = useState<{ high: SwingPoint | null; low: SwingPoint | null }>({
        high: null,
        low: null
    });
    const [currentTrend, setCurrentTrend] = useState<'up' | 'down' | 'neutral'>('neutral');
    const [activeLevel, setActiveLevel] = useState<FibonacciLevel | null>(null);

    // Standard Fibonacci ratios
    const fibRatios = [
        { ratio: 0, label: '0%', color: '#4CAF50', strength: 'strong' as const },
        { ratio: 0.236, label: '23.6%', color: '#8BC34A', strength: 'weak' as const },
        { ratio: 0.382, label: '38.2%', color: '#FFC107', strength: 'medium' as const },
        { ratio: 0.5, label: '50%', color: '#FF9800', strength: 'medium' as const },
        { ratio: 0.618, label: '61.8%', color: '#FF5722', strength: 'strong' as const }, // Golden ratio
        { ratio: 0.786, label: '78.6%', color: '#F44336', strength: 'medium' as const },
        { ratio: 1, label: '100%', color: '#9C27B0', strength: 'strong' as const }
    ];

    // Fibonacci extensions for profit targets
    const fibExtensions = [
        { ratio: 1.272, label: '127.2%', color: '#3F51B5', strength: 'medium' as const },
        { ratio: 1.618, label: '161.8%', color: '#2196F3', strength: 'strong' as const },
        { ratio: 2.618, label: '261.8%', color: '#00BCD4', strength: 'weak' as const }
    ];

    // Auto-detect swing points
    const detectSwingPoints = () => {
        if (!chartData || chartData.length < 20) return;

        const lookback = 10; // Number of candles to look back/forward
        let highPoint: SwingPoint | null = null;
        let lowPoint: SwingPoint | null = null;

        // Find swing high
        for (let i = lookback; i < chartData.length - lookback; i++) {
            const current = chartData[i];
            let isSwingHigh = true;

            // Check if current high is higher than surrounding candles
            for (let j = i - lookback; j <= i + lookback; j++) {
                if (j !== i && chartData[j].high >= current.high) {
                    isSwingHigh = false;
                    break;
                }
            }

            if (isSwingHigh && (!highPoint || current.high > highPoint.price)) {
                highPoint = {
                    price: current.high,
                    time: current.time,
                    type: 'high'
                };
            }
        }

        // Find swing low
        for (let i = lookback; i < chartData.length - lookback; i++) {
            const current = chartData[i];
            let isSwingLow = true;

            // Check if current low is lower than surrounding candles
            for (let j = i - lookback; j <= i + lookback; j++) {
                if (j !== i && chartData[j].low <= current.low) {
                    isSwingLow = false;
                    break;
                }
            }

            if (isSwingLow && (!lowPoint || current.low < lowPoint.price)) {
                lowPoint = {
                    price: current.low,
                    time: current.time,
                    type: 'low'
                };
            }
        }

        if (highPoint && lowPoint) {
            setSwingPoints({ high: highPoint, low: lowPoint });

            // Determine trend direction
            const trend = highPoint.time > lowPoint.time ? 'down' : 'up';
            setCurrentTrend(trend);

            // Calculate Fibonacci levels
            calculateFibonacciLevels(highPoint, lowPoint, trend);
        }
    };

    // Calculate Fibonacci retracement levels
    const calculateFibonacciLevels = (high: SwingPoint, low: SwingPoint, trend: 'up' | 'down') => {
        const range = high.price - low.price;
        const levels: FibonacciLevel[] = [];

        // Calculate retracement levels
        fibRatios.forEach(({ ratio, label, color, strength }) => {
            const level = trend === 'up'
                ? high.price - (range * ratio)  // For uptrend, measure from high
                : low.price + (range * ratio);  // For downtrend, measure from low

            levels.push({
                level: ratio,
                price: level,
                label,
                color,
                strength
            });
        });

        // Add extension levels for profit targets
        if (trend === 'up') {
            fibExtensions.forEach(({ ratio, label, color, strength }) => {
                const level = high.price + (range * (ratio - 1));
                levels.push({
                    level: ratio,
                    price: level,
                    label: `EXT ${label}`,
                    color,
                    strength
                });
            });
        } else {
            fibExtensions.forEach(({ ratio, label, color, strength }) => {
                const level = low.price - (range * (ratio - 1));
                levels.push({
                    level: ratio,
                    price: level,
                    label: `EXT ${label}`,
                    color,
                    strength
                });
            });
        }

        setFibLevels(levels);
        if (onLevelsCalculated) {
            onLevelsCalculated(levels);
        }

        // Check for potential signals
        checkForSignals(levels);
    };

    // Check if price is near any Fibonacci level
    const checkForSignals = (levels: FibonacciLevel[]) => {
        if (!chartData || chartData.length === 0) return;

        const currentPrice = chartData[chartData.length - 1].close;
        const tolerance = 0.002; // 0.2% tolerance

        levels.forEach(level => {
            const distance = Math.abs(currentPrice - level.price) / level.price;

            if (distance < tolerance) {
                setActiveLevel(level);

                // Generate signal based on level strength and trend
                if (level.strength === 'strong' && onSignalGenerated) {
                    const signal = {
                        type: currentTrend === 'up' ? 'BUY' : 'SELL',
                        level: level.label,
                        price: level.price,
                        currentPrice,
                        strength: level.strength,
                        confidence: level.level === 0.618 ? 'HIGH' : 'MEDIUM', // Golden ratio has highest confidence
                        stopLoss: calculateStopLoss(level, currentTrend),
                        takeProfit: calculateTakeProfit(level, currentTrend, levels),
                        reason: `Price approaching ${level.label} Fibonacci level (${level.strength} support/resistance)`,
                        timestamp: new Date().toISOString()
                    };

                    onSignalGenerated(signal);
                }
            }
        });
    };

    // Calculate stop loss based on Fibonacci level
    const calculateStopLoss = (level: FibonacciLevel, trend: 'up' | 'down'): number => {
        const buffer = 0.002; // 0.2% buffer beyond level

        if (trend === 'up') {
            // For uptrend, stop loss below the level
            return level.price * (1 - buffer);
        } else {
            // For downtrend, stop loss above the level
            return level.price * (1 + buffer);
        }
    };

    // Calculate take profit using Fibonacci extensions
    const calculateTakeProfit = (
        entryLevel: FibonacciLevel,
        trend: 'up' | 'down',
        allLevels: FibonacciLevel[]
    ): number[] => {
        const extensionLevels = allLevels.filter(l => l.label.includes('EXT'));
        const takeProfits: number[] = [];

        if (trend === 'up') {
            // For uptrend, take profits at extension levels above entry
            extensionLevels.forEach(ext => {
                if (ext.price > entryLevel.price) {
                    takeProfits.push(ext.price);
                }
            });
        } else {
            // For downtrend, take profits at extension levels below entry
            extensionLevels.forEach(ext => {
                if (ext.price < entryLevel.price) {
                    takeProfits.push(ext.price);
                }
            });
        }

        return takeProfits.slice(0, 3); // Return top 3 targets
    };

    // Effect to auto-detect swing points when data changes
    useEffect(() => {
        if (autoDetect && chartData && chartData.length > 0) {
            detectSwingPoints();
        }
    }, [chartData, autoDetect]);

    // Effect to continuously check for signals
    useEffect(() => {
        if (fibLevels.length > 0) {
            const interval = setInterval(() => {
                checkForSignals(fibLevels);
            }, 5000); // Check every 5 seconds

            return () => clearInterval(interval);
        }
    }, [fibLevels, chartData]);

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Header */}
            <Box sx={{
                p: 2,
                borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)'
            }}>
                <Stack direction="row" spacing={2} alignItems="center">
                    <TimelineIcon sx={{ color: '#8B5CF6' }} />
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        Fibonacci Retracement Analysis
                    </Typography>
                    <Chip
                        label={currentTrend === 'up' ? 'UPTREND' : currentTrend === 'down' ? 'DOWNTREND' : 'NEUTRAL'}
                        color={currentTrend === 'up' ? 'success' : currentTrend === 'down' ? 'error' : 'default'}
                        size="small"
                        icon={currentTrend === 'up' ? <TrendingUpIcon /> : currentTrend === 'down' ? <TrendingDownIcon /> : undefined}
                    />
                </Stack>
            </Box>

            {/* Swing Points Info */}
            {swingPoints.high && swingPoints.low && (
                <Box sx={{ p: 2, borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}>
                    <Stack spacing={1}>
                        <Typography variant="body2" color="text.secondary">
                            Detected Swing Points:
                        </Typography>
                        <Stack direction="row" spacing={2}>
                            <Chip
                                label={`High: $${swingPoints.high.price.toFixed(2)}`}
                                size="small"
                                sx={{ backgroundColor: 'rgba(239, 68, 68, 0.2)' }}
                            />
                            <Chip
                                label={`Low: $${swingPoints.low.price.toFixed(2)}`}
                                size="small"
                                sx={{ backgroundColor: 'rgba(34, 197, 94, 0.2)' }}
                            />
                            <Chip
                                label={`Range: $${(swingPoints.high.price - swingPoints.low.price).toFixed(2)}`}
                                size="small"
                                variant="outlined"
                            />
                        </Stack>
                    </Stack>
                </Box>
            )}

            {/* Fibonacci Levels */}
            <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
                <Stack spacing={1}>
                    {fibLevels.map((level, index) => (
                        <Paper
                            key={index}
                            sx={{
                                p: 2,
                                backgroundColor: activeLevel?.level === level.level
                                    ? 'rgba(139, 92, 246, 0.2)'
                                    : 'rgba(255, 255, 255, 0.02)',
                                border: '1px solid',
                                borderColor: activeLevel?.level === level.level
                                    ? level.color
                                    : 'rgba(255, 255, 255, 0.1)',
                                transition: 'all 0.3s ease',
                                '&:hover': {
                                    backgroundColor: 'rgba(255, 255, 255, 0.05)',
                                    borderColor: level.color
                                }
                            }}
                        >
                            <Stack direction="row" alignItems="center" justifyContent="space-between">
                                <Stack direction="row" spacing={2} alignItems="center">
                                    <Box
                                        sx={{
                                            width: 4,
                                            height: 40,
                                            backgroundColor: level.color,
                                            borderRadius: 1
                                        }}
                                    />
                                    <Box>
                                        <Typography variant="body1" sx={{ fontWeight: 600 }}>
                                            {level.label}
                                        </Typography>
                                        <Typography variant="body2" color="text.secondary">
                                            ${level.price.toFixed(2)}
                                        </Typography>
                                    </Box>
                                </Stack>

                                <Stack direction="row" spacing={1} alignItems="center">
                                    {level.strength === 'strong' && (
                                        <Tooltip title="Strong level - High probability of reaction">
                                            <Badge badgeContent="STRONG" color="error">
                                                <CheckIcon sx={{ color: '#22C55E' }} />
                                            </Badge>
                                        </Tooltip>
                                    )}
                                    {level.level === 0.618 && (
                                        <Tooltip title="Golden Ratio - Most significant Fibonacci level">
                                            <Chip
                                                label="GOLDEN RATIO"
                                                size="small"
                                                sx={{
                                                    backgroundColor: 'rgba(255, 215, 0, 0.2)',
                                                    color: '#FFD700',
                                                    fontWeight: 600
                                                }}
                                            />
                                        </Tooltip>
                                    )}
                                    {level.label.includes('EXT') && (
                                        <Tooltip title="Extension level - Potential profit target">
                                            <InfoIcon sx={{ color: '#3B82F6' }} />
                                        </Tooltip>
                                    )}
                                </Stack>
                            </Stack>

                            {activeLevel?.level === level.level && (
                                <Box sx={{ mt: 2, p: 1, backgroundColor: 'rgba(139, 92, 246, 0.1)', borderRadius: 1 }}>
                                    <Typography variant="caption" sx={{ color: '#8B5CF6' }}>
                                        ⚡ Price is approaching this level - Monitor for potential reversal
                                    </Typography>
                                </Box>
                            )}
                        </Paper>
                    ))}
                </Stack>
            </Box>

            {/* Legend */}
            {showLabels && (
                <Box sx={{ p: 2, borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
                    <Typography variant="caption" color="text.secondary" gutterBottom>
                        Key Levels:
                    </Typography>
                    <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: 'wrap' }}>
                        <Chip label="23.6% - Shallow retracement" size="small" variant="outlined" />
                        <Chip label="38.2% - Moderate retracement" size="small" variant="outlined" />
                        <Chip label="50% - Psychological level" size="small" variant="outlined" />
                        <Chip label="61.8% - Golden ratio (strongest)" size="small" variant="outlined" sx={{ borderColor: '#FFD700' }} />
                        <Chip label="78.6% - Deep retracement" size="small" variant="outlined" />
                    </Stack>
                </Box>
            )}
        </Box>
    );
};

export default FibonacciRetracement; 