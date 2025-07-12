/**
 * MiniChart Component - Lightweight Chart for Signal Visualization
 * 
 * Displays price action with entry, stop loss, and target levels
 */

import React, { useMemo } from 'react';
import { Box, Typography, useTheme, alpha } from '@mui/material';
import { LineChart, Line, ResponsiveContainer, YAxis, Tooltip } from 'recharts';
import { TrendingUp, TrendingDown, TrendingFlat } from '@mui/icons-material';

interface MiniChartProps {
    symbol: string;
    data: Array<{ time: string; value: number }>;
    height?: number;
    showLabel?: boolean;
    color?: 'primary' | 'success' | 'error' | 'warning';
    sparkline?: boolean;
}

export const MiniChart: React.FC<MiniChartProps> = ({
    symbol,
    data,
    height = 50,
    showLabel = true,
    color = 'primary',
    sparkline = false,
}) => {
    const theme = useTheme();

    const { currentPrice, change, changePercent, trend } = useMemo(() => {
        if (!data || data.length === 0) {
            return { currentPrice: 0, change: 0, changePercent: 0, trend: 'flat' };
        }

        const current = data[data.length - 1].value;
        const previous = data[0].value;
        const changeAmount = current - previous;
        const changePct = (changeAmount / previous) * 100;

        return {
            currentPrice: current,
            change: changeAmount,
            changePercent: changePct,
            trend: changeAmount > 0 ? 'up' : changeAmount < 0 ? 'down' : 'flat',
        };
    }, [data]);

    const chartColor = useMemo(() => {
        if (color === 'primary') {
            return trend === 'up' ? theme.palette.success.main : theme.palette.error.main;
        }
        return theme.palette[color].main;
    }, [color, trend, theme]);

    const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : TrendingFlat;

    if (!data || data.length === 0) {
        return (
            <Box sx={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Typography variant="caption" color="text.secondary">No data</Typography>
            </Box>
        );
    }

    return (
        <Box sx={{ width: '100%', height: showLabel ? height + 30 : height }}>
            {showLabel && (
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
                    <Typography variant="body2" fontWeight="bold">{symbol}</Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Typography
                            variant="body2"
                            sx={{
                                color: trend === 'up' ? theme.palette.success.main : trend === 'down' ? theme.palette.error.main : theme.palette.text.secondary,
                                fontWeight: 'bold',
                            }}
                        >
                            ${currentPrice.toFixed(2)}
                        </Typography>
                        <TrendIcon
                            fontSize="small"
                            sx={{
                                color: trend === 'up' ? theme.palette.success.main : trend === 'down' ? theme.palette.error.main : theme.palette.text.secondary,
                            }}
                        />
                        <Typography
                            variant="caption"
                            sx={{
                                color: trend === 'up' ? theme.palette.success.main : trend === 'down' ? theme.palette.error.main : theme.palette.text.secondary,
                            }}
                        >
                            {change >= 0 ? '+' : ''}{changePercent.toFixed(2)}%
                        </Typography>
                    </Box>
                </Box>
            )}

            <ResponsiveContainer width="100%" height={height}>
                <LineChart data={data} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
                    {!sparkline && (
                        <YAxis
                            hide
                            domain={['dataMin - 1', 'dataMax + 1']}
                        />
                    )}
                    <Tooltip
                        contentStyle={{
                            backgroundColor: alpha(theme.palette.background.paper, 0.95),
                            border: `1px solid ${alpha(theme.palette.divider, 0.2)}`,
                            borderRadius: theme.shape.borderRadius,
                            padding: theme.spacing(1),
                        }}
                        labelStyle={{ color: theme.palette.text.primary }}
                        formatter={(value: number) => [`$${value.toFixed(2)}`, 'Price']}
                        labelFormatter={(label) => `Time: ${label}`}
                    />
                    <Line
                        type="monotone"
                        dataKey="value"
                        stroke={chartColor}
                        strokeWidth={sparkline ? 1 : 2}
                        dot={false}
                        animationDuration={300}
                    />
                </LineChart>
            </ResponsiveContainer>
        </Box>
    );
};

export default MiniChart; 