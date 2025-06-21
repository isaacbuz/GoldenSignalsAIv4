/**
 * MiniChart Component - Lightweight Chart for Signal Visualization
 * 
 * Displays price action with entry, stop loss, and target levels
 */

import React, { useEffect, useRef } from 'react';
import { Box, useTheme, alpha } from '@mui/material';
import { createChart, IChartApi, ISeriesApi, Time } from 'lightweight-charts';

interface MiniChartProps {
    symbol: string;
    entryPrice: number;
    stopLoss: number;
    targets: number[];
    signalType: 'BUY_CALL' | 'BUY_PUT';
    height?: number;
}

const MiniChart: React.FC<MiniChartProps> = ({
    symbol,
    entryPrice,
    stopLoss,
    targets,
    signalType,
    height = 400,
}) => {
    const theme = useTheme();
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);

    useEffect(() => {
        if (!chartContainerRef.current) return;

        // Create chart
        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height,
            layout: {
                background: { color: 'transparent' },
                textColor: theme.palette.text.secondary,
            },
            grid: {
                vertLines: { color: alpha(theme.palette.divider, 0.1) },
                horzLines: { color: alpha(theme.palette.divider, 0.1) },
            },
            crosshair: {
                mode: 1,
            },
            rightPriceScale: {
                borderColor: alpha(theme.palette.divider, 0.1),
            },
            timeScale: {
                borderColor: alpha(theme.palette.divider, 0.1),
                timeVisible: true,
            },
        });

        chartRef.current = chart;

        // Generate sample data around entry price
        const generateData = () => {
            const data = [];
            const basePrice = entryPrice;
            const now = Math.floor(Date.now() / 1000);

            for (let i = -50; i < 50; i++) {
                const time = now + i * 900; // 15-minute intervals
                const volatility = 0.02;
                const trend = signalType === 'BUY_CALL' ? 0.0001 * i : -0.0001 * i;

                const open = basePrice * (1 + (Math.random() - 0.5) * volatility + trend);
                const close = basePrice * (1 + (Math.random() - 0.5) * volatility + trend);
                const high = Math.max(open, close) * (1 + Math.random() * volatility * 0.5);
                const low = Math.min(open, close) * (1 - Math.random() * volatility * 0.5);

                data.push({
                    time: time as Time,
                    open,
                    high,
                    low,
                    close,
                });
            }

            return data;
        };

        // Add candlestick series
        const candlestickSeries = chart.addCandlestickSeries({
            upColor: theme.palette.success.main,
            downColor: theme.palette.error.main,
            borderUpColor: theme.palette.success.main,
            borderDownColor: theme.palette.error.main,
            wickUpColor: theme.palette.success.main,
            wickDownColor: theme.palette.error.main,
        });

        candlestickSeries.setData(generateData());

        // Add entry line
        candlestickSeries.createPriceLine({
            price: entryPrice,
            color: theme.palette.primary.main,
            lineWidth: 2,
            lineStyle: 2,
            axisLabelVisible: true,
            title: 'Entry',
        });

        // Add stop loss line
        candlestickSeries.createPriceLine({
            price: stopLoss,
            color: theme.palette.error.main,
            lineWidth: 2,
            lineStyle: 2,
            axisLabelVisible: true,
            title: 'Stop Loss',
        });

        // Add target lines
        targets.forEach((target, index) => {
            candlestickSeries.createPriceLine({
                price: target,
                color: theme.palette.success.main,
                lineWidth: 2,
                lineStyle: 2,
                axisLabelVisible: true,
                title: `Target ${index + 1}`,
            });
        });

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
    }, [theme, entryPrice, stopLoss, targets, signalType, height]);

    return (
        <Box
            ref={chartContainerRef}
            sx={{
                width: '100%',
                height,
                position: 'relative',
                borderRadius: 1,
                overflow: 'hidden',
            }}
        />
    );
};

export default MiniChart; 