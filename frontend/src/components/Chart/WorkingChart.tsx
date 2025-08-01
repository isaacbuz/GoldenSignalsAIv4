/**
 * Working Chart Component
 * A simplified, functional chart that displays real market data
 */

import React, { useEffect, useRef, useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  CircularProgress,
  Alert,
  useTheme,
  alpha,
  Stack,
  Chip,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';
import { createChart, ColorType, IChartApi, ISeriesApi } from 'lightweight-charts';
import axios from 'axios';

interface ChartData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface WorkingChartProps {
  symbol: string;
  height?: number;
  showVolume?: boolean;
  showIndicators?: boolean;
}

export const WorkingChart: React.FC<WorkingChartProps> = ({
  symbol = 'AAPL',
  height = 500,
  showVolume = true,
  showIndicators = true
}) => {
  const theme = useTheme();
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const ma20SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ma50SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [latestPrice, setLatestPrice] = useState<number | null>(null);
  const [priceChange, setPriceChange] = useState<number>(0);
  const [aiSignal, setAiSignal] = useState<'BUY' | 'SELL' | 'HOLD' | null>(null);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: theme.palette.text.primary,
      },
      grid: {
        vertLines: { color: alpha(theme.palette.divider, 0.1) },
        horzLines: { color: alpha(theme.palette.divider, 0.1) },
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
      timeScale: {
        borderColor: alpha(theme.palette.divider, 0.2),
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: alpha(theme.palette.divider, 0.2),
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: alpha(theme.palette.primary.main, 0.4),
          width: 1,
          style: 2,
          labelBackgroundColor: theme.palette.primary.main,
        },
        horzLine: {
          color: alpha(theme.palette.primary.main, 0.4),
          width: 1,
          style: 2,
          labelBackgroundColor: theme.palette.primary.main,
        },
      },
    });

    // Add candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: theme.palette.success.main,
      downColor: theme.palette.error.main,
      borderUpColor: theme.palette.success.main,
      borderDownColor: theme.palette.error.main,
      wickUpColor: theme.palette.success.main,
      wickDownColor: theme.palette.error.main,
    });

    // Add volume series
    const volumeSeries = chart.addHistogramSeries({
      color: alpha(theme.palette.info.main, 0.3),
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'volume',
    });

    chart.priceScale('volume').applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    // Add moving average lines
    const ma20Series = chart.addLineSeries({
      color: theme.palette.info.main,
      lineWidth: 2,
      title: 'MA20',
      crosshairMarkerVisible: false,
    });

    const ma50Series = chart.addLineSeries({
      color: theme.palette.warning.main,
      lineWidth: 2,
      title: 'MA50',
      crosshairMarkerVisible: false,
    });

    chartRef.current = chart;
    candlestickSeriesRef.current = candlestickSeries;
    volumeSeriesRef.current = volumeSeries;
    ma20SeriesRef.current = ma20Series;
    ma50SeriesRef.current = ma50Series;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth
        });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [theme, height]);

  // Fetch and display data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await axios.get(
          `http://localhost:8000/api/v1/market-data/${symbol}/historical`,
          {
            params: {
              period: '5d',
              interval: '15m'
            }
          }
        );

        const data = response.data.data || response.data;

        if (!Array.isArray(data) || data.length === 0) {
          throw new Error('No data available');
        }

        // Transform data for lightweight-charts
        const chartData: ChartData[] = data.map((item: any) => ({
          time: typeof item.time === 'number'
            ? item.time
            : Math.floor(new Date(item.time).getTime() / 1000),
          open: parseFloat(item.open),
          high: parseFloat(item.high),
          low: parseFloat(item.low),
          close: parseFloat(item.close),
          volume: parseInt(item.volume) || 0,
        }));

        // Sort by time
        chartData.sort((a, b) => a.time - b.time);

        // Update chart
        if (candlestickSeriesRef.current && chartData.length > 0) {
          candlestickSeriesRef.current.setData(chartData);

          // Update volume
          if (volumeSeriesRef.current) {
            const volumeData = chartData.map(item => ({
              time: item.time,
              value: item.volume || 0,
              color: item.close >= item.open
                ? alpha(theme.palette.success.main, 0.3)
                : alpha(theme.palette.error.main, 0.3),
            }));
            volumeSeriesRef.current.setData(volumeData);
          }

          // Calculate and display moving averages
          if (showIndicators && ma20SeriesRef.current && ma50SeriesRef.current) {
            // Calculate MA20
            const ma20Data = chartData.map((item, index) => {
              if (index < 19) return null;
              const sum = chartData.slice(index - 19, index + 1).reduce((acc, d) => acc + d.close, 0);
              return {
                time: item.time,
                value: sum / 20,
              };
            }).filter(item => item !== null);

            // Calculate MA50
            const ma50Data = chartData.map((item, index) => {
              if (index < 49) return null;
              const sum = chartData.slice(index - 49, index + 1).reduce((acc, d) => acc + d.close, 0);
              return {
                time: item.time,
                value: sum / 50,
              };
            }).filter(item => item !== null);

            ma20SeriesRef.current.setData(ma20Data as any);
            ma50SeriesRef.current.setData(ma50Data as any);

            // Simple AI signal based on MA crossover
            if (ma20Data.length > 0 && ma50Data.length > 0) {
              const latestMA20 = ma20Data[ma20Data.length - 1];
              const latestMA50 = ma50Data[ma50Data.length - 1];
              const prevMA20 = ma20Data[ma20Data.length - 2];
              const prevMA50 = ma50Data[ma50Data.length - 2];

              if (latestMA20 && latestMA50 && prevMA20 && prevMA50) {
                // Golden cross (bullish)
                if (prevMA20.value < prevMA50.value && latestMA20.value > latestMA50.value) {
                  setAiSignal('BUY');
                  candlestickSeriesRef.current?.setMarkers([{
                    time: latestMA20.time,
                    position: 'belowBar',
                    color: theme.palette.success.main,
                    shape: 'arrowUp',
                    text: 'AI: BUY',
                  }]);
                }
                // Death cross (bearish)
                else if (prevMA20.value > prevMA50.value && latestMA20.value < latestMA50.value) {
                  setAiSignal('SELL');
                  candlestickSeriesRef.current?.setMarkers([{
                    time: latestMA20.time,
                    position: 'aboveBar',
                    color: theme.palette.error.main,
                    shape: 'arrowDown',
                    text: 'AI: SELL',
                  }]);
                }
                // Trending
                else if (latestMA20.value > latestMA50.value) {
                  setAiSignal('BUY');
                } else {
                  setAiSignal('HOLD');
                }
              }
            }
          }

          // Fit content
          chartRef.current?.timeScale().fitContent();

          // Calculate price change
          const firstCandle = chartData[0];
          const lastCandle = chartData[chartData.length - 1];
          setLatestPrice(lastCandle.close);
          setPriceChange(((lastCandle.close - firstCandle.open) / firstCandle.open) * 100);
        }

      } catch (err) {
        console.error('Chart data error:', err);
        setError(err instanceof Error ? err.message : 'Failed to load chart data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    // Refresh every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [symbol, theme]);

  // Update with WebSocket data
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
      console.log('Chart WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.symbol === symbol && data.price && candlestickSeriesRef.current) {
          const timestamp = Math.floor(Date.now() / 1000);
          const price = parseFloat(data.price);

          // Update the latest candle
          candlestickSeriesRef.current.update({
            time: timestamp,
            open: price,
            high: price,
            low: price,
            close: price,
          });

          setLatestPrice(price);
        }
      } catch (err) {
        console.error('WebSocket message error:', err);
      }
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [symbol]);

  return (
    <Paper
      sx={{
        p: 2,
        height: height + 80,
        position: 'relative',
        background: alpha(theme.palette.background.paper, 0.8),
        backdropFilter: 'blur(10px)',
        border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
      }}
    >
      {/* Header */}
      <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            {symbol}
          </Typography>
          {latestPrice && (
            <Stack direction="row" spacing={1} alignItems="center">
              <Typography variant="h4" sx={{ fontWeight: 700 }}>
                ${latestPrice.toFixed(2)}
              </Typography>
              <Chip
                label={`${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)}%`}
                size="small"
                color={priceChange >= 0 ? 'success' : 'error'}
                sx={{ fontWeight: 600 }}
              />
            </Stack>
          )}
        </Box>
        <Stack direction="row" spacing={1}>
          {aiSignal && (
            <Chip
              label={`AI: ${aiSignal}`}
              size="small"
              color={aiSignal === 'BUY' ? 'success' : aiSignal === 'SELL' ? 'error' : 'default'}
              icon={
                aiSignal === 'BUY' ? (
                  <TrendingUpIcon sx={{ fontSize: 16 }} />
                ) : aiSignal === 'SELL' ? (
                  <TrendingDownIcon sx={{ fontSize: 16 }} />
                ) : undefined
              }
              sx={{ fontWeight: 600 }}
            />
          )}
          <Chip
            label="LIVE"
            size="small"
            color="primary"
            sx={{
              animation: 'pulse 2s infinite',
              '@keyframes pulse': {
                '0%': { opacity: 1 },
                '50%': { opacity: 0.6 },
                '100%': { opacity: 1 },
              }
            }}
          />
        </Stack>
      </Stack>

      {/* Chart Container */}
      <Box
        ref={chartContainerRef}
        sx={{
          height: height,
          position: 'relative',
          '& .tv-lightweight-charts': {
            borderRadius: 1,
          }
        }}
      >
        {loading && (
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
            }}
          >
            <CircularProgress />
          </Box>
        )}

        {error && (
          <Alert
            severity="error"
            sx={{
              position: 'absolute',
              top: 16,
              left: 16,
              right: 16,
            }}
          >
            {error}
          </Alert>
        )}
      </Box>
    </Paper>
  );
};
