/**
 * Focused Trading Chart
 * A single, clean implementation that just works
 */

import React, { useEffect, useRef, useState } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  LineData,
  Time,
  ColorType,
  CrosshairMode,
  LineStyle,
} from 'lightweight-charts';
import { Box, Typography, ToggleButton, ToggleButtonGroup, CircularProgress } from '@mui/material';
import { styled } from '@mui/material/styles';

// Simple, clean styling
const Container = styled(Box)({
  width: '100%',
  height: '100vh',
  backgroundColor: '#131722',
  display: 'flex',
  flexDirection: 'column',
});

const Header = styled(Box)({
  height: 50,
  backgroundColor: '#1e222d',
  borderBottom: '1px solid #2a2e39',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '0 20px',
});

const ChartArea = styled(Box)({
  flex: 1,
  position: 'relative',
  backgroundColor: '#131722',
});

const LoadingOverlay = styled(Box)({
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: 'rgba(19, 23, 34, 0.9)',
  zIndex: 1000,
});

// Clean color scheme
const COLORS = {
  background: '#131722',
  card: '#1e222d',
  border: '#2a2e39',

  text: {
    primary: '#d1d4dc',
    secondary: '#787b86',
  },

  candle: {
    up: '#26a69a',
    down: '#ef5350',
  },

  ma: {
    20: '#2962ff',
    50: '#ff6d00',
    200: '#d500f9',
  },

  volume: {
    up: 'rgba(38, 166, 154, 0.5)',
    down: 'rgba(239, 83, 80, 0.5)',
  },

  grid: 'rgba(42, 46, 57, 0.5)',
};

interface ChartProps {
  symbol?: string;
}

export const FocusedTradingChart: React.FC<ChartProps> = ({ symbol = 'TSLA' }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const rsiChartRef = useRef<IChartApi | null>(null);
  const macdChartRef = useRef<IChartApi | null>(null);

  const [loading, setLoading] = useState(true);
  const [timeframe, setTimeframe] = useState('1D');
  const [price, setPrice] = useState<number>(0);
  const [change, setChange] = useState<number>(0);
  const [changePercent, setChangePercent] = useState<number>(0);
  const [isLive, setIsLive] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // Create the chart layout
  useEffect(() => {
    if (!chartContainerRef.current || loading) return;

    const container = chartContainerRef.current;
    const containerHeight = container.clientHeight;
    const containerWidth = container.clientWidth;

    // Clear previous charts
    container.innerHTML = '';

    // Create layout divs
    const priceDiv = document.createElement('div');
    priceDiv.style.height = '60%';
    priceDiv.style.width = '100%';
    container.appendChild(priceDiv);

    const rsiDiv = document.createElement('div');
    rsiDiv.style.height = '20%';
    rsiDiv.style.width = '100%';
    rsiDiv.style.borderTop = '1px solid #2a2e39';
    container.appendChild(rsiDiv);

    const macdDiv = document.createElement('div');
    macdDiv.style.height = '20%';
    macdDiv.style.width = '100%';
    macdDiv.style.borderTop = '1px solid #2a2e39';
    container.appendChild(macdDiv);

    // Common chart options
    const commonOptions = {
      layout: {
        background: { type: ColorType.Solid, color: COLORS.background },
        textColor: COLORS.text.secondary,
        fontSize: 11,
      },
      grid: {
        vertLines: { color: COLORS.grid },
        horzLines: { color: COLORS.grid },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: COLORS.text.secondary,
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: COLORS.card,
        },
        horzLine: {
          color: COLORS.text.secondary,
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: COLORS.card,
        },
      },
      rightPriceScale: {
        borderColor: COLORS.border,
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      timeScale: {
        borderColor: COLORS.border,
        timeVisible: true,
        secondsVisible: false,
      },
    };

    // Create price chart
    const priceChart = createChart(priceDiv, {
      ...commonOptions,
      width: containerWidth,
      height: containerHeight * 0.6,
    });

    // Create RSI chart
    const rsiChart = createChart(rsiDiv, {
      ...commonOptions,
      width: containerWidth,
      height: containerHeight * 0.2,
      timeScale: { visible: false },
    });

    // Create MACD chart
    const macdChart = createChart(macdDiv, {
      ...commonOptions,
      width: containerWidth,
      height: containerHeight * 0.2,
    });

    // Sync crosshairs
    const charts = [priceChart, rsiChart, macdChart];
    charts.forEach((chart, index) => {
      chart.subscribeCrosshairMove((param) => {
        charts.forEach((otherChart, otherIndex) => {
          if (index !== otherIndex && param.time) {
            otherChart.setCrosshairPosition(param.point?.x || 0, 0, param.time);
          }
        });
      });
    });

    // Sync visible range
    priceChart.timeScale().subscribeVisibleTimeRangeChange(() => {
      const range = priceChart.timeScale().getVisibleRange();
      if (range) {
        rsiChart.timeScale().setVisibleRange(range);
        macdChart.timeScale().setVisibleRange(range);
      }
    });

    chartRef.current = priceChart;
    rsiChartRef.current = rsiChart;
    macdChartRef.current = macdChart;

    // Fetch and display data
    fetchData();

    // Set up periodic refresh based on timeframe
    const refreshInterval = timeframe === '1D' ? 10000 : // 10 seconds for intraday
                           timeframe === '5D' ? 30000 : // 30 seconds for 5 days
                           60000; // 1 minute for longer timeframes

    const intervalId = setInterval(() => {
      fetchData();
    }, refreshInterval);

    // Handle resize
    const handleResize = () => {
      const newWidth = container.clientWidth;
      const newHeight = container.clientHeight;

      priceChart.applyOptions({ width: newWidth, height: newHeight * 0.6 });
      rsiChart.applyOptions({ width: newWidth, height: newHeight * 0.2 });
      macdChart.applyOptions({ width: newWidth, height: newHeight * 0.2 });
    };

    window.addEventListener('resize', handleResize);

    return () => {
      clearInterval(intervalId);
      window.removeEventListener('resize', handleResize);
      priceChart.remove();
      rsiChart.remove();
      macdChart.remove();
    };
  }, [loading, timeframe]);

  // Fetch data
  const fetchData = async () => {
    try {
      setLoading(true);

      // Timeframe mapping
      const periodMap: Record<string, string> = {
        '1D': '1d',
        '5D': '5d',
        '1M': '1mo',
        '3M': '3mo',
        '1Y': '1y',
      };

      const intervalMap: Record<string, string> = {
        '1D': '5m',
        '5D': '30m',
        '1M': '1d',
        '3M': '1d',
        '1Y': '1wk',
      };

      const response = await fetch(
        `http://localhost:8000/api/v1/market-data/${symbol}/history?period=${periodMap[timeframe]}&interval=${intervalMap[timeframe]}`
      );

      if (!response.ok) throw new Error('Failed to fetch data');

      const result = await response.json();
      const data = result.data || [];

      if (data.length === 0) throw new Error('No data available');

      // Process data
      const candles: CandlestickData[] = data.map((d: any) => ({
        time: d.time as Time,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }));

      const volumes = data.map((d: any, i: number) => ({
        time: d.time as Time,
        value: d.volume || 0,
        color: candles[i].close >= candles[i].open ? COLORS.volume.up : COLORS.volume.down,
      }));

      // Update price display
      const lastCandle = candles[candles.length - 1];
      const firstCandle = candles[0];
      setPrice(lastCandle.close);
      setChange(lastCandle.close - firstCandle.open);
      setChangePercent(((lastCandle.close - firstCandle.open) / firstCandle.open) * 100);

      // Add data to charts
      if (chartRef.current && rsiChartRef.current && macdChartRef.current) {
        // Price chart
        const candleSeries = chartRef.current.addCandlestickSeries({
          upColor: COLORS.candle.up,
          downColor: COLORS.candle.down,
          borderUpColor: COLORS.candle.up,
          borderDownColor: COLORS.candle.down,
          wickUpColor: COLORS.candle.up,
          wickDownColor: COLORS.candle.down,
        });
        candleSeries.setData(candles);
        candleSeriesRef.current = candleSeries;

        // Volume
        const volumeSeries = chartRef.current.addHistogramSeries({
          color: COLORS.volume.up,
          priceFormat: { type: 'volume' },
          priceScaleId: 'volume',
        });
        volumeSeries.priceScale().applyOptions({
          scaleMargins: { top: 0.8, bottom: 0 },
        });
        volumeSeries.setData(volumes);

        // Moving averages
        const ma20 = chartRef.current.addLineSeries({
          color: COLORS.ma[20],
          lineWidth: 2,
          title: 'MA 20',
          lastValueVisible: false,
        });
        ma20.setData(calculateMA(candles, 20));

        const ma50 = chartRef.current.addLineSeries({
          color: COLORS.ma[50],
          lineWidth: 2,
          title: 'MA 50',
          lastValueVisible: false,
        });
        ma50.setData(calculateMA(candles, 50));

        // RSI
        const rsiLine = rsiChartRef.current.addLineSeries({
          color: '#ab47bc',
          lineWidth: 2,
          title: 'RSI',
        });
        const rsiData = calculateRSI(candles, 14);
        rsiLine.setData(rsiData);

        // RSI levels
        rsiLine.createPriceLine({ price: 70, color: '#ef5350', lineWidth: 1, lineStyle: LineStyle.Dashed });
        rsiLine.createPriceLine({ price: 30, color: '#26a69a', lineWidth: 1, lineStyle: LineStyle.Dashed });

        // MACD
        const macdData = calculateMACD(candles, 12, 26, 9);

        const macdHistogram = macdChartRef.current.addHistogramSeries({
          color: '#26a69a',
          title: 'MACD Histogram',
        });
        macdHistogram.setData(macdData.histogram.map(h => ({
          ...h,
          color: h.value >= 0 ? '#26a69a' : '#ef5350',
        })));

        const macdLine = macdChartRef.current.addLineSeries({
          color: '#2196f3',
          lineWidth: 2,
          title: 'MACD',
        });
        macdLine.setData(macdData.macd);

        const signalLine = macdChartRef.current.addLineSeries({
          color: '#ff6d00',
          lineWidth: 2,
          title: 'Signal',
        });
        signalLine.setData(macdData.signal);

        // Zero line
        signalLine.createPriceLine({ price: 0, color: COLORS.border, lineWidth: 1 });
      }

      setLoading(false);

      // Connect WebSocket for real-time updates
      connectWebSocket();

    } catch (error) {
      console.error('Error fetching data:', error);
      setLoading(false);
    }
  };

  // WebSocket connection for live updates
  const connectWebSocket = () => {
    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      const ws = new WebSocket(`ws://localhost:8000/ws/v2/signals/${symbol}`);

      ws.onopen = () => {
        console.log('WebSocket connected for live updates');
        setIsLive(true);
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);

          // Update price in real-time
          if (message.type === 'price_update' || message.price) {
            const newPrice = message.price || message.data?.price;
            if (newPrice && candleSeriesRef.current) {
              setPrice(newPrice);

              // Update the last candle
              const time = Math.floor(Date.now() / 1000) as Time;
              candleSeriesRef.current.update({
                time,
                open: newPrice,
                high: newPrice,
                low: newPrice,
                close: newPrice,
              });
            }
          }
        } catch (error) {
          console.error('WebSocket message error:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsLive(false);
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsLive(false);
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('WebSocket connection error:', error);
      setIsLive(false);
    }
  };

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Calculate indicators
  const calculateMA = (candles: CandlestickData[], period: number): LineData[] => {
    const ma: LineData[] = [];
    for (let i = period - 1; i < candles.length; i++) {
      const sum = candles.slice(i - period + 1, i + 1).reduce((acc, c) => acc + c.close, 0);
      ma.push({ time: candles[i].time, value: sum / period });
    }
    return ma;
  };

  const calculateRSI = (candles: CandlestickData[], period: number): LineData[] => {
    const rsi: LineData[] = [];
    const gains: number[] = [];
    const losses: number[] = [];

    // Calculate initial gains/losses
    for (let i = 1; i < candles.length; i++) {
      const change = candles[i].close - candles[i - 1].close;
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? -change : 0);
    }

    // Calculate RSI
    for (let i = period; i < gains.length; i++) {
      const avgGain = gains.slice(i - period, i).reduce((a, b) => a + b) / period;
      const avgLoss = losses.slice(i - period, i).reduce((a, b) => a + b) / period;
      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      const rsiValue = 100 - (100 / (1 + rs));
      rsi.push({ time: candles[i + 1].time, value: rsiValue });
    }

    return rsi;
  };

  const calculateEMA = (values: number[], period: number): number[] => {
    const ema: number[] = [];
    const multiplier = 2 / (period + 1);

    // Start with SMA
    let sum = 0;
    for (let i = 0; i < period; i++) {
      sum += values[i];
    }
    ema[period - 1] = sum / period;

    // Calculate EMA
    for (let i = period; i < values.length; i++) {
      ema[i] = (values[i] - ema[i - 1]) * multiplier + ema[i - 1];
    }

    return ema;
  };

  const calculateMACD = (candles: CandlestickData[], fast: number, slow: number, signal: number) => {
    const closes = candles.map(c => c.close);
    const ema12 = calculateEMA(closes, fast);
    const ema26 = calculateEMA(closes, slow);

    const macdLine: LineData[] = [];
    const macdValues: number[] = [];

    for (let i = slow - 1; i < candles.length; i++) {
      const macdValue = ema12[i] - ema26[i];
      macdValues.push(macdValue);
      macdLine.push({ time: candles[i].time, value: macdValue });
    }

    const signalEMA = calculateEMA(macdValues, signal);
    const signalLine: LineData[] = [];
    const histogram: LineData[] = [];

    for (let i = signal - 1; i < signalEMA.length; i++) {
      const time = candles[i + slow - 1].time;
      signalLine.push({ time, value: signalEMA[i] });
      histogram.push({ time, value: macdValues[i] - signalEMA[i] });
    }

    return { macd: macdLine, signal: signalLine, histogram };
  };

  // Handle timeframe change
  const handleTimeframeChange = (_: any, newTimeframe: string) => {
    if (newTimeframe && newTimeframe !== timeframe) {
      setTimeframe(newTimeframe);
      if (chartRef.current && rsiChartRef.current && macdChartRef.current) {
        // Clear existing series
        chartRef.current.removeAllSeries();
        rsiChartRef.current.removeAllSeries();
        macdChartRef.current.removeAllSeries();
        // Fetch new data
        fetchData();
      }
    }
  };

  return (
    <Container>
      <Header>
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="h6" sx={{ color: COLORS.text.primary, fontWeight: 600 }}>
            {symbol}
          </Typography>
          {isLive && (
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                padding: '2px 8px',
                borderRadius: 1,
                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                color: '#4caf50',
                fontSize: '0.75rem',
                fontWeight: 600,
              }}
            >
              <Box
                sx={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  backgroundColor: '#4caf50',
                  animation: 'pulse 2s infinite',
                  '@keyframes pulse': {
                    '0%': { opacity: 1 },
                    '50%': { opacity: 0.5 },
                    '100%': { opacity: 1 },
                  },
                }}
              />
              LIVE
            </Box>
          )}
          {price > 0 && (
            <>
              <Typography variant="h6" sx={{ color: COLORS.text.primary }}>
                ${price.toFixed(2)}
              </Typography>
              <Typography
                variant="body2"
                sx={{
                  color: change >= 0 ? COLORS.candle.up : COLORS.candle.down,
                  fontWeight: 500,
                }}
              >
                {change >= 0 ? '+' : ''}{change.toFixed(2)} ({changePercent >= 0 ? '+' : ''}{changePercent.toFixed(2)}%)
              </Typography>
            </>
          )}
        </Box>

        <ToggleButtonGroup
          value={timeframe}
          exclusive
          onChange={handleTimeframeChange}
          size="small"
          sx={{
            '& .MuiToggleButton-root': {
              color: COLORS.text.secondary,
              borderColor: COLORS.border,
              padding: '4px 16px',
              textTransform: 'none',
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
              },
              '&.Mui-selected': {
                backgroundColor: 'rgba(33, 150, 243, 0.15)',
                color: '#2196f3',
                '&:hover': {
                  backgroundColor: 'rgba(33, 150, 243, 0.25)',
                },
              },
            },
          }}
        >
          <ToggleButton value="1D">1D</ToggleButton>
          <ToggleButton value="5D">5D</ToggleButton>
          <ToggleButton value="1M">1M</ToggleButton>
          <ToggleButton value="3M">3M</ToggleButton>
          <ToggleButton value="1Y">1Y</ToggleButton>
        </ToggleButtonGroup>
      </Header>

      <ChartArea>
        <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />
        {loading && (
          <LoadingOverlay>
            <CircularProgress sx={{ color: '#2196f3' }} />
          </LoadingOverlay>
        )}
      </ChartArea>
    </Container>
  );
};

export default FocusedTradingChart;
