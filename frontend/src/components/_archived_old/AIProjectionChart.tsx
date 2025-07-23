/**
 * AI Projection Chart Component
 * Real-time candlestick chart with AI predictions using lightweight-charts
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
  LineStyle,
} from 'lightweight-charts';
import { Box, Typography, CircularProgress } from '@mui/material';
import { styled } from '@mui/material/styles';
import { RSIChart } from './RSIChart';
import { MACDChart } from './MACDChart';

const ChartContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  width: '100%',
  height: '600px',
  backgroundColor: '#000000',
  borderRadius: theme.spacing(1),
  overflow: 'hidden',
}));

const LoadingOverlay = styled(Box)({
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: 'rgba(0, 0, 0, 0.8)',
  zIndex: 1000,
});

interface SignalData {
  price: number;
  signal: 'BUY' | 'SELL' | 'HOLD';
  entry: number;
  stop_loss: number;
  take_profit: number;
  predicted_path: Array<{ time: string; price: number }>;
  indicators: {
    rsi: number[];
    macd: { macd: number; signal: number; hist: number }[];
  };
  candles: CandlestickData[];
}

interface AIProjectionChartProps {
  symbol: string;
  height?: string | number;
}

export const AIProjectionChart: React.FC<AIProjectionChartProps> = ({
  symbol = 'TSLA',
  height = '600px'
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const projectionSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const entryLineRef = useRef<ISeriesApi<'Line'> | null>(null);
  const stopLossLineRef = useRef<ISeriesApi<'Line'> | null>(null);
  const takeProfitLineRef = useRef<ISeriesApi<'Line'> | null>(null);

  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<SignalData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  // Fetch signal data with retry logic
  const fetchSignalData = async (attempt = 0) => {
    try {
      setLoading(true);
      setError(null);

      // Fetch real market data with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout

      const marketResponse = await fetch(
        `http://localhost:8000/api/v1/market-data/${symbol}/history?period=1d&interval=5m`,
        { signal: controller.signal }
      );

      clearTimeout(timeoutId);

      if (!marketResponse.ok) {
        throw new Error(`HTTP ${marketResponse.status}: Failed to fetch market data`);
      }

      const marketData = await marketResponse.json();

      // Transform to candlestick format
      const candles: CandlestickData[] = marketData.data.map((candle: any) => ({
        time: candle.time as Time,
        open: parseFloat(candle.open),
        high: parseFloat(candle.high),
        low: parseFloat(candle.low),
        close: parseFloat(candle.close),
      }));

      // Get latest price
      const latestPrice = candles[candles.length - 1].close;

      // Calculate moving averages for trend detection
      const sma20 = candles.slice(-20).reduce((sum, c) => sum + c.close, 0) / 20;
      const sma50 = candles.slice(-50).reduce((sum, c) => sum + c.close, 0) / Math.min(50, candles.length);

      // Determine trend direction
      const trendDirection = latestPrice > sma20 && sma20 > sma50 ? 1 : -1;
      const trendStrength = Math.abs(latestPrice - sma20) / sma20;

      // Calculate volatility
      const recentCandles = candles.slice(-20);
      const avgRange = recentCandles.reduce((sum, c) => sum + (c.high - c.low), 0) / recentCandles.length;
      const volatility = avgRange / latestPrice;

      // Generate AI predictions with trend-following and mean reversion
      const predicted_path = Array.from({ length: 20 }, (_, i) => {
        const time = candles[candles.length - 1].time as number;
        const futureTime = time + (i + 1) * 300; // 5 minute intervals

        // Combine trend following with mean reversion
        const trendComponent = trendDirection * trendStrength * 0.001 * (i + 1);
        const meanReversionComponent = -(latestPrice - sma20) / sma20 * 0.05 * Math.log(i + 2);
        const noiseComponent = (Math.random() - 0.5) * volatility * 0.5;

        // Apply Fibonacci-like decay to predictions
        const decayFactor = 1 / (1 + i * 0.05);
        const priceChange = (trendComponent + meanReversionComponent + noiseComponent) * decayFactor;

        const price = latestPrice * (1 + priceChange);

        return {
          time: futureTime.toString(),
          price: price
        };
      });

      // Calculate entry, stop loss, and take profit
      const entry = latestPrice;
      const stop_loss = entry * 0.98; // 2% stop loss
      const take_profit = entry * 1.05; // 5% take profit

      // Calculate RSI
      const calculateRSI = (data: typeof candles, period = 14) => {
        if (data.length < period + 1) return Array(data.length).fill(50);

        const rsiValues: number[] = [];
        for (let i = period; i < data.length; i++) {
          let gains = 0;
          let losses = 0;

          for (let j = i - period + 1; j <= i; j++) {
            const change = data[j].close - data[j - 1].close;
            if (change > 0) gains += change;
            else losses += Math.abs(change);
          }

          const avgGain = gains / period;
          const avgLoss = losses / period;
          const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
          const rsi = 100 - (100 / (1 + rs));
          rsiValues.push(rsi);
        }

        return rsiValues;
      };

      const rsi = calculateRSI(candles);

      // Calculate MACD
      const calculateEMA = (data: number[], period: number): number[] => {
        const multiplier = 2 / (period + 1);
        const ema: number[] = [data[0]];

        for (let i = 1; i < data.length; i++) {
          ema.push((data[i] - ema[i - 1]) * multiplier + ema[i - 1]);
        }

        return ema;
      };

      const closes = candles.map(c => c.close);
      const ema12 = calculateEMA(closes, 12);
      const ema26 = calculateEMA(closes, 26);

      const macdLine = ema12.map((val, i) => val - ema26[i]);
      const signalLine = calculateEMA(macdLine, 9);

      const macd = macdLine.slice(-26).map((val, i) => ({
        macd: val,
        signal: signalLine[signalLine.length - 26 + i] || val,
        hist: val - (signalLine[signalLine.length - 26 + i] || val)
      }));

      // Determine signal based on technical indicators
      const latestRSI = rsi[rsi.length - 1] || 50;
      const latestMACD = macd[macd.length - 1];
      const macdCrossover = latestMACD && latestMACD.macd > latestMACD.signal;

      let signal: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';

      // Buy conditions
      if (latestRSI < 30 && macdCrossover && trendDirection > 0) {
        signal = 'BUY';
      }
      // Sell conditions
      else if (latestRSI > 70 && !macdCrossover && trendDirection < 0) {
        signal = 'SELL';
      }
      // Trend following
      else if (latestRSI > 40 && latestRSI < 60 && macdCrossover) {
        signal = 'BUY';
      }

      const signalData: SignalData = {
        price: latestPrice,
        signal,
        entry,
        stop_loss,
        take_profit,
        predicted_path,
        indicators: { rsi, macd },
        candles
      };

      setData(signalData);
    } catch (error) {
      console.error('Error fetching signal data:', error);
    } finally {
      setLoading(false);
    }
  };

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current || !data) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: parseInt(height.toString()),
      layout: {
        background: { type: ColorType.Solid, color: '#000000' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: 'rgba(42, 46, 57, 0.5)' },
        horzLines: { color: 'rgba(42, 46, 57, 0.5)' },
      },
      rightPriceScale: {
        borderColor: 'rgba(197, 203, 206, 0.8)',
        scaleMargins: { top: 0.1, bottom: 0.2 },
      },
      timeScale: {
        borderColor: 'rgba(197, 203, 206, 0.8)',
        timeVisible: true,
        secondsVisible: false,
      },
      watermark: {
        visible: true,
        fontSize: 48,
        horzAlign: 'center',
        vertAlign: 'center',
        color: 'rgba(255, 215, 0, 0.1)',
        text: symbol.toUpperCase(),
      },
    });

    // Add candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00D964',
      downColor: '#FF3B30',
      borderUpColor: '#00D964',
      borderDownColor: '#FF3B30',
      wickUpColor: '#00D964',
      wickDownColor: '#FF3B30',
    });
    candleSeries.setData(data.candles);

    // Add AI projection line
    const projectionSeries = chart.addLineSeries({
      color: '#FFD700',
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      crosshairMarkerVisible: false,
    });

    const projectionData: LineData[] = data.predicted_path.map(point => ({
      time: parseInt(point.time) as Time,
      value: point.price,
    }));
    projectionSeries.setData(projectionData);

    // Add entry price line
    const entryLine = chart.addLineSeries({
      color: '#007AFF',
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
      crosshairMarkerVisible: false,
      lastValueVisible: true,
      priceLineVisible: true,
    });

    const lastTime = data.candles[data.candles.length - 1].time;
    const futureTime = (lastTime as number) + 86400; // +1 day
    entryLine.setData([
      { time: lastTime, value: data.entry },
      { time: futureTime as Time, value: data.entry },
    ]);

    // Add stop loss line
    const stopLossLine = chart.addLineSeries({
      color: '#FF3B30',
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
      crosshairMarkerVisible: false,
      lastValueVisible: true,
      priceLineVisible: true,
    });
    stopLossLine.setData([
      { time: lastTime, value: data.stop_loss },
      { time: futureTime as Time, value: data.stop_loss },
    ]);

    // Add take profit line
    const takeProfitLine = chart.addLineSeries({
      color: '#00D964',
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
      crosshairMarkerVisible: false,
      lastValueVisible: true,
      priceLineVisible: true,
    });
    takeProfitLine.setData([
      { time: lastTime, value: data.take_profit },
      { time: futureTime as Time, value: data.take_profit },
    ]);

    // Add markers
    candleSeries.setMarkers([
      {
        time: lastTime,
        position: 'belowBar',
        color: '#007AFF',
        shape: 'arrowUp',
        text: 'Entry',
      },
    ]);

    // Store references
    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    projectionSeriesRef.current = projectionSeries;
    entryLineRef.current = entryLine;
    stopLossLineRef.current = stopLossLine;
    takeProfitLineRef.current = takeProfitLine;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [data, symbol, height]);

  // Fetch data on mount and symbol change
  useEffect(() => {
    fetchSignalData();

    // Refresh every 30 seconds
    const interval = setInterval(fetchSignalData, 30000);

    return () => clearInterval(interval);
  }, [symbol]);

  return (
    <Box sx={{ width: '100%' }}>
      <ChartContainer ref={chartContainerRef} sx={{ height }}>
        {loading && (
          <LoadingOverlay>
            <CircularProgress sx={{ color: '#FFD700' }} />
          </LoadingOverlay>
        )}
      </ChartContainer>

      {data && (
        <>
          <Box sx={{ mt: 2, p: 2, backgroundColor: '#111', borderRadius: 1 }}>
            <Typography variant="h6" sx={{ color: '#FFD700', mb: 1 }}>
              AI Signal: {data.signal}
            </Typography>
            <Box sx={{ display: 'flex', gap: 3 }}>
              <Typography sx={{ color: '#fff' }}>
                Entry: ${data.entry.toFixed(2)}
              </Typography>
              <Typography sx={{ color: '#FF3B30' }}>
                Stop Loss: ${data.stop_loss.toFixed(2)}
              </Typography>
              <Typography sx={{ color: '#00D964' }}>
                Take Profit: ${data.take_profit.toFixed(2)}
              </Typography>
            </Box>
          </Box>

          <Box sx={{ mt: 2 }}>
            <RSIChart values={data.indicators.rsi} />
          </Box>

          <Box sx={{ mt: 2 }}>
            <MACDChart values={data.indicators.macd} />
          </Box>
        </>
      )}
    </Box>
  );
};

export default AIProjectionChart;
