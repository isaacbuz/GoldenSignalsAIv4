/**
 * Robinhood-Style Trading Chart
 * Clean, modern interface with all indicators in one view
 */

import React, { useEffect, useRef, useState } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  LineData,
  HistogramData,
  Time,
  ColorType,
  CrosshairMode,
} from 'lightweight-charts';
import {
  Box,
  Typography,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Paper,
  IconButton,
  Stack,
  Chip,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { ArrowBack as ArrowBackIcon, Search as SearchIcon } from '@mui/icons-material';

const Container = styled(Box)({
  display: 'flex',
  width: '100%',
  height: '100vh',
  backgroundColor: '#ffffff',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
});

const Sidebar = styled(Paper)({
  width: '200px',
  padding: '20px',
  borderRight: '1px solid #e0e0e0',
  backgroundColor: '#fafafa',
  boxShadow: 'none',
  borderRadius: 0,
  overflowY: 'auto',
});

const MainContent = styled(Box)({
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: '#ffffff',
});

const Header = styled(Box)({
  padding: '16px 24px',
  borderBottom: '1px solid #e0e0e0',
  display: 'flex',
  alignItems: 'center',
  gap: '16px',
});

const ChartContainer = styled(Box)({
  flex: 1,
  position: 'relative',
  backgroundColor: '#ffffff',
});

const IndicatorLabel = styled(FormControlLabel)({
  margin: '4px 0',
  '& .MuiTypography-root': {
    fontSize: '14px',
    color: '#333',
  },
  '& .MuiCheckbox-root': {
    padding: '4px 8px',
  },
});

const PriceDisplay = styled(Typography)({
  fontSize: '24px',
  fontWeight: 500,
  color: '#000',
});

const ChangeDisplay = styled(Typography)<{ positive: boolean }>(({ positive }) => ({
  fontSize: '18px',
  fontWeight: 400,
  color: positive ? '#00c853' : '#ff1744',
}));

interface IndicatorConfig {
  ma10: boolean;
  ma50: boolean;
  ma200: boolean;
  ema9: boolean;
  ema12: boolean;
  vwap: boolean;
  bollinger: boolean;
  volume: boolean;
  rsi: boolean;
  macd: boolean;
}

interface RobinhoodChartProps {
  symbol?: string;
}

export const RobinhoodChart: React.FC<RobinhoodChartProps> = ({ symbol = 'TSLA' }) => {
  const mainChartRef = useRef<HTMLDivElement>(null);
  const rsiChartRef = useRef<HTMLDivElement>(null);
  const macdChartRef = useRef<HTMLDivElement>(null);

  const [price, setPrice] = useState<number>(264.00);
  const [change, setChange] = useState<number>(-18.76);
  const [changePercent, setChangePercent] = useState<number>(-6.62);
  const [loading, setLoading] = useState(false);

  const [indicators, setIndicators] = useState<IndicatorConfig>({
    ma10: true,
    ma50: true,
    ma200: true,
    ema9: true,
    ema12: false,
    vwap: false,
    bollinger: true,
    volume: true,
    rsi: true,
    macd: true,
  });

  const toggleIndicator = (indicator: keyof IndicatorConfig) => {
    setIndicators(prev => ({ ...prev, [indicator]: !prev[indicator] }));
  };

  useEffect(() => {
    if (!mainChartRef.current) return;

    // Create main chart
    const chart = createChart(mainChartRef.current, {
      width: mainChartRef.current.clientWidth,
      height: mainChartRef.current.clientHeight * 0.6,
      layout: {
        background: { type: ColorType.Solid, color: '#ffffff' },
        textColor: '#333',
      },
      grid: {
        vertLines: {
          color: '#f0f0f0',
          style: 1,
          visible: true,
        },
        horzLines: {
          color: '#f0f0f0',
          style: 1,
          visible: true,
        },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: '#758696',
          width: 1,
          style: 0,
          visible: true,
          labelVisible: true,
        },
        horzLine: {
          color: '#758696',
          width: 1,
          style: 0,
          visible: true,
          labelVisible: true,
        },
      },
      rightPriceScale: {
        borderColor: '#e0e0e0',
        visible: true,
      },
      timeScale: {
        borderColor: '#e0e0e0',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    // Add candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00c853',
      downColor: '#ff1744',
      borderUpColor: '#00c853',
      borderDownColor: '#ff1744',
      wickUpColor: '#00c853',
      wickDownColor: '#ff1744',
    });

    // Fetch and set data
    fetchData(symbol).then(data => {
      if (data) {
        candleSeries.setData(data.candles);

        // Update price display
        const lastCandle = data.candles[data.candles.length - 1];
        setPrice(lastCandle.close);
        setChange(lastCandle.close - data.candles[0].open);
        setChangePercent(((lastCandle.close - data.candles[0].open) / data.candles[0].open) * 100);

        // Add moving averages
        if (indicators.ma10) addMovingAverage(chart, data.candles, 10, '#2196f3', 'MA(10)');
        if (indicators.ma50) addMovingAverage(chart, data.candles, 50, '#9c27b0', 'MA(50)');
        if (indicators.ma200) addMovingAverage(chart, data.candles, 200, '#ff9800', 'MA(200)');
        if (indicators.ema9) addEMA(chart, data.candles, 9, '#4caf50', 'EMA(9)');

        // Add Bollinger Bands
        if (indicators.bollinger) addBollingerBands(chart, data.candles);

        // Add volume
        if (indicators.volume) addVolume(chart, data.candles);
      }
    });

    // Add RSI chart
    if (indicators.rsi && rsiChartRef.current) {
      createRSIChart(rsiChartRef.current, symbol);
    }

    // Add MACD chart
    if (indicators.macd && macdChartRef.current) {
      createMACDChart(macdChartRef.current, symbol);
    }

    // Handle resize
    const handleResize = () => {
      if (mainChartRef.current) {
        chart.applyOptions({
          width: mainChartRef.current.clientWidth,
          height: mainChartRef.current.clientHeight * 0.6,
        });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [symbol, indicators]);

  return (
    <Container>
      <Sidebar>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          Indicators
        </Typography>

        <FormGroup>
          <IndicatorLabel
            control={<Checkbox checked={indicators.volume} onChange={() => toggleIndicator('volume')} size="small" />}
            label="Volume"
          />
          <IndicatorLabel
            control={<Checkbox checked={indicators.ma10} onChange={() => toggleIndicator('ma10')} size="small" />}
            label="MA(10)"
          />
          <IndicatorLabel
            control={<Checkbox checked={indicators.ma50} onChange={() => toggleIndicator('ma50')} size="small" />}
            label="MA(50)"
          />
          <IndicatorLabel
            control={<Checkbox checked={indicators.ma200} onChange={() => toggleIndicator('ma200')} size="small" />}
            label="MA(200)"
          />
          <IndicatorLabel
            control={<Checkbox checked={indicators.ema9} onChange={() => toggleIndicator('ema9')} size="small" />}
            label="EMA(9)"
          />
          <IndicatorLabel
            control={<Checkbox checked={indicators.vwap} onChange={() => toggleIndicator('vwap')} size="small" />}
            label="VWAP"
          />
          <IndicatorLabel
            control={<Checkbox checked={indicators.bollinger} onChange={() => toggleIndicator('bollinger')} size="small" />}
            label="BOLL"
          />
          <IndicatorLabel
            control={<Checkbox checked={indicators.rsi} onChange={() => toggleIndicator('rsi')} size="small" />}
            label="RSI(14)"
          />
          <IndicatorLabel
            control={<Checkbox checked={indicators.macd} onChange={() => toggleIndicator('macd')} size="small" />}
            label="MACD(12, 26, 9)"
          />
        </FormGroup>
      </Sidebar>

      <MainContent>
        <Header>
          <IconButton size="small">
            <ArrowBackIcon />
          </IconButton>

          <Box sx={{ flex: 1 }}>
            <Stack direction="row" spacing={2} alignItems="baseline">
              <PriceDisplay>${price.toFixed(2)}</PriceDisplay>
              <ChangeDisplay positive={change >= 0}>
                ({change >= 0 ? '+' : ''}{change.toFixed(2)}) {changePercent.toFixed(2)}%
              </ChangeDisplay>
            </Stack>
            <Typography variant="body2" color="text.secondary">
              {symbol}
            </Typography>
          </Box>

          <IconButton size="small">
            <SearchIcon />
          </IconButton>
        </Header>

        <ChartContainer>
          <div ref={mainChartRef} style={{ width: '100%', height: '60%' }} />
          {indicators.rsi && (
            <div ref={rsiChartRef} style={{ width: '100%', height: '20%' }} />
          )}
          {indicators.macd && (
            <div ref={macdChartRef} style={{ width: '100%', height: '20%' }} />
          )}
        </ChartContainer>
      </MainContent>
    </Container>
  );
};

// Helper functions
async function fetchData(symbol: string) {
  try {
    const response = await fetch(
      `http://localhost:8000/api/v1/market-data/${symbol}/history?period=1d&interval=5m`
    );
    const data = await response.json();

    const candles = data.data.map((item: any) => ({
      time: item.time as Time,
      open: parseFloat(item.open),
      high: parseFloat(item.high),
      low: parseFloat(item.low),
      close: parseFloat(item.close),
      volume: item.volume ? parseInt(item.volume) : 0,
    }));

    return { candles };
  } catch (error) {
    console.error('Error fetching data:', error);
    return null;
  }
}

function calculateMA(data: CandlestickData[], period: number): LineData[] {
  const ma: LineData[] = [];
  for (let i = period - 1; i < data.length; i++) {
    const sum = data.slice(i - period + 1, i + 1).reduce((acc, d) => acc + d.close, 0);
    ma.push({ time: data[i].time, value: sum / period });
  }
  return ma;
}

function calculateEMA(data: CandlestickData[], period: number): LineData[] {
  const multiplier = 2 / (period + 1);
  const ema: LineData[] = [];

  // Start with SMA
  const firstSum = data.slice(0, period).reduce((acc, d) => acc + d.close, 0);
  ema.push({ time: data[period - 1].time, value: firstSum / period });

  // Calculate EMA
  for (let i = period; i < data.length; i++) {
    const prevEMA = ema[ema.length - 1].value;
    const currentEMA = (data[i].close - prevEMA) * multiplier + prevEMA;
    ema.push({ time: data[i].time, value: currentEMA });
  }

  return ema;
}

function addMovingAverage(chart: IChartApi, data: CandlestickData[], period: number, color: string, title: string) {
  const maSeries = chart.addLineSeries({
    color: color,
    lineWidth: 2,
    title: title,
    crosshairMarkerVisible: false,
  });
  maSeries.setData(calculateMA(data, period));
  return maSeries;
}

function addEMA(chart: IChartApi, data: CandlestickData[], period: number, color: string, title: string) {
  const emaSeries = chart.addLineSeries({
    color: color,
    lineWidth: 2,
    title: title,
    crosshairMarkerVisible: false,
  });
  emaSeries.setData(calculateEMA(data, period));
  return emaSeries;
}

function addBollingerBands(chart: IChartApi, data: CandlestickData[]) {
  const period = 20;
  const stdDev = 2;

  const sma = calculateMA(data, period);
  const upper: LineData[] = [];
  const lower: LineData[] = [];

  for (let i = 0; i < sma.length; i++) {
    const dataSlice = data.slice(i + data.length - sma.length - period + 1, i + data.length - sma.length + 1);
    const mean = sma[i].value;
    const variance = dataSlice.reduce((acc, d) => acc + Math.pow(d.close - mean, 2), 0) / period;
    const std = Math.sqrt(variance);

    upper.push({ time: sma[i].time, value: mean + stdDev * std });
    lower.push({ time: sma[i].time, value: mean - stdDev * std });
  }

  const upperSeries = chart.addLineSeries({
    color: 'rgba(96, 125, 139, 0.5)',
    lineWidth: 1,
    crosshairMarkerVisible: false,
  });
  upperSeries.setData(upper);

  const lowerSeries = chart.addLineSeries({
    color: 'rgba(96, 125, 139, 0.5)',
    lineWidth: 1,
    crosshairMarkerVisible: false,
  });
  lowerSeries.setData(lower);

  const middleSeries = chart.addLineSeries({
    color: 'rgba(96, 125, 139, 0.3)',
    lineWidth: 1,
    lineStyle: 2,
    crosshairMarkerVisible: false,
  });
  middleSeries.setData(sma);
}

function addVolume(chart: IChartApi, data: CandlestickData[]) {
  const volumeSeries = chart.addHistogramSeries({
    color: '#e0e0e0',
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

  const volumeData = data.map(d => ({
    time: d.time,
    value: d.volume || 0,
    color: d.close >= d.open ? 'rgba(0, 200, 83, 0.3)' : 'rgba(255, 23, 68, 0.3)',
  }));

  volumeSeries.setData(volumeData);
}

function createRSIChart(container: HTMLDivElement, symbol: string) {
  const chart = createChart(container, {
    width: container.clientWidth,
    height: container.clientHeight,
    layout: {
      background: { type: ColorType.Solid, color: '#ffffff' },
      textColor: '#333',
    },
    grid: {
      vertLines: { color: '#f0f0f0' },
      horzLines: { color: '#f0f0f0' },
    },
    rightPriceScale: {
      borderColor: '#e0e0e0',
    },
    timeScale: {
      borderColor: '#e0e0e0',
      visible: false,
    },
  });

  const rsiSeries = chart.addLineSeries({
    color: '#ff9800',
    lineWidth: 2,
  });

  // Add 70 and 30 lines
  const upperLine = chart.addLineSeries({
    color: '#e0e0e0',
    lineWidth: 1,
    lineStyle: 2,
  });

  const lowerLine = chart.addLineSeries({
    color: '#e0e0e0',
    lineWidth: 1,
    lineStyle: 2,
  });

  // Generate RSI data (simplified)
  const currentTime = Math.floor(Date.now() / 1000);
  const rsiData = Array.from({ length: 100 }, (_, i) => ({
    time: (currentTime - (100 - i) * 300) as Time,
    value: 30 + Math.random() * 40,
  }));

  const lineData = rsiData.map(d => ({ time: d.time, value: 70 }));
  const lowerData = rsiData.map(d => ({ time: d.time, value: 30 }));

  rsiSeries.setData(rsiData);
  upperLine.setData(lineData);
  lowerLine.setData(lowerData);

  chart.applyOptions({
    watermark: {
      visible: true,
      fontSize: 14,
      horzAlign: 'left',
      vertAlign: 'top',
      color: 'rgba(0, 0, 0, 0.3)',
      text: 'RSI(14)',
    },
  });
}

function createMACDChart(container: HTMLDivElement, symbol: string) {
  const chart = createChart(container, {
    width: container.clientWidth,
    height: container.clientHeight,
    layout: {
      background: { type: ColorType.Solid, color: '#ffffff' },
      textColor: '#333',
    },
    grid: {
      vertLines: { color: '#f0f0f0' },
      horzLines: { color: '#f0f0f0' },
    },
    rightPriceScale: {
      borderColor: '#e0e0e0',
    },
    timeScale: {
      borderColor: '#e0e0e0',
      visible: false,
    },
  });

  const macdLine = chart.addLineSeries({
    color: '#2196f3',
    lineWidth: 2,
  });

  const signalLine = chart.addLineSeries({
    color: '#f44336',
    lineWidth: 2,
  });

  const histogram = chart.addHistogramSeries({
    color: '#4caf50',
  });

  // Generate MACD data (simplified)
  const currentTime = Math.floor(Date.now() / 1000);
  const macdData = Array.from({ length: 100 }, (_, i) => {
    const time = (currentTime - (100 - i) * 300) as Time;
    const macd = (Math.random() - 0.5) * 2;
    const signal = macd * 0.8 + (Math.random() - 0.5) * 0.5;
    return {
      time,
      macd,
      signal,
      histogram: macd - signal,
    };
  });

  macdLine.setData(macdData.map(d => ({ time: d.time, value: d.macd })));
  signalLine.setData(macdData.map(d => ({ time: d.time, value: d.signal })));
  histogram.setData(macdData.map(d => ({
    time: d.time,
    value: d.histogram,
    color: d.histogram >= 0 ? 'rgba(0, 200, 83, 0.5)' : 'rgba(255, 23, 68, 0.5)',
  })));

  chart.applyOptions({
    watermark: {
      visible: true,
      fontSize: 14,
      horzAlign: 'left',
      vertAlign: 'top',
      color: 'rgba(0, 0, 0, 0.3)',
      text: 'MACD(12, 26, 9)',
    },
  });
}

export default RobinhoodChart;
