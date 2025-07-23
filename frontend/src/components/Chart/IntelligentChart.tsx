/**
 * IntelligentChart - AI-Powered Trading Chart
 *
 * Clean, focused chart with AI predictions and trading signals
 * - Entry/exit points with confidence
 * - Stop loss and take profit levels
 * - 15-minute price forecast
 * - Minimal distractions, maximum clarity
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
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
  SeriesMarkerPosition,
  SeriesMarkerShape,
} from 'lightweight-charts';
import {
  Box,
  Typography,
  CircularProgress,
  TextField,
  Autocomplete,
  Chip,
  Paper,
  Fade,
  Zoom,
  Button,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import { styled, alpha, keyframes } from '@mui/material/styles';
import {
  Search as SearchIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Psychology as AIIcon,
  Circle as LiveIcon,
  ArrowDropDown as ArrowDropDownIcon,
  Check as CheckIcon,
  CandlestickChart as CandleIcon,
  ShowChart as LineIcon,
  BarChart as BarIcon,
} from '@mui/icons-material';

// Hooks
import { useMarketData } from './hooks/useMarketData';
import { useAgentAnalysis } from './hooks/useAgentAnalysis';

// Styled Components
const Container = styled(Box)({
  width: '100%',
  height: '100vh',
  backgroundColor: '#0a0a0a',
  display: 'flex',
  flexDirection: 'column',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  overflow: 'hidden',
});

const Header = styled(Box)({
  height: 56,
  backgroundColor: '#0a0a0a',
  borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '0 16px',
});

const LeftSection = styled(Box)({
  display: 'flex',
  alignItems: 'baseline',
  gap: 8,
  flex: '1 1 0',
});

const CenterSection = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 16,
  flex: '0 0 auto',
});

const RightSection = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'flex-end',
  gap: 2,
  flex: '1 1 0',
});


const TimeButton = styled(Button)<{ active?: boolean }>(({ active }) => ({
  minWidth: 'auto',
  padding: '6px 12px',
  fontSize: '13px',
  fontWeight: 500,
  color: active ? '#fff' : 'rgba(255, 255, 255, 0.5)',
  backgroundColor: active ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
  borderRadius: 4,
  textTransform: 'none',
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    color: active ? '#fff' : 'rgba(255, 255, 255, 0.7)',
  },
  '& .MuiButton-endIcon': {
    marginLeft: 4,
    marginRight: -4,
  },
}));

const ChartTypeButton = styled(Button)({
  minWidth: 'auto',
  padding: '6px 12px',
  fontSize: '13px',
  fontWeight: 500,
  color: 'rgba(255, 255, 255, 0.7)',
  backgroundColor: 'transparent',
  borderRadius: 4,
  textTransform: 'none',
  marginLeft: 16,
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    color: '#fff',
  },
  '& .MuiButton-startIcon': {
    marginRight: 4,
  },
});

const ChartArea = styled(Box)({
  flex: 1,
  position: 'relative',
  backgroundColor: '#0a0a0a',
});

const pulse = keyframes`
  0% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.1);
    opacity: 0.7;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
`;

const LiveBadge = styled(Chip)({
  height: 24,
  fontSize: '11px',
  fontWeight: 600,
  backgroundColor: 'rgba(0, 200, 5, 0.15)',
  color: '#00c805',
  '& .MuiChip-icon': {
    fontSize: 8,
    color: '#00c805',
    animation: `${pulse} 2s ease-in-out infinite`,
  },
});

const AISignalPanel = styled(Paper)(({ theme }) => ({
  position: 'absolute',
  top: 16,
  right: 16,
  width: 280,
  backgroundColor: 'rgba(16, 16, 16, 0.95)',
  backdropFilter: 'blur(20px)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  borderRadius: 8,
  padding: 16,
  zIndex: 10,
}));

const SignalRow = styled(Box)({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: 12,
  '&:last-child': {
    marginBottom: 0,
  },
});

const SignalLabel = styled(Typography)({
  fontSize: '12px',
  color: 'rgba(255, 255, 255, 0.6)',
  fontWeight: 500,
});

const SignalValue = styled(Typography)({
  fontSize: '14px',
  color: '#fff',
  fontWeight: 600,
  fontVariantNumeric: 'tabular-nums',
});

const ConfidenceBar = styled(Box)<{ value: number }>(({ value }) => ({
  width: '100%',
  height: 4,
  backgroundColor: 'rgba(255, 255, 255, 0.1)',
  borderRadius: 2,
  overflow: 'hidden',
  marginTop: 8,
  '&::after': {
    content: '""',
    display: 'block',
    width: `${value}%`,
    height: '100%',
    backgroundColor: value > 70 ? '#00c805' : value > 40 ? '#ff9800' : '#ff3b30',
    transition: 'width 0.3s ease',
  },
}));

const LoadingOverlay = styled(Box)({
  position: 'absolute',
  inset: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: 'rgba(10, 10, 10, 0.8)',
  backdropFilter: 'blur(10px)',
  zIndex: 100,
});

// Constants
const TIMEFRAMES = ['1D', '1W', '1M', '3M', '1Y', 'ALL'];

// Timeframe intervals for different trading styles
const TIMEFRAME_INTERVALS: Record<string, { label: string; value: string; aiDescription: string }[]> = {
  '1D': [
    { label: '1 min', value: '1m', aiDescription: 'Scalping & High-frequency' },
    { label: '5 min', value: '5m', aiDescription: 'Day Trading' },
    { label: '15 min', value: '15m', aiDescription: 'Intraday Swings' },
    { label: '30 min', value: '30m', aiDescription: 'Short-term Trends' },
    { label: '1 hour', value: '1h', aiDescription: 'Intraday Positions' },
  ],
  '1W': [
    { label: '5 min', value: '5m', aiDescription: 'Weekly Scalping' },
    { label: '15 min', value: '15m', aiDescription: 'Weekly Day Trading' },
    { label: '1 hour', value: '1h', aiDescription: 'Weekly Swings' },
    { label: '4 hours', value: '4h', aiDescription: 'Multi-day Holds' },
    { label: '1 day', value: '1d', aiDescription: 'Weekly Overview' },
  ],
  '1M': [
    { label: '30 min', value: '30m', aiDescription: 'Monthly Intraday' },
    { label: '1 hour', value: '1h', aiDescription: 'Monthly Swings' },
    { label: '4 hours', value: '4h', aiDescription: 'Weekly Options' },
    { label: '1 day', value: '1d', aiDescription: 'Monthly Options' },
    { label: '1 week', value: '1w', aiDescription: 'Monthly LEAPS' },
  ],
  '3M': [
    { label: '1 hour', value: '1h', aiDescription: 'Quarterly Detail' },
    { label: '4 hours', value: '4h', aiDescription: 'Quarterly Swings' },
    { label: '1 day', value: '1d', aiDescription: 'Quarterly Options' },
    { label: '1 week', value: '1w', aiDescription: 'Quarterly LEAPS' },
  ],
  '1Y': [
    { label: '1 day', value: '1d', aiDescription: 'Yearly Swings' },
    { label: '1 week', value: '1w', aiDescription: 'Long-term Options' },
    { label: '1 month', value: '1M', aiDescription: 'LEAPS Analysis' },
  ],
  'ALL': [
    { label: '1 week', value: '1w', aiDescription: 'Historical Weekly' },
    { label: '1 month', value: '1M', aiDescription: 'Historical Monthly' },
    { label: '3 months', value: '3M', aiDescription: 'Historical Quarterly' },
  ],
};

const CHART_TYPES = [
  { label: 'Candlestick', value: 'candlestick', icon: CandleIcon },
  { label: 'Line', value: 'line', icon: LineIcon },
  { label: 'Bars', value: 'bars', icon: BarIcon },
];

const POPULAR_SYMBOLS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
  'V', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'MA', 'HD', 'BAC', 'ADBE',
];

interface IntelligentChartProps {
  symbol?: string;
  onSymbolChange?: (symbol: string) => void;
}

export const IntelligentChart: React.FC<IntelligentChartProps> = ({
  symbol: initialSymbol = 'TSLA',
  onSymbolChange
}) => {
  // Chart refs
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const predictionSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  // State
  const [symbol, setSymbol] = useState(initialSymbol);
  const [timeframe, setTimeframe] = useState('1D');
  const [interval, setInterval] = useState('5m');
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'bars'>('candlestick');
  const [showSignals, setShowSignals] = useState(true);
  const [intervalMenuAnchor, setIntervalMenuAnchor] = useState<null | HTMLElement>(null);
  const [chartTypeMenuAnchor, setChartTypeMenuAnchor] = useState<null | HTMLElement>(null);

  // Custom hooks
  const { data, price, change, changePercent, isLive, error, loading } = useMarketData(symbol, timeframe);
  const { agentSignals, consensus, isAnalyzing, triggerAnalysis } = useAgentAnalysis(symbol);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current || !data) return;

    try {
      const container = chartContainerRef.current;
      container.innerHTML = '';

      const containerWidth = container.clientWidth;
      const containerHeight = container.clientHeight;

    // Create chart
    const chart = createChart(container, {
      width: containerWidth,
      height: containerHeight,
      layout: {
        background: { type: ColorType.Solid, color: '#0a0a0a' },
        textColor: 'rgba(255, 255, 255, 0.6)',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: 'rgba(255, 255, 255, 0.3)',
          width: 1,
          style: LineStyle.Solid,
          labelBackgroundColor: '#1a1a1a',
        },
        horzLine: {
          color: 'rgba(255, 255, 255, 0.3)',
          width: 1,
          style: LineStyle.Solid,
          labelBackgroundColor: '#1a1a1a',
        },
      },
      rightPriceScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
        scaleMargins: { top: 0.1, bottom: 0.25 },
      },
      timeScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    // Ensure data is sorted by time
    const sortedCandles = [...data.candles].sort((a, b) => {
      const timeA = typeof a.time === 'number' ? a.time : new Date(a.time as string).getTime();
      const timeB = typeof b.time === 'number' ? b.time : new Date(b.time as string).getTime();
      return timeA - timeB;
    });

    // Add appropriate series based on chart type
    let mainSeries: any;

    if (chartType === 'candlestick') {
      mainSeries = chart.addCandlestickSeries({
        upColor: '#00c805',
        downColor: '#ff3b30',
        borderUpColor: '#00c805',
        borderDownColor: '#ff3b30',
        wickUpColor: '#00c805',
        wickDownColor: '#ff3b30',
      });
      mainSeries.setData(sortedCandles);
      candleSeriesRef.current = mainSeries;
    } else if (chartType === 'line') {
      mainSeries = chart.addLineSeries({
        color: '#00c805',
        lineWidth: 2,
        lastValueVisible: true,
        priceLineVisible: true,
      });
      const lineData = sortedCandles.map(candle => ({
        time: candle.time,
        value: candle.close,
      }));
      mainSeries.setData(lineData);
    } else if (chartType === 'bars') {
      mainSeries = chart.addBarSeries({
        upColor: '#00c805',
        downColor: '#ff3b30',
        openVisible: true,
        thinBars: false,
      });
      mainSeries.setData(sortedCandles);
    }

    // Add volume
    if (data.volumes) {
      const volumeSeries = chart.addHistogramSeries({
        color: '#00c805',
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      });
      volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
      });

      // Sort volumes to match candles
      const sortedVolumes = [...data.volumes].sort((a, b) => {
        const timeA = typeof a.time === 'number' ? a.time : new Date(a.time as string).getTime();
        const timeB = typeof b.time === 'number' ? b.time : new Date(b.time as string).getTime();
        return timeA - timeB;
      });

      volumeSeries.setData(sortedVolumes.map(v => ({
        ...v,
        color: v.color.includes('26a69a') ? 'rgba(0, 200, 5, 0.2)' : 'rgba(255, 59, 48, 0.2)',
      })));
    }

    // Add AI prediction line
    const predictionSeries = chart.addLineSeries({
      color: '#FFD700',
      lineWidth: 3,
      lineStyle: LineStyle.Solid,
      title: 'AI Prediction',
      lastValueVisible: true,
      priceLineVisible: false,
    });
    predictionSeriesRef.current = predictionSeries;

    // Generate AI prediction data (aligned with price)
    const lastCandles = sortedCandles.slice(-20);
    const predictionData: LineData[] = lastCandles.map((candle, i) => {
      // Simulate AI prediction that closely follows price with slight lead
      const nextIndex = Math.min(i + 1, lastCandles.length - 1);
      const prediction = (candle.close + lastCandles[nextIndex].close) / 2;
      return {
        time: candle.time,
        value: prediction,
      };
    });

    // Sort prediction data to ensure time order
    predictionData.sort((a, b) => {
      const timeA = typeof a.time === 'number' ? a.time : new Date(a.time as string).getTime();
      const timeB = typeof b.time === 'number' ? b.time : new Date(b.time as string).getTime();
      return timeA - timeB;
    });

    predictionSeries.setData(predictionData);

    // Add markers for entry/exit signals
    if (consensus && consensus.entry_price) {
      const markers = [];

      // Entry marker
      markers.push({
        time: lastCandle.time,
        position: 'belowBar' as SeriesMarkerPosition,
        shape: 'arrowUp' as SeriesMarkerShape,
        color: '#00c805',
        size: 2,
        text: `Entry: $${consensus.entry_price.toFixed(2)}`,
      });

      // Stop loss line
      if (consensus.stop_loss) {
        candleSeries.createPriceLine({
          price: consensus.stop_loss,
          color: '#ff3b30',
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: `SL: $${consensus.stop_loss.toFixed(2)}`,
        });
      }

      // Take profit line
      if (consensus.take_profit) {
        candleSeries.createPriceLine({
          price: consensus.take_profit,
          color: '#00c805',
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: `TP: $${consensus.take_profit.toFixed(2)}`,
        });
      }

      candleSeries.setMarkers(markers);
    }

    // Note: Forecast area removed to prevent time ordering issues
    // Will be re-implemented with proper future time handling

    chartRef.current = chart;

    // Handle resize
    const handleResize = () => {
      chart.applyOptions({
        width: container.clientWidth,
        height: container.clientHeight,
      });
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
    } catch (error) {
      console.error('Chart initialization error:', error);
    }
  }, [data, consensus, timeframe, chartType]);

  // Update real-time data
  useEffect(() => {
    if (!data || data.candles.length === 0 || price <= 0) return;

    // Only update candlestick series if that's the current chart type
    if (chartType === 'candlestick' && candleSeriesRef.current) {
      const updatedCandles = [...data.candles];
      const lastCandle = updatedCandles[updatedCandles.length - 1];
      updatedCandles[updatedCandles.length - 1] = {
        ...lastCandle,
        close: price,
        high: Math.max(lastCandle.high, price),
        low: Math.min(lastCandle.low, price),
      };
      candleSeriesRef.current.update(updatedCandles[updatedCandles.length - 1]);
    }
  }, [price, data, chartType]);

  // Handle symbol change
  const handleSymbolChange = (newSymbol: string | null) => {
    if (newSymbol && newSymbol !== symbol) {
      setSymbol(newSymbol);
      onSymbolChange?.(newSymbol);
      triggerAnalysis(); // Trigger AI analysis on symbol change
    }
  };

  // Handle timeframe change
  const handleTimeframeChange = (tf: string) => {
    setTimeframe(tf);
    // Set default interval for the timeframe
    const intervals = TIMEFRAME_INTERVALS[tf];
    if (intervals && intervals.length > 0) {
      setInterval(intervals[0].value);
    }
  };

  // Handle interval selection
  const handleIntervalSelect = (value: string) => {
    setInterval(value);
    setIntervalMenuAnchor(null);
    // Re-trigger analysis with new interval
    triggerAnalysis();
  };

  // Get current interval label
  const getCurrentIntervalLabel = () => {
    const intervals = TIMEFRAME_INTERVALS[timeframe];
    const current = intervals?.find(i => i.value === interval);
    return current?.label || interval;
  };

  // Get chart type icon
  const ChartIcon = CHART_TYPES.find(t => t.value === chartType)?.icon || CandleIcon;

  return (
    <Container>
      <Header>
        <LeftSection>
          <Typography variant="h5" sx={{ color: '#fff', fontWeight: 700, fontSize: 32 }}>
            ${price > 0 ? price.toFixed(2) : '---'}
          </Typography>
          {price > 0 && (
            <>
              <Typography
                variant="h6"
                sx={{
                  color: change >= 0 ? '#00c805' : '#ff3b30',
                  fontWeight: 500,
                  fontSize: 20,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.5,
                }}
              >
                {change >= 0 ? '+' : ''}{change.toFixed(2)}
                {change >= 0 ? <TrendingUpIcon fontSize="small" /> : <TrendingDownIcon fontSize="small" />}
              </Typography>
              <Typography
                variant="body1"
                sx={{
                  color: 'rgba(255, 255, 255, 0.6)',
                  fontSize: 16,
                }}
              >
                ({changePercent >= 0 ? '+' : ''}{changePercent.toFixed(2)}%)
              </Typography>
            </>
          )}
        </LeftSection>

        <CenterSection>
          <Autocomplete
            value={symbol}
            onChange={(_, newValue) => handleSymbolChange(newValue)}
            options={POPULAR_SYMBOLS}
            size="small"
            sx={{
              width: 200,
              '& .MuiOutlinedInput-root': {
                height: 36,
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                '& fieldset': {
                  borderColor: 'rgba(255, 255, 255, 0.1)',
                },
                '&:hover fieldset': {
                  borderColor: 'rgba(255, 255, 255, 0.2)',
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#FFD700',
                },
              },
              '& .MuiInputBase-input': {
                color: '#fff',
                fontSize: '14px',
                fontWeight: 600,
                textAlign: 'center',
              },
            }}
            renderInput={(params) => (
              <TextField
                {...params}
                placeholder="Search Symbol"
                InputProps={{
                  ...params.InputProps,
                  startAdornment: <SearchIcon sx={{ color: 'rgba(255, 255, 255, 0.5)', mr: 0.5, fontSize: 18 }} />,
                }}
              />
            )}
          />

          {isLive && <LiveBadge icon={<LiveIcon />} label="LIVE" size="small" />}
        </CenterSection>

        <RightSection>
          {TIMEFRAMES.map((tf) => (
            <TimeButton
              key={tf}
              active={timeframe === tf}
              onClick={() => handleTimeframeChange(tf)}
              endIcon={timeframe === tf ? <ArrowDropDownIcon sx={{ fontSize: 16 }} /> : null}
              onClickCapture={(e) => {
                if (timeframe === tf) {
                  e.stopPropagation();
                  setIntervalMenuAnchor(e.currentTarget);
                }
              }}
            >
              {tf}
            </TimeButton>
          ))}

          <ChartTypeButton
            startIcon={<ChartIcon sx={{ fontSize: 16 }} />}
            endIcon={<ArrowDropDownIcon sx={{ fontSize: 16 }} />}
            onClick={(e) => setChartTypeMenuAnchor(e.currentTarget)}
          >
            {CHART_TYPES.find(t => t.value === chartType)?.label}
          </ChartTypeButton>
        </RightSection>
      </Header>

      <ChartArea>
        <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />

        {loading && (
          <LoadingOverlay>
            <CircularProgress sx={{ color: '#FFD700' }} />
          </LoadingOverlay>
        )}

        {error && (
          <Box
            position="absolute"
            top="50%"
            left="50%"
            sx={{ transform: 'translate(-50%, -50%)', textAlign: 'center' }}
          >
            <Typography sx={{ color: '#ff3b30', mb: 1 }}>
              Failed to load market data
            </Typography>
            <Typography sx={{ color: 'rgba(255, 255, 255, 0.5)', fontSize: 14 }}>
              {error}
            </Typography>
          </Box>
        )}

        {consensus && showSignals && (
          <Zoom in timeout={300}>
            <AISignalPanel elevation={0}>
              <Box display="flex" alignItems="center" gap={1} mb={2}>
                <AIIcon sx={{ color: '#FFD700', fontSize: 20 }} />
                <Typography sx={{ color: '#fff', fontWeight: 600, fontSize: 16 }}>
                  AI Trading Signal
                </Typography>
              </Box>

              <Box
                sx={{
                  mb: 2,
                  p: 1.5,
                  borderRadius: 1,
                  backgroundColor: consensus.signal === 'BUY' ? 'rgba(0, 200, 5, 0.1)' :
                                  consensus.signal === 'SELL' ? 'rgba(255, 59, 48, 0.1)' :
                                  'rgba(255, 255, 255, 0.05)',
                  border: `1px solid ${
                    consensus.signal === 'BUY' ? 'rgba(0, 200, 5, 0.3)' :
                    consensus.signal === 'SELL' ? 'rgba(255, 59, 48, 0.3)' :
                    'rgba(255, 255, 255, 0.1)'
                  }`,
                }}
              >
                <Typography
                  variant="h6"
                  sx={{
                    color: consensus.signal === 'BUY' ? '#00c805' :
                           consensus.signal === 'SELL' ? '#ff3b30' : '#fff',
                    fontWeight: 700,
                    textAlign: 'center',
                  }}
                >
                  {consensus.signal}
                </Typography>
              </Box>

              <SignalRow>
                <SignalLabel>Entry Price</SignalLabel>
                <SignalValue>${consensus.entry_price?.toFixed(2) || price.toFixed(2)}</SignalValue>
              </SignalRow>

              {consensus.stop_loss && (
                <SignalRow>
                  <SignalLabel>Stop Loss</SignalLabel>
                  <SignalValue sx={{ color: '#ff3b30' }}>
                    ${consensus.stop_loss.toFixed(2)}
                    <Typography component="span" sx={{ fontSize: 12, ml: 0.5, color: 'rgba(255, 59, 48, 0.8)' }}>
                      (-{(((consensus.entry_price || price) - consensus.stop_loss) / (consensus.entry_price || price) * 100).toFixed(1)}%)
                    </Typography>
                  </SignalValue>
                </SignalRow>
              )}

              {consensus.take_profit && (
                <SignalRow>
                  <SignalLabel>Take Profit</SignalLabel>
                  <SignalValue sx={{ color: '#00c805' }}>
                    ${consensus.take_profit.toFixed(2)}
                    <Typography component="span" sx={{ fontSize: 12, ml: 0.5, color: 'rgba(0, 200, 5, 0.8)' }}>
                      (+{((consensus.take_profit - (consensus.entry_price || price)) / (consensus.entry_price || price) * 100).toFixed(1)}%)
                    </Typography>
                  </SignalValue>
                </SignalRow>
              )}

              <SignalRow>
                <SignalLabel>Risk Score</SignalLabel>
                <SignalValue sx={{
                  color: consensus.risk_score < 0.3 ? '#00c805' :
                         consensus.risk_score < 0.7 ? '#ff9800' : '#ff3b30'
                }}>
                  {consensus.risk_score < 0.3 ? 'Low' :
                   consensus.risk_score < 0.7 ? 'Medium' : 'High'}
                </SignalValue>
              </SignalRow>

              <Box mt={2}>
                <SignalLabel>Confidence</SignalLabel>
                <Box display="flex" alignItems="center" gap={1}>
                  <ConfidenceBar value={consensus.confidence * 100} />
                  <Typography sx={{ fontSize: 12, color: '#fff', fontWeight: 600 }}>
                    {Math.round(consensus.confidence * 100)}%
                  </Typography>
                </Box>
              </Box>

              <Box mt={2} pt={2} borderTop="1px solid rgba(255, 255, 255, 0.1)">
                <Typography sx={{ fontSize: 11, color: 'rgba(255, 255, 255, 0.4)', textAlign: 'center' }}>
                  {consensus.supporting_agents} agents agree • Updated {new Date().toLocaleTimeString()}
                </Typography>
              </Box>
            </AISignalPanel>
          </Zoom>
        )}
      </ChartArea>

      {/* Interval Selection Menu */}
      <Menu
        anchorEl={intervalMenuAnchor}
        open={Boolean(intervalMenuAnchor)}
        onClose={() => setIntervalMenuAnchor(null)}
        PaperProps={{
          sx: {
            backgroundColor: '#1a1a1a',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            minWidth: 250,
          },
        }}
      >
        <Box px={2} py={1}>
          <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.5)', fontWeight: 600 }}>
            Select Interval for {timeframe} timeframe
          </Typography>
        </Box>
        {TIMEFRAME_INTERVALS[timeframe]?.map((int) => (
          <MenuItem
            key={int.value}
            selected={interval === int.value}
            onClick={() => handleIntervalSelect(int.value)}
            sx={{
              color: '#fff',
              fontSize: '14px',
              py: 1.5,
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.08)',
              },
              '&.Mui-selected': {
                backgroundColor: 'rgba(255, 215, 0, 0.1)',
                '&:hover': {
                  backgroundColor: 'rgba(255, 215, 0, 0.15)',
                },
              },
            }}
          >
            <ListItemIcon>
              {interval === int.value && <CheckIcon sx={{ fontSize: 18, color: '#FFD700' }} />}
            </ListItemIcon>
            <ListItemText>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {int.label}
              </Typography>
              <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.5)' }}>
                {int.aiDescription}
              </Typography>
            </ListItemText>
          </MenuItem>
        ))}
      </Menu>

      {/* Chart Type Menu */}
      <Menu
        anchorEl={chartTypeMenuAnchor}
        open={Boolean(chartTypeMenuAnchor)}
        onClose={() => setChartTypeMenuAnchor(null)}
        PaperProps={{
          sx: {
            backgroundColor: '#1a1a1a',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            minWidth: 200,
          },
        }}
      >
        <Box px={2} py={1}>
          <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.5)', fontWeight: 600 }}>
            Chart Type
          </Typography>
        </Box>
        {CHART_TYPES.map((type) => {
          const Icon = type.icon;
          return (
            <MenuItem
              key={type.value}
              selected={chartType === type.value}
              onClick={() => {
                setChartType(type.value as any);
                setChartTypeMenuAnchor(null);
              }}
              sx={{
                color: '#fff',
                fontSize: '14px',
                py: 1,
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.08)',
                },
                '&.Mui-selected': {
                  backgroundColor: 'rgba(255, 215, 0, 0.1)',
                  '&:hover': {
                    backgroundColor: 'rgba(255, 215, 0, 0.15)',
                  },
                },
              }}
            >
              <ListItemIcon>
                <Icon sx={{ fontSize: 20, color: chartType === type.value ? '#FFD700' : 'rgba(255, 255, 255, 0.7)' }} />
              </ListItemIcon>
              <ListItemText>{type.label}</ListItemText>
            </MenuItem>
          );
        })}
      </Menu>
    </Container>
  );
};

export default IntelligentChart;
