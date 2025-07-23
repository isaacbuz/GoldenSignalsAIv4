/**
 * SignalsChart Component
 *
 * A professional trading signals chart for GoldenSignalsAI
 * Focused on displaying market data and AI-generated signals
 *
 * Features:
 * - Clean candlestick chart with integrated volume
 * - AI signal overlays
 * - Real-time market data
 * - Professional dark theme
 * - Signal accuracy display
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from 'highcharts-react-official';

import {
  Box,
  TextField,
  Typography,
  Select,
  MenuItem,
  FormControl,
  alpha,
  CircularProgress,
  IconButton,
  Chip,
  Autocomplete,
  Paper,
  Fade,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Search as SearchIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Psychology as PsychologyIcon,
  Analytics as AnalyticsIcon,
  Insights as InsightsIcon,
  ShowChart as ShowChartIcon,
  CandlestickChart as CandlestickChartIcon,
  BarChart as BarChartIcon,
  AutoGraph as AutoGraphIcon,
} from '@mui/icons-material';

// Hooks
import { useMarketData } from './hooks/useMarketData';
import { useAgentAnalysis } from './hooks/useAgentAnalysis';
import { aiPredictionService } from '../../services/aiPredictionService';

// Professional dark theme colors with better contrast
const THEME = {
  background: '#0a0a0a',
  surface: '#1a1d23',
  elevated: '#252830',
  border: '#2a2e38',
  text: {
    primary: '#ffffff',
    secondary: '#b8bcc8',
    muted: '#8b92a8',
  },
  accent: {
    green: '#00d68f',
    red: '#ff3b30',
    blue: '#0095ff',
    gold: '#FFD700',
    purple: '#7c3aed',
  },
  chart: {
    grid: '#2a2e38',
    candle: {
      up: '#00d68f',
      down: '#ff3b30',
    },
    volume: 'rgba(139, 146, 168, 0.3)',
  },
};

// Styled components
const Container = styled(Box)({
  width: '100%',
  height: '100vh',
  backgroundColor: THEME.background,
  display: 'flex',
  flexDirection: 'column',
  fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif',
  overflow: 'hidden',
});

const Header = styled(Box)({
  height: 56,
  backgroundColor: THEME.surface,
  borderBottom: `1px solid ${THEME.border}`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '0 20px',
  boxShadow: '0 1px 3px rgba(0, 0, 0, 0.5)',
  position: 'relative',
  zIndex: 10,
});

const SearchContainer = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  gap: 16,
  flex: 1,
  maxWidth: 400,
});

const TimeframeSelect = styled(Select)({
  height: 32,
  backgroundColor: THEME.elevated,
  color: THEME.text.primary,
  fontSize: 13,
  fontWeight: 500,
  '& .MuiOutlinedInput-notchedOutline': {
    border: `1px solid ${THEME.border}`,
  },
  '&:hover .MuiOutlinedInput-notchedOutline': {
    borderColor: THEME.accent.blue,
  },
  '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
    borderColor: THEME.accent.blue,
    borderWidth: 2,
  },
  '& .MuiSelect-select': {
    padding: '6px 32px 6px 12px',
  },
  '& .MuiSvgIcon-root': {
    color: THEME.text.secondary,
  },
});

const PriceDisplay = styled(Box)({
  padding: '16px 20px',
  backgroundColor: THEME.surface,
  borderBottom: `1px solid ${THEME.border}`,
  display: 'flex',
  alignItems: 'flex-start',
  justifyContent: 'space-between',
});

const ChartContainer = styled(Box)({
  flex: 1,
  backgroundColor: THEME.background,
  position: 'relative',
  overflow: 'hidden',
});

const LiveBadge = styled(Chip)({
  height: 20,
  fontSize: 11,
  fontWeight: 600,
  backgroundColor: THEME.accent.green,
  color: THEME.background,
  '& .MuiChip-label': {
    padding: '0 8px',
  },
});

const SignalPanel = styled(Paper)({
  position: 'absolute',
  top: 16,
  right: 16,
  padding: '16px',
  backgroundColor: THEME.elevated,
  backdropFilter: 'blur(20px)',
  border: `1px solid ${THEME.border}`,
  borderRadius: 12,
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
  zIndex: 5,
  minWidth: 220,
});

const AnalyzeButton = styled(Box)({
  display: 'inline-flex',
  alignItems: 'center',
  gap: 8,
  padding: '8px 16px',
  backgroundColor: THEME.accent.gold,
  color: THEME.background,
  borderRadius: 20,
  fontSize: 13,
  fontWeight: 600,
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  '&:hover': {
    backgroundColor: alpha(THEME.accent.gold, 0.8),
    transform: 'translateY(-1px)',
    boxShadow: `0 4px 12px ${alpha(THEME.accent.gold, 0.3)}`,
  },
});

// Symbol suggestions
const SYMBOL_SUGGESTIONS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ',
  'WMT', 'PG', 'UNH', 'MA', 'HD', 'DIS', 'PYPL', 'BAC', 'NFLX', 'ADBE',
];

// Timeframe options
const TIMEFRAMES = [
  { value: '1D', label: '1D' },
  { value: '1W', label: '1W' },
  { value: '1M', label: '1M' },
  { value: '3M', label: '3M' },
  { value: '6M', label: '6M' },
  { value: '1Y', label: '1Y' },
  { value: '5Y', label: '5Y' },
  { value: 'ALL', label: 'ALL' },
];

// Configure Highcharts theme
const chartTheme = {
  colors: [THEME.accent.green, THEME.accent.red, THEME.accent.blue],
  chart: {
    backgroundColor: THEME.background,
    style: {
      fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif',
    },
    animation: true,
  },
  title: {
    style: {
      display: 'none',
    },
  },
  xAxis: {
    gridLineColor: THEME.chart.grid,
    gridLineWidth: 1,
    gridLineDashStyle: 'Dot',
    labels: {
      style: {
        color: THEME.text.secondary,
        fontSize: '12px',
        fontWeight: '500',
      },
    },
    lineColor: THEME.chart.grid,
    tickColor: THEME.chart.grid,
    minorGridLineColor: 'transparent',
  },
  yAxis: {
    gridLineColor: THEME.chart.grid,
    gridLineWidth: 1,
    gridLineDashStyle: 'Dot',
    labels: {
      style: {
        color: THEME.text.secondary,
        fontSize: '12px',
        fontWeight: '500',
      },
      align: 'left',
      x: 8,
      y: -2,
    },
    lineWidth: 0,
    tickWidth: 0,
  },
  tooltip: {
    backgroundColor: THEME.elevated,
    borderColor: THEME.accent.blue,
    borderRadius: 8,
    borderWidth: 1,
    shadow: true,
    style: {
      color: THEME.text.primary,
      fontSize: '13px',
    },
    useHTML: true,
    headerFormat: `<div style="font-size: 12px; color: ${THEME.text.secondary}; margin-bottom: 4px; font-weight: 500;">{point.x:%b %d, %Y}</div>`,
    pointFormat: `<div style="font-size: 16px; font-weight: 600; margin-bottom: 4px;"><b>\${point.close:.2f}</b></div>` +
                 `<div style="font-size: 12px; color: ${THEME.text.secondary};">O: \${point.open:.2f} H: \${point.high:.2f} L: \${point.low:.2f}</div>`,
  },
  plotOptions: {
    candlestick: {
      lineColor: THEME.chart.candle.down,
      upLineColor: THEME.chart.candle.up,
      color: THEME.chart.candle.down,
      upColor: THEME.chart.candle.up,
      lineWidth: 1,
    },
    column: {
      borderWidth: 0,
      borderRadius: 0,
    },
    series: {
      animation: {
        duration: 500,
      },
    },
  },
  rangeSelector: {
    enabled: false,
  },
  navigator: {
    enabled: false,
  },
  scrollbar: {
    enabled: false,
  },
  credits: {
    enabled: false,
  },
  legend: {
    enabled: false,
  },
};

Highcharts.setOptions(chartTheme);

interface SignalsChartProps {
  initialSymbol?: string;
}

export const SignalsChart: React.FC<SignalsChartProps> = ({
  initialSymbol = 'AAPL',
}) => {
  const chartRef = useRef<HighchartsReact.RefObject>(null);

  // State
  const [symbol, setSymbol] = useState(initialSymbol);
  const [timeframe, setTimeframe] = useState('1D');
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'area'>('candlestick');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Hooks
  const { data, loading, error, refetch } = useMarketData(symbol, timeframe);
  const { consensus, analyzeSymbol } = useAgentAnalysis(symbol);

  // Real-time state
  const [isConnected, setIsConnected] = useState(false);
  const [currentPrice, setCurrentPrice] = useState<number | null>(null);
  const [priceChange, setPriceChange] = useState<number>(0);
  const [priceChangePercent, setPriceChangePercent] = useState<number>(0);

  // Calculate price changes
  useEffect(() => {
    if (data && data.candles && data.candles.length > 0) {
      const current = data.candles[data.candles.length - 1].close;
      const previous = data.candles[0].open;
      const change = current - previous;
      const changePercent = (change / previous) * 100;

      setCurrentPrice(current);
      setPriceChange(change);
      setPriceChangePercent(changePercent);
    }
  }, [data]);

  // Convert data to Highcharts format
  const getChartData = useCallback(() => {
    if (!data || !data.candles) return { ohlc: [], volume: [] };

    const ohlc = data.candles.map(candle => ({
      x: new Date(candle.time).getTime(),
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    }));

    const volume = data.candles.map(candle => ({
      x: new Date(candle.time).getTime(),
      y: candle.volume || 0,
      color: candle.close >= candle.open ? alpha(THEME.chart.candle.up, 0.3) : alpha(THEME.chart.candle.down, 0.3),
    }));

    return { ohlc, volume };
  }, [data]);

  // Handle symbol change
  const handleSymbolChange = (event: any, newValue: string | null) => {
    if (newValue) {
      setSymbol(newValue);
    }
  };

  // Handle AI analysis
  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    try {
      await analyzeSymbol(symbol);

      // Get AI prediction
      const prediction = await aiPredictionService.getPrediction(symbol);

      // Add prediction to chart if available
      if (chartRef.current && prediction) {
        const chart = chartRef.current.chart;
        const lastCandle = data.candles[data.candles.length - 1];
        const startTime = new Date(lastCandle.time).getTime();

        // Create prediction line
        const predictionData = [];
        for (let i = 0; i <= 60; i += 5) { // 60 minutes ahead
          const time = startTime + (i * 60000);
          const value = prediction.price_prediction + (prediction.price_prediction * prediction.confidence_interval[0] * i / 60);
          predictionData.push([time, value]);
        }

        // Add prediction series
        chart.addSeries({
          type: 'line',
          name: 'AI Prediction',
          data: predictionData,
          color: THEME.accent.gold,
          lineWidth: 2,
          dashStyle: 'Dash',
          marker: {
            enabled: false,
          },
          zIndex: 10,
        });

        // Add signal markers
        if (prediction.signals && prediction.signals.length > 0) {
          const signalData = prediction.signals.map(signal => ({
            x: new Date(signal.time).getTime(),
            y: signal.price,
            marker: {
              symbol: signal.type === 'BUY' ? 'triangle' : 'triangle-down',
              fillColor: signal.type === 'BUY' ? THEME.accent.green : THEME.accent.red,
              lineColor: signal.type === 'BUY' ? THEME.accent.green : THEME.accent.red,
              lineWidth: 2,
              radius: 8,
            },
          }));

          chart.addSeries({
            type: 'scatter',
            name: 'Trading Signals',
            data: signalData,
            zIndex: 11,
          });
        }
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Initialize and update chart
  useEffect(() => {
    if (!data || !chartRef.current) return;

    const { ohlc, volume } = getChartData();

    const chartOptions: Highcharts.Options = {
      chart: {
        height: '100%',
        spacing: [0, 0, 0, 0],
      },

      time: {
        useUTC: false,
      },

      xAxis: {
        type: 'datetime',
        crosshair: {
          color: THEME.text.muted,
          dashStyle: 'Solid',
          width: 1,
        },
      },

      yAxis: [{
        // Price axis
        labels: {
          align: 'right',
          x: -5,
          formatter: function() {
            return '$' + this.value;
          },
        },
        height: '80%',
        resize: {
          enabled: false,
        },
        crosshair: {
          color: THEME.text.muted,
          dashStyle: 'Solid',
          width: 1,
          snap: false,
        },
      }, {
        // Volume axis
        labels: {
          enabled: false,
        },
        top: '80%',
        height: '20%',
        offset: 0,
        gridLineWidth: 0,
      }],

      series: [{
        type: chartType === 'candlestick' ? 'candlestick' : chartType === 'line' ? 'line' : 'area',
        name: symbol,
        id: 'price',
        data: chartType === 'candlestick' ? ohlc : ohlc.map(point => [point.x, point.close]),
        yAxis: 0,
        color: chartType !== 'candlestick' ? THEME.accent.green : undefined,
        fillColor: chartType === 'area' ? {
          linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
          stops: [
            [0, alpha(THEME.accent.green, 0.3)],
            [1, alpha(THEME.accent.green, 0.0)]
          ]
        } : undefined,
        lineWidth: chartType !== 'candlestick' ? 2 : undefined,
      }, {
        type: 'column',
        name: 'Volume',
        id: 'volume',
        data: volume,
        yAxis: 1,
        opacity: 0.5,
      }],
    };

    chartRef.current.chart.update(chartOptions, true, true);

  }, [data, symbol, chartType]);

  // WebSocket for real-time updates
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/market-data/${symbol}`);

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const update = JSON.parse(event.data);
        if (update.type === 'price_update' && chartRef.current) {
          const chart = chartRef.current.chart;
          const priceSeries = chart.get('price');

          if (priceSeries && priceSeries.data.length > 0) {
            const lastPoint = priceSeries.data[priceSeries.data.length - 1];

            // Update the last candlestick
            lastPoint.update({
              high: Math.max(lastPoint.high, update.price),
              low: Math.min(lastPoint.low, update.price),
              close: update.price,
            }, true, false);

            // Update current price
            setCurrentPrice(update.price);

            // Recalculate price change
            if (priceSeries.data.length > 0) {
              const firstPoint = priceSeries.data[0];
              const change = update.price - firstPoint.open;
              const changePercent = (change / firstPoint.open) * 100;
              setPriceChange(change);
              setPriceChangePercent(changePercent);
            }
          }
        }
      } catch (err) {
        console.error('WebSocket error:', err);
      }
    };

    ws.onerror = () => {
      setIsConnected(false);
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [symbol]);

  const isPositive = priceChange >= 0;

  return (
    <Container>
      {/* Header */}
      <Header>
        <SearchContainer>
          <Autocomplete
            value={symbol}
            onChange={handleSymbolChange}
            options={SYMBOL_SUGGESTIONS}
            sx={{
              width: 240,
              '& .MuiInputBase-root': {
                height: 36,
                backgroundColor: THEME.elevated,
                borderRadius: '18px',
                border: `1px solid ${THEME.border}`,
                '& fieldset': {
                  border: 'none',
                },
                '&:hover': {
                  backgroundColor: THEME.elevated,
                  borderColor: THEME.accent.blue,
                },
                '&.Mui-focused': {
                  backgroundColor: THEME.elevated,
                  borderColor: THEME.accent.blue,
                  boxShadow: `0 0 0 2px ${alpha(THEME.accent.blue, 0.2)}`,
                },
              },
              '& .MuiInputBase-input': {
                color: THEME.text.primary,
                fontSize: 14,
                fontWeight: 500,
                padding: '0 !important',
              },
              '& .MuiAutocomplete-popupIndicator': {
                color: THEME.text.secondary,
              },
              '& .MuiAutocomplete-clearIndicator': {
                color: THEME.text.secondary,
              },
            }}
            renderInput={(params) => (
              <TextField
                {...params}
                placeholder="Search"
                InputProps={{
                  ...params.InputProps,
                  startAdornment: <SearchIcon sx={{ color: THEME.text.muted, mr: 1, fontSize: 18 }} />,
                }}
              />
            )}
            freeSolo
            disableClearable
          />
        </SearchContainer>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <FormControl size="small">
            <TimeframeSelect
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              MenuProps={{
                PaperProps: {
                  sx: {
                    backgroundColor: THEME.elevated,
                    border: `1px solid ${THEME.border}`,
                    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
                    '& .MuiMenuItem-root': {
                      fontSize: 13,
                      color: THEME.text.secondary,
                      padding: '8px 16px',
                      '&:hover': {
                        backgroundColor: alpha(THEME.accent.blue, 0.1),
                        color: THEME.text.primary,
                      },
                      '&.Mui-selected': {
                        backgroundColor: alpha(THEME.accent.blue, 0.2),
                        color: THEME.accent.blue,
                        fontWeight: 600,
                        '&:hover': {
                          backgroundColor: alpha(THEME.accent.blue, 0.3),
                        },
                      },
                    },
                  },
                },
              }}
            >
              {TIMEFRAMES.map(tf => (
                <MenuItem key={tf.value} value={tf.value}>
                  {tf.label}
                </MenuItem>
              ))}
            </TimeframeSelect>
          </FormControl>

          {/* Chart Type Icons */}
          <Box sx={{ display: 'flex', gap: 0.5 }}>
            <IconButton
              size="small"
              sx={{
                backgroundColor: chartType === 'candlestick' ? alpha(THEME.accent.gold, 0.1) : 'transparent',
                color: chartType === 'candlestick' ? THEME.accent.gold : THEME.text.secondary,
                '&:hover': {
                  backgroundColor: alpha(THEME.accent.gold, 0.2),
                  color: THEME.accent.gold
                }
              }}
              onClick={() => setChartType('candlestick')}
            >
              <CandlestickChartIcon fontSize="small" />
            </IconButton>
            <IconButton
              size="small"
              sx={{
                backgroundColor: chartType === 'line' ? alpha(THEME.accent.gold, 0.1) : 'transparent',
                color: chartType === 'line' ? THEME.accent.gold : THEME.text.secondary,
                '&:hover': {
                  backgroundColor: alpha(THEME.accent.gold, 0.2),
                  color: THEME.accent.gold
                }
              }}
              onClick={() => setChartType('line')}
            >
              <ShowChartIcon fontSize="small" />
            </IconButton>
            <IconButton
              size="small"
              sx={{
                backgroundColor: chartType === 'area' ? alpha(THEME.accent.gold, 0.1) : 'transparent',
                color: chartType === 'area' ? THEME.accent.gold : THEME.text.secondary,
                '&:hover': {
                  backgroundColor: alpha(THEME.accent.gold, 0.2),
                  color: THEME.accent.gold
                }
              }}
              onClick={() => setChartType('area')}
            >
              <BarChartIcon fontSize="small" />
            </IconButton>
          </Box>

          {isConnected && <LiveBadge label="LIVE" size="small" />}
        </Box>
      </Header>

      {/* Price Display */}
      <PriceDisplay>
        <Box>
          <Typography variant="body2" sx={{ color: THEME.text.muted, fontSize: 13, mb: 0.5 }}>
            {symbol}
          </Typography>

          {loading ? (
            <Box sx={{ py: 1 }}>
              <CircularProgress size={20} sx={{ color: THEME.text.muted }} />
            </Box>
          ) : currentPrice ? (
            <>
              <Typography variant="h4" sx={{
                color: THEME.text.primary,
                fontWeight: 700,
                fontSize: 32,
                letterSpacing: '-0.02em',
                mb: 0.5,
              }}>
                ${currentPrice.toFixed(2)}
              </Typography>

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {isPositive ? (
                  <TrendingUpIcon sx={{ fontSize: 16, color: THEME.accent.green }} />
                ) : (
                  <TrendingDownIcon sx={{ fontSize: 16, color: THEME.accent.red }} />
                )}
                <Typography sx={{
                  color: isPositive ? THEME.accent.green : THEME.accent.red,
                  fontSize: 14,
                  fontWeight: 500,
                }}>
                  {isPositive ? '+' : ''}{priceChange.toFixed(2)} ({isPositive ? '+' : ''}{priceChangePercent.toFixed(2)}%)
                </Typography>
                <Typography sx={{
                  color: THEME.text.muted,
                  fontSize: 12,
                }}>
                  {timeframe}
                </Typography>
              </Box>
            </>
          ) : null}
        </Box>

        {/* AI Analysis Button */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {consensus && (
            <Chip
              icon={consensus.signal === 'BUY' ? <TrendingUpIcon /> :
                    consensus.signal === 'SELL' ? <TrendingDownIcon /> : <AutoGraphIcon />}
              label={`${consensus.signal} ${consensus.confidence}%`}
              size="small"
              sx={{
                backgroundColor:
                  consensus.signal === 'BUY' ? alpha(THEME.accent.green, 0.2) :
                  consensus.signal === 'SELL' ? alpha(THEME.accent.red, 0.2) :
                  alpha(THEME.accent.blue, 0.2),
                color:
                  consensus.signal === 'BUY' ? THEME.accent.green :
                  consensus.signal === 'SELL' ? THEME.accent.red :
                  THEME.accent.blue,
                border: `1px solid ${
                  consensus.signal === 'BUY' ? THEME.accent.green :
                  consensus.signal === 'SELL' ? THEME.accent.red :
                  THEME.accent.blue
                }`,
                fontWeight: 600,
                fontSize: 11,
              }}
            />
          )}

          <AnalyzeButton onClick={handleAnalyze}>
            {isAnalyzing ? (
              <>
                <CircularProgress size={16} sx={{ color: THEME.background }} />
                Analyzing...
              </>
            ) : (
              <>
                <PsychologyIcon fontSize="small" />
                Analyze with AI
              </>
            )}
          </AnalyzeButton>
        </Box>
      </PriceDisplay>

      {/* Chart */}
      <ChartContainer>
        {error ? (
          <Box sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: THEME.text.muted,
          }}>
            <Typography>Unable to load chart data</Typography>
          </Box>
        ) : (
          <>
            <HighchartsReact
              highcharts={Highcharts}
              constructorType={'stockChart'}
              options={{}}
              ref={chartRef}
              containerProps={{ style: { width: '100%', height: '100%' } }}
            />

            {/* Signal Panel */}
            {consensus && (
              <Fade in>
                <SignalPanel elevation={0}>
                  <Typography variant="caption" sx={{ color: THEME.text.muted, mb: 1, display: 'block', fontWeight: 600 }}>
                    AI Signal Analysis
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 3 }}>
                      <Typography variant="caption" sx={{ color: THEME.text.muted }}>
                        Signal:
                      </Typography>
                      <Typography variant="caption" sx={{
                        color: consensus.signal === 'BUY' ? THEME.accent.green :
                               consensus.signal === 'SELL' ? THEME.accent.red :
                               THEME.accent.blue,
                        fontWeight: 600
                      }}>
                        {consensus.signal}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 3 }}>
                      <Typography variant="caption" sx={{ color: THEME.text.muted }}>
                        Confidence:
                      </Typography>
                      <Typography variant="caption" sx={{ color: THEME.accent.gold, fontWeight: 600 }}>
                        {consensus.confidence}%
                      </Typography>
                    </Box>
                    {consensus.entry_price && (
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 3 }}>
                        <Typography variant="caption" sx={{ color: THEME.text.muted }}>
                          Entry:
                        </Typography>
                        <Typography variant="caption" sx={{ color: THEME.text.primary, fontWeight: 600 }}>
                          ${consensus.entry_price.toFixed(2)}
                        </Typography>
                      </Box>
                    )}
                    {consensus.take_profit && (
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 3 }}>
                        <Typography variant="caption" sx={{ color: THEME.text.muted }}>
                          Target:
                        </Typography>
                        <Typography variant="caption" sx={{ color: THEME.accent.green, fontWeight: 600 }}>
                          ${consensus.take_profit.toFixed(2)}
                        </Typography>
                      </Box>
                    )}
                    {consensus.stop_loss && (
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 3 }}>
                        <Typography variant="caption" sx={{ color: THEME.text.muted }}>
                          Stop:
                        </Typography>
                        <Typography variant="caption" sx={{ color: THEME.accent.red, fontWeight: 600 }}>
                          ${consensus.stop_loss.toFixed(2)}
                        </Typography>
                      </Box>
                    )}
                    {consensus.historical_accuracy && (
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 3, mt: 1, pt: 1, borderTop: `1px solid ${THEME.border}` }}>
                        <Typography variant="caption" sx={{ color: THEME.text.muted }}>
                          Accuracy:
                        </Typography>
                        <Typography variant="caption" sx={{ color: THEME.accent.gold, fontWeight: 600 }}>
                          {consensus.historical_accuracy}%
                        </Typography>
                      </Box>
                    )}
                  </Box>
                </SignalPanel>
              </Fade>
            )}
          </>
        )}
      </ChartContainer>
    </Container>
  );
};
