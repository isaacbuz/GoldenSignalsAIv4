/**
 * RobinhoodDarkChart Component
 *
 * A minimal, professional trading chart inspired by Robinhood's design
 * with Material dark theme for stock trading.
 *
 * Features:
 * - Clean candlestick chart with integrated volume
 * - Minimal UI with focus on data
 * - Professional dark theme
 * - Simple timeframe dropdown
 * - Real-time updates
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
  InputBase,
  alpha,
  useTheme,
  CircularProgress,
  IconButton,
  Chip,
  Autocomplete,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Search as SearchIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  MoreVert as MoreVertIcon,
  Add as AddIcon,
  Remove as RemoveIcon,
  Notifications as NotificationsIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  ShowChart as ShowChartIcon,
  CandlestickChart as CandlestickChartIcon,
  BarChart as BarChartIcon,
  Check as CheckIcon,
} from '@mui/icons-material';

// Hooks
import { useMarketData } from './hooks/useMarketData';
import { useAgentAnalysis } from './hooks/useAgentAnalysis';

// Professional dark theme colors
const THEME = {
  background: '#000000',
  surface: '#0d0d0d',
  border: '#1a1a1a',
  text: {
    primary: '#ffffff',
    secondary: '#999999',
    muted: '#666666',
  },
  accent: {
    green: '#00ff41',
    red: '#ff0033',
    blue: '#0084ff',
  },
  chart: {
    grid: '#1a1a1a',
    candle: {
      up: '#00ff41',
      down: '#ff0033',
    },
    volume: '#333333',
  },
};

// Styled components
const Container = styled(Box)({
  width: '100%',
  height: '100vh',
  backgroundColor: THEME.background,
  display: 'flex',
  fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif',
  overflow: 'hidden',
});

const MainContent = styled(Box)({
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  minWidth: 0,
});

const Sidebar = styled(Box)({
  width: 320,
  backgroundColor: THEME.surface,
  borderLeft: `1px solid ${THEME.border}`,
  display: 'flex',
  flexDirection: 'column',
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
  backgroundColor: 'transparent',
  color: THEME.text.secondary,
  fontSize: 13,
  fontWeight: 500,
  '& .MuiOutlinedInput-notchedOutline': {
    border: `1px solid ${THEME.border}`,
  },
  '&:hover .MuiOutlinedInput-notchedOutline': {
    borderColor: THEME.text.muted,
  },
  '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
    borderColor: THEME.accent.blue,
    borderWidth: 1,
  },
  '& .MuiSelect-select': {
    padding: '6px 32px 6px 12px',
  },
});

const PriceDisplay = styled(Box)({
  padding: '16px 20px',
  backgroundColor: THEME.surface,
  borderBottom: `1px solid ${THEME.border}`,
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

const OrderPanel = styled(Box)({
  padding: 20,
  borderBottom: `1px solid ${THEME.border}`,
});

const OrderButton = styled(Box)<{ variant: 'buy' | 'sell' }>(({ variant }) => ({
  flex: 1,
  padding: '12px 0',
  textAlign: 'center',
  backgroundColor: variant === 'buy' ? THEME.accent.green : THEME.accent.red,
  color: THEME.background,
  fontWeight: 600,
  fontSize: 14,
  borderRadius: 4,
  cursor: 'pointer',
  transition: 'all 0.2s ease',
  '&:hover': {
    opacity: 0.9,
    transform: 'translateY(-1px)',
  },
  '&:active': {
    transform: 'translateY(0)',
  },
}));

const WatchlistItem = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '12px 20px',
  cursor: 'pointer',
  transition: 'background-color 0.2s ease',
  '&:hover': {
    backgroundColor: THEME.border,
  },
});

const PriceAlert = styled(Box)({
  position: 'fixed',
  top: 80,
  right: 20,
  backgroundColor: THEME.surface,
  border: `1px solid ${THEME.border}`,
  borderRadius: 4,
  padding: '12px 16px',
  boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
  zIndex: 1000,
  minWidth: 300,
});

// Symbol suggestions
const SYMBOL_SUGGESTIONS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ',
  'WMT', 'PG', 'UNH', 'MA', 'HD', 'DIS', 'PYPL', 'BAC', 'NFLX', 'ADBE',
];

// Timeframe options
const TIMEFRAMES = [
  { value: '1D', label: '1D', multiplier: 1 },
  { value: '1W', label: '1W', multiplier: 7 },
  { value: '1M', label: '1M', multiplier: 30 },
  { value: '3M', label: '3M', multiplier: 90 },
  { value: '6M', label: '6M', multiplier: 180 },
  { value: '1Y', label: '1Y', multiplier: 365 },
  { value: '5Y', label: '5Y', multiplier: 1825 },
  { value: 'ALL', label: 'ALL', multiplier: 3650 },
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
    gridLineDashStyle: 'Solid',
    labels: {
      style: {
        color: THEME.text.muted,
        fontSize: '11px',
      },
    },
    lineColor: THEME.chart.grid,
    tickColor: THEME.chart.grid,
    minorGridLineColor: 'transparent',
  },
  yAxis: {
    gridLineColor: THEME.chart.grid,
    gridLineWidth: 1,
    gridLineDashStyle: 'Solid',
    labels: {
      style: {
        color: THEME.text.muted,
        fontSize: '11px',
      },
      align: 'left',
      x: 5,
      y: -2,
    },
    lineWidth: 0,
    tickWidth: 0,
  },
  tooltip: {
    backgroundColor: THEME.surface,
    borderColor: THEME.border,
    borderRadius: 4,
    borderWidth: 1,
    shadow: false,
    style: {
      color: THEME.text.primary,
      fontSize: '12px',
    },
    useHTML: true,
    headerFormat: '<div style="font-size: 11px; color: #999; margin-bottom: 4px;">{point.x:%b %d, %Y}</div>',
    pointFormat: '<div><b>${point.close:.2f}</b></div>' +
                 '<div style="font-size: 11px; color: #999;">O: ${point.open:.2f} H: ${point.high:.2f} L: ${point.low:.2f}</div>',
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

interface RobinhoodDarkChartProps {
  initialSymbol?: string;
}

// Watchlist data
const DEFAULT_WATCHLIST = [
  { symbol: 'AAPL', name: 'Apple Inc.', price: 182.52, change: 2.15, changePercent: 1.19 },
  { symbol: 'MSFT', name: 'Microsoft', price: 378.85, change: -1.23, changePercent: -0.32 },
  { symbol: 'GOOGL', name: 'Alphabet', price: 142.65, change: 3.45, changePercent: 2.48 },
  { symbol: 'TSLA', name: 'Tesla', price: 238.45, change: 5.67, changePercent: 2.44 },
  { symbol: 'NVDA', name: 'NVIDIA', price: 485.09, change: 12.34, changePercent: 2.61 },
];

export const RobinhoodDarkChart: React.FC<RobinhoodDarkChartProps> = ({
  initialSymbol = 'AAPL',
}) => {
  const chartRef = useRef<HighchartsReact.RefObject>(null);

  // State
  const [symbol, setSymbol] = useState(initialSymbol);
  const [timeframe, setTimeframe] = useState('1D');
  const [orderQuantity, setOrderQuantity] = useState('1');
  const [watchlist, setWatchlist] = useState(DEFAULT_WATCHLIST);
  const [showAlert, setShowAlert] = useState(false);
  const [alertMessage, setAlertMessage] = useState('');
  const [isFavorite, setIsFavorite] = useState(false);
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'area'>('candlestick');

  // Hooks
  const { data, loading, error, refetch } = useMarketData(symbol, timeframe);
  const { consensus } = useAgentAnalysis(symbol);

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
      setIsFavorite(false);
    }
  };

  // Handle order
  const handleOrder = (type: 'buy' | 'sell') => {
    const quantity = parseInt(orderQuantity) || 1;
    const action = type === 'buy' ? 'Bought' : 'Sold';
    setAlertMessage(`${action} ${quantity} share${quantity > 1 ? 's' : ''} of ${symbol} at $${currentPrice?.toFixed(2)}`);
    setShowAlert(true);
    setTimeout(() => setShowAlert(false), 5000);
  };

  // Handle watchlist item click
  const handleWatchlistClick = (item: typeof DEFAULT_WATCHLIST[0]) => {
    setSymbol(item.symbol);
  };

  // Toggle favorite
  const toggleFavorite = () => {
    setIsFavorite(!isFavorite);
    if (!isFavorite) {
      setAlertMessage(`${symbol} added to favorites`);
      setShowAlert(true);
      setTimeout(() => setShowAlert(false), 3000);
    }
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;

      switch(e.key.toLowerCase()) {
        case 'b':
          handleOrder('buy');
          break;
        case 's':
          handleOrder('sell');
          break;
        case 'f':
          toggleFavorite();
          break;
        case '1':
          setTimeframe('1D');
          break;
        case '2':
          setTimeframe('1W');
          break;
        case '3':
          setTimeframe('1M');
          break;
        case 'c':
          setChartType(prev =>
            prev === 'candlestick' ? 'line' :
            prev === 'line' ? 'area' : 'candlestick'
          );
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [currentPrice, symbol, isFavorite]);

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
          label: {
            enabled: true,
            backgroundColor: THEME.surface,
            borderColor: THEME.border,
            borderRadius: 2,
            borderWidth: 1,
            style: {
              color: THEME.text.primary,
              fontSize: '11px',
            },
          },
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
        dataGrouping: {
          enabled: true,
          forced: true,
          units: [
            ['hour', [1]],
            ['day', [1]],
            ['week', [1]],
            ['month', [1]],
          ],
        },
      }, {
        type: 'column',
        name: 'Volume',
        id: 'volume',
        data: volume,
        yAxis: 1,
        dataGrouping: {
          enabled: true,
          forced: true,
        },
        opacity: 0.5,
      }],

      plotOptions: {
        series: {
          point: {
            events: {
              mouseOver: function() {
                // Update price display on hover
                if (this.series.type === 'candlestick') {
                  setCurrentPrice(this.close);
                }
              },
            },
          },
        },
      },
    };

    chartRef.current.chart.update(chartOptions, true, true);

  }, [data, symbol]);

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
      <MainContent>
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
                backgroundColor: THEME.border,
                borderRadius: '18px',
                '& fieldset': {
                  border: 'none',
                },
                '&:hover': {
                  backgroundColor: alpha(THEME.border, 0.8),
                },
                '&.Mui-focused': {
                  backgroundColor: THEME.border,
                  outline: `1px solid ${THEME.text.muted}`,
                },
              },
              '& .MuiInputBase-input': {
                color: THEME.text.primary,
                fontSize: 14,
                fontWeight: 500,
                padding: '0 !important',
              },
              '& .MuiAutocomplete-popupIndicator': {
                display: 'none',
              },
              '& .MuiAutocomplete-clearIndicator': {
                color: THEME.text.muted,
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
                    backgroundColor: THEME.surface,
                    border: `1px solid ${THEME.border}`,
                    '& .MuiMenuItem-root': {
                      fontSize: 13,
                      color: THEME.text.secondary,
                      '&:hover': {
                        backgroundColor: THEME.border,
                      },
                      '&.Mui-selected': {
                        backgroundColor: THEME.border,
                        color: THEME.text.primary,
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

          {isConnected && <LiveBadge label="LIVE" size="small" />}

          <IconButton size="small" sx={{ color: THEME.text.muted }}>
            <NotificationsIcon />
          </IconButton>

          <IconButton
            size="small"
            sx={{ color: isFavorite ? THEME.accent.green : THEME.text.muted }}
            onClick={toggleFavorite}
          >
            {isFavorite ? <StarIcon /> : <StarBorderIcon />}
          </IconButton>

          <IconButton size="small" sx={{ color: THEME.text.muted }}>
            <MoreVertIcon />
          </IconButton>
        </Box>
      </Header>

      {/* Price Display */}
      <PriceDisplay>
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

            {consensus && (
              <Box sx={{ mt: 1 }}>
                <Chip
                  icon={consensus.signal === 'BUY' ? <TrendingUpIcon /> : <TrendingDownIcon />}
                  label={`AI: ${consensus.signal} ${consensus.confidence}%`}
                  size="small"
                  sx={{
                    backgroundColor: consensus.signal === 'BUY' ?
                      alpha(THEME.accent.green, 0.2) : alpha(THEME.accent.red, 0.2),
                    color: consensus.signal === 'BUY' ? THEME.accent.green : THEME.accent.red,
                    border: `1px solid ${consensus.signal === 'BUY' ? THEME.accent.green : THEME.accent.red}`,
                    fontWeight: 600,
                    fontSize: 11,
                  }}
                />
              </Box>
            )}
          </>
        ) : null}
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
          <HighchartsReact
            highcharts={Highcharts}
            constructorType={'stockChart'}
            options={{}}
            ref={chartRef}
            containerProps={{ style: { width: '100%', height: '100%' } }}
          />
        )}
      </ChartContainer>
      </MainContent>

      {/* Sidebar */}
      <Sidebar>
        {/* Order Panel */}
        <OrderPanel>
          <Typography variant="h6" sx={{ color: THEME.text.primary, mb: 2, fontWeight: 600 }}>
            Trade {symbol}
          </Typography>

          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" sx={{ color: THEME.text.muted, mb: 1, display: 'block' }}>
              Quantity
            </Typography>
            <TextField
              value={orderQuantity}
              onChange={(e) => setOrderQuantity(e.target.value)}
              type="number"
              size="small"
              fullWidth
              sx={{
                '& .MuiInputBase-root': {
                  backgroundColor: THEME.border,
                  color: THEME.text.primary,
                  '& fieldset': {
                    border: 'none',
                  },
                },
                '& .MuiInputBase-input': {
                  textAlign: 'center',
                },
              }}
            />
          </Box>

          <Box sx={{ display: 'flex', gap: 1 }}>
            <OrderButton variant="buy" onClick={() => handleOrder('buy')}>
              Buy
            </OrderButton>
            <OrderButton variant="sell" onClick={() => handleOrder('sell')}>
              Sell
            </OrderButton>
          </Box>

          <Box sx={{ mt: 2, pt: 2, borderTop: `1px solid ${THEME.border}` }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="caption" sx={{ color: THEME.text.muted }}>
                Market Price
              </Typography>
              <Typography variant="caption" sx={{ color: THEME.text.primary }}>
                ${currentPrice?.toFixed(2) || '---'}
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="caption" sx={{ color: THEME.text.muted }}>
                Total
              </Typography>
              <Typography variant="caption" sx={{ color: THEME.text.primary, fontWeight: 600 }}>
                ${((currentPrice || 0) * parseInt(orderQuantity || '1')).toFixed(2)}
              </Typography>
            </Box>
          </Box>
        </OrderPanel>

        {/* Watchlist */}
        <Box sx={{ flex: 1, overflow: 'auto' }}>
          <Box sx={{ p: 2, borderBottom: `1px solid ${THEME.border}` }}>
            <Typography variant="h6" sx={{ color: THEME.text.primary, fontWeight: 600 }}>
              Watchlist
            </Typography>
          </Box>

          {watchlist.map((item) => (
            <WatchlistItem key={item.symbol} onClick={() => handleWatchlistClick(item)}>
              <Box>
                <Typography sx={{ color: THEME.text.primary, fontWeight: 500, fontSize: 14 }}>
                  {item.symbol}
                </Typography>
                <Typography sx={{ color: THEME.text.muted, fontSize: 12 }}>
                  {item.name}
                </Typography>
              </Box>
              <Box sx={{ textAlign: 'right' }}>
                <Typography sx={{ color: THEME.text.primary, fontWeight: 500, fontSize: 14 }}>
                  ${item.price.toFixed(2)}
                </Typography>
                <Typography sx={{
                  color: item.change >= 0 ? THEME.accent.green : THEME.accent.red,
                  fontSize: 12,
                }}>
                  {item.change >= 0 ? '+' : ''}{item.changePercent.toFixed(2)}%
                </Typography>
              </Box>
            </WatchlistItem>
          ))}
        </Box>

        {/* Keyboard Shortcuts */}
        <Box sx={{ p: 2, borderTop: `1px solid ${THEME.border}` }}>
          <Typography variant="caption" sx={{ color: THEME.text.muted, display: 'block', mb: 1 }}>
            Keyboard Shortcuts
          </Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0.5 }}>
            <Typography variant="caption" sx={{ color: THEME.text.muted }}>B - Buy</Typography>
            <Typography variant="caption" sx={{ color: THEME.text.muted }}>S - Sell</Typography>
            <Typography variant="caption" sx={{ color: THEME.text.muted }}>F - Favorite</Typography>
            <Typography variant="caption" sx={{ color: THEME.text.muted }}>C - Chart Type</Typography>
            <Typography variant="caption" sx={{ color: THEME.text.muted }}>1/2/3 - Timeframe</Typography>
          </Box>
        </Box>
      </Sidebar>

      {/* Price Alert */}
      {showAlert && (
        <PriceAlert>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CheckIcon sx={{ color: THEME.accent.green, fontSize: 20 }} />
            <Typography sx={{ color: THEME.text.primary, fontSize: 14 }}>
              {alertMessage}
            </Typography>
          </Box>
        </PriceAlert>
      )}
    </Container>
  );
};
