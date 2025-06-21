import React, { useEffect, useRef, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  IconButton,
  Chip,
  Typography,
  Stack,
  Button,
  InputAdornment,
  useTheme,
  alpha,
  CircularProgress,
  ButtonGroup,
  Divider,
  Tooltip,
  Grid,
  Paper,
} from '@mui/material';
import {
  Search as SearchIcon,
  TrendingUp,
  TrendingDown,
  Refresh,
  Fullscreen,
  Timeline,
  ShowChart,
  BarChart,
  Flag as TargetIcon,
  Warning as ShieldAlertIcon,
  AttachMoney as DollarSignIcon,
  Speed as ActivityIcon,
} from '@mui/icons-material';
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time, LineData, SeriesMarker } from 'lightweight-charts';
import { useQuery } from '@tanstack/react-query';
import { apiClient, MarketData, Signal, SignalsResponse } from '../../services/api';

interface SignalsChartProps {
  defaultSymbol?: string;
  height?: number;
  onSelectSignal?: (signal: Signal) => void;
}

interface ChartData {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface Prediction {
  time: Time;
  price: number;
  confidence: number;
}

export default function SignalsChart({ defaultSymbol = 'AAPL', height = 600, onSelectSignal }: SignalsChartProps) {
  const theme = useTheme();
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const predictionSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  
  const [symbol, setSymbol] = useState(defaultSymbol);
  const [searchInput, setSearchInput] = useState(defaultSymbol);
  const [selectedPeriod, setSelectedPeriod] = useState('1D');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'bar'>('candlestick');
  const [showPrediction, setShowPrediction] = useState(true);
  const [activeSignals, setActiveSignals] = useState<Signal[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);

  // Time periods
  const timePeriods = [
    { label: '1m', value: '1m' },
    { label: '5m', value: '5m' },
    { label: '15m', value: '15m' },
    { label: '30m', value: '30m' },
    { label: '1h', value: '1h' },
    { label: '4h', value: '4h' },
    { label: '1D', value: '1D' },
  ];

  // Fetch market data
  const { data: marketData, isLoading, error, refetch } = useQuery<MarketData>({
    queryKey: ['market-data', symbol, selectedPeriod],
    queryFn: () => apiClient.getMarketData({ symbol, timeframe: selectedPeriod }),
    refetchInterval: 30000,
    enabled: !!symbol,
  });

  // Fetch signals
  const { data: signalsData } = useQuery<SignalsResponse>({
    queryKey: ['signals', symbol],
    queryFn: () => apiClient.getSignals({ symbols: [symbol] }),
    refetchInterval: 60000,
    enabled: !!symbol,
  });

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Clean up previous chart
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: height - 120,
      layout: {
        background: { color: 'transparent' },
        textColor: theme.palette.text.primary,
        fontSize: 12,
        fontFamily: theme.typography.fontFamily,
      },
      grid: {
        vertLines: { color: alpha(theme.palette.divider, 0.1) },
        horzLines: { color: alpha(theme.palette.divider, 0.1) },
      },
      crosshair: {
        mode: 1,
        vertLine: { color: theme.palette.primary.main, width: 1, style: 2 },
        horzLine: { color: theme.palette.primary.main, width: 1, style: 2 },
      },
      rightPriceScale: {
        borderColor: alpha(theme.palette.divider, 0.2),
        textColor: theme.palette.text.secondary,
      },
      timeScale: {
        borderColor: alpha(theme.palette.divider, 0.2),
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    // Add candlestick series
    candlestickSeriesRef.current = chart.addCandlestickSeries({
      upColor: '#00D4AA',
      downColor: '#FF3B30',
      borderUpColor: '#00D4AA',
      borderDownColor: '#FF3B30',
      wickUpColor: '#00D4AA',
      wickDownColor: '#FF3B30',
    });

    // Add prediction series
    predictionSeriesRef.current = chart.addLineSeries({
      color: '#8B5CF6',
      lineWidth: 2,
      lineStyle: 2,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
      crosshairMarkerBorderColor: '#8B5CF6',
      crosshairMarkerBackgroundColor: '#8B5CF6',
    });

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
      }
    };
  }, [theme, height]);

  // Update chart data
  useEffect(() => {
    if (!chartRef.current || !candlestickSeriesRef.current || !marketData) return;

    const chartData = marketData.data.map((item: ChartData) => ({
      time: item.time,
      open: item.open,
      high: item.high,
      low: item.low,
      close: item.close,
    }));

    candlestickSeriesRef.current.setData(chartData);

    // Add signals as markers
    if (signalsData) {
      const markers: SeriesMarker<Time>[] = signalsData.signals.map((signal: Signal) => ({
        time: signal.time,
        position: signal.type === 'ENTRY' ? 'belowBar' as const : 'aboveBar' as const,
        color: signal.type === 'ENTRY' ? '#00D4AA' : '#FF3B30',
        shape: signal.type === 'ENTRY' ? 'arrowUp' as const : 'arrowDown' as const,
        text: `${signal.type}: $${signal.price} (${signal.confidence}%)`,
      }));

      candlestickSeriesRef.current.setMarkers(markers);
    }

    // Add prediction line
    if (showPrediction && predictions.length > 0) {
      const predictionData = predictions.map(p => ({
        time: p.time,
        value: p.price,
      })) as LineData<Time>[];
      predictionSeriesRef.current?.setData(predictionData);
    }

    // Add price lines for stop loss and take profit
    if (signalsData) {
      const stopLoss = signalsData.signals.find((s: Signal) => s.type === 'STOP_LOSS');
      const takeProfit = signalsData.signals.find((s: Signal) => s.type === 'TAKE_PROFIT');

      if (stopLoss) {
        candlestickSeriesRef.current.createPriceLine({
          price: stopLoss.price,
          color: '#FF3B30',
          lineWidth: 2,
          lineStyle: 2,
          axisLabelVisible: true,
          title: 'Stop Loss',
        });
      }

      if (takeProfit) {
        candlestickSeriesRef.current.createPriceLine({
          price: takeProfit.price,
          color: '#00D4AA',
          lineWidth: 2,
          lineStyle: 2,
          axisLabelVisible: true,
          title: 'Take Profit',
        });
      }
    }
  }, [marketData, signalsData, predictions, showPrediction]);

  const handleSearch = () => {
    if (searchInput.trim()) {
      setSymbol(searchInput.trim().toUpperCase());
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <Card
      sx={{
        height: '100%',
        background: 'rgba(15, 15, 15, 0.8)',
        backdropFilter: 'blur(40px) saturate(180%)',
        border: '1px solid rgba(255, 255, 255, 0.05)',
        borderRadius: 3,
        overflow: 'hidden',
      }}
    >
      <CardContent sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
        {/* Controls Row */}
        <Grid container spacing={2} alignItems="center" sx={{ mb: 2 }}>
          <Grid item xs={12} sm={6}>
            <ButtonGroup size="small" sx={{ borderRadius: 2 }}>
              <Button
                startIcon={<ShowChart />}
                variant={chartType === 'candlestick' ? 'contained' : 'outlined'}
                onClick={() => setChartType('candlestick')}
              >Candlestick</Button>
              <Button
                startIcon={<Timeline />}
                variant={chartType === 'line' ? 'contained' : 'outlined'}
                onClick={() => setChartType('line')}
              >Line</Button>
              <Button
                startIcon={<BarChart />}
                variant={chartType === 'bar' ? 'contained' : 'outlined'}
                onClick={() => setChartType('bar')}
              >Bar</Button>
            </ButtonGroup>
          </Grid>
          <Grid item xs={12} sm={6} sx={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              variant={showPrediction ? 'contained' : 'outlined'}
              color={showPrediction ? 'warning' : 'inherit'}
              onClick={() => setShowPrediction(v => !v)}
            >
              {showPrediction ? 'Hide Prediction' : 'Show Prediction'}
            </Button>
          </Grid>
        </Grid>

        {/* Header */}
        <Grid container spacing={2} alignItems="center" sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <TextField
                size="small"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value.toUpperCase())}
                onKeyPress={handleKeyPress}
                placeholder="Enter symbol..."
                sx={{
                  width: 140,
                  '& .MuiOutlinedInput-root': {
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: 2,
                    '& fieldset': { border: '1px solid rgba(255, 255, 255, 0.2)' },
                    '&:hover fieldset': { border: '1px solid rgba(255, 255, 255, 0.3)' },
                    '&.Mui-focused fieldset': { border: '2px solid #0066FF' },
                  },
                }}
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton size="small" onClick={handleSearch}>
                        <SearchIcon sx={{ color: 'rgba(255, 255, 255, 0.6)' }} />
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
              />
              {isLoading && <CircularProgress size={20} />}
            </Box>
          </Grid>
          <Grid item xs={12} sm={6} sx={{ textAlign: 'right' }}>
            <Typography variant="h4" sx={{ fontWeight: 700, color: 'white', mb: 0.5 }}>
              {symbol}
            </Typography>
            <Stack direction="row" alignItems="center" spacing={1} justifyContent="flex-end">
              <Typography variant="h5" sx={{ fontWeight: 600, color: 'white' }}>
                ${marketData?.price?.toFixed(2) || '0.00'}
              </Typography>
              {marketData?.change !== undefined && (
                <Chip
                  icon={marketData.change >= 0 ? <TrendingUp /> : <TrendingDown />}
                  label={`${marketData.change >= 0 ? '+' : ''}${marketData.change.toFixed(2)} (${marketData.change >= 0 ? '+' : ''}${marketData.change_percent?.toFixed(2)}%)`}
                  size="small"
                  sx={{
                    backgroundColor: marketData.change >= 0 ? 'rgba(0, 212, 170, 0.2)' : 'rgba(255, 59, 48, 0.2)',
                    color: marketData.change >= 0 ? '#00D4AA' : '#FF3B30',
                    fontWeight: 600,
                    '& .MuiChip-icon': { color: 'inherit' },
                  }}
                />
              )}
            </Stack>
          </Grid>
        </Grid>

        {/* Time Period Selector */}
        <Grid container spacing={2} alignItems="center" sx={{ mb: 2 }}>
          <Grid item xs={12} sm={6}>
            <ButtonGroup size="small" sx={{ borderRadius: 2 }}>
              {timePeriods.map((period) => (
                <Button
                  key={period.value}
                  variant={selectedPeriod === period.value ? 'contained' : 'outlined'}
                  onClick={() => setSelectedPeriod(period.value)}
                  sx={{
                    minWidth: 40,
                    fontWeight: 500,
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    color: selectedPeriod === period.value ? 'white' : 'rgba(255, 255, 255, 0.7)',
                    ...(selectedPeriod === period.value && {
                      background: 'linear-gradient(135deg, #0066FF 0%, #0052CC 100%)',
                      borderColor: '#0066FF',
                    }),
                    '&:hover': {
                      background: selectedPeriod === period.value 
                        ? 'linear-gradient(135deg, #3D8BFF 0%, #0066FF 100%)'
                        : 'rgba(255, 255, 255, 0.05)',
                      borderColor: 'rgba(255, 255, 255, 0.3)',
                    },
                  }}
                >
                  {period.label}
                </Button>
              ))}
            </ButtonGroup>
          </Grid>
          <Grid item xs={12} sm={6} sx={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Stack direction="row" spacing={1}>
              <IconButton 
                size="small" 
                onClick={() => refetch()}
                sx={{ color: 'rgba(255, 255, 255, 0.7)' }}
              >
                <Refresh />
              </IconButton>
              <IconButton 
                size="small" 
                onClick={() => setIsFullscreen(!isFullscreen)}
                sx={{ color: 'rgba(255, 255, 255, 0.7)' }}
              >
                <Fullscreen />
              </IconButton>
            </Stack>
          </Grid>
        </Grid>

        <Divider sx={{ mb: 2, borderColor: 'rgba(255, 255, 255, 0.1)' }} />

        {/* Active Signals */}
        {signalsData && signalsData.signals.length > 0 && (
          <Paper 
            elevation={0} 
            sx={{ 
              p: 1, 
              mb: 2, 
              overflowX: 'auto', 
              background: 'transparent',
              '&::-webkit-scrollbar': { height: 6 },
              '&::-webkit-scrollbar-thumb': { background: 'rgba(255, 255, 255, 0.2)', borderRadius: 3 },
            }}
          >
            <Stack direction="row" spacing={1} sx={{ minWidth: 'max-content' }}>
              {signalsData.signals.map((signal) => (
                <Chip
                  key={`${signal.type}-${signal.time}`}
                  icon={
                    signal.type === 'ENTRY' ? <DollarSignIcon /> :
                    signal.type === 'EXIT' ? <ActivityIcon /> :
                    signal.type === 'STOP_LOSS' ? <ShieldAlertIcon /> :
                    <TargetIcon />
                  }
                  label={`${signal.type}: $${signal.price} (${signal.confidence}%)`}
                  size="small"
                  sx={{
                    backgroundColor: 
                      signal.type === 'ENTRY' ? 'rgba(0, 212, 170, 0.2)' :
                      signal.type === 'EXIT' ? 'rgba(59, 130, 246, 0.2)' :
                      signal.type === 'STOP_LOSS' ? 'rgba(255, 59, 48, 0.2)' :
                      'rgba(0, 212, 170, 0.2)',
                    color: 
                      signal.type === 'ENTRY' ? '#00D4AA' :
                      signal.type === 'EXIT' ? '#3B82F6' :
                      signal.type === 'STOP_LOSS' ? '#FF3B30' :
                      '#00D4AA',
                    fontWeight: 500,
                    '& .MuiChip-icon': { color: 'inherit' },
                  }}
                />
              ))}
            </Stack>
          </Paper>
        )}

        {/* Chart Container */}
        <Box 
          ref={chartContainerRef}
          sx={{ 
            flex: 1,
            minHeight: 400,
            maxHeight: '60vh',
            position: 'relative',
            borderRadius: 2,
            overflow: 'hidden',
          }}
        />
      </CardContent>
    </Card>
  );
} 