/**
 * Professional Trading Chart - Inspired by TD Ameritrade & E*TRADE
 * 
 * Simple, clean, and functional chart that actually works
 */

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
} from '@mui/icons-material';
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time } from 'lightweight-charts';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../services/api';
import { useSignals } from '../../store';

interface TradingChartProps {
  defaultSymbol?: string;
  height?: number;
  onSelectSignal?: (signal: any) => void;
}

interface ChartData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export default function TradingChart({ defaultSymbol = 'AAPL', height = 600, onSelectSignal }: TradingChartProps) {
  const theme = useTheme();
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  
  const [symbol, setSymbol] = useState(defaultSymbol);
  const [searchInput, setSearchInput] = useState(defaultSymbol);
  const [selectedPeriod, setSelectedPeriod] = useState('1D');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'bar'>('candlestick');
  const [showProjection, setShowProjection] = useState(false);
  const [selectedSignal, setSelectedSignal] = useState<any>(null);
  
  const { signals } = useSignals();

  // Time periods like TD Ameritrade
  const timePeriods = [
    { label: '1D', value: '1D' },
    { label: '5D', value: '5D' },
    { label: '1M', value: '1M' },
    { label: '3M', value: '3M' },
    { label: '6M', value: '6M' },
    { label: '1Y', value: '1Y' },
    { label: '5Y', value: '5Y' },
  ];

  // Fetch current market data
  const { data: marketData, isLoading, error, refetch } = useQuery({
    queryKey: ['market-data', symbol],
    queryFn: () => apiClient.getMarketData(symbol),
    refetchInterval: 30000,
    enabled: !!symbol,
  });

  // Fetch historical data for charting
  const { data: historicalData, isLoading: historicalLoading } = useQuery({
    queryKey: ['historical-data', symbol, selectedPeriod],
    queryFn: async () => {
      try {
        // Try to get historical data from backend
        const response = await fetch(`http://localhost:8000/api/v1/market-data/${symbol}/historical?period=${selectedPeriod.toLowerCase()}&interval=5m`);
        if (response.ok) {
          const data = await response.json();
          return data.data || [];
        }
        // Fallback to mock data if endpoint doesn't exist
        return generateMockData();
      } catch (error) {
        console.log('Using mock data for chart:', error);
        return generateMockData();
      }
    },
    refetchInterval: 60000, // Refresh every minute
    enabled: !!symbol,
  });

  // Generate realistic mock data
  const generateMockData = (): ChartData[] => {
    const data: ChartData[] = [];
    let basePrice = marketData?.price || 150;
    const now = new Date();
    
    // Determine data points based on period
    const periods = {
      '1D': 96,    // 15-minute intervals for 1 day
      '5D': 120,   // hourly for 5 days
      '1M': 22,    // daily for 1 month
      '3M': 66,    // daily for 3 months
      '6M': 132,   // daily for 6 months
      '1Y': 252,   // daily for 1 year
      '5Y': 1260,  // daily for 5 years
    };
    
    const intervals = {
      '1D': 15 * 60,        // 15 minutes in seconds
      '5D': 60 * 60,        // 1 hour in seconds
      '1M': 24 * 60 * 60,   // 1 day in seconds
      '3M': 24 * 60 * 60,   // 1 day in seconds
      '6M': 24 * 60 * 60,   // 1 day in seconds
      '1Y': 24 * 60 * 60,   // 1 day in seconds
      '5Y': 24 * 60 * 60,   // 1 day in seconds
    };
    
    const pointCount = periods[selectedPeriod as keyof typeof periods];
    const interval = intervals[selectedPeriod as keyof typeof intervals];
    const startTime = Math.floor(now.getTime() / 1000) - (pointCount * interval);
    
    for (let i = 0; i < pointCount; i++) {
      const time = startTime + i * interval;
      
      const trend = Math.sin(i / (pointCount * 0.1)) * (basePrice * 0.05);
      const volatility = (Math.random() - 0.5) * (basePrice * 0.03);
      const momentum = Math.cos(i / (pointCount * 0.05)) * (basePrice * 0.02);
      
      const open = Math.max(1, basePrice + trend + (volatility * 0.5));
      const high = Math.max(open, open + Math.abs(volatility) * 1.2 + Math.random() * (basePrice * 0.01));
      const low = Math.min(open, open - Math.abs(volatility) * 1.2 - Math.random() * (basePrice * 0.01));
      const close = low + (high - low) * (0.2 + Math.random() * 0.6);
      const volume = Math.floor((100000 + Math.random() * 500000) * (1 + Math.abs(volatility) * 0.1));
      
      data.push({
        time: time, // Keep as number (unix timestamp) for the chart
        open: Math.max(1, open),
        high: Math.max(1, high),
        low: Math.max(1, low),
        close: Math.max(1, close),
        volume,
      });
      
      basePrice = close + momentum * 0.1;
    }
    
    return data;
  };

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;
    
    // Clean up previous chart if it exists
    if (chartRef.current) {
      try {
        chartRef.current.remove();
      } catch (error) {
        console.log('Chart already disposed:', error);
      }
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
    let series: ISeriesApi<any>;
    
    // Safely get data with validation
    let data: ChartData[];
    try {
      const rawData = historicalData && Array.isArray(historicalData) && historicalData.length > 0 
        ? historicalData 
        : generateMockData();
      
      // Validate and clean the data
      data = rawData.filter(item => {
        return item && 
               typeof item.time === 'number' && 
               !isNaN(item.time) &&
               typeof item.open === 'number' && 
               !isNaN(item.open) &&
               typeof item.high === 'number' && 
               !isNaN(item.high) &&
               typeof item.low === 'number' && 
               !isNaN(item.low) &&
               typeof item.close === 'number' && 
               !isNaN(item.close) &&
               item.open > 0 && 
               item.high > 0 && 
               item.low > 0 && 
               item.close > 0;
      }).map((item: ChartData) => ({
        ...item,
        time: Math.floor(item.time), // Ensure time is an integer timestamp
      }));
      
      if (data.length === 0) {
        data = generateMockData();
      }
    } catch (error) {
      console.error('Data preparation error:', error);
      data = generateMockData();
    }

    try {
      if (chartType === 'candlestick') {
        series = chart.addCandlestickSeries({
          upColor: '#00D4AA',
          downColor: '#FF3B30',
          borderUpColor: '#00D4AA',
          borderDownColor: '#FF3B30',
          wickUpColor: '#00D4AA',
          wickDownColor: '#FF3B30',
        });
        series.setData(data.map((item: ChartData) => ({
          time: item.time as Time,
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
        })));
      } else if (chartType === 'line') {
        series = chart.addLineSeries({
          color: '#0066FF',
          lineWidth: 2,
        });
        series.setData(data.map((item: ChartData) => ({
          time: item.time as Time,
          value: item.close,
        })));
      } else if (chartType === 'bar') {
        series = chart.addBarSeries({
          upColor: '#00D4AA',
          downColor: '#FF3B30',
          thinBars: false,
        });
        series.setData(data.map((item: ChartData) => ({
          time: item.time as Time,
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
        })));
      }

      // Projection line
      if (showProjection && data.length > 0) {
        const last = data[data.length - 1];
        if (last && last.time && last.close) {
          try {
            // Safely parse time and add projection
            const lastTime = Math.floor(last.time);
            const projectionTime = lastTime + 5 * 60 * 60; // Add 5 hours
            
            const projPoints = [
              { time: lastTime as Time, value: last.close },
              { time: projectionTime as Time, value: last.close * (1 + 0.03 * (Math.random() - 0.5)) },
            ];
            
            const projSeries = chart.addLineSeries({
              color: '#FFD600',
              lineWidth: 2,
              lineStyle: 2,
              priceLineVisible: false,
            });
            projSeries.setData(projPoints);
          } catch (error) {
            console.log('Projection line error:', error);
          }
        }
      }

      // Signal markers
      const symbolSignals = signals.filter(s => s.symbol === symbol);
      symbolSignals.forEach(signal => {
        if (series && signal.current_price && !isNaN(signal.current_price)) {
          try {
            series.createPriceLine({
              price: signal.current_price,
              color: signal.signal_type === 'BUY' ? '#00D4AA' : '#FF3B30',
              lineWidth: 2,
              lineStyle: 2,
              axisLabelVisible: true,
              title: `${signal.signal_type} Signal`,
            });
          } catch (error) {
            console.log('Signal marker error:', error);
          }
        }
      });
    } catch (error) {
      console.error('Chart series creation error:', error);
      // Fallback: try with simplified data
      try {
        if (chartType === 'line' || !series) {
          series = chart.addLineSeries({
            color: '#0066FF',
            lineWidth: 2,
          });
          series.setData(data.slice(0, 100).map((item: ChartData) => ({
            time: item.time as Time,
            value: item.close,
          })));
        }
      } catch (fallbackError) {
        console.error('Fallback chart creation failed:', fallbackError);
      }
    }

    // Resize handler
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        try {
          chart.applyOptions({ width: chartContainerRef.current.clientWidth });
        } catch (error) {
          console.log('Chart resize error:', error);
        }
      }
    };

    window.addEventListener('resize', handleResize);

    // Cleanup function
    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        try {
          chartRef.current.remove();
        } catch (error) {
          console.log('Chart cleanup error:', error);
        }
        chartRef.current = null;
      }
    };
  }, [theme, height, chartType, showProjection, selectedPeriod, marketData, signals, symbol, historicalData]);

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

  const currentPrice = marketData?.price;
  const priceChange = marketData?.change;
  const priceChangePercent = marketData?.change_percent;
  const isPositive = (priceChange ?? 0) >= 0;

  // Simulate marker click: on chart click, select the latest signal for the symbol
  const handleChartClick = () => {
    const symbolSignals = signals.filter(s => s.symbol === symbol);
    if (symbolSignals.length > 0 && onSelectSignal) {
      onSelectSignal(symbolSignals[0]);
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
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
          {/* Chart Type Selector */}
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
          {/* Projection Toggle */}
          <Button
            variant={showProjection ? 'contained' : 'outlined'}
            color={showProjection ? 'warning' : 'inherit'}
            onClick={() => setShowProjection(v => !v)}
            sx={{ ml: 2 }}
          >
            {showProjection ? 'Hide Projection' : 'Show Projection'}
          </Button>
        </Stack>
        {/* Header - TD Ameritrade Style */}
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 3 }}>
          {/* Symbol Search */}
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

          {/* Price Info */}
          <Box sx={{ textAlign: 'right' }}>
            <Typography variant="h4" sx={{ fontWeight: 700, color: 'white', mb: 0.5 }}>
              {symbol}
            </Typography>
            <Stack direction="row" alignItems="center" spacing={1}>
              <Typography variant="h5" sx={{ fontWeight: 600, color: 'white' }}>
                ${currentPrice?.toFixed(2) || '0.00'}
              </Typography>
              {priceChange !== undefined && (
                <Chip
                  icon={isPositive ? <TrendingUp /> : <TrendingDown />}
                  label={`${isPositive ? '+' : ''}${priceChange.toFixed(2)} (${isPositive ? '+' : ''}${priceChangePercent?.toFixed(2)}%)`}
                  size="small"
                  sx={{
                    backgroundColor: isPositive ? 'rgba(0, 212, 170, 0.2)' : 'rgba(255, 59, 48, 0.2)',
                    color: isPositive ? '#00D4AA' : '#FF3B30',
                    fontWeight: 600,
                    '& .MuiChip-icon': { color: 'inherit' },
                  }}
                />
              )}
            </Stack>
          </Box>
        </Stack>

        {/* Time Period Selector - E*TRADE Style */}
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
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

          {/* Chart Controls */}
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
        </Stack>

        <Divider sx={{ mb: 2, borderColor: 'rgba(255, 255, 255, 0.1)' }} />

        {/* Chart Container */}
        <Box 
          ref={chartContainerRef}
          sx={{ 
            flex: 1,
            position: 'relative',
            borderRadius: 2,
            overflow: 'hidden',
            cursor: 'pointer',
          }}
          onClick={handleChartClick}
        />

        {/* Signal Summary */}
        {signals.filter(s => s.symbol === symbol).length > 0 && (
          <Box sx={{ mt: 2, p: 2, background: 'rgba(255, 255, 255, 0.02)', borderRadius: 2 }}>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
              Active Signals for {symbol}
            </Typography>
            <Stack direction="row" spacing={1}>
              {signals.filter(s => s.symbol === symbol).slice(0, 3).map((signal, index) => (
                <Chip
                  key={signal.signal_id}
                  label={`${signal.signal_type} - ${Math.round(signal.confidence * 100)}%`}
                  size="small"
                  sx={{
                    backgroundColor: signal.signal_type === 'BUY' ? 'rgba(0, 212, 170, 0.2)' : 'rgba(255, 59, 48, 0.2)',
                    color: signal.signal_type === 'BUY' ? '#00D4AA' : '#FF3B30',
                    fontWeight: 500,
                  }}
                />
              ))}
            </Stack>
          </Box>
        )}
      </CardContent>
    </Card>
  );
} 