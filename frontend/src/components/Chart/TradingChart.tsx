/**
 * Professional Trading Chart with AI Insights
 * 
 * Advanced charting component with AI-driven analysis overlay
 */

import React, { useEffect, useRef, useState, useMemo, useCallback, memo } from 'react';
import {
  Box,
  IconButton,
  Chip,
  Typography,
  Stack,
  Button,
  useTheme,
  CircularProgress,
  ButtonGroup,
  Tooltip,
  TextField,
  InputAdornment,
  Autocomplete,
  Skeleton,
  Divider,
  ToggleButton,
  ToggleButtonGroup,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Switch,
  FormControlLabel,
  alpha,
  Select,
  FormControl,
  SelectChangeEvent,
  Fade,
  Zoom,
  Paper,
  ClickAwayListener,
  Popper,
  Grow,
  Badge,
  LinearProgress,
  Alert,
  Collapse,
  Card,
  InputBase,
} from '@mui/material';
import {
  Fullscreen,
  FullscreenExit,
  Refresh,
  MoreVert,
  TrendingUp,
  TrendingDown,
  ShowChart,
  CandlestickChart,
  Timeline,
  Insights,
  Search,
  ExpandMore,
  CameraAlt,
  RestartAlt,
  AccessTime,
  Speed,
  Pattern as PatternIcon,
  ShowChart as AreaChart,
  PlayArrow as PlayIcon,
  AutoAwesome as AIIcon,
  Close as CloseIcon,
  Analytics,
  Assessment,
} from '@mui/icons-material';
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time, LineStyle, CrosshairMode, HistogramData, SeriesMarker } from 'lightweight-charts';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../services/api';
import { useSignals } from '../../store';
import { fetchMarketData, fetchAIInsights } from '../../services/api';
import { useHotkeys } from 'react-hotkeys-hook';
import { PreciseOptionsSignal } from '../../types/signals';
import { motion, AnimatePresence } from 'framer-motion';
import { debounce } from 'lodash';
import html2canvas from 'html2canvas';

// Professional Chart Theme Configuration
const CHART_THEMES = {
  dark: {
    background: '#0A0A0A',
    text: '#E2E8F0',
    grid: 'rgba(255, 255, 255, 0.04)',
    crosshair: '#2563EB',
    upColor: '#10B981',
    downColor: '#EF4444',
    volumeUp: 'rgba(16, 185, 129, 0.2)',
    volumeDown: 'rgba(239, 68, 68, 0.2)',
    border: 'rgba(255, 255, 255, 0.1)',
  },
  light: {
    background: '#FFFFFF',
    text: '#1F2937',
    grid: 'rgba(0, 0, 0, 0.06)',
    crosshair: '#3B82F6',
    upColor: '#059669',
    downColor: '#DC2626',
    volumeUp: 'rgba(5, 150, 105, 0.2)',
    volumeDown: 'rgba(220, 38, 38, 0.2)',
    border: 'rgba(0, 0, 0, 0.1)',
  },
};

interface TradingChartProps {
  defaultSymbol?: string;
  height?: number;
  showAIInsights?: boolean;
  onSelectSignal?: (signal: any) => void;
  onSymbolChange?: (symbol: string) => void;
  theme?: 'dark' | 'light';
  enableDrawingTools?: boolean;
  enableAlerts?: boolean;
}

interface ChartData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface AIInsight {
  price: number;
  type: 'SUPPORT' | 'RESISTANCE' | 'TARGET' | 'ENTRY' | 'STOP';
  confidence: number;
  label: string;
  strength?: number;
}

interface Pattern {
  type: string;
  confidence: number;
  points: { time: number; price: number }[];
}

interface Divergence {
  startTime: number;
  endTime: number;
  startPrice: number;
  endPrice: number;
  type: 'bullish' | 'bearish';
  indicator: string;
}

interface TimeframeOption {
  value: string;
  label: string;
  shortLabel: string;
  interval: string;
  dataPoints: number;
  description: string;
  category: 'minutes' | 'hours' | 'days' | 'weeks' | 'months' | 'years';
}

interface PriceAlert {
  id: string;
  symbol: string;
  price: number;
  type: 'above' | 'below';
  active: boolean;
  triggered: boolean;
}

// Enhanced timeframe options with standard intervals organized by category
const TIMEFRAME_OPTIONS: TimeframeOption[] = [
  // Minutes
  { value: '1m', label: '1 Minute', shortLabel: '1m', interval: '1m', dataPoints: 60, description: '1-minute bars', category: 'minutes' },
  { value: '5m', label: '5 Minutes', shortLabel: '5m', interval: '1m', dataPoints: 300, description: '1-minute bars', category: 'minutes' },
  { value: '15m', label: '15 Minutes', shortLabel: '15m', interval: '1m', dataPoints: 900, description: '1-minute bars', category: 'minutes' },
  { value: '30m', label: '30 Minutes', shortLabel: '30m', interval: '5m', dataPoints: 360, description: '5-minute bars', category: 'minutes' },

  // Hours
  { value: '1h', label: '1 Hour', shortLabel: '1h', interval: '5m', dataPoints: 720, description: '5-minute bars', category: 'hours' },
  { value: '2h', label: '2 Hours', shortLabel: '2h', interval: '5m', dataPoints: 1440, description: '5-minute bars', category: 'hours' },
  { value: '4h', label: '4 Hours', shortLabel: '4h', interval: '15m', dataPoints: 960, description: '15-minute bars', category: 'hours' },
  { value: '6h', label: '6 Hours', shortLabel: '6h', interval: '15m', dataPoints: 1440, description: '15-minute bars', category: 'hours' },
  { value: '12h', label: '12 Hours', shortLabel: '12h', interval: '30m', dataPoints: 1440, description: '30-minute bars', category: 'hours' },

  // Days
  { value: '1D', label: '1 Day', shortLabel: '1D', interval: '5m', dataPoints: 390, description: '5-minute bars', category: 'days' },
  { value: '2D', label: '2 Days', shortLabel: '2D', interval: '10m', dataPoints: 390, description: '10-minute bars', category: 'days' },
  { value: '3D', label: '3 Days', shortLabel: '3D', interval: '15m', dataPoints: 390, description: '15-minute bars', category: 'days' },
  { value: '5D', label: '5 Days', shortLabel: '5D', interval: '30m', dataPoints: 390, description: '30-minute bars', category: 'days' },
  { value: '10D', label: '10 Days', shortLabel: '10D', interval: '1h', dataPoints: 390, description: 'Hourly bars', category: 'days' },

  // Weeks
  { value: '1W', label: '1 Week', shortLabel: '1W', interval: '1h', dataPoints: 168, description: 'Hourly bars', category: 'weeks' },
  { value: '2W', label: '2 Weeks', shortLabel: '2W', interval: '2h', dataPoints: 168, description: '2-hour bars', category: 'weeks' },
  { value: '3W', label: '3 Weeks', shortLabel: '3W', interval: '4h', dataPoints: 126, description: '4-hour bars', category: 'weeks' },
  { value: '4W', label: '4 Weeks', shortLabel: '4W', interval: '4h', dataPoints: 168, description: '4-hour bars', category: 'weeks' },

  // Months
  { value: '1M', label: '1 Month', shortLabel: '1M', interval: '4h', dataPoints: 180, description: '4-hour bars', category: 'months' },
  { value: '2M', label: '2 Months', shortLabel: '2M', interval: '1d', dataPoints: 60, description: 'Daily bars', category: 'months' },
  { value: '3M', label: '3 Months', shortLabel: '3M', interval: '1d', dataPoints: 90, description: 'Daily bars', category: 'months' },
  { value: '6M', label: '6 Months', shortLabel: '6M', interval: '1d', dataPoints: 180, description: 'Daily bars', category: 'months' },
  { value: '9M', label: '9 Months', shortLabel: '9M', interval: '1d', dataPoints: 270, description: 'Daily bars', category: 'months' },

  // Years
  { value: '1Y', label: '1 Year', shortLabel: '1Y', interval: '1d', dataPoints: 252, description: 'Daily bars', category: 'years' },
  { value: '2Y', label: '2 Years', shortLabel: '2Y', interval: '1w', dataPoints: 104, description: 'Weekly bars', category: 'years' },
  { value: '3Y', label: '3 Years', shortLabel: '3Y', interval: '1w', dataPoints: 156, description: 'Weekly bars', category: 'years' },
  { value: '5Y', label: '5 Years', shortLabel: '5Y', interval: '1w', dataPoints: 260, description: 'Weekly bars', category: 'years' },
  { value: 'YTD', label: 'Year to Date', shortLabel: 'YTD', interval: '1d', dataPoints: 0, description: 'Daily bars', category: 'years' },
  { value: 'ALL', label: 'All Time', shortLabel: 'ALL', interval: '1M', dataPoints: 0, description: 'Monthly bars', category: 'years' },
];

const TradingChart = memo(({
  defaultSymbol = 'AAPL',
  height = 600,
  showAIInsights = true,
  onSelectSignal,
  onSymbolChange,
  theme: chartTheme = 'dark',
  enableDrawingTools = false,
  enableAlerts = false,
}: TradingChartProps) => {
  const theme = useTheme();
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  // Chart state
  const symbol = defaultSymbol; // Use symbol from props
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'area'>('candlestick');
  const [selectedTimeframe, setSelectedTimeframe] = useState(TIMEFRAME_OPTIONS[4]); // 1D default
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showExtendedHours, setShowExtendedHours] = useState(false);
  const [hoveredPrice, setHoveredPrice] = useState<number | null>(null);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [showOrderBook, setShowOrderBook] = useState(false);

  // AI Analysis state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [detectedPatterns, setDetectedPatterns] = useState<Pattern[]>([]);
  const [aiInsights, setAIInsights] = useState<AIInsight[]>([]);
  const [showAIPanel, setShowAIPanel] = useState(false);

  const [showAlertDialog, setShowAlertDialog] = useState(false);
  const [showVolumeProfile, setShowVolumeProfile] = useState(false);
  const [alerts, setAlerts] = useState<PriceAlert[]>([]);
  const [indicatorsMenuAnchor, setIndicatorsMenuAnchor] = useState<null | HTMLElement>(null);

  // Indicators state - all indicators available
  const [indicators, setIndicators] = useState({
    // Moving Averages
    ma: true,
    ema: true,

    // Bands & Channels
    bollinger: true,

    // Volume
    volume: true,

    // Oscillators (shown in separate panel)
    rsi: false,
    macd: false,

    // Support/Resistance
    supportResistance: true,

    // Fibonacci
    fibonacci: true,

    // AI Features
    predictions: false,
    trendPrediction: false,
    patterns: true,
    divergences: false,
  });

  const { signals } = useSignals();

  // Fetch current market data
  const { data: marketData, isLoading, error, refetch } = useQuery({
    queryKey: ['market-data', symbol],
    queryFn: () => apiClient.getMarketData(symbol),
    refetchInterval: 5000, // Update every 5 seconds for real-time feel
    enabled: !!symbol,
  });

  // Fetch historical data for charting
  const { data: historicalData, isLoading: historicalLoading } = useQuery({
    queryKey: ['historical-data', symbol, selectedTimeframe.value, selectedTimeframe.interval],
    queryFn: async () => {
      try {
        // Try to get historical data from backend
        const response = await fetch(
          `http://localhost:8000/api/v1/market-data/${symbol}/historical?period=${selectedTimeframe.value.toLowerCase()}&interval=${selectedTimeframe.interval}`
        );
        if (response.ok) {
          const data = await response.json();
          return data;
        }
      } catch (error) {
        console.error('Error fetching historical data:', error);
      }
      // Fallback to mock data if backend fails
      return generateMockData();
    },
    refetchInterval: selectedTimeframe.interval === '5m' ? 60000 : 300000, // Refresh more frequently for intraday
    enabled: !!symbol,
  });

  // Fetch AI insights
  const { data: aiInsightsData } = useQuery({
    queryKey: ['ai-insights', symbol],
    queryFn: () => apiClient.getAIInsights(symbol),
    refetchInterval: 60000,
    enabled: showAIInsights,
  });

  // Update chart with real-time data
  useEffect(() => {
    if (marketData && candlestickSeriesRef.current && historicalData && historicalData.length > 0) {
      try {
        const lastCandle = historicalData[historicalData.length - 1];
        const currentTime = lastCandle.time;

        // Create updated candle with real-time price
        const updatedCandle: CandlestickData = {
          time: currentTime as Time,
          open: lastCandle.open,
          high: Math.max(lastCandle.high, marketData.price),
          low: Math.min(lastCandle.low, marketData.price),
          close: marketData.price,
        };

        // Update the series
        candlestickSeriesRef.current.update(updatedCandle);

        // Update volume if available
        if (volumeSeriesRef.current && marketData.volume) {
          const volumeData: HistogramData = {
            time: currentTime as Time,
            value: marketData.volume,
            color: marketData.price >= lastCandle.open ? '#26a69a' : '#ef5350',
          };
          volumeSeriesRef.current.update(volumeData);
        }

        // Update price line
        candlestickSeriesRef.current.createPriceLine({
          price: marketData.price,
          color: marketData.change >= 0 ? '#26a69a' : '#ef5350',
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: 'Current',
        });
      } catch (error) {
        console.error('Error updating real-time data:', error);
      }
    }
  }, [marketData, historicalData]);

  // Symbol is now controlled by parent component

  // Handle timeframe change
  const handleTimeframeChange = (event: SelectChangeEvent<string>) => {
    const newTimeframe = TIMEFRAME_OPTIONS.find(tf => tf.value === event.target.value);
    if (newTimeframe) {
      setSelectedTimeframe(newTimeframe);
    }
  };

  // Keyboard shortcuts
  useHotkeys('1', () => setSelectedTimeframe(TIMEFRAME_OPTIONS[0]));
  useHotkeys('2', () => setSelectedTimeframe(TIMEFRAME_OPTIONS[1]));
  useHotkeys('3', () => setSelectedTimeframe(TIMEFRAME_OPTIONS[2]));
  useHotkeys('4', () => setSelectedTimeframe(TIMEFRAME_OPTIONS[3]));
  useHotkeys('r', () => refetch());
  useHotkeys('f', () => setIsFullscreen(!isFullscreen));
  useHotkeys('cmd+z, ctrl+z', () => handleResetZoom());
  useHotkeys('cmd+plus, ctrl+plus', () => handleZoomIn());
  useHotkeys('cmd+minus, ctrl+minus', () => handleZoomOut());
  useHotkeys('cmd+s, ctrl+s', (e) => { e.preventDefault(); handleScreenshot(); });

  // Calculate technical indicators
  const calculateSMA = (data: ChartData[], period: number): { time: Time; value: number }[] => {
    const sma = [];
    for (let i = period - 1; i < data.length; i++) {
      const sum = data.slice(i - period + 1, i + 1).reduce((acc, d) => acc + d.close, 0);
      sma.push({ time: data[i].time as Time, value: sum / period });
    }
    return sma;
  };

  const calculateEMA = (data: ChartData[], period: number): { time: Time; value: number }[] => {
    const ema = [];
    const multiplier = 2 / (period + 1);
    let previousEMA = data[0].close;

    for (let i = 0; i < data.length; i++) {
      const currentEMA = (data[i].close - previousEMA) * multiplier + previousEMA;
      ema.push({ time: data[i].time as Time, value: currentEMA });
      previousEMA = currentEMA;
    }
    return ema;
  };

  const calculateBollingerBands = (data: ChartData[], period: number = 20, stdDev: number = 2) => {
    const sma = calculateSMA(data, period);
    const upper = [];
    const lower = [];

    for (let i = period - 1; i < data.length; i++) {
      const slice = data.slice(i - period + 1, i + 1);
      const avg = sma[i - period + 1].value;
      const variance = slice.reduce((sum, d) => sum + Math.pow(d.close - avg, 2), 0) / period;
      const std = Math.sqrt(variance);

      upper.push({ time: data[i].time as Time, value: avg + stdDev * std });
      lower.push({ time: data[i].time as Time, value: avg - stdDev * std });
    }

    return { upper, lower, middle: sma };
  };

  const calculateRSI = (data: ChartData[], period: number = 14): { time: Time; value: number }[] => {
    const rsi = [];
    let gains = 0;
    let losses = 0;

    // Calculate initial average gain/loss
    for (let i = 1; i <= period; i++) {
      const change = data[i].close - data[i - 1].close;
      if (change > 0) gains += change;
      else losses -= change;
    }

    let avgGain = gains / period;
    let avgLoss = losses / period;

    for (let i = period; i < data.length; i++) {
      const change = data[i].close - data[i - 1].close;
      avgGain = (avgGain * (period - 1) + (change > 0 ? change : 0)) / period;
      avgLoss = (avgLoss * (period - 1) + (change < 0 ? -change : 0)) / period;

      const rs = avgGain / avgLoss;
      const rsiValue = 100 - (100 / (1 + rs));

      rsi.push({ time: data[i].time as Time, value: rsiValue });
    }

    return rsi;
  };

  // Generate realistic mock data
  const generateMockData = (): ChartData[] => {
    const data: ChartData[] = [];
    let basePrice = 150; // Default base price
    const now = new Date();

    // Calculate data points for YTD
    if (selectedTimeframe.value === 'YTD') {
      const yearStart = new Date(now.getFullYear(), 0, 1);
      const daysSinceYearStart = Math.floor((now.getTime() - yearStart.getTime()) / (1000 * 60 * 60 * 24));
      selectedTimeframe.dataPoints = daysSinceYearStart;
    }

    const pointCount = selectedTimeframe.dataPoints || 100;
    const intervalMap: { [key: string]: number } = {
      '1m': 60,
      '5m': 5 * 60,
      '10m': 10 * 60,
      '15m': 15 * 60,
      '30m': 30 * 60,
      '1h': 60 * 60,
      '2h': 2 * 60 * 60,
      '4h': 4 * 60 * 60,
      '6h': 6 * 60 * 60,
      '12h': 12 * 60 * 60,
      '1d': 24 * 60 * 60,
      '1w': 7 * 24 * 60 * 60,
      '1M': 30 * 24 * 60 * 60,
    };

    const interval = intervalMap[selectedTimeframe.interval] || 24 * 60 * 60;
    const startTime = Math.floor(now.getTime() / 1000) - (pointCount * interval);

    // Volatility based on timeframe - shorter timeframes have higher volatility
    const volatilityMultiplier = {
      '1m': 0.001,
      '5m': 0.002,
      '10m': 0.0025,
      '15m': 0.003,
      '30m': 0.004,
      '1h': 0.005,
      '2h': 0.006,
      '4h': 0.008,
      '6h': 0.010,
      '12h': 0.012,
      '1d': 0.015,
      '1w': 0.025,
      '1M': 0.035,
    }[selectedTimeframe.interval] || 0.015;

    // Generate more realistic price movements
    let trendDirection = Math.random() > 0.5 ? 1 : -1;
    let trendStrength = 0.001 + Math.random() * 0.002;
    let consecutiveSameDirection = 0;

    for (let i = 0; i < pointCount; i++) {
      const time = startTime + i * interval;

      // Change trend occasionally
      if (Math.random() < 0.1 || consecutiveSameDirection > 5) {
        trendDirection *= -1;
        consecutiveSameDirection = 0;
        trendStrength = 0.001 + Math.random() * 0.002;
      }

      // Base movement with trend
      const trendComponent = trendDirection * trendStrength * basePrice;

      // Random walk component
      const randomWalk = (Math.random() - 0.5) * basePrice * volatilityMultiplier * 2;

      // Momentum from previous candles
      const momentum = i > 0 ? (data[i - 1].close - data[i - 1].open) * 0.3 : 0;

      // Calculate OHLC with realistic relationships
      const open = i > 0 ? data[i - 1].close : basePrice;

      // Body size varies but is realistic
      const bodySize = Math.abs(randomWalk + trendComponent + momentum);
      const wickSize = bodySize * (0.5 + Math.random() * 1.5);

      // Determine if bullish or bearish candle
      const isBullish = (randomWalk + trendComponent + momentum) > 0;

      let high, low, close;

      if (isBullish) {
        close = open + bodySize;
        high = Math.max(open, close) + wickSize * Math.random();
        low = Math.min(open, close) - wickSize * Math.random() * 0.5;
      } else {
        close = open - bodySize;
        high = Math.max(open, close) + wickSize * Math.random() * 0.5;
        low = Math.min(open, close) - wickSize * Math.random();
      }

      // Add some doji candles (small body)
      if (Math.random() < 0.1) {
        close = open + (Math.random() - 0.5) * bodySize * 0.1;
        high = Math.max(open, close) + wickSize;
        low = Math.min(open, close) - wickSize;
      }

      // Volume correlates with price movement
      const priceMovement = Math.abs(close - open) / open;
      const baseVolume = 100000 + Math.random() * 400000;
      const volume = Math.floor(baseVolume * (1 + priceMovement * 50));

      data.push({
        time: time,
        open: Math.max(1, open),
        high: Math.max(1, high),
        low: Math.max(1, low),
        close: Math.max(1, close),
        volume,
      });

      basePrice = close;
      consecutiveSameDirection++;
    }

    return data;
  };

  // Format time based on timeframe
  const getTimeFormatter = (timeframe: string) => {
    return (time: number) => {
      const date = new Date(time * 1000);
      const selectedTf = TIMEFRAME_OPTIONS.find(tf => tf.value === timeframe);

      if (!selectedTf) {
        return date.toLocaleDateString();
      }

      switch (selectedTf.category) {
        case 'minutes':
          // For minute timeframes, show time with seconds
          return date.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
            second: selectedTf.value === '1m' ? '2-digit' : undefined
          });

        case 'hours':
          // For hour timeframes, show date and time
          if (selectedTf.value === '1h' || selectedTf.value === '2h') {
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
          }
          return date.toLocaleDateString([], {
            month: 'short',
            day: 'numeric',
            hour: '2-digit'
          });

        case 'days':
          // For day timeframes, show date
          if (selectedTf.value === '1D') {
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
          }
          return date.toLocaleDateString([], {
            month: 'short',
            day: 'numeric'
          });

        case 'weeks':
          // For week timeframes, show date
          return date.toLocaleDateString([], {
            month: 'short',
            day: 'numeric'
          });

        case 'months':
          // For month timeframes, show month and year
          if (selectedTf.value === '1M' || selectedTf.value === '2M') {
            return date.toLocaleDateString([], {
              month: 'short',
              day: 'numeric'
            });
          }
          return date.toLocaleDateString([], {
            month: 'short',
            year: '2-digit'
          });

        case 'years':
          // For year timeframes, show year
          if (selectedTf.value === '1Y' || selectedTf.value === 'YTD') {
            return date.toLocaleDateString([], {
              month: 'short',
              year: '2-digit'
            });
          }
          return date.toLocaleDateString([], {
            year: 'numeric'
          });

        default:
          return date.toLocaleDateString();
      }
    };
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

    const chartColors = CHART_THEMES[chartTheme];

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: height - 80,
      layout: {
        background: { color: chartColors.background },
        textColor: chartColors.text,
        fontSize: 11,
        fontFamily: '"SF Pro Display", "Inter", -apple-system, BlinkMacSystemFont, sans-serif',
      },
      grid: {
        vertLines: {
          color: chartColors.grid,
          style: LineStyle.Dotted,
        },
        horzLines: {
          color: chartColors.grid,
          style: LineStyle.Dotted,
        },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: chartColors.crosshair,
          width: 1,
          style: LineStyle.Solid,
          labelBackgroundColor: chartColors.crosshair,
          labelVisible: true,
        },
        horzLine: {
          color: chartColors.crosshair,
          width: 1,
          style: LineStyle.Solid,
          labelBackgroundColor: chartColors.crosshair,
          labelVisible: true,
        },
      },
      rightPriceScale: {
        borderColor: chartColors.grid,
        textColor: chartColors.text,
        scaleMargins: {
          top: 0.1,
          bottom: indicators.volume ? 0.25 : 0.1,
        },
        autoScale: true,
        alignLabels: true,
        borderVisible: false,
        entireTextOnly: true,
      },
      timeScale: {
        borderColor: chartColors.border,
        borderVisible: false,
        rightOffset: 12,
        barSpacing: 8,
        minBarSpacing: 4,
        fixLeftEdge: false,
        fixRightEdge: false,
        lockVisibleTimeRangeOnResize: true,
        rightBarStaysOnScroll: true,
        borderVisible: false,
        visible: true,
        timeVisible: true,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: {
        axisPressedMouseMove: {
          time: true,
          price: true,
        },
        axisDoubleClickReset: true,
        mouseWheel: true,
        pinch: true,
      },
      kineticScroll: {
        mouse: true,
        touch: true,
      },
    });

    chartRef.current = chart;
    let series: ISeriesApi<any> | null = null;

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
          upColor: chartColors.upColor,
          downColor: chartColors.downColor,
          borderUpColor: chartColors.upColor,
          borderDownColor: chartColors.downColor,
          wickUpColor: chartColors.upColor,
          wickDownColor: chartColors.downColor,
          priceLineVisible: true,
          priceLineWidth: 1,
          priceLineColor: chartColors.text,
          priceLineStyle: LineStyle.Dashed,
          lastValueVisible: true,
          priceFormat: {
            type: 'price',
            precision: 2,
            minMove: 0.01,
          },
        });
        candlestickSeriesRef.current = series as ISeriesApi<'Candlestick'>;
        series.setData(data);
      } else if (chartType === 'line') {
        series = chart.addLineSeries({
          color: chartColors.crosshair,
          lineWidth: 2,
          lineStyle: LineStyle.Solid,
          crosshairMarkerVisible: true,
          crosshairMarkerRadius: 5,
          crosshairMarkerBorderColor: chartColors.crosshair,
          crosshairMarkerBackgroundColor: chartColors.background,
          lastValueVisible: true,
          priceLineVisible: true,
          priceLineWidth: 1,
          priceLineColor: chartColors.crosshair,
          priceLineStyle: LineStyle.Dashed,
        });
        series.setData(data.map((item: ChartData) => ({
          time: item.time as Time,
          value: item.close,
        })));
      } else if (chartType === 'area') {
        series = chart.addAreaSeries({
          lineColor: chartColors.crosshair,
          topColor: alpha(chartColors.crosshair, 0.4),
          bottomColor: alpha(chartColors.crosshair, 0.05),
          lineWidth: 2,
          lineStyle: LineStyle.Solid,
          crosshairMarkerVisible: true,
          crosshairMarkerRadius: 5,
          crosshairMarkerBorderColor: chartColors.crosshair,
          crosshairMarkerBackgroundColor: chartColors.background,
          lastValueVisible: true,
          priceLineVisible: true,
          priceLineWidth: 1,
          priceLineColor: chartColors.crosshair,
          priceLineStyle: LineStyle.Dashed,
        });
        series.setData(data.map((item: ChartData) => ({
          time: item.time as Time,
          value: item.close,
        })));
      }

      // Add volume series with professional styling
      if (indicators.volume) {
        const volumeSeries = chart.addHistogramSeries({
          color: chartColors.volumeUp,
          priceFormat: {
            type: 'volume',
          },
          priceScaleId: '',
          scaleMargins: {
            top: 0.85,
            bottom: 0,
          },
        });
        volumeSeriesRef.current = volumeSeries;

        // Set volume data with dynamic colors
        volumeSeries.setData(data.map((item: ChartData, index: number) => {
          const prevClose = index > 0 ? data[index - 1].close : item.open;
          const volumeColor = item.close >= prevClose ? chartColors.volumeUp : chartColors.volumeDown;

          return {
            time: item.time as Time,
            value: item.volume,
            color: volumeColor,
          };
        }));
      }

      // Add Moving Averages
      if (indicators.ma) {
        const ma20 = calculateSMA(data, 20);
        const ma50 = calculateSMA(data, 50);

        const ma20Series = chart.addLineSeries({
          color: '#FFA500',
          lineWidth: 1,
          lineStyle: LineStyle.Solid,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        ma20Series.setData(ma20);

        const ma50Series = chart.addLineSeries({
          color: '#FF6347',
          lineWidth: 1,
          lineStyle: LineStyle.Solid,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        ma50Series.setData(ma50);
      }

      // Add EMA
      if (indicators.ema) {
        const ema12 = calculateEMA(data, 12);
        const ema26 = calculateEMA(data, 26);

        const ema12Series = chart.addLineSeries({
          color: '#00CED1',
          lineWidth: 1,
          lineStyle: LineStyle.Solid,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        ema12Series.setData(ema12);

        const ema26Series = chart.addLineSeries({
          color: '#4169E1',
          lineWidth: 1,
          lineStyle: LineStyle.Solid,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        ema26Series.setData(ema26);
      }

      // Add Bollinger Bands
      if (indicators.bollinger) {
        const bands = calculateBollingerBands(data);

        const upperBand = chart.addLineSeries({
          color: '#8B5CF6',
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        upperBand.setData(bands.upper);

        const lowerBand = chart.addLineSeries({
          color: '#8B5CF6',
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        lowerBand.setData(bands.lower);

        const middleBand = chart.addLineSeries({
          color: '#8B5CF6',
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        middleBand.setData(bands.middle);
      }

      // Signal markers
      const symbolSignals = signals.filter(s => s.symbol === symbol);
      if (series && symbolSignals.length > 0 && data.length > 0) {
        const markers = symbolSignals.map((signal: PreciseOptionsSignal, index: number) => {
          const timeIndex = Math.max(0, data.length - 1 - index * 5);
          const isBuy = signal.type === 'CALL';
          return {
            time: data[timeIndex].time as Time,
            position: isBuy ? 'belowBar' : 'aboveBar' as 'belowBar' | 'aboveBar',
            color: isBuy ? '#00D4AA' : '#FF3B30',
            shape: isBuy ? 'arrowUp' : 'arrowDown' as 'arrowUp' | 'arrowDown',
            text: `${signal.type} (${Math.round(signal.confidence)}%)`,
            size: 2,
          };
        });
        series.setMarkers(markers);
      }

      // Add AI insights if enabled
      if (showAIInsights && aiInsights) {
        // Support and resistance levels with strength
        aiInsights.levels?.forEach((level: AIInsight) => {
          const lineWidth = level.strength && level.strength > 0.8 ? 2 : 1;
          const lineStyle = level.strength && level.strength > 0.8 ? LineStyle.Solid : LineStyle.Dashed;

          if (series) {
            series.createPriceLine({
              price: level.price,
              color: level.type === 'SUPPORT' ? '#10B981' : '#EF4444',
              lineWidth: lineWidth,
              lineStyle: lineStyle,
              axisLabelVisible: true,
              title: `${level.label} (${Math.round((level.strength || 0.5) * 100)}%)`,
            });
          }
        });

        // Entry and exit points
        if (series && aiInsights.signals && data.length > 0) {
          const aiMarkers = aiInsights.signals.map((signal: AIInsight, index: number) => ({
            time: data[Math.max(0, data.length - 20 + index * 5)].time as Time,
            position: signal.type === 'ENTRY' ? 'belowBar' : 'aboveBar' as 'belowBar' | 'aboveBar',
            color: signal.type === 'ENTRY' ? '#10B981' : '#EF4444',
            shape: signal.type === 'ENTRY' ? 'arrowUp' : 'arrowDown' as 'arrowUp' | 'arrowDown',
            text: `${signal.label} (${signal.confidence}%)`,
          }));
          series.setMarkers(aiMarkers);
        }

        // Prediction line
        if (indicators.predictions && aiInsights.predictions && data.length > 0) {
          const predictionSeries = chart.addLineSeries({
            color: '#8B5CF6',
            lineWidth: 2,
            lineStyle: LineStyle.Dashed,
            priceLineVisible: false,
            lastValueVisible: true,
            crosshairMarkerVisible: true,
          });

          const lastDataPoint = data[data.length - 1];
          const predictionData = [
            { time: lastDataPoint.time as Time, value: lastDataPoint.close },
            ...aiInsights.predictions.map((pred: any, index: number) => ({
              time: (lastDataPoint.time + (index + 1) * 3600) as Time,
              value: pred.price,
            }))
          ];
          predictionSeries.setData(predictionData);
        }
      }

      // Add Entry/Exit Markers with Stop Loss and Take Profit
      if (data.length > 0) {
        const currentPrice = data[data.length - 1].close;
        const entryIndex = Math.max(0, data.length - 30);
        const entryPrice = data[entryIndex].close;

        // Calculate stop loss and take profit based on ATR or percentage
        const stopLossPercent = 0.02; // 2% stop loss
        const takeProfitPercent = 0.05; // 5% take profit
        const stopLossPrice = entryPrice * (1 - stopLossPercent);
        const takeProfitPrice = entryPrice * (1 + takeProfitPercent);

        // Entry marker
        const entryExitMarkers = [
          {
            time: data[entryIndex].time as Time,
            position: 'belowBar' as const,
            color: '#00D4AA',
            shape: 'arrowUp' as const,  // Changed from 'circle' to 'arrowUp'
            text: 'BUY',
            size: 1,  // Reduced size for smaller arrow
          }
        ];

        // Add exit marker if we've reached target or stop
        if (currentPrice >= takeProfitPrice || currentPrice <= stopLossPrice) {
          const exitIndex = data.length - 1;
          entryExitMarkers.push({
            time: data[exitIndex].time as Time,
            position: 'aboveBar' as const,
            color: currentPrice >= takeProfitPrice ? '#00D4AA' : '#FF3B30',
            shape: 'arrowDown' as const,  // Changed from 'circle' to 'arrowDown'
            text: currentPrice >= takeProfitPrice ? 'SELL (TP)' : 'SELL (SL)',
            size: 1,  // Reduced size for smaller arrow
          });
        }

        if (series) {
          series.setMarkers(entryExitMarkers);

          // Stop Loss line
          series.createPriceLine({
            price: stopLossPrice,
            color: '#FF3B30',
            lineWidth: 2,
            lineStyle: LineStyle.Dashed,
            axisLabelVisible: true,
            title: `Stop Loss (-${(stopLossPercent * 100).toFixed(1)}%)`,
          });

          // Take Profit line
          series.createPriceLine({
            price: takeProfitPrice,
            color: '#00D4AA',
            lineWidth: 2,
            lineStyle: LineStyle.Dashed,
            axisLabelVisible: true,
            title: `Take Profit (+${(takeProfitPercent * 100).toFixed(1)}%)`,
          });

          // Entry line
          series.createPriceLine({
            price: entryPrice,
            color: '#0066FF',
            lineWidth: 1,
            lineStyle: LineStyle.Solid,
            axisLabelVisible: true,
            title: 'Entry',
          });
        }
      }

      // Add Predictive Trendline
      if (indicators.trendPrediction && data.length > 20) {
        // Calculate linear regression for trend
        const recentData = data.slice(-20);
        const n = recentData.length;
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

        recentData.forEach((point, i) => {
          sumX += i;
          sumY += point.close;
          sumXY += i * point.close;
          sumX2 += i * i;
        });

        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;

        // Create trendline data
        const trendlineData = [];
        const futurePoints = 10; // Predict 10 periods into the future

        for (let i = 0; i < n + futurePoints; i++) {
          const timeIndex = i < n ? data.length - n + i : data.length - 1 + (i - n + 1);
          const time = i < n
            ? recentData[i].time
            : data[data.length - 1].time + (i - n + 1) * (recentData[1].time - recentData[0].time);

          trendlineData.push({
            time: time as Time,
            value: intercept + slope * i,
          });
        }

        const trendlineSeries = chart.addLineSeries({
          color: '#F59E0B',
          lineWidth: 2,
          lineStyle: LineStyle.Solid,
          priceLineVisible: false,
          lastValueVisible: true,
          crosshairMarkerVisible: false,
          title: 'Trend Prediction',
        });
        trendlineSeries.setData(trendlineData);

        // Add confidence bands
        const upperBandData = trendlineData.map(point => ({
          ...point,
          value: point.value * 1.02, // 2% upper band
        }));
        const lowerBandData = trendlineData.map(point => ({
          ...point,
          value: point.value * 0.98, // 2% lower band
        }));

        const upperBandSeries = chart.addLineSeries({
          color: 'rgba(245, 158, 11, 0.3)',
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        upperBandSeries.setData(upperBandData);

        const lowerBandSeries = chart.addLineSeries({
          color: 'rgba(245, 158, 11, 0.3)',
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        lowerBandSeries.setData(lowerBandData);
      }

      // Add pattern overlays
      if (indicators.patterns && aiInsights?.patterns) {
        aiInsights.patterns.forEach((pattern: Pattern) => {
          const patternSeries = chart.addLineSeries({
            color: '#06B6D4',
            lineWidth: 2,
            lineStyle: LineStyle.Solid,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
          });

          const patternData = pattern.points.map(p => ({
            time: p.time as Time,
            value: p.price,
          }));
          patternSeries.setData(patternData);
        });
      }

      // Add divergence indicators
      if (indicators.divergences && aiInsights?.divergences) {
        aiInsights.divergences.forEach((div: Divergence) => {
          const divergenceSeries = chart.addLineSeries({
            color: div.type === 'bullish' ? '#10B981' : '#EF4444',
            lineWidth: 2,
            lineStyle: LineStyle.SparseDotted,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
          });

          divergenceSeries.setData([
            { time: div.startTime as Time, value: div.startPrice },
            { time: div.endTime as Time, value: div.endPrice },
          ]);
        });
      }

    } catch (error) {
      console.error('Chart series creation error:', error);
      // Fallback: try with simplified data
      try {
        if (!series) {
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
  }, [theme, height, chartType, selectedTimeframe, marketData, signals, symbol, historicalData, aiInsights, indicators, showAIInsights, chartTheme]);

  const currentPrice = marketData?.price;
  const priceChange = marketData?.change;
  const priceChangePercent = marketData?.change_percent;
  const isPositive = (priceChange ?? 0) >= 0;

  // Get chart colors based on theme
  const chartColors = CHART_THEMES[chartTheme];

  // Market status
  const marketStatus = useMemo(() => {
    const now = new Date();
    const hours = now.getHours();
    const minutes = now.getMinutes();
    const day = now.getDay();

    // Market hours: 9:30 AM - 4:00 PM ET (adjust for your timezone)
    const isWeekday = day >= 1 && day <= 5;
    const isMarketHours = hours >= 9 && (hours < 16 || (hours === 16 && minutes === 0));
    const isPreMarket = hours >= 4 && hours < 9.5;
    const isAfterHours = (hours === 16 && minutes > 0) || (hours > 16 && hours < 20);

    if (!isWeekday) return { status: 'CLOSED', label: 'Market Closed', color: '#6B7280' };
    if (isMarketHours) return { status: 'OPEN', label: 'Market Open', color: '#10B981' };
    if (isPreMarket) return { status: 'PRE', label: 'Pre-Market', color: '#F59E0B' };
    if (isAfterHours) return { status: 'AFTER', label: 'After Hours', color: '#8B5CF6' };
    return { status: 'CLOSED', label: 'Market Closed', color: '#6B7280' };
  }, []);

  // Zoom control functions
  const handleZoomIn = () => {
    if (chartRef.current) {
      const timeScale = chartRef.current.timeScale();
      const currentBarSpacing = timeScale.options().barSpacing || 6;
      timeScale.applyOptions({
        barSpacing: currentBarSpacing * 1.1,
      });
    }
  };

  const handleZoomOut = () => {
    if (chartRef.current) {
      const timeScale = chartRef.current.timeScale();
      const currentBarSpacing = timeScale.options().barSpacing || 6;
      timeScale.applyOptions({
        barSpacing: currentBarSpacing * 0.9,
      });
    }
  };

  const handleResetZoom = () => {
    if (chartRef.current) {
      chartRef.current.timeScale().resetTimeScale();
      chartRef.current.priceScale('right').applyOptions({ autoScale: true });
    }
  };

  const handleScreenshot = () => {
    if (chartContainerRef.current) {
      html2canvas(chartContainerRef.current).then((canvas) => {
        const link = document.createElement('a');
        link.download = `${symbol}-chart-${new Date().toISOString()}.png`;
        link.href = canvas.toDataURL();
        link.click();
      });
    }
  };

  const handleFullscreen = () => {
    if (!document.fullscreenElement) {
      chartContainerRef.current?.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const handleIndicatorToggle = (indicator: keyof typeof indicators) => {
    setIndicators(prev => ({ ...prev, [indicator]: !prev[indicator] }));
  };

  // AI Analysis Functions
  const runAIAnalysis = async () => {
    if (!symbol || isAnalyzing) return;

    setIsAnalyzing(true);
    setDetectedPatterns([]);
    setAIInsights([]);
    setShowAIPanel(false);

    // Clear previous analysis overlays
    if (chartRef.current && candlestickSeriesRef.current) {
      // Remove all price lines
      const series = candlestickSeriesRef.current;
      try {
        // Clear any existing price lines (this is a workaround as lightweight-charts doesn't have a clear method)
        chartRef.current.timeScale().fitContent();
      } catch (e) {
        console.log('Clearing previous analysis');
      }
    }

    // Simulate AI analysis with a brief delay
    await new Promise(resolve => setTimeout(resolve, 1500));

    // Run all technical analysis functions
    drawSupportResistanceLevels();
    drawTrendLines();
    drawFibonacciLevels();
    drawVolumeProfile();

    // Detect and draw patterns
    const detected = detectAndDrawPatterns();
    setDetectedPatterns(detected);

    // Automatically enable relevant indicators
    setIndicators(prev => ({
      ...prev,
      supportResistance: true,
      fibonacci: true,
      patterns: true,
      ma: true,
      ema: true,
      volume: true,
    }));

    // Add market structure analysis
    if (historicalData && historicalData.length > 0) {
      const currentPrice = historicalData[historicalData.length - 1].close;
      const ma20 = calculateSMA(historicalData, 20);
      const ma50 = calculateSMA(historicalData, 50);

      // Trend analysis
      const shortTermTrend = ma20.length > 0 && currentPrice > ma20[ma20.length - 1].value ? 'BULLISH' : 'BEARISH';
      const longTermTrend = ma50.length > 0 && currentPrice > ma50[ma50.length - 1].value ? 'BULLISH' : 'BEARISH';

      // Add market structure insight
      setAIInsights(prev => [...prev, {
        price: currentPrice,
        type: 'ENTRY',
        confidence: 85,
        label: `Market Structure: ${shortTermTrend} short-term, ${longTermTrend} long-term`,
      }]);

      // Volume analysis
      const avgVolume = historicalData.slice(-20).reduce((sum, d) => sum + d.volume, 0) / 20;
      const currentVolume = historicalData[historicalData.length - 1].volume;
      const volumeStrength = currentVolume > avgVolume * 1.5 ? 'HIGH' : currentVolume < avgVolume * 0.5 ? 'LOW' : 'NORMAL';

      if (volumeStrength !== 'NORMAL') {
        setAIInsights(prev => [...prev, {
          price: currentPrice,
          type: 'ENTRY',
          confidence: 70,
          label: `Volume: ${volumeStrength} - ${volumeStrength === 'HIGH' ? 'Increased interest' : 'Decreased activity'}`,
        }]);
      }

      // Price action analysis
      const priceRange = Math.max(...historicalData.slice(-20).map(d => d.high)) -
        Math.min(...historicalData.slice(-20).map(d => d.low));
      const currentRange = historicalData[historicalData.length - 1].high - historicalData[historicalData.length - 1].low;

      if (currentRange > priceRange * 0.7) {
        setAIInsights(prev => [...prev, {
          price: currentPrice,
          type: 'ENTRY',
          confidence: 75,
          label: 'Wide range bar - Potential volatility expansion',
        }]);
      }
    }

    setIsAnalyzing(false);
    setShowAIPanel(true);

    // Auto-hide panel after 30 seconds
    setTimeout(() => {
      setShowAIPanel(false);
    }, 30000);
  };

  // Enhanced pattern visualization with annotations
  const visualizePatterns = () => {
    if (!chartRef.current || !candlestickSeriesRef.current || detectedPatterns.length === 0) return;

    detectedPatterns.forEach(pattern => {
      if (pattern.points.length >= 2) {
        // Draw pattern lines
        const patternSeries = chartRef.current!.addLineSeries({
          color: CHART_THEMES[chartTheme].crosshair,
          lineWidth: 2,
          lineStyle: LineStyle.Solid,
          lastValueVisible: false,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
        });

        patternSeries.setData(pattern.points.map(p => ({
          time: p.time as Time,
          value: p.price,
        })));

        // Add pattern markers
        const markers: SeriesMarker<Time>[] = pattern.points.map((p, index) => ({
          time: p.time as Time,
          position: 'aboveBar' as const,
          color: CHART_THEMES[chartTheme].crosshair,
          shape: 'circle' as const,
          text: index === 0 ? pattern.type.charAt(0) : '',
        }));

        candlestickSeriesRef.current!.setMarkers(markers);
      }
    });
  };

  // Enhanced Technical Analysis Drawing Functions
  const drawSupportResistanceLevels = () => {
    if (!chartRef.current || !candlestickSeriesRef.current || !historicalData || historicalData.length === 0) return;

    const chart = chartRef.current;
    const recentData = historicalData.slice(-100);

    // Find key levels
    const levels: { price: number; strength: number; type: string; touches: number }[] = [];

    // Analyze price action for support/resistance
    for (let i = 10; i < recentData.length - 10; i++) {
      const current = recentData[i];
      const isLocalHigh = current.high > Math.max(...recentData.slice(i - 5, i).map(d => d.high)) &&
        current.high > Math.max(...recentData.slice(i + 1, i + 6).map(d => d.high));
      const isLocalLow = current.low < Math.min(...recentData.slice(i - 5, i).map(d => d.low)) &&
        current.low < Math.min(...recentData.slice(i + 1, i + 6).map(d => d.low));

      if (isLocalHigh) {
        levels.push({ price: current.high, strength: 1, type: 'RESISTANCE', touches: 1 });
      }
      if (isLocalLow) {
        levels.push({ price: current.low, strength: 1, type: 'SUPPORT', touches: 1 });
      }
    }

    // Consolidate nearby levels
    const consolidatedLevels = levels.reduce((acc, level) => {
      const existing = acc.find(l => Math.abs(l.price - level.price) / level.price < 0.005);
      if (existing) {
        existing.touches += 1;
        existing.strength = Math.min(existing.touches * 0.2, 1);
      } else {
        acc.push(level);
      }
      return acc;
    }, [] as typeof levels);

    // Sort by strength and limit to top levels
    const topLevels = consolidatedLevels
      .sort((a, b) => b.strength - a.strength)
      .slice(0, 6);

    // Draw levels with annotations
    topLevels.forEach((level) => {
      const lineColor = level.type === 'SUPPORT'
        ? CHART_THEMES[chartTheme].upColor
        : CHART_THEMES[chartTheme].downColor;

      // Create price line
      const priceLine = candlestickSeriesRef.current!.createPriceLine({
        price: level.price,
        color: lineColor,
        lineWidth: Math.max(1, Math.floor(level.strength * 3)),
        lineStyle: level.strength > 0.7 ? LineStyle.Solid : LineStyle.Dashed,
        axisLabelVisible: true,
        title: `${level.type} (${level.touches} touches)`,
      });

      // Add zone visualization
      if (level.strength > 0.5) {
        const zoneHeight = level.price * 0.002;
        candlestickSeriesRef.current!.createPriceLine({
          price: level.price + (level.type === 'RESISTANCE' ? zoneHeight : -zoneHeight),
          color: alpha(lineColor, 0.1),
          lineWidth: 10,
          lineStyle: LineStyle.Solid,
          axisLabelVisible: false,
        });
      }
    });

    // Add annotations for key levels
    const currentPrice = historicalData[historicalData.length - 1].close;
    const nearestSupport = topLevels
      .filter(l => l.type === 'SUPPORT' && l.price < currentPrice)
      .sort((a, b) => b.price - a.price)[0];
    const nearestResistance = topLevels
      .filter(l => l.type === 'RESISTANCE' && l.price > currentPrice)
      .sort((a, b) => a.price - b.price)[0];

    if (nearestSupport) {
      setAIInsights(prev => [...prev, {
        price: nearestSupport.price,
        type: 'SUPPORT',
        confidence: nearestSupport.strength * 100,
        label: `Key Support - ${nearestSupport.touches} bounces`,
        strength: nearestSupport.strength,
      }]);
    }

    if (nearestResistance) {
      setAIInsights(prev => [...prev, {
        price: nearestResistance.price,
        type: 'RESISTANCE',
        confidence: nearestResistance.strength * 100,
        label: `Key Resistance - ${nearestResistance.touches} rejections`,
        strength: nearestResistance.strength,
      }]);
    }
  };

  const drawTrendLines = () => {
    if (!chartRef.current || !candlestickSeriesRef.current || !historicalData || historicalData.length === 0) return;

    const data = historicalData.slice(-100);

    // Find swing highs and lows
    const swingHighs: { index: number; price: number; time: number }[] = [];
    const swingLows: { index: number; price: number; time: number }[] = [];

    for (let i = 5; i < data.length - 5; i++) {
      const isSwingHigh = data[i].high > Math.max(...data.slice(i - 5, i).map(d => d.high)) &&
        data[i].high > Math.max(...data.slice(i + 1, i + 6).map(d => d.high));
      const isSwingLow = data[i].low < Math.min(...data.slice(i - 5, i).map(d => d.low)) &&
        data[i].low < Math.min(...data.slice(i + 1, i + 6).map(d => d.low));

      if (isSwingHigh) {
        swingHighs.push({ index: i, price: data[i].high, time: data[i].time });
      }
      if (isSwingLow) {
        swingLows.push({ index: i, price: data[i].low, time: data[i].time });
      }
    }

    // Draw uptrend lines (connecting swing lows)
    if (swingLows.length >= 2) {
      const lastTwo = swingLows.slice(-2);
      const slope = (lastTwo[1].price - lastTwo[0].price) / (lastTwo[1].index - lastTwo[0].index);

      if (slope > 0) {
        const trendLine = candlestickSeriesRef.current!.createPriceLine({
          price: lastTwo[1].price,
          color: CHART_THEMES[chartTheme].upColor,
          lineWidth: 2,
          lineStyle: LineStyle.Solid,
          axisLabelVisible: false,
          title: 'Uptrend Support',
        });

        // Add trend channel
        const channelWidth = Math.abs(swingHighs[swingHighs.length - 1]?.price - lastTwo[1].price) || lastTwo[1].price * 0.02;
        candlestickSeriesRef.current!.createPriceLine({
          price: lastTwo[1].price + channelWidth,
          color: alpha(CHART_THEMES[chartTheme].upColor, 0.5),
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: false,
          title: 'Channel Top',
        });

        setDetectedPatterns(prev => [...prev, {
          type: 'UPTREND CHANNEL',
          confidence: 85,
          points: lastTwo.map(p => ({ time: p.time, price: p.price })),
        }]);
      }
    }

    // Draw downtrend lines (connecting swing highs)
    if (swingHighs.length >= 2) {
      const lastTwo = swingHighs.slice(-2);
      const slope = (lastTwo[1].price - lastTwo[0].price) / (lastTwo[1].index - lastTwo[0].index);

      if (slope < 0) {
        const trendLine = candlestickSeriesRef.current!.createPriceLine({
          price: lastTwo[1].price,
          color: CHART_THEMES[chartTheme].downColor,
          lineWidth: 2,
          lineStyle: LineStyle.Solid,
          axisLabelVisible: false,
          title: 'Downtrend Resistance',
        });

        setDetectedPatterns(prev => [...prev, {
          type: 'DOWNTREND',
          confidence: 80,
          points: lastTwo.map(p => ({ time: p.time, price: p.price })),
        }]);
      }
    }
  };

  const drawFibonacciLevels = () => {
    if (!chartRef.current || !candlestickSeriesRef.current || !historicalData || historicalData.length === 0) return;

    const data = historicalData.slice(-50);
    const high = Math.max(...data.map(d => d.high));
    const low = Math.min(...data.map(d => d.low));
    const diff = high - low;

    const fibLevels = [
      { level: 0, label: '0%' },
      { level: 0.236, label: '23.6%' },
      { level: 0.382, label: '38.2%' },
      { level: 0.5, label: '50%' },
      { level: 0.618, label: '61.8%' },
      { level: 0.786, label: '78.6%' },
      { level: 1, label: '100%' },
    ];

    fibLevels.forEach((fib) => {
      const price = low + (diff * fib.level);
      const isKeyLevel = [0.382, 0.5, 0.618].includes(fib.level);

      candlestickSeriesRef.current!.createPriceLine({
        price: price,
        color: isKeyLevel ? CHART_THEMES[chartTheme].crosshair : alpha(CHART_THEMES[chartTheme].text, 0.3),
        lineWidth: isKeyLevel ? 2 : 1,
        lineStyle: isKeyLevel ? LineStyle.Solid : LineStyle.Dashed,
        axisLabelVisible: true,
        title: `Fib ${fib.label}`,
      });

      // Add to insights for key levels
      if (isKeyLevel) {
        setAIInsights(prev => [...prev, {
          price: price,
          type: 'SUPPORT',
          confidence: 70,
          label: `Fibonacci ${fib.label} - Key retracement level`,
        }]);
      }
    });

    // Add extension levels
    const extensions = [
      { level: 1.272, label: '127.2%' },
      { level: 1.618, label: '161.8%' },
    ];

    extensions.forEach((ext) => {
      const price = low + (diff * ext.level);
      candlestickSeriesRef.current!.createPriceLine({
        price: price,
        color: alpha(CHART_THEMES[chartTheme].crosshair, 0.5),
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: true,
        title: `Fib Ext ${ext.label}`,
      });
    });
  };

  const detectAndDrawPatterns = () => {
    if (!historicalData || historicalData.length < 20) return [];

    const patterns: Pattern[] = [];
    const data = historicalData.slice(-50);

    // Head and Shoulders Detection
    for (let i = 10; i < data.length - 10; i++) {
      const leftShoulder = data[i - 5];
      const head = data[i];
      const rightShoulder = data[i + 5];
      const neckline = Math.min(data[i - 2].low, data[i + 2].low);

      if (head.high > leftShoulder.high &&
        head.high > rightShoulder.high &&
        Math.abs(leftShoulder.high - rightShoulder.high) / leftShoulder.high < 0.02) {

        patterns.push({
          type: 'HEAD & SHOULDERS',
          confidence: 85,
          points: [
            { time: leftShoulder.time, price: leftShoulder.high },
            { time: head.time, price: head.high },
            { time: rightShoulder.time, price: rightShoulder.high },
          ],
        });

        // Draw neckline
        if (candlestickSeriesRef.current) {
          candlestickSeriesRef.current.createPriceLine({
            price: neckline,
            color: CHART_THEMES[chartTheme].downColor,
            lineWidth: 2,
            lineStyle: LineStyle.Dashed,
            axisLabelVisible: true,
            title: 'Neckline - Breakdown target',
          });
        }
      }
    }

    // Triangle Pattern Detection
    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const isConverging = highs[highs.length - 1] - lows[lows.length - 1] <
      (highs[0] - lows[0]) * 0.5;

    if (isConverging && data.length > 20) {
      patterns.push({
        type: 'TRIANGLE PATTERN',
        confidence: 75,
        points: [
          { time: data[0].time, price: highs[0] },
          { time: data[data.length - 1].time, price: highs[highs.length - 1] },
        ],
      });
    }

    // Flag Pattern Detection
    const trend = data[data.length - 1].close - data[0].close;
    const recentRange = Math.max(...data.slice(-10).map(d => d.high)) -
      Math.min(...data.slice(-10).map(d => d.low));
    const previousRange = Math.max(...data.slice(0, 10).map(d => d.high)) -
      Math.min(...data.slice(0, 10).map(d => d.low));

    if (Math.abs(trend) > previousRange && recentRange < previousRange * 0.5) {
      patterns.push({
        type: trend > 0 ? 'BULL FLAG' : 'BEAR FLAG',
        confidence: 80,
        points: [
          { time: data[data.length - 10].time, price: data[data.length - 10].close },
          { time: data[data.length - 1].time, price: data[data.length - 1].close },
        ],
      });

      // Add breakout target
      const target = data[data.length - 1].close + (trend > 0 ? previousRange : -previousRange);
      setAIInsights(prev => [...prev, {
        price: target,
        type: 'TARGET',
        confidence: 80,
        label: `${trend > 0 ? 'Bull' : 'Bear'} Flag Target`,
      }]);
    }

    return patterns;
  };

  const drawVolumeProfile = () => {
    if (!historicalData || !volumeSeriesRef.current) return;

    const data = historicalData.slice(-50);
    const priceRange = Math.max(...data.map(d => d.high)) - Math.min(...data.map(d => d.low));
    const priceBins = 20;
    const binSize = priceRange / priceBins;
    const volumeProfile: { price: number; volume: number }[] = [];

    // Calculate volume at each price level
    for (let i = 0; i < priceBins; i++) {
      const binPrice = Math.min(...data.map(d => d.low)) + (i * binSize);
      const binVolume = data
        .filter(d => d.low <= binPrice && d.high >= binPrice)
        .reduce((sum, d) => sum + d.volume, 0);

      volumeProfile.push({ price: binPrice, volume: binVolume });
    }

    // Find high volume nodes (areas of high trading activity)
    const maxVolume = Math.max(...volumeProfile.map(vp => vp.volume));
    const hvnThreshold = maxVolume * 0.7;

    volumeProfile
      .filter(vp => vp.volume > hvnThreshold)
      .forEach(hvn => {
        if (candlestickSeriesRef.current) {
          candlestickSeriesRef.current.createPriceLine({
            price: hvn.price,
            color: alpha(CHART_THEMES[chartTheme].crosshair, 0.3),
            lineWidth: 3,
            lineStyle: LineStyle.Solid,
            axisLabelVisible: false,
            title: 'High Volume Node',
          });

          setAIInsights(prev => [...prev, {
            price: hvn.price,
            type: 'SUPPORT',
            confidence: 65,
            label: 'High Volume Node - Potential support/resistance',
          }]);
        }
      });
  };

  return (
    <Card sx={{
      p: 2,
      backgroundColor: chartColors.background,
      border: `1px solid ${chartColors.border}`,
      borderRadius: '12px',
      height: isFullscreen ? '100vh' : height + 100,
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Chart Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1, flexShrink: 0 }}>
        {/* Left Side: Symbol, Price, Timeframe */}
        <Stack direction="row" spacing={2} alignItems="center">
          {/* Symbol Display */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ShowChart sx={{ color: chartColors.text, fontSize: '1.2rem' }} />
            <Typography
              variant="h6"
              sx={{
                color: chartColors.text,
                fontWeight: 'bold',
                letterSpacing: '0.5px'
              }}
            >
              {symbol}
            </Typography>
          </Box>

          {/* Price Info */}
          <Stack direction="row" spacing={1} alignItems="baseline">
            {currentPrice ? (
              <>
                <Typography variant="h6" sx={{ color: isPositive ? chartColors.upColor : chartColors.downColor }}>
                  {currentPrice.toFixed(2)}
                </Typography>
                <Typography variant="body2" sx={{ color: isPositive ? chartColors.upColor : chartColors.downColor }}>
                  {priceChange?.toFixed(2)} ({priceChangePercent?.toFixed(2)}%)
                </Typography>
              </>
            ) : <Skeleton variant="text" width={150} />}
          </Stack>

          {/* Market Status */}
          <Tooltip title={marketStatus.label}>
            <Chip
              label={marketStatus.status}
              size="small"
              sx={{
                backgroundColor: alpha(marketStatus.color, 0.2),
                color: marketStatus.color,
                fontWeight: 'bold'
              }}
            />
          </Tooltip>

          {/* Timeframe Selector */}
          <Select
            value={selectedTimeframe.value}
            onChange={handleTimeframeChange}
            variant="standard"
            size="small"
            sx={{
              color: chartColors.text,
              fontSize: '0.9rem',
              '& .MuiSelect-icon': {
                color: chartColors.text,
              },
              '&:before': {
                borderBottom: 'none',
              },
              '&:hover:not(.Mui-disabled):before': {
                borderBottom: 'none',
              },
            }}
          >
            {TIMEFRAME_OPTIONS.map(tf => (
              <MenuItem key={tf.value} value={tf.value}>{tf.shortLabel}</MenuItem>
            ))}
          </Select>
        </Stack>

        {/* Right Side: Chart Controls */}
        <Stack direction="row" spacing={0.5} alignItems="center">
          <ToggleButtonGroup size="small" value={chartType} exclusive onChange={(e, newType) => newType && setChartType(newType)}>
            <ToggleButton value="candlestick" sx={{ color: chartColors.text }}><Tooltip title="Candlestick"><CandlestickChart fontSize="small" /></Tooltip></ToggleButton>
            <ToggleButton value="line" sx={{ color: chartColors.text }}><Tooltip title="Line"><ShowChart fontSize="small" /></Tooltip></ToggleButton>
            <ToggleButton value="area" sx={{ color: chartColors.text }}><Tooltip title="Area"><AreaChart fontSize="small" /></Tooltip></ToggleButton>
          </ToggleButtonGroup>

          <Tooltip title="Indicators">
            <IconButton onClick={(e) => setIndicatorsMenuAnchor(e.currentTarget)} size="small">
              <Analytics sx={{ color: chartColors.text }} />
            </IconButton>
          </Tooltip>
          <Menu
            anchorEl={indicatorsMenuAnchor}
            open={Boolean(indicatorsMenuAnchor)}
            onClose={() => setIndicatorsMenuAnchor(null)}
          >
            {Object.keys(indicators).map(key => (
              <MenuItem key={key}>
                <FormControlLabel
                  control={<Switch checked={indicators[key as keyof typeof indicators]} onChange={() => handleIndicatorToggle(key as keyof typeof indicators)} />}
                  label={key.charAt(0).toUpperCase() + key.slice(1)}
                />
              </MenuItem>
            ))}
          </Menu>

          <Tooltip title="AI Analysis">
            <IconButton onClick={runAIAnalysis} size="small" disabled={isAnalyzing}>
              {isAnalyzing ? <CircularProgress size={20} /> : <AIIcon sx={{ color: chartColors.text }} />}
            </IconButton>
          </Tooltip>

          <Tooltip title="Take Screenshot">
            <IconButton onClick={handleScreenshot} size="small"><CameraAlt sx={{ color: chartColors.text }} /></IconButton>
          </Tooltip>

          <Tooltip title="Reset View">
            <IconButton onClick={handleResetZoom} size="small"><RestartAlt sx={{ color: chartColors.text }} /></IconButton>
          </Tooltip>

          <Tooltip title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}>
            <IconButton onClick={handleFullscreen} size="small">
              {isFullscreen ? <FullscreenExit sx={{ color: chartColors.text }} /> : <Fullscreen sx={{ color: chartColors.text }} />}
            </IconButton>
          </Tooltip>
        </Stack>
      </Box>

      {/* Main Chart Area */}
      <Box sx={{ flex: 1, position: 'relative' }}>
        <AnimatePresence>
          {historicalLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: alpha(chartColors.background, 0.7),
                zIndex: 10,
              }}
            >
              <CircularProgress />
            </motion.div>
          )}
        </AnimatePresence>

        <Box ref={chartContainerRef} sx={{ width: '100%', height: '100%' }} />

        <AnimatePresence>
          {showAIPanel && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
            >
              <Paper sx={{
                position: 'absolute',
                bottom: '20px',
                left: '20px',
                p: 2,
                backgroundColor: alpha(chartColors.background, 0.9),
                backdropFilter: 'blur(5px)',
                borderRadius: '8px',
                border: `1px solid ${chartColors.border}`,
                maxWidth: '300px'
              }}>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Typography variant="subtitle2" sx={{ color: chartColors.text }}>AI Analysis</Typography>
                  <IconButton size="small" onClick={() => setShowAIPanel(false)}><CloseIcon fontSize="small" sx={{ color: chartColors.text }} /></IconButton>
                </Stack>
                <Divider sx={{ my: 1, borderColor: chartColors.border }} />
                {detectedPatterns.map((p, i) => (
                  <Chip key={i} label={`${p.type} (${p.confidence.toFixed(0)}%)`} size="small" sx={{ mr: 1, mb: 1, backgroundColor: alpha('#06B6D4', 0.3), color: '#06B6D4' }} />
                ))}
                {aiInsights.map((insight, i) => (
                  <Typography key={i} variant="caption" display="block" sx={{ color: chartColors.text, mt: 0.5 }}>- {insight.label}</Typography>
                ))}
              </Paper>
            </motion.div>
          )}
        </AnimatePresence>
      </Box>
    </Card>
  );
});

TradingChart.displayName = 'TradingChart';

export default TradingChart;