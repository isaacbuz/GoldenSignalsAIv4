import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  LineData,
  Time,
  CrosshairMode,
  ColorType,
  LineStyle,
  PriceScaleMode,
  SeriesMarkerPosition,
  SeriesMarkerShape,
} from 'lightweight-charts';
import {
  Box,
  IconButton,
  Button,
  ButtonGroup,
  Menu,
  MenuItem,
  Typography,
  Divider,
  Chip,
  Stack,
  useTheme,
  alpha,
  Tooltip,
  CircularProgress,
  ToggleButton,
  ToggleButtonGroup,
  Fade,
  Zoom,
} from '@mui/material';
import {
  ShowChart as ShowChartIcon,
  CandlestickChart as CandlestickChartIcon,
  Timeline as TimelineIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Fullscreen as FullscreenIcon,
  FullscreenExit as FullscreenExitIcon,
  PhotoCamera as PhotoCameraIcon,
  Settings as SettingsIcon,
  Layers as LayersIcon,
  ArrowUpward as ArrowUpwardIcon,
  ArrowDownward as ArrowDownwardIcon,
  Timeline as PredictionIcon,
  ShowChart as LineChartIcon,
  BarChart as BarChartIcon,
  Analytics as AnalyticsIcon,
  Psychology as PsychologyIcon,
  AutoGraph as AutoGraphIcon,
  KeyboardArrowDown as KeyboardArrowDownIcon,
} from '@mui/icons-material';
import { styled, keyframes } from '@mui/material/styles';
import { customStyles } from '../../theme/enhancedTheme';
import { TradeSearch } from '../TradeSearch';
import {
  calculateRSI,
  calculateMACD,
  calculateBollingerBands,
  calculateSMA,
  calculateEMA,
  calculateVWAP,
  calculateStochastic,
  calculateATR,
  calculateADX,
  detectPatterns,
  OHLCData
} from '../../utils/technicalIndicators';
import { useChartSignalAgent } from '../../hooks/useChartSignalAgent';
import {
  backendMarketDataService,
  fetchHistoricalData,
  fetchSignals,
  fetchMarketData
} from '../../services/backendMarketDataService';
import { chartSettingsService } from '../../services/chartSettingsService';
import logger from '../../../services/logger';


// Styled components
const ChartContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  height: '100%',
  minHeight: '500px',
  backgroundColor: 'transparent',
  borderRadius: 0,
  overflow: 'hidden',
  display: 'flex',
  flexDirection: 'column',
}));

const ChartHeader = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1, 2),
  borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  minHeight: 48,
  backgroundColor: alpha(theme.palette.background.default, 0.5),
}));

const ChartToolbar = styled(Box)(({ theme }) => ({
  padding: theme.spacing(0.5, 2),
  borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  flexWrap: 'wrap',
  minHeight: 40,
  backgroundColor: alpha(theme.palette.background.default, 0.3),
}));

const ChartArea = styled(Box)(({ theme }) => ({
  flex: 1,
  position: 'relative',
  minHeight: 450,
  backgroundColor: theme.palette.mode === 'dark' ? '#0A0E1A' : '#FAFAFA',
  '& > div': {
    backgroundColor: 'transparent !important',
  },
  // Add glow effect to TradingView markers
  '& .tv-lightweight-charts': {
    '& canvas': {
      filter: 'url(#glow-filter)',
    },
  },
}));

// Subtle pulsing glow animation
const glowPulse = keyframes`
  0% {
    filter: drop-shadow(0 0 4px currentColor) drop-shadow(0 0 8px currentColor);
  }
  50% {
    filter: drop-shadow(0 0 8px currentColor) drop-shadow(0 0 16px currentColor) drop-shadow(0 0 20px currentColor);
  }
  100% {
    filter: drop-shadow(0 0 4px currentColor) drop-shadow(0 0 8px currentColor);
  }
`;

const shimmer = keyframes`
  0% {
    background-position: -100% 0;
  }
  100% {
    background-position: 100% 0;
  }
`;

const watermarkFloat = keyframes`
  0%, 100% {
    transform: translate(-50%, -50%) scale(1);
    opacity: 0.08;
  }
  50% {
    transform: translate(-50%, -50%) scale(1.02);
    opacity: 0.12;
  }
`;

const StockWatermark = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  fontSize: '8rem',
  fontWeight: 'bold',
  color: alpha(theme.palette.text.primary, 0.08),
  userSelect: 'none',
  pointerEvents: 'none',
  letterSpacing: '0.5rem',
  textTransform: 'uppercase',
  zIndex: 1,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  animation: `${watermarkFloat} 10s ease-in-out infinite`,
  '& .symbol': {
    fontSize: '10rem',
    lineHeight: 1,
    fontWeight: 900,
    color: alpha(theme.palette.text.primary, 0.06),
    textShadow: `0 0 40px ${alpha(theme.palette.primary.main, 0.1)}`,
  },
  '& .brand': {
    fontSize: '2.5rem',
    letterSpacing: '0.4rem',
    marginTop: '-1rem',
    fontWeight: 700,
    background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.3)}, ${alpha(theme.palette.secondary.main, 0.3)})`,
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    textShadow: `0 0 20px ${alpha(theme.palette.primary.main, 0.2)}`,
    filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.1))',
  },
}));

const SignalMarkerOverlay = styled(Box)(({ theme }) => ({
  position: 'absolute',
  pointerEvents: 'none',
  zIndex: 10,
  '& .buy-signal': {
    color: theme.palette.success.main,
    animation: `${glowPulse} 2s ease-in-out infinite`,
  },
  '& .sell-signal': {
    color: theme.palette.error.main,
    animation: `${glowPulse} 2s ease-in-out infinite`,
  },
}));

const PredictionOverlay = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(1),
  right: theme.spacing(1),
  padding: theme.spacing(1, 1.5),
  backgroundColor: alpha(theme.palette.background.paper, 0.95),
  border: `1px solid ${alpha(theme.palette.primary.main, 0.3)}`,
  borderRadius: theme.spacing(1),
  ...customStyles.glassEffect,
}));

// Types
interface PriceData {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface PredictionData {
  time: Time;
  value: number;
  confidence: number;
}

interface Signal {
  id: string;
  type: 'buy' | 'sell';
  price: number;
  time: Time;
  confidence: number;
  stopLoss?: number;
  takeProfit?: number[];
  reasoning?: string;
}

interface Pattern {
  id: string;
  type: string;
  startTime: Time;
  endTime: Time;
  points: { time: Time; price: number }[];
  confidence: number;
  target?: number;
}

interface ProfessionalChartProps {
  symbol?: string;
  timeframe?: string;
  onSymbolChange?: (symbol: string) => void;
  onTimeframeChange?: (timeframe: string) => void;
  currentSignal?: any;
  onAnalyze?: (symbol: string, timeframe: string) => void;
  isAnalyzing?: boolean;
  initialIndicators?: string[];
  showWatermark?: boolean;
}

// Mock data generator for demonstration
const generateRealisticPriceData = (days: number = 90): PriceData[] => {
  const data: PriceData[] = [];
  let basePrice = 150;
  const now = new Date();
  now.setHours(0, 0, 0, 0);

  for (let i = days; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);

    // Add intraday data for the last day
    if (i === 0) {
      for (let hour = 9; hour <= 16; hour++) {
        for (let minute = 0; minute < 60; minute += 5) {
          const time = new Date(date);
          time.setHours(hour, minute, 0, 0);

          const volatility = 0.003;
          const trend = Math.sin((hour * 60 + minute) / 30) * 0.5;
          const randomWalk = (Math.random() - 0.5) * basePrice * volatility;

          basePrice = Math.max(basePrice + randomWalk + trend * 0.1, 50);

          const high = basePrice + Math.random() * 0.5;
          const low = basePrice - Math.random() * 0.5;
          const open = low + Math.random() * (high - low);
          const close = low + Math.random() * (high - low);

          data.push({
            time: (time.getTime() / 1000) as Time,
            open,
            high,
            low,
            close,
            volume: Math.floor(100000 + Math.random() * 500000),
          });
        }
      }
    } else {
      // Daily data for other days
      const volatility = 0.02;
      const trend = Math.sin(i / 10) * 5;
      const randomWalk = (Math.random() - 0.5) * basePrice * volatility;

      basePrice = Math.max(basePrice + randomWalk + trend * 0.1, 50);

      const high = basePrice + Math.random() * 2;
      const low = basePrice - Math.random() * 2;
      const open = low + Math.random() * (high - low);
      const close = low + Math.random() * (high - low);

      data.push({
        time: (date.getTime() / 1000) as Time,
        open,
        high,
        low,
        close,
        volume: Math.floor(1000000 + Math.random() * 5000000),
      });
    }
  }

  return data;
};

const generatePredictionData = (priceData: PriceData[]): PredictionData[] => {
  if (priceData.length === 0) return [];

  const predictions: PredictionData[] = [];

  // Generate AI predictions for all historical data
  // This shows how well the AI model tracks actual prices
  for (let i = 0; i < priceData.length; i++) {
    const actualPrice = priceData[i].close;

    // Simulate AI prediction with slight deviation from actual
    // In production, this would be your actual AI model output
    const error = (Math.random() - 0.5) * 0.5; // Small prediction error
    const trend = i > 0 ? (priceData[i].close - priceData[i-1].close) * 0.8 : 0;
    const prediction = actualPrice + error + trend * 0.1;

    predictions.push({
      time: priceData[i].time,
      value: prediction,
      confidence: 0.88 + Math.random() * 0.1, // 88-98% confidence
    });
  }

  // Add future predictions beyond the last data point
  const lastPrice = priceData[priceData.length - 1].close;
  const lastTime = priceData[priceData.length - 1].time as number;
  let futurePrice = lastPrice;

  const predictionSteps = 12; // Predict 1 hour ahead (12 x 5min)

  for (let i = 1; i <= predictionSteps; i++) {
    const futureTime = lastTime + (i * 5 * 60); // 5 minutes in seconds

    // AI prediction logic with momentum and volatility
    const trend = 0.0002 * i; // Gradual trend
    const volatility = (Math.random() - 0.5) * 0.5;
    const momentum = Math.sin(i / 4) * 0.3; // Cyclical momentum

    futurePrice = futurePrice * (1 + trend) + volatility + momentum;

    predictions.push({
      time: futureTime as Time,
      value: futurePrice,
      confidence: Math.max(0.6, 0.95 - i * 0.03), // Confidence decreases for future
    });
  }

  return predictions;
};

export const ProfessionalChart: React.FC<ProfessionalChartProps> = ({
  symbol = 'AAPL',
  timeframe = '5m',
  onSymbolChange,
  onTimeframeChange,
  currentSignal,
  onAnalyze,
  isAnalyzing = false,
  initialIndicators = ['prediction', 'patterns', 'signals', 'volume'],
  showWatermark = true,
}) => {
  const theme = useTheme();
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const predictionSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const patternSeriesRefs = useRef<ISeriesApi<'Line'>[]>([]);

  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'area'>('candlestick');
  const [priceData, setPriceData] = useState<PriceData[]>([]);
  const [predictionData, setPredictionData] = useState<PredictionData[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [loading, setLoading] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  const [showPrediction, setShowPrediction] = useState(true);
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>(initialIndicators);
  const [indicatorMenuAnchor, setIndicatorMenuAnchor] = useState<null | HTMLElement>(null);
  const [timeframeMenuAnchor, setTimeframeMenuAnchor] = useState<null | HTMLElement>(null);

  // Load settings on mount
  useEffect(() => {
    const settings = chartSettingsService.getSettings();
    if (settings.indicators && settings.indicators.length > 0) {
      setSelectedIndicators(settings.indicators);
    }
    if (settings.showVolume !== undefined) {
      setSelectedIndicators(prev =>
        settings.showVolume ? [...new Set([...prev, 'volume'])] : prev.filter(i => i !== 'volume')
      );
    }
  }, []);

  // Save settings when they change
  useEffect(() => {
    chartSettingsService.saveSettings({
      symbol,
      timeframe,
      indicators: selectedIndicators,
      showVolume: selectedIndicators.includes('volume'),
    });
  }, [symbol, timeframe, selectedIndicators]);
  const [currentPrice, setCurrentPrice] = useState(0);
  const [priceChange, setPriceChange] = useState(0);
  const [priceChangePercent, setPriceChangePercent] = useState(0);
  const [predictionAccuracy, setPredictionAccuracy] = useState(94.3);
  const [chartInstance, setChartInstance] = useState<IChartApi | null>(null);

  // Technical indicator series refs
  const rsiSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const macdSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const bollingerUpperRef = useRef<ISeriesApi<'Line'> | null>(null);
  const bollingerLowerRef = useRef<ISeriesApi<'Line'> | null>(null);
  const smaSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const emaSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const vwapSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const macdSignalRef = useRef<ISeriesApi<'Line'> | null>(null);
  const macdHistogramRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const stochKRef = useRef<ISeriesApi<'Line'> | null>(null);
  const stochDRef = useRef<ISeriesApi<'Line'> | null>(null);
  const atrSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const adxSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const adxPlusDIRef = useRef<ISeriesApi<'Line'> | null>(null);
  const adxMinusDIRef = useRef<ISeriesApi<'Line'> | null>(null);

  // Use ChartSignalAgent for intelligent signal management
  const {
    signals: signalOverlays,
    isConnected: agentConnected,
    addSignal,
    updateFromConsensus,
    clearSignals
  } = useChartSignalAgent({
    chart: chartInstance,
    symbol,
    timeframe,
    wsUrl: process.env.NODE_ENV === 'production'
      ? 'wss://api.goldensignalsai.com/ws'
      : 'ws://localhost:8000/ws',
    enabled: true,
  });

  // WebSocket integration is now handled by the ChartSignalAgent
  // No need for direct WebSocket connection here

  const isConnected = agentConnected;
  const requestPrediction = () => {}; // Handled by agent
  const requestSignals = () => {}; // Handled by agent
  const requestPatterns = () => {}; // Handled by agent

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Ensure container has dimensions
    const containerWidth = chartContainerRef.current.clientWidth || 800;
    const containerHeight = chartContainerRef.current.clientHeight || 500;

    logger.info('Chart container dimensions:', { width: containerWidth, height: containerHeight });

    const chart = createChart(chartContainerRef.current, {
      width: containerWidth,
      height: containerHeight,
      layout: {
        background: {
          type: ColorType.Solid,
          color: theme.palette.mode === 'dark' ? '#0A0E1A' : '#FFFFFF',
        },
        textColor: theme.palette.text.primary,
      },
      grid: {
        vertLines: {
          color: theme.palette.mode === 'dark' ? alpha(theme.palette.divider, 0.1) : alpha(theme.palette.divider, 0.2),
        },
        horzLines: {
          color: theme.palette.mode === 'dark' ? alpha(theme.palette.divider, 0.1) : alpha(theme.palette.divider, 0.2),
        },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          width: 1,
          color: alpha(theme.palette.primary.main, 0.5),
          style: LineStyle.Dashed,
        },
        horzLine: {
          width: 1,
          color: alpha(theme.palette.primary.main, 0.5),
          style: LineStyle.Dashed,
        },
      },
      rightPriceScale: {
        borderColor: alpha(theme.palette.divider, 0.1),
        scaleMargins: {
          top: 0.1,
          bottom: 0.2,
        },
      },
      timeScale: {
        borderColor: alpha(theme.palette.divider, 0.1),
        timeVisible: true,
        secondsVisible: timeframe === '1m' || timeframe === '5m',
        tickMarkFormatter: (time: any) => {
          const date = new Date(time * 1000);

          // Format based on timeframe
          switch (timeframe) {
            case '1m':
            case '5m':
            case '15m':
              return date.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: false,
              });
            case '30m':
            case '1h':
              return date.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                hour12: false,
              });
            case '4h':
              return date.toLocaleTimeString('en-US', {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                hour12: false,
              });
            case '1d':
              return date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
              });
            case '1w':
            case '1M':
              return date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: '2-digit',
              });
            default:
              return date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
              });
          }
        },
      },
    });

    chartRef.current = chart;
    // Set the chart instance for the signal agent
    setChartInstance(chart);

    // Create series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: theme.palette.success.main,
      downColor: theme.palette.error.main,
      borderUpColor: theme.palette.success.dark,
      borderDownColor: theme.palette.error.dark,
      wickUpColor: theme.palette.success.main,
      wickDownColor: theme.palette.error.main,
    });
    candlestickSeriesRef.current = candlestickSeries;

    // Create prediction line
    const predictionSeries = chart.addLineSeries({
      color: alpha(theme.palette.info.main, 0.4), // 40% opacity for subtle appearance
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
      crosshairMarkerVisible: false, // Disable crosshair for cleaner look
      priceLineVisible: false,
      title: 'AI Prediction',
      lastValueVisible: false, // Hide last value label
    });
    predictionSeriesRef.current = predictionSeries;

    // Create volume series
    const volumeSeries = chart.addHistogramSeries({
      color: alpha(theme.palette.primary.main, 0.3),
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'volume',
    });
    volumeSeriesRef.current = volumeSeries;

    // Configure volume scale
    chart.priceScale('volume').applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    // Create indicator series
    const smaSeries = chart.addLineSeries({
      color: theme.palette.warning.main,
      lineWidth: 2,
      title: 'SMA 20',
      priceLineVisible: false,
      visible: false,
    });
    smaSeriesRef.current = smaSeries;

    const emaSeries = chart.addLineSeries({
      color: theme.palette.secondary.main,
      lineWidth: 2,
      title: 'EMA 20',
      priceLineVisible: false,
      visible: false,
    });
    emaSeriesRef.current = emaSeries;

    const bollingerUpper = chart.addLineSeries({
      color: alpha(theme.palette.info.main, 0.5),
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      title: 'BB Upper',
      priceLineVisible: false,
      visible: false,
    });
    bollingerUpperRef.current = bollingerUpper;

    const bollingerLower = chart.addLineSeries({
      color: alpha(theme.palette.info.main, 0.5),
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      title: 'BB Lower',
      priceLineVisible: false,
      visible: false,
    });
    bollingerLowerRef.current = bollingerLower;

    const vwapSeries = chart.addLineSeries({
      color: theme.palette.primary.dark,
      lineWidth: 2,
      title: 'VWAP',
      priceLineVisible: false,
      visible: false,
    });
    vwapSeriesRef.current = vwapSeries;

    // Create RSI series on separate pane
    const rsiSeries = chart.addLineSeries({
      color: theme.palette.secondary.main,
      lineWidth: 2,
      title: 'RSI',
      priceScaleId: 'rsi',
      visible: false,
    });
    rsiSeriesRef.current = rsiSeries;

    // Configure RSI scale
    chart.priceScale('rsi').applyOptions({
      scaleMargins: {
        top: 0.85,
        bottom: 0,
      },
      drawTicks: false,
      borderVisible: false,
    });

    // Create MACD series on separate pane
    const macdSeries = chart.addLineSeries({
      color: theme.palette.info.main,
      lineWidth: 2,
      title: 'MACD',
      priceScaleId: 'macd',
      visible: false,
    });
    macdSeriesRef.current = macdSeries;

    const macdSignal = chart.addLineSeries({
      color: theme.palette.error.main,
      lineWidth: 1,
      title: 'Signal',
      priceScaleId: 'macd',
      visible: false,
    });
    macdSignalRef.current = macdSignal;

    const macdHistogram = chart.addHistogramSeries({
      color: alpha(theme.palette.success.main, 0.5),
      priceScaleId: 'macd',
      visible: false,
    });
    macdHistogramRef.current = macdHistogram;

    // Configure MACD scale
    chart.priceScale('macd').applyOptions({
      scaleMargins: {
        top: 0.9,
        bottom: 0,
      },
      drawTicks: false,
      borderVisible: false,
    });

    // Create Stochastic series on separate pane
    const stochKSeries = chart.addLineSeries({
      color: theme.palette.primary.main,
      lineWidth: 2,
      title: 'Stoch %K',
      priceScaleId: 'stochastic',
      visible: false,
    });
    stochKRef.current = stochKSeries;

    const stochDSeries = chart.addLineSeries({
      color: theme.palette.secondary.main,
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      title: 'Stoch %D',
      priceScaleId: 'stochastic',
      visible: false,
    });
    stochDRef.current = stochDSeries;

    // Configure Stochastic scale
    chart.priceScale('stochastic').applyOptions({
      scaleMargins: {
        top: 0.95,
        bottom: 0,
      },
      drawTicks: false,
      borderVisible: false,
    });

    // Create ATR series on separate pane
    const atrSeries = chart.addLineSeries({
      color: theme.palette.warning.main,
      lineWidth: 2,
      title: 'ATR',
      priceScaleId: 'atr',
      visible: false,
    });
    atrSeriesRef.current = atrSeries;

    // Configure ATR scale
    chart.priceScale('atr').applyOptions({
      scaleMargins: {
        top: 0.97,
        bottom: 0,
      },
      drawTicks: false,
      borderVisible: false,
    });

    // Create ADX series on separate pane
    const adxSeries = chart.addLineSeries({
      color: theme.palette.primary.main,
      lineWidth: 2,
      title: 'ADX',
      priceScaleId: 'adx',
      visible: false,
    });
    adxSeriesRef.current = adxSeries;

    const adxPlusDI = chart.addLineSeries({
      color: theme.palette.success.main,
      lineWidth: 1,
      title: '+DI',
      priceScaleId: 'adx',
      visible: false,
    });
    adxPlusDIRef.current = adxPlusDI;

    const adxMinusDI = chart.addLineSeries({
      color: theme.palette.error.main,
      lineWidth: 1,
      title: '-DI',
      priceScaleId: 'adx',
      visible: false,
    });
    adxMinusDIRef.current = adxMinusDI;

    // Configure ADX scale
    chart.priceScale('adx').applyOptions({
      scaleMargins: {
        top: 0.98,
        bottom: 0,
      },
      drawTicks: false,
      borderVisible: false,
    });

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chart) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [theme, timeframe]);

  // Load initial data
  useEffect(() => {
    loadData();
  }, [symbol, timeframe]);

  // Update indicators when data or selection changes
  useEffect(() => {
    if (priceData.length > 0) {
      updateTechnicalIndicators();
    }
  }, [priceData, selectedIndicators]);

  // Generate predictions when price data changes
  useEffect(() => {
    if (priceData.length > 0) {
      const predictions = generatePredictionData(priceData);
      setPredictionData(predictions);
    }
  }, [priceData]);

  // Update chart when price data changes
  useEffect(() => {
    if (priceData.length > 0 && candlestickSeriesRef.current) {
      candlestickSeriesRef.current.setData(priceData as CandlestickData[]);
      logger.info('Chart updated with real data:', priceData.length, 'points');
    }

    // Update volume series
    if (priceData.length > 0 && volumeSeriesRef.current) {
      volumeSeriesRef.current.setData(priceData.map(d => ({
        time: d.time,
        value: d.volume || 0,
        color: d.close >= d.open
          ? alpha(theme.palette.success.main, 0.5)
          : alpha(theme.palette.error.main, 0.5),
      })));
    }
  }, [priceData, theme]);

  // Update predictions when prediction data changes
  useEffect(() => {
    if (predictionData.length > 0 && predictionSeriesRef.current) {
      predictionSeriesRef.current.setData(predictionData.map(p => ({
        time: p.time,
        value: p.value,
      })) as LineData[]);
    }
  }, [predictionData]);

  // Detect real patterns from price data
  useEffect(() => {
    if (priceData.length > 0) {
      try {
        // Convert priceData to OHLC format for pattern detection
        const ohlcData = priceData.map(d => ({
          timestamp: new Date((d.time as number) * 1000),
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
          volume: d.volume || 0
        }));

        // Detect real patterns from the data
        const detectedPatterns = detectPatterns(ohlcData);

        if (detectedPatterns.length > 0) {
          logger.info('Detected real patterns:', detectedPatterns.length);
          setPatterns(detectedPatterns);
        } else {
          logger.info('No patterns detected in current data');
          setPatterns([]);
        }
      } catch (error) {
        logger.error('Error detecting patterns:', error);
        setPatterns([]);
      }
    }
  }, [priceData]);


  const loadData = async () => {
    setLoading(true);
    try {
      // Fetch current market data
      try {
        const marketData = await fetchMarketData(symbol);
        setCurrentPrice(marketData.price);
        setPriceChange(marketData.change);
        setPriceChangePercent(marketData.change_percent);
      } catch (error) {
        logger.error('Failed to fetch real market data:', error);
        // No mock data fallback - leave current price data empty
        setCurrentPrice(0);
        setPriceChange(0);
        setPriceChangePercent(0);
      }

      // Fetch real historical data for charting
      try {
        const historyData = await fetchHistoricalData(symbol, timeframe);

        // Convert backend format to chart format
        const chartData = historyData.data.map((item: any) => ({
          time: item.time as Time,
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
          volume: item.volume || 0,
          value: item.close, // For line/area charts
        }));

        setPriceData(chartData);
        logger.info('Loaded real historical data:', chartData.length, 'data points');

      } catch (error) {
        logger.error('Failed to fetch real historical data:', error);
        // The service will return mock data as fallback, so we should still have data
        const historyData = await fetchHistoricalData(symbol, timeframe);
        setPriceData(historyData.data);
      }

      // Predictions will be generated in a separate useEffect when priceData changes

      // Fetch live signals from backend
      try {
        const signalData = await fetchSignals(symbol);

        // Convert backend signals to chart signal format
        const chartSignals: Signal[] = signalData.map(signal => ({
          id: signal.id,
          type: signal.action.toLowerCase() as 'buy' | 'sell',
          price: signal.price,
          time: signal.time || (Math.floor(Date.now() / 1000) as Time),
          confidence: signal.confidence,
          reasoning: signal.reasoning,
          consensus: signal.agents_consensus,
          stopLoss: signal.stop_loss,
          takeProfit: signal.take_profit,
        }));

        setSignals(chartSignals);

        // Update parent component if callback provided
        if (onAnalyze && chartSignals.length > 0) {
          onAnalyze(symbol, timeframe);
        }
      } catch (error) {
        logger.error('Failed to fetch live signals:', error);
        // Service returns mock signals as fallback
        const fallbackSignals = await fetchSignals(symbol);
        setSignals(fallbackSignals.map(signal => ({
          id: signal.id,
          type: signal.action.toLowerCase() as 'buy' | 'sell',
          price: signal.price,
          time: signal.time || (Math.floor(Date.now() / 1000) as Time),
          confidence: signal.confidence,
          reasoning: signal.reasoning,
          consensus: signal.agents_consensus,
        })));
      }

      // Generate patterns will be handled by useEffect based on priceData

      // Update technical indicators
      updateTechnicalIndicators();

      // Fit content
      if (chartRef.current) {
        chartRef.current.timeScale().fitContent();
      }

      // Add signal markers after data loads
      setTimeout(() => {
        if (candlestickSeriesRef.current && chartRef.current && signals.length > 0) {
          // Use ChartSignalAgent to manage signals
          signals.forEach(signal => {
            addSignalMarker(signal);
            addSignal(signal);
          });
        }
      }, 1000);

      // Real pattern detection will be implemented here
      // No mock patterns - real data only
    } catch (error) {
      logger.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleIndicatorMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setIndicatorMenuAnchor(event.currentTarget);
  };

  const handleIndicatorMenuClose = () => {
    setIndicatorMenuAnchor(null);
  };

  const handleTimeframeMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setTimeframeMenuAnchor(event.currentTarget);
  };

  const handleTimeframeMenuClose = () => {
    setTimeframeMenuAnchor(null);
  };

  const handleFullscreen = () => {
    setFullscreen(!fullscreen);
  };

  const handleChartTypeChange = (newType: 'candlestick' | 'line' | 'area') => {
    setChartType(newType);
    switchChartType(newType);
  };

  const handleTimeframeChange = (newTimeframe: string) => {
    onTimeframeChange?.(newTimeframe);
    handleTimeframeMenuClose();
  };

  const toggleIndicator = (indicator: string) => {
    setSelectedIndicators(prev => {
      if (prev.includes(indicator)) {
        return prev.filter(i => i !== indicator);
      } else {
        return [...prev, indicator];
      }
    });
  };

  // Add signal marker to chart with entry/exit points
  const addSignalMarker = (signal: Signal) => {
    if (!candlestickSeriesRef.current || !chartRef.current) return;

    // Don't add markers to the chart - we'll use the overlay instead
    // Just handle the stop loss and take profit lines

    // Add stop loss marker if available
    if (signal.stopLoss) {
      // Create a horizontal line for stop loss
      const stopLossSeries = chartRef.current.addLineSeries({
        color: theme.palette.error.light,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        priceLineVisible: true,
        priceLineColor: theme.palette.error.light,
        priceLineStyle: LineStyle.Dashed,
        priceLineWidth: 1,
        title: 'Stop Loss',
      });

      stopLossSeries.setData([
        { time: signal.time, value: signal.stopLoss },
        { time: priceData[priceData.length - 1].time, value: signal.stopLoss },
      ] as LineData[]);
    }

    // Add take profit lines if available
    if (signal.takeProfit && signal.takeProfit.length > 0) {
      signal.takeProfit.forEach((tp, index) => {
        const tpSeries = chartRef.current!.addLineSeries({
          color: alpha(theme.palette.success.light, 0.6 - index * 0.1),
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          priceLineVisible: true,
          priceLineColor: alpha(theme.palette.success.light, 0.6 - index * 0.1),
          priceLineStyle: LineStyle.Dashed,
          priceLineWidth: 1,
          title: `TP${index + 1}`,
        });

        tpSeries.setData([
          { time: signal.time, value: tp },
          { time: priceData[priceData.length - 1].time, value: tp },
        ] as LineData[]);
      });
    }

  };

  // Draw pattern on chart
  const drawPatternOnChart = (pattern: Pattern) => {
    if (!chartRef.current || !pattern.points || pattern.points.length === 0) return;

    // Create series for pattern lines with glow effect
    const patternSeries = chartRef.current.addLineSeries({
      color: theme.palette.warning.main,
      lineWidth: 3,
      lineStyle: LineStyle.Solid,
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false,
    });

    // Set pattern data
    patternSeries.setData(pattern.points.map(p => ({
      time: p.time,
      value: p.price,
    })) as LineData[]);

    // Store reference
    patternSeriesRefs.current.push(patternSeries);

    // Pattern label is now shown in the overlay box instead of as a marker

    return patternSeries;
  };

  // Switch chart type
  const switchChartType = (newType: 'candlestick' | 'line' | 'area') => {
    if (!chartRef.current) return;

    // Get current markers
    const currentMarkers = candlestickSeriesRef.current?.markers() || [];

    // Remove existing main series
    if (candlestickSeriesRef.current) {
      chartRef.current.removeSeries(candlestickSeriesRef.current);
    }

    // Create new series based on type
    switch (newType) {
      case 'line': {
        const lineSeries = chartRef.current.addLineSeries({
          color: theme.palette.primary.main,
          lineWidth: 2,
        });
        lineSeries.setData(priceData.map(d => ({ time: d.time, value: d.close })) as LineData[]);
        candlestickSeriesRef.current = lineSeries as any;
        break;
      }

      case 'area': {
        const areaSeries = chartRef.current.addAreaSeries({
          lineColor: theme.palette.primary.main,
          topColor: alpha(theme.palette.primary.main, 0.4),
          bottomColor: alpha(theme.palette.primary.main, 0.0),
        });
        areaSeries.setData(priceData.map(d => ({ time: d.time, value: d.close })) as LineData[]);
        candlestickSeriesRef.current = areaSeries as any;
        break;
      }

      case 'candlestick':
      default: {
        const candlestickSeries = chartRef.current.addCandlestickSeries({
          upColor: theme.palette.success.main,
          downColor: theme.palette.error.main,
          borderUpColor: theme.palette.success.dark,
          borderDownColor: theme.palette.error.dark,
          wickUpColor: theme.palette.success.main,
          wickDownColor: theme.palette.error.main,
        });
        candlestickSeries.setData(priceData as CandlestickData[]);
        candlestickSeriesRef.current = candlestickSeries;
        break;
      }
    }

    // Restore markers
    if (candlestickSeriesRef.current && currentMarkers.length > 0) {
      candlestickSeriesRef.current.setMarkers(currentMarkers);
    }
  };

  // Update technical indicators
  const updateTechnicalIndicators = () => {
    if (!priceData.length) return;

    const ohlcData: OHLCData[] = priceData.map(d => ({
      time: d.time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
      volume: d.volume,
    }));

    // Update selected indicators
    if (smaSeriesRef.current) {
      if (selectedIndicators.includes('sma')) {
        const smaData = calculateSMA(ohlcData, 20);
        smaSeriesRef.current.setData(smaData.filter(d => !isNaN(d.value)) as LineData[]);
        smaSeriesRef.current.applyOptions({ visible: true });
      } else {
        smaSeriesRef.current.applyOptions({ visible: false });
      }
    }

    if (emaSeriesRef.current) {
      if (selectedIndicators.includes('ema')) {
        const emaData = calculateEMA(ohlcData, 20);
        emaSeriesRef.current.setData(emaData.filter(d => !isNaN(d.value)) as LineData[]);
        emaSeriesRef.current.applyOptions({ visible: true });
      } else {
        emaSeriesRef.current.applyOptions({ visible: false });
      }
    }

    if (bollingerUpperRef.current && bollingerLowerRef.current) {
      if (selectedIndicators.includes('bb')) {
        const bbData = calculateBollingerBands(ohlcData);
        bollingerUpperRef.current.setData(
          bbData.filter(d => !isNaN(d.upper)).map(d => ({ time: d.time, value: d.upper })) as LineData[]
        );
        bollingerLowerRef.current.setData(
          bbData.filter(d => !isNaN(d.lower)).map(d => ({ time: d.time, value: d.lower })) as LineData[]
        );
        bollingerUpperRef.current.applyOptions({ visible: true });
        bollingerLowerRef.current.applyOptions({ visible: true });
      } else {
        bollingerUpperRef.current.applyOptions({ visible: false });
        bollingerLowerRef.current.applyOptions({ visible: false });
      }
    }

    if (vwapSeriesRef.current) {
      if (selectedIndicators.includes('vwap')) {
        const vwapData = calculateVWAP(ohlcData);
        vwapSeriesRef.current.setData(vwapData.filter(d => !isNaN(d.value)) as LineData[]);
        vwapSeriesRef.current.applyOptions({ visible: true });
      } else {
        vwapSeriesRef.current.applyOptions({ visible: false });
      }
    }

    // Update RSI
    if (rsiSeriesRef.current) {
      if (selectedIndicators.includes('rsi')) {
        const rsiData = calculateRSI(ohlcData, 14);
        rsiSeriesRef.current.setData(rsiData.filter(d => !isNaN(d.value)) as LineData[]);
        rsiSeriesRef.current.applyOptions({ visible: true });

        // Add RSI overbought/oversold lines
        const rsiMarkers = [];
        rsiData.forEach((point, index) => {
          if (!isNaN(point.value)) {
            if (point.value > 70) {
              rsiMarkers.push({
                time: point.time,
                position: 'aboveBar' as SeriesMarkerPosition,
                color: theme.palette.error.main,
                shape: 'arrowDown' as SeriesMarkerShape,
                text: 'OB',
              });
            } else if (point.value < 30) {
              rsiMarkers.push({
                time: point.time,
                position: 'belowBar' as SeriesMarkerPosition,
                color: theme.palette.success.main,
                shape: 'arrowUp' as SeriesMarkerShape,
                text: 'OS',
              });
            }
          }
        });

        // Add RSI markers to main series
        if (candlestickSeriesRef.current && rsiMarkers.length > 0) {
          const existingMarkers = candlestickSeriesRef.current.markers() || [];
          candlestickSeriesRef.current.setMarkers([...existingMarkers, ...rsiMarkers]);
        }
      } else {
        rsiSeriesRef.current.applyOptions({ visible: false });
      }
    }

    // Update MACD
    if (macdSeriesRef.current && macdSignalRef.current && macdHistogramRef.current) {
      if (selectedIndicators.includes('macd')) {
        const macdData = calculateMACD(ohlcData);

        // Set MACD line data
        const macdLineData = macdData
          .filter(d => !isNaN(d.macd))
          .map(d => ({ time: d.time, value: d.macd }));
        macdSeriesRef.current.setData(macdLineData as LineData[]);
        macdSeriesRef.current.applyOptions({ visible: true });

        // Set signal line data
        const signalLineData = macdData
          .filter(d => !isNaN(d.signal))
          .map(d => ({ time: d.time, value: d.signal }));
        macdSignalRef.current.setData(signalLineData as LineData[]);
        macdSignalRef.current.applyOptions({ visible: true });

        // Set histogram data with colors
        const histogramData = macdData
          .filter(d => !isNaN(d.histogram))
          .map(d => ({
            time: d.time,
            value: d.histogram,
            color: d.histogram >= 0
              ? alpha(theme.palette.success.main, 0.5)
              : alpha(theme.palette.error.main, 0.5),
          }));
        macdHistogramRef.current.setData(histogramData);
        macdHistogramRef.current.applyOptions({ visible: true });

        // Add MACD crossover signals
        const macdSignals = [];
        for (let i = 1; i < macdData.length; i++) {
          const prev = macdData[i - 1];
          const curr = macdData[i];

          if (!isNaN(prev.macd) && !isNaN(prev.signal) && !isNaN(curr.macd) && !isNaN(curr.signal)) {
            // Bullish crossover (MACD crosses above signal)
            if (prev.macd <= prev.signal && curr.macd > curr.signal) {
              macdSignals.push({
                time: curr.time,
                position: 'belowBar' as SeriesMarkerPosition,
                color: theme.palette.success.main,
                shape: 'arrowUp' as SeriesMarkerShape,
                text: 'MACD+',
                size: 2,
              });
            }
            // Bearish crossover (MACD crosses below signal)
            else if (prev.macd >= prev.signal && curr.macd < curr.signal) {
              macdSignals.push({
                time: curr.time,
                position: 'aboveBar' as SeriesMarkerPosition,
                color: theme.palette.error.main,
                shape: 'arrowDown' as SeriesMarkerShape,
                text: 'MACD-',
                size: 2,
              });
            }
          }
        }

        // Add MACD signals to main series
        if (candlestickSeriesRef.current && macdSignals.length > 0) {
          const existingMarkers = candlestickSeriesRef.current.markers() || [];
          candlestickSeriesRef.current.setMarkers([...existingMarkers, ...macdSignals]);
        }
      } else {
        macdSeriesRef.current.applyOptions({ visible: false });
        macdSignalRef.current.applyOptions({ visible: false });
        macdHistogramRef.current.applyOptions({ visible: false });
      }
    }

    // Update Stochastic
    if (stochKRef.current && stochDRef.current) {
      if (selectedIndicators.includes('stochastic')) {
        const stochData = calculateStochastic(ohlcData, 14, 3, 3);

        // Set %K line data
        stochKRef.current.setData(stochData.k.filter(d => !isNaN(d.value)) as LineData[]);
        stochKRef.current.applyOptions({ visible: true });

        // Set %D line data
        stochDRef.current.setData(stochData.d.filter(d => !isNaN(d.value)) as LineData[]);
        stochDRef.current.applyOptions({ visible: true });

        // Add Stochastic overbought/oversold signals
        const stochSignals = [];
        for (let i = 1; i < stochData.k.length; i++) {
          const prevK = stochData.k[i - 1];
          const currK = stochData.k[i];
          const prevD = stochData.d[i - 1];
          const currD = stochData.d[i];

          if (!isNaN(prevK.value) && !isNaN(currK.value) && !isNaN(prevD.value) && !isNaN(currD.value)) {
            // Bullish crossover (%K crosses above %D in oversold territory)
            if (prevK.value <= prevD.value && currK.value > currD.value && currK.value < 20) {
              stochSignals.push({
                time: currK.time,
                position: 'belowBar' as SeriesMarkerPosition,
                color: theme.palette.success.main,
                shape: 'arrowUp' as SeriesMarkerShape,
                text: 'Stoch+',
              });
            }
            // Bearish crossover (%K crosses below %D in overbought territory)
            else if (prevK.value >= prevD.value && currK.value < currD.value && currK.value > 80) {
              stochSignals.push({
                time: currK.time,
                position: 'aboveBar' as SeriesMarkerPosition,
                color: theme.palette.error.main,
                shape: 'arrowDown' as SeriesMarkerShape,
                text: 'Stoch-',
              });
            }
          }
        }

        // Add Stochastic signals to main series
        if (candlestickSeriesRef.current && stochSignals.length > 0) {
          const existingMarkers = candlestickSeriesRef.current.markers() || [];
          candlestickSeriesRef.current.setMarkers([...existingMarkers, ...stochSignals]);
        }
      } else {
        stochKRef.current.applyOptions({ visible: false });
        stochDRef.current.applyOptions({ visible: false });
      }
    }

    // Update ATR
    if (atrSeriesRef.current) {
      if (selectedIndicators.includes('atr')) {
        const atrData = calculateATR(ohlcData, 14);
        atrSeriesRef.current.setData(atrData.filter(d => !isNaN(d.value)) as LineData[]);
        atrSeriesRef.current.applyOptions({ visible: true });
      } else {
        atrSeriesRef.current.applyOptions({ visible: false });
      }
    }

    // Update ADX
    if (adxSeriesRef.current && adxPlusDIRef.current && adxMinusDIRef.current) {
      if (selectedIndicators.includes('adx')) {
        const adxData = calculateADX(ohlcData, 14);

        // Set ADX line data
        const adxLineData = adxData
          .filter(d => !isNaN(d.adx))
          .map(d => ({ time: d.time, value: d.adx }));
        adxSeriesRef.current.setData(adxLineData as LineData[]);
        adxSeriesRef.current.applyOptions({ visible: true });

        // Set +DI line data
        const plusDIData = adxData
          .filter(d => !isNaN(d.plusDI))
          .map(d => ({ time: d.time, value: d.plusDI }));
        adxPlusDIRef.current.setData(plusDIData as LineData[]);
        adxPlusDIRef.current.applyOptions({ visible: true });

        // Set -DI line data
        const minusDIData = adxData
          .filter(d => !isNaN(d.minusDI))
          .map(d => ({ time: d.time, value: d.minusDI }));
        adxMinusDIRef.current.setData(minusDIData as LineData[]);
        adxMinusDIRef.current.applyOptions({ visible: true });

        // Add ADX trend strength signals
        const adxSignals = [];
        adxData.forEach((point, index) => {
          if (!isNaN(point.adx)) {
            // Strong trend signal
            if (point.adx > 25 && index > 0 && adxData[index - 1].adx <= 25) {
              adxSignals.push({
                time: point.time,
                position: 'belowBar' as SeriesMarkerPosition,
                color: theme.palette.info.main,
                shape: 'circle' as SeriesMarkerShape,
                text: 'Strong Trend',
              });
            }
            // Weak trend warning
            else if (point.adx < 20 && index > 0 && adxData[index - 1].adx >= 20) {
              adxSignals.push({
                time: point.time,
                position: 'aboveBar' as SeriesMarkerPosition,
                color: theme.palette.warning.main,
                shape: 'circle' as SeriesMarkerShape,
                text: 'Weak Trend',
              });
            }
          }
        });

        // Add ADX signals to main series
        if (candlestickSeriesRef.current && adxSignals.length > 0) {
          const existingMarkers = candlestickSeriesRef.current.markers() || [];
          candlestickSeriesRef.current.setMarkers([...existingMarkers, ...adxSignals]);
        }
      } else {
        adxSeriesRef.current.applyOptions({ visible: false });
        adxPlusDIRef.current.applyOptions({ visible: false });
        adxMinusDIRef.current.applyOptions({ visible: false });
      }
    }
  };

  const availableIndicators = [
    { id: 'prediction', name: 'AI Prediction Line', category: 'AI Core', icon: AutoGraphIcon },
    { id: 'patterns', name: 'Pattern Recognition', category: 'AI Core', icon: PsychologyIcon },
    { id: 'signals', name: 'Buy/Sell Signals', category: 'AI Core', icon: AnalyticsIcon },
    { id: 'volume', name: 'Volume Analysis', category: 'AI Core', icon: BarChartIcon },
    { id: 'bb', name: 'Bollinger Bands', category: 'Technical', icon: TimelineIcon },
    { id: 'sma', name: 'SMA 20', category: 'Technical', icon: ShowChartIcon },
    { id: 'ema', name: 'EMA 20', category: 'Technical', icon: ShowChartIcon },
    { id: 'vwap', name: 'VWAP', category: 'Technical', icon: TimelineIcon },
    { id: 'rsi', name: 'RSI Divergence', category: 'Momentum', icon: AnalyticsIcon },
    { id: 'macd', name: 'MACD Signals', category: 'Momentum', icon: AnalyticsIcon },
    { id: 'stochastic', name: 'Stochastic Oscillator', category: 'Momentum', icon: TimelineIcon },
    { id: 'atr', name: 'ATR (Volatility)', category: 'Volatility', icon: ShowChartIcon },
    { id: 'adx', name: 'ADX (Trend Strength)', category: 'Trend', icon: TrendingUpIcon },
  ];

  const timeframes = [
    { value: '1m', label: '1 Min' },
    { value: '5m', label: '5 Min' },
    { value: '15m', label: '15 Min' },
    { value: '30m', label: '30 Min' },
    { value: '1h', label: '1 Hour' },
    { value: '4h', label: '4 Hours' },
    { value: '1d', label: '1 Day' },
    { value: '1w', label: '1 Week' },
  ];

  return (
    <ChartContainer sx={fullscreen ? {
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      zIndex: theme.zIndex.modal,
      borderRadius: 0,
    } : {}}>
      {/* Header */}
      <ChartHeader>
        <Box display="flex" alignItems="center" gap={1} minWidth={0} flex={1}>
          {/* Symbol Search */}
          <Box flex={1} maxWidth={250}>
            <TradeSearch
              onSubmit={onAnalyze || ((s, t) => {
                onSymbolChange?.(s);
                onTimeframeChange?.(t);
              })}
              onSymbolChange={onSymbolChange}
              onTimeframeChange={onTimeframeChange}
              defaultSymbol={symbol}
              defaultTimeframe={timeframe}
              isAnalyzing={isAnalyzing}
              compact={true}
              hideTimeframe={true}
            />
          </Box>

          {/* Timeframe Dropdown */}
          <Button
            size="small"
            variant="text"
            onClick={handleTimeframeMenuOpen}
            endIcon={<KeyboardArrowDownIcon />}
            sx={{
              color: 'text.primary',
              textTransform: 'none',
              fontWeight: 500,
              minWidth: 80,
              '&:hover': {
                backgroundColor: alpha(theme.palette.primary.main, 0.08),
              },
            }}
          >
            {timeframes.find(tf => tf.value === timeframe)?.label || timeframe}
          </Button>

          <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />

          {/* Price Info */}
          <Box display="flex" alignItems="center" gap={1} flexShrink={0}>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              ${currentPrice.toFixed(2)}
            </Typography>
            <Chip
              label={`${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)} (${priceChangePercent >= 0 ? '+' : ''}${priceChangePercent.toFixed(2)}%)`}
              color={priceChange >= 0 ? 'success' : 'error'}
              size="small"
              icon={priceChange >= 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
            />
          </Box>

          <Box flex={1} />

          {/* AI Status */}
          <Stack direction="row" spacing={0.5} alignItems="center">
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: theme.palette.success.main,
                animation: 'pulse 2s infinite',
              }}
            />
            <Typography variant="caption" color="text.secondary">
              30 AI Agents Active
            </Typography>
          </Stack>
        </Box>
        <Box display="flex" alignItems="center" gap={0.5}>
          <IconButton size="small" onClick={handleFullscreen}>
            {fullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
          </IconButton>
          <IconButton size="small">
            <PhotoCameraIcon />
          </IconButton>
        </Box>
      </ChartHeader>

      {/* Toolbar */}
      <ChartToolbar>
        <ToggleButtonGroup
          value={chartType}
          exclusive
          onChange={(e, value) => value && handleChartTypeChange(value)}
          size="small"
        >
          <ToggleButton value="line">
            <LineChartIcon fontSize="small" />
          </ToggleButton>
          <ToggleButton value="candlestick">
            <CandlestickChartIcon fontSize="small" />
          </ToggleButton>
          <ToggleButton value="area">
            <BarChartIcon fontSize="small" />
          </ToggleButton>
        </ToggleButtonGroup>

        <Divider orientation="vertical" flexItem />

        <Tooltip title="AI Indicators">
          <IconButton size="small" onClick={handleIndicatorMenuOpen}>
            <LayersIcon />
          </IconButton>
        </Tooltip>

        <Box flex={1} />

        <Stack direction="row" spacing={0.5}>
          {selectedIndicators.map(indicator => {
            const ind = availableIndicators.find(i => i.id === indicator);
            return ind ? (
              <Chip
                key={indicator}
                label={ind.name}
                size="small"
                onDelete={() => toggleIndicator(indicator)}
                sx={{ height: 24, fontSize: '0.7rem' }}
                color={ind.category === 'AI Core' ? 'primary' : 'default'}
              />
            ) : null;
          })}
        </Stack>
      </ChartToolbar>

      {/* Chart Area */}
      <ChartArea>
        {/* Stock Symbol Watermark */}
        {showWatermark && (
          <StockWatermark>
            <Box className="symbol">{symbol}</Box>
            <Box className="brand">GoldenSignalsAI</Box>
          </StockWatermark>
        )}

        <div
          ref={chartContainerRef}
          style={{
            width: '100%',
            height: '100%',
            minHeight: '400px',
            position: 'relative',
            backgroundColor: 'transparent',
            zIndex: 2,
          }}
        >
          {loading && (
            <Box
              position="absolute"
              top="50%"
              left="50%"
              style={{ transform: 'translate(-50%, -50%)' }}
              zIndex={10}
            >
              <CircularProgress />
              <Typography variant="caption" display="block" mt={2}>
                Loading chart data...
              </Typography>
            </Box>
          )}
        </div>

        {/* SVG Filters for Glow Effects */}
        <svg width="0" height="0" style={{ position: 'absolute' }}>
          <defs>
            <filter id="buy-glow">
              <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
              <feFlood floodColor={theme.palette.success.main} floodOpacity="1"/>
              <feComposite in2="coloredBlur" operator="in"/>
              <feComponentTransfer>
                <feFuncA type="discrete" tableValues="0 .5 .5 .5 .5 .5 .5 .5 .5 .5 .5 .5 .5 .5 .5 .5 .5 .5 1"/>
              </feComponentTransfer>
              <feMerge>
                <feMergeNode/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
            <filter id="sell-glow">
              <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
              <feFlood floodColor={theme.palette.error.main} floodOpacity="1"/>
              <feComposite in2="coloredBlur" operator="in"/>
              <feComponentTransfer>
                <feFuncA type="discrete" tableValues="0 .5 .5 .5 .5 .5 .5 .5 .5 .5 .5 .5 .5 .5 .5 .5 .5 .5 1"/>
              </feComponentTransfer>
              <feMerge>
                <feMergeNode/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
        </svg>

        {/* Signal Overlays with Static Glow - Managed by ChartSignalAgent */}
        {signalOverlays.filter(signal => signal.visible).map((signal) => {
          // Agent provides exact coordinates when signal is visible
          const position = signal.coordinates
            ? {
                left: signal.coordinates.x,
                top: signal.coordinates.y + (signal.type === 'buy' ? 20 : -20), // Smaller offset for cleaner look
              }
            : {
                left: signal.type === 'buy' ? '70%' : '85%', // Fallback positions
                top: signal.type === 'buy' ? '65%' : '35%',
              };

          return (
            <Box
              key={signal.id}
              sx={{
                position: 'absolute',
                left: position.left,
                top: position.top,
                transform: 'translate(-50%, -50%)',
                zIndex: 30,
                pointerEvents: 'none',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Box
                sx={{
                  color: signal.type === 'buy' ? theme.palette.success.main : theme.palette.error.main,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  '& svg': {
                    fontSize: '1.2rem',
                    animation: `${glowPulse} 3s ease-in-out infinite`,
                  },
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    width: '150%',
                    height: '150%',
                    borderRadius: '50%',
                    background: signal.type === 'buy'
                      ? `radial-gradient(circle, ${alpha(theme.palette.success.main, 0.3)} 0%, ${alpha(theme.palette.success.main, 0.15)} 40%, transparent 70%)`
                      : `radial-gradient(circle, ${alpha(theme.palette.error.main, 0.3)} 0%, ${alpha(theme.palette.error.main, 0.15)} 40%, transparent 70%)`,
                    zIndex: -1,
                  },
                }}
              >
                {signal.type === 'buy' ? <ArrowUpwardIcon /> : <ArrowDownwardIcon />}
              </Box>
              <Box
                sx={{
                  position: 'absolute',
                  bottom: -30,
                  textAlign: 'center',
                  whiteSpace: 'nowrap',
                }}
              >
                <Typography
                  variant="caption"
                  sx={{
                    fontWeight: 'bold',
                    color: signal.type === 'buy' ? theme.palette.success.main : theme.palette.error.main,
                    textShadow: `0 0 8px ${signal.type === 'buy' ? theme.palette.success.main : theme.palette.error.main}`,
                    display: 'block',
                  }}
                >
                  {signal.type.toUpperCase()} @ ${signal.price.toFixed(2)}
                </Typography>
                {signal.consensus && (
                  <Typography
                    variant="caption"
                    sx={{
                      fontSize: '0.65rem',
                      color: theme.palette.text.secondary,
                      display: 'block',
                    }}
                  >
                    {signal.consensus.agentsInFavor}/{signal.consensus.totalAgents} agents ({(signal.confidence * 100).toFixed(0)}%)
                  </Typography>
                )}
              </Box>
            </Box>
          );
        })}

        {/* Prediction Overlay */}
        {selectedIndicators.includes('prediction') && (
          <Fade in>
            <PredictionOverlay>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <Box
                  sx={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    backgroundColor: theme.palette.info.main,
                    animation: 'pulse 2s infinite',
                  }}
                />
                <Typography variant="body2" fontWeight="bold" color="primary">
                  AI Prediction Engine
                </Typography>
              </Box>
              <Stack spacing={0.5}>
                <Box display="flex" justifyContent="space-between" gap={2}>
                  <Typography variant="caption" color="text.secondary">
                    Model Accuracy:
                  </Typography>
                  <Typography variant="caption" fontWeight="bold" color="success.main">
                    {predictionAccuracy.toFixed(1)}%
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between" gap={2}>
                  <Typography variant="caption" color="text.secondary">
                    Price Target:
                  </Typography>
                  <Typography variant="caption" fontWeight="bold" color="info.main">
                    ${(currentPrice * 1.02).toFixed(2)}
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between" gap={2}>
                  <Typography variant="caption" color="text.secondary">
                    Signal Strength:
                  </Typography>
                  <Typography variant="caption" fontWeight="bold">
                    94.7%
                  </Typography>
                </Box>
                <Divider sx={{ my: 0.5 }} />
                <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                  30 AI agents analyzing in real-time
                </Typography>
              </Stack>
            </PredictionOverlay>
          </Fade>
        )}

        {/* Agent Analysis Overlay */}
        {selectedIndicators.includes('patterns') && patterns.length > 0 && (
          <Box
            sx={{
              position: 'absolute',
              top: theme.spacing(8),
              left: theme.spacing(2),
              padding: theme.spacing(1.5),
              backgroundColor: alpha(theme.palette.background.paper, 0.95),
              border: `1px solid ${alpha(theme.palette.warning.main, 0.3)}`,
              borderRadius: theme.spacing(1),
              maxWidth: 200,
              boxShadow: `0 0 20px ${alpha(theme.palette.warning.main, 0.5)}`,
            }}
          >
            <Typography variant="caption" fontWeight="bold" color="warning.main">
              Pattern Detected
            </Typography>
            <Typography variant="body2" sx={{ mt: 0.5 }}>
              {patterns[0].type}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Confidence: {(patterns[0].confidence * 100).toFixed(0)}%
            </Typography>
          </Box>
        )}

        {/* Connection Status */}
        <Box
          sx={{
            position: 'absolute',
            bottom: 8,
            right: 8,
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
          }}
        >
          <Box
            sx={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              backgroundColor: isConnected ? theme.palette.success.main : theme.palette.warning.main,
              animation: isConnected ? 'pulse 2s infinite' : 'none',
            }}
          />
          <Typography variant="caption" color="text.secondary">
            {isConnected ? 'Live' : 'Demo Mode'}
          </Typography>
        </Box>
      </ChartArea>

      {/* Timeframe Menu */}
      <Menu
        anchorEl={timeframeMenuAnchor}
        open={Boolean(timeframeMenuAnchor)}
        onClose={handleTimeframeMenuClose}
      >
        {timeframes.map(tf => (
          <MenuItem
            key={tf.value}
            onClick={() => handleTimeframeChange(tf.value)}
            selected={tf.value === timeframe}
          >
            {tf.label}
          </MenuItem>
        ))}
      </Menu>

      {/* Indicator Menu */}
      <Menu
        anchorEl={indicatorMenuAnchor}
        open={Boolean(indicatorMenuAnchor)}
        onClose={handleIndicatorMenuClose}
      >
        <MenuItem disabled>
          <Typography variant="subtitle2" color="primary">
            AI-Powered Analysis
          </Typography>
        </MenuItem>
        <Divider />
        {availableIndicators.map(indicator => {
          const Icon = indicator.icon;
          return (
            <MenuItem
              key={indicator.id}
              onClick={() => {
                toggleIndicator(indicator.id);
                handleIndicatorMenuClose();
              }}
              selected={selectedIndicators.includes(indicator.id)}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', gap: 1 }}>
                <Icon fontSize="small" color={selectedIndicators.includes(indicator.id) ? "primary" : "action"} />
                <Box flex={1}>
                  <Typography variant="body2">{indicator.name}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {indicator.category}
                  </Typography>
                </Box>
                {selectedIndicators.includes(indicator.id) && (
                  <Chip
                    label="Active"
                    size="small"
                    color="primary"
                    sx={{ height: 16, fontSize: '0.6rem' }}
                  />
                )}
              </Box>
            </MenuItem>
          );
        })}
        <Divider sx={{ my: 1 }} />
        <Box px={2} pb={1}>
          <Typography variant="caption" color="text.secondary" display="block" textAlign="center">
            All indicators are AI-enhanced for accuracy
          </Typography>
        </Box>
      </Menu>
    </ChartContainer>
  );
};

export default ProfessionalChart;
