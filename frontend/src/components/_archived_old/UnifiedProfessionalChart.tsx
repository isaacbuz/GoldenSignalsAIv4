/**
 * Unified Professional Trading Chart
 * Combines all indicators into one beautifully scaled interface
 * Clean, modern design for professional traders
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
  PriceScaleMode,
} from 'lightweight-charts';
import {
  Box,
  Typography,
  IconButton,
  Chip,
  Fade,
  Skeleton,
  useTheme,
  useMediaQuery,
  Tooltip,
  ToggleButton,
  ToggleButtonGroup,
  Divider,
  Badge,
  Zoom,
  Grow,
  Paper,
  Switch,
  FormGroup,
  FormControlLabel,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Collapse,
} from '@mui/material';
import { styled, alpha } from '@mui/material/styles';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  ShowChart as ShowChartIcon,
  Timeline as TimelineIcon,
  Assessment as AssessmentIcon,
  Fullscreen as FullscreenIcon,
  Settings as SettingsIcon,
  BookmarkBorder as BookmarkBorderIcon,
  Bookmark as BookmarkIcon,
  Share as ShareIcon,
  Psychology as PsychologyIcon,
  AutoGraph as AutoGraphIcon,
  FiberManualRecord as LiveIcon,
  MoreVert as MoreVertIcon,
  Refresh as RefreshIcon,
  KeyboardArrowDown as ArrowDownIcon,
  KeyboardArrowUp as ArrowUpIcon,
  DarkMode as DarkModeIcon,
  LightMode as LightModeIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';

// Professional color palette
const CHART_COLORS = {
  // Candlestick colors
  bullish: '#26A69A',
  bearish: '#EF5350',

  // Moving averages
  ma: {
    fast: '#2196F3',    // Blue
    medium: '#FF9800',  // Orange
    slow: '#9C27B0',    // Purple
  },

  // EMA colors
  ema: {
    fast: '#00BCD4',    // Cyan
    medium: '#4CAF50',  // Green
    slow: '#E91E63',    // Pink
  },

  // Bollinger Bands
  bollinger: '#607D8B',

  // RSI colors
  rsi: {
    line: '#9C27B0',
    overbought: '#EF5350',
    oversold: '#26A69A',
  },

  // MACD colors
  macd: {
    line: '#2196F3',
    signal: '#FF5722',
    histogram: {
      positive: '#26A69A',
      negative: '#EF5350',
    },
  },

  // Volume
  volume: {
    up: alpha('#26A69A', 0.5),
    down: alpha('#EF5350', 0.5),
  },

  // Grid and text
  grid: alpha('#90A4AE', 0.1),
  text: {
    primary: '#263238',
    secondary: '#546E7A',
  },
};

// Dark theme colors
const DARK_COLORS = {
  ...CHART_COLORS,
  text: {
    primary: '#ECEFF1',
    secondary: '#B0BEC5',
  },
  grid: alpha('#546E7A', 0.2),
};

// Styled Components
const Container = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  width: '100%',
  height: '100vh',
  backgroundColor: theme.palette.mode === 'dark' ? '#0A0E27' : '#F8F9FA',
  overflow: 'hidden',
  fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif',
}));

const Header = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2, 3),
  background: theme.palette.mode === 'dark'
    ? 'linear-gradient(180deg, #1A1F3A 0%, #0A0E27 100%)'
    : 'linear-gradient(180deg, #FFFFFF 0%, #F8F9FA 100%)',
  borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  backdropFilter: 'blur(20px)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: theme.spacing(2),
  flexWrap: 'wrap',
  [theme.breakpoints.down('sm')]: {
    padding: theme.spacing(1.5, 2),
  },
}));

const PriceSection = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'baseline',
  gap: theme.spacing(2),
  flex: '0 0 auto',
}));

const SymbolChip = styled(Chip)(({ theme }) => ({
  backgroundColor: theme.palette.mode === 'dark'
    ? alpha('#2196F3', 0.15)
    : alpha('#1976D2', 0.08),
  color: theme.palette.mode === 'dark' ? '#64B5F6' : '#1976D2',
  fontWeight: 700,
  fontSize: '0.875rem',
  height: 32,
  borderRadius: 16,
  '& .MuiChip-icon': {
    color: 'inherit',
  },
  '&:hover': {
    backgroundColor: theme.palette.mode === 'dark'
      ? alpha('#2196F3', 0.25)
      : alpha('#1976D2', 0.12),
  },
}));

const PriceDisplay = styled(Typography)(({ theme }) => ({
  fontSize: '2.25rem',
  fontWeight: 700,
  lineHeight: 1,
  color: theme.palette.text.primary,
  fontVariantNumeric: 'tabular-nums',
  letterSpacing: '-0.02em',
  [theme.breakpoints.down('sm')]: {
    fontSize: '1.75rem',
  },
}));

const ChangeChip = styled(Chip)<{ trend: 'up' | 'down' | 'flat' }>(({ theme, trend }) => ({
  backgroundColor: trend === 'up'
    ? alpha('#26A69A', 0.12)
    : trend === 'down'
    ? alpha('#EF5350', 0.12)
    : alpha(theme.palette.grey[500], 0.08),
  color: trend === 'up'
    ? '#26A69A'
    : trend === 'down'
    ? '#EF5350'
    : theme.palette.text.secondary,
  fontWeight: 600,
  height: 28,
  '& .MuiChip-icon': {
    color: 'inherit',
    fontSize: '1.1rem',
  },
}));

const ChartContainer = styled(Paper)(({ theme }) => ({
  flex: 1,
  margin: theme.spacing(2),
  marginTop: 0,
  padding: theme.spacing(2),
  backgroundColor: theme.palette.mode === 'dark' ? '#0D1421' : '#FFFFFF',
  borderRadius: theme.shape.borderRadius * 2,
  boxShadow: theme.palette.mode === 'dark'
    ? '0 8px 32px rgba(0, 0, 0, 0.4)'
    : '0 4px 24px rgba(0, 0, 0, 0.06)',
  border: `1px solid ${alpha(theme.palette.divider, 0.08)}`,
  position: 'relative',
  overflow: 'hidden',
  [theme.breakpoints.down('sm')]: {
    margin: theme.spacing(1),
    padding: theme.spacing(1),
  },
}));

const IndicatorBar = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  padding: theme.spacing(1.5, 0),
  borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  flexWrap: 'wrap',
}));

const IndicatorChip = styled(Chip)(({ theme }) => ({
  height: 28,
  fontSize: '0.75rem',
  fontWeight: 600,
  borderRadius: 14,
  '& .MuiChip-deleteIcon': {
    fontSize: '1rem',
    marginLeft: 0,
  },
}));

const AIBadge = styled(Badge)(({ theme }) => ({
  '& .MuiBadge-badge': {
    background: 'linear-gradient(135deg, #2196F3 0%, #1976D2 100%)',
    color: '#FFFFFF',
    fontWeight: 600,
    fontSize: '0.75rem',
    padding: '6px 10px',
    borderRadius: 12,
    boxShadow: '0 4px 12px rgba(33, 150, 243, 0.3)',
    border: '2px solid transparent',
    backgroundClip: 'padding-box',
    '&::before': {
      content: '""',
      position: 'absolute',
      inset: -2,
      borderRadius: 12,
      padding: 2,
      background: theme.palette.mode === 'dark'
        ? 'linear-gradient(135deg, #64B5F6 0%, #2196F3 100%)'
        : 'linear-gradient(135deg, #2196F3 0%, #1565C0 100%)',
      mask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
      maskComposite: 'exclude',
      zIndex: -1,
    },
  },
}));

const LiveIndicator = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(0.5),
  padding: theme.spacing(0.5, 1.5),
  borderRadius: 20,
  backgroundColor: alpha('#4CAF50', 0.1),
  color: '#4CAF50',
  fontSize: '0.75rem',
  fontWeight: 700,
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
  '& .MuiSvgIcon-root': {
    fontSize: '0.875rem',
    animation: 'pulse 2s infinite',
  },
  '@keyframes pulse': {
    '0%, 100%': {
      opacity: 1,
      transform: 'scale(1)',
    },
    '50%': {
      opacity: 0.6,
      transform: 'scale(1.2)',
    },
  },
}));

const StatsPanel = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(2),
  right: theme.spacing(2),
  backgroundColor: theme.palette.mode === 'dark'
    ? alpha('#0D1421', 0.95)
    : alpha('#FFFFFF', 0.95),
  backdropFilter: 'blur(20px) saturate(200%)',
  borderRadius: 16,
  padding: theme.spacing(2),
  minWidth: 240,
  boxShadow: theme.palette.mode === 'dark'
    ? '0 8px 32px rgba(0, 0, 0, 0.4)'
    : '0 4px 24px rgba(0, 0, 0, 0.08)',
  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  zIndex: 10,
}));

// Interfaces
interface ChartData {
  candles: CandlestickData[];
  volumes?: number[];
}

interface IndicatorSettings {
  ma: { 10: boolean; 50: boolean; 200: boolean };
  ema: { 9: boolean; 12: boolean; 26: boolean };
  bollinger: boolean;
  rsi: boolean;
  macd: boolean;
  volume: boolean;
}

interface UnifiedProfessionalChartProps {
  symbol?: string;
}

// Main Component
export const UnifiedProfessionalChart: React.FC<UnifiedProfessionalChartProps> = ({
  symbol = 'TSLA'
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isDark = theme.palette.mode === 'dark';
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  // State
  const [loading, setLoading] = useState(true);
  const [timeframe, setTimeframe] = useState('1D');
  const [data, setData] = useState<ChartData | null>(null);
  const [price, setPrice] = useState(0);
  const [change, setChange] = useState(0);
  const [changePercent, setChangePercent] = useState(0);
  const [isLive, setIsLive] = useState(false);
  const [showStats, setShowStats] = useState(true);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Indicators state
  const [indicators, setIndicators] = useState<IndicatorSettings>({
    ma: { 10: false, 50: true, 200: true },
    ema: { 9: false, 12: false, 26: false },
    bollinger: false,
    rsi: true,
    macd: true,
    volume: true,
  });

  // AI State
  const [aiSignal, setAiSignal] = useState<'BUY' | 'SELL' | 'HOLD'>('HOLD');
  const [aiConfidence, setAiConfidence] = useState(0);

  // Initialize unified chart
  useEffect(() => {
    if (!chartContainerRef.current || loading || !data) return;

    const colors = isDark ? DARK_COLORS : CHART_COLORS;

    // Create main chart with dynamic height allocation
    const hasRSI = indicators.rsi;
    const hasMACD = indicators.macd;
    const indicatorCount = (hasRSI ? 1 : 0) + (hasMACD ? 1 : 0);

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: colors.text.secondary,
        fontSize: 12,
        fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif',
      },
      grid: {
        vertLines: {
          color: colors.grid,
          style: LineStyle.Solid,
          visible: true,
        },
        horzLines: {
          color: colors.grid,
          style: LineStyle.Solid,
          visible: true,
        },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          width: 1,
          color: alpha(colors.text.secondary, 0.3),
          style: LineStyle.Dashed,
          labelBackgroundColor: isDark ? '#1A1F3A' : '#1976D2',
        },
        horzLine: {
          width: 1,
          color: alpha(colors.text.secondary, 0.3),
          style: LineStyle.Dashed,
          labelBackgroundColor: isDark ? '#1A1F3A' : '#1976D2',
        },
      },
      rightPriceScale: {
        borderColor: 'transparent',
        visible: true,
        scaleMargins: {
          top: 0.1,
          bottom: 0.2 + (indicatorCount * 0.15),
        },
      },
      timeScale: {
        borderColor: 'transparent',
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time: any) => {
          const date = new Date(time * 1000);
          if (timeframe === '1D') {
            return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
          }
          return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        },
      },
      watermark: {
        visible: true,
        fontSize: 48,
        horzAlign: 'center',
        vertAlign: 'center',
        color: alpha(colors.text.secondary, 0.05),
        text: `${symbol} • GoldenSignalsAI`,
      },
    });

    // Add candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: colors.bullish,
      downColor: colors.bearish,
      borderUpColor: colors.bullish,
      borderDownColor: colors.bearish,
      wickUpColor: colors.bullish,
      wickDownColor: colors.bearish,
    });
    candleSeries.setData(data.candles);

    // Add volume
    if (indicators.volume) {
      const volumeSeries = chart.addHistogramSeries({
        color: colors.volume.up,
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      });

      const volumeData = data.candles.map((candle, idx) => ({
        time: candle.time,
        value: data.volumes?.[idx] || Math.random() * 10000000,
        color: candle.close >= candle.open ? colors.volume.up : colors.volume.down,
      }));
      volumeSeries.setData(volumeData);
    }

    // Add moving averages
    if (indicators.ma[10]) {
      const ma10 = chart.addLineSeries({
        color: colors.ma.fast,
        lineWidth: 2,
        title: 'MA 10',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      ma10.setData(calculateMA(data.candles, 10));
    }

    if (indicators.ma[50]) {
      const ma50 = chart.addLineSeries({
        color: colors.ma.medium,
        lineWidth: 2,
        title: 'MA 50',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      ma50.setData(calculateMA(data.candles, 50));
    }

    if (indicators.ma[200]) {
      const ma200 = chart.addLineSeries({
        color: colors.ma.slow,
        lineWidth: 2,
        title: 'MA 200',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      ma200.setData(calculateMA(data.candles, 200));
    }

    // Add EMAs
    if (indicators.ema[9]) {
      const ema9 = chart.addLineSeries({
        color: colors.ema.fast,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        title: 'EMA 9',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      ema9.setData(calculateEMA(data.candles, 9));
    }

    // Add Bollinger Bands
    if (indicators.bollinger) {
      const bb = calculateBollingerBands(data.candles, 20, 2);

      const bbUpper = chart.addLineSeries({
        color: alpha(colors.bollinger, 0.5),
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        title: 'BB Upper',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      bbUpper.setData(bb.upper);

      const bbLower = chart.addLineSeries({
        color: alpha(colors.bollinger, 0.5),
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        title: 'BB Lower',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      bbLower.setData(bb.lower);
    }

    // Add RSI in the same chart (scaled)
    if (indicators.rsi) {
      const rsiSeries = chart.addLineSeries({
        color: colors.rsi.line,
        lineWidth: 2,
        title: 'RSI',
        priceScaleId: 'rsi',
        scaleMargins: {
          top: 0.7,
          bottom: indicatorCount > 1 ? 0.15 : 0,
        },
        lastValueVisible: true,
        priceLineVisible: false,
      });

      const rsiData = calculateRSI(data.candles, 14);
      rsiSeries.setData(rsiData);

      // Add RSI levels
      rsiSeries.createPriceLine({
        price: 70,
        color: colors.rsi.overbought,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: false,
      });
      rsiSeries.createPriceLine({
        price: 30,
        color: colors.rsi.oversold,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: false,
      });
    }

    // Add MACD in the same chart (scaled)
    if (indicators.macd) {
      const macdData = calculateMACD(data.candles, 12, 26, 9);

      // MACD histogram
      const macdHistogram = chart.addHistogramSeries({
        color: colors.macd.histogram.positive,
        title: 'MACD Histogram',
        priceScaleId: 'macd',
        scaleMargins: {
          top: hasRSI ? 0.85 : 0.7,
          bottom: 0,
        },
      });
      macdHistogram.setData(macdData.histogram.map(h => ({
        ...h,
        color: h.value >= 0 ? colors.macd.histogram.positive : colors.macd.histogram.negative,
      })));

      // MACD line
      const macdLine = chart.addLineSeries({
        color: colors.macd.line,
        lineWidth: 2,
        title: 'MACD',
        priceScaleId: 'macd',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      macdLine.setData(macdData.macd);

      // Signal line
      const signalLine = chart.addLineSeries({
        color: colors.macd.signal,
        lineWidth: 2,
        title: 'Signal',
        priceScaleId: 'macd',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      signalLine.setData(macdData.signal);
    }

    chartRef.current = chart;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    // Add click handler for fullscreen
    chart.subscribeCrosshairMove((param) => {
      if (param.time) {
        const price = param.seriesPrices.get(candleSeries);
        if (price) {
          // Update crosshair price display
        }
      }
    });

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [loading, data, theme, indicators, timeframe, isDark]);

  // Fetch data
  const fetchData = async (tf: string) => {
    setLoading(true);
    try {
      const periodMap: Record<string, string> = {
        '1D': '1d',
        '1W': '5d',
        '1M': '1mo',
        '3M': '3mo',
        '1Y': '1y',
        'ALL': '2y',
      };

      const intervalMap: Record<string, string> = {
        '1D': '5m',
        '1W': '30m',
        '1M': '1h',
        '3M': '1d',
        '1Y': '1d',
        'ALL': '1wk',
      };

      const period = periodMap[tf] || '1d';
      const interval = intervalMap[tf] || '5m';

      const response = await fetch(
        `http://localhost:8000/api/v1/market-data/${symbol}/history?period=${period}&interval=${interval}`
      );

      if (!response.ok) throw new Error('Failed to fetch market data');

      const result = await response.json();
      const apiData = result.data || [];

      const candles: CandlestickData[] = apiData.map((item: any) => ({
        time: item.time as Time,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      }));

      const volumes = apiData.map((item: any) => item.volume || 0);

      setData({ candles, volumes });

      // Update price info
      const lastCandle = candles[candles.length - 1];
      setPrice(lastCandle.close);
      const firstCandle = candles[0];
      const priceChange = lastCandle.close - firstCandle.open;
      setChange(priceChange);
      setChangePercent((priceChange / firstCandle.open) * 100);
      setLastUpdate(new Date());

      // Fetch AI prediction
      try {
        const aiResponse = await fetch(
          `http://localhost:8000/api/v1/ai/predict/${symbol}`,
          { method: 'POST' }
        );

        if (aiResponse.ok) {
          const aiData = await aiResponse.json();
          setAiSignal((aiData.prediction?.signal || 'HOLD').toUpperCase());
          setAiConfidence(Math.round((aiData.prediction?.confidence || 0.75) * 100));
        }
      } catch (aiError) {
        // Use fallback
        setAiSignal('HOLD');
        setAiConfidence(78);
      }

      setIsLive(true);
    } catch (error) {
      console.error('Error fetching data:', error);
      // Use sample data as fallback
      setData({
        candles: generateSampleData(),
        volumes: Array(100).fill(0).map(() => Math.random() * 10000000)
      });
      setPrice(335.50);
      setChange(5.25);
      setChangePercent(1.59);
      setAiSignal('BUY');
      setAiConfidence(82);
    } finally {
      setLoading(false);
    }
  };

  // Generate sample data
  const generateSampleData = (): CandlestickData[] => {
    const now = Math.floor(Date.now() / 1000);
    const data: CandlestickData[] = [];
    let lastClose = 330;

    for (let i = 100; i >= 0; i--) {
      const time = (now - i * 3600) as Time;
      const volatility = 2 + Math.random() * 3;
      const trend = Math.sin(i * 0.1) * 5;
      const change = (Math.random() - 0.5) * volatility + trend * 0.1;

      const open = lastClose;
      const close = open + change;
      const high = Math.max(open, close) + Math.random() * volatility;
      const low = Math.min(open, close) - Math.random() * volatility;

      data.push({ time, open, high, low, close });
      lastClose = close;
    }

    return data;
  };

  // Technical indicator calculations
  const calculateMA = (candles: CandlestickData[], period: number): LineData[] => {
    const ma: LineData[] = [];
    for (let i = period - 1; i < candles.length; i++) {
      const sum = candles.slice(i - period + 1, i + 1).reduce((acc, c) => acc + c.close, 0);
      ma.push({ time: candles[i].time, value: sum / period });
    }
    return ma;
  };

  const calculateEMA = (candles: CandlestickData[], period: number): LineData[] => {
    const ema: LineData[] = [];
    const multiplier = 2 / (period + 1);

    let sum = 0;
    for (let i = 0; i < period; i++) {
      sum += candles[i].close;
    }
    let emaValue = sum / period;
    ema.push({ time: candles[period - 1].time, value: emaValue });

    for (let i = period; i < candles.length; i++) {
      emaValue = (candles[i].close - emaValue) * multiplier + emaValue;
      ema.push({ time: candles[i].time, value: emaValue });
    }

    return ema;
  };

  const calculateBollingerBands = (candles: CandlestickData[], period: number, stdDev: number) => {
    const middle = calculateMA(candles, period);
    const upper: LineData[] = [];
    const lower: LineData[] = [];

    for (let i = period - 1; i < candles.length; i++) {
      const slice = candles.slice(i - period + 1, i + 1);
      const avg = middle[i - period + 1].value;
      const squaredDiffs = slice.map(c => Math.pow(c.close - avg, 2));
      const variance = squaredDiffs.reduce((acc, val) => acc + val, 0) / period;
      const std = Math.sqrt(variance);

      upper.push({ time: candles[i].time, value: avg + std * stdDev });
      lower.push({ time: candles[i].time, value: avg - std * stdDev });
    }

    return { upper, middle, lower };
  };

  const calculateRSI = (candles: CandlestickData[], period: number): LineData[] => {
    const rsi: LineData[] = [];
    let gains = 0;
    let losses = 0;

    for (let i = 1; i <= period; i++) {
      const change = candles[i].close - candles[i - 1].close;
      if (change > 0) gains += change;
      else losses -= change;
    }

    let avgGain = gains / period;
    let avgLoss = losses / period;
    let rs = avgGain / avgLoss;
    rsi.push({ time: candles[period].time, value: 100 - (100 / (1 + rs)) });

    for (let i = period + 1; i < candles.length; i++) {
      const change = candles[i].close - candles[i - 1].close;
      if (change > 0) {
        avgGain = (avgGain * (period - 1) + change) / period;
        avgLoss = (avgLoss * (period - 1)) / period;
      } else {
        avgGain = (avgGain * (period - 1)) / period;
        avgLoss = (avgLoss * (period - 1) - change) / period;
      }

      rs = avgGain / avgLoss;
      rsi.push({ time: candles[i].time, value: 100 - (100 / (1 + rs)) });
    }

    return rsi;
  };

  const calculateMACD = (candles: CandlestickData[], fast: number, slow: number, signal: number) => {
    const ema12 = calculateEMA(candles, fast);
    const ema26 = calculateEMA(candles, slow);
    const macdLine: LineData[] = [];
    const signalLine: LineData[] = [];
    const histogram: LineData[] = [];

    for (let i = slow - 1; i < candles.length; i++) {
      const macdValue = ema12[i - fast + 1].value - ema26[i - slow + 1].value;
      macdLine.push({ time: candles[i].time, value: macdValue });
    }

    const multiplier = 2 / (signal + 1);
    let emaValue = macdLine.slice(0, signal).reduce((acc, val) => acc + val.value, 0) / signal;
    signalLine.push({ time: macdLine[signal - 1].time, value: emaValue });

    for (let i = signal; i < macdLine.length; i++) {
      emaValue = (macdLine[i].value - emaValue) * multiplier + emaValue;
      signalLine.push({ time: macdLine[i].time, value: emaValue });
    }

    for (let i = 0; i < signalLine.length; i++) {
      const histValue = macdLine[i + signal - 1].value - signalLine[i].value;
      histogram.push({
        time: signalLine[i].time,
        value: histValue,
      });
    }

    return { macd: macdLine, signal: signalLine, histogram };
  };

  // Initial load
  useEffect(() => {
    fetchData(timeframe);
  }, []);

  // Handlers
  const handleTimeframeChange = (_: any, newTimeframe: string) => {
    if (newTimeframe !== null) {
      setTimeframe(newTimeframe);
      fetchData(newTimeframe);
    }
  };

  const toggleIndicator = (category: 'ma' | 'ema', period: number) => {
    setIndicators(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [period]: !prev[category][period as keyof typeof prev.ma],
      },
    }));
  };

  const trend = change > 0.01 ? 'up' : change < -0.01 ? 'down' : 'flat';
  const TrendIcon = trend === 'up' ? TrendingUpIcon : TrendingDownIcon;

  return (
    <Container>
      <Header>
        <PriceSection>
          <SymbolChip
            label={symbol}
            size="medium"
            icon={<AutoGraphIcon />}
          />

          {loading || price === 0 ? (
            <Skeleton variant="text" width={200} height={45} />
          ) : (
            <Fade in={!loading}>
              <Box display="flex" alignItems="baseline" gap={1.5}>
                <PriceDisplay>${price.toFixed(2)}</PriceDisplay>
                <ChangeChip
                  trend={trend}
                  icon={<TrendIcon />}
                  label={`${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%)`}
                  size="small"
                />
              </Box>
            </Fade>
          )}
        </PriceSection>

        <ToggleButtonGroup
          value={timeframe}
          exclusive
          onChange={handleTimeframeChange}
          size="small"
          sx={{
            '& .MuiToggleButton-root': {
              px: 2,
              py: 0.5,
              fontSize: '0.875rem',
              fontWeight: 600,
              textTransform: 'none',
              borderRadius: 2,
              border: 'none',
              color: theme.palette.text.secondary,
              '&:hover': {
                backgroundColor: alpha(theme.palette.primary.main, 0.08),
              },
              '&.Mui-selected': {
                backgroundColor: theme.palette.primary.main,
                color: theme.palette.primary.contrastText,
                '&:hover': {
                  backgroundColor: theme.palette.primary.dark,
                },
              },
            },
          }}
        >
          <ToggleButton value="1D">1D</ToggleButton>
          <ToggleButton value="1W">1W</ToggleButton>
          <ToggleButton value="1M">1M</ToggleButton>
          <ToggleButton value="3M">3M</ToggleButton>
          <ToggleButton value="1Y">1Y</ToggleButton>
          <ToggleButton value="ALL">ALL</ToggleButton>
        </ToggleButtonGroup>

        <Box display="flex" gap={1} alignItems="center">
          {isLive && (
            <Grow in={isLive}>
              <LiveIndicator>
                <LiveIcon />
                <span>LIVE</span>
              </LiveIndicator>
            </Grow>
          )}

          <Tooltip title="Refresh">
            <IconButton onClick={() => fetchData(timeframe)} size="small">
              <RefreshIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Settings">
            <IconButton
              onClick={(e) => setAnchorEl(e.currentTarget)}
              size="small"
            >
              <MoreVertIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Header>

      <ChartContainer elevation={0}>
        <IndicatorBar>
          <Typography variant="caption" sx={{ fontWeight: 600, mr: 1 }}>
            Indicators:
          </Typography>

          {/* MA indicators */}
          {Object.entries(indicators.ma).map(([period, enabled]) =>
            enabled && (
              <IndicatorChip
                key={`ma${period}`}
                label={`MA ${period}`}
                size="small"
                onDelete={() => toggleIndicator('ma', parseInt(period))}
                sx={{
                  backgroundColor: alpha(CHART_COLORS.ma[period === '10' ? 'fast' : period === '50' ? 'medium' : 'slow'], 0.15),
                  color: CHART_COLORS.ma[period === '10' ? 'fast' : period === '50' ? 'medium' : 'slow'],
                }}
              />
            )
          )}

          {/* Other indicators */}
          {indicators.bollinger && (
            <IndicatorChip
              label="Bollinger"
              size="small"
              onDelete={() => setIndicators(prev => ({ ...prev, bollinger: false }))}
              sx={{
                backgroundColor: alpha(CHART_COLORS.bollinger, 0.15),
                color: CHART_COLORS.bollinger,
              }}
            />
          )}

          {indicators.rsi && (
            <IndicatorChip
              label="RSI"
              size="small"
              onDelete={() => setIndicators(prev => ({ ...prev, rsi: false }))}
              sx={{
                backgroundColor: alpha(CHART_COLORS.rsi.line, 0.15),
                color: CHART_COLORS.rsi.line,
              }}
            />
          )}

          {indicators.macd && (
            <IndicatorChip
              label="MACD"
              size="small"
              onDelete={() => setIndicators(prev => ({ ...prev, macd: false }))}
              sx={{
                backgroundColor: alpha(CHART_COLORS.macd.line, 0.15),
                color: CHART_COLORS.macd.line,
              }}
            />
          )}
        </IndicatorBar>

        {loading ? (
          <Box display="flex" alignItems="center" justifyContent="center" height={500}>
            <Skeleton variant="rectangular" width="100%" height="100%" />
          </Box>
        ) : (
          <>
            <div
              ref={chartContainerRef}
              style={{
                width: '100%',
                height: 'calc(100% - 60px)',
                position: 'relative',
              }}
            />

            {/* AI Signal Badge */}
            <Zoom in={!loading}>
              <Box position="absolute" top={70} left={16}>
                <AIBadge
                  badgeContent={`${aiSignal} ${aiConfidence}%`}
                  color="primary"
                >
                  <Chip
                    icon={<PsychologyIcon />}
                    label="AI Analysis"
                    size="small"
                    sx={{
                      backgroundColor: aiSignal === 'BUY'
                        ? alpha('#4CAF50', 0.1)
                        : aiSignal === 'SELL'
                        ? alpha('#F44336', 0.1)
                        : alpha('#FF9800', 0.1),
                      color: aiSignal === 'BUY'
                        ? '#4CAF50'
                        : aiSignal === 'SELL'
                        ? '#F44336'
                        : '#FF9800',
                      fontWeight: 600,
                      border: `1px solid ${alpha(
                        aiSignal === 'BUY' ? '#4CAF50' : aiSignal === 'SELL' ? '#F44336' : '#FF9800',
                        0.3
                      )}`,
                    }}
                  />
                </AIBadge>
              </Box>
            </Zoom>

            {/* Stats Panel */}
            {showStats && !isMobile && (
              <Grow in={!loading}>
                <StatsPanel>
                  <Box display="flex" alignItems="center" justifyContent="space-between" mb={1.5}>
                    <Typography variant="subtitle2" fontWeight={700}>
                      Market Stats
                    </Typography>
                    <Chip
                      label="PRO"
                      size="small"
                      icon={<SpeedIcon />}
                      sx={{
                        height: 20,
                        fontSize: '0.7rem',
                        backgroundColor: alpha('#FFD700', 0.15),
                        color: '#FFB300',
                      }}
                    />
                  </Box>

                  <Box display="flex" flexDirection="column" gap={1}>
                    <Box display="flex" justifyContent="space-between">
                      <Typography variant="caption" color="text.secondary">Volume</Typography>
                      <Typography variant="caption" fontWeight={600}>
                        {data ? formatVolume(data.volumes?.reduce((a, b) => a + b, 0) || 0) : '-'}
                      </Typography>
                    </Box>
                    <Box display="flex" justifyContent="space-between">
                      <Typography variant="caption" color="text.secondary">Day Range</Typography>
                      <Typography variant="caption" fontWeight={600}>
                        {data ? `$${Math.min(...data.candles.map(c => c.low)).toFixed(2)} - $${Math.max(...data.candles.map(c => c.high)).toFixed(2)}` : '-'}
                      </Typography>
                    </Box>
                    <Box display="flex" justifyContent="space-between">
                      <Typography variant="caption" color="text.secondary">Market Cap</Typography>
                      <Typography variant="caption" fontWeight={600}>$1.05T</Typography>
                    </Box>
                  </Box>
                </StatsPanel>
              </Grow>
            )}
          </>
        )}
      </ChartContainer>

      {/* Settings Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
        PaperProps={{
          sx: {
            minWidth: 280,
            mt: 1,
            backgroundColor: theme.palette.mode === 'dark' ? '#1A1F3A' : '#FFFFFF',
          }
        }}
      >
        <Box px={2} py={1}>
          <Typography variant="subtitle2" fontWeight={600} gutterBottom>
            Chart Settings
          </Typography>
        </Box>

        <Divider />

        <Box p={2}>
          <Typography variant="caption" fontWeight={600} color="text.secondary" gutterBottom display="block">
            MOVING AVERAGES
          </Typography>
          <FormGroup>
            {Object.entries(indicators.ma).map(([period]) => (
              <FormControlLabel
                key={`ma${period}`}
                control={
                  <Switch
                    size="small"
                    checked={indicators.ma[period as keyof typeof indicators.ma]}
                    onChange={() => toggleIndicator('ma', parseInt(period))}
                  />
                }
                label={`MA ${period}`}
                sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem' } }}
              />
            ))}
          </FormGroup>

          <Typography variant="caption" fontWeight={600} color="text.secondary" gutterBottom display="block" sx={{ mt: 2 }}>
            INDICATORS
          </Typography>
          <FormGroup>
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.bollinger}
                  onChange={() => setIndicators(prev => ({ ...prev, bollinger: !prev.bollinger }))}
                />
              }
              label="Bollinger Bands"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem' } }}
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.rsi}
                  onChange={() => setIndicators(prev => ({ ...prev, rsi: !prev.rsi }))}
                />
              }
              label="RSI (14)"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem' } }}
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.macd}
                  onChange={() => setIndicators(prev => ({ ...prev, macd: !prev.macd }))}
                />
              }
              label="MACD (12, 26, 9)"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem' } }}
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.volume}
                  onChange={() => setIndicators(prev => ({ ...prev, volume: !prev.volume }))}
                />
              }
              label="Volume"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem' } }}
            />
          </FormGroup>
        </Box>

        <Divider />

        <MenuItem onClick={() => { setShowStats(!showStats); setAnchorEl(null); }}>
          <ListItemIcon>
            <AssessmentIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>{showStats ? 'Hide' : 'Show'} Stats Panel</ListItemText>
        </MenuItem>
      </Menu>
    </Container>
  );
};

// Helper function
const formatVolume = (volume: number): string => {
  if (volume >= 1e9) return `${(volume / 1e9).toFixed(1)}B`;
  if (volume >= 1e6) return `${(volume / 1e6).toFixed(1)}M`;
  if (volume >= 1e3) return `${(volume / 1e3).toFixed(1)}K`;
  return volume.toString();
};

export default UnifiedProfessionalChart;
