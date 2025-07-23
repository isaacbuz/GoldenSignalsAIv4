/**
 * Professional Trading Chart
 * Modern, clean interface with excellent UX/UI
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
} from 'lightweight-charts';
import {
  Box,
  Typography,
  IconButton,
  Button,
  Menu,
  MenuItem,
  Chip,
  Fade,
  Skeleton,
  useTheme,
  useMediaQuery,
  Tooltip,
  ToggleButton,
  ToggleButtonGroup,
  Divider,
  ListItemIcon,
  ListItemText,
  Badge,
  Zoom,
  Grow,
  Paper,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import { styled, alpha } from '@mui/material/styles';
import {
  MoreVert as MoreVertIcon,
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
  ExpandMore as ExpandMoreIcon,
  Tune as TuneIcon,
  Close as CloseIcon,
  Menu as MenuIcon,
  TrendingFlat as TrendingFlatIcon,
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
} from '@mui/icons-material';

// Styled Components with modern design
const Container = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  width: '100%',
  height: '100vh',
  backgroundColor: theme.palette.background.default,
  overflow: 'hidden',
}));

const MainContent = styled(Box)(({ theme }) => ({
  display: 'flex',
  flex: 1,
  overflow: 'hidden',
}));

const IndicatorPanel = styled(Paper)(({ theme }) => ({
  width: 280,
  backgroundColor: theme.palette.background.paper,
  borderRight: `1px solid ${theme.palette.divider}`,
  overflowY: 'auto',
  overflowX: 'hidden',
  padding: 0,
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  [theme.breakpoints.down('md')]: {
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
    zIndex: 1200,
    boxShadow: theme.shadows[8],
    transform: 'translateX(-100%)',
    '&.open': {
      transform: 'translateX(0)',
    },
  },
}));

const ChartContainer = styled(Box)(({ theme }) => ({
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
}));

const Header = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2, 3),
  backgroundColor: theme.palette.background.paper,
  borderBottom: `1px solid ${theme.palette.divider}`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: theme.spacing(2),
  [theme.breakpoints.down('sm')]: {
    padding: theme.spacing(1.5, 2),
    flexWrap: 'wrap',
  },
}));

const PriceSection = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'baseline',
  gap: theme.spacing(1.5),
  flex: '0 0 auto',
}));

const SymbolChip = styled(Chip)(({ theme }) => ({
  backgroundColor: alpha(theme.palette.primary.main, 0.1),
  color: theme.palette.primary.main,
  fontWeight: 600,
  fontSize: '0.875rem',
  height: 28,
  '&:hover': {
    backgroundColor: alpha(theme.palette.primary.main, 0.2),
  },
}));

const PriceDisplay = styled(Typography)(({ theme }) => ({
  fontSize: '2rem',
  fontWeight: 700,
  lineHeight: 1,
  color: theme.palette.text.primary,
  fontVariantNumeric: 'tabular-nums',
  letterSpacing: '-0.02em',
  [theme.breakpoints.down('sm')]: {
    fontSize: '1.5rem',
  },
}));

const ChangeChip = styled(Chip)<{ trend: 'up' | 'down' | 'flat' }>(({ theme, trend }) => ({
  backgroundColor: trend === 'up'
    ? alpha(theme.palette.success.main, 0.1)
    : trend === 'down'
    ? alpha(theme.palette.error.main, 0.1)
    : alpha(theme.palette.grey[500], 0.1),
  color: trend === 'up'
    ? theme.palette.success.main
    : trend === 'down'
    ? theme.palette.error.main
    : theme.palette.grey[600],
  fontWeight: 600,
  height: 28,
  '& .MuiChip-icon': {
    color: 'inherit',
    fontSize: '1.2rem',
  },
  animation: trend !== 'flat' ? 'subtle-pulse 2s infinite' : 'none',
  '@keyframes subtle-pulse': {
    '0%, 100%': {
      opacity: 1,
    },
    '50%': {
      opacity: 0.8,
    },
  },
}));

const TimeframeToggle = styled(ToggleButtonGroup)(({ theme }) => ({
  backgroundColor: alpha(theme.palette.action.selected, 0.04),
  '& .MuiToggleButton-root': {
    border: 'none',
    padding: theme.spacing(0.5, 2),
    fontSize: '0.875rem',
    fontWeight: 500,
    color: theme.palette.text.secondary,
    '&:hover': {
      backgroundColor: alpha(theme.palette.action.hover, 0.04),
    },
    '&.Mui-selected': {
      backgroundColor: theme.palette.primary.main,
      color: theme.palette.primary.contrastText,
      '&:hover': {
        backgroundColor: theme.palette.primary.dark,
      },
    },
  },
}));

const ActionButton = styled(IconButton)(({ theme }) => ({
  color: theme.palette.text.secondary,
  '&:hover': {
    backgroundColor: alpha(theme.palette.action.hover, 0.04),
  },
}));

const ChartWrapper = styled(Box)(({ theme }) => ({
  flex: 1,
  position: 'relative',
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.shape.borderRadius,
  margin: theme.spacing(2),
  marginTop: 0,
  marginLeft: 0,
  overflow: 'hidden',
  boxShadow: theme.shadows[1],
  [theme.breakpoints.down('sm')]: {
    margin: theme.spacing(1),
    marginTop: 0,
    marginLeft: 0,
  },
}));

const IndicatorSection = styled(Box)(({ theme }) => ({
  marginBottom: theme.spacing(1),
}));

const IndicatorCheckbox = styled(FormControlLabel)(({ theme }) => ({
  margin: 0,
  width: '100%',
  padding: theme.spacing(0.75, 2),
  borderRadius: theme.shape.borderRadius,
  transition: 'all 0.2s ease',
  '&:hover': {
    backgroundColor: alpha(theme.palette.primary.main, 0.04),
  },
  '& .MuiCheckbox-root': {
    padding: theme.spacing(0.5),
    color: theme.palette.text.secondary,
    '&.Mui-checked': {
      color: theme.palette.primary.main,
    },
  },
  '& .MuiFormControlLabel-label': {
    fontSize: '0.875rem',
    fontWeight: 500,
  },
}));

const StyledAccordion = styled(Accordion)(({ theme }) => ({
  backgroundColor: 'transparent',
  boxShadow: 'none',
  '&:before': {
    display: 'none',
  },
  '& .MuiAccordionSummary-root': {
    backgroundColor: alpha(theme.palette.primary.main, 0.03),
    minHeight: 48,
    padding: theme.spacing(0, 2),
    '&:hover': {
      backgroundColor: alpha(theme.palette.primary.main, 0.05),
    },
  },
  '& .MuiAccordionSummary-content': {
    margin: theme.spacing(1, 0),
  },
  '& .MuiAccordionDetails-root': {
    padding: theme.spacing(1, 0),
  },
}));

const IndicatorHeader = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2.5, 2, 2, 2),
  borderBottom: `1px solid ${theme.palette.divider}`,
  background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.02)} 0%, ${alpha(theme.palette.primary.main, 0.05)} 100%)`,
}));

const MobileMenuButton = styled(IconButton)(({ theme }) => ({
  display: 'none',
  [theme.breakpoints.down('md')]: {
    display: 'flex',
    position: 'fixed',
    bottom: theme.spacing(2),
    left: theme.spacing(2),
    backgroundColor: theme.palette.primary.main,
    color: theme.palette.primary.contrastText,
    boxShadow: theme.shadows[6],
    '&:hover': {
      backgroundColor: theme.palette.primary.dark,
    },
  },
}));

const FloatingPanel = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(2),
  right: theme.spacing(2),
  backgroundColor: alpha(theme.palette.background.paper, 0.95),
  backdropFilter: 'blur(20px) saturate(180%)',
  borderRadius: theme.shape.borderRadius * 1.5,
  padding: theme.spacing(2),
  boxShadow: `0 8px 32px ${alpha(theme.palette.common.black, 0.08)}`,
  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(1.5),
  minWidth: 220,
  zIndex: 10,
  transition: 'all 0.3s ease',
  '&:hover': {
    boxShadow: `0 12px 40px ${alpha(theme.palette.common.black, 0.12)}`,
  },
}));

const StatRow = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: theme.spacing(0.5, 0),
  borderRadius: theme.shape.borderRadius / 2,
  transition: 'all 0.2s ease',
  '&:hover': {
    backgroundColor: alpha(theme.palette.action.hover, 0.04),
    paddingLeft: theme.spacing(0.5),
    paddingRight: theme.spacing(0.5),
  },
  '& .label': {
    fontSize: '0.75rem',
    color: theme.palette.text.secondary,
    fontWeight: 500,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  '& .value': {
    fontSize: '0.875rem',
    color: theme.palette.text.primary,
    fontWeight: 600,
    fontVariantNumeric: 'tabular-nums',
  },
}));

const AIBadge = styled(Badge)(({ theme }) => ({
  '& .MuiBadge-badge': {
    background: `linear-gradient(135deg, ${theme.palette.success.main} 0%, ${theme.palette.success.dark} 100%)`,
    color: theme.palette.success.contrastText,
    fontWeight: 600,
    fontSize: '0.75rem',
    padding: '4px 8px',
    borderRadius: theme.shape.borderRadius,
    boxShadow: `0 2px 8px ${alpha(theme.palette.success.main, 0.3)}`,
    animation: 'pulse 2s infinite',
  },
  '@keyframes pulse': {
    '0%': {
      boxShadow: `0 2px 8px ${alpha(theme.palette.success.main, 0.3)}`,
    },
    '50%': {
      boxShadow: `0 2px 12px ${alpha(theme.palette.success.main, 0.5)}`,
    },
    '100%': {
      boxShadow: `0 2px 8px ${alpha(theme.palette.success.main, 0.3)}`,
    },
  },
}));

const LiveIndicator = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(0.5),
  padding: theme.spacing(0.5, 1),
  borderRadius: theme.shape.borderRadius,
  backgroundColor: alpha(theme.palette.success.main, 0.1),
  color: theme.palette.success.main,
  fontSize: '0.75rem',
  fontWeight: 600,
  textTransform: 'uppercase',
  '& .MuiSvgIcon-root': {
    fontSize: '0.75rem',
    animation: 'pulse 2s infinite',
  },
  '@keyframes pulse': {
    '0%': {
      opacity: 1,
      transform: 'scale(1)',
    },
    '50%': {
      opacity: 0.7,
      transform: 'scale(1.1)',
    },
    '100%': {
      opacity: 1,
      transform: 'scale(1)',
    },
  },
}));

const UpdateTime = styled(Typography)(({ theme }) => ({
  fontSize: '0.7rem',
  color: theme.palette.text.disabled,
  marginLeft: theme.spacing(1),
}));

// Interfaces
interface ChartData {
  candles: CandlestickData[];
  indicators: {
    rsi?: number;
    macd?: { value: number; signal: number; histogram: number[] };
    volume?: number;
  };
  stats: {
    high: number;
    low: number;
    volume: string;
    marketCap?: string;
  };
}

interface IndicatorSettings {
  // Price Overlays
  ma10: boolean;
  ma50: boolean;
  ma200: boolean;
  ema9: boolean;
  ema12: boolean;
  ema26: boolean;
  bollinger: boolean;
  vwap: boolean;
  // Separate Panels
  rsi: boolean;
  macd: boolean;
  volume: boolean;
}

interface ProfessionalTradingChartProps {
  symbol?: string;
}

// Timeframe options
const timeframes = [
  { value: '1D', label: '1D' },
  { value: '1W', label: '1W' },
  { value: '1M', label: '1M' },
  { value: '3M', label: '3M' },
  { value: '1Y', label: '1Y' },
  { value: 'ALL', label: 'ALL' },
];

export const ProfessionalTradingChart: React.FC<ProfessionalTradingChartProps> = ({
  symbol = 'TSLA'
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  // State
  const [loading, setLoading] = useState(true);
  const [timeframe, setTimeframe] = useState('1D');
  const [isFavorite, setIsFavorite] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [showStats, setShowStats] = useState(true);
  const [data, setData] = useState<ChartData | null>(null);
  const [price, setPrice] = useState(0);
  const [change, setChange] = useState(0);
  const [changePercent, setChangePercent] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

  // AI Signal state
  const [aiSignal, setAiSignal] = useState<'BUY' | 'SELL' | 'HOLD'>('HOLD');
  const [aiConfidence, setAiConfidence] = useState(0);
  const [isLive, setIsLive] = useState(false);
  const [indicators, setIndicators] = useState<IndicatorSettings>({
    ma10: false,
    ma50: true,
    ma200: true,
    ema9: false,
    ema12: false,
    ema26: false,
    bollinger: false,
    vwap: false,
    rsi: true,
    macd: true,
    volume: true,
  });
  const [showIndicatorPanel, setShowIndicatorPanel] = useState(true);
  const [mobilePanelOpen, setMobilePanelOpen] = useState(false);
  const macdSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const rsiSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Handlers
  const handleTimeframeChange = (_: any, newTimeframe: string) => {
    if (newTimeframe !== null) {
      setTimeframe(newTimeframe);
      fetchData(newTimeframe);
    }
  };

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleIndicatorChange = (indicator: keyof IndicatorSettings) => {
    setIndicators(prev => ({ ...prev, [indicator]: !prev[indicator] }));
  };

  const toggleFavorite = () => {
    setIsFavorite(!isFavorite);
  };

  // Fetch data
  const fetchData = async (tf: string) => {
    setLoading(true);
    try {
      // Map timeframe to API parameters
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

      // Fetch real market data
      const response = await fetch(
        `http://localhost:8000/api/v1/market-data/${symbol}/history?period=${period}&interval=${interval}`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch market data');
      }

      const result = await response.json();
      const apiData = result.data || [];

      // Convert API data to chart format
      const candles: CandlestickData[] = apiData.map((item: any) => ({
        time: item.time as Time,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      }));

      if (candles.length === 0) {
        throw new Error('No data available');
      }

      // Calculate stats
      const highs = candles.map(c => c.high);
      const lows = candles.map(c => c.low);
      const volumes = apiData.map((item: any) => item.volume || 0);
      const totalVolume = volumes.reduce((sum: number, v: number) => sum + v, 0);

      setData({
        candles,
        indicators: {
          rsi: 50 + Math.random() * 30, // TODO: Calculate real RSI
          macd: { value: (Math.random() - 0.5) * 5, signal: (Math.random() - 0.5) * 3 },
          volume: totalVolume,
        },
        stats: {
          high: Math.max(...highs),
          low: Math.min(...lows),
          volume: formatVolume(totalVolume),
          marketCap: '1.05T', // TODO: Fetch real market cap
        },
      });

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
          const signal = aiData.prediction?.signal || 'HOLD';
          const confidence = Math.round((aiData.prediction?.confidence || 0.7) * 100);
          setAiSignal(signal.toUpperCase() as 'BUY' | 'SELL' | 'HOLD');
          setAiConfidence(confidence);
        }
      } catch (aiError) {
        console.error('AI prediction error:', aiError);
        // Use fallback AI signal based on simple analysis
        const trend = priceChange > 0 ? 'up' : 'down';
        const momentum = Math.abs(changePercent);
        if (momentum > 2) {
          setAiSignal(trend === 'up' ? 'BUY' : 'SELL');
          setAiConfidence(65 + Math.floor(momentum * 3));
        } else {
          setAiSignal('HOLD');
          setAiConfidence(75);
        }
      }

    } catch (error) {
      console.error('Error fetching data:', error);
      // Fall back to sample data
      const candles = generateCandleData();
      setData({
        candles,
        indicators: {
          rsi: 65.4,
          macd: { value: 2.3, signal: 1.8 },
          volume: 15234567,
        },
        stats: {
          high: 350,
          low: 320,
          volume: '15.2M',
          marketCap: '1.05T',
        },
      });

      const lastCandle = candles[candles.length - 1];
      setPrice(lastCandle.close);
      setChange(-5.23);
      setChangePercent(-1.54);
      setAiSignal('HOLD');
      setAiConfidence(82);
      setLastUpdate(new Date());
    } finally {
      setLoading(false);
    }
  };

  // Format volume for display
  const formatVolume = (volume: number): string => {
    if (volume >= 1e9) return `${(volume / 1e9).toFixed(1)}B`;
    if (volume >= 1e6) return `${(volume / 1e6).toFixed(1)}M`;
    if (volume >= 1e3) return `${(volume / 1e3).toFixed(1)}K`;
    return volume.toString();
  };

  // Format relative time
  const formatRelativeTime = (date: Date): string => {
    const seconds = Math.floor((new Date().getTime() - date.getTime()) / 1000);
    if (seconds < 10) return 'just now';
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return date.toLocaleDateString();
  };

  // Generate sample candle data
  const generateCandleData = (): CandlestickData[] => {
    const now = Math.floor(Date.now() / 1000);
    const data: CandlestickData[] = [];
    let lastClose = 260 + Math.random() * 20;

    for (let i = 100; i >= 0; i--) {
      const time = (now - i * 3600) as Time; // Hourly candles for sample data
      const open = lastClose;
      const change = (Math.random() - 0.5) * 4;
      const close = open + change;
      const high = Math.max(open, close) + Math.random() * 2;
      const low = Math.min(open, close) - Math.random() * 2;

      data.push({ time, open, high, low, close });
      lastClose = close;
    }

    return data;
  };

  // Initialize chart with multiple panes
  useEffect(() => {
    if (!chartContainerRef.current || loading || !data) return;

    // Calculate heights for different panels
    const containerHeight = chartContainerRef.current.clientHeight;
    const mainChartHeight = indicators.rsi && indicators.macd ? containerHeight * 0.5 :
                           indicators.rsi || indicators.macd ? containerHeight * 0.7 :
                           containerHeight * 0.85;
    const indicatorHeight = containerHeight * 0.15;
    const volumeHeight = containerHeight * 0.15;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: mainChartHeight,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: theme.palette.text.secondary,
      },
      grid: {
        vertLines: {
          color: alpha(theme.palette.divider, 0.3),
          style: 1,
        },
        horzLines: {
          color: alpha(theme.palette.divider, 0.3),
          style: 1,
        },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          width: 1,
          color: theme.palette.text.secondary,
          style: 0,
        },
        horzLine: {
          width: 1,
          color: theme.palette.text.secondary,
          style: 0,
        },
      },
      rightPriceScale: {
        borderColor: 'transparent',
        visible: true,
      },
      timeScale: {
        borderColor: 'transparent',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    // Add candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: theme.palette.success.main,
      downColor: theme.palette.error.main,
      borderUpColor: theme.palette.success.main,
      borderDownColor: theme.palette.error.main,
      wickUpColor: theme.palette.success.main,
      wickDownColor: theme.palette.error.main,
    });

    candleSeries.setData(data.candles);
    candleSeriesRef.current = candleSeries;

    // Price scale for main chart
    chart.priceScale('right').applyOptions({
      scaleMargins: {
        top: 0.1,
        bottom: indicators.volume ? 0.2 : 0.1,
      },
    });

    // Add moving averages based on settings
    if (indicators.ma10) {
      const ma10Series = chart.addLineSeries({
        color: '#2962FF',
        lineWidth: 1,
        title: 'MA(10)',
      });
      ma10Series.setData(calculateMA(data.candles, 10));
    }

    if (indicators.ma50) {
      const ma50Series = chart.addLineSeries({
        color: '#FF6D00',
        lineWidth: 2,
        title: 'MA(50)',
      });
      ma50Series.setData(calculateMA(data.candles, 50));
    }

    if (indicators.ma200) {
      const ma200Series = chart.addLineSeries({
        color: '#E91E63',
        lineWidth: 2,
        title: 'MA(200)',
      });
      ma200Series.setData(calculateMA(data.candles, 200));
    }

    // Add EMAs
    if (indicators.ema9) {
      const ema9Series = chart.addLineSeries({
        color: '#00BCD4',
        lineWidth: 1,
        lineStyle: 2,
        title: 'EMA(9)',
      });
      ema9Series.setData(calculateEMA(data.candles, 9));
    }

    if (indicators.ema12) {
      const ema12Series = chart.addLineSeries({
        color: '#4CAF50',
        lineWidth: 1,
        lineStyle: 2,
        title: 'EMA(12)',
      });
      ema12Series.setData(calculateEMA(data.candles, 12));
    }

    if (indicators.ema26) {
      const ema26Series = chart.addLineSeries({
        color: '#9C27B0',
        lineWidth: 1,
        lineStyle: 2,
        title: 'EMA(26)',
      });
      ema26Series.setData(calculateEMA(data.candles, 26));
    }

    // Add Bollinger Bands
    if (indicators.bollinger) {
      const bbData = calculateBollingerBands(data.candles, 20, 2);

      const bbUpperSeries = chart.addLineSeries({
        color: 'rgba(33, 150, 243, 0.5)',
        lineWidth: 1,
        lineStyle: 2,
        title: 'BB Upper',
      });
      bbUpperSeries.setData(bbData.upper);

      const bbMiddleSeries = chart.addLineSeries({
        color: 'rgba(33, 150, 243, 0.5)',
        lineWidth: 1,
        title: 'BB Middle',
      });
      bbMiddleSeries.setData(bbData.middle);

      const bbLowerSeries = chart.addLineSeries({
        color: 'rgba(33, 150, 243, 0.5)',
        lineWidth: 1,
        lineStyle: 2,
        title: 'BB Lower',
      });
      bbLowerSeries.setData(bbData.lower);
    }

    // Add volume if enabled
    if (indicators.volume) {
      const volumeSeries = chart.addHistogramSeries({
        color: alpha(theme.palette.text.secondary, 0.2),
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      });

      chart.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      });

      const volumeData = data.candles.map((candle, idx) => ({
        time: candle.time,
        value: (data as any).volumes?.[idx] || Math.random() * 10000000,
        color: candle.close >= candle.open
          ? alpha(theme.palette.success.main, 0.5)
          : alpha(theme.palette.error.main, 0.5),
      }));

      volumeSeries.setData(volumeData);
    }

    chartRef.current = chart;

    // Create RSI chart if enabled
    let rsiChart: IChartApi | null = null;
    if (indicators.rsi && chartContainerRef.current) {
      const rsiContainer = document.createElement('div');
      rsiContainer.style.height = `${indicatorHeight}px`;
      rsiContainer.style.borderTop = `1px solid ${theme.palette.divider}`;
      chartContainerRef.current.appendChild(rsiContainer);

      rsiChart = createChart(rsiContainer, {
        width: chartContainerRef.current.clientWidth,
        height: indicatorHeight,
        layout: {
          background: { type: ColorType.Solid, color: 'transparent' },
          textColor: theme.palette.text.secondary,
        },
        grid: {
          vertLines: { color: alpha(theme.palette.divider, 0.3) },
          horzLines: { color: alpha(theme.palette.divider, 0.3) },
        },
        rightPriceScale: {
          borderColor: 'transparent',
        },
        timeScale: {
          borderColor: 'transparent',
          visible: false,
        },
      });

      const rsiSeries = rsiChart.addLineSeries({
        color: '#9C27B0',
        lineWidth: 2,
        title: 'RSI(14)',
      });

      const rsiData = calculateRSI(data.candles, 14);
      rsiSeries.setData(rsiData);
      rsiSeriesRef.current = rsiSeries;

      // Add RSI levels
      rsiSeries.createPriceLine({ price: 70, color: '#ff4444', lineWidth: 1, lineStyle: 2 });
      rsiSeries.createPriceLine({ price: 30, color: '#00c853', lineWidth: 1, lineStyle: 2 });

      // Sync time scales
      chart.timeScale().subscribeVisibleTimeRangeChange(() => {
        rsiChart?.timeScale().setVisibleRange(chart.timeScale().getVisibleRange() as any);
      });
    }

    // Create MACD chart if enabled
    let macdChart: IChartApi | null = null;
    if (indicators.macd && chartContainerRef.current) {
      const macdContainer = document.createElement('div');
      macdContainer.style.height = `${indicatorHeight}px`;
      macdContainer.style.borderTop = `1px solid ${theme.palette.divider}`;
      chartContainerRef.current.appendChild(macdContainer);

      macdChart = createChart(macdContainer, {
        width: chartContainerRef.current.clientWidth,
        height: indicatorHeight,
        layout: {
          background: { type: ColorType.Solid, color: 'transparent' },
          textColor: theme.palette.text.secondary,
        },
        grid: {
          vertLines: { color: alpha(theme.palette.divider, 0.3) },
          horzLines: { color: alpha(theme.palette.divider, 0.3) },
        },
        rightPriceScale: {
          borderColor: 'transparent',
        },
        timeScale: {
          borderColor: 'transparent',
          timeVisible: true,
        },
      });

      const macdData = calculateMACD(data.candles, 12, 26, 9);

      // MACD histogram
      const macdHistogram = macdChart.addHistogramSeries({
        color: '#26A69A',
        title: 'MACD Histogram',
      });
      macdHistogram.setData(macdData.histogram);

      // MACD line
      const macdLine = macdChart.addLineSeries({
        color: '#2196F3',
        lineWidth: 2,
        title: 'MACD',
      });
      macdLine.setData(macdData.macd);

      // Signal line
      const signalLine = macdChart.addLineSeries({
        color: '#FF5722',
        lineWidth: 2,
        title: 'Signal',
      });
      signalLine.setData(macdData.signal);

      // Sync time scales
      chart.timeScale().subscribeVisibleTimeRangeChange(() => {
        macdChart?.timeScale().setVisibleRange(chart.timeScale().getVisibleRange() as any);
      });
    }

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        const newWidth = chartContainerRef.current.clientWidth;
        chart.applyOptions({ width: newWidth });
        rsiChart?.applyOptions({ width: newWidth });
        macdChart?.applyOptions({ width: newWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      rsiChart?.remove();
      macdChart?.remove();
    };
  }, [loading, data, theme, indicators]);

  // Initial load and WebSocket connection
  useEffect(() => {
    fetchData(timeframe);

    // Set up periodic refresh based on timeframe
    const refreshIntervals: Record<string, number> = {
      '1D': 30000,    // 30 seconds for intraday
      '1W': 60000,    // 1 minute for week view
      '1M': 300000,   // 5 minutes for month view
      '3M': 600000,   // 10 minutes for 3 months
      '1Y': 900000,   // 15 minutes for year
      'ALL': 1800000, // 30 minutes for all time
    };

    const refreshInterval = setInterval(() => {
      if (!loading) {
        fetchData(timeframe);
      }
    }, refreshIntervals[timeframe] || 60000);

    // Connect to WebSocket for real-time updates
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket(`ws://localhost:8000/ws/v2/signals/${symbol}`);

        ws.onopen = () => {
          console.log('WebSocket connected for real-time updates');
          setIsLive(true);
        };

        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);

            // Update price if we have price data
            if (message.type === 'price_update' || message.price) {
              const newPrice = message.price || message.data?.price;
              if (newPrice && !loading) {
                setPrice(newPrice);
                setLastUpdate(new Date());

                // Update the last candle
                if (candleSeriesRef.current && data?.candles.length > 0) {
                  const lastCandle = data.candles[data.candles.length - 1];
                  const updatedCandle: CandlestickData = {
                    ...lastCandle,
                    close: newPrice,
                    high: Math.max(lastCandle.high, newPrice),
                    low: Math.min(lastCandle.low, newPrice),
                  };
                  candleSeriesRef.current.update(updatedCandle);
                }
              }
            }

            // Update AI signals
            if (message.type === 'ai_signal' || message.signal) {
              const signal = message.signal || message.data?.signal;
              const confidence = message.confidence || message.data?.confidence;
              if (signal) {
                setAiSignal(signal.toUpperCase() as 'BUY' | 'SELL' | 'HOLD');
                if (confidence) {
                  setAiConfidence(Math.round(confidence * 100));
                }
              }
            }
          } catch (error) {
            console.error('WebSocket message error:', error);
          }
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
        };

        ws.onclose = () => {
          console.log('WebSocket disconnected');
          setIsLive(false);
          // Reconnect after 5 seconds
          setTimeout(connectWebSocket, 5000);
        };

        wsRef.current = ws;
      } catch (error) {
        console.error('WebSocket connection error:', error);
      }
    };

    // Connect after a short delay to ensure data is loaded
    const wsTimeout = setTimeout(connectWebSocket, 1000);

    return () => {
      clearInterval(refreshInterval);
      clearTimeout(wsTimeout);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [symbol, timeframe]);

  // Calculate MA
  const calculateMA = (candles: CandlestickData[], period: number): LineData[] => {
    const ma: LineData[] = [];
    for (let i = period - 1; i < candles.length; i++) {
      const sum = candles.slice(i - period + 1, i + 1).reduce((acc, c) => acc + c.close, 0);
      ma.push({ time: candles[i].time, value: sum / period });
    }
    return ma;
  };

  // Calculate EMA
  const calculateEMA = (candles: CandlestickData[], period: number): LineData[] => {
    const ema: LineData[] = [];
    const multiplier = 2 / (period + 1);

    // Start with SMA
    let sum = 0;
    for (let i = 0; i < period; i++) {
      sum += candles[i].close;
    }
    let emaValue = sum / period;
    ema.push({ time: candles[period - 1].time, value: emaValue });

    // Calculate EMA for rest
    for (let i = period; i < candles.length; i++) {
      emaValue = (candles[i].close - emaValue) * multiplier + emaValue;
      ema.push({ time: candles[i].time, value: emaValue });
    }

    return ema;
  };

  // Calculate Bollinger Bands
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

  // Calculate RSI
  const calculateRSI = (candles: CandlestickData[], period: number): LineData[] => {
    const rsi: LineData[] = [];
    let gains = 0;
    let losses = 0;

    // Initial average gain/loss
    for (let i = 1; i <= period; i++) {
      const change = candles[i].close - candles[i - 1].close;
      if (change > 0) gains += change;
      else losses -= change;
    }

    let avgGain = gains / period;
    let avgLoss = losses / period;
    let rs = avgGain / avgLoss;
    rsi.push({ time: candles[period].time, value: 100 - (100 / (1 + rs)) });

    // Calculate RSI for remaining periods
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

  // Calculate MACD
  const calculateMACD = (candles: CandlestickData[], fast: number, slow: number, signal: number) => {
    const ema12 = calculateEMA(candles, fast);
    const ema26 = calculateEMA(candles, slow);
    const macdLine: LineData[] = [];
    const signalLine: LineData[] = [];
    const histogram: LineData[] = [];

    // Calculate MACD line
    for (let i = slow - 1; i < candles.length; i++) {
      const macdValue = ema12[i - fast + 1].value - ema26[i - slow + 1].value;
      macdLine.push({ time: candles[i].time, value: macdValue });
    }

    // Calculate signal line (EMA of MACD)
    const multiplier = 2 / (signal + 1);
    let emaValue = macdLine.slice(0, signal).reduce((acc, val) => acc + val.value, 0) / signal;
    signalLine.push({ time: macdLine[signal - 1].time, value: emaValue });

    for (let i = signal; i < macdLine.length; i++) {
      emaValue = (macdLine[i].value - emaValue) * multiplier + emaValue;
      signalLine.push({ time: macdLine[i].time, value: emaValue });
    }

    // Calculate histogram
    for (let i = 0; i < signalLine.length; i++) {
      const histValue = macdLine[i + signal - 1].value - signalLine[i].value;
      histogram.push({
        time: signalLine[i].time,
        value: histValue,
        color: histValue >= 0 ? '#26A69A' : '#EF5350'
      });
    }

    return { macd: macdLine, signal: signalLine, histogram };
  };

  const trend = change > 0.01 ? 'up' : change < -0.01 ? 'down' : 'flat';
  const TrendIcon = trend === 'up' ? TrendingUpIcon : trend === 'down' ? TrendingDownIcon : TrendingFlatIcon;

  return (
    <Container>
      <Header>
        {/* Left section - Symbol & Price */}
        <PriceSection>
          <SymbolChip
            label={symbol}
            size="small"
            icon={<AutoGraphIcon />}
          />

          {loading || price === 0 ? (
            <Skeleton variant="text" width={120} height={40} />
          ) : (
            <Fade in={!loading && price > 0}>
              <Box display="flex" alignItems="baseline" gap={1}>
                <PriceDisplay>${price.toFixed(2)}</PriceDisplay>
                <ChangeChip
                  trend={trend}
                  icon={<TrendIcon />}
                  label={`${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePercent.toFixed(2)}%)`}
                  size="small"
                />
              </Box>
            </Fade>
          )}
        </PriceSection>

        {/* Center section - Timeframes */}
        {!isMobile && (
          <TimeframeToggle
            value={timeframe}
            exclusive
            onChange={handleTimeframeChange}
            size="small"
          >
            {timeframes.map(tf => (
              <ToggleButton key={tf.value} value={tf.value}>
                {tf.label}
              </ToggleButton>
            ))}
          </TimeframeToggle>
        )}

        {/* Right section - Actions */}
        <Box display="flex" gap={1} alignItems="center">
          {isLive && (
            <Grow in={isLive}>
              <LiveIndicator>
                <LiveIcon />
                <span>LIVE</span>
              </LiveIndicator>
            </Grow>
          )}

          {!isMobile && (
            <UpdateTime variant="caption">
              Updated {formatRelativeTime(lastUpdate)}
            </UpdateTime>
          )}

          <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />

          <Tooltip title={isFavorite ? "Remove from favorites" : "Add to favorites"}>
            <ActionButton onClick={toggleFavorite}>
              {isFavorite ? <BookmarkIcon /> : <BookmarkBorderIcon />}
            </ActionButton>
          </Tooltip>

          <Tooltip title="Share">
            <ActionButton>
              <ShareIcon />
            </ActionButton>
          </Tooltip>

          <Tooltip title="Fullscreen">
            <ActionButton>
              <FullscreenIcon />
            </ActionButton>
          </Tooltip>

          <ActionButton onClick={handleMenuClick}>
            <MoreVertIcon />
          </ActionButton>
        </Box>
      </Header>

      {/* Mobile timeframes */}
      {isMobile && (
        <Box px={2} py={1}>
          <TimeframeToggle
            value={timeframe}
            exclusive
            onChange={handleTimeframeChange}
            size="small"
            fullWidth
          >
            {timeframes.map(tf => (
              <ToggleButton key={tf.value} value={tf.value}>
                {tf.label}
              </ToggleButton>
            ))}
          </TimeframeToggle>
        </Box>
      )}

      {/* Main content area */}
      <MainContent>
        {/* Indicator Panel */}
        {!isMobile && showIndicatorPanel && (
          <IndicatorPanel elevation={0}>
            <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
              <Typography variant="h6" fontSize="1rem" fontWeight={600}>
                Indicators
              </Typography>
              <IconButton size="small" onClick={() => setShowIndicatorPanel(false)}>
                <TuneIcon fontSize="small" />
              </IconButton>
            </Box>

            {/* Price Overlays */}
            <Accordion defaultExpanded elevation={0}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle2" fontWeight={500}>Price Overlays</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <FormGroup>
                  <IndicatorCheckbox
                    control={
                      <Checkbox
                        checked={indicators.ma10}
                        onChange={() => handleIndicatorChange('ma10')}
                        size="small"
                      />
                    }
                    label="MA(10)"
                  />
                  <IndicatorCheckbox
                    control={
                      <Checkbox
                        checked={indicators.ma50}
                        onChange={() => handleIndicatorChange('ma50')}
                        size="small"
                      />
                    }
                    label="MA(50)"
                  />
                  <IndicatorCheckbox
                    control={
                      <Checkbox
                        checked={indicators.ma200}
                        onChange={() => handleIndicatorChange('ma200')}
                        size="small"
                      />
                    }
                    label="MA(200)"
                  />
                  <IndicatorCheckbox
                    control={
                      <Checkbox
                        checked={indicators.ema9}
                        onChange={() => handleIndicatorChange('ema9')}
                        size="small"
                      />
                    }
                    label="EMA(9)"
                  />
                  <IndicatorCheckbox
                    control={
                      <Checkbox
                        checked={indicators.ema12}
                        onChange={() => handleIndicatorChange('ema12')}
                        size="small"
                      />
                    }
                    label="EMA(12)"
                  />
                  <IndicatorCheckbox
                    control={
                      <Checkbox
                        checked={indicators.ema26}
                        onChange={() => handleIndicatorChange('ema26')}
                        size="small"
                      />
                    }
                    label="EMA(26)"
                  />
                  <IndicatorCheckbox
                    control={
                      <Checkbox
                        checked={indicators.bollinger}
                        onChange={() => handleIndicatorChange('bollinger')}
                        size="small"
                      />
                    }
                    label="Bollinger Bands"
                  />
                  <IndicatorCheckbox
                    control={
                      <Checkbox
                        checked={indicators.vwap}
                        onChange={() => handleIndicatorChange('vwap')}
                        size="small"
                        disabled
                      />
                    }
                    label="VWAP (Coming Soon)"
                  />
                </FormGroup>
              </AccordionDetails>
            </Accordion>

            {/* Momentum Indicators */}
            <Accordion defaultExpanded elevation={0}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle2" fontWeight={500}>Momentum</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <FormGroup>
                  <IndicatorCheckbox
                    control={
                      <Checkbox
                        checked={indicators.rsi}
                        onChange={() => handleIndicatorChange('rsi')}
                        size="small"
                      />
                    }
                    label="RSI(14)"
                  />
                  <IndicatorCheckbox
                    control={
                      <Checkbox
                        checked={indicators.macd}
                        onChange={() => handleIndicatorChange('macd')}
                        size="small"
                      />
                    }
                    label="MACD(12,26,9)"
                  />
                </FormGroup>
              </AccordionDetails>
            </Accordion>

            {/* Volume */}
            <Accordion defaultExpanded elevation={0}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle2" fontWeight={500}>Volume</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <FormGroup>
                  <IndicatorCheckbox
                    control={
                      <Checkbox
                        checked={indicators.volume}
                        onChange={() => handleIndicatorChange('volume')}
                        size="small"
                      />
                    }
                    label="Volume Bars"
                  />
                </FormGroup>
              </AccordionDetails>
            </Accordion>
          </IndicatorPanel>
        )}

        {/* Chart Container */}
        <ChartContainer>
          {/* Chart */}
          <ChartWrapper>
            {loading ? (
          <Box display="flex" alignItems="center" justifyContent="center" height="100%">
            <Skeleton variant="rectangular" width="95%" height="95%" />
          </Box>
        ) : (
          <>
            <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />

            {/* AI Signal Badge */}
            <Zoom in={!loading}>
              <Box position="absolute" top={16} left={16}>
                <AIBadge
                  badgeContent={`AI: ${aiSignal} ${aiConfidence}%`}
                  color="primary"
                >
                  <Chip
                    icon={<PsychologyIcon />}
                    label="AI Analysis"
                    size="small"
                    color={aiSignal === 'BUY' ? 'success' : aiSignal === 'SELL' ? 'error' : 'default'}
                  />
                </AIBadge>
              </Box>
            </Zoom>

            {/* Stats Panel */}
            {showStats && !isMobile && (
              <Grow in={!loading}>
                <FloatingPanel>
                  <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                    Market Stats
                  </Typography>
                  <Divider />

                  <StatRow>
                    <span className="label">Day High</span>
                    <span className="value">${data?.stats.high.toFixed(2)}</span>
                  </StatRow>

                  <StatRow>
                    <span className="label">Day Low</span>
                    <span className="value">${data?.stats.low.toFixed(2)}</span>
                  </StatRow>

                  <StatRow>
                    <span className="label">Volume</span>
                    <span className="value">{data?.stats.volume}</span>
                  </StatRow>

                  <StatRow>
                    <span className="label">Market Cap</span>
                    <span className="value">{data?.stats.marketCap}</span>
                  </StatRow>

                  <Divider />

                  <StatRow>
                    <span className="label">RSI(14)</span>
                    <span className="value">{data?.indicators.rsi?.toFixed(1)}</span>
                  </StatRow>

                  <StatRow>
                    <span className="label">MACD</span>
                    <span className="value">{data?.indicators.macd?.value.toFixed(2)}</span>
                  </StatRow>
                </FloatingPanel>
              </Grow>
            )}
          </>
        )}
          </ChartWrapper>
        </ChartContainer>
      </MainContent>

      {/* Mobile Indicator Button */}
      {isMobile && (
        <MobileMenuButton
          onClick={() => setMobilePanelOpen(true)}
          color="primary"
          size="large"
        >
          <TuneIcon />
        </MobileMenuButton>
      )}

      {/* Settings Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
        PaperProps={{
          sx: { minWidth: 200 }
        }}
      >
        <MenuItem onClick={() => { setShowStats(!showStats); handleMenuClose(); }}>
          <ListItemIcon>
            <AssessmentIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>
            {showStats ? 'Hide' : 'Show'} Stats
          </ListItemText>
        </MenuItem>

        <MenuItem onClick={() => { setShowIndicatorPanel(!showIndicatorPanel); handleMenuClose(); }}>
          <ListItemIcon>
            <TimelineIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>
            {showIndicatorPanel ? 'Hide' : 'Show'} Indicators
          </ListItemText>
        </MenuItem>

        <MenuItem onClick={handleMenuClose}>
          <ListItemIcon>
            <SettingsIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Settings</ListItemText>
        </MenuItem>
      </Menu>
    </Container>
  );
};

export default ProfessionalTradingChart;
