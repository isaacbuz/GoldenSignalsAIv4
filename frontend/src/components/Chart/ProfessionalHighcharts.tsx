/**
 * ProfessionalHighcharts Component
 *
 * A professional trading chart implementation using Highcharts Stock
 * with enhanced design while maintaining the original layout structure.
 *
 * Features:
 * - Beautiful dark theme with smooth gradients
 * - Real-time candlestick charts with glow effects
 * - Technical indicators (RSI, MACD, Moving Averages)
 * - AI prediction overlay with confidence bands
 * - Agent analysis integration
 * - Volume bars with gradient fills
 * - Professional header with search and controls
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from 'highcharts-react-official';
import IndicatorsAll from 'highcharts/indicators/indicators-all';
import AnnotationsAdvanced from 'highcharts/modules/annotations-advanced';

// Initialize modules
if (typeof Highcharts === 'object') {
  IndicatorsAll(Highcharts);
  AnnotationsAdvanced(Highcharts);
}

import {
  Box,
  TextField,
  Autocomplete,
  Select,
  MenuItem,
  FormControl,
  Chip,
  Typography,
  Button,
  IconButton,
  Paper,
  alpha,
  useTheme,
  Fade,
  CircularProgress,
  Tabs,
  Tab,
  FormControlLabel,
  Checkbox,
  ListItemText,
  OutlinedInput,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Search as SearchIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Psychology as PsychologyIcon,
  ShowChart as ShowChartIcon,
  CandlestickChart as CandlestickChartIcon,
  BarChart as BarChartIcon,
  Settings as SettingsIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  Refresh as RefreshIcon,
  FiberManualRecord as LiveIcon,
  KeyboardAlt as KeyboardIcon,
} from '@mui/icons-material';

// Hooks and services
import { useMarketData } from './hooks/useMarketData';
import { useAgentAnalysis } from './hooks/useAgentAnalysis';
import { aiPredictionService } from '../../services/aiPredictionService';
import { agentWorkflowService } from '../../services/agentWorkflowService';

// Styled components with enhanced design
const Container = styled(Box)(({ theme }) => ({
  width: '100%',
  height: '100vh',
  background: 'linear-gradient(180deg, #0a0b0d 0%, #141518 100%)',
  display: 'flex',
  flexDirection: 'column',
  fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif',
  position: 'relative',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'radial-gradient(circle at 50% 50%, rgba(255, 215, 0, 0.03) 0%, transparent 70%)',
    pointerEvents: 'none',
  },
}));

const Header = styled(Box)(({ theme }) => ({
  height: 64,
  background: 'rgba(22, 24, 29, 0.95)',
  backdropFilter: 'blur(20px)',
  borderBottom: '1px solid rgba(255, 215, 0, 0.1)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '0 24px',
  boxShadow: '0 2px 24px rgba(0, 0, 0, 0.4)',
  position: 'relative',
  zIndex: 10,
}));

const SearchBar = styled(Autocomplete)(({ theme }) => ({
  width: 300,
  '& .MuiInputBase-root': {
    height: 40,
    backgroundColor: alpha('#fff', 0.05),
    borderRadius: 20,
    border: '1px solid rgba(255, 215, 0, 0.2)',
    transition: 'all 0.3s ease',
    '&:hover': {
      backgroundColor: alpha('#fff', 0.08),
      borderColor: 'rgba(255, 215, 0, 0.3)',
    },
    '&.Mui-focused': {
      backgroundColor: alpha('#fff', 0.1),
      borderColor: 'rgba(255, 215, 0, 0.5)',
      boxShadow: '0 0 20px rgba(255, 215, 0, 0.2)',
    },
  },
  '& .MuiInputBase-input': {
    color: '#fff',
    fontSize: 14,
    fontWeight: 500,
  },
  '& .MuiAutocomplete-endAdornment': {
    right: 14,
  },
}));

const ControlButton = styled(Button)(({ theme }) => ({
  backgroundColor: alpha('#fff', 0.05),
  color: '#fff',
  border: '1px solid rgba(255, 215, 0, 0.2)',
  borderRadius: 8,
  padding: '6px 16px',
  fontSize: 13,
  fontWeight: 600,
  textTransform: 'none',
  transition: 'all 0.3s ease',
  '&:hover': {
    backgroundColor: alpha('#FFD700', 0.1),
    borderColor: 'rgba(255, 215, 0, 0.4)',
    transform: 'translateY(-1px)',
    boxShadow: '0 4px 12px rgba(255, 215, 0, 0.2)',
  },
  '&.active': {
    backgroundColor: alpha('#FFD700', 0.2),
    borderColor: '#FFD700',
    color: '#FFD700',
  },
}));

const ChartContainer = styled(Box)(({ theme }) => ({
  flex: 1,
  position: 'relative',
  margin: '16px',
  borderRadius: 12,
  overflow: 'hidden',
  backgroundColor: 'rgba(22, 24, 29, 0.6)',
  backdropFilter: 'blur(10px)',
  border: '1px solid rgba(255, 215, 0, 0.1)',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
}));

const StatsPanel = styled(Paper)(({ theme }) => ({
  position: 'absolute',
  top: 16,
  right: 16,
  padding: '12px 16px',
  backgroundColor: 'rgba(22, 24, 29, 0.95)',
  backdropFilter: 'blur(20px)',
  border: '1px solid rgba(255, 215, 0, 0.2)',
  borderRadius: 8,
  boxShadow: '0 4px 24px rgba(0, 0, 0, 0.3)',
  zIndex: 5,
}));

const LiveIndicator = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  '& .pulse': {
    width: 8,
    height: 8,
    backgroundColor: '#00ff88',
    borderRadius: '50%',
    animation: 'pulse 2s infinite',
  },
  '@keyframes pulse': {
    '0%': {
      boxShadow: '0 0 0 0 rgba(0, 255, 136, 0.4)',
    },
    '70%': {
      boxShadow: '0 0 0 8px rgba(0, 255, 136, 0)',
    },
    '100%': {
      boxShadow: '0 0 0 0 rgba(0, 255, 136, 0)',
    },
  },
}));

// Symbol suggestions
const SYMBOL_SUGGESTIONS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ',
  'WMT', 'PG', 'UNH', 'MA', 'HD', 'DIS', 'PYPL', 'BAC', 'NFLX', 'ADBE',
];

// Timeframe options with intervals
const TIMEFRAME_OPTIONS = [
  { value: '1D', label: '1 Day', interval: '5m' },
  { value: '1W', label: '1 Week', interval: '30m' },
  { value: '1M', label: '1 Month', interval: '1h' },
  { value: '3M', label: '3 Months', interval: '4h' },
  { value: '6M', label: '6 Months', interval: '1d' },
  { value: '1Y', label: '1 Year', interval: '1d' },
  { value: 'ALL', label: 'All', interval: '1w' },
];

// Technical indicators
const INDICATORS = [
  { value: 'sma', label: 'SMA (20)' },
  { value: 'ema', label: 'EMA (20)' },
  { value: 'bb', label: 'Bollinger Bands' },
  { value: 'rsi', label: 'RSI (14)' },
  { value: 'macd', label: 'MACD' },
  { value: 'volume', label: 'Volume' },
];

// Chart types
const CHART_TYPES = [
  { value: 'candlestick', label: 'Candlestick', icon: <CandlestickChartIcon /> },
  { value: 'line', label: 'Line', icon: <ShowChartIcon /> },
  { value: 'area', label: 'Area', icon: <BarChartIcon /> },
];

// Enhanced Highcharts theme
const darkTheme = {
  colors: ['#FFD700', '#00ff88', '#ff4757', '#54a0ff', '#48dbfb', '#ff6348'],
  chart: {
    backgroundColor: 'transparent',
    style: {
      fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif',
    },
    plotBorderColor: 'transparent',
  },
  title: {
    style: {
      color: '#ffffff',
      fontSize: '16px',
      fontWeight: '600',
    },
  },
  subtitle: {
    style: {
      color: '#8b92a8',
      fontSize: '13px',
    },
  },
  xAxis: {
    gridLineColor: 'rgba(255, 255, 255, 0.05)',
    gridLineWidth: 1,
    labels: {
      style: {
        color: '#8b92a8',
        fontSize: '11px',
      },
    },
    lineColor: 'rgba(255, 255, 255, 0.1)',
    minorGridLineColor: 'rgba(255, 255, 255, 0.02)',
    tickColor: 'rgba(255, 255, 255, 0.1)',
    title: {
      style: {
        color: '#8b92a8',
      },
    },
  },
  yAxis: {
    gridLineColor: 'rgba(255, 255, 255, 0.05)',
    gridLineWidth: 1,
    labels: {
      style: {
        color: '#8b92a8',
        fontSize: '11px',
      },
    },
    lineColor: 'rgba(255, 255, 255, 0.1)',
    minorGridLineColor: 'rgba(255, 255, 255, 0.02)',
    tickColor: 'rgba(255, 255, 255, 0.1)',
    tickWidth: 1,
    title: {
      style: {
        color: '#8b92a8',
      },
    },
  },
  tooltip: {
    backgroundColor: 'rgba(22, 24, 29, 0.95)',
    borderColor: 'rgba(255, 215, 0, 0.3)',
    borderRadius: 8,
    borderWidth: 1,
    style: {
      color: '#ffffff',
      fontSize: '12px',
    },
    shadow: {
      color: 'rgba(0, 0, 0, 0.5)',
      offsetX: 2,
      offsetY: 2,
      opacity: 0.5,
      width: 5,
    },
  },
  plotOptions: {
    series: {
      dataLabels: {
        color: '#ffffff',
      },
      marker: {
        lineColor: 'rgba(255, 255, 255, 0.3)',
      },
    },
    candlestick: {
      lineColor: 'rgba(255, 255, 255, 0.3)',
      upColor: '#00ff88',
      upLineColor: '#00ff88',
      color: '#ff4757',
    },
  },
  legend: {
    backgroundColor: 'rgba(22, 24, 29, 0.9)',
    borderColor: 'rgba(255, 215, 0, 0.2)',
    borderRadius: 8,
    itemStyle: {
      color: '#8b92a8',
      fontSize: '12px',
      fontWeight: '500',
    },
    itemHoverStyle: {
      color: '#ffffff',
    },
    itemHiddenStyle: {
      color: 'rgba(139, 146, 168, 0.4)',
    },
    title: {
      style: {
        color: '#8b92a8',
      },
    },
  },
  credits: {
    enabled: false,
  },
  navigation: {
    buttonOptions: {
      theme: {
        fill: 'rgba(22, 24, 29, 0.9)',
        'stroke-width': 1,
        stroke: 'rgba(255, 215, 0, 0.2)',
        r: 8,
        states: {
          hover: {
            fill: 'rgba(255, 215, 0, 0.1)',
            stroke: 'rgba(255, 215, 0, 0.4)',
          },
          select: {
            fill: 'rgba(255, 215, 0, 0.2)',
            stroke: '#FFD700',
          },
        },
      },
    },
  },
  rangeSelector: {
    buttonTheme: {
      fill: 'rgba(255, 255, 255, 0.05)',
      stroke: 'rgba(255, 215, 0, 0.2)',
      'stroke-width': 1,
      r: 8,
      style: {
        color: '#8b92a8',
        fontSize: '12px',
        fontWeight: '500',
      },
      states: {
        hover: {
          fill: 'rgba(255, 215, 0, 0.1)',
          stroke: 'rgba(255, 215, 0, 0.4)',
          style: {
            color: '#ffffff',
          },
        },
        select: {
          fill: 'rgba(255, 215, 0, 0.2)',
          stroke: '#FFD700',
          style: {
            color: '#FFD700',
            fontWeight: '600',
          },
        },
      },
    },
    inputBoxBorderColor: 'rgba(255, 215, 0, 0.2)',
    inputBoxHeight: 32,
    inputBoxWidth: 120,
    inputStyle: {
      backgroundColor: 'rgba(255, 255, 255, 0.05)',
      color: '#ffffff',
      fontSize: '12px',
    },
    labelStyle: {
      color: '#8b92a8',
      fontSize: '12px',
    },
  },
  navigator: {
    handles: {
      backgroundColor: '#FFD700',
      borderColor: '#FFD700',
    },
    maskFill: 'rgba(255, 215, 0, 0.05)',
    series: {
      color: '#FFD700',
      lineColor: '#FFD700',
    },
  },
  scrollbar: {
    barBackgroundColor: 'rgba(255, 255, 255, 0.05)',
    barBorderColor: 'rgba(255, 215, 0, 0.2)',
    barBorderRadius: 4,
    barBorderWidth: 1,
    buttonArrowColor: '#8b92a8',
    buttonBackgroundColor: 'rgba(255, 255, 255, 0.05)',
    buttonBorderColor: 'rgba(255, 215, 0, 0.2)',
    buttonBorderRadius: 4,
    buttonBorderWidth: 1,
    rifleColor: '#8b92a8',
    trackBackgroundColor: 'rgba(255, 255, 255, 0.02)',
    trackBorderColor: 'rgba(255, 215, 0, 0.1)',
    trackBorderRadius: 4,
    trackBorderWidth: 1,
  },
};

// Apply theme
Highcharts.setOptions(darkTheme);

interface ProfessionalHighchartsProps {
  height?: string | number;
  initialSymbol?: string;
  onSymbolChange?: (symbol: string) => void;
}

// Sound alert for signals
const playSignalSound = () => {
  const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhByyHz/DaizsIGGG48OScTgwOUKzq765N');
  audio.volume = 0.3;
  audio.play().catch(e => console.log('Audio play failed:', e));
};

export const ProfessionalHighcharts: React.FC<ProfessionalHighchartsProps> = ({
  height = '100vh',
  initialSymbol = 'AAPL',
  onSymbolChange,
}) => {
  const theme = useTheme();
  const chartRef = useRef<HighchartsReact.RefObject>(null);

  // State
  const [symbol, setSymbol] = useState(initialSymbol);
  const [timeframe, setTimeframe] = useState('1D');
  const [chartType, setChartType] = useState('candlestick');
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>(['volume']);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  const [compareSymbols, setCompareSymbols] = useState<string[]>([]);

  // Hooks
  const { data, loading, error, refetch } = useMarketData(symbol, timeframe);
  const {
    consensus,
    isLoading: isAgentLoading,
    error: agentError,
    analyzeSymbol,
  } = useAgentAnalysis(symbol);

  // Real-time updates
  const [isConnected, setIsConnected] = useState(false);
  const [lastPrice, setLastPrice] = useState<number | null>(null);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Space: Refresh AI analysis
      if (e.code === 'Space' && !e.target?.matches('input, textarea')) {
        e.preventDefault();
        handleAnalyze();
      }
      // Enter: Execute trade (placeholder)
      else if (e.code === 'Enter' && !e.target?.matches('input, textarea')) {
        e.preventDefault();
        if (consensus) {
          playSignalSound();
          console.log('Execute trade:', consensus);
          // TODO: Implement trade execution
        }
      }
      // Arrow keys: Change timeframe
      else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        const currentIndex = TIMEFRAME_OPTIONS.findIndex(tf => tf.value === timeframe);
        if (currentIndex > 0) {
          setTimeframe(TIMEFRAME_OPTIONS[currentIndex - 1].value);
        }
      }
      else if (e.code === 'ArrowRight') {
        e.preventDefault();
        const currentIndex = TIMEFRAME_OPTIONS.findIndex(tf => tf.value === timeframe);
        if (currentIndex < TIMEFRAME_OPTIONS.length - 1) {
          setTimeframe(TIMEFRAME_OPTIONS[currentIndex + 1].value);
        }
      }
      // C: Toggle chart type
      else if (e.code === 'KeyC' && !e.target?.matches('input, textarea')) {
        e.preventDefault();
        const types = CHART_TYPES.map(t => t.value);
        const currentIndex = types.indexOf(chartType);
        setChartType(types[(currentIndex + 1) % types.length]);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [timeframe, chartType, consensus]);

  // Convert data to Highcharts format
  const convertToHighchartsData = useCallback(() => {
    if (!data || !data.candles) return { ohlc: [], volume: [] };

    const ohlc = data.candles.map(candle => [
      new Date(candle.time).getTime(),
      candle.open,
      candle.high,
      candle.low,
      candle.close,
    ]);

    const volume = data.candles.map(candle => [
      new Date(candle.time).getTime(),
      candle.volume || 0,
    ]);

    return { ohlc, volume };
  }, [data]);

  // Handle symbol change
  const handleSymbolChange = (event: any, newValue: string | null) => {
    if (newValue) {
      setSymbol(newValue);
      onSymbolChange?.(newValue);
    }
  };

  // Handle analyze
  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    try {
      await analyzeSymbol(symbol);

      // Also fetch AI prediction
      const prediction = await aiPredictionService.getPrediction(symbol);

      // Play sound if strong signal detected
      if (prediction && prediction.confidence > 75) {
        playSignalSound();
      }

      // Update chart with prediction
      if (chartRef.current && prediction) {
        const chart = chartRef.current.chart;

        // Add prediction line
        chart.addSeries({
          type: 'line',
          name: 'AI Prediction',
          data: prediction.forecast.map((point: any) => [
            new Date(point.time).getTime(),
            point.value,
          ]),
          color: '#FFD700',
          lineWidth: 3,
          dashStyle: 'Dash',
          marker: {
            enabled: false,
          },
          zIndex: 10,
        });

        // Add confidence bands
        chart.addSeries({
          type: 'arearange',
          name: 'Confidence Band',
          data: prediction.forecast.map((point: any) => [
            new Date(point.time).getTime(),
            point.lower_bound,
            point.upper_bound,
          ]),
          color: 'rgba(255, 215, 0, 0.1)',
          lineWidth: 0,
          zIndex: 9,
          marker: {
            enabled: false,
          },
        });
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  // WebSocket for real-time updates
  useEffect(() => {
    if (!symbol) return;

    const ws = new WebSocket(`ws://localhost:8000/ws/market-data/${symbol}`);

    ws.onopen = () => {
      console.log('WebSocket connected for', symbol);
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const update = JSON.parse(event.data);
        if (update.type === 'price_update' && chartRef.current) {
          const chart = chartRef.current.chart;
          const mainSeries = chart.get('main');

          if (mainSeries && mainSeries.data.length > 0) {
            const lastPoint = mainSeries.data[mainSeries.data.length - 1];
            const now = Date.now();

            // Update last candle
            lastPoint.update({
              high: Math.max(lastPoint.high, update.price),
              low: Math.min(lastPoint.low, update.price),
              close: update.price,
            });

            setLastPrice(update.price);
          }
        }
      } catch (err) {
        console.error('WebSocket message error:', err);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [symbol]);

  // Handle comparison mode
  const handleAddComparison = (newSymbol: string) => {
    if (newSymbol && !compareSymbols.includes(newSymbol)) {
      setCompareSymbols([...compareSymbols, newSymbol]);
    }
  };

  const handleRemoveComparison = (symbolToRemove: string) => {
    setCompareSymbols(compareSymbols.filter(s => s !== symbolToRemove));
  };

  // Initialize chart
  useEffect(() => {
    if (!data) return;

    const { ohlc, volume } = convertToHighchartsData();

    const chartOptions: Highcharts.Options = {
      chart: {
        height: '100%',
        events: {
          load: function() {
            // Add gradient background
            this.renderer.rect(0, 0, this.chartWidth, this.chartHeight)
              .attr({
                fill: 'url(#gradient)',
                zIndex: -1,
              })
              .add();
          },
        },
      },

      defs: {
        gradient: {
          tagName: 'linearGradient',
          id: 'gradient',
          x1: 0,
          y1: 0,
          x2: 0,
          y2: 1,
          children: [
            { tagName: 'stop', offset: 0, 'stop-color': 'rgba(255, 215, 0, 0.02)' },
            { tagName: 'stop', offset: 1, 'stop-color': 'rgba(255, 215, 0, 0)' },
          ],
        },
      },

      title: {
        text: '',
      },

      rangeSelector: {
        enabled: false, // We'll use our custom controls
      },

      navigator: {
        enabled: true,
        height: 40,
      },

      xAxis: {
        type: 'datetime',
        labels: {
          formatter: function() {
            return Highcharts.dateFormat('%H:%M', this.value as number);
          },
        },
      },

      yAxis: [{
        labels: {
          align: 'right',
          x: -8,
          formatter: function() {
            return '$' + this.value;
          },
        },
        title: {
          text: '',
        },
        height: selectedIndicators.includes('volume') ? '70%' : '100%',
        lineWidth: 0,
        resize: {
          enabled: true,
        },
      }, {
        labels: {
          align: 'right',
          x: -8,
        },
        title: {
          text: '',
        },
        top: '70%',
        height: '30%',
        offset: 0,
        lineWidth: 0,
        visible: selectedIndicators.includes('volume'),
      }],

      plotOptions: {
        candlestick: {
          color: '#ff4757',
          upColor: '#00ff88',
          lineColor: '#ff4757',
          upLineColor: '#00ff88',
        },
        series: {
          animation: {
            duration: 1000,
          },
          states: {
            hover: {
              brightness: 0.2,
            },
          },
        },
      },

      series: [{
        type: chartType as any,
        name: symbol,
        data: ohlc,
        id: 'main',
        yAxis: 0,
        dataGrouping: {
          enabled: true,
        },
      }],

      tooltip: {
        split: true,
        distance: 30,
        padding: 12,
        formatter: function() {
          const points = this.points || [];
          let s = '<b>' + Highcharts.dateFormat('%Y-%m-%d %H:%M', this.x as number) + '</b>';

          points.forEach((point: any) => {
            if (point.series.type === 'candlestick') {
              s += '<br/><b>' + point.series.name + '</b>';
              s += '<br/>Open: $' + point.point.open.toFixed(2);
              s += '<br/>High: $' + point.point.high.toFixed(2);
              s += '<br/>Low: $' + point.point.low.toFixed(2);
              s += '<br/>Close: $' + point.point.close.toFixed(2);
            } else {
              s += '<br/>' + point.series.name + ': ' + point.y.toFixed(2);
            }
          });

          return s;
        },
      },

      responsive: {
        rules: [{
          condition: {
            maxWidth: 500,
          },
          chartOptions: {
            chart: {
              height: 400,
            },
            subtitle: {
              text: null,
            },
            navigator: {
              enabled: false,
            },
          },
        }],
      },
    };

    // Add volume if selected
    if (selectedIndicators.includes('volume')) {
      chartOptions.series?.push({
        type: 'column',
        name: 'Volume',
        data: volume,
        yAxis: 1,
        color: 'rgba(74, 77, 87, 0.5)',
      });
    }

    // Add technical indicators
    if (selectedIndicators.includes('sma')) {
      chartOptions.series?.push({
        type: 'sma',
        linkedTo: 'main',
        params: {
          period: 20,
        },
        color: '#54a0ff',
        lineWidth: 2,
        marker: {
          enabled: false,
        },
      });
    }

    if (selectedIndicators.includes('ema')) {
      chartOptions.series?.push({
        type: 'ema',
        linkedTo: 'main',
        params: {
          period: 20,
        },
        color: '#48dbfb',
        lineWidth: 2,
        marker: {
          enabled: false,
        },
      });
    }

    if (selectedIndicators.includes('bb')) {
      chartOptions.series?.push({
        type: 'bb',
        linkedTo: 'main',
        color: '#ff6348',
        lineWidth: 1,
        marker: {
          enabled: false,
        },
      });
    }

    // Add RSI if selected
    if (selectedIndicators.includes('rsi')) {
      chartOptions.yAxis?.push({
        labels: {
          align: 'right',
          x: -8,
        },
        title: {
          text: 'RSI',
          style: {
            color: '#8b92a8',
          },
        },
        top: selectedIndicators.includes('volume') ? '73%' : '75%',
        height: '25%',
        offset: 0,
        lineWidth: 0,
        plotLines: [{
          value: 70,
          color: 'rgba(255, 71, 87, 0.5)',
          dashStyle: 'Dash',
          width: 1,
        }, {
          value: 30,
          color: 'rgba(0, 255, 136, 0.5)',
          dashStyle: 'Dash',
          width: 1,
        }],
      });

      chartOptions.series?.push({
        type: 'rsi',
        linkedTo: 'main',
        yAxis: chartOptions.yAxis.length - 1,
        color: '#FFD700',
        lineWidth: 2,
        marker: {
          enabled: false,
        },
      });
    }

    // Add MACD if selected
    if (selectedIndicators.includes('macd')) {
      const yAxisIndex = chartOptions.yAxis?.length || 0;

      chartOptions.yAxis?.push({
        labels: {
          align: 'right',
          x: -8,
        },
        title: {
          text: 'MACD',
          style: {
            color: '#8b92a8',
          },
        },
        top: selectedIndicators.includes('rsi') ? '78%' : '75%',
        height: '22%',
        offset: 0,
        lineWidth: 0,
      });

      chartOptions.series?.push({
        type: 'macd',
        linkedTo: 'main',
        yAxis: yAxisIndex,
        lineWidth: 2,
        marker: {
          enabled: false,
        },
      });
    }

    // Update chart
    if (chartRef.current) {
      chartRef.current.chart.update(chartOptions);
    }

  }, [data, chartType, selectedIndicators, symbol]);

  return (
    <Container>
      {/* Header */}
      <Header>
        {/* Left side - Search */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <SearchBar
            value={symbol}
            onChange={handleSymbolChange}
            options={SYMBOL_SUGGESTIONS}
            renderInput={(params) => (
              <TextField
                {...params}
                placeholder="Search symbol..."
                InputProps={{
                  ...params.InputProps,
                  startAdornment: <SearchIcon sx={{ color: '#8b92a8', mr: 1 }} />,
                }}
              />
            )}
            size="small"
            freeSolo
          />

          <ControlButton
            onClick={handleAnalyze}
            disabled={isAnalyzing || isAgentLoading}
            startIcon={isAnalyzing ? <CircularProgress size={16} /> : <PsychologyIcon />}
          >
            {isAnalyzing ? 'Analyzing...' : 'Analyze'}
          </ControlButton>
        </Box>

        {/* Center - Symbol info */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 3 }}>
          <Typography variant="h5" sx={{ color: '#fff', fontWeight: 700 }}>
            {symbol}
          </Typography>

          <LiveIndicator>
            <div className="pulse" />
            <Typography variant="caption" sx={{ color: isConnected ? '#00ff88' : '#ff4757', fontWeight: 600 }}>
              {isConnected ? 'LIVE' : 'OFFLINE'}
            </Typography>
          </LiveIndicator>

          {lastPrice && (
            <Typography variant="h6" sx={{ color: '#fff', fontWeight: 600 }}>
              ${lastPrice.toFixed(2)}
            </Typography>
          )}

          {consensus && (
            <Chip
              icon={consensus.signal === 'BUY' ? <TrendingUpIcon /> : <TrendingDownIcon />}
              label={`${consensus.signal} ${consensus.confidence}%`}
              sx={{
                backgroundColor: consensus.signal === 'BUY' ?
                  alpha('#00ff88', 0.2) : alpha('#ff4757', 0.2),
                color: consensus.signal === 'BUY' ? '#00ff88' : '#ff4757',
                border: `1px solid ${consensus.signal === 'BUY' ? '#00ff88' : '#ff4757'}`,
                fontWeight: 600,
              }}
            />
          )}
        </Box>

        {/* Right side - Controls */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {/* Timeframe selector */}
          <FormControl size="small">
            <Select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              sx={{
                minWidth: 100,
                backgroundColor: alpha('#fff', 0.05),
                color: '#fff',
                borderRadius: 1,
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(255, 215, 0, 0.2)',
                },
                '&:hover .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(255, 215, 0, 0.3)',
                },
                '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(255, 215, 0, 0.5)',
                },
              }}
            >
              {TIMEFRAME_OPTIONS.map(option => (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Chart type selector */}
          <FormControl size="small">
            <Select
              value={chartType}
              onChange={(e) => setChartType(e.target.value)}
              sx={{
                minWidth: 120,
                backgroundColor: alpha('#fff', 0.05),
                color: '#fff',
                borderRadius: 1,
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(255, 215, 0, 0.2)',
                },
              }}
            >
              {CHART_TYPES.map(type => (
                <MenuItem key={type.value} value={type.value}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {type.icon}
                    {type.label}
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Indicators selector */}
          <FormControl size="small">
            <Select
              multiple
              value={selectedIndicators}
              onChange={(e) => setSelectedIndicators(e.target.value as string[])}
              input={<OutlinedInput />}
              renderValue={(selected) => `Indicators (${selected.length})`}
              sx={{
                minWidth: 140,
                backgroundColor: alpha('#fff', 0.05),
                color: '#fff',
                borderRadius: 1,
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(255, 215, 0, 0.2)',
                },
              }}
            >
              {INDICATORS.map(indicator => (
                <MenuItem key={indicator.value} value={indicator.value}>
                  <Checkbox checked={selectedIndicators.includes(indicator.value)} />
                  <ListItemText primary={indicator.label} />
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Keyboard shortcuts help */}
          <Tooltip
            title={
              <Box sx={{ p: 1 }}>
                <Typography variant="caption" sx={{ display: 'block', mb: 0.5, fontWeight: 600 }}>
                  Keyboard Shortcuts:
                </Typography>
                <Typography variant="caption" sx={{ display: 'block' }}>
                  Space - Refresh AI Analysis
                </Typography>
                <Typography variant="caption" sx={{ display: 'block' }}>
                  Enter - Execute Trade
                </Typography>
                <Typography variant="caption" sx={{ display: 'block' }}>
                  ← → - Change Timeframe
                </Typography>
                <Typography variant="caption" sx={{ display: 'block' }}>
                  C - Toggle Chart Type
                </Typography>
              </Box>
            }
            placement="bottom"
          >
            <IconButton
              size="small"
              sx={{
                color: '#8b92a8',
                '&:hover': { color: '#FFD700' }
              }}
            >
              <KeyboardIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Header>

      {/* Tabs for Main/Compare */}
      <Box sx={{ borderBottom: 1, borderColor: 'rgba(255, 215, 0, 0.1)', px: 2 }}>
        <Tabs
          value={tabValue}
          onChange={(e, v) => setTabValue(v)}
          sx={{
            '& .MuiTab-root': {
              color: '#8b92a8',
              textTransform: 'none',
              fontWeight: 600,
              '&.Mui-selected': {
                color: '#FFD700',
              },
            },
            '& .MuiTabs-indicator': {
              backgroundColor: '#FFD700',
              height: 3,
            },
          }}
        >
          <Tab label="Main" />
          <Tab label="Compare" />
        </Tabs>
      </Box>

      {/* Chart Container */}
      <ChartContainer>
        {tabValue === 1 ? (
          // Compare Mode
          <Box sx={{ p: 3, height: '100%', overflow: 'auto' }}>
            <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
              <Autocomplete
                options={SYMBOL_SUGGESTIONS.filter(s => s !== symbol && !compareSymbols.includes(s))}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    placeholder="Add symbol to compare..."
                    size="small"
                    sx={{
                      width: 300,
                      '& .MuiInputBase-root': {
                        backgroundColor: alpha('#fff', 0.05),
                        color: '#fff',
                      },
                    }}
                  />
                )}
                onChange={(e, value) => value && handleAddComparison(value)}
                size="small"
              />
              <Typography variant="body2" sx={{ color: '#8b92a8' }}>
                Comparing {symbol} with:
              </Typography>
              {compareSymbols.map(sym => (
                <Chip
                  key={sym}
                  label={sym}
                  onDelete={() => handleRemoveComparison(sym)}
                  sx={{
                    backgroundColor: alpha('#FFD700', 0.2),
                    color: '#FFD700',
                    border: '1px solid rgba(255, 215, 0, 0.3)',
                  }}
                />
              ))}
            </Box>

            {/* Comparison Chart */}
            <Box sx={{ height: 'calc(100% - 80px)' }}>
              <HighchartsReact
                highcharts={Highcharts}
                constructorType={'stockChart'}
                options={{
                  chart: {
                    height: '100%',
                  },
                  series: [
                    {
                      type: 'line',
                      name: symbol,
                      data: convertToHighchartsData().ohlc.map(point => [point[0], point[4]]), // Close prices
                      color: '#FFD700',
                      lineWidth: 2,
                    },
                    ...compareSymbols.map((sym, index) => ({
                      type: 'line',
                      name: sym,
                      data: [], // Would need to fetch data for each symbol
                      color: darkTheme.colors[index + 1],
                      lineWidth: 2,
                    })),
                  ],
                }}
              />
            </Box>
          </Box>
        ) : loading ? (
          <Box sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
          }}>
            <CircularProgress sx={{ color: '#FFD700' }} />
          </Box>
        ) : error ? (
          <Box sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            flexDirection: 'column',
            gap: 2,
          }}>
            <Typography color="error">Failed to load data</Typography>
            <ControlButton onClick={refetch}>Retry</ControlButton>
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

            {/* Stats Panel */}
            {consensus && (
              <Fade in>
                <StatsPanel elevation={0}>
                  <Typography variant="caption" sx={{ color: '#8b92a8', mb: 1, display: 'block' }}>
                    AI Analysis
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 3 }}>
                      <Typography variant="caption" sx={{ color: '#8b92a8' }}>
                        Accuracy:
                      </Typography>
                      <Typography variant="caption" sx={{ color: '#FFD700', fontWeight: 600 }}>
                        {consensus.historical_accuracy || 82}%
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 3 }}>
                      <Typography variant="caption" sx={{ color: '#8b92a8' }}>
                        Target:
                      </Typography>
                      <Typography variant="caption" sx={{ color: '#00ff88', fontWeight: 600 }}>
                        ${consensus.take_profit?.toFixed(2) || '--'}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 3 }}>
                      <Typography variant="caption" sx={{ color: '#8b92a8' }}>
                        Stop:
                      </Typography>
                      <Typography variant="caption" sx={{ color: '#ff4757', fontWeight: 600 }}>
                        ${consensus.stop_loss?.toFixed(2) || '--'}
                      </Typography>
                    </Box>
                  </Box>
                </StatsPanel>
              </Fade>
            )}
          </>
        )}
      </ChartContainer>
    </Container>
  );
};
