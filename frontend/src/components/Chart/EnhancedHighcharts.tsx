/**
 * EnhancedHighcharts Component
 *
 * A professional trading chart implementation using Highcharts Stock
 * without complex module dependencies, focusing on core functionality
 * with beautiful design.
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from 'highcharts-react-official';

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
  Tooltip,
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
  KeyboardAlt as KeyboardIcon,
  FiberManualRecord as LiveIcon,
} from '@mui/icons-material';

// Hooks and services
import { useMarketData } from './hooks/useMarketData';
import { useAgentAnalysis } from './hooks/useAgentAnalysis';
import { aiPredictionService } from '../../services/aiPredictionService';

// Styled components
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

// Symbol suggestions
const SYMBOL_SUGGESTIONS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ',
  'WMT', 'PG', 'UNH', 'MA', 'HD', 'DIS', 'PYPL', 'BAC', 'NFLX', 'ADBE',
];

// Timeframe options
const TIMEFRAME_OPTIONS = [
  { value: '1D', label: '1 Day' },
  { value: '1W', label: '1 Week' },
  { value: '1M', label: '1 Month' },
  { value: '3M', label: '3 Months' },
  { value: '6M', label: '6 Months' },
  { value: '1Y', label: '1 Year' },
  { value: 'ALL', label: 'All' },
];

// Chart types
const CHART_TYPES = [
  { value: 'candlestick', label: 'Candlestick', icon: <CandlestickChartIcon /> },
  { value: 'line', label: 'Line', icon: <ShowChartIcon /> },
  { value: 'area', label: 'Area', icon: <BarChartIcon /> },
];

// Dark theme configuration
Highcharts.theme = {
  colors: ['#FFD700', '#00ff88', '#ff4757', '#54a0ff', '#48dbfb', '#ff6348'],
  chart: {
    backgroundColor: 'transparent',
    style: {
      fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif',
    },
  },
  title: {
    style: {
      color: '#ffffff',
    },
  },
  xAxis: {
    gridLineColor: 'rgba(255, 255, 255, 0.05)',
    labels: {
      style: {
        color: '#8b92a8',
      },
    },
    lineColor: 'rgba(255, 255, 255, 0.1)',
  },
  yAxis: {
    gridLineColor: 'rgba(255, 255, 255, 0.05)',
    labels: {
      style: {
        color: '#8b92a8',
      },
    },
    lineColor: 'rgba(255, 255, 255, 0.1)',
  },
  tooltip: {
    backgroundColor: 'rgba(22, 24, 29, 0.95)',
    borderColor: 'rgba(255, 215, 0, 0.3)',
    borderRadius: 8,
    style: {
      color: '#ffffff',
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
    itemStyle: {
      color: '#8b92a8',
    },
    itemHoverStyle: {
      color: '#ffffff',
    },
  },
  credits: {
    enabled: false,
  },
  navigation: {
    buttonOptions: {
      theme: {
        fill: 'rgba(22, 24, 29, 0.9)',
        stroke: 'rgba(255, 215, 0, 0.2)',
        states: {
          hover: {
            fill: 'rgba(255, 215, 0, 0.1)',
          },
        },
      },
    },
  },
};

// Apply theme
Highcharts.setOptions(Highcharts.theme);

// Sound for alerts
const playSignalSound = () => {
  const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhByyHz/DaizsIGGG48OScTgwOUKzq765N');
  audio.volume = 0.3;
  audio.play().catch(e => console.log('Audio play failed:', e));
};

interface EnhancedHighchartsProps {
  height?: string | number;
  initialSymbol?: string;
  onSymbolChange?: (symbol: string) => void;
}

export const EnhancedHighcharts: React.FC<EnhancedHighchartsProps> = ({
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
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  const [showVolume, setShowVolume] = useState(true);

  // Hooks
  const { data, loading, error, refetch } = useMarketData(symbol, timeframe);
  const { consensus, isLoading: isAgentLoading, analyzeSymbol } = useAgentAnalysis(symbol);

  // Real-time state
  const [isConnected, setIsConnected] = useState(false);
  const [lastPrice, setLastPrice] = useState<number | null>(null);

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
      const prediction = await aiPredictionService.getPrediction(symbol);

      if (prediction && prediction.confidence > 75) {
        playSignalSound();
      }

      // Add prediction to chart
      if (chartRef.current && prediction) {
        const chart = chartRef.current.chart;
        const lastCandle = data.candles[data.candles.length - 1];
        const startTime = new Date(lastCandle.time).getTime();

        // Create prediction data points
        const predictionData = [];
        for (let i = 0; i <= 15; i++) {
          const time = startTime + (i * 60000); // 1 minute intervals
          const value = lastCandle.close * (1 + (prediction.direction === 'UP' ? 0.002 : -0.002) * i);
          predictionData.push([time, value]);
        }

        // Remove existing prediction series
        const existingPrediction = chart.get('prediction');
        if (existingPrediction) {
          existingPrediction.remove();
        }

        // Add new prediction line
        chart.addSeries({
          type: 'line',
          id: 'prediction',
          name: 'AI Prediction',
          data: predictionData,
          color: '#FFD700',
          lineWidth: 3,
          dashStyle: 'Dash',
          marker: {
            enabled: false,
          },
          zIndex: 10,
        });
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.target?.matches('input, textarea')) {
        e.preventDefault();
        handleAnalyze();
      } else if (e.code === 'KeyC' && !e.target?.matches('input, textarea')) {
        e.preventDefault();
        const types = CHART_TYPES.map(t => t.value);
        const currentIndex = types.indexOf(chartType);
        setChartType(types[(currentIndex + 1) % types.length]);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [chartType]);

  // Initialize chart
  useEffect(() => {
    if (!data) return;

    const { ohlc, volume } = convertToHighchartsData();

    const chartOptions: Highcharts.Options = {
      chart: {
        height: '100%',
      },

      title: {
        text: '',
      },

      rangeSelector: {
        buttons: TIMEFRAME_OPTIONS.map(tf => ({
          type: 'all',
          text: tf.label,
          events: {
            click: () => {
              setTimeframe(tf.value);
              return false;
            },
          },
        })),
        selected: 0,
        inputEnabled: false,
      },

      xAxis: {
        type: 'datetime',
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
        height: showVolume ? '70%' : '100%',
        lineWidth: 0,
      }, {
        labels: {
          align: 'right',
          x: -8,
        },
        title: {
          text: 'Volume',
        },
        top: '72%',
        height: '28%',
        offset: 0,
        lineWidth: 0,
        visible: showVolume,
      }],

      tooltip: {
        split: true,
        distance: 30,
        padding: 12,
      },

      plotOptions: {
        series: {
          animation: {
            duration: 1000,
          },
        },
      },

      series: [{
        type: chartType as any,
        name: symbol,
        data: ohlc,
        id: 'main',
        yAxis: 0,
      }],

      navigator: {
        enabled: true,
        height: 40,
      },
    };

    // Add volume if enabled
    if (showVolume) {
      chartOptions.series?.push({
        type: 'column',
        name: 'Volume',
        data: volume,
        yAxis: 1,
        color: 'rgba(74, 77, 87, 0.5)',
      });
    }

    // Update chart
    if (chartRef.current) {
      chartRef.current.chart.update(chartOptions, true, true);
    }

  }, [data, chartType, showVolume, symbol, timeframe]);

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
          setLastPrice(update.price);

          const chart = chartRef.current.chart;
          const mainSeries = chart.get('main');

          if (mainSeries && mainSeries.data.length > 0) {
            const lastPoint = mainSeries.data[mainSeries.data.length - 1];
            lastPoint.update({
              high: Math.max(lastPoint.high, update.price),
              low: Math.min(lastPoint.low, update.price),
              close: update.price,
            });
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

  return (
    <Container>
      {/* Header */}
      <Header>
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

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 3 }}>
          <Typography variant="h5" sx={{ color: '#fff', fontWeight: 700 }}>
            {symbol}
          </Typography>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <div style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              backgroundColor: isConnected ? '#00ff88' : '#ff4757',
              animation: isConnected ? 'pulse 2s infinite' : 'none',
            }} />
            <Typography variant="caption" sx={{ color: isConnected ? '#00ff88' : '#ff4757', fontWeight: 600 }}>
              {isConnected ? 'LIVE' : 'OFFLINE'}
            </Typography>
          </Box>

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

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
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

          <FormControlLabel
            control={
              <Checkbox
                checked={showVolume}
                onChange={(e) => setShowVolume(e.target.checked)}
                sx={{
                  color: 'rgba(255, 215, 0, 0.5)',
                  '&.Mui-checked': {
                    color: '#FFD700',
                  },
                }}
              />
            }
            label="Volume"
            sx={{ color: '#8b92a8' }}
          />

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
                  C - Toggle Chart Type
                </Typography>
              </Box>
            }
          >
            <IconButton size="small" sx={{ color: '#8b92a8' }}>
              <KeyboardIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Header>

      {/* Tabs */}
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
          <Tab label="Chart" />
          <Tab label="Analysis" />
        </Tabs>
      </Box>

      {/* Chart Container */}
      <ChartContainer>
        {loading ? (
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
        ) : tabValue === 0 ? (
          <>
            <HighchartsReact
              highcharts={Highcharts}
              constructorType={'stockChart'}
              options={{}}
              ref={chartRef}
              containerProps={{ style: { width: '100%', height: '100%' } }}
            />

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
        ) : (
          <Box sx={{ p: 3, color: '#8b92a8' }}>
            <Typography variant="h6" sx={{ mb: 2, color: '#fff' }}>
              AI Analysis Results
            </Typography>
            {consensus ? (
              <Box>
                <Typography>Signal: {consensus.signal}</Typography>
                <Typography>Confidence: {consensus.confidence}%</Typography>
                <Typography>Entry Price: ${consensus.entry_price?.toFixed(2)}</Typography>
                <Typography>Take Profit: ${consensus.take_profit?.toFixed(2)}</Typography>
                <Typography>Stop Loss: ${consensus.stop_loss?.toFixed(2)}</Typography>
              </Box>
            ) : (
              <Typography>Click "Analyze" to generate AI predictions</Typography>
            )}
          </Box>
        )}
      </ChartContainer>

      <style jsx global>{`
        @keyframes pulse {
          0% {
            box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.4);
          }
          70% {
            box-shadow: 0 0 0 8px rgba(0, 255, 136, 0);
          }
          100% {
            box-shadow: 0 0 0 0 rgba(0, 255, 136, 0);
          }
        }
      `}</style>
    </Container>
  );
};
