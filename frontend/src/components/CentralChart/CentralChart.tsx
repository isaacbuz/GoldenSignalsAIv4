import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Box,
  Paper,
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
} from '@mui/material';
import {
  ShowChart as ShowChartIcon,
  CandlestickChart as CandlestickChartIcon,
  Timeline as TimelineIcon,
  TrendingUp as TrendingUpIcon,
  Fullscreen as FullscreenIcon,
  FullscreenExit as FullscreenExitIcon,
  PhotoCamera as PhotoCameraIcon,
  Settings as SettingsIcon,
  Layers as LayersIcon,
  CompareArrows as CompareArrowsIcon,
  Brush as BrushIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  Filler,
  TimeScale,
  TimeSeriesScale,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import zoomPlugin from 'chartjs-plugin-zoom';
import annotationPlugin from 'chartjs-plugin-annotation';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  ChartTooltip,
  Legend,
  Filler,
  TimeScale,
  TimeSeriesScale,
  zoomPlugin,
  annotationPlugin
);

// Styled components
const ChartContainer = styled(Paper)(({ theme }) => ({
  position: 'relative',
  height: '100%',
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.spacing(1),
  overflow: 'hidden',
  display: 'flex',
  flexDirection: 'column',
}));

const ChartHeader = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1.5, 2),
  borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  minHeight: 56,
}));

const ChartToolbar = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1, 2),
  borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  flexWrap: 'wrap',
}));

const ChartArea = styled(Box)(({ theme }) => ({
  flex: 1,
  position: 'relative',
  padding: theme.spacing(2),
  minHeight: 400,
}));

const SignalOverlay = styled(Box)(({ theme }) => ({
  position: 'absolute',
  padding: theme.spacing(1, 2),
  backgroundColor: alpha(theme.palette.background.paper, 0.95),
  border: `1px solid ${theme.palette.divider}`,
  borderRadius: theme.spacing(1),
  boxShadow: theme.shadows[4],
  zIndex: 10,
}));

// Types
interface PriceData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface Signal {
  type: 'buy' | 'sell';
  price: number;
  time: string;
  confidence: number;
  stopLoss?: number;
  takeProfit?: number[];
}

interface TechnicalIndicator {
  id: string;
  name: string;
  type: 'overlay' | 'panel';
  data?: number[];
  config?: any;
}

interface CentralChartProps {
  symbol?: string;
  timeframe?: string;
  onSymbolChange?: (symbol: string) => void;
  onTimeframeChange?: (timeframe: string) => void;
  className?: string;
}

// Mock data generator
const generateMockPriceData = (days: number = 90): PriceData[] => {
  const data: PriceData[] = [];
  let basePrice = 150;
  const now = new Date();

  for (let i = days; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    
    const volatility = 0.02;
    const trend = Math.sin(i / 10) * 5;
    const randomWalk = (Math.random() - 0.5) * basePrice * volatility;
    
    basePrice = Math.max(basePrice + randomWalk + trend * 0.1, 50);
    
    const high = basePrice + Math.random() * 2;
    const low = basePrice - Math.random() * 2;
    const open = low + Math.random() * (high - low);
    const close = low + Math.random() * (high - low);
    
    data.push({
      time: date.toISOString(),
      open,
      high,
      low,
      close,
      volume: Math.floor(1000000 + Math.random() * 5000000),
    });
  }

  return data;
};

export const CentralChart: React.FC<CentralChartProps> = ({
  symbol = 'AAPL',
  timeframe = '1d',
  onSymbolChange,
  onTimeframeChange,
  className,
}) => {
  const theme = useTheme();
  const chartRef = useRef<any>(null);
  const [chartType, setChartType] = useState<'line' | 'candlestick' | 'bar'>('candlestick');
  const [priceData, setPriceData] = useState<PriceData[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [indicators, setIndicators] = useState<TechnicalIndicator[]>([]);
  const [loading, setLoading] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  const [drawingMode, setDrawingMode] = useState<string | null>(null);
  const [indicatorMenuAnchor, setIndicatorMenuAnchor] = useState<null | HTMLElement>(null);

  // Fetch data
  useEffect(() => {
    fetchData();
  }, [symbol, timeframe]);

  const fetchData = async () => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const mockData = generateMockPriceData(90);
      setPriceData(mockData);
      
      // Generate mock signals
      const mockSignals: Signal[] = [
        {
          type: 'buy',
          price: mockData[70].close,
          time: mockData[70].time,
          confidence: 0.947,
          stopLoss: mockData[70].close * 0.976,
          takeProfit: [
            mockData[70].close * 1.04,
            mockData[70].close * 1.076,
          ],
        },
        {
          type: 'sell',
          price: mockData[85].close,
          time: mockData[85].time,
          confidence: 0.923,
        },
      ];
      setSignals(mockSignals);
      
      // Add default indicators
      setIndicators([
        { id: 'sma20', name: 'SMA 20', type: 'overlay' },
        { id: 'sma50', name: 'SMA 50', type: 'overlay' },
        { id: 'volume', name: 'Volume', type: 'panel' },
      ]);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  // Calculate indicators
  const calculateSMA = (data: number[], period: number): (number | null)[] => {
    const sma: (number | null)[] = [];
    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) {
        sma.push(null);
      } else {
        const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
        sma.push(sum / period);
      }
    }
    return sma;
  };

  // Chart data
  const chartData = useMemo(() => {
    const labels = priceData.map(d => d.time);
    const closePrices = priceData.map(d => d.close);
    
    const datasets: any[] = [
      {
        label: symbol,
        data: closePrices,
        borderColor: theme.palette.primary.main,
        backgroundColor: alpha(theme.palette.primary.main, 0.1),
        fill: true,
        tension: 0.1,
        pointRadius: 0,
        borderWidth: 2,
      },
    ];

    // Add indicator overlays
    indicators.forEach(indicator => {
      if (indicator.type === 'overlay') {
        if (indicator.id === 'sma20') {
          datasets.push({
            label: 'SMA 20',
            data: calculateSMA(closePrices, 20),
            borderColor: theme.palette.warning.main,
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0,
          });
        } else if (indicator.id === 'sma50') {
          datasets.push({
            label: 'SMA 50',
            data: calculateSMA(closePrices, 50),
            borderColor: theme.palette.info.main,
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0,
          });
        }
      }
    });

    return {
      labels,
      datasets,
    };
  }, [priceData, indicators, symbol, theme]);

  // Chart options
  const chartOptions = useMemo(() => {
    const annotations: any = {};
    
    // Add signal annotations
    signals.forEach((signal, index) => {
      const dataIndex = priceData.findIndex(d => d.time === signal.time);
      if (dataIndex >= 0) {
        annotations[`signal_${index}`] = {
          type: 'point',
          xValue: signal.time,
          yValue: signal.price,
          backgroundColor: signal.type === 'buy' ? theme.palette.success.main : theme.palette.error.main,
          borderColor: signal.type === 'buy' ? theme.palette.success.dark : theme.palette.error.dark,
          borderWidth: 2,
          radius: 8,
        };

        // Add stop loss line
        if (signal.stopLoss) {
          annotations[`stoploss_${index}`] = {
            type: 'line',
            yMin: signal.stopLoss,
            yMax: signal.stopLoss,
            borderColor: theme.palette.error.main,
            borderWidth: 1,
            borderDash: [5, 5],
          };
        }

        // Add take profit lines
        if (signal.takeProfit) {
          signal.takeProfit.forEach((tp, tpIndex) => {
            annotations[`takeprofit_${index}_${tpIndex}`] = {
              type: 'line',
              yMin: tp,
              yMax: tp,
              borderColor: theme.palette.success.main,
              borderWidth: 1,
              borderDash: [5, 5],
            };
          });
        }
      }
    });

    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index' as const,
        intersect: false,
      },
      plugins: {
        legend: {
          display: true,
          position: 'top' as const,
        },
        tooltip: {
          callbacks: {
            label: (context: any) => {
              const label = context.dataset.label || '';
              const value = context.parsed.y;
              return `${label}: $${value.toFixed(2)}`;
            },
          },
        },
        zoom: {
          zoom: {
            wheel: {
              enabled: true,
            },
            pinch: {
              enabled: true,
            },
            mode: 'x' as const,
          },
          pan: {
            enabled: true,
            mode: 'x' as const,
          },
        },
        annotation: {
          annotations,
        },
      },
      scales: {
        x: {
          type: 'time' as const,
          time: {
            unit: 'day' as const,
          },
          grid: {
            color: alpha(theme.palette.divider, 0.1),
          },
        },
        y: {
          position: 'right' as const,
          grid: {
            color: alpha(theme.palette.divider, 0.1),
          },
          ticks: {
            callback: (value: any) => `$${value}`,
          },
        },
      },
    };
  }, [signals, priceData, theme]);

  // Handlers
  const handleIndicatorMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setIndicatorMenuAnchor(event.currentTarget);
  };

  const handleIndicatorMenuClose = () => {
    setIndicatorMenuAnchor(null);
  };

  const handleAddIndicator = (indicatorId: string) => {
    // Add indicator logic
    handleIndicatorMenuClose();
  };

  const handleFullscreen = () => {
    setFullscreen(!fullscreen);
  };

  const availableIndicators = [
    { id: 'rsi', name: 'RSI', category: 'Momentum' },
    { id: 'macd', name: 'MACD', category: 'Momentum' },
    { id: 'bb', name: 'Bollinger Bands', category: 'Volatility' },
    { id: 'ema', name: 'EMA', category: 'Trend' },
    { id: 'vwap', name: 'VWAP', category: 'Volume' },
    { id: 'stoch', name: 'Stochastic', category: 'Momentum' },
  ];

  if (loading) {
    return (
      <ChartContainer className={className}>
        <Box display="flex" justifyContent="center" alignItems="center" height="100%">
          <CircularProgress />
        </Box>
      </ChartContainer>
    );
  }

  return (
    <ChartContainer className={className} elevation={fullscreen ? 0 : 1} sx={fullscreen ? {
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
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="h6" fontWeight="bold">
            {symbol}
          </Typography>
          <Chip
            label={`$${priceData[priceData.length - 1]?.close.toFixed(2)}`}
            color="primary"
            size="small"
          />
          <Chip
            label={`+2.34 (1.35%)`}
            color="success"
            size="small"
            icon={<TrendingUpIcon />}
          />
        </Box>
        <Box display="flex" alignItems="center" gap={1}>
          <IconButton size="small" onClick={handleFullscreen}>
            {fullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
          </IconButton>
          <IconButton size="small">
            <PhotoCameraIcon />
          </IconButton>
          <IconButton size="small">
            <SettingsIcon />
          </IconButton>
        </Box>
      </ChartHeader>

      {/* Toolbar */}
      <ChartToolbar>
        <ButtonGroup size="small" variant="outlined">
          <Button
            variant={chartType === 'line' ? 'contained' : 'outlined'}
            onClick={() => setChartType('line')}
          >
            <ShowChartIcon />
          </Button>
          <Button
            variant={chartType === 'candlestick' ? 'contained' : 'outlined'}
            onClick={() => setChartType('candlestick')}
          >
            <CandlestickChartIcon />
          </Button>
          <Button
            variant={chartType === 'bar' ? 'contained' : 'outlined'}
            onClick={() => setChartType('bar')}
          >
            <TimelineIcon />
          </Button>
        </ButtonGroup>

        <Divider orientation="vertical" flexItem />

        <Tooltip title="Indicators">
          <IconButton size="small" onClick={handleIndicatorMenuOpen}>
            <LayersIcon />
          </IconButton>
        </Tooltip>

        <Tooltip title="Drawing Tools">
          <IconButton 
            size="small" 
            color={drawingMode ? 'primary' : 'default'}
            onClick={() => setDrawingMode(drawingMode ? null : 'trendline')}
          >
            <BrushIcon />
          </IconButton>
        </Tooltip>

        <Tooltip title="Compare">
          <IconButton size="small">
            <CompareArrowsIcon />
          </IconButton>
        </Tooltip>

        <Box flex={1} />

        <Stack direction="row" spacing={1}>
          {indicators.map(indicator => (
            <Chip
              key={indicator.id}
              label={indicator.name}
              size="small"
              onDelete={() => setIndicators(indicators.filter(i => i.id !== indicator.id))}
            />
          ))}
        </Stack>
      </ChartToolbar>

      {/* Chart Area */}
      <ChartArea>
        <Line ref={chartRef} data={chartData} options={chartOptions} />
        
        {/* Signal Overlay */}
        {signals.length > 0 && (
          <SignalOverlay sx={{ top: 16, right: 16 }}>
            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
              Active Signals
            </Typography>
            {signals.map((signal, index) => (
              <Box key={index} mb={1}>
                <Chip
                  label={signal.type.toUpperCase()}
                  color={signal.type === 'buy' ? 'success' : 'error'}
                  size="small"
                  sx={{ mr: 1 }}
                />
                <Typography variant="caption">
                  ${signal.price.toFixed(2)} • {(signal.confidence * 100).toFixed(1)}%
                </Typography>
              </Box>
            ))}
          </SignalOverlay>
        )}
      </ChartArea>

      {/* Indicator Menu */}
      <Menu
        anchorEl={indicatorMenuAnchor}
        open={Boolean(indicatorMenuAnchor)}
        onClose={handleIndicatorMenuClose}
      >
        <MenuItem disabled>
          <Typography variant="subtitle2">Add Indicator</Typography>
        </MenuItem>
        <Divider />
        {availableIndicators.map(indicator => (
          <MenuItem key={indicator.id} onClick={() => handleAddIndicator(indicator.id)}>
            <Box>
              <Typography variant="body2">{indicator.name}</Typography>
              <Typography variant="caption" color="text.secondary">
                {indicator.category}
              </Typography>
            </Box>
          </MenuItem>
        ))}
      </Menu>
    </ChartContainer>
  );
};

export default CentralChart;