import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  IconButton,
  Menu,
  MenuItem,
  Divider,
  Typography,
  Chip,
  Stack,
  ToggleButton,
  ToggleButtonGroup,
  useTheme,
  alpha,
  Tooltip,
} from '@mui/material';
import {
  Fullscreen as FullscreenIcon,
  FullscreenExit as FullscreenExitIcon,
  Settings as SettingsIcon,
  Save as SaveIcon,
  PhotoCamera as PhotoCameraIcon,
  Timeline as TimelineIcon,
  ShowChart as ShowChartIcon,
  BarChart as BarChartIcon,
  CandlestickChart as CandlestickChartIcon,
  Analytics as AnalyticsIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import EnhancedCustomChart from './EnhancedCustomChart';
import { fetchHistoricalData } from '../../services/backendMarketDataService';
import { chartSettingsService } from '../../services/chartSettingsService';
import logger from '../../../services/logger';


interface GoldenSignalsChartProps {
  symbol?: string;
  timeframe?: string;
  onSymbolChange?: (symbol: string) => void;
  onTimeframeChange?: (timeframe: string) => void;
  showWatermark?: boolean;
  height?: string | number;
}

const ChartContainer = styled(Paper)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: theme.palette.background.default,
  borderRadius: theme.spacing(2),
  overflow: 'hidden',
  boxShadow: theme.palette.mode === 'dark'
    ? `0 8px 32px ${alpha('#000', 0.4)}`
    : `0 8px 32px ${alpha('#000', 0.1)}`,
}));

const ChartHeader = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1.5, 2),
  borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  backgroundColor: alpha(theme.palette.background.paper, 0.5),
  backdropFilter: 'blur(10px)',
}));

const ChartBody = styled(Box)({
  flex: 1,
  position: 'relative',
  overflow: 'hidden',
});

const TimeframeButton = styled(ToggleButton)(({ theme }) => ({
  padding: theme.spacing(0.5, 1.5),
  fontSize: '0.875rem',
  fontWeight: 600,
  '&.Mui-selected': {
    backgroundColor: alpha(theme.palette.primary.main, 0.2),
    '&:hover': {
      backgroundColor: alpha(theme.palette.primary.main, 0.3),
    },
  },
}));

const timeframes = [
  { value: '5m', label: '5M' },
  { value: '15m', label: '15M' },
  { value: '1h', label: '1H' },
  { value: '4h', label: '4H' },
  { value: '1d', label: '1D' },
  { value: '1w', label: '1W' },
];

export const GoldenSignalsChart: React.FC<GoldenSignalsChartProps> = ({
  symbol = 'AAPL',
  timeframe = '5m',
  onSymbolChange,
  onTimeframeChange,
  showWatermark = true,
  height = '100%',
}) => {
  const theme = useTheme();
  const [chartData, setChartData] = useState<any[]>([]);
  const [signals, setSignals] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState(timeframe);
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>(['sma', 'volume']);
  const [settingsAnchor, setSettingsAnchor] = useState<null | HTMLElement>(null);
  const [chartType, setChartType] = useState<'line' | 'candle'>('candle');

  // Load saved settings
  useEffect(() => {
    const settings = chartSettingsService.getSettings();
    if (settings.indicators) {
      setSelectedIndicators(settings.indicators);
    }
    if (settings.timeframe) {
      setSelectedTimeframe(settings.timeframe);
    }
  }, []);

  // Fetch chart data
  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetchHistoricalData(symbol, selectedTimeframe);
      if (response.data) {
        // Transform data for our custom chart
        const transformedData = response.data.map((d: any) => ({
          time: d.time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
          volume: d.volume,
        }));
        setChartData(transformedData);
      }

      // Generate some mock signals for demonstration
      const mockSignals = response.data.slice(0, 5).map((d: any, i: number) => ({
        time: d.time,
        type: i % 2 === 0 ? 'buy' : 'sell',
        price: d.close,
        confidence: 0.75 + Math.random() * 0.2,
      }));
      setSignals(mockSignals);
    } catch (error) {
      logger.error('Failed to fetch chart data:', error);
      // Use mock data as fallback
      generateMockData();
    } finally {
      setLoading(false);
    }
  }, [symbol, selectedTimeframe]);

  // Generate mock data
  const generateMockData = () => {
    const data = [];
    const now = Date.now() / 1000;
    const interval = selectedTimeframe === '5m' ? 300 :
                     selectedTimeframe === '15m' ? 900 :
                     selectedTimeframe === '1h' ? 3600 :
                     selectedTimeframe === '4h' ? 14400 :
                     selectedTimeframe === '1d' ? 86400 : 604800;

    let price = 150;
    for (let i = 0; i < 100; i++) {
      const time = now - (100 - i) * interval;
      const volatility = 0.02;
      const trend = Math.sin(i / 20) * 5;

      const open = price;
      const change = (Math.random() - 0.5) * volatility * price;
      const high = Math.max(open, open + Math.random() * volatility * price);
      const low = Math.min(open, open - Math.random() * volatility * price);
      const close = open + change + trend * 0.1;
      const volume = 1000000 + Math.random() * 5000000;

      data.push({ time, open, high, low, close, volume });
      price = close;
    }

    setChartData(data);

    // Generate mock signals
    const mockSignals = data.slice(20, 25).map((d, i) => ({
      time: d.time,
      type: i % 2 === 0 ? 'buy' : 'sell',
      price: d.close,
      confidence: 0.75 + Math.random() * 0.2,
    }));
    setSignals(mockSignals);
  };

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Handle timeframe change
  const handleTimeframeChange = (event: React.MouseEvent<HTMLElement>, newTimeframe: string | null) => {
    if (newTimeframe) {
      setSelectedTimeframe(newTimeframe);
      onTimeframeChange?.(newTimeframe);
      chartSettingsService.saveSettings({ timeframe: newTimeframe });
    }
  };

  // Handle indicator toggle
  const toggleIndicator = (indicator: string) => {
    setSelectedIndicators(prev => {
      const newIndicators = prev.includes(indicator)
        ? prev.filter(i => i !== indicator)
        : [...prev, indicator];

      chartSettingsService.saveSettings({ indicators: newIndicators });
      return newIndicators;
    });
  };

  // Handle save
  const handleSave = () => {
    const layoutName = prompt('Enter a name for this chart layout:');
    if (layoutName) {
      chartSettingsService.saveLayout(layoutName, symbol, {
        timeframe: selectedTimeframe,
        indicators: selectedIndicators,
        showWatermark,
      });
      alert(`Layout "${layoutName}" saved!`);
    }
  };

  // Handle screenshot
  const handleScreenshot = () => {
    // In a real implementation, this would capture the canvas
    alert('Screenshot feature coming soon!');
  };

  // Render loading state
  if (loading) {
    return (
      <ChartContainer sx={{ height }}>
        <Box
          display="flex"
          alignItems="center"
          justifyContent="center"
          height="100%"
        >
          <Typography variant="h6" color="text.secondary">
            Loading chart data...
          </Typography>
        </Box>
      </ChartContainer>
    );
  }

  return (
    <ChartContainer sx={{ height }}>
      <ChartHeader>
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="h6" fontWeight="bold">
            {symbol}
          </Typography>

          <ToggleButtonGroup
            value={selectedTimeframe}
            exclusive
            onChange={handleTimeframeChange}
            size="small"
          >
            {timeframes.map(tf => (
              <TimeframeButton key={tf.value} value={tf.value}>
                {tf.label}
              </TimeframeButton>
            ))}
          </ToggleButtonGroup>
        </Box>

        <Box display="flex" alignItems="center" gap={1}>
          <Tooltip title="Save Layout">
            <IconButton size="small" onClick={handleSave}>
              <SaveIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Screenshot">
            <IconButton size="small" onClick={handleScreenshot}>
              <PhotoCameraIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Settings">
            <IconButton
              size="small"
              onClick={(e) => setSettingsAnchor(e.currentTarget)}
            >
              <SettingsIcon />
            </IconButton>
          </Tooltip>

          <Divider orientation="vertical" flexItem />

          <Tooltip title={fullscreen ? 'Exit Fullscreen' : 'Fullscreen'}>
            <IconButton
              size="small"
              onClick={() => setFullscreen(!fullscreen)}
            >
              {fullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
            </IconButton>
          </Tooltip>
        </Box>
      </ChartHeader>

      <ChartBody>
        <EnhancedCustomChart
          data={chartData}
          signals={signals}
          symbol={symbol}
          showWatermark={showWatermark}
          indicators={selectedIndicators}
          theme={theme.palette.mode}
        />
      </ChartBody>

      {/* Settings Menu */}
      <Menu
        anchorEl={settingsAnchor}
        open={Boolean(settingsAnchor)}
        onClose={() => setSettingsAnchor(null)}
      >
        <MenuItem disabled>
          <Typography variant="body2">Indicators</Typography>
        </MenuItem>
        <Divider />

        <MenuItem onClick={() => toggleIndicator('sma')}>
          <Box display="flex" alignItems="center" gap={1} width="100%">
            <TimelineIcon fontSize="small" />
            <Typography variant="body2" flex={1}>Moving Averages</Typography>
            {selectedIndicators.includes('sma') && '✓'}
          </Box>
        </MenuItem>

        <MenuItem onClick={() => toggleIndicator('volume')}>
          <Box display="flex" alignItems="center" gap={1} width="100%">
            <BarChartIcon fontSize="small" />
            <Typography variant="body2" flex={1}>Volume</Typography>
            {selectedIndicators.includes('volume') && '✓'}
          </Box>
        </MenuItem>

        <MenuItem onClick={() => toggleIndicator('rsi')}>
          <Box display="flex" alignItems="center" gap={1} width="100%">
            <ShowChartIcon fontSize="small" />
            <Typography variant="body2" flex={1}>RSI</Typography>
            {selectedIndicators.includes('rsi') && '✓'}
          </Box>
        </MenuItem>

        <MenuItem onClick={() => toggleIndicator('macd')}>
          <Box display="flex" alignItems="center" gap={1} width="100%">
            <AnalyticsIcon fontSize="small" />
            <Typography variant="body2" flex={1}>MACD</Typography>
            {selectedIndicators.includes('macd') && '✓'}
          </Box>
        </MenuItem>

        <Divider />

        <MenuItem disabled>
          <Typography variant="body2">Chart Type</Typography>
        </MenuItem>

        <MenuItem onClick={() => setChartType('candle')}>
          <Box display="flex" alignItems="center" gap={1} width="100%">
            <CandlestickChartIcon fontSize="small" />
            <Typography variant="body2" flex={1}>Candlestick</Typography>
            {chartType === 'candle' && '✓'}
          </Box>
        </MenuItem>

        <MenuItem onClick={() => setChartType('line')}>
          <Box display="flex" alignItems="center" gap={1} width="100%">
            <ShowChartIcon fontSize="small" />
            <Typography variant="body2" flex={1}>Line</Typography>
            {chartType === 'line' && '✓'}
          </Box>
        </MenuItem>
      </Menu>
    </ChartContainer>
  );
};

export default GoldenSignalsChart;
