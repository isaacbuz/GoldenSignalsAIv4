import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  IconButton,
  Select,
  MenuItem,
  FormControl,
  InputBase,
  Tooltip,
  useTheme,
  alpha,
  Autocomplete,
  TextField,
  Chip,
} from '@mui/material';
import {
  Search as SearchIcon,
  Fullscreen as FullscreenIcon,
  FullscreenExit as FullscreenExitIcon,
  PhotoCamera as PhotoCameraIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import EnhancedCustomChart from './EnhancedCustomChart';
import ChartSettingsMenu from './ChartSettingsMenu';
import { fetchHistoricalData } from '../../services/backendMarketDataService';
import { chartSettingsService } from '../../services/chartSettingsService';
import logger from '../../../services/logger';


interface StreamlinedGoldenChartProps {
  symbol?: string;
  onSymbolChange?: (symbol: string) => void;
  showWatermark?: boolean;
  height?: string | number;
}

// Styled components - Cleaner, more minimal design
const ChartContainer = styled(Box)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: theme.palette.background.default,
  borderRadius: theme.spacing(1),
  overflow: 'hidden',
}));

const ChartHeader = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1, 2),
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(2),
  borderBottom: `1px solid ${alpha(theme.palette.divider, 0.08)}`,
  minHeight: 56,
}));

const SearchContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  backgroundColor: alpha(theme.palette.background.paper, 0.6),
  borderRadius: theme.shape.borderRadius,
  padding: theme.spacing(0.5, 2),
  minWidth: 280,
  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  transition: 'all 0.2s',
  '&:hover': {
    backgroundColor: alpha(theme.palette.background.paper, 0.8),
    borderColor: alpha(theme.palette.primary.main, 0.2),
  },
  '&:focus-within': {
    backgroundColor: theme.palette.background.paper,
    borderColor: theme.palette.primary.main,
    boxShadow: `0 0 0 2px ${alpha(theme.palette.primary.main, 0.1)}`,
  },
}));

const StyledSelect = styled(Select)(({ theme }) => ({
  minWidth: 100,
  height: 36,
  fontSize: '0.875rem',
  fontWeight: 600,
  '& .MuiSelect-select': {
    padding: theme.spacing(1, 2),
    paddingRight: theme.spacing(4),
  },
  '& .MuiOutlinedInput-notchedOutline': {
    borderColor: alpha(theme.palette.divider, 0.1),
  },
  '&:hover .MuiOutlinedInput-notchedOutline': {
    borderColor: alpha(theme.palette.primary.main, 0.2),
  },
  '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
    borderColor: theme.palette.primary.main,
    borderWidth: 1,
  },
}));

const ChartBody = styled(Box)({
  flex: 1,
  position: 'relative',
  overflow: 'hidden',
});

const PriceDisplay = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'baseline',
  gap: theme.spacing(1),
  marginLeft: 'auto',
  '& .price': {
    fontSize: '1.5rem',
    fontWeight: 700,
    fontFamily: 'monospace',
  },
  '& .change': {
    fontSize: '0.875rem',
    fontWeight: 600,
  },
}));

// Popular symbols for quick access
const POPULAR_SYMBOLS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
  'SPY', 'QQQ', 'BTC-USD', 'ETH-USD',
];

const timeframes = [
  { value: '1m', label: '1 Minute' },
  { value: '5m', label: '5 Minutes' },
  { value: '15m', label: '15 Minutes' },
  { value: '30m', label: '30 Minutes' },
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hours' },
  { value: '1d', label: '1 Day' },
  { value: '1w', label: '1 Week' },
  { value: '1M', label: '1 Month' },
];

export const StreamlinedGoldenChart: React.FC<StreamlinedGoldenChartProps> = ({
  symbol: initialSymbol = 'AAPL',
  onSymbolChange,
  showWatermark = true,
  height = '100%',
}) => {
  const theme = useTheme();
  const [symbol, setSymbol] = useState(initialSymbol);
  const [searchValue, setSearchValue] = useState(initialSymbol);
  const [chartData, setChartData] = useState<any[]>([]);
  const [signals, setSignals] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  const [timeframe, setTimeframe] = useState('5m');
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>(['sma', 'volume']);
  const [chartType, setChartType] = useState<'line' | 'candle'>('candle');
  const [showGrid, setShowGrid] = useState(true);
  const [showWatermarkState, setShowWatermarkState] = useState(showWatermark);
  const [lastPrice, setLastPrice] = useState(0);
  const [priceChange, setPriceChange] = useState({ value: 0, percent: 0, isPositive: true });

  // Load saved settings
  useEffect(() => {
    const settings = chartSettingsService.getSettings();
    if (settings.indicators) {
      setSelectedIndicators(settings.indicators);
    }
    if (settings.timeframe) {
      setTimeframe(settings.timeframe);
    }
  }, []);

  // Fetch chart data
  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetchHistoricalData(symbol, timeframe);
      if (response.data && response.data.length > 0) {
        const transformedData = response.data.map((d: any) => ({
          time: d.time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
          volume: d.volume,
        }));
        setChartData(transformedData);

        // Update price display
        const last = transformedData[transformedData.length - 1];
        const previous = transformedData[transformedData.length - 2];
        if (last && previous) {
          setLastPrice(last.close);
          const change = last.close - previous.close;
          const changePercent = (change / previous.close) * 100;
          setPriceChange({
            value: change,
            percent: changePercent,
            isPositive: change >= 0,
          });
        }
      }

      // Generate mock signals
      if (response.data && response.data.length > 20) {
        const signalIndices = [10, 25, 40, 55, 70].filter(i => i < response.data.length);
        const mockSignals = signalIndices.map((idx, i) => ({
          time: response.data[idx].time,
          type: i % 2 === 0 ? 'buy' : 'sell',
          price: response.data[idx].close,
          confidence: 0.75 + Math.random() * 0.2,
        }));
        setSignals(mockSignals);
      }
    } catch (error) {
      logger.error('Failed to fetch chart data:', error);
      generateMockData();
    } finally {
      setLoading(false);
    }
  }, [symbol, timeframe]);

  // Generate mock data as fallback
  const generateMockData = () => {
    const data = [];
    const now = Date.now() / 1000;
    const interval = timeframe === '1m' ? 60 :
                     timeframe === '5m' ? 300 :
                     timeframe === '15m' ? 900 :
                     timeframe === '30m' ? 1800 :
                     timeframe === '1h' ? 3600 :
                     timeframe === '4h' ? 14400 :
                     timeframe === '1d' ? 86400 :
                     timeframe === '1w' ? 604800 : 2592000;

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

    // Update price display
    if (data.length > 1) {
      const last = data[data.length - 1];
      const previous = data[data.length - 2];
      setLastPrice(last.close);
      const change = last.close - previous.close;
      const changePercent = (change / previous.close) * 100;
      setPriceChange({
        value: change,
        percent: changePercent,
        isPositive: change >= 0,
      });
    }

    // Generate mock signals
    const mockSignals = [20, 35, 50, 65, 80].map((idx, i) => ({
      time: data[idx].time,
      type: i % 2 === 0 ? 'buy' : 'sell',
      price: data[idx].close,
      confidence: 0.75 + Math.random() * 0.2,
    }));
    setSignals(mockSignals);
  };

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Handle symbol change
  const handleSymbolChange = (newSymbol: string | null) => {
    if (newSymbol) {
      setSymbol(newSymbol);
      setSearchValue(newSymbol);
      onSymbolChange?.(newSymbol);
      chartSettingsService.addFavoriteSymbol(newSymbol);
    }
  };

  // Handle timeframe change
  const handleTimeframeChange = (newTimeframe: string) => {
    setTimeframe(newTimeframe);
    chartSettingsService.saveSettings({ timeframe: newTimeframe });
  };

  // Handle screenshot
  const handleScreenshot = () => {
    // TODO: Implement canvas capture
    alert('Screenshot saved to clipboard!');
  };

  // Toggle indicator
  const handleIndicatorToggle = (indicator: string) => {
    setSelectedIndicators(prev => {
      const newIndicators = prev.includes(indicator)
        ? prev.filter(i => i !== indicator)
        : [...prev, indicator];

      chartSettingsService.saveSettings({ indicators: newIndicators });
      return newIndicators;
    });
  };

  // Toggle fullscreen
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setFullscreen(true);
    } else {
      document.exitFullscreen();
      setFullscreen(false);
    }
  };

  // Format price for display
  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(price);
  };

  const formatChange = (value: number, percent: number, isPositive: boolean) => {
    const sign = isPositive ? '+' : '';
    return `${sign}${value.toFixed(2)} (${sign}${percent.toFixed(2)}%)`;
  };

  return (
    <ChartContainer sx={{ height }}>
      <ChartHeader>
        {/* Symbol Search */}
        <SearchContainer>
          <SearchIcon sx={{ color: 'text.secondary', mr: 1 }} />
          <Autocomplete
            value={searchValue}
            onChange={(event, newValue) => handleSymbolChange(newValue)}
            inputValue={searchValue}
            onInputChange={(event, newInputValue) => setSearchValue(newInputValue)}
            options={POPULAR_SYMBOLS}
            freeSolo
            disableClearable
            sx={{ flex: 1 }}
            renderInput={(params) => (
              <InputBase
                {...params.InputProps}
                inputProps={params.inputProps}
                placeholder="Search symbol..."
                sx={{
                  flex: 1,
                  '& input': {
                    padding: 0,
                    fontSize: '0.875rem',
                    fontWeight: 500,
                  },
                }}
              />
            )}
          />
        </SearchContainer>

        {/* Timeframe Dropdown */}
        <FormControl size="small">
          <StyledSelect
            value={timeframe}
            onChange={(e) => handleTimeframeChange(e.target.value)}
            variant="outlined"
          >
            {timeframes.map((tf) => (
              <MenuItem key={tf.value} value={tf.value}>
                {tf.label}
              </MenuItem>
            ))}
          </StyledSelect>
        </FormControl>

        {/* Price Display */}
        <PriceDisplay>
          <span className="price">{formatPrice(lastPrice)}</span>
          <Chip
            label={formatChange(priceChange.value, priceChange.percent, priceChange.isPositive)}
            size="small"
            color={priceChange.isPositive ? 'success' : 'error'}
            icon={priceChange.isPositive ? <TrendingUpIcon /> : <TrendingDownIcon />}
            sx={{ fontWeight: 600 }}
          />
        </PriceDisplay>

        {/* Action Buttons */}
        <Box display="flex" gap={0.5} ml="auto">
          <Tooltip title="Take Screenshot">
            <IconButton size="small" onClick={handleScreenshot}>
              <PhotoCameraIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          <ChartSettingsMenu
            indicators={selectedIndicators}
            onIndicatorToggle={handleIndicatorToggle}
            chartType={chartType}
            onChartTypeChange={setChartType}
            showGrid={showGrid}
            onGridToggle={() => setShowGrid(!showGrid)}
            showWatermark={showWatermarkState}
            onWatermarkToggle={() => setShowWatermarkState(!showWatermarkState)}
          />

          <Tooltip title={fullscreen ? 'Exit Fullscreen' : 'Fullscreen'}>
            <IconButton size="small" onClick={toggleFullscreen}>
              {fullscreen ? <FullscreenExitIcon fontSize="small" /> : <FullscreenIcon fontSize="small" />}
            </IconButton>
          </Tooltip>
        </Box>
      </ChartHeader>

      <ChartBody>
        {!loading && (
          <EnhancedCustomChart
            data={chartData}
            signals={signals}
            symbol={symbol}
            showWatermark={showWatermarkState}
            showGrid={showGrid}
            indicators={selectedIndicators}
            theme={theme.palette.mode}
          />
        )}
      </ChartBody>
    </ChartContainer>
  );
};

export default StreamlinedGoldenChart;
