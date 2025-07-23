import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  TextField,
  Button,
  Paper,
  Autocomplete,
  Chip,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
  InputAdornment,
  CircularProgress,
  alpha,
  useTheme,
  Popper,
} from '@mui/material';
import {
  Search as SearchIcon,
  TrendingUp as TrendingUpIcon,
  Schedule as ScheduleIcon,
  Analytics as AnalyticsIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { motion } from 'framer-motion';
import debounce from 'lodash/debounce';
import logger from '../../services/logger';


// Styled components
const SearchContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.05)} 0%, ${alpha(
    theme.palette.primary.main,
    0.02
  )} 100%)`,
  border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
  borderRadius: theme.spacing(1.5),
  transition: 'all 0.3s ease',
  '&:hover': {
    boxShadow: theme.shadows[4],
    borderColor: alpha(theme.palette.primary.main, 0.3),
  },
}));

const TimeframeButton = styled(ToggleButton)(({ theme }) => ({
  padding: theme.spacing(1, 2),
  borderRadius: theme.spacing(1),
  textTransform: 'none',
  fontWeight: 600,
  '&.Mui-selected': {
    backgroundColor: theme.palette.primary.main,
    color: theme.palette.primary.contrastText,
    '&:hover': {
      backgroundColor: theme.palette.primary.dark,
    },
  },
}));

const AnalyzeButton = styled(Button)(({ theme }) => ({
  padding: theme.spacing(1.5, 4),
  borderRadius: theme.spacing(1),
  fontSize: '1rem',
  fontWeight: 600,
  textTransform: 'none',
  background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
  boxShadow: `0 4px 12px ${alpha(theme.palette.primary.main, 0.3)}`,
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: `0 6px 20px ${alpha(theme.palette.primary.main, 0.4)}`,
  },
  '&:active': {
    transform: 'translateY(0)',
  },
}));

const CustomPopper = styled(Popper)(({ theme }) => ({
  '& .MuiAutocomplete-paper': {
    marginTop: theme.spacing(1),
    boxShadow: theme.shadows[8],
    borderRadius: theme.spacing(1),
    border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
  },
}));

// Types
interface StockOption {
  symbol: string;
  name: string;
  exchange: string;
  type: 'stock' | 'etf' | 'crypto';
  marketCap?: string;
  sector?: string;
}

export interface TradeSearchProps {
  onSubmit: (symbol: string, timeframe: string) => void;
  onSymbolChange?: (symbol: string) => void;
  onTimeframeChange?: (timeframe: string) => void;
  defaultSymbol?: string;
  defaultTimeframe?: string;
  isAnalyzing?: boolean;
  compact?: boolean;
  hideTimeframe?: boolean;
}

// Mock data - replace with API call
const mockStockOptions: StockOption[] = [
  { symbol: 'AAPL', name: 'Apple Inc.', exchange: 'NASDAQ', type: 'stock', marketCap: '$2.95T', sector: 'Technology' },
  { symbol: 'MSFT', name: 'Microsoft Corporation', exchange: 'NASDAQ', type: 'stock', marketCap: '$2.85T', sector: 'Technology' },
  { symbol: 'GOOGL', name: 'Alphabet Inc.', exchange: 'NASDAQ', type: 'stock', marketCap: '$1.75T', sector: 'Technology' },
  { symbol: 'AMZN', name: 'Amazon.com Inc.', exchange: 'NASDAQ', type: 'stock', marketCap: '$1.56T', sector: 'Consumer Cyclical' },
  { symbol: 'NVDA', name: 'NVIDIA Corporation', exchange: 'NASDAQ', type: 'stock', marketCap: '$1.12T', sector: 'Technology' },
  { symbol: 'TSLA', name: 'Tesla Inc.', exchange: 'NASDAQ', type: 'stock', marketCap: '$795B', sector: 'Automotive' },
  { symbol: 'META', name: 'Meta Platforms Inc.', exchange: 'NASDAQ', type: 'stock', marketCap: '$892B', sector: 'Technology' },
  { symbol: 'BRK.B', name: 'Berkshire Hathaway Inc.', exchange: 'NYSE', type: 'stock', marketCap: '$785B', sector: 'Financial' },
  { symbol: 'SPY', name: 'SPDR S&P 500 ETF', exchange: 'NYSE', type: 'etf', marketCap: '$423B' },
  { symbol: 'BTC-USD', name: 'Bitcoin', exchange: 'Crypto', type: 'crypto', marketCap: '$680B' },
];

const timeframes = [
  { value: '5m', label: '5m', description: '5 minutes' },
  { value: '15m', label: '15m', description: '15 minutes' },
  { value: '30m', label: '30m', description: '30 minutes' },
  { value: '1h', label: '1h', description: '1 hour' },
  { value: '4h', label: '4h', description: '4 hours' },
  { value: '1d', label: '1D', description: '1 day' },
  { value: '1w', label: '1W', description: '1 week' },
];

export const TradeSearch: React.FC<TradeSearchProps> = ({
  onSubmit,
  onSymbolChange,
  onTimeframeChange,
  defaultSymbol = '',
  defaultTimeframe = '1d',
  isAnalyzing = false,
  compact = false,
  hideTimeframe = false,
}) => {
  const theme = useTheme();
  const [selectedStock, setSelectedStock] = useState<StockOption | null>(null);
  const [inputValue, setInputValue] = useState(defaultSymbol);
  const [timeframe, setTimeframe] = useState(defaultTimeframe);
  const [options, setOptions] = useState<StockOption[]>(mockStockOptions);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);

  // Use external analyzing state if provided
  const isCurrentlyAnalyzing = isAnalyzing || analyzing;

  // Debounced search function
  const searchStocks = useCallback(
    debounce(async (query: string) => {
      if (!query) {
        setOptions(mockStockOptions);
        return;
      }

      setLoading(true);
      try {
        // Simulate API call - replace with actual API
        await new Promise(resolve => setTimeout(resolve, 300));

        const filtered = mockStockOptions.filter(
          stock =>
            stock.symbol.toLowerCase().includes(query.toLowerCase()) ||
            stock.name.toLowerCase().includes(query.toLowerCase())
        );

        setOptions(filtered);
      } catch (error) {
        logger.error('Error searching stocks:', error);
      } finally {
        setLoading(false);
      }
    }, 300),
    []
  );

  useEffect(() => {
    searchStocks(inputValue);
  }, [inputValue, searchStocks]);

  const handleAnalyze = async () => {
    if (!selectedStock) return;

    setAnalyzing(true);
    try {
      await onSubmit(selectedStock.symbol, timeframe);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleTimeframeChange = (event: React.MouseEvent<HTMLElement>, newTimeframe: string | null) => {
    if (newTimeframe) {
      setTimeframe(newTimeframe);
      onTimeframeChange?.(newTimeframe);
    }
  };

  const handleStockChange = (event: any, newValue: StockOption | null) => {
    setSelectedStock(newValue);
    if (newValue) {
      onSymbolChange?.(newValue.symbol);
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'stock':
        return theme.palette.primary.main;
      case 'etf':
        return theme.palette.success.main;
      case 'crypto':
        return theme.palette.warning.main;
      default:
        return theme.palette.grey[500];
    }
  };

  // Compact mode for header display
  if (compact) {
    return (
      <Box display="flex" gap={1} alignItems="center">
        <Autocomplete
          sx={{ width: 250 }}
          size="small"
          value={selectedStock}
          onChange={(event, newValue) => {
            setSelectedStock(newValue);
            if (newValue) {
              onSymbolChange?.(newValue.symbol);
              // Auto-analyze when symbol changes in compact mode
              onSubmit(newValue.symbol, timeframe);
            }
          }}
          inputValue={inputValue}
          onInputChange={(event, newInputValue) => {
            setInputValue(newInputValue);
          }}
          options={options}
          loading={loading}
          getOptionLabel={(option) => option.symbol}
          renderInput={(params) => (
            <TextField
              {...params}
              placeholder="Search symbol..."
              size="small"
              InputProps={{
                ...params.InputProps,
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon fontSize="small" />
                  </InputAdornment>
                ),
                endAdornment: (
                  <>
                    {loading ? <CircularProgress color="inherit" size={20} /> : null}
                    {params.InputProps.endAdornment}
                  </>
                ),
              }}
            />
          )}
          renderOption={(props, option) => (
            <Box component="li" {...props}>
              <Box>
                <Typography variant="body2" fontWeight="bold">
                  {option.symbol}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {option.name}
                </Typography>
              </Box>
            </Box>
          )}
        />

        {!hideTimeframe && (
          <ToggleButtonGroup
            size="small"
            value={timeframe}
            exclusive
            onChange={handleTimeframeChange}
          >
            {timeframes.slice(0, 5).map((tf) => (
              <TimeframeButton key={tf.value} value={tf.value} size="small">
                {tf.label}
              </TimeframeButton>
            ))}
          </ToggleButtonGroup>
        )}

        <Button
          variant="contained"
          size="small"
          startIcon={isCurrentlyAnalyzing ? <CircularProgress size={16} color="inherit" /> : <AnalyticsIcon />}
          onClick={handleAnalyze}
          disabled={!selectedStock || isCurrentlyAnalyzing}
        >
          {isCurrentlyAnalyzing ? 'Analyzing...' : 'Analyze'}
        </Button>
      </Box>
    );
  }

  // Full mode for standalone display
  return (
    <SearchContainer elevation={0}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Box display="flex" flexDirection="column" gap={3}>
          {/* Search Input */}
          <Box>
            <Typography variant="h5" fontWeight="bold" gutterBottom>
              Analyze Stocks with AI-Powered Signals
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Enter a ticker symbol or company name to generate high-accuracy trading signals
            </Typography>
          </Box>

          <Box display="flex" gap={2} alignItems="flex-start" flexWrap="wrap">
            <Autocomplete
              sx={{ flex: 1, minWidth: 300 }}
              options={options}
              getOptionLabel={(option) => `${option.symbol} - ${option.name}`}
              value={selectedStock}
              onChange={handleStockChange}
              inputValue={inputValue}
              onInputChange={(event, newInputValue) => setInputValue(newInputValue)}
              loading={loading}
              PopperComponent={CustomPopper}
              renderInput={(params) => (
                <TextField
                  {...params}
                  placeholder="Search by ticker or company name..."
                  variant="outlined"
                  InputProps={{
                    ...params.InputProps,
                    startAdornment: (
                      <InputAdornment position="start">
                        <SearchIcon color="action" />
                      </InputAdornment>
                    ),
                    endAdornment: (
                      <>
                        {loading ? <CircularProgress color="inherit" size={20} /> : null}
                        {params.InputProps.endAdornment}
                      </>
                    ),
                  }}
                />
              )}
              renderOption={(props, option) => (
                <Box component="li" {...props}>
                  <Box display="flex" alignItems="center" width="100%" gap={2}>
                    <Box flex={1}>
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography variant="subtitle1" fontWeight="600">
                          {option.symbol}
                        </Typography>
                        <Chip
                          label={option.type.toUpperCase()}
                          size="small"
                          sx={{
                            backgroundColor: alpha(getTypeColor(option.type), 0.1),
                            color: getTypeColor(option.type),
                            fontWeight: 600,
                            fontSize: '0.7rem',
                          }}
                        />
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {option.name}
                      </Typography>
                    </Box>
                    <Box textAlign="right">
                      <Typography variant="caption" color="text.secondary">
                        {option.exchange}
                      </Typography>
                      {option.marketCap && (
                        <Typography variant="caption" display="block" fontWeight="600">
                          {option.marketCap}
                        </Typography>
                      )}
                    </Box>
                  </Box>
                </Box>
              )}
            />

            <AnalyzeButton
              variant="contained"
              size="large"
              startIcon={isCurrentlyAnalyzing ? <CircularProgress size={20} color="inherit" /> : <AnalyticsIcon />}
              onClick={handleAnalyze}
              disabled={!selectedStock || isCurrentlyAnalyzing}
            >
              {isCurrentlyAnalyzing ? 'Analyzing...' : 'Analyze'}
            </AnalyzeButton>
          </Box>

          {/* Timeframe Selector */}
          {!hideTimeframe && (
            <Box>
              <Box display="flex" alignItems="center" gap={1} mb={2}>
                <ScheduleIcon fontSize="small" color="action" />
                <Typography variant="subtitle2" color="text.secondary">
                  Select Timeframe
                </Typography>
              </Box>
              <ToggleButtonGroup
                value={timeframe}
                exclusive
                onChange={handleTimeframeChange}
                aria-label="timeframe selection"
              >
                {timeframes.map((tf) => (
                  <TimeframeButton key={tf.value} value={tf.value} aria-label={tf.description}>
                    {tf.label}
                  </TimeframeButton>
                ))}
              </ToggleButtonGroup>
            </Box>
          )}

          {/* Popular Searches */}
          <Box>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Popular Searches
            </Typography>
            <Box display="flex" gap={1} flexWrap="wrap">
              {['AAPL', 'TSLA', 'SPY', 'NVDA', 'BTC-USD'].map((symbol) => (
                <Chip
                  key={symbol}
                  label={symbol}
                  onClick={() => {
                    const stock = mockStockOptions.find(s => s.symbol === symbol);
                    if (stock) {
                      setSelectedStock(stock);
                      setInputValue(symbol);
                      onSymbolChange?.(symbol);
                    }
                  }}
                  icon={<TrendingUpIcon />}
                  variant="outlined"
                  sx={{
                    cursor: 'pointer',
                    '&:hover': {
                      backgroundColor: alpha(theme.palette.primary.main, 0.1),
                      borderColor: theme.palette.primary.main,
                    },
                  }}
                />
              ))}
            </Box>
          </Box>
        </Box>
      </motion.div>
    </SearchContainer>
  );
};
