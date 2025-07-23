import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Chip,
  IconButton,
  TextField,
  Typography,
  Autocomplete,
  ToggleButton,
  ToggleButtonGroup,
  useTheme,
  alpha,
  Stack,
  Tooltip,
} from '@mui/material';
import {
  Add as AddIcon,
  Close as CloseIcon,
  Timeline as TimelineIcon,
  ShowChart as ShowChartIcon,
  Percent as PercentIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  LineData,
  Time,
  CrosshairMode,
  ColorType,
} from 'lightweight-charts';
import { fetchHistoricalData } from '../../services/backendMarketDataService';
import logger from '../../../services/logger';


const ChartContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  height: '100%',
  backgroundColor: 'transparent',
  borderRadius: theme.spacing(1),
  overflow: 'hidden',
}));

const ControlPanel = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(2),
  left: theme.spacing(2),
  zIndex: 10,
  backgroundColor: alpha(theme.palette.background.paper, 0.95),
  borderRadius: theme.spacing(1),
  padding: theme.spacing(1.5),
  border: `1px solid ${alpha(theme.palette.divider, 0.2)}`,
  maxWidth: 400,
}));

const SymbolChip = styled(Chip)(({ theme }) => ({
  margin: theme.spacing(0.5),
  fontWeight: 600,
}));

export interface ComparisonChartProps {
  primarySymbol: string;
  timeframe?: string;
  height?: string | number;
  onSymbolChange?: (symbol: string) => void;
}

type ComparisonMode = 'absolute' | 'percentage' | 'normalized';

interface SymbolData {
  symbol: string;
  color: string;
  series: ISeriesApi<'Line'> | null;
  data: LineData[];
  visible: boolean;
}

const SYMBOL_COLORS = [
  '#2962FF', // Blue
  '#00E676', // Green
  '#FF6D00', // Orange
  '#D500F9', // Purple
  '#FFD600', // Yellow
  '#00E5FF', // Cyan
  '#FF1744', // Red
  '#76FF03', // Light Green
];

// Popular symbols for quick selection
const POPULAR_SYMBOLS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
  'SPY', 'QQQ', 'DIA', 'IWM', 'VIX', 'GLD', 'TLT', 'USO',
];

export const ComparisonChart: React.FC<ComparisonChartProps> = ({
  primarySymbol,
  timeframe = '1d',
  height = '100%',
  onSymbolChange,
}) => {
  const theme = useTheme();
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [comparisonMode, setComparisonMode] = useState<ComparisonMode>('percentage');
  const [symbols, setSymbols] = useState<SymbolData[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: theme.palette.text.primary,
      },
      grid: {
        vertLines: { color: alpha(theme.palette.divider, 0.1) },
        horzLines: { color: alpha(theme.palette.divider, 0.1) },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: alpha(theme.palette.primary.main, 0.4),
          width: 1,
          style: 1,
        },
        horzLine: {
          color: alpha(theme.palette.primary.main, 0.4),
          width: 1,
          style: 1,
        },
      },
      rightPriceScale: {
        borderColor: alpha(theme.palette.divider, 0.1),
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      timeScale: {
        borderColor: alpha(theme.palette.divider, 0.1),
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

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
  }, [theme]);

  // Load initial symbol
  useEffect(() => {
    if (primarySymbol) {
      addSymbol(primarySymbol, true);
    }
  }, [primarySymbol]);

  const addSymbol = async (symbol: string, isPrimary = false) => {
    if (symbols.some(s => s.symbol === symbol)) return;

    setLoading(true);
    try {
      const historicalData = await fetchHistoricalData(symbol, timeframe);

      if (!chartRef.current) return;

      const colorIndex = isPrimary ? 0 : (symbols.length % SYMBOL_COLORS.length);
      const color = SYMBOL_COLORS[colorIndex];

      const series = chartRef.current.addLineSeries({
        color,
        lineWidth: isPrimary ? 3 : 2,
        title: symbol,
        priceLineVisible: false,
        lastValueVisible: true,
        crosshairMarkerVisible: true,
      });

      // Transform data based on comparison mode
      const transformedData = transformData(historicalData.data, comparisonMode);
      series.setData(transformedData);

      const newSymbolData: SymbolData = {
        symbol,
        color,
        series,
        data: transformedData,
        visible: true,
      };

      setSymbols(prev => [...prev, newSymbolData]);

      if (isPrimary && onSymbolChange) {
        onSymbolChange(symbol);
      }
    } catch (error) {
      logger.error(`Error loading data for ${symbol}:`, error);
    } finally {
      setLoading(false);
    }
  };

  const removeSymbol = (symbol: string) => {
    const symbolData = symbols.find(s => s.symbol === symbol);
    if (symbolData?.series && chartRef.current) {
      chartRef.current.removeSeries(symbolData.series);
    }
    setSymbols(prev => prev.filter(s => s.symbol !== symbol));
  };

  const toggleSymbolVisibility = (symbol: string) => {
    setSymbols(prev =>
      prev.map(s => {
        if (s.symbol === symbol) {
          const visible = !s.visible;
          s.series?.applyOptions({ visible });
          return { ...s, visible };
        }
        return s;
      })
    );
  };

  const transformData = (data: any[], mode: ComparisonMode): LineData[] => {
    if (!data.length) return [];

    switch (mode) {
      case 'percentage':
        // Calculate percentage change from first value
        const firstValue = data[0].close;
        return data.map(d => ({
          time: d.time as Time,
          value: ((d.close - firstValue) / firstValue) * 100,
        }));

      case 'normalized':
        // Normalize to 0-100 scale
        const min = Math.min(...data.map(d => d.close));
        const max = Math.max(...data.map(d => d.close));
        const range = max - min;
        return data.map(d => ({
          time: d.time as Time,
          value: range > 0 ? ((d.close - min) / range) * 100 : 50,
        }));

      case 'absolute':
      default:
        // Return absolute prices
        return data.map(d => ({
          time: d.time as Time,
          value: d.close,
        }));
    }
  };

  const handleModeChange = (event: React.MouseEvent<HTMLElement>, newMode: ComparisonMode | null) => {
    if (newMode !== null) {
      setComparisonMode(newMode);
      // Re-transform all data
      symbols.forEach(symbolData => {
        if (symbolData.series) {
          const transformedData = transformData(symbolData.data, newMode);
          symbolData.series.setData(transformedData);
        }
      });
    }
  };

  const handleAddSymbol = () => {
    if (inputValue && !symbols.some(s => s.symbol === inputValue.toUpperCase())) {
      addSymbol(inputValue.toUpperCase());
      setInputValue('');
    }
  };

  return (
    <ChartContainer sx={{ height }}>
      <ControlPanel>
        <Stack spacing={2}>
          <Typography variant="subtitle2" fontWeight="bold">
            Symbol Comparison
          </Typography>

          <Box display="flex" alignItems="center" gap={1}>
            <Autocomplete
              value={inputValue}
              onChange={(event, newValue) => {
                if (newValue) {
                  addSymbol(newValue);
                  setInputValue('');
                }
              }}
              inputValue={inputValue}
              onInputChange={(event, newInputValue) => {
                setInputValue(newInputValue);
              }}
              options={POPULAR_SYMBOLS.filter(s => !symbols.some(sym => sym.symbol === s))}
              renderInput={(params) => (
                <TextField
                  {...params}
                  size="small"
                  placeholder="Add symbol..."
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      handleAddSymbol();
                    }
                  }}
                />
              )}
              sx={{ minWidth: 150 }}
            />
            <IconButton size="small" onClick={handleAddSymbol} disabled={!inputValue}>
              <AddIcon />
            </IconButton>
          </Box>

          <Box>
            <Typography variant="caption" color="text.secondary" gutterBottom>
              Active Symbols
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={0.5} mt={1}>
              {symbols.map((symbolData) => (
                <SymbolChip
                  key={symbolData.symbol}
                  label={symbolData.symbol}
                  size="small"
                  onClick={() => toggleSymbolVisibility(symbolData.symbol)}
                  onDelete={() => removeSymbol(symbolData.symbol)}
                  style={{
                    backgroundColor: symbolData.visible
                      ? alpha(symbolData.color, 0.2)
                      : alpha(theme.palette.action.disabled, 0.1),
                    color: symbolData.visible
                      ? symbolData.color
                      : theme.palette.text.disabled,
                    borderColor: symbolData.visible
                      ? symbolData.color
                      : theme.palette.action.disabled,
                  }}
                  variant="outlined"
                />
              ))}
            </Box>
          </Box>

          <Box>
            <Typography variant="caption" color="text.secondary" gutterBottom>
              Comparison Mode
            </Typography>
            <ToggleButtonGroup
              value={comparisonMode}
              exclusive
              onChange={handleModeChange}
              size="small"
              fullWidth
            >
              <ToggleButton value="absolute" aria-label="absolute prices">
                <Tooltip title="Absolute Prices">
                  <ShowChartIcon fontSize="small" />
                </Tooltip>
              </ToggleButton>
              <ToggleButton value="percentage" aria-label="percentage change">
                <Tooltip title="Percentage Change">
                  <PercentIcon fontSize="small" />
                </Tooltip>
              </ToggleButton>
              <ToggleButton value="normalized" aria-label="normalized">
                <Tooltip title="Normalized (0-100)">
                  <TimelineIcon fontSize="small" />
                </Tooltip>
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>
        </Stack>
      </ControlPanel>

      <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />

      {loading && (
        <Box
          position="absolute"
          top="50%"
          left="50%"
          sx={{ transform: 'translate(-50%, -50%)' }}
        >
          <Typography variant="body2" color="text.secondary">
            Loading data...
          </Typography>
        </Box>
      )}
    </ChartContainer>
  );
};

export default ComparisonChart;
