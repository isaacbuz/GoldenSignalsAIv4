/**
 * AITradingChart Component
 *
 * Main intelligent trading chart component that integrates:
 * - Real-time market data visualization
 * - Multi-agent AI analysis system
 * - Interactive trading interface
 * - Advanced technical indicators
 *
 * This is the primary chart component for GoldenSignalsAI,
 * providing institutional-grade analysis through AI agents.
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import {
  Box,
  Select,
  MenuItem,
  FormControl,
  InputBase,
  Chip,
  Typography,
  CircularProgress,
  alpha,
  useTheme,
  Fade,
  Zoom,
  ListSubheader,
  Checkbox,
  ListItemText,
  Button,
  Tabs,
  Tab,
  IconButton,
  Tooltip,
  Paper,
  Divider,
} from '@mui/material';
import {
  Search as SearchIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  AutoGraph as AutoGraphIcon,
  Psychology as PsychologyIcon,
  QueryStats as QueryStatsIcon,
  CompareArrows as CompareArrowsIcon,
  Add as AddIcon,
  Close as CloseIcon,
  Analytics as AnalyticsIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  ZoomOutMap as ZoomOutMapIcon,
  FiberManualRecord as FiberManualRecordIcon,
} from '@mui/icons-material';
import { styled, keyframes } from '@mui/material/styles';
import { debounce } from 'lodash';

// Services
import { fetchHistoricalData } from '../../services/backendMarketDataService';
import { chartSettingsService } from '../../services/chartSettingsService';
import { aiPredictionService } from '../../services/aiPredictionService';
import { agentWorkflowService } from '../../services/agentWorkflowService';

// Hooks
import { useRealtimeChart } from '../../hooks/useRealtimeChart';
import { useAgentAnalysis } from '../../hooks/useAgentAnalysis';
import { useAgentWebSocket } from '../../hooks/useAgentWebSocket';
import { useChartZoom } from './hooks/useChartZoom';
import { useDrawingTools } from './hooks/useDrawingTools';

// Components
import { AgentSignalOverlay } from './AgentSignalOverlay';
import { AgentProgressIndicator } from './components/AgentAnalysis/AgentProgressIndicator';
import { AgentErrorAlert } from './components/AgentAnalysis/AgentErrorAlert';
import { LoadingOverlay } from './components/ChartOverlays/LoadingOverlay';
import { ChartCanvas, type ChartCanvasHandle } from './components/ChartCanvas/ChartCanvas';
import { OscillatorPanel } from './components/OscillatorPanel/OscillatorPanel';
import { DrawingToolbar } from './components/DrawingToolbar/DrawingToolbar';

// Context
import { ChartProvider, useChartContext } from './context/ChartContext';

// Utils
import { candleNormalizer } from '../../utils/candleNormalizer';

// Types
import { WorkflowResult, TradingLevels } from '../../types/agent.types';
import logger from '../../services/logger';


interface AITradingChartProps {
  height?: string | number;
  symbol?: string;
  onSymbolAnalyze?: (symbol: string, analysis: any) => void;
  autoAnalyze?: boolean;  // New prop to control auto-analysis
}

interface ChartDataPoint {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface AISignal {
  time: number;
  type: 'buy' | 'sell';
  price: number;
  confidence: number;
  source?: string;
  metadata?: {
    stopLoss?: number;
    takeProfit?: number[];
    riskReward?: number;
  };
}

interface Pattern {
  type: string;
  confidence: number;
  points: { x: number; y: number }[];
  label?: string;
}

interface Trendline {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  type: 'support' | 'resistance';
}

// Styled components
const pulse = keyframes`
  0% { opacity: 0.6; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.05); }
  100% { opacity: 0.6; transform: scale(1); }
`;

const ChartContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  width: '100%',
  height: '100%',
  minHeight: '400px', // Ensure minimum height
  backgroundColor: '#000000', // Pure black like Yahoo Finance
  overflow: 'hidden',
  display: 'flex',
  flexDirection: 'column',
  // Remove gradient overlay for cleaner look
}));

const SearchContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  backgroundColor: 'rgba(255, 255, 255, 0.05)',
  backdropFilter: 'blur(10px)',
  borderRadius: theme.shape.borderRadius,
  padding: theme.spacing(0.5, 2),
  minWidth: 240,
  border: '1px solid rgba(255, 255, 255, 0.1)',
  '& input': {
    color: '#FFFFFF',
  },
  '& svg': {
    color: 'rgba(255, 255, 255, 0.7)',
  },
}));

const TimeframeSelect = styled(Select)(({ theme }) => ({
  backgroundColor: 'rgba(255, 255, 255, 0.05)',
  color: '#FFFFFF',
  '& .MuiSelect-select': {
    paddingTop: theme.spacing(1),
    paddingBottom: theme.spacing(1),
  },
  '& .MuiOutlinedInput-notchedOutline': {
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  '&:hover .MuiOutlinedInput-notchedOutline': {
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  '& .MuiSvgIcon-root': {
    color: 'rgba(255, 255, 255, 0.7)',
  },
}));

const LiveChip = styled(Chip)(({ theme }) => ({
  animation: `${pulse} 2s infinite`,
  backgroundColor: '#00FF88',
  color: theme.palette.getContrastText('#00FF88'),
}));

const AccuracyChip = styled(Chip)(({ theme }) => ({
  backgroundColor: alpha(theme.palette.primary.main, 0.1),
  color: theme.palette.primary.main,
  fontWeight: 600,
}));

// Available indicators
interface IndicatorOption {
  value: string;
  label: string;
  group: string;
  requiresPanel?: boolean;
}

const AVAILABLE_INDICATORS: IndicatorOption[] = [
  // AI Analysis
  { value: 'ai-signals', label: 'AI Signals', group: 'AI Analysis' },
  { value: 'ai-predictions', label: 'AI Predictions', group: 'AI Analysis' },
  { value: 'agent-levels', label: 'Agent Levels', group: 'AI Analysis' },

  // Trend Indicators
  { value: 'sma-20', label: 'SMA 20', group: 'Trend' },
  { value: 'sma-50', label: 'SMA 50', group: 'Trend' },
  { value: 'sma-200', label: 'SMA 200', group: 'Trend' },
  { value: 'ema-12', label: 'EMA 12', group: 'Trend' },
  { value: 'ema-26', label: 'EMA 26', group: 'Trend' },
  { value: 'vwap', label: 'VWAP', group: 'Trend' },

  // Momentum Oscillators (require separate panel)
  { value: 'rsi', label: 'RSI (14)', group: 'Momentum', requiresPanel: true },
  { value: 'macd', label: 'MACD (12,26,9)', group: 'Momentum', requiresPanel: true },
  { value: 'stochastic', label: 'Stochastic (14,3,3)', group: 'Momentum', requiresPanel: true },
  { value: 'adx', label: 'ADX (14)', group: 'Momentum', requiresPanel: true },

  // Volatility Indicators
  { value: 'bollinger', label: 'Bollinger Bands', group: 'Volatility' },
  { value: 'atr', label: 'ATR (14)', group: 'Volatility', requiresPanel: true },

  // Volume Indicators
  { value: 'volume', label: 'Volume', group: 'Volume' },
];

/**
 * Main AITradingChart component wrapped with ChartProvider
 */
export const AITradingChart: React.FC<AITradingChartProps> = (props) => {
  return (
    <ChartProvider>
      <AITradingChartContent {...props} />
    </ChartProvider>
  );
};

/**
 * Inner content component that uses the chart context
 */
const AITradingChartContent: React.FC<AITradingChartProps> = ({
  height = '600px',
  symbol: propSymbol,
  onSymbolAnalyze,
  autoAnalyze = true,  // Default to true for backwards compatibility
}) => {
  const theme = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const chartCanvasRef = useRef<ChartCanvasHandle>(null);

  // Context
  const {
    state,
    dispatch,
    setLoading,
    setTradingLevels,
    setAgentSignals,
    setChartData,
    setError,
  } = useChartContext();

  // State
  const [symbol, setSymbol] = useState(propSymbol || 'AAPL');
  const [searchValue, setSearchValue] = useState(propSymbol || 'AAPL');
  const [chartType, setChartType] = useState('candlestick');
  const [selectedTimeframe, setSelectedTimeframe] = useState('5m');
  const timeframe = selectedTimeframe;
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>(['ai-signals', 'ai-predictions', 'agent-levels']);
  const [activeTab, setActiveTab] = useState(0); // 0: Main, 1: Compare
  const [compareSymbols, setCompareSymbols] = useState<string[]>([]);
  const [dimensions, setDimensions] = useState({ width: 1200, height: 600 });
  const [isChartReady, setIsChartReady] = useState(false);

  // Agent Analysis Hook with timeframe awareness
  const {
    isAnalyzing,
    progress,
    error: analysisError,
    analyze: analyzeSymbol,
    cancel: cancelAnalysis,
    retry: retryAnalysis,
    tradingLevels: analysisLevels,
  } = useAgentAnalysis(symbol, timeframe);

  // Agent WebSocket Hook
  const {
    agents: realtimeAgents,
    isConnected: isAgentConnected,
    connectionStatus,
    subscribeToAgent,
    unsubscribeFromAgent,
  } = useAgentWebSocket(symbol);

  // Zoom functionality
  const {
    zoomState,
    visibleData,
    zoomIn,
    zoomOut,
    resetZoom,
    handleMouseDown: handleZoomMouseDown,
    handleMouseMove: handleZoomMouseMove,
    handleMouseUp: handleZoomMouseUp,
    isZoomed,
  } = useChartZoom({
    data: state.chartData.data,
    containerRef,
    minZoom: 0.5,
    maxZoom: 10,
  });

  // Drawing tools functionality
  const {
    selectedTool,
    setSelectedTool,
    drawings,
    isDrawing,
    handleMouseDown: handleDrawingMouseDown,
    handleMouseMove: handleDrawingMouseMove,
    handleMouseUp: handleDrawingMouseUp,
    deleteDrawing,
    clearAllDrawings,
    drawAllDrawings,
  } = useDrawingTools({
    onDrawingComplete: (drawing) => {
      logger.info('Drawing completed:', drawing);
      // Could save to localStorage or backend here
    },
  });

  // Real-time chart updates hook
  const { currentPrice, isConnected: isRealtimeConnected } = useRealtimeChart({
    symbol,
    onPriceUpdate: (price) => {
      // Update last candle when price changes
      if (chartCanvasRef.current && state.chartData.data.length > 0) {
        const lastCandle = state.chartData.data[state.chartData.data.length - 1];
        const updatedCandle = {
          ...lastCandle,
          close: price.price,
          high: Math.max(lastCandle.high, price.price),
          low: Math.min(lastCandle.low, price.price),
        };
        chartCanvasRef.current.updateLastCandle(updatedCandle);
      }
    },
    updateInterval: timeframe === '1m' ? 1000 :     // 1 second for 1-minute charts
                    timeframe === '5m' ? 2000 :     // 2 seconds for 5-minute charts
                    timeframe === '15m' ? 5000 :    // 5 seconds for 15-minute charts
                    timeframe === '30m' ? 10000 :   // 10 seconds for 30-minute charts
                    timeframe === '1h' ? 15000 :    // 15 seconds for hourly charts
                    30000,                          // 30 seconds for daily and above
  });

  /**
   * Handle symbol analysis with agent workflow
   */
  const handleAnalyze = useCallback(async () => {
    if (!symbol) return;

    try {
      // Update UI state
      dispatch({ type: 'SET_LOADING', payload: true });

      // Start agent analysis
      await analyzeSymbol();

      if (analysisLevels) {
        setTradingLevels(analysisLevels);

        // Update signals from real-time agents - convert to array format
        const signalsArray = Object.entries(realtimeAgents).map(([agentName, signal]) => ({
          ...signal,
          agent: agentName
        }));
        setAgentSignals(signalsArray);

        // Notify parent component
        if (onSymbolAnalyze) {
          onSymbolAnalyze(symbol, { levels: analysisLevels, signals: Object.values(realtimeAgents) });
        }
      }
    } catch (error) {
      logger.error('Analysis failed:', error);
    } finally {
      setLoading(false);
    }
  }, [symbol, analyzeSymbol, dispatch, onSymbolAnalyze, analysisLevels, realtimeAgents, setTradingLevels, setAgentSignals, setLoading]);

  /**
   * Subscribe to real-time agent signals
   */
  useEffect(() => {
    if (symbol && isAgentConnected) {
      // Subscribe to all agents for current symbol
      const agentNames = ['RSIAgent', 'MACDAgent', 'PatternAgent', 'VolumeAgent'];
      agentNames.forEach(agent => subscribeToAgent(agent));

      return () => {
        agentNames.forEach(agent => unsubscribeFromAgent(agent));
      };
    }
  }, [symbol, isAgentConnected, subscribeToAgent, unsubscribeFromAgent]);

  /**
   * Update chart with real-time signals
   */
  useEffect(() => {
    if (realtimeAgents && Object.keys(realtimeAgents).length > 0) {
      // Convert agent signals to chart format
      const newSignals = Object.entries(realtimeAgents).map(([agentName, agentData]) => ({
        agent: agentName,
        signal: agentData.signal,
        confidence: agentData.confidence,
        metadata: agentData.metadata || {},
      }));

      // Update agent signals in context - convert to array format
      const signalsArray = Object.entries(newSignals).map(([agentName, signal]) => ({
        ...signal,
        agent: agentName
      }));
      setAgentSignals(signalsArray);
    }
  }, [realtimeAgents, setAgentSignals]);

  /**
   * Fetch and update chart data
   */
  const fetchChartData = useCallback(async (retryCount = 0) => {
    try {
      setLoading(true);
      logger.debug('Fetching chart data for:', { symbol, timeframe, attempt: retryCount + 1 });

      const response = await fetchHistoricalData(symbol, timeframe);

      // Extract data from response object
      const data = response.data || response;

      if (!data || !Array.isArray(data) || data.length === 0) {
        logger.error('No chart data received from backend');
        setError(new Error('No data available for this symbol'));
        return;
      }

      // Check for sparse data and retry
      const minDataPoints = {
        // Intraday
        '1m': 50,
        '5m': 30,
        '15m': 20,
        '30m': 15,
        '1h': 10,
        '4h': 8,
        // Daily & Above
        '1d': 30,   // At least 30 days
        '1w': 20,   // At least 20 weeks
        '1M': 12,   // At least 12 months
        // Long Term - Lower requirements
        '3M': 8,    // At least 8 quarters (2 years)
        '6M': 4,    // At least 4 half-years (2 years)
        '1y': 5,    // At least 5 years
        '2y': 3,    // At least 3 data points
        '5y': 3,    // At least 3 data points
        '10y': 3,   // At least 3 data points
        'max': 3,   // At least 3 data points
      }[timeframe] || 10;

      if (data.length < minDataPoints) {
        if (retryCount < 3) {
          logger.warn(`Sparse data detected: ${data.length} points (expected ${minDataPoints}+). Retrying...`);
          setTimeout(() => fetchChartData(retryCount + 1), 1000 * (retryCount + 1));
          return;
        } else {
          // After 3 retries, show a specific error for sparse data
          const sparseError = new Error(
            `Insufficient data for ${symbol} on ${timeframe} timeframe. ` +
            `Only ${data.length} data points available (minimum ${minDataPoints} required). ` +
            `Try a different timeframe or symbol.`
          );
          setError(sparseError);
          // Still show the available data if any
          if (data.length > 0) {
            logger.info('Showing limited data available');
          }
        }
      }

      // Just ensure data types are correct without normalizing ranges
      const formattedData = data.map(candle => ({
        time: typeof candle.time === 'number' ? candle.time : new Date(candle.time as string).getTime() / 1000,
        open: parseFloat(String(candle.open)),
        high: parseFloat(String(candle.high)),
        low: parseFloat(String(candle.low)),
        close: parseFloat(String(candle.close)),
        volume: parseInt(String(candle.volume)) || 0,
      }));

      // Apply normalization to fix large/unrealistic candlesticks
      const normalizedData = candleNormalizer.normalizeData(formattedData, timeframe);

      // Debug info
      if (normalizedData.length > 0) {
        const firstCandle = normalizedData[0];
        const lastCandle = normalizedData[normalizedData.length - 1];
        const priceRange = Math.max(...normalizedData.map(d => d.high)) - Math.min(...normalizedData.map(d => d.low));
        const avgPrice = (Math.max(...normalizedData.map(d => d.high)) + Math.min(...normalizedData.map(d => d.low))) / 2;
        const rangePercent = (priceRange / avgPrice) * 100;

        logger.info(`Chart data loaded: ${normalizedData.length} candles for ${symbol}`);
        logger.info(`Price range: $${Math.min(...normalizedData.map(d => d.low)).toFixed(2)} - $${Math.max(...normalizedData.map(d => d.high)).toFixed(2)} (${rangePercent.toFixed(2)}% range)`);
        logger.info(`Time range: ${new Date(firstCandle.time * 1000).toLocaleString()} - ${new Date(lastCandle.time * 1000).toLocaleString()}`);
      }

      setChartData(normalizedData);
      setError(null); // Clear any previous errors

      // Update current price
      if (normalizedData.length > 0) {
        const lastCandle = normalizedData[normalizedData.length - 1];
        logger.debug('Latest price:', lastCandle.close);
      }
    } catch (error) {
      logger.error('Failed to fetch chart data:', error);
      setError(error as Error);
      // Do NOT use mock data - show error to user instead
      setChartData([]);
    } finally {
      setLoading(false);
    }
  }, [symbol, timeframe, setLoading, setChartData, setError]);

  /**
   * Initialize chart data on mount and symbol/timeframe change
   */
  useEffect(() => {
    fetchChartData();
  }, [fetchChartData]);

  /**
   * Auto-analyze on symbol or timeframe change
   * This implements Grok's suggestion for automatic analysis
   */
  useEffect(() => {
    // Only run auto-analysis if enabled
    if (!autoAnalyze) return;

    // Add a small delay to avoid rapid consecutive calls
    const analyzeTimeout = setTimeout(() => {
      // Only auto-analyze if we have data and not already analyzing
      if (state.chartData.data.length > 0 && !isAnalyzing) {
        handleAnalyze();
      }
    }, 1000); // 1 second delay

    return () => clearTimeout(analyzeTimeout);
  }, [symbol, timeframe, autoAnalyze]); // Trigger on symbol or timeframe change

  /**
   * Update dimensions on container resize
   */
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const newWidth = rect.width || 1200;
        const newHeight = rect.height || 600;

        logger.debug('Chart container dimensions:', { width: newWidth, height: newHeight });

        // Only update if dimensions are valid
        if (newWidth > 0 && newHeight > 0) {
          setDimensions({
            width: newWidth,
            height: newHeight,
          });
          setIsChartReady(true);
        }
      }
    };

    // Use a small delay to ensure container is properly laid out
    const timer = setTimeout(updateDimensions, 100);
    window.addEventListener('resize', updateDimensions);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', updateDimensions);
    };
  }, []);

  /**
   * Handle search input change
   */
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchValue(event.target.value.toUpperCase());
  };

  /**
   * Handle search submit
   */
  const handleSearchSubmit = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && searchValue) {
      setSymbol(searchValue);
    }
  };

  /**
   * Handle indicator selection
   */
  const handleIndicatorChange = (event: any) => {
    const value = event.target.value;
    setSelectedIndicators(typeof value === 'string' ? value.split(',') : value);
  };

  /**
   * Handle compare symbol addition
   */
  const handleAddCompareSymbol = () => {
    if (searchValue && !compareSymbols.includes(searchValue) && compareSymbols.length < 3) {
      setCompareSymbols([...compareSymbols, searchValue]);
      setSearchValue('');
    }
  };

  /**
   * Handle compare symbol removal
   */
  const handleRemoveCompareSymbol = (symbolToRemove: string) => {
    setCompareSymbols(compareSymbols.filter(s => s !== symbolToRemove));
  };

  /**
   * Calculate height needed for oscillator panels
   */
  const getOscillatorPanelHeight = () => {
    const oscillators = selectedIndicators.filter(ind =>
      ['rsi', 'macd', 'stochastic', 'atr', 'adx'].includes(ind)
    );
    return oscillators.length * 104; // 100px per panel + 4px margin
  };

  /**
   * Calculate AI accuracy display
   */
  const aiAccuracy = analysisLevels ? {
    historical: Math.round(0.75 * 100),
    confidence: Math.round(0.8 * 100),
  } : { historical: 0, confidence: 0 };

  return (
    <ChartContainer ref={containerRef} sx={{ height }}>
      {/* Header Controls */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          zIndex: 10,
          p: 2,
          display: 'flex',
          gap: 2,
          flexWrap: 'wrap',
          alignItems: 'center',
          background: 'linear-gradient(to bottom, rgba(0,0,0,0.95), transparent)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        {/* Tabs for Main/Compare */}
        <Tabs
          value={activeTab}
          onChange={(e, v) => setActiveTab(v)}
          sx={{
            minHeight: 0,
            '& .MuiTab-root': {
              color: 'rgba(255, 255, 255, 0.7)',
              '&.Mui-selected': {
                color: '#FFFFFF',
              },
            },
          }}
        >
          <Tab label="Main" sx={{ minHeight: 48 }} />
          <Tab
            label="Compare"
            icon={<CompareArrowsIcon sx={{ fontSize: 16, ml: 0.5 }} />}
            iconPosition="end"
            sx={{ minHeight: 48 }}
          />
        </Tabs>

        {/* Search Bar */}
        <SearchContainer>
          <InputBase
            placeholder="Search symbol..."
            value={searchValue}
            onChange={handleSearchChange}
            onKeyPress={handleSearchSubmit}
            startAdornment={<SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />}
            sx={{ width: '100%' }}
          />
        </SearchContainer>

        {/* Analyze Button */}
        <Button
          variant="contained"
          onClick={handleAnalyze}
          disabled={isAnalyzing || !symbol}
          startIcon={isAnalyzing ? <CircularProgress size={16} /> : <PsychologyIcon />}
          sx={{
            backgroundColor: '#007AFF',
            color: 'white',
            fontWeight: 500,
            '&:hover': {
              backgroundColor: '#0051D5',
            },
          }}
        >
          {isAnalyzing ? 'Analyzing...' : 'Analyze'}
        </Button>

        {/* Timeframe Selector */}
        <FormControl size="small">
          <TimeframeSelect
            value={timeframe}
            onChange={(e) => setSelectedTimeframe(e.target.value as any)}
          >
            <ListSubheader>Intraday</ListSubheader>
            <MenuItem value="1m">1 Minute</MenuItem>
            <MenuItem value="5m">5 Minutes</MenuItem>
            <MenuItem value="15m">15 Minutes</MenuItem>
            <MenuItem value="30m">30 Minutes</MenuItem>
            <MenuItem value="1h">1 Hour</MenuItem>
            <MenuItem value="4h">4 Hours</MenuItem>
            <Divider />
            <ListSubheader>Daily & Above</ListSubheader>
            <MenuItem value="1d">1 Day</MenuItem>
            <MenuItem value="1w">1 Week</MenuItem>
            <MenuItem value="1M">1 Month</MenuItem>
            <Divider />
            <ListSubheader>Long Term</ListSubheader>
            <MenuItem value="3M">3 Months</MenuItem>
            <MenuItem value="6M">6 Months</MenuItem>
            <MenuItem value="1y">1 Year</MenuItem>
            <MenuItem value="2y">2 Years</MenuItem>
            <MenuItem value="5y">5 Years</MenuItem>
            <MenuItem value="10y">10 Years</MenuItem>
            <MenuItem value="max">All Time</MenuItem>
          </TimeframeSelect>
        </FormControl>

        {/* Indicators Selector */}
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <Select
            multiple
            value={selectedIndicators}
            onChange={handleIndicatorChange}
            sx={{
              backgroundColor: 'rgba(255, 255, 255, 0.05)',
              color: '#FFFFFF',
              '& .MuiOutlinedInput-notchedOutline': {
                borderColor: 'rgba(255, 255, 255, 0.1)',
              },
              '&:hover .MuiOutlinedInput-notchedOutline': {
                borderColor: 'rgba(255, 255, 255, 0.3)',
              },
              '& .MuiSvgIcon-root': {
                color: 'rgba(255, 255, 255, 0.7)',
              },
            }}
            renderValue={(selected) => `${selected.length} indicators`}
            MenuProps={{
              PaperProps: {
                style: { maxHeight: 300 },
              },
            }}
          >
            {Object.entries(
              AVAILABLE_INDICATORS.reduce((acc, indicator) => {
                if (!acc[indicator.group]) acc[indicator.group] = [];
                acc[indicator.group].push(indicator);
                return acc;
              }, {} as Record<string, typeof AVAILABLE_INDICATORS>)
            ).map(([group, indicators]) => [
              <ListSubheader key={group}>{group}</ListSubheader>,
              ...indicators.map((indicator) => (
                <MenuItem key={indicator.value} value={indicator.value}>
                  <Checkbox checked={selectedIndicators.indexOf(indicator.value) > -1} />
                  <ListItemText primary={indicator.label} />
                </MenuItem>
              )),
            ])}
          </Select>
        </FormControl>

        {/* Zoom Controls */}
        <Box sx={{ display: 'flex', gap: 0.5 }}>
          <IconButton
            size="small"
            onClick={zoomIn}
            disabled={zoomState.scale >= 10}
            title="Zoom In (Ctrl +)"
          >
            <ZoomInIcon fontSize="small" />
          </IconButton>
          <IconButton
            size="small"
            onClick={zoomOut}
            disabled={zoomState.scale <= 0.5}
            title="Zoom Out (Ctrl -)"
          >
            <ZoomOutIcon fontSize="small" />
          </IconButton>
          <IconButton
            size="small"
            onClick={resetZoom}
            disabled={!isZoomed}
            title="Reset Zoom (Ctrl 0)"
          >
            <ZoomOutMapIcon fontSize="small" />
          </IconButton>
        </Box>

        {/* Status Chips */}
        <Box sx={{ display: 'flex', gap: 1, ml: 'auto', alignItems: 'center' }}>
          {/* Live Connection Status */}
          {isRealtimeConnected ? (
            <LiveChip
              icon={<FiberManualRecordIcon sx={{ fontSize: 12 }} />}
              label="LIVE"
              size="small"
            />
          ) : (
            <Chip
              icon={<FiberManualRecordIcon sx={{ fontSize: 12 }} />}
              label="CONNECTING..."
              size="small"
              sx={{
                backgroundColor: alpha(theme.palette.warning.main, 0.2),
                color: theme.palette.warning.main
              }}
            />
          )}

          {/* Current Price Display */}
          {currentPrice > 0 && (
            <Typography
              variant="body1"
              sx={{
                fontWeight: 700,
                color: theme.palette.mode === 'dark' ? '#FFD700' : theme.palette.primary.main,
                fontSize: '1.1rem'
              }}
            >
              ${currentPrice.toFixed(2)}
            </Typography>
          )}

          {/* Zoom Level */}
          {isZoomed && (
            <Chip
              label={`${Math.round(zoomState.scale * 100)}%`}
              size="small"
              variant="outlined"
            />
          )}
          {aiAccuracy.historical > 0 && (
            <AccuracyChip
              label={`AI: ${aiAccuracy.historical}% accurate`}
              size="small"
              icon={<AutoGraphIcon />}
            />
          )}
        </Box>
      </Box>

      {/* Progress Indicator */}
      {isAnalyzing && progress && (
        <Box sx={{ position: 'absolute', top: 80, left: 20, zIndex: 20 }}>
          <AgentProgressIndicator
            progress={progress}
            currentStage={null}
            messages={[]}
          />
        </Box>
      )}

      {/* Error Alert */}
      {analysisError && (
        <Box sx={{ position: 'absolute', top: 80, right: 20, zIndex: 20 }}>
          <AgentErrorAlert
            error={analysisError}
            onRetry={retryAnalysis}
          />
        </Box>
      )}

      {/* Loading Overlay */}
      {state.chartData.loading && (
        <LoadingOverlay
          visible={state.chartData.loading}
          message={isAnalyzing ? 'AI agents analyzing market...' : 'Loading chart data...'}
          progress={isAnalyzing && progress ? progress : undefined}
        />
      )}

      {/* Main Chart Canvas */}
      <Box
        sx={{
          flex: 1,
          position: 'relative',
          minHeight: '300px',
          cursor: selectedTool !== 'none' ? 'crosshair' : isZoomed ? 'move' : 'default'
        }}
        onMouseDown={selectedTool !== 'none' ? undefined : handleZoomMouseDown}
        onMouseMove={selectedTool !== 'none' ? undefined : handleZoomMouseMove}
        onMouseUp={selectedTool !== 'none' ? undefined : handleZoomMouseUp}
        onMouseLeave={selectedTool !== 'none' ? undefined : handleZoomMouseUp}
      >
        {state.chartData.error ? (
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              textAlign: 'center',
            }}
          >
            <Typography variant="h6" color="error" gutterBottom>
              Unable to Load Chart Data
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {state.chartData.error.message}
            </Typography>
            <Button
              variant="contained"
              onClick={() => fetchChartData()}
              startIcon={<TrendingUpIcon />}
            >
              Retry
            </Button>
            <Typography variant="caption" display="block" sx={{ mt: 2 }}>
              Make sure the backend is running on port 8000
            </Typography>
          </Box>
        ) : isChartReady && dimensions.width > 0 && dimensions.height > 0 ? (
          <ChartCanvas
            ref={chartCanvasRef}
            data={state.chartData.data}
            visibleData={visibleData}
            zoomState={zoomState}
            width={dimensions.width}
            height={Math.max(dimensions.height - 100 - getOscillatorPanelHeight(), 300)} // Account for header and oscillators
            indicators={selectedIndicators}
            signals={Object.values(state.agentSignals)}
            agentLevels={state.tradingLevels}
            currentPrice={currentPrice}
            theme={theme}
            timeframe={timeframe}
            drawings={drawings}
            onDrawingMouseDown={handleDrawingMouseDown}
            onDrawingMouseMove={handleDrawingMouseMove}
            onDrawingMouseUp={handleDrawingMouseUp}
            drawAllDrawings={drawAllDrawings}
          />
        ) : (
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              textAlign: 'center',
            }}
          >
            <CircularProgress size={40} />
            <Typography variant="body2" sx={{ mt: 2 }}>
              Loading real market data...
            </Typography>
          </Box>
        )}

        {/* Drawing Toolbar */}
        {state.chartData.data.length > 0 && !state.chartData.error && (
          <DrawingToolbar
            selectedTool={selectedTool}
            onToolChange={setSelectedTool}
            onClearAll={clearAllDrawings}
            drawingCount={drawings.length}
            position="left"
          />
        )}
      </Box>

      {/* Compare Mode Overlay */}
      {activeTab === 1 && (
        <Box
          sx={{
            position: 'absolute',
            top: 120,
            left: 20,
            zIndex: 15,
            backgroundColor: alpha(theme.palette.background.paper, 0.9),
            borderRadius: 1,
            p: 2,
            backdropFilter: 'blur(10px)',
          }}
        >
          <Typography variant="subtitle2" gutterBottom>
            Compare Symbols
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 1 }}>
            {compareSymbols.map((sym) => (
              <Chip
                key={sym}
                label={sym}
                onDelete={() => handleRemoveCompareSymbol(sym)}
                size="small"
              />
            ))}
          </Box>
          {compareSymbols.length < 3 && (
            <Button
              size="small"
              startIcon={<AddIcon />}
              onClick={handleAddCompareSymbol}
              disabled={!searchValue}
            >
              Add Symbol
            </Button>
          )}
        </Box>
      )}

      {/* Agent Signal Overlay (legacy compatibility) */}
      {selectedIndicators.includes('agent-levels') && state.tradingLevels && (
        <AgentSignalOverlay
          agentSignals={{}}
          workflowDecision={null}
          onClose={() => {}}
        />
      )}

      {/* Oscillator Panel for RSI, MACD, etc. */}
      {state.chartData.data.length > 0 && (
        <OscillatorPanel
          data={state.chartData.data}
          indicators={selectedIndicators}
          width={dimensions.width}
          onRemoveIndicator={(indicator) => {
            setSelectedIndicators(prev => prev.filter(ind => ind !== indicator));
          }}
        />
      )}
    </ChartContainer>
  );
};

export default AITradingChart;
