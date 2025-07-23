/**
 * AITradingChart - The Definitive Trading Chart for GoldenSignalsAI
 *
 * This is the ONE chart component that combines:
 * - Clean, focused chart rendering (from FocusedTradingChart)
 * - AI agent integration for trading signals
 * - Professional technical analysis tools
 * - Real-time data with WebSocket support
 *
 * Architecture:
 * - Modular design with separate components
 * - Clean separation of concerns
 * - Efficient rendering with lightweight-charts
 * - Agent signals overlay without clutter
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
} from 'lightweight-charts';
import {
  Box,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  CircularProgress,
  TextField,
  Autocomplete,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Divider,
  FormGroup,
  FormControlLabel,
  Switch,
  Tooltip,
  Badge,
} from '@mui/material';
import { styled, alpha } from '@mui/material/styles';
import {
  Search as SearchIcon,
  Settings as SettingsIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Psychology as PsychologyIcon,
  Refresh as RefreshIcon,
  Fullscreen as FullscreenIcon,
  Draw as DrawIcon,
} from '@mui/icons-material';

// Components
import { DrawingToolbar } from './components/DrawingToolbar';
import { AgentSignalOverlay } from './components/AgentSignalOverlay';
import { IndicatorSettings } from './components/IndicatorSettings';

// Hooks
import { useMarketData } from './hooks/useMarketData';
import { useAgentAnalysis } from './hooks/useAgentAnalysis';
import { useChartDrawings } from './hooks/useChartDrawings';

// Utils
import { calculateIndicators } from './utils/indicators';
import { CHART_COLORS } from './utils/colors';

// Styled Components
const Container = styled(Box)({
  width: '100%',
  height: '100vh',
  backgroundColor: '#131722',
  display: 'flex',
  flexDirection: 'column',
  fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif',
});

const Header = styled(Box)({
  height: 60,
  backgroundColor: '#1e222d',
  borderBottom: '1px solid #2a2e39',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '0 20px',
  gap: 20,
});

const SymbolSearch = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  gap: 12,
  flex: '0 0 auto',
});

const PriceInfo = styled(Box)({
  display: 'flex',
  alignItems: 'baseline',
  gap: 12,
  flex: '0 0 auto',
});

const ChartArea = styled(Box)({
  flex: 1,
  position: 'relative',
  backgroundColor: '#131722',
});

const LoadingOverlay = styled(Box)({
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: 'rgba(19, 23, 34, 0.9)',
  zIndex: 1000,
});

const LiveBadge = styled(Badge)(({ theme }) => ({
  '& .MuiBadge-badge': {
    backgroundColor: '#4caf50',
    color: '#4caf50',
    boxShadow: `0 0 0 2px ${theme.palette.background.paper}`,
    '&::after': {
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      borderRadius: '50%',
      animation: 'ripple 1.2s infinite ease-in-out',
      border: '1px solid currentColor',
      content: '""',
    },
  },
  '@keyframes ripple': {
    '0%': {
      transform: 'scale(.8)',
      opacity: 1,
    },
    '100%': {
      transform: 'scale(2.4)',
      opacity: 0,
    },
  },
}));

// Popular symbols for autocomplete
const POPULAR_SYMBOLS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
  'V', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'MA', 'HD', 'BAC', 'ADBE',
  'NFLX', 'CRM', 'XOM', 'VZ', 'CMCSA', 'PFE', 'KO', 'INTC', 'CSCO',
  'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV', 'USO', 'TLT', 'VXX'
];

interface AITradingChartProps {
  symbol?: string;
  onSymbolChange?: (symbol: string) => void;
}

export const AITradingChart: React.FC<AITradingChartProps> = ({
  symbol: initialSymbol = 'TSLA',
  onSymbolChange
}) => {
  // Chart refs
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const rsiChartRef = useRef<IChartApi | null>(null);
  const macdChartRef = useRef<IChartApi | null>(null);

  // State
  const [symbol, setSymbol] = useState(initialSymbol);
  const [timeframe, setTimeframe] = useState('1D');
  const [loading, setLoading] = useState(true);
  const [indicators, setIndicators] = useState({
    ma20: true,
    ma50: true,
    ma200: false,
    bollinger: false,
    volume: true,
    rsi: true,
    macd: true,
  });
  const [drawingMode, setDrawingMode] = useState<string | null>(null);
  const [settingsAnchor, setSettingsAnchor] = useState<null | HTMLElement>(null);

  // Custom hooks
  const { data, price, change, changePercent, isLive, error } = useMarketData(symbol, timeframe);
  const { agentSignals, consensus, isAnalyzing } = useAgentAnalysis(symbol);
  const { drawings, addDrawing, removeDrawing, clearDrawings } = useChartDrawings();

  // Initialize charts
  useEffect(() => {
    if (!chartContainerRef.current || !data) return;

    const container = chartContainerRef.current;
    const containerHeight = container.clientHeight;
    const containerWidth = container.clientWidth;

    // Clear previous charts
    container.innerHTML = '';

    // Calculate heights
    const hasOscillators = indicators.rsi || indicators.macd;
    const oscillatorCount = (indicators.rsi ? 1 : 0) + (indicators.macd ? 1 : 0);
    const priceHeight = hasOscillators ? containerHeight * 0.6 : containerHeight;
    const oscillatorHeight = hasOscillators ? (containerHeight * 0.4) / oscillatorCount : 0;

    // Create layout divs
    const priceDiv = document.createElement('div');
    priceDiv.style.height = `${priceHeight}px`;
    priceDiv.style.width = '100%';
    container.appendChild(priceDiv);

    // Common chart options
    const commonOptions = {
      layout: {
        background: { type: ColorType.Solid, color: CHART_COLORS.background },
        textColor: CHART_COLORS.text.secondary,
        fontSize: 11,
      },
      grid: {
        vertLines: { color: CHART_COLORS.grid },
        horzLines: { color: CHART_COLORS.grid },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: CHART_COLORS.text.secondary,
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: CHART_COLORS.card,
        },
        horzLine: {
          color: CHART_COLORS.text.secondary,
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: CHART_COLORS.card,
        },
      },
      rightPriceScale: {
        borderColor: CHART_COLORS.border,
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      timeScale: {
        borderColor: CHART_COLORS.border,
        timeVisible: true,
        secondsVisible: false,
      },
    };

    // Create price chart
    const priceChart = createChart(priceDiv, {
      ...commonOptions,
      width: containerWidth,
      height: priceHeight,
    });

    // Add candlestick series
    const candleSeries = priceChart.addCandlestickSeries({
      upColor: CHART_COLORS.candle.up,
      downColor: CHART_COLORS.candle.down,
      borderUpColor: CHART_COLORS.candle.up,
      borderDownColor: CHART_COLORS.candle.down,
      wickUpColor: CHART_COLORS.candle.up,
      wickDownColor: CHART_COLORS.candle.down,
    });
    candleSeries.setData(data.candles);
    candleSeriesRef.current = candleSeries;

    // Add indicators
    if (indicators.volume) {
      const volumeSeries = priceChart.addHistogramSeries({
        color: CHART_COLORS.volume.up,
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      });
      volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      });
      volumeSeries.setData(data.volumes || []);
    }

    // Add moving averages
    const { ma20, ma50, ma200, bollingerBands } = calculateIndicators(data.candles);

    if (indicators.ma20 && ma20) {
      const ma20Series = priceChart.addLineSeries({
        color: CHART_COLORS.ma[20],
        lineWidth: 2,
        title: 'MA 20',
        lastValueVisible: false,
      });
      ma20Series.setData(ma20);
    }

    if (indicators.ma50 && ma50) {
      const ma50Series = priceChart.addLineSeries({
        color: CHART_COLORS.ma[50],
        lineWidth: 2,
        title: 'MA 50',
        lastValueVisible: false,
      });
      ma50Series.setData(ma50);
    }

    if (indicators.ma200 && ma200) {
      const ma200Series = priceChart.addLineSeries({
        color: CHART_COLORS.ma[200],
        lineWidth: 2,
        title: 'MA 200',
        lastValueVisible: false,
      });
      ma200Series.setData(ma200);
    }

    if (indicators.bollinger && bollingerBands) {
      // Add Bollinger Bands
      const bbUpperSeries = priceChart.addLineSeries({
        color: alpha(CHART_COLORS.bollinger, 0.5),
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        title: 'BB Upper',
        lastValueVisible: false,
      });
      bbUpperSeries.setData(bollingerBands.upper);

      const bbLowerSeries = priceChart.addLineSeries({
        color: alpha(CHART_COLORS.bollinger, 0.5),
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        title: 'BB Lower',
        lastValueVisible: false,
      });
      bbLowerSeries.setData(bollingerBands.lower);
    }

    // Create RSI chart if enabled
    if (indicators.rsi) {
      const rsiDiv = document.createElement('div');
      rsiDiv.style.height = `${oscillatorHeight}px`;
      rsiDiv.style.width = '100%';
      rsiDiv.style.borderTop = '1px solid #2a2e39';
      container.appendChild(rsiDiv);

      const rsiChart = createChart(rsiDiv, {
        ...commonOptions,
        width: containerWidth,
        height: oscillatorHeight,
        timeScale: { visible: indicators.macd ? false : true },
      });

      const rsiLine = rsiChart.addLineSeries({
        color: CHART_COLORS.rsi.line,
        lineWidth: 2,
        title: 'RSI',
      });
      const rsiData = calculateIndicators(data.candles).rsi;
      if (rsiData) rsiLine.setData(rsiData);

      // RSI levels
      rsiLine.createPriceLine({ price: 70, color: CHART_COLORS.rsi.overbought, lineWidth: 1, lineStyle: LineStyle.Dashed });
      rsiLine.createPriceLine({ price: 30, color: CHART_COLORS.rsi.oversold, lineWidth: 1, lineStyle: LineStyle.Dashed });

      rsiChartRef.current = rsiChart;
    }

    // Create MACD chart if enabled
    if (indicators.macd) {
      const macdDiv = document.createElement('div');
      macdDiv.style.height = `${oscillatorHeight}px`;
      macdDiv.style.width = '100%';
      macdDiv.style.borderTop = '1px solid #2a2e39';
      container.appendChild(macdDiv);

      const macdChart = createChart(macdDiv, {
        ...commonOptions,
        width: containerWidth,
        height: oscillatorHeight,
      });

      const macdData = calculateIndicators(data.candles).macd;
      if (macdData) {
        const macdHistogram = macdChart.addHistogramSeries({
          color: CHART_COLORS.macd.histogram.positive,
          title: 'MACD Histogram',
        });
        macdHistogram.setData(macdData.histogram.map(h => ({
          ...h,
          color: h.value >= 0 ? CHART_COLORS.macd.histogram.positive : CHART_COLORS.macd.histogram.negative,
        })));

        const macdLine = macdChart.addLineSeries({
          color: CHART_COLORS.macd.line,
          lineWidth: 2,
          title: 'MACD',
        });
        macdLine.setData(macdData.macd);

        const signalLine = macdChart.addLineSeries({
          color: CHART_COLORS.macd.signal,
          lineWidth: 2,
          title: 'Signal',
        });
        signalLine.setData(macdData.signal);

        // Zero line
        signalLine.createPriceLine({ price: 0, color: CHART_COLORS.border, lineWidth: 1 });
      }

      macdChartRef.current = macdChart;
    }

    // Sync charts
    const charts = [priceChart, rsiChartRef.current, macdChartRef.current].filter(Boolean) as IChartApi[];
    charts.forEach((chart, index) => {
      chart.subscribeCrosshairMove((param) => {
        charts.forEach((otherChart, otherIndex) => {
          if (index !== otherIndex && param.time) {
            otherChart.setCrosshairPosition(param.point?.x || 0, 0, param.time);
          }
        });
      });
    });

    // Sync visible range
    priceChart.timeScale().subscribeVisibleTimeRangeChange(() => {
      const range = priceChart.timeScale().getVisibleRange();
      if (range) {
        charts.forEach((chart) => {
          if (chart !== priceChart) {
            chart.timeScale().setVisibleRange(range);
          }
        });
      }
    });

    chartRef.current = priceChart;

    // Handle resize
    const handleResize = () => {
      const newWidth = container.clientWidth;
      const newHeight = container.clientHeight;

      priceChart.applyOptions({ width: newWidth, height: priceHeight });
      if (rsiChartRef.current) {
        rsiChartRef.current.applyOptions({ width: newWidth, height: oscillatorHeight });
      }
      if (macdChartRef.current) {
        macdChartRef.current.applyOptions({ width: newWidth, height: oscillatorHeight });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      charts.forEach(chart => chart.remove());
    };
  }, [data, indicators]);

  // Handle symbol change
  const handleSymbolChange = (newSymbol: string | null) => {
    if (newSymbol && newSymbol !== symbol) {
      setSymbol(newSymbol);
      onSymbolChange?.(newSymbol);
    }
  };

  // Handle timeframe change
  const handleTimeframeChange = (_: any, newTimeframe: string) => {
    if (newTimeframe) {
      setTimeframe(newTimeframe);
    }
  };

  // Handle fullscreen
  const handleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  };

  return (
    <Container>
      <Header>
        <SymbolSearch>
          <Autocomplete
            value={symbol}
            onChange={(_, newValue) => handleSymbolChange(newValue)}
            options={POPULAR_SYMBOLS}
            size="small"
            sx={{
              width: 150,
              '& .MuiOutlinedInput-root': {
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                '& fieldset': {
                  borderColor: '#2a2e39',
                },
                '&:hover fieldset': {
                  borderColor: '#3a3e49',
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#2196f3',
                },
              },
              '& .MuiInputBase-input': {
                color: '#d1d4dc',
                fontWeight: 600,
              },
            }}
            renderInput={(params) => (
              <TextField
                {...params}
                placeholder="Symbol"
                InputProps={{
                  ...params.InputProps,
                  startAdornment: <SearchIcon sx={{ color: '#787b86', mr: 1 }} />,
                }}
              />
            )}
          />

          {isLive && (
            <LiveBadge
              overlap="circular"
              anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
              variant="dot"
            >
              <Chip
                label="LIVE"
                size="small"
                sx={{
                  backgroundColor: 'rgba(76, 175, 80, 0.1)',
                  color: '#4caf50',
                  fontWeight: 600,
                  fontSize: '0.75rem',
                }}
              />
            </LiveBadge>
          )}
        </SymbolSearch>

        <PriceInfo>
          {price > 0 && (
            <>
              <Typography
                variant="h5"
                sx={{
                  color: '#d1d4dc',
                  fontWeight: 700,
                  fontVariantNumeric: 'tabular-nums'
                }}
              >
                ${price.toFixed(2)}
              </Typography>
              <Typography
                variant="body1"
                sx={{
                  color: change >= 0 ? CHART_COLORS.candle.up : CHART_COLORS.candle.down,
                  fontWeight: 500,
                }}
              >
                {change >= 0 ? '+' : ''}{change.toFixed(2)} ({changePercent >= 0 ? '+' : ''}{changePercent.toFixed(2)}%)
                {change >= 0 ? <TrendingUpIcon sx={{ fontSize: 16, ml: 0.5 }} /> : <TrendingDownIcon sx={{ fontSize: 16, ml: 0.5 }} />}
              </Typography>
            </>
          )}
        </PriceInfo>

        <ToggleButtonGroup
          value={timeframe}
          exclusive
          onChange={handleTimeframeChange}
          size="small"
          sx={{
            '& .MuiToggleButton-root': {
              color: '#787b86',
              borderColor: '#2a2e39',
              padding: '4px 16px',
              textTransform: 'none',
              fontWeight: 600,
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
              },
              '&.Mui-selected': {
                backgroundColor: 'rgba(33, 150, 243, 0.15)',
                color: '#2196f3',
                '&:hover': {
                  backgroundColor: 'rgba(33, 150, 243, 0.25)',
                },
              },
            },
          }}
        >
          <ToggleButton value="1D">1D</ToggleButton>
          <ToggleButton value="5D">5D</ToggleButton>
          <ToggleButton value="1M">1M</ToggleButton>
          <ToggleButton value="3M">3M</ToggleButton>
          <ToggleButton value="1Y">1Y</ToggleButton>
          <ToggleButton value="ALL">ALL</ToggleButton>
        </ToggleButtonGroup>

        <Box display="flex" gap={1} ml="auto">
          {agentSignals && (
            <Tooltip title="AI Analysis">
              <IconButton
                size="small"
                sx={{
                  color: consensus?.signal === 'BUY' ? CHART_COLORS.candle.up :
                         consensus?.signal === 'SELL' ? CHART_COLORS.candle.down :
                         '#787b86'
                }}
              >
                <Badge
                  badgeContent={consensus?.confidence ? `${Math.round(consensus.confidence * 100)}%` : ''}
                  color="primary"
                >
                  <PsychologyIcon />
                </Badge>
              </IconButton>
            </Tooltip>
          )}

          <Tooltip title="Drawing Tools">
            <IconButton
              size="small"
              onClick={() => setDrawingMode(drawingMode ? null : 'line')}
              sx={{ color: drawingMode ? '#2196f3' : '#787b86' }}
            >
              <DrawIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Refresh">
            <IconButton
              size="small"
              onClick={() => window.location.reload()}
              sx={{ color: '#787b86' }}
            >
              <RefreshIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Fullscreen">
            <IconButton
              size="small"
              onClick={handleFullscreen}
              sx={{ color: '#787b86' }}
            >
              <FullscreenIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Settings">
            <IconButton
              size="small"
              onClick={(e) => setSettingsAnchor(e.currentTarget)}
              sx={{ color: '#787b86' }}
            >
              <SettingsIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Header>

      <ChartArea>
        <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />

        {loading && (
          <LoadingOverlay>
            <CircularProgress sx={{ color: '#2196f3' }} />
          </LoadingOverlay>
        )}

        {error && (
          <Box
            position="absolute"
            top="50%"
            left="50%"
            sx={{ transform: 'translate(-50%, -50%)', textAlign: 'center' }}
          >
            <Typography color="error" gutterBottom>
              Failed to load market data
            </Typography>
            <Typography color="text.secondary" variant="body2">
              {error}
            </Typography>
          </Box>
        )}

        {agentSignals && chartRef.current && (
          <AgentSignalOverlay
            signals={agentSignals}
            consensus={consensus}
            isAnalyzing={isAnalyzing}
            chart={chartRef.current}
          />
        )}

        {drawingMode && chartRef.current && (
          <DrawingToolbar
            mode={drawingMode}
            onModeChange={setDrawingMode}
            chart={chartRef.current}
            onDrawingComplete={addDrawing}
          />
        )}
      </ChartArea>

      {/* Settings Menu */}
      <Menu
        anchorEl={settingsAnchor}
        open={Boolean(settingsAnchor)}
        onClose={() => setSettingsAnchor(null)}
        PaperProps={{
          sx: {
            backgroundColor: '#1e222d',
            border: '1px solid #2a2e39',
            minWidth: 200,
          }
        }}
      >
        <Box px={2} py={1}>
          <Typography variant="subtitle2" sx={{ color: '#d1d4dc', fontWeight: 600 }}>
            Indicators
          </Typography>
        </Box>
        <Divider sx={{ borderColor: '#2a2e39' }} />
        <Box px={1} py={1}>
          <FormGroup>
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.ma20}
                  onChange={(e) => setIndicators({ ...indicators, ma20: e.target.checked })}
                />
              }
              label="MA 20"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.ma50}
                  onChange={(e) => setIndicators({ ...indicators, ma50: e.target.checked })}
                />
              }
              label="MA 50"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.ma200}
                  onChange={(e) => setIndicators({ ...indicators, ma200: e.target.checked })}
                />
              }
              label="MA 200"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.bollinger}
                  onChange={(e) => setIndicators({ ...indicators, bollinger: e.target.checked })}
                />
              }
              label="Bollinger Bands"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
          </FormGroup>
        </Box>
        <Divider sx={{ borderColor: '#2a2e39' }} />
        <Box px={1} py={1}>
          <FormGroup>
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.rsi}
                  onChange={(e) => setIndicators({ ...indicators, rsi: e.target.checked })}
                />
              }
              label="RSI"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.macd}
                  onChange={(e) => setIndicators({ ...indicators, macd: e.target.checked })}
                />
              }
              label="MACD"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={indicators.volume}
                  onChange={(e) => setIndicators({ ...indicators, volume: e.target.checked })}
                />
              }
              label="Volume"
              sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem', color: '#d1d4dc' } }}
            />
          </FormGroup>
        </Box>
      </Menu>
    </Container>
  );
};

export default AITradingChart;
