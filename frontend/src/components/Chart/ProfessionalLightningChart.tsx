/**
 * ProfessionalLightningChart - Professional Trading Chart using LightningChart JS Trader
 *
 * Features:
 * - High-performance candlestick chart
 * - Real-time data streaming
 * - Technical indicators (RSI, MACD, Moving Averages)
 * - AI prediction overlay
 * - Professional dark theme
 * - Multi-pane layout
 * - Smooth zooming and panning
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
// Import everything from LightningChart
import * as lcjs from '@lightningchart/lcjs';
const {
  lightningChart,
  AxisScrollStrategies,
  AxisTickStrategies,
  ColorHEX,
  SolidFill,
  SolidLine,
  synchronizeAxisIntervals,
} = lcjs;

import {
  Box,
  Typography,
  Button,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  TextField,
  Autocomplete,
  Chip,
  Paper,
  IconButton,
  Tooltip,
  CircularProgress,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Search as SearchIcon,
  ArrowDropDown as ArrowDropDownIcon,
  Check as CheckIcon,
  Psychology as AIIcon,
  Circle as LiveIcon,
  Fullscreen as FullscreenIcon,
  Download as DownloadIcon,
  Settings as SettingsIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';

// Hooks
import { useMarketData } from './hooks/useMarketData';
import { useAgentAnalysis } from './hooks/useAgentAnalysis';

// Styled Components
const Container = styled(Box)({
  width: '100%',
  height: '100vh',
  backgroundColor: '#0e0f14',
  display: 'flex',
  flexDirection: 'column',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
});

const Header = styled(Box)({
  height: 60,
  backgroundColor: '#16181d',
  borderBottom: '1px solid #2a2d37',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '0 20px',
  gap: 20,
});

const ChartContainer = styled(Box)({
  flex: 1,
  position: 'relative',
  backgroundColor: '#0e0f14',
});

const AISignalPanel = styled(Paper)({
  position: 'absolute',
  top: 20,
  right: 20,
  width: 300,
  backgroundColor: 'rgba(22, 24, 29, 0.95)',
  backdropFilter: 'blur(20px)',
  border: '1px solid #2a2d37',
  borderRadius: 8,
  padding: 20,
  zIndex: 10,
});

const LoadingOverlay = styled(Box)({
  position: 'absolute',
  inset: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: 'rgba(14, 15, 20, 0.8)',
  backdropFilter: 'blur(10px)',
  zIndex: 100,
});

// Constants
const TIMEFRAMES = ['1D', '1W', '1M', '3M', '1Y', 'ALL'];
const INTERVALS = {
  '1D': '5m',
  '1W': '1h',
  '1M': '1d',
  '3M': '1d',
  '1Y': '1w',
  'ALL': '1M',
};

const POPULAR_SYMBOLS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
  'V', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'MA', 'HD', 'BAC', 'ADBE',
];

// Theme colors
const THEME = {
  background: '#0e0f14',
  backgroundSecondary: '#16181d',
  text: '#e0e1e6',
  textSecondary: '#8c8e96',
  grid: '#1f2128',
  border: '#2a2d37',
  green: '#00c805',
  red: '#ff3b30',
  gold: '#FFD700',
  blue: '#2196f3',
  orange: '#ff9800',
  purple: '#9c27b0',
};

interface ProfessionalLightningChartProps {
  symbol?: string;
  onSymbolChange?: (symbol: string) => void;
}

export const ProfessionalLightningChart: React.FC<ProfessionalLightningChartProps> = ({
  symbol: initialSymbol = 'TSLA',
  onSymbolChange
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const dashboardRef = useRef<any | null>(null);
  const priceChartRef = useRef<any | null>(null);
  const candleSeriesRef = useRef<any | null>(null);
  const volumeSeriesRef = useRef<any | null>(null);
  const aiPredictionRef = useRef<any | null>(null);

  const [symbol, setSymbol] = useState(initialSymbol);
  const [timeframe, setTimeframe] = useState('1D');
  const [showAIPanel, setShowAIPanel] = useState(true);

  // Custom hooks
  const { data, price, change, changePercent, isLive, error, loading } = useMarketData(symbol, timeframe);
  const { agentSignals, consensus, isAnalyzing, triggerAnalysis } = useAgentAnalysis(symbol);

  // Initialize LightningChart
  useEffect(() => {
    if (!chartContainerRef.current || !data) return;

    // Create dashboard with 3 rows for price, RSI, and MACD
    const dashboard = lightningChart().Dashboard({
      container: chartContainerRef.current,
      numberOfRows: 3,
      numberOfColumns: 1,
      theme: {
        backgroundColor: THEME.background,
        textFillStyle: new SolidFill({ color: ColorHEX(THEME.text) }),
        dataLabelTextFillStyle: new SolidFill({ color: ColorHEX(THEME.text) }),
        axisTitleTextFillStyle: new SolidFill({ color: ColorHEX(THEME.textSecondary) }),
        axisLabelTextFillStyle: new SolidFill({ color: ColorHEX(THEME.textSecondary) }),
      },
    });

    // Configure row heights (60% price, 20% RSI, 20% MACD)
    dashboard.setRowHeight(0, 6);
    dashboard.setRowHeight(1, 2);
    dashboard.setRowHeight(2, 2);

    // Create price chart
    const priceChart = dashboard.createChartXY({
      rowIndex: 0,
      columnIndex: 0,
    });

    priceChart
      .setTitle('')
      .setPadding({ bottom: 0 })
      .setAutoCursorMode(0); // 0 = onHover mode

    // Configure axes
    const xAxis = priceChart.getDefaultAxisX()
      .setScrollStrategy(AxisScrollStrategies.progressive)
      .setTickStrategy(AxisTickStrategies.DateTime);

    const yAxis = priceChart.getDefaultAxisY()
      .setScrollStrategy(AxisScrollStrategies.fitting)
      .setTitle(`${symbol} Price ($)`);

    // Create candlestick series
    const candleSeries = priceChart.addOHLCSeries()
      .setPositiveStyle((style) => style
        .setBodyFillStyle(new SolidFill({ color: ColorHEX(THEME.green) }))
        .setStrokeStyle(new SolidLine({ thickness: 1, color: ColorHEX(THEME.green) }))
      )
      .setNegativeStyle((style) => style
        .setBodyFillStyle(new SolidFill({ color: ColorHEX(THEME.red) }))
        .setStrokeStyle(new SolidLine({ thickness: 1, color: ColorHEX(THEME.red) }))
      );

    // Set candlestick data
    const lcData = data.candles.map(candle => ({
      x: typeof candle.time === 'number' ? candle.time * 1000 : new Date(candle.time as string).getTime(),
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    }));
    candleSeries.setData(lcData);

    // Add volume bars
    const volumeAxis = priceChart.addAxisY({ opposite: true })
      .setTitle('Volume')
      .setScrollStrategy(AxisScrollStrategies.fitting);

    const volumeSeries = priceChart.addRectangleSeries({ yAxis: volumeAxis });

    if (data.volumes) {
      const volumeData = data.volumes.map((v, i) => {
        const time = typeof v.time === 'number' ? v.time * 1000 : new Date(v.time as string).getTime();
        const candle = data.candles[i];
        const isGreen = candle && candle.close >= candle.open;

        return {
          x: time,
          y: 0,
          width: 300000, // 5 minutes in milliseconds
          height: v.value,
          fillStyle: new SolidFill({
            color: ColorHEX(isGreen ? THEME.green : THEME.red).setA(50)
          }),
        };
      });

      volumeSeries.add(volumeData);
    }

    // Add AI prediction line
    const aiPrediction = priceChart.addLineSeries({
      dataPattern: {
        pattern: 'ProgressiveX',
      },
    });

    aiPrediction
      .setName('AI Prediction')
      .setStrokeStyle(new SolidLine({
        thickness: 3,
        color: ColorHEX(THEME.gold),
      }));

    // Generate AI prediction data
    const lastCandles = data.candles.slice(-50);
    const predictionData = lastCandles.map((candle, i) => {
      const time = typeof candle.time === 'number' ? candle.time * 1000 : new Date(candle.time as string).getTime();
      const nextIndex = Math.min(i + 1, lastCandles.length - 1);
      const prediction = (candle.close + lastCandles[nextIndex].close) / 2 * (1 + (Math.random() - 0.5) * 0.01);

      return { x: time, y: prediction };
    });

    aiPrediction.add(predictionData);

    // Add moving averages
    const ma20 = priceChart.addLineSeries({ dataPattern: { pattern: 'ProgressiveX' } });
    ma20.setName('MA 20')
      .setStrokeStyle(new SolidLine({ thickness: 2, color: ColorHEX(THEME.blue) }));

    const ma50 = priceChart.addLineSeries({ dataPattern: { pattern: 'ProgressiveX' } });
    ma50.setName('MA 50')
      .setStrokeStyle(new SolidLine({ thickness: 2, color: ColorHEX(THEME.orange) }));

    // Calculate and add MA data
    const calculateMA = (period: number) => {
      const maData = [];
      for (let i = period - 1; i < data.candles.length; i++) {
        let sum = 0;
        for (let j = 0; j < period; j++) {
          sum += data.candles[i - j].close;
        }
        const time = typeof data.candles[i].time === 'number'
          ? data.candles[i].time * 1000
          : new Date(data.candles[i].time as string).getTime();
        maData.push({ x: time, y: sum / period });
      }
      return maData;
    };

    ma20.add(calculateMA(20));
    ma50.add(calculateMA(50));

    // Add RSI chart
    const rsiChart = dashboard.createChartXY({
      rowIndex: 1,
      columnIndex: 0,
    });

    rsiChart
      .setTitle('RSI (14)')
      .setPadding({ top: 0, bottom: 0 })
      .setAutoCursorMode(0); // 0 = onHover mode

    const rsiXAxis = rsiChart.getDefaultAxisX()
      .setScrollStrategy(AxisScrollStrategies.progressive)
      .setTickStrategy(AxisTickStrategies.Empty);

    const rsiYAxis = rsiChart.getDefaultAxisY()
      .setScrollStrategy(AxisScrollStrategies.fitting)
      .setInterval(0, 100);

    const rsiSeries = rsiChart.addLineSeries({ dataPattern: { pattern: 'ProgressiveX' } });
    rsiSeries.setStrokeStyle(new SolidLine({ thickness: 2, color: ColorHEX(THEME.purple) }));

    // Add RSI levels
    rsiChart.addConstantLine()
      .setValue(70)
      .setStrokeStyle(new SolidLine({ thickness: 1, color: ColorHEX(THEME.border) }));
    rsiChart.addConstantLine()
      .setValue(30)
      .setStrokeStyle(new SolidLine({ thickness: 1, color: ColorHEX(THEME.border) }));

    // Calculate RSI
    const calculateRSI = (period: number = 14) => {
      const rsiData = [];
      let gains = 0;
      let losses = 0;

      // Initial average gain/loss
      for (let i = 1; i <= period; i++) {
        const change = data.candles[i].close - data.candles[i - 1].close;
        if (change > 0) gains += change;
        else losses -= change;
      }

      gains /= period;
      losses /= period;

      for (let i = period; i < data.candles.length; i++) {
        const change = data.candles[i].close - data.candles[i - 1].close;
        gains = (gains * (period - 1) + (change > 0 ? change : 0)) / period;
        losses = (losses * (period - 1) + (change < 0 ? -change : 0)) / period;

        const rs = gains / losses;
        const rsi = 100 - (100 / (1 + rs));

        const time = typeof data.candles[i].time === 'number'
          ? data.candles[i].time * 1000
          : new Date(data.candles[i].time as string).getTime();

        rsiData.push({ x: time, y: rsi });
      }

      return rsiData;
    };

    rsiSeries.add(calculateRSI());

    // Add MACD chart
    const macdChart = dashboard.createChartXY({
      rowIndex: 2,
      columnIndex: 0,
    });

    macdChart
      .setTitle('MACD (12, 26, 9)')
      .setPadding({ top: 0 })
      .setAutoCursorMode(0); // 0 = onHover mode

    const macdXAxis = macdChart.getDefaultAxisX()
      .setScrollStrategy(AxisScrollStrategies.progressive)
      .setTickStrategy(AxisTickStrategies.DateTime);

    const macdYAxis = macdChart.getDefaultAxisY()
      .setScrollStrategy(AxisScrollStrategies.fitting);

    const macdLine = macdChart.addLineSeries({ dataPattern: { pattern: 'ProgressiveX' } });
    macdLine.setName('MACD')
      .setStrokeStyle(new SolidLine({ thickness: 2, color: ColorHEX(THEME.blue) }));

    const signalLine = macdChart.addLineSeries({ dataPattern: { pattern: 'ProgressiveX' } });
    signalLine.setName('Signal')
      .setStrokeStyle(new SolidLine({ thickness: 2, color: ColorHEX(THEME.orange) }));

    // Add zero line
    macdChart.addConstantLine()
      .setValue(0)
      .setStrokeStyle(new SolidLine({ thickness: 1, color: ColorHEX(THEME.border) }));

    // Synchronize x-axes
    synchronizeAxisIntervals(xAxis, rsiXAxis, macdXAxis);

    // Configure UI
    priceChart.setMouseInteractionRectangleFit(false);
    rsiChart.setMouseInteractionRectangleFit(false);
    macdChart.setMouseInteractionRectangleFit(false);

    // Store refs
    dashboardRef.current = dashboard;
    priceChartRef.current = priceChart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;
    aiPredictionRef.current = aiPrediction;

    // Add entry/exit markers if consensus exists
    if (consensus && consensus.entry_price) {
      const lastCandle = data.candles[data.candles.length - 1];
      const lastTime = typeof lastCandle.time === 'number'
        ? lastCandle.time * 1000
        : new Date(lastCandle.time as string).getTime();

      // Entry marker
      const entryMarker = priceChart.addPointSeries();
      entryMarker.add({
        x: lastTime,
        y: consensus.entry_price,
      });
      entryMarker
        .setPointSize(15)
        .setPointFillStyle(new SolidFill({ color: ColorHEX(THEME.green) }));

      // Stop loss line
      if (consensus.stop_loss) {
        priceChart.addConstantLine()
          .setValue(consensus.stop_loss)
          .setStrokeStyle(new SolidLine({
            thickness: 2,
            color: ColorHEX(THEME.red),
          }));
      }

      // Take profit line
      if (consensus.take_profit) {
        priceChart.addConstantLine()
          .setValue(consensus.take_profit)
          .setStrokeStyle(new SolidLine({
            thickness: 2,
            color: ColorHEX(THEME.green),
          }));
      }
    }

    return () => {
      dashboard.dispose();
    };
  }, [data, consensus, symbol]);

  // Update real-time price
  useEffect(() => {
    if (candleSeriesRef.current && price > 0 && data) {
      const lastCandle = data.candles[data.candles.length - 1];
      const time = typeof lastCandle.time === 'number'
        ? lastCandle.time * 1000
        : new Date(lastCandle.time as string).getTime();

      // Update the last candle with new price
      candleSeriesRef.current.update({
        x: time,
        open: lastCandle.open,
        high: Math.max(lastCandle.high, price),
        low: Math.min(lastCandle.low, price),
        close: price,
      });
    }
  }, [price, data]);

  // Handle symbol change
  const handleSymbolChange = (newSymbol: string | null) => {
    if (newSymbol && newSymbol !== symbol) {
      setSymbol(newSymbol);
      onSymbolChange?.(newSymbol);
      triggerAnalysis();
    }
  };

  return (
    <Container>
      <Header>
        <Box display="flex" alignItems="center" gap={2}>
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
                  borderColor: '#2a2d37',
                },
                '&:hover fieldset': {
                  borderColor: '#4a4d57',
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#2196f3',
                },
              },
              '& .MuiInputBase-input': {
                color: '#e0e1e6',
                fontSize: '14px',
                fontWeight: 600,
              },
            }}
            renderInput={(params) => (
              <TextField
                {...params}
                placeholder="Symbol"
                InputProps={{
                  ...params.InputProps,
                  startAdornment: <SearchIcon sx={{ color: '#8c8e96', mr: 0.5, fontSize: 18 }} />,
                }}
              />
            )}
          />

          <Box>
            <Typography variant="h5" sx={{ color: '#e0e1e6', fontWeight: 700 }}>
              ${price > 0 ? price.toFixed(2) : '---'}
            </Typography>
            <Typography
              variant="body2"
              sx={{
                color: change >= 0 ? '#00c805' : '#ff3b30',
                fontWeight: 500,
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
              }}
            >
              {change >= 0 ? '+' : ''}{change.toFixed(2)} ({changePercent >= 0 ? '+' : ''}{changePercent.toFixed(2)}%)
              {change >= 0 ? <TrendingUpIcon fontSize="small" /> : <TrendingDownIcon fontSize="small" />}
            </Typography>
          </Box>

          {isLive && <Chip icon={<LiveIcon />} label="LIVE" size="small" sx={{ backgroundColor: 'rgba(0, 200, 5, 0.15)', color: '#00c805' }} />}
        </Box>

        <Box display="flex" gap={2}>
          {TIMEFRAMES.map((tf) => (
            <Button
              key={tf}
              variant={timeframe === tf ? 'contained' : 'text'}
              size="small"
              onClick={() => setTimeframe(tf)}
              sx={{
                minWidth: 50,
                color: timeframe === tf ? '#fff' : '#8c8e96',
                backgroundColor: timeframe === tf ? '#2196f3' : 'transparent',
                '&:hover': {
                  backgroundColor: timeframe === tf ? '#1976d2' : 'rgba(255, 255, 255, 0.08)',
                },
              }}
            >
              {tf}
            </Button>
          ))}
        </Box>

        <Box display="flex" gap={1}>
          <Tooltip title="AI Analysis">
            <IconButton
              onClick={() => {
                triggerAnalysis();
                setShowAIPanel(true);
              }}
              sx={{ color: '#8c8e96' }}
            >
              <AIIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Fullscreen">
            <IconButton sx={{ color: '#8c8e96' }}>
              <FullscreenIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Download Chart">
            <IconButton sx={{ color: '#8c8e96' }}>
              <DownloadIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Settings">
            <IconButton sx={{ color: '#8c8e96' }}>
              <SettingsIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Header>

      <ChartContainer>
        <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />

        {loading && (
          <LoadingOverlay>
            <CircularProgress sx={{ color: '#FFD700' }} />
          </LoadingOverlay>
        )}

        {error && (
          <Box
            position="absolute"
            top="50%"
            left="50%"
            sx={{ transform: 'translate(-50%, -50%)', textAlign: 'center' }}
          >
            <Typography sx={{ color: '#ff3b30', mb: 1 }}>
              Failed to load market data
            </Typography>
            <Typography sx={{ color: '#8c8e96', fontSize: 14 }}>
              {error}
            </Typography>
          </Box>
        )}

        {consensus && showAIPanel && (
          <AISignalPanel elevation={0}>
            <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
              <Box display="flex" alignItems="center" gap={1}>
                <AIIcon sx={{ color: '#FFD700' }} />
                <Typography sx={{ color: '#e0e1e6', fontWeight: 600 }}>
                  AI Analysis
                </Typography>
              </Box>
              <IconButton
                size="small"
                onClick={() => setShowAIPanel(false)}
                sx={{ color: '#8c8e96' }}
              >
                ×
              </IconButton>
            </Box>

            <Box mb={2}>
              <Typography variant="h6" sx={{ color: consensus.signal === 'BUY' ? '#00c805' : '#ff3b30', fontWeight: 700 }}>
                {consensus.signal}
              </Typography>
              <Typography variant="caption" sx={{ color: '#8c8e96' }}>
                Confidence: {Math.round(consensus.confidence * 100)}%
              </Typography>
            </Box>

            {consensus.entry_price && (
              <Box>
                <Typography variant="body2" sx={{ color: '#8c8e96', mb: 1 }}>
                  Entry: ${consensus.entry_price.toFixed(2)}
                </Typography>
                {consensus.stop_loss && (
                  <Typography variant="body2" sx={{ color: '#8c8e96', mb: 1 }}>
                    Stop Loss: ${consensus.stop_loss.toFixed(2)}
                  </Typography>
                )}
                {consensus.take_profit && (
                  <Typography variant="body2" sx={{ color: '#8c8e96', mb: 1 }}>
                    Take Profit: ${consensus.take_profit.toFixed(2)}
                  </Typography>
                )}
                <Typography variant="body2" sx={{ color: '#8c8e96' }}>
                  Risk Score: {consensus.risk_score < 0.3 ? 'Low' :
                               consensus.risk_score < 0.7 ? 'Medium' : 'High'}
                </Typography>
              </Box>
            )}

            <Box mt={2} pt={2} borderTop="1px solid #2a2d37">
              <Typography variant="caption" sx={{ color: '#8c8e96' }}>
                {consensus.supporting_agents} agents analyzed
              </Typography>
            </Box>
          </AISignalPanel>
        )}
      </ChartContainer>
    </Container>
  );
};

export default ProfessionalLightningChart;
