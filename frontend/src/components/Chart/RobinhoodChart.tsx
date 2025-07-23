/**
 * RobinhoodChart Component
 *
 * A minimal, professional trading chart inspired by Robinhood's design
 * with Material dark theme for stock trading.
 *
 * Features:
 * - Clean candlestick chart with integrated volume
 * - Minimal UI with focus on data
 * - Professional dark theme
 * - Simple timeframe dropdown
 * - Real-time updates
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from 'highcharts-react-official';
import {
  Box,
  Typography,
  CircularProgress,
  TextField,
  Autocomplete,
  Button,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Checkbox,
  IconButton,
  Divider,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Search as SearchIcon,
  ArrowDropDown as ArrowDropDownIcon,
  Refresh as RefreshIcon,
  Share as ShareIcon,
  MoreHoriz as MoreIcon,
  Check as CheckIcon,
} from '@mui/icons-material';

// Hooks
import { useMarketData } from './hooks/useMarketData';
import { useAgentAnalysis } from './hooks/useAgentAnalysis';

// Hooks
import { useMarketData } from './hooks/useMarketData';
import { useAgentAnalysis } from './hooks/useAgentAnalysis';

// Styled Components
const Container = styled(Box)({
  width: '100%',
  height: '100vh',
  backgroundColor: '#ffffff',
  display: 'flex',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
});

const Sidebar = styled(Box)({
  width: 220,
  backgroundColor: '#ffffff',
  borderRight: '1px solid #e0e0e0',
  display: 'flex',
  flexDirection: 'column',
  padding: '16px 0',
});

const MainContent = styled(Box)({
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
});

const Header = styled(Box)({
  padding: '16px 24px',
  borderBottom: '1px solid #e0e0e0',
  backgroundColor: '#ffffff',
});

const PriceRow = styled(Box)({
  display: 'flex',
  alignItems: 'baseline',
  gap: 8,
  marginBottom: 16,
});

const TimeframeTabs = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  gap: 2,
});

const TimeTab = styled(Button)<{ active?: boolean }>(({ active }) => ({
  minWidth: 'auto',
  padding: '6px 12px',
  fontSize: '13px',
  fontWeight: 500,
  color: active ? '#000' : '#6f6f6f',
  backgroundColor: active ? '#f0f0f0' : 'transparent',
  borderRadius: 4,
  textTransform: 'none',
  '&:hover': {
    backgroundColor: '#f0f0f0',
  },
}));

const IntervalButton = styled(Button)({
  padding: '6px 12px',
  fontSize: '13px',
  fontWeight: 500,
  color: '#000',
  backgroundColor: '#00c805',
  borderRadius: 4,
  textTransform: 'none',
  marginLeft: 8,
  '&:hover': {
    backgroundColor: '#00a004',
  },
});

const ChartArea = styled(Box)({
  flex: 1,
  position: 'relative',
  backgroundColor: '#ffffff',
});

const IndicatorRow = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  padding: '8px 16px',
  cursor: 'pointer',
  '&:hover': {
    backgroundColor: '#f7f7f7',
  },
});

const IndicatorLabel = styled(Typography)({
  fontSize: '13px',
  fontWeight: 500,
  color: '#000',
  flex: 1,
});

const IndicatorValue = styled(Typography)({
  fontSize: '13px',
  color: '#6f6f6f',
});

const TradeButton = styled(Button)({
  backgroundColor: '#00c805',
  color: '#ffffff',
  fontWeight: 600,
  fontSize: '14px',
  padding: '10px 24px',
  borderRadius: 4,
  textTransform: 'none',
  position: 'fixed',
  bottom: 24,
  right: 24,
  '&:hover': {
    backgroundColor: '#00a004',
  },
});

// Time periods and intervals
const TIME_PERIODS = ['1D', '1W', '1M', '3M', 'YTD', '1Y', '5Y', 'MAX'];
const INTERVALS = {
  '1D': ['5 min', '10 min', '30 min', '1 hour'],
  '1W': ['5 min', '10 min', '30 min', '1 hour'],
  '1M': ['30 min', '1 hour', '1 day'],
  '3M': ['1 hour', '1 day', '1 week'],
  'YTD': ['1 day', '1 week'],
  '1Y': ['1 day', '1 week', '1 month'],
  '5Y': ['1 week', '1 month'],
  'MAX': ['1 month', '3 months'],
};

// Popular symbols
const POPULAR_SYMBOLS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
  'V', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'MA', 'HD', 'BAC', 'ADBE',
];

interface RobinhoodChartProps {
  symbol?: string;
  onSymbolChange?: (symbol: string) => void;
}

export const RobinhoodChart: React.FC<RobinhoodChartProps> = ({
  symbol: initialSymbol = 'TSLA',
  onSymbolChange
}) => {
  // Chart refs
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const rsiChartRef = useRef<IChartApi | null>(null);
  const macdChartRef = useRef<IChartApi | null>(null);
  const currentPriceLineRef = useRef<any>(null);

  // State
  const [symbol, setSymbol] = useState(initialSymbol);
  const [timeframe, setTimeframe] = useState('1W');
  const [interval, setInterval] = useState('1 hour');
  const [intervalAnchor, setIntervalAnchor] = useState<null | HTMLElement>(null);
  const [indicators, setIndicators] = useState({
    price: true,
    change: true,
    volume: true,
    ma10: true,
    ma50: true,
    ma200: true,
    ema9: true,
    ema12: true,
    vwap: true,
    bollinger: true,
    bollingerMA: true,
    bollingerUpper: true,
    bollingerLower: true,
    rsi: true,
    macd: true,
    signal: true,
  });

  // Custom hooks
  const { data, price, change, changePercent, isLive, error, loading } = useMarketData(symbol, timeframe);
  const { agentSignals, consensus, isAnalyzing, triggerAnalysis } = useAgentAnalysis(symbol);

  // Initialize charts
  useEffect(() => {
    if (!chartContainerRef.current || !data) return;

    const container = chartContainerRef.current;
    container.innerHTML = '';

    // Calculate heights
    const containerHeight = container.clientHeight;
    const containerWidth = container.clientWidth;
    const priceHeight = Math.floor(containerHeight * 0.65);
    const rsiHeight = Math.floor(containerHeight * 0.175);
    const macdHeight = Math.floor(containerHeight * 0.175);

    // Chart options
    const commonOptions = {
      layout: {
        background: { type: ColorType.Solid, color: '#ffffff' },
        textColor: '#666666',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: '#f0f0f0' },
        horzLines: { color: '#f0f0f0' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: '#999999',
          width: 1,
          style: LineStyle.Solid,
          labelBackgroundColor: '#ffffff',
        },
        horzLine: {
          color: '#999999',
          width: 1,
          style: LineStyle.Solid,
          labelBackgroundColor: '#ffffff',
        },
      },
      rightPriceScale: {
        borderColor: '#e0e0e0',
        scaleMargins: { top: 0.1, bottom: 0.2 },
      },
      timeScale: {
        borderColor: '#e0e0e0',
        timeVisible: true,
        secondsVisible: false,
      },
    };

    // Create price chart
    const priceDiv = document.createElement('div');
    priceDiv.style.height = `${priceHeight}px`;
    priceDiv.style.width = '100%';
    container.appendChild(priceDiv);

    const priceChart = createChart(priceDiv, {
      ...commonOptions,
      width: containerWidth,
      height: priceHeight,
    });

    // Add candlestick series
    const candleSeries = priceChart.addCandlestickSeries({
      upColor: '#00c805',
      downColor: '#ff3b30',
      borderUpColor: '#00c805',
      borderDownColor: '#ff3b30',
      wickUpColor: '#00c805',
      wickDownColor: '#ff3b30',
    });
    candleSeries.setData(data.candles);
    candleSeriesRef.current = candleSeries;

    // Add current price line
    if (price > 0) {
      currentPriceLineRef.current = candleSeries.createPriceLine({
        price: price,
        color: '#000000',
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        axisLabelVisible: true,
        title: '',
      });
    }

    // Add volume
    if (indicators.volume && data.volumes) {
      const volumeSeries = priceChart.addHistogramSeries({
        color: '#00c805',
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      });
      volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.9, bottom: 0 },
      });
      volumeSeries.setData(data.volumes.map(v => ({
        ...v,
        color: v.color.includes('26a69a') ? 'rgba(0, 200, 5, 0.2)' : 'rgba(255, 59, 48, 0.2)',
      })));
    }

    // Calculate indicators
    const indicatorData = calculateIndicators(data.candles);

    // Add moving averages
    if (indicators.ma10 && indicatorData.ma20) {
      const ma10Series = priceChart.addLineSeries({
        color: '#2196f3',
        lineWidth: 1,
        title: 'MA(10, 50, 200)',
        lastValueVisible: false,
      });
      ma10Series.setData(indicatorData.ma20);
    }

    if (indicators.ma50 && indicatorData.ma50) {
      const ma50Series = priceChart.addLineSeries({
        color: '#ff6d00',
        lineWidth: 1,
        title: 'MA(50)',
        lastValueVisible: false,
      });
      ma50Series.setData(indicatorData.ma50);
    }

    if (indicators.ma200 && indicatorData.ma200) {
      const ma200Series = priceChart.addLineSeries({
        color: '#9c27b0',
        lineWidth: 1,
        title: 'MA(200)',
        lastValueVisible: false,
      });
      ma200Series.setData(indicatorData.ma200);
    }

    // Add EMAs
    if (indicators.ema9) {
      const ema9Series = priceChart.addLineSeries({
        color: '#4caf50',
        lineWidth: 1,
        title: 'EMA(9)',
        lastValueVisible: false,
      });
      // Use MA20 as placeholder for EMA9
      if (indicatorData.ma20) ema9Series.setData(indicatorData.ma20);
    }

    if (indicators.ema12) {
      const ema12Series = priceChart.addLineSeries({
        color: '#ff5722',
        lineWidth: 1,
        title: 'EMA(12)',
        lastValueVisible: false,
      });
      // Use MA50 as placeholder for EMA12
      if (indicatorData.ma50) ema12Series.setData(indicatorData.ma50);
    }

    // Add VWAP
    if (indicators.vwap && data.volumes) {
      const vwapSeries = priceChart.addLineSeries({
        color: '#000000',
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        title: 'VWAP',
        lastValueVisible: false,
      });
      // Calculate simple VWAP
      const vwapData: LineData[] = [];
      let cumulativeTPV = 0;
      let cumulativeVolume = 0;

      data.candles.forEach((candle, i) => {
        const typicalPrice = (candle.high + candle.low + candle.close) / 3;
        const volume = data.volumes![i]?.value || 0;
        cumulativeTPV += typicalPrice * volume;
        cumulativeVolume += volume;

        vwapData.push({
          time: candle.time,
          value: cumulativeVolume > 0 ? cumulativeTPV / cumulativeVolume : typicalPrice,
        });
      });
      vwapSeries.setData(vwapData);
    }

    // Add Bollinger Bands
    if (indicators.bollinger && indicatorData.bollingerBands) {
      if (indicators.bollingerUpper) {
        const bbUpperSeries = priceChart.addLineSeries({
          color: '#9e9e9e',
          lineWidth: 1,
          title: 'BB Upper',
          lastValueVisible: false,
        });
        bbUpperSeries.setData(indicatorData.bollingerBands.upper);
      }

      if (indicators.bollingerMA) {
        const bbMiddleSeries = priceChart.addLineSeries({
          color: '#757575',
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          title: 'BB Middle',
          lastValueVisible: false,
        });
        bbMiddleSeries.setData(indicatorData.bollingerBands.middle);
      }

      if (indicators.bollingerLower) {
        const bbLowerSeries = priceChart.addLineSeries({
          color: '#9e9e9e',
          lineWidth: 1,
          title: 'BB Lower',
          lastValueVisible: false,
        });
        bbLowerSeries.setData(indicatorData.bollingerBands.lower);
      }
    }

    // Create RSI chart
    if (indicators.rsi) {
      const rsiDiv = document.createElement('div');
      rsiDiv.style.height = `${rsiHeight}px`;
      rsiDiv.style.width = '100%';
      rsiDiv.style.borderTop = '1px solid #e0e0e0';
      container.appendChild(rsiDiv);

      const rsiChart = createChart(rsiDiv, {
        ...commonOptions,
        width: containerWidth,
        height: rsiHeight,
        timeScale: { visible: false },
      });

      // Add RSI label
      const rsiLabel = document.createElement('div');
      rsiLabel.style.position = 'absolute';
      rsiLabel.style.top = '8px';
      rsiLabel.style.left = '8px';
      rsiLabel.style.fontSize = '11px';
      rsiLabel.style.color = '#666666';
      rsiLabel.style.fontWeight = '500';
      rsiLabel.textContent = 'RSI(14)';
      rsiDiv.appendChild(rsiLabel);

      const rsiLine = rsiChart.addLineSeries({
        color: '#ff6d00',
        lineWidth: 2,
      });
      const rsiData = indicatorData.rsi;
      if (rsiData) rsiLine.setData(rsiData);

      // RSI levels
      rsiLine.createPriceLine({ price: 70, color: '#e0e0e0', lineWidth: 1, lineStyle: LineStyle.Dashed });
      rsiLine.createPriceLine({ price: 50, color: '#e0e0e0', lineWidth: 1, lineStyle: LineStyle.Dotted });
      rsiLine.createPriceLine({ price: 30, color: '#e0e0e0', lineWidth: 1, lineStyle: LineStyle.Dashed });

      rsiChartRef.current = rsiChart;
    }

    // Create MACD chart
    if (indicators.macd) {
      const macdDiv = document.createElement('div');
      macdDiv.style.height = `${macdHeight}px`;
      macdDiv.style.width = '100%';
      macdDiv.style.borderTop = '1px solid #e0e0e0';
      container.appendChild(macdDiv);

      const macdChart = createChart(macdDiv, {
        ...commonOptions,
        width: containerWidth,
        height: macdHeight,
      });

      // Add MACD label
      const macdLabel = document.createElement('div');
      macdLabel.style.position = 'absolute';
      macdLabel.style.top = '8px';
      macdLabel.style.left = '8px';
      macdLabel.style.fontSize = '11px';
      macdLabel.style.color = '#666666';
      macdLabel.style.fontWeight = '500';
      macdLabel.textContent = 'MACD(12, 26, 9)';
      macdDiv.appendChild(macdLabel);

      const macdData = indicatorData.macd;
      if (macdData) {
        const macdHistogram = macdChart.addHistogramSeries({
          color: '#4caf50',
          title: 'MACD Histogram',
        });
        macdHistogram.setData(macdData.histogram.map(h => ({
          ...h,
          color: h.value >= 0 ? 'rgba(0, 200, 5, 0.3)' : 'rgba(255, 59, 48, 0.3)',
        })));

        const macdLine = macdChart.addLineSeries({
          color: '#2196f3',
          lineWidth: 2,
          title: 'MACD',
        });
        macdLine.setData(macdData.macd);

        if (indicators.signal) {
          const signalLine = macdChart.addLineSeries({
            color: '#ff6d00',
            lineWidth: 2,
            title: 'Signal',
          });
          signalLine.setData(macdData.signal);
        }

        // Zero line
        macdLine.createPriceLine({ price: 0, color: '#e0e0e0', lineWidth: 1 });
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
      priceChart.applyOptions({ width: newWidth });
      if (rsiChartRef.current) {
        rsiChartRef.current.applyOptions({ width: newWidth });
      }
      if (macdChartRef.current) {
        macdChartRef.current.applyOptions({ width: newWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      charts.forEach(chart => chart.remove());
    };
  }, [data, indicators]);

  // Update current price line
  useEffect(() => {
    if (candleSeriesRef.current && currentPriceLineRef.current && price > 0) {
      candleSeriesRef.current.updatePriceLine(currentPriceLineRef.current, {
        price: price,
      });
    }
  }, [price]);

  // Handle symbol change
  const handleSymbolChange = (newSymbol: string | null) => {
    if (newSymbol && newSymbol !== symbol) {
      setSymbol(newSymbol);
      onSymbolChange?.(newSymbol);
    }
  };

  // Toggle indicator
  const toggleIndicator = (indicator: keyof typeof indicators) => {
    setIndicators(prev => ({
      ...prev,
      [indicator]: !prev[indicator],
    }));
  };

  return (
    <Container>
      <Sidebar>
        <Box px={2} mb={2}>
          <Autocomplete
            value={symbol}
            onChange={(_, newValue) => handleSymbolChange(newValue)}
            options={POPULAR_SYMBOLS}
            size="small"
            sx={{
              '& .MuiOutlinedInput-root': {
                backgroundColor: '#f7f7f7',
                '& fieldset': {
                  borderColor: '#e0e0e0',
                },
                '&:hover fieldset': {
                  borderColor: '#00c805',
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#00c805',
                },
              },
              '& .MuiInputBase-input': {
                color: '#000',
                fontWeight: 500,
                fontSize: '14px',
              },
            }}
            renderInput={(params) => (
              <TextField
                {...params}
                placeholder="Search"
                InputProps={{
                  ...params.InputProps,
                  startAdornment: <SearchIcon sx={{ color: '#6f6f6f', mr: 1, fontSize: 20 }} />,
                }}
              />
            )}
          />
        </Box>

        <Box px={2} mb={1}>
          <Typography variant="caption" sx={{ color: '#6f6f6f', fontWeight: 600 }}>
            Indicators
          </Typography>
        </Box>

        <Box sx={{ flex: 1, overflowY: 'auto' }}>
          <IndicatorRow onClick={() => toggleIndicator('price')}>
            <CheckIcon sx={{ fontSize: 18, color: indicators.price ? '#00c805' : 'transparent', mr: 1 }} />
            <IndicatorLabel>Price</IndicatorLabel>
            <IndicatorValue>${price.toFixed(2)}</IndicatorValue>
          </IndicatorRow>

          <IndicatorRow onClick={() => toggleIndicator('change')}>
            <CheckIcon sx={{ fontSize: 18, color: indicators.change ? '#00c805' : 'transparent', mr: 1 }} />
            <IndicatorLabel>Change</IndicatorLabel>
            <IndicatorValue sx={{ color: change >= 0 ? '#00c805' : '#ff3b30' }}>
              {changePercent.toFixed(2)}%
            </IndicatorValue>
          </IndicatorRow>

          <IndicatorRow onClick={() => toggleIndicator('volume')}>
            <CheckIcon sx={{ fontSize: 18, color: indicators.volume ? '#00c805' : 'transparent', mr: 1 }} />
            <IndicatorLabel>Volume</IndicatorLabel>
            <IndicatorValue>{data?.volumes?.[data.volumes.length - 1]?.value.toLocaleString() || '0'}</IndicatorValue>
          </IndicatorRow>

          <Divider sx={{ my: 1 }} />

          <IndicatorRow onClick={() => toggleIndicator('ma10')}>
            <CheckIcon sx={{ fontSize: 18, color: indicators.ma10 ? '#00c805' : 'transparent', mr: 1 }} />
            <IndicatorLabel>MA(10)</IndicatorLabel>
            <IndicatorValue>$327.59</IndicatorValue>
          </IndicatorRow>

          <IndicatorRow onClick={() => toggleIndicator('ma50')}>
            <CheckIcon sx={{ fontSize: 18, color: indicators.ma50 ? '#00c805' : 'transparent', mr: 1 }} />
            <IndicatorLabel>MA(50)</IndicatorLabel>
            <IndicatorValue>$323.44</IndicatorValue>
          </IndicatorRow>

          <IndicatorRow onClick={() => toggleIndicator('ma200')}>
            <CheckIcon sx={{ fontSize: 18, color: indicators.ma200 ? '#00c805' : 'transparent', mr: 1 }} />
            <IndicatorLabel>MA(200)</IndicatorLabel>
            <IndicatorValue>$321.38</IndicatorValue>
          </IndicatorRow>

          <IndicatorRow onClick={() => toggleIndicator('ema9')}>
            <CheckIcon sx={{ fontSize: 18, color: indicators.ema9 ? '#00c805' : 'transparent', mr: 1 }} />
            <IndicatorLabel>EMA(9)</IndicatorLabel>
            <IndicatorValue>$327.57</IndicatorValue>
          </IndicatorRow>

          <IndicatorRow onClick={() => toggleIndicator('vwap')}>
            <CheckIcon sx={{ fontSize: 18, color: indicators.vwap ? '#00c805' : 'transparent', mr: 1 }} />
            <IndicatorLabel>VWAP</IndicatorLabel>
            <IndicatorValue>$326.96</IndicatorValue>
          </IndicatorRow>

          <IndicatorRow onClick={() => toggleIndicator('bollinger')}>
            <CheckIcon sx={{ fontSize: 18, color: indicators.bollinger ? '#00c805' : 'transparent', mr: 1 }} />
            <IndicatorLabel>BOLL(20)</IndicatorLabel>
            <IndicatorValue>$329.69</IndicatorValue>
          </IndicatorRow>

          <Divider sx={{ my: 1 }} />

          <IndicatorRow onClick={() => toggleIndicator('rsi')}>
            <CheckIcon sx={{ fontSize: 18, color: indicators.rsi ? '#00c805' : 'transparent', mr: 1 }} />
            <IndicatorLabel>RSI(14)</IndicatorLabel>
            <IndicatorValue>55.81</IndicatorValue>
          </IndicatorRow>

          <IndicatorRow onClick={() => toggleIndicator('macd')}>
            <CheckIcon sx={{ fontSize: 18, color: indicators.macd ? '#00c805' : 'transparent', mr: 1 }} />
            <IndicatorLabel>MACD(12, 26, 9)</IndicatorLabel>
            <IndicatorValue>1.24</IndicatorValue>
          </IndicatorRow>

          <IndicatorRow onClick={() => toggleIndicator('signal')}>
            <CheckIcon sx={{ fontSize: 18, color: indicators.signal ? '#00c805' : 'transparent', mr: 1 }} />
            <IndicatorLabel>Signal</IndicatorLabel>
            <IndicatorValue>1.55</IndicatorValue>
          </IndicatorRow>
        </Box>
      </Sidebar>

      <MainContent>
        <Header>
          <PriceRow>
            <Typography variant="h4" sx={{ fontWeight: 700, color: '#000' }}>
              ${price > 0 ? price.toFixed(2) : '---'}
            </Typography>
            <Typography
              variant="h6"
              sx={{
                color: change >= 0 ? '#00c805' : '#ff3b30',
                fontWeight: 500,
              }}
            >
              {change >= 0 ? '+' : ''}${Math.abs(change).toFixed(2)}
            </Typography>
            <Typography
              variant="body1"
              sx={{
                color: change >= 0 ? '#00c805' : '#ff3b30',
                fontWeight: 400,
              }}
            >
              ({changePercent >= 0 ? '+' : ''}{changePercent.toFixed(2)}%)
            </Typography>
          </PriceRow>

          <TimeframeTabs>
            {TIME_PERIODS.map((period) => (
              <TimeTab
                key={period}
                active={timeframe === period}
                onClick={() => setTimeframe(period)}
              >
                {period}
              </TimeTab>
            ))}

            <IntervalButton
              endIcon={<ArrowDropDownIcon />}
              onClick={(e) => setIntervalAnchor(e.currentTarget)}
            >
              {timeframe} +
            </IntervalButton>

            <Box ml="auto" display="flex" gap={1}>
              <IconButton size="small" sx={{ color: '#6f6f6f' }}>
                <ShareIcon fontSize="small" />
              </IconButton>
              <IconButton size="small" sx={{ color: '#6f6f6f' }}>
                <RefreshIcon fontSize="small" />
              </IconButton>
              <IconButton size="small" sx={{ color: '#6f6f6f' }}>
                <MoreIcon fontSize="small" />
              </IconButton>
            </Box>
          </TimeframeTabs>
        </Header>

        <ChartArea>
          <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />

          {loading && (
            <Box
              position="absolute"
              top="50%"
              left="50%"
              sx={{ transform: 'translate(-50%, -50%)' }}
            >
              <CircularProgress sx={{ color: '#00c805' }} />
            </Box>
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
        </ChartArea>

        <TradeButton variant="contained">
          Trade {symbol}
        </TradeButton>
      </MainContent>

      {/* Interval dropdown menu */}
      <Menu
        anchorEl={intervalAnchor}
        open={Boolean(intervalAnchor)}
        onClose={() => setIntervalAnchor(null)}
        PaperProps={{
          sx: {
            minWidth: 150,
            boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
          },
        }}
      >
        <Box px={2} py={1}>
          <Typography variant="caption" sx={{ color: '#6f6f6f', fontWeight: 600 }}>
            Time Interval
          </Typography>
        </Box>
        {INTERVALS[timeframe]?.map((int) => (
          <MenuItem
            key={int}
            selected={interval === int}
            onClick={() => {
              setInterval(int);
              setIntervalAnchor(null);
            }}
            sx={{
              fontSize: '14px',
              py: 1,
              '&.Mui-selected': {
                backgroundColor: 'rgba(0, 200, 5, 0.1)',
              },
            }}
          >
            <ListItemIcon>
              {interval === int && <CheckIcon sx={{ fontSize: 18, color: '#00c805' }} />}
            </ListItemIcon>
            <ListItemText>{int}</ListItemText>
          </MenuItem>
        ))}
      </Menu>
    </Container>
  );
};

export default RobinhoodChart;
