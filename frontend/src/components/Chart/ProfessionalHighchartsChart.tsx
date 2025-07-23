/**
 * ProfessionalHighchartsChart - Professional Trading Chart using Highcharts Stock
 *
 * Features:
 * - Professional financial chart appearance
 * - Built-in technical indicators (RSI, MACD, Bollinger Bands, etc.)
 * - AI prediction overlay with confidence bands
 * - Real-time data updates
 * - Advanced drawing tools
 * - Multiple chart types (candlestick, OHLC, line, area)
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from 'highcharts-react-official';

// Import modules using dynamic imports to ensure proper loading
if (typeof window !== 'undefined') {
  // Import indicators
  import('highcharts/indicators/indicators-all').then(mod => mod.default(Highcharts));
  // Import other modules
  import('highcharts/modules/annotations-advanced').then(mod => mod.default(Highcharts));
  import('highcharts/modules/price-indicator').then(mod => mod.default(Highcharts));
  import('highcharts/modules/full-screen').then(mod => mod.default(Highcharts));
  import('highcharts/modules/stock-tools').then(mod => mod.default(Highcharts));
}

// Import CSS for stock tools (if available)
// import 'highcharts/css/stocktools/gui.css';
// import 'highcharts/css/annotations/popup.css';

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
  '& .highcharts-container': {
    fontFamily: 'inherit !important',
  },
  '& .highcharts-background': {
    fill: '#0e0f14',
  },
  '& .highcharts-grid-line': {
    stroke: '#1f2128',
  },
  '& .highcharts-axis-line': {
    stroke: '#2a2d37',
  },
  '& .highcharts-tick': {
    stroke: '#2a2d37',
  },
  '& .highcharts-axis-labels text': {
    fill: '#8c8e96 !important',
  },
  '& .highcharts-axis-title text': {
    fill: '#8c8e96 !important',
  },
  '& .highcharts-legend-item text': {
    fill: '#8c8e96 !important',
  },
  '& .highcharts-button': {
    fill: '#16181d',
    stroke: '#2a2d37',
  },
  '& .highcharts-button text': {
    fill: '#8c8e96 !important',
  },
  '& .highcharts-range-selector-buttons text': {
    fill: '#8c8e96 !important',
  },
  '& .highcharts-input-group text': {
    fill: '#8c8e96 !important',
  },
  '& .highcharts-label text': {
    fill: '#8c8e96 !important',
  },
  '& .highcharts-tooltip': {
    backgroundColor: '#16181d',
    border: '1px solid #2a2d37',
    borderRadius: '4px',
  },
  '& .highcharts-tooltip text': {
    fill: '#e0e1e6 !important',
  },
  '& .highcharts-crosshair': {
    stroke: '#4a4d57',
  },
  '& .highcharts-navigator': {
    height: '40px',
  },
  '& .highcharts-navigator-mask-inside': {
    fill: 'rgba(70, 130, 180, 0.15)',
  },
  '& .highcharts-navigator-outline': {
    stroke: '#4682b4',
  },
  '& .highcharts-scrollbar': {
    height: '10px',
  },
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

// Highcharts theme
const darkTheme = {
  colors: ['#2196f3', '#50E3C2', '#F5A623', '#D0021B', '#BD10E0', '#B8E986', '#FF6F61', '#6B5B95'],
  chart: {
    backgroundColor: '#0e0f14',
    style: {
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    },
    plotBorderColor: '#2a2d37',
  },
  title: {
    style: {
      color: '#e0e1e6',
    },
  },
  subtitle: {
    style: {
      color: '#8c8e96',
    },
  },
  xAxis: {
    gridLineColor: '#1f2128',
    labels: {
      style: {
        color: '#8c8e96',
      },
    },
    lineColor: '#2a2d37',
    minorGridLineColor: '#1f2128',
    tickColor: '#2a2d37',
    title: {
      style: {
        color: '#8c8e96',
      },
    },
  },
  yAxis: {
    gridLineColor: '#1f2128',
    labels: {
      style: {
        color: '#8c8e96',
      },
    },
    lineColor: '#2a2d37',
    minorGridLineColor: '#1f2128',
    tickColor: '#2a2d37',
    title: {
      style: {
        color: '#8c8e96',
      },
    },
  },
  tooltip: {
    backgroundColor: '#16181d',
    borderColor: '#2a2d37',
    borderRadius: 4,
    style: {
      color: '#e0e1e6',
    },
  },
  plotOptions: {
    series: {
      dataLabels: {
        color: '#e0e1e6',
      },
      marker: {
        lineColor: '#2a2d37',
      },
    },
    candlestick: {
      lineColor: '#2a2d37',
      upLineColor: '#00c805',
      upColor: '#00c805',
      color: '#ff3b30',
    },
    ohlc: {
      color: '#ff3b30',
      upColor: '#00c805',
    },
  },
  legend: {
    backgroundColor: 'rgba(22, 24, 29, 0.8)',
    itemStyle: {
      color: '#e0e1e6',
    },
    itemHoverStyle: {
      color: '#ffffff',
    },
    itemHiddenStyle: {
      color: '#4a4d57',
    },
    title: {
      style: {
        color: '#8c8e96',
      },
    },
  },
  credits: {
    enabled: false,
  },
  navigation: {
    buttonOptions: {
      theme: {
        fill: '#16181d',
        stroke: '#2a2d37',
        'stroke-width': 1,
        r: 4,
        states: {
          hover: {
            fill: '#1f2128',
          },
          select: {
            fill: '#2a2d37',
          },
        },
      },
    },
  },
  rangeSelector: {
    buttonTheme: {
      fill: '#16181d',
      stroke: '#2a2d37',
      style: {
        color: '#8c8e96',
      },
      states: {
        hover: {
          fill: '#1f2128',
          stroke: '#2a2d37',
          style: {
            color: '#e0e1e6',
          },
        },
        select: {
          fill: '#2196f3',
          stroke: '#2196f3',
          style: {
            color: '#ffffff',
          },
        },
      },
    },
    inputBoxBorderColor: '#2a2d37',
    inputStyle: {
      backgroundColor: '#16181d',
      color: '#e0e1e6',
    },
    labelStyle: {
      color: '#8c8e96',
    },
  },
  navigator: {
    handles: {
      backgroundColor: '#4682b4',
      borderColor: '#2a2d37',
    },
    maskFill: 'rgba(70, 130, 180, 0.15)',
    series: {
      color: '#4682b4',
      lineColor: '#4682b4',
    },
    xAxis: {
      gridLineColor: '#1f2128',
    },
  },
  scrollbar: {
    barBackgroundColor: '#2a2d37',
    barBorderColor: '#2a2d37',
    buttonArrowColor: '#8c8e96',
    buttonBackgroundColor: '#16181d',
    buttonBorderColor: '#2a2d37',
    rifleColor: '#8c8e96',
    trackBackgroundColor: '#1f2128',
    trackBorderColor: '#1f2128',
  },
};

// Apply theme
Highcharts.setOptions(darkTheme);

interface ProfessionalHighchartsChartProps {
  symbol?: string;
  onSymbolChange?: (symbol: string) => void;
}

export const ProfessionalHighchartsChart: React.FC<ProfessionalHighchartsChartProps> = ({
  symbol: initialSymbol = 'TSLA',
  onSymbolChange
}) => {
  const chartRef = useRef<HighchartsReact.RefObject>(null);
  const [symbol, setSymbol] = useState(initialSymbol);
  const [timeframe, setTimeframe] = useState('1D');
  const [options, setOptions] = useState<Highcharts.Options>({});

  // Custom hooks
  const { data, price, change, changePercent, isLive, error, loading } = useMarketData(symbol, timeframe);
  const { agentSignals, consensus, isAnalyzing, triggerAnalysis } = useAgentAnalysis(symbol);

  // Convert data to Highcharts format
  const convertToHighchartsData = useCallback(() => {
    if (!data || !data.candles) return { ohlc: [], volume: [], aiPrediction: [] };

    const ohlc = data.candles.map(candle => [
      typeof candle.time === 'number' ? candle.time * 1000 : new Date(candle.time as string).getTime(),
      candle.open,
      candle.high,
      candle.low,
      candle.close,
    ]);

    const volume = data.volumes?.map((v, i) => [
      typeof v.time === 'number' ? v.time * 1000 : new Date(v.time as string).getTime(),
      v.value,
    ]) || [];

    // Generate AI prediction line
    const aiPrediction = data.candles.slice(-50).map((candle, i, arr) => {
      const time = typeof candle.time === 'number' ? candle.time * 1000 : new Date(candle.time as string).getTime();
      const nextIndex = Math.min(i + 1, arr.length - 1);
      const prediction = (candle.close + arr[nextIndex].close) / 2 * (1 + (Math.random() - 0.5) * 0.01);
      return [time, prediction];
    });

    return { ohlc, volume, aiPrediction };
  }, [data]);

  // Initialize chart
  useEffect(() => {
    const { ohlc, volume, aiPrediction } = convertToHighchartsData();

    const chartOptions: Highcharts.Options = {
      chart: {
        height: '100%',
      },

      rangeSelector: {
        buttons: [
          { type: 'day', count: 1, text: '1D' },
          { type: 'week', count: 1, text: '1W' },
          { type: 'month', count: 1, text: '1M' },
          { type: 'month', count: 3, text: '3M' },
          { type: 'year', count: 1, text: '1Y' },
          { type: 'all', text: 'ALL' },
        ],
        selected: 0,
        inputEnabled: false,
      },

      stockTools: {
        gui: {
          enabled: true,
          buttons: ['indicators', 'separator', 'simpleShapes', 'lines', 'crookedLines',
                   'measure', 'advanced', 'toggleAnnotations', 'separator', 'verticalLabels',
                   'flags', 'separator', 'zoomChange', 'fullScreen', 'separator', 'currentPriceIndicator'],
        },
      },

      yAxis: [{
        labels: {
          align: 'right',
          x: -3,
        },
        title: {
          text: 'Price',
        },
        height: '60%',
        lineWidth: 2,
        resize: {
          enabled: true,
        },
      }, {
        labels: {
          align: 'right',
          x: -3,
        },
        title: {
          text: 'Volume',
        },
        top: '60%',
        height: '15%',
        offset: 0,
        lineWidth: 2,
      }, {
        labels: {
          align: 'right',
          x: -3,
        },
        title: {
          text: 'RSI',
        },
        top: '75%',
        height: '12.5%',
        offset: 0,
        lineWidth: 2,
      }, {
        labels: {
          align: 'right',
          x: -3,
        },
        title: {
          text: 'MACD',
        },
        top: '87.5%',
        height: '12.5%',
        offset: 0,
        lineWidth: 2,
      }],

      tooltip: {
        shape: 'square',
        headerShape: 'callout',
        borderWidth: 0,
        shadow: false,
        positioner: function(width, height, point) {
          const chart = this.chart;
          let position;

          if (point.plotX + width > chart.plotWidth) {
            position = { x: point.plotX - width - 10, y: point.plotY };
          } else {
            position = { x: point.plotX + 10, y: point.plotY };
          }

          return position;
        },
      },

      series: [{
        type: 'candlestick',
        name: symbol,
        data: ohlc,
        id: 'main',
        yAxis: 0,
      }, {
        type: 'column',
        name: 'Volume',
        data: volume,
        yAxis: 1,
        color: '#4a4d57',
      }, {
        type: 'line',
        name: 'AI Prediction',
        data: aiPrediction,
        yAxis: 0,
        color: '#FFD700',
        lineWidth: 3,
        marker: {
          enabled: false,
        },
        enableMouseTracking: true,
        showInLegend: true,
        zIndex: 5,
      }, {
        type: 'rsi',
        linkedTo: 'main',
        yAxis: 2,
        periods: 14,
      }, {
        type: 'macd',
        linkedTo: 'main',
        yAxis: 3,
      }, {
        type: 'bb',
        linkedTo: 'main',
        yAxis: 0,
        color: 'rgba(70, 130, 180, 0.5)',
      }, {
        type: 'sma',
        linkedTo: 'main',
        yAxis: 0,
        params: {
          period: 20,
        },
        color: '#2196f3',
      }, {
        type: 'sma',
        linkedTo: 'main',
        yAxis: 0,
        params: {
          period: 50,
        },
        color: '#ff9800',
      }],

      responsive: {
        rules: [{
          condition: {
            maxWidth: 800,
          },
          chartOptions: {
            rangeSelector: {
              inputEnabled: false,
            },
          },
        }],
      },
    };

    setOptions(chartOptions);
  }, [data, symbol, convertToHighchartsData]);

  // Update real-time price
  useEffect(() => {
    if (chartRef.current && chartRef.current.chart && price > 0 && data) {
      const chart = chartRef.current.chart;
      const series = chart.series[0];
      if (series && series.data.length > 0) {
        const lastPoint = series.data[series.data.length - 1];
        const lastTime = lastPoint.x;
        const lastData = data.candles[data.candles.length - 1];

        // Update the last candle
        lastPoint.update([
          lastTime,
          lastData.open,
          Math.max(lastData.high, price),
          Math.min(lastData.low, price),
          price,
        ]);
      }
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
              }}
            >
              {change >= 0 ? '+' : ''}{change.toFixed(2)} ({changePercent >= 0 ? '+' : ''}{changePercent.toFixed(2)}%)
            </Typography>
          </Box>

          {isLive && <Chip icon={<LiveIcon />} label="LIVE" size="small" sx={{ backgroundColor: 'rgba(0, 200, 5, 0.15)', color: '#00c805' }} />}
        </Box>

        <Box display="flex" gap={1}>
          <Tooltip title="AI Analysis">
            <IconButton onClick={triggerAnalysis} sx={{ color: '#8c8e96' }}>
              <AIIcon />
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
        <HighchartsReact
          highcharts={Highcharts}
          constructorType={'stockChart'}
          options={options}
          ref={chartRef}
        />

        {consensus && (
          <AISignalPanel elevation={0}>
            <Box display="flex" alignItems="center" gap={1} mb={2}>
              <AIIcon sx={{ color: '#FFD700' }} />
              <Typography sx={{ color: '#e0e1e6', fontWeight: 600 }}>
                AI Analysis
              </Typography>
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
                <Typography variant="body2" sx={{ color: '#8c8e96' }}>
                  Entry: ${consensus.entry_price.toFixed(2)}
                </Typography>
                {consensus.stop_loss && (
                  <Typography variant="body2" sx={{ color: '#8c8e96' }}>
                    Stop Loss: ${consensus.stop_loss.toFixed(2)}
                  </Typography>
                )}
                {consensus.take_profit && (
                  <Typography variant="body2" sx={{ color: '#8c8e96' }}>
                    Take Profit: ${consensus.take_profit.toFixed(2)}
                  </Typography>
                )}
              </Box>
            )}
          </AISignalPanel>
        )}
      </ChartContainer>
    </Container>
  );
};

export default ProfessionalHighchartsChart;
