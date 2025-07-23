/**
 * SimpleHighchartsChart - Simplified Highcharts implementation
 *
 * A clean implementation without complex module dependencies
 */

import React, { useEffect, useRef, useState } from 'react';
import Highcharts from 'highcharts';
import HighchartsReact from 'highcharts-react-official';

import {
  Box,
  Typography,
  TextField,
  Autocomplete,
  Chip,
  Paper,
  Button,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Search as SearchIcon,
  Circle as LiveIcon,
  Psychology as AIIcon,
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

// Constants
const TIMEFRAMES = ['1D', '1W', '1M', '3M', '1Y', 'ALL'];

const POPULAR_SYMBOLS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
  'V', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'MA', 'HD', 'BAC', 'ADBE',
];

// Dark theme for Highcharts
Highcharts.theme = {
  colors: ['#2196f3', '#00c805', '#ff3b30', '#FFD700', '#ff9800', '#9c27b0'],
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
  xAxis: {
    gridLineColor: '#1f2128',
    labels: {
      style: {
        color: '#8c8e96',
      },
    },
    lineColor: '#2a2d37',
    tickColor: '#2a2d37',
  },
  yAxis: {
    gridLineColor: '#1f2128',
    labels: {
      style: {
        color: '#8c8e96',
      },
    },
    lineColor: '#2a2d37',
    tickColor: '#2a2d37',
  },
  tooltip: {
    backgroundColor: '#16181d',
    borderColor: '#2a2d37',
    style: {
      color: '#e0e1e6',
    },
  },
  plotOptions: {
    candlestick: {
      lineColor: '#2a2d37',
      upLineColor: '#00c805',
      upColor: '#00c805',
      color: '#ff3b30',
    },
  },
  credits: {
    enabled: false,
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
        },
        select: {
          fill: '#2196f3',
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
  },
};

// Apply theme
Highcharts.setOptions(Highcharts.theme);

interface SimpleHighchartsChartProps {
  symbol?: string;
  onSymbolChange?: (symbol: string) => void;
}

export const SimpleHighchartsChart: React.FC<SimpleHighchartsChartProps> = ({
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
  const convertToHighchartsData = () => {
    if (!data || !data.candles) return { ohlc: [], volume: [] };

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

    return { ohlc, volume };
  };

  // Initialize chart
  useEffect(() => {
    const { ohlc, volume } = convertToHighchartsData();

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

      yAxis: [{
        labels: {
          align: 'right',
          x: -3,
        },
        title: {
          text: 'Price',
        },
        height: '70%',
        lineWidth: 2,
      }, {
        labels: {
          align: 'right',
          x: -3,
        },
        title: {
          text: 'Volume',
        },
        top: '70%',
        height: '30%',
        offset: 0,
        lineWidth: 2,
      }],

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
        color: 'rgba(74, 77, 87, 0.5)',
      }],
    };

    // Add AI prediction line if we have consensus
    if (consensus && consensus.entry_price && ohlc.length > 0) {
      // Create a simple prediction line
      const lastTime = ohlc[ohlc.length - 1][0];
      const predictionData = [
        [lastTime, consensus.entry_price],
        [lastTime + 3600000, consensus.take_profit || consensus.entry_price * 1.02], // 1 hour ahead
      ];

      chartOptions.series?.push({
        type: 'line',
        name: 'AI Prediction',
        data: predictionData,
        yAxis: 0,
        color: '#FFD700',
        lineWidth: 3,
        dashStyle: 'Dash',
        marker: {
          enabled: false,
        },
      });
    }

    setOptions(chartOptions);
  }, [data, symbol, consensus]);

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

          <Button
            variant="contained"
            startIcon={<AIIcon />}
            onClick={triggerAnalysis}
            disabled={isAnalyzing}
            sx={{
              ml: 2,
              backgroundColor: '#FFD700',
              color: '#000',
              '&:hover': {
                backgroundColor: '#FFC700',
              },
            }}
          >
            Analyze
          </Button>
        </Box>
      </Header>

      <ChartContainer>
        {!loading && !error && (
          <HighchartsReact
            highcharts={Highcharts}
            constructorType={'stockChart'}
            options={options}
            ref={chartRef}
          />
        )}

        {loading && (
          <Box display="flex" justifyContent="center" alignItems="center" height="100%">
            <Typography sx={{ color: '#8c8e96' }}>Loading market data...</Typography>
          </Box>
        )}

        {error && (
          <Box display="flex" justifyContent="center" alignItems="center" height="100%">
            <Typography sx={{ color: '#ff3b30' }}>{error}</Typography>
          </Box>
        )}

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

export default SimpleHighchartsChart;
