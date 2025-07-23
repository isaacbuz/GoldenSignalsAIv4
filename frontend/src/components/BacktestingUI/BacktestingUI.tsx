import React, { useState, useCallback, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Slider,
  Switch,
  FormControlLabel,
  Chip,
  Alert,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tabs,
  Tab,
  useTheme,
  alpha,
} from '@mui/material';
import {
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon,
  Save as SaveIcon,
  History as HistoryIcon,
  Settings as SettingsIcon,
  ExpandMore as ExpandMoreIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  ShowChart as ShowChartIcon,
  Assessment as AssessmentIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { Line, Bar } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import { styled } from '@mui/material/styles';
import { motion, AnimatePresence } from 'framer-motion';

// Styled components
const BacktestContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(3),
  backgroundColor: theme.palette.background.default,
  minHeight: '100vh',
}));

const StyledCard = styled(Card)(({ theme }) => ({
  background: theme.palette.background.paper,
  border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
  transition: 'all 0.3s ease',
  '&:hover': {
    borderColor: theme.palette.primary.main,
    boxShadow: theme.shadows[4],
  },
}));

const MetricBox = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  borderRadius: theme.shape.borderRadius,
  backgroundColor: alpha(theme.palette.primary.main, 0.05),
  border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
  textAlign: 'center',
}));

// Types
interface BacktestConfig {
  symbols: string[];
  startDate: Date;
  endDate: Date;
  initialCapital: number;
  strategy: string;
  stopLoss: number;
  takeProfit: number;
  positionSizing: 'fixed' | 'kelly' | 'risk-based';
  maxPositions: number;
  commission: number;
  slippage: number;
  agents: string[];
}

interface BacktestResult {
  totalReturn: number;
  annualizedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  profitableTrades: number;
  averageWin: number;
  averageLoss: number;
  profitFactor: number;
  expectancy: number;
  equity: number[];
  dates: string[];
  trades: TradeResult[];
}

interface TradeResult {
  id: string;
  symbol: string;
  entryDate: string;
  exitDate: string;
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  pnl: number;
  pnlPercent: number;
  duration: number;
  signal: string;
}

const BacktestingUI: React.FC = () => {
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [config, setConfig] = useState<BacktestConfig>({
    symbols: ['AAPL', 'GOOGL', 'MSFT'],
    startDate: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000), // 1 year ago
    endDate: new Date(),
    initialCapital: 100000,
    strategy: 'multi-agent',
    stopLoss: 2,
    takeProfit: 5,
    positionSizing: 'kelly',
    maxPositions: 5,
    commission: 0.1,
    slippage: 0.05,
    agents: ['RSI', 'MACD', 'Sentiment', 'ML'],
  });
  const [result, setResult] = useState<BacktestResult | null>(null);

  // Mock backtest execution
  const runBacktest = useCallback(async () => {
    setIsRunning(true);
    setProgress(0);

    // Simulate backtest progress
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(progressInterval);
          return 100;
        }
        return prev + 10;
      });
    }, 500);

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 5000));

    // Generate mock results
    const mockResult: BacktestResult = {
      totalReturn: 23.45,
      annualizedReturn: 28.12,
      sharpeRatio: 1.85,
      maxDrawdown: -12.34,
      winRate: 0.67,
      totalTrades: 145,
      profitableTrades: 97,
      averageWin: 234.56,
      averageLoss: -123.45,
      profitFactor: 2.34,
      expectancy: 89.12,
      equity: Array.from({ length: 252 }, (_, i) =>
        100000 * (1 + 0.0015 * i + Math.random() * 0.02 - 0.01)
      ),
      dates: Array.from({ length: 252 }, (_, i) => {
        const date = new Date(config.startDate);
        date.setDate(date.getDate() + i);
        return date.toISOString();
      }),
      trades: generateMockTrades(),
    };

    setResult(mockResult);
    setIsRunning(false);
    setProgress(100);
  }, [config]);

  const generateMockTrades = (): TradeResult[] => {
    return Array.from({ length: 20 }, (_, i) => ({
      id: `trade_${i}`,
      symbol: config.symbols[Math.floor(Math.random() * config.symbols.length)],
      entryDate: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
      exitDate: new Date(Date.now() - Math.random() * 10 * 24 * 60 * 60 * 1000).toISOString(),
      entryPrice: 100 + Math.random() * 100,
      exitPrice: 100 + Math.random() * 100,
      quantity: Math.floor(10 + Math.random() * 90),
      pnl: -500 + Math.random() * 1500,
      pnlPercent: -5 + Math.random() * 15,
      duration: Math.floor(1 + Math.random() * 20),
      signal: ['BUY', 'SELL'][Math.floor(Math.random() * 2)],
    }));
  };

  const handleConfigChange = (field: keyof BacktestConfig, value: any) => {
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  const equityChartData = useMemo(() => {
    if (!result) return null;

    return {
      labels: result.dates.map(d => new Date(d).toLocaleDateString()),
      datasets: [{
        label: 'Portfolio Value',
        data: result.equity,
        borderColor: theme.palette.primary.main,
        backgroundColor: alpha(theme.palette.primary.main, 0.1),
        fill: true,
        tension: 0.1,
      }],
    };
  }, [result, theme]);

  const drawdownChartData = useMemo(() => {
    if (!result) return null;

    const drawdowns = result.equity.map((value, i) => {
      const peak = Math.max(...result.equity.slice(0, i + 1));
      return ((value - peak) / peak) * 100;
    });

    return {
      labels: result.dates.map(d => new Date(d).toLocaleDateString()),
      datasets: [{
        label: 'Drawdown %',
        data: drawdowns,
        backgroundColor: alpha(theme.palette.error.main, 0.6),
        borderColor: theme.palette.error.main,
        fill: true,
      }],
    };
  }, [result, theme]);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <BacktestContainer>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Strategy Backtesting
        </Typography>

        <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ mb: 3 }}>
          <Tab label="Configuration" icon={<SettingsIcon />} iconPosition="start" />
          <Tab label="Results" icon={<AssessmentIcon />} iconPosition="start" disabled={!result} />
          <Tab label="Trades" icon={<HistoryIcon />} iconPosition="start" disabled={!result} />
        </Tabs>

        <AnimatePresence mode="wait">
          {activeTab === 0 && (
            <motion.div
              key="config"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              <Grid container spacing={3}>
                {/* Configuration Form */}
                <Grid item xs={12} md={8}>
                  <StyledCard>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Backtest Configuration
                      </Typography>

                      <Grid container spacing={2}>
                        <Grid item xs={12}>
                          <TextField
                            fullWidth
                            label="Symbols (comma separated)"
                            value={config.symbols.join(', ')}
                            onChange={(e) => handleConfigChange('symbols', e.target.value.split(',').map(s => s.trim()))}
                          />
                        </Grid>

                        <Grid item xs={12} sm={6}>
                          <DatePicker
                            label="Start Date"
                            value={config.startDate}
                            onChange={(date) => date && handleConfigChange('startDate', date)}
                            renderInput={(params) => <TextField {...params} fullWidth />}
                          />
                        </Grid>

                        <Grid item xs={12} sm={6}>
                          <DatePicker
                            label="End Date"
                            value={config.endDate}
                            onChange={(date) => date && handleConfigChange('endDate', date)}
                            renderInput={(params) => <TextField {...params} fullWidth />}
                          />
                        </Grid>

                        <Grid item xs={12} sm={6}>
                          <TextField
                            fullWidth
                            type="number"
                            label="Initial Capital"
                            value={config.initialCapital}
                            onChange={(e) => handleConfigChange('initialCapital', Number(e.target.value))}
                            InputProps={{ startAdornment: '$' }}
                          />
                        </Grid>

                        <Grid item xs={12} sm={6}>
                          <FormControl fullWidth>
                            <InputLabel>Position Sizing</InputLabel>
                            <Select
                              value={config.positionSizing}
                              label="Position Sizing"
                              onChange={(e) => handleConfigChange('positionSizing', e.target.value)}
                            >
                              <MenuItem value="fixed">Fixed Amount</MenuItem>
                              <MenuItem value="kelly">Kelly Criterion</MenuItem>
                              <MenuItem value="risk-based">Risk-Based</MenuItem>
                            </Select>
                          </FormControl>
                        </Grid>

                        <Grid item xs={12}>
                          <Typography gutterBottom>Stop Loss: {config.stopLoss}%</Typography>
                          <Slider
                            value={config.stopLoss}
                            onChange={(_, v) => handleConfigChange('stopLoss', v)}
                            min={0.5}
                            max={10}
                            step={0.5}
                            marks
                            valueLabelDisplay="auto"
                          />
                        </Grid>

                        <Grid item xs={12}>
                          <Typography gutterBottom>Take Profit: {config.takeProfit}%</Typography>
                          <Slider
                            value={config.takeProfit}
                            onChange={(_, v) => handleConfigChange('takeProfit', v)}
                            min={1}
                            max={20}
                            step={1}
                            marks
                            valueLabelDisplay="auto"
                          />
                        </Grid>

                        <Grid item xs={12}>
                          <Typography variant="subtitle2" gutterBottom>
                            Select Agents
                          </Typography>
                          <Box display="flex" gap={1} flexWrap="wrap">
                            {['RSI', 'MACD', 'Sentiment', 'Volume', 'ML', 'Options'].map(agent => (
                              <Chip
                                key={agent}
                                label={agent}
                                color={config.agents.includes(agent) ? 'primary' : 'default'}
                                onClick={() => {
                                  const newAgents = config.agents.includes(agent)
                                    ? config.agents.filter(a => a !== agent)
                                    : [...config.agents, agent];
                                  handleConfigChange('agents', newAgents);
                                }}
                              />
                            ))}
                          </Box>
                        </Grid>
                      </Grid>

                      <Box mt={3} display="flex" gap={2}>
                        <Button
                          variant="contained"
                          color="primary"
                          startIcon={<PlayArrowIcon />}
                          onClick={runBacktest}
                          disabled={isRunning}
                          fullWidth
                        >
                          {isRunning ? 'Running Backtest...' : 'Run Backtest'}
                        </Button>
                        <Button
                          variant="outlined"
                          startIcon={<SaveIcon />}
                          disabled={isRunning}
                        >
                          Save Config
                        </Button>
                      </Box>

                      {isRunning && (
                        <Box mt={2}>
                          <LinearProgress variant="determinate" value={progress} />
                          <Typography variant="caption" color="text.secondary" align="center" display="block" mt={1}>
                            Processing... {progress}%
                          </Typography>
                        </Box>
                      )}
                    </CardContent>
                  </StyledCard>
                </Grid>

                {/* Quick Stats */}
                <Grid item xs={12} md={4}>
                  <StyledCard>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Configuration Summary
                      </Typography>
                      <Box display="flex" flexDirection="column" gap={2}>
                        <MetricBox>
                          <Typography variant="caption" color="text.secondary">
                            Trading Period
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {Math.floor((config.endDate.getTime() - config.startDate.getTime()) / (1000 * 60 * 60 * 24))} days
                          </Typography>
                        </MetricBox>
                        <MetricBox>
                          <Typography variant="caption" color="text.secondary">
                            Symbols
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {config.symbols.length} selected
                          </Typography>
                        </MetricBox>
                        <MetricBox>
                          <Typography variant="caption" color="text.secondary">
                            Active Agents
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {config.agents.length} agents
                          </Typography>
                        </MetricBox>
                        <MetricBox>
                          <Typography variant="caption" color="text.secondary">
                            Risk Per Trade
                          </Typography>
                          <Typography variant="body2" fontWeight="bold">
                            {((config.stopLoss / 100) * config.initialCapital / config.maxPositions).toFixed(0)}
                          </Typography>
                        </MetricBox>
                      </Box>
                    </CardContent>
                  </StyledCard>
                </Grid>
              </Grid>
            </motion.div>
          )}

          {activeTab === 1 && result && (
            <motion.div
              key="results"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              <Grid container spacing={3}>
                {/* Performance Metrics */}
                <Grid item xs={12}>
                  <Grid container spacing={2}>
                    <Grid item xs={6} sm={4} md={2}>
                      <MetricBox>
                        <Typography variant="caption" color="text.secondary">
                          Total Return
                        </Typography>
                        <Typography
                          variant="h6"
                          fontWeight="bold"
                          color={result.totalReturn >= 0 ? 'success.main' : 'error.main'}
                        >
                          {result.totalReturn.toFixed(2)}%
                        </Typography>
                      </MetricBox>
                    </Grid>
                    <Grid item xs={6} sm={4} md={2}>
                      <MetricBox>
                        <Typography variant="caption" color="text.secondary">
                          Sharpe Ratio
                        </Typography>
                        <Typography variant="h6" fontWeight="bold">
                          {result.sharpeRatio.toFixed(2)}
                        </Typography>
                      </MetricBox>
                    </Grid>
                    <Grid item xs={6} sm={4} md={2}>
                      <MetricBox>
                        <Typography variant="caption" color="text.secondary">
                          Max Drawdown
                        </Typography>
                        <Typography variant="h6" fontWeight="bold" color="error.main">
                          {result.maxDrawdown.toFixed(2)}%
                        </Typography>
                      </MetricBox>
                    </Grid>
                    <Grid item xs={6} sm={4} md={2}>
                      <MetricBox>
                        <Typography variant="caption" color="text.secondary">
                          Win Rate
                        </Typography>
                        <Typography variant="h6" fontWeight="bold">
                          {(result.winRate * 100).toFixed(1)}%
                        </Typography>
                      </MetricBox>
                    </Grid>
                    <Grid item xs={6} sm={4} md={2}>
                      <MetricBox>
                        <Typography variant="caption" color="text.secondary">
                          Total Trades
                        </Typography>
                        <Typography variant="h6" fontWeight="bold">
                          {result.totalTrades}
                        </Typography>
                      </MetricBox>
                    </Grid>
                    <Grid item xs={6} sm={4} md={2}>
                      <MetricBox>
                        <Typography variant="caption" color="text.secondary">
                          Profit Factor
                        </Typography>
                        <Typography variant="h6" fontWeight="bold">
                          {result.profitFactor.toFixed(2)}
                        </Typography>
                      </MetricBox>
                    </Grid>
                  </Grid>
                </Grid>

                {/* Equity Curve */}
                <Grid item xs={12} md={8}>
                  <StyledCard>
                    <CardContent>
                      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                        <Typography variant="h6">Equity Curve</Typography>
                        <IconButton size="small">
                          <DownloadIcon />
                        </IconButton>
                      </Box>
                      <Box height={300}>
                        {equityChartData && (
                          <Line
                            data={equityChartData}
                            options={{
                              responsive: true,
                              maintainAspectRatio: false,
                              plugins: {
                                legend: { display: false },
                              },
                              scales: {
                                y: {
                                  ticks: {
                                    callback: (value) => formatCurrency(Number(value)),
                                  },
                                },
                              },
                            }}
                          />
                        )}
                      </Box>
                    </CardContent>
                  </StyledCard>
                </Grid>

                {/* Drawdown Chart */}
                <Grid item xs={12} md={4}>
                  <StyledCard>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Drawdown Analysis
                      </Typography>
                      <Box height={300}>
                        {drawdownChartData && (
                          <Bar
                            data={drawdownChartData}
                            options={{
                              responsive: true,
                              maintainAspectRatio: false,
                              plugins: {
                                legend: { display: false },
                              },
                              scales: {
                                y: {
                                  max: 0,
                                  ticks: {
                                    callback: (value) => `${value}%`,
                                  },
                                },
                              },
                            }}
                          />
                        )}
                      </Box>
                    </CardContent>
                  </StyledCard>
                </Grid>

                {/* Additional Metrics */}
                <Grid item xs={12}>
                  <StyledCard>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Detailed Statistics
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid item xs={12} sm={6} md={3}>
                          <Typography variant="subtitle2" color="text.secondary">
                            Average Win
                          </Typography>
                          <Typography variant="h6" color="success.main">
                            {formatCurrency(result.averageWin)}
                          </Typography>
                        </Grid>
                        <Grid item xs={12} sm={6} md={3}>
                          <Typography variant="subtitle2" color="text.secondary">
                            Average Loss
                          </Typography>
                          <Typography variant="h6" color="error.main">
                            {formatCurrency(result.averageLoss)}
                          </Typography>
                        </Grid>
                        <Grid item xs={12} sm={6} md={3}>
                          <Typography variant="subtitle2" color="text.secondary">
                            Expectancy
                          </Typography>
                          <Typography variant="h6">
                            {formatCurrency(result.expectancy)}
                          </Typography>
                        </Grid>
                        <Grid item xs={12} sm={6} md={3}>
                          <Typography variant="subtitle2" color="text.secondary">
                            Annualized Return
                          </Typography>
                          <Typography variant="h6">
                            {result.annualizedReturn.toFixed(2)}%
                          </Typography>
                        </Grid>
                      </Grid>
                    </CardContent>
                  </StyledCard>
                </Grid>
              </Grid>
            </motion.div>
          )}

          {activeTab === 2 && result && (
            <motion.div
              key="trades"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              <StyledCard>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Typography variant="h6">Trade History</Typography>
                    <Button startIcon={<DownloadIcon />} size="small">
                      Export CSV
                    </Button>
                  </Box>
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Symbol</TableCell>
                          <TableCell>Signal</TableCell>
                          <TableCell>Entry Date</TableCell>
                          <TableCell>Exit Date</TableCell>
                          <TableCell align="right">Entry Price</TableCell>
                          <TableCell align="right">Exit Price</TableCell>
                          <TableCell align="right">Quantity</TableCell>
                          <TableCell align="right">P&L</TableCell>
                          <TableCell align="right">P&L %</TableCell>
                          <TableCell align="right">Duration</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {result.trades.map((trade) => (
                          <TableRow key={trade.id} hover>
                            <TableCell>
                              <Chip label={trade.symbol} size="small" />
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={trade.signal}
                                size="small"
                                color={trade.signal === 'BUY' ? 'success' : 'error'}
                              />
                            </TableCell>
                            <TableCell>{new Date(trade.entryDate).toLocaleDateString()}</TableCell>
                            <TableCell>{new Date(trade.exitDate).toLocaleDateString()}</TableCell>
                            <TableCell align="right">${trade.entryPrice.toFixed(2)}</TableCell>
                            <TableCell align="right">${trade.exitPrice.toFixed(2)}</TableCell>
                            <TableCell align="right">{trade.quantity}</TableCell>
                            <TableCell
                              align="right"
                              sx={{
                                color: trade.pnl >= 0 ? 'success.main' : 'error.main',
                                fontWeight: 'medium',
                              }}
                            >
                              {formatCurrency(trade.pnl)}
                            </TableCell>
                            <TableCell
                              align="right"
                              sx={{
                                color: trade.pnlPercent >= 0 ? 'success.main' : 'error.main',
                              }}
                            >
                              {trade.pnlPercent.toFixed(2)}%
                            </TableCell>
                            <TableCell align="right">{trade.duration}d</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </StyledCard>
            </motion.div>
          )}
        </AnimatePresence>
      </BacktestContainer>
    </LocalizationProvider>
  );
};

export default BacktestingUI;
