/**
 * Backtesting Interface Component
 * 
 * Provides a comprehensive interface for running and analyzing backtests
 * using the backend's advanced backtesting capabilities.
 * 
 * Features:
 * - Agent-specific backtesting
 * - Strategy comparison
 * - Performance metrics visualization
 * - Historical signal analysis
 * - Risk assessment
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
    Box,
    Card,
    CardContent,
    Typography,
    Grid,
    Stack,
    Button,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    TextField,
    Chip,
    LinearProgress,
    Alert,
    Tabs,
    Tab,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    IconButton,
    Tooltip,
    useTheme,
    alpha,
    Divider,
} from '@mui/material';
import {
    PlayArrow,
    Stop,
    Download,
    Compare,
    TrendingUp,
    TrendingDown,
    Assessment,
    Timeline,
    ShowChart,
    Speed,
    Warning,
    CheckCircle,
    Info,
} from '@mui/icons-material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Line, Bar } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip as ChartTooltip,
    Legend,
} from 'chart.js';

import { apiClient, BacktestRequest, BacktestResult } from '../../services/api/apiClient';

// Register Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    ChartTooltip,
    Legend
);

interface BacktestingInterfaceProps {
    defaultSymbol?: string;
    defaultAgent?: string;
}

interface BacktestConfig {
    agent: string;
    symbol: string;
    startDate: Date;
    endDate: Date;
    initialCapital: number;
    strategy: string;
}

interface ComparisonResult {
    config: BacktestConfig;
    result: BacktestResult;
    id: string;
}

const AVAILABLE_AGENTS = [
    'all',
    'rsi',
    'macd',
    'volume_spike',
    'ma_crossover',
    'bollinger',
    'stochastic',
    'ema',
    'atr',
    'vwap',
    'ichimoku',
    'fibonacci',
    'adx',
    'parabolic_sar',
    'std_dev',
    'volume_profile',
    'market_profile',
    'order_flow',
    'sentiment',
    'options_flow'
];

const AVAILABLE_STRATEGIES = [
    'ALL',
    'pairs_trading',
    'momentum',
    'volatility_breakout',
    'machine_learning',
    'mean_reversion',
    'trend_following'
];

const POPULAR_SYMBOLS = [
    'SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'
];

export const BacktestingInterface: React.FC<BacktestingInterfaceProps> = ({
    defaultSymbol = 'SPY',
    defaultAgent = 'all'
}) => {
    const theme = useTheme();
    const queryClient = useQueryClient();
    const [activeTab, setActiveTab] = useState(0);
    const [isRunning, setIsRunning] = useState(false);
    const [comparisons, setComparisons] = useState<ComparisonResult[]>([]);

    // Backtest configuration
    const [config, setConfig] = useState<BacktestConfig>({
        agent: defaultAgent,
        symbol: defaultSymbol,
        startDate: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000), // 1 year ago
        endDate: new Date(),
        initialCapital: 10000,
        strategy: 'ALL'
    });

    // Current backtest result
    const [currentResult, setCurrentResult] = useState<BacktestResult | null>(null);

    // Fetch backtest history
    const { data: backtestHistory, isLoading: historyLoading } = useQuery({
        queryKey: ['backtest-history'],
        queryFn: () => apiClient.getBacktestHistory(),
        staleTime: 60000, // 1 minute
    });

    // Run backtest mutation
    const runBacktestMutation = useMutation({
        mutationFn: (request: BacktestRequest) => apiClient.runBacktest(request),
        onMutate: () => {
            setIsRunning(true);
        },
        onSuccess: (result) => {
            setCurrentResult(result);
            setIsRunning(false);
            queryClient.invalidateQueries({ queryKey: ['backtest-history'] });
        },
        onError: (error) => {
            console.error('Backtest failed:', error);
            setIsRunning(false);
        }
    });

    const handleRunBacktest = () => {
        const request: BacktestRequest = {
            agent: config.agent,
            symbol: config.symbol,
            start_date: config.startDate.toISOString().split('T')[0],
            end_date: config.endDate.toISOString().split('T')[0],
            initial_capital: config.initialCapital,
            strategy: config.strategy
        };

        runBacktestMutation.mutate(request);
    };

    const handleAddToComparison = () => {
        if (currentResult) {
            const comparisonResult: ComparisonResult = {
                config: { ...config },
                result: currentResult,
                id: Date.now().toString()
            };
            setComparisons(prev => [...prev, comparisonResult]);
        }
    };

    const handleRemoveComparison = (id: string) => {
        setComparisons(prev => prev.filter(c => c.id !== id));
    };

    // Performance metrics component
    const MetricCard: React.FC<{
        title: string;
        value: string | number;
        change?: number;
        icon: React.ReactNode;
        color?: string;
        format?: 'percentage' | 'currency' | 'number';
    }> = ({ title, value, change, icon, color = theme.palette.primary.main, format = 'number' }) => {
        const formatValue = (val: string | number) => {
            if (typeof val === 'number') {
                switch (format) {
                    case 'percentage':
                        return `${(val * 100).toFixed(2)}%`;
                    case 'currency':
                        return `$${val.toLocaleString()}`;
                    default:
                        return val.toLocaleString();
                }
            }
            return val;
        };

        return (
            <Card sx={{ height: '100%', background: alpha(color, 0.05) }}>
                <CardContent>
                    <Stack direction="row" alignItems="center" spacing={2}>
                        <Box sx={{ color, fontSize: 24 }}>
                            {icon}
                        </Box>
                        <Box sx={{ flex: 1 }}>
                            <Typography variant="body2" color="text.secondary">
                                {title}
                            </Typography>
                            <Typography variant="h6" fontWeight="bold">
                                {formatValue(value)}
                            </Typography>
                            {change !== undefined && (
                                <Stack direction="row" alignItems="center" spacing={0.5}>
                                    {change > 0 ? (
                                        <TrendingUp sx={{ fontSize: 16, color: 'success.main' }} />
                                    ) : (
                                        <TrendingDown sx={{ fontSize: 16, color: 'error.main' }} />
                                    )}
                                    <Typography
                                        variant="caption"
                                        color={change > 0 ? 'success.main' : 'error.main'}
                                    >
                                        {Math.abs(change).toFixed(2)}%
                                    </Typography>
                                </Stack>
                            )}
                        </Box>
                    </Stack>
                </CardContent>
            </Card>
        );
    };

    // Configuration panel
    const ConfigurationPanel = () => (
        <Card>
            <CardContent>
                <Typography variant="h6" gutterBottom>
                    Backtest Configuration
                </Typography>
                <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                        <FormControl fullWidth>
                            <InputLabel>Agent</InputLabel>
                            <Select
                                value={config.agent}
                                onChange={(e) => setConfig(prev => ({ ...prev, agent: e.target.value }))}
                                label="Agent"
                            >
                                {AVAILABLE_AGENTS.map(agent => (
                                    <MenuItem key={agent} value={agent}>
                                        {agent.replace('_', ' ').toUpperCase()}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Grid>
                    <Grid item xs={12} md={6}>
                        <FormControl fullWidth>
                            <InputLabel>Symbol</InputLabel>
                            <Select
                                value={config.symbol}
                                onChange={(e) => setConfig(prev => ({ ...prev, symbol: e.target.value }))}
                                label="Symbol"
                            >
                                {POPULAR_SYMBOLS.map(symbol => (
                                    <MenuItem key={symbol} value={symbol}>
                                        {symbol}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Grid>
                    <Grid item xs={12} md={6}>
                        <LocalizationProvider dateAdapter={AdapterDateFns}>
                            <DatePicker
                                label="Start Date"
                                value={config.startDate}
                                onChange={(date) => date && setConfig(prev => ({ ...prev, startDate: date }))}
                                slotProps={{ textField: { fullWidth: true } }}
                            />
                        </LocalizationProvider>
                    </Grid>
                    <Grid item xs={12} md={6}>
                        <LocalizationProvider dateAdapter={AdapterDateFns}>
                            <DatePicker
                                label="End Date"
                                value={config.endDate}
                                onChange={(date) => date && setConfig(prev => ({ ...prev, endDate: date }))}
                                slotProps={{ textField: { fullWidth: true } }}
                            />
                        </LocalizationProvider>
                    </Grid>
                    <Grid item xs={12} md={6}>
                        <TextField
                            fullWidth
                            label="Initial Capital"
                            type="number"
                            value={config.initialCapital}
                            onChange={(e) => setConfig(prev => ({ ...prev, initialCapital: Number(e.target.value) }))}
                            InputProps={{
                                startAdornment: '$'
                            }}
                        />
                    </Grid>
                    <Grid item xs={12} md={6}>
                        <FormControl fullWidth>
                            <InputLabel>Strategy</InputLabel>
                            <Select
                                value={config.strategy}
                                onChange={(e) => setConfig(prev => ({ ...prev, strategy: e.target.value }))}
                                label="Strategy"
                            >
                                {AVAILABLE_STRATEGIES.map(strategy => (
                                    <MenuItem key={strategy} value={strategy}>
                                        {strategy.replace('_', ' ').toUpperCase()}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Grid>
                </Grid>
                <Stack direction="row" spacing={2} sx={{ mt: 3 }}>
                    <Button
                        variant="contained"
                        startIcon={<PlayArrow />}
                        onClick={handleRunBacktest}
                        disabled={isRunning}
                        sx={{ minWidth: 120 }}
                    >
                        {isRunning ? 'Running...' : 'Run Backtest'}
                    </Button>
                    {currentResult && (
                        <Button
                            variant="outlined"
                            startIcon={<Compare />}
                            onClick={handleAddToComparison}
                        >
                            Add to Comparison
                        </Button>
                    )}
                </Stack>
                {isRunning && (
                    <Box sx={{ mt: 2 }}>
                        <LinearProgress />
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                            Running backtest for {config.agent} on {config.symbol}...
                        </Typography>
                    </Box>
                )}
            </CardContent>
        </Card>
    );

    // Results panel
    const ResultsPanel = () => {
        if (!currentResult) {
            return (
                <Card>
                    <CardContent>
                        <Alert severity="info">
                            Configure and run a backtest to see results here.
                        </Alert>
                    </CardContent>
                </Card>
            );
        }

        const { results, comparison } = currentResult;

        return (
            <Stack spacing={3}>
                {/* Performance Metrics */}
                <Grid container spacing={3}>
                    <Grid item xs={12} sm={6} md={3}>
                        <MetricCard
                            title="Total Return"
                            value={results.total_return}
                            icon={<TrendingUp />}
                            color={theme.palette.success.main}
                            format="percentage"
                        />
                    </Grid>
                    <Grid item xs={12} sm={6} md={3}>
                        <MetricCard
                            title="Sharpe Ratio"
                            value={results.sharpe_ratio}
                            icon={<Assessment />}
                            color={theme.palette.primary.main}
                        />
                    </Grid>
                    <Grid item xs={12} sm={6} md={3}>
                        <MetricCard
                            title="Max Drawdown"
                            value={results.max_drawdown}
                            icon={<TrendingDown />}
                            color={theme.palette.error.main}
                            format="percentage"
                        />
                    </Grid>
                    <Grid item xs={12} sm={6} md={3}>
                        <MetricCard
                            title="Win Rate"
                            value={results.win_rate}
                            icon={<Speed />}
                            color={theme.palette.info.main}
                            format="percentage"
                        />
                    </Grid>
                </Grid>

                {/* Detailed Results */}
                <Card>
                    <CardContent>
                        <Typography variant="h6" gutterBottom>
                            Detailed Results
                        </Typography>
                        <Grid container spacing={3}>
                            <Grid item xs={12} md={6}>
                                <TableContainer component={Paper} variant="outlined">
                                    <Table size="small">
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Metric</TableCell>
                                                <TableCell align="right">Value</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            <TableRow>
                                                <TableCell>Total Trades</TableCell>
                                                <TableCell align="right">{results.total_trades}</TableCell>
                                            </TableRow>
                                            <TableRow>
                                                <TableCell>Profit Factor</TableCell>
                                                <TableCell align="right">{results.profit_factor.toFixed(2)}</TableCell>
                                            </TableRow>
                                            <TableRow>
                                                <TableCell>Annualized Return</TableCell>
                                                <TableCell align="right">{(results.annualized_return * 100).toFixed(2)}%</TableCell>
                                            </TableRow>
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                            </Grid>
                            <Grid item xs={12} md={6}>
                                <TableContainer component={Paper} variant="outlined">
                                    <Table size="small">
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Benchmark Comparison</TableCell>
                                                <TableCell align="right">Value</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            <TableRow>
                                                <TableCell>Buy & Hold Return</TableCell>
                                                <TableCell align="right">{(comparison.buy_hold_return * 100).toFixed(2)}%</TableCell>
                                            </TableRow>
                                            <TableRow>
                                                <TableCell>Outperformance</TableCell>
                                                <TableCell align="right">
                                                    <Chip
                                                        label={`${(comparison.outperformance * 100).toFixed(2)}%`}
                                                        color={comparison.outperformance > 0 ? 'success' : 'error'}
                                                        size="small"
                                                    />
                                                </TableCell>
                                            </TableRow>
                                            <TableRow>
                                                <TableCell>Alpha</TableCell>
                                                <TableCell align="right">{(comparison.alpha * 100).toFixed(2)}%</TableCell>
                                            </TableRow>
                                            <TableRow>
                                                <TableCell>Beta</TableCell>
                                                <TableCell align="right">{comparison.beta.toFixed(2)}</TableCell>
                                            </TableRow>
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                            </Grid>
                        </Grid>
                    </CardContent>
                </Card>
            </Stack>
        );
    };

    // Comparison panel
    const ComparisonPanel = () => {
        if (comparisons.length === 0) {
            return (
                <Card>
                    <CardContent>
                        <Alert severity="info">
                            Run backtests and add them to comparison to analyze performance differences.
                        </Alert>
                    </CardContent>
                </Card>
            );
        }

        const chartData = {
            labels: ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
            datasets: comparisons.map((comp, index) => ({
                label: `${comp.config.agent} (${comp.config.symbol})`,
                data: [
                    comp.result.results.total_return * 100,
                    comp.result.results.sharpe_ratio,
                    Math.abs(comp.result.results.max_drawdown * 100),
                    comp.result.results.win_rate * 100
                ],
                backgroundColor: `hsl(${(index * 360) / comparisons.length}, 70%, 50%)`,
                borderColor: `hsl(${(index * 360) / comparisons.length}, 70%, 40%)`,
                borderWidth: 2,
            }))
        };

        return (
            <Stack spacing={3}>
                <Card>
                    <CardContent>
                        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2 }}>
                            <Typography variant="h6">
                                Performance Comparison
                            </Typography>
                            <Button
                                variant="outlined"
                                size="small"
                                onClick={() => setComparisons([])}
                            >
                                Clear All
                            </Button>
                        </Stack>
                        <Box sx={{ height: 400 }}>
                            <Bar
                                data={chartData}
                                options={{
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    plugins: {
                                        legend: {
                                            position: 'top' as const,
                                        },
                                        title: {
                                            display: true,
                                            text: 'Backtest Performance Comparison',
                                        },
                                    },
                                    scales: {
                                        y: {
                                            beginAtZero: true,
                                        },
                                    },
                                }}
                            />
                        </Box>
                    </CardContent>
                </Card>

                <Card>
                    <CardContent>
                        <Typography variant="h6" gutterBottom>
                            Comparison Table
                        </Typography>
                        <TableContainer component={Paper} variant="outlined">
                            <Table>
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Agent</TableCell>
                                        <TableCell>Symbol</TableCell>
                                        <TableCell align="right">Total Return</TableCell>
                                        <TableCell align="right">Sharpe Ratio</TableCell>
                                        <TableCell align="right">Max Drawdown</TableCell>
                                        <TableCell align="right">Win Rate</TableCell>
                                        <TableCell align="center">Actions</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {comparisons.map((comp) => (
                                        <TableRow key={comp.id}>
                                            <TableCell>{comp.config.agent.toUpperCase()}</TableCell>
                                            <TableCell>{comp.config.symbol}</TableCell>
                                            <TableCell align="right">
                                                <Chip
                                                    label={`${(comp.result.results.total_return * 100).toFixed(2)}%`}
                                                    color={comp.result.results.total_return > 0 ? 'success' : 'error'}
                                                    size="small"
                                                />
                                            </TableCell>
                                            <TableCell align="right">{comp.result.results.sharpe_ratio.toFixed(2)}</TableCell>
                                            <TableCell align="right">{(comp.result.results.max_drawdown * 100).toFixed(2)}%</TableCell>
                                            <TableCell align="right">{(comp.result.results.win_rate * 100).toFixed(2)}%</TableCell>
                                            <TableCell align="center">
                                                <IconButton
                                                    size="small"
                                                    onClick={() => handleRemoveComparison(comp.id)}
                                                >
                                                    <Stop />
                                                </IconButton>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </CardContent>
                </Card>
            </Stack>
        );
    };

    return (
        <Box sx={{ p: 3 }}>
            <Typography variant="h4" fontWeight="bold" gutterBottom>
                Backtesting Interface
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                Test agent performance and strategies against historical data
            </Typography>

            <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
                <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
                    <Tab label="Configuration" />
                    <Tab label="Results" />
                    <Tab label="Comparison" />
                </Tabs>
            </Box>

            {activeTab === 0 && (
                <Stack spacing={3}>
                    <ConfigurationPanel />
                </Stack>
            )}

            {activeTab === 1 && (
                <ResultsPanel />
            )}

            {activeTab === 2 && (
                <ComparisonPanel />
            )}
        </Box>
    );
}; 