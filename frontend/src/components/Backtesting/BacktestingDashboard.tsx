import React, { useState, useEffect } from 'react';
import {
    Box,
    Card,
    CardContent,
    Typography,
    Button,
    Grid,
    TextField,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    LinearProgress,
    Chip,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    Alert,
    Tabs,
    Tab,
    IconButton,
    Tooltip,
} from '@mui/material';
import {
    PlayArrow,
    Stop,
    Refresh,
    TrendingUp,
    TrendingDown,
    Assessment,
    Timeline,
    CheckCircle,
    Warning,
    Error as ErrorIcon,
} from '@mui/icons-material';
import { Line, Bar } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import axios from 'axios';
import logger from '../../services/logger';


interface BacktestRequest {
    symbols: string[];
    startDate?: string;
    endDate?: string;
    strategyType: string;
    useWalkForward: boolean;
    includeTransactionCosts: boolean;
}

interface BacktestResult {
    backtestId: string;
    status: string;
    progress: number;
    results?: any;
    error?: string;
}

interface SignalValidation {
    symbol: string;
    action: string;
    confidence: number;
    validation: {
        expectedAnnualReturn: number;
        currentVolatility: number;
        riskAdjustedScore: number;
        historicalWinRate: number;
        recommendation: string;
        confidenceLevel: string;
    };
}

const BacktestingDashboard: React.FC = () => {
    const [activeTab, setActiveTab] = useState(0);
    const [symbols, setSymbols] = useState<string[]>(['AAPL', 'GOOGL', 'MSFT']);
    const [strategyType, setStrategyType] = useState('ml_ensemble');
    const [startDate, setStartDate] = useState(
        new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
    );
    const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);
    const [runningBacktests, setRunningBacktests] = useState<Map<string, BacktestResult>>(new Map());
    const [completedBacktests, setCompletedBacktests] = useState<BacktestResult[]>([]);
    const [selectedBacktest, setSelectedBacktest] = useState<BacktestResult | null>(null);
    const [signalValidations, setSignalValidations] = useState<SignalValidation[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const API_BASE_URL = 'http://localhost:8001';

    useEffect(() => {
        // Poll for running backtest updates
        const interval = setInterval(() => {
            runningBacktests.forEach(async (backtest, id) => {
                if (backtest.status === 'running' || backtest.status === 'started') {
                    try {
                        const response = await axios.get(`${API_BASE_URL}/backtest/${id}`);
                        const updatedBacktest = response.data;

                        if (updatedBacktest.status === 'completed') {
                            setRunningBacktests(prev => {
                                const newMap = new Map(prev);
                                newMap.delete(id);
                                return newMap;
                            });
                            setCompletedBacktests(prev => [...prev, updatedBacktest]);
                        } else {
                            setRunningBacktests(prev => {
                                const newMap = new Map(prev);
                                newMap.set(id, updatedBacktest);
                                return newMap;
                            });
                        }
                    } catch (err) {
                        logger.error('Error polling backtest:', err);
                    }
                }
            });
        }, 2000);

        return () => clearInterval(interval);
    }, [runningBacktests]);

    const runBacktest = async () => {
        try {
            setLoading(true);
            setError(null);

            const request: BacktestRequest = {
                symbols,
                startDate,
                endDate,
                strategyType,
                useWalkForward: true,
                includeTransactionCosts: true,
            };

            const response = await axios.post(`${API_BASE_URL}/backtest`, request);
            const { backtestId } = response.data;

            setRunningBacktests(prev => {
                const newMap = new Map(prev);
                newMap.set(backtestId, response.data);
                return newMap;
            });
        } catch (err: any) {
            setError(err.message || 'Failed to start backtest');
        } finally {
            setLoading(false);
        }
    };

    const validateSignal = async (symbol: string, action: string, confidence: number) => {
        try {
            const response = await axios.post(`${API_BASE_URL}/validate-signal`, {
                symbol,
                action,
                confidence,
            });

            setSignalValidations(prev => [...prev, response.data]);
        } catch (err) {
            logger.error('Error validating signal:', err);
        }
    };

    const getPerformanceMetrics = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/performance-metrics`);
            return response.data;
        } catch (err) {
            logger.error('Error fetching performance metrics:', err);
            return null;
        }
    };

    const renderBacktestResults = (backtest: BacktestResult) => {
        if (!backtest.results) return null;

        const results = backtest.results.results || backtest.results;

        return (
            <Box>
                <Typography variant="h6" gutterBottom>
                    Backtest Results - {backtest.backtestId}
                </Typography>

                <Grid container spacing={3}>
                    {Object.entries(results).map(([symbol, data]: [string, any]) => (
                        <Grid item xs={12} md={6} key={symbol}>
                            <Card>
                                <CardContent>
                                    <Typography variant="h6" color="primary">
                                        {symbol}
                                    </Typography>

                                    {data.backtest_metrics && (
                                        <Box mt={2}>
                                            <Grid container spacing={2}>
                                                <Grid item xs={6}>
                                                    <Typography variant="body2" color="textSecondary">
                                                        Sharpe Ratio
                                                    </Typography>
                                                    <Typography variant="h6">
                                                        {data.backtest_metrics.sharpe_ratio.toFixed(2)}
                                                    </Typography>
                                                </Grid>
                                                <Grid item xs={6}>
                                                    <Typography variant="body2" color="textSecondary">
                                                        Annual Return
                                                    </Typography>
                                                    <Typography variant="h6" color={data.backtest_metrics.annual_return > 0 ? 'success.main' : 'error.main'}>
                                                        {(data.backtest_metrics.annual_return * 100).toFixed(1)}%
                                                    </Typography>
                                                </Grid>
                                                <Grid item xs={6}>
                                                    <Typography variant="body2" color="textSecondary">
                                                        Max Drawdown
                                                    </Typography>
                                                    <Typography variant="h6" color="error.main">
                                                        {(data.backtest_metrics.max_drawdown * 100).toFixed(1)}%
                                                    </Typography>
                                                </Grid>
                                                <Grid item xs={6}>
                                                    <Typography variant="body2" color="textSecondary">
                                                        Win Rate
                                                    </Typography>
                                                    <Typography variant="h6">
                                                        {(data.backtest_metrics.win_rate * 100).toFixed(1)}%
                                                    </Typography>
                                                </Grid>
                                            </Grid>
                                        </Box>
                                    )}

                                    {data.feature_importance && (
                                        <Box mt={3}>
                                            <Typography variant="subtitle2" gutterBottom>
                                                Top Features
                                            </Typography>
                                            {data.feature_importance.slice(0, 5).map(([feature, importance]: [string, number], idx: number) => (
                                                <Box key={idx} display="flex" justifyContent="space-between" mb={1}>
                                                    <Typography variant="body2">{feature}</Typography>
                                                    <Typography variant="body2" color="primary">
                                                        {(importance * 100).toFixed(1)}%
                                                    </Typography>
                                                </Box>
                                            ))}
                                        </Box>
                                    )}
                                </CardContent>
                            </Card>
                        </Grid>
                    ))}
                </Grid>
            </Box>
        );
    };

    const renderSignalValidations = () => {
        return (
            <TableContainer component={Paper}>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell>Symbol</TableCell>
                            <TableCell>Action</TableCell>
                            <TableCell>Confidence</TableCell>
                            <TableCell>Expected Return</TableCell>
                            <TableCell>Risk Score</TableCell>
                            <TableCell>Win Rate</TableCell>
                            <TableCell>Recommendation</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {signalValidations.map((validation, idx) => (
                            <TableRow key={idx}>
                                <TableCell>{validation.symbol}</TableCell>
                                <TableCell>
                                    <Chip
                                        label={validation.action}
                                        color={validation.action === 'BUY' ? 'success' : validation.action === 'SELL' ? 'error' : 'default'}
                                        size="small"
                                    />
                                </TableCell>
                                <TableCell>{(validation.confidence * 100).toFixed(0)}%</TableCell>
                                <TableCell>
                                    {(validation.validation.expectedAnnualReturn * 100).toFixed(1)}%
                                </TableCell>
                                <TableCell>{validation.validation.riskAdjustedScore.toFixed(2)}</TableCell>
                                <TableCell>{(validation.validation.historicalWinRate * 100).toFixed(1)}%</TableCell>
                                <TableCell>
                                    <Chip
                                        label={validation.validation.recommendation}
                                        color={validation.validation.recommendation === 'PROCEED' ? 'success' : 'warning'}
                                        size="small"
                                        icon={validation.validation.recommendation === 'PROCEED' ? <CheckCircle /> : <Warning />}
                                    />
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        );
    };

    return (
        <Box>
            <Typography variant="h4" gutterBottom>
                ML Backtesting Dashboard
            </Typography>

            <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)} sx={{ mb: 3 }}>
                <Tab label="Run Backtest" />
                <Tab label="Results" />
                <Tab label="Signal Validation" />
                <Tab label="Performance Metrics" />
            </Tabs>

            {activeTab === 0 && (
                <Card>
                    <CardContent>
                        <Typography variant="h6" gutterBottom>
                            Configure Backtest
                        </Typography>

                        <Grid container spacing={3}>
                            <Grid item xs={12} md={6}>
                                <TextField
                                    fullWidth
                                    label="Symbols (comma-separated)"
                                    value={symbols.join(', ')}
                                    onChange={(e) => setSymbols(e.target.value.split(',').map(s => s.trim()))}
                                />
                            </Grid>

                            <Grid item xs={12} md={6}>
                                <FormControl fullWidth>
                                    <InputLabel>Strategy Type</InputLabel>
                                    <Select
                                        value={strategyType}
                                        onChange={(e) => setStrategyType(e.target.value)}
                                    >
                                        <MenuItem value="ml_ensemble">ML Ensemble</MenuItem>
                                        <MenuItem value="technical">Technical Analysis</MenuItem>
                                        <MenuItem value="hybrid">Hybrid</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>

                            <Grid item xs={12} md={6}>
                                <TextField
                                    fullWidth
                                    type="date"
                                    label="Start Date"
                                    value={startDate}
                                    onChange={(e) => setStartDate(e.target.value)}
                                    InputLabelProps={{ shrink: true }}
                                />
                            </Grid>

                            <Grid item xs={12} md={6}>
                                <TextField
                                    fullWidth
                                    type="date"
                                    label="End Date"
                                    value={endDate}
                                    onChange={(e) => setEndDate(e.target.value)}
                                    InputLabelProps={{ shrink: true }}
                                />
                            </Grid>
                        </Grid>

                        <Box mt={3}>
                            <Button
                                variant="contained"
                                color="primary"
                                onClick={runBacktest}
                                disabled={loading}
                                startIcon={<PlayArrow />}
                            >
                                Run Backtest
                            </Button>
                        </Box>

                        {error && (
                            <Alert severity="error" sx={{ mt: 2 }}>
                                {error}
                            </Alert>
                        )}

                        {runningBacktests.size > 0 && (
                            <Box mt={3}>
                                <Typography variant="h6" gutterBottom>
                                    Running Backtests
                                </Typography>
                                {Array.from(runningBacktests.entries()).map(([id, backtest]) => (
                                    <Box key={id} mb={2}>
                                        <Typography variant="body2" color="textSecondary">
                                            {id}
                                        </Typography>
                                        <LinearProgress
                                            variant="determinate"
                                            value={backtest.progress * 100}
                                            sx={{ mt: 1 }}
                                        />
                                    </Box>
                                ))}
                            </Box>
                        )}
                    </CardContent>
                </Card>
            )}

            {activeTab === 1 && (
                <Box>
                    {completedBacktests.length === 0 ? (
                        <Alert severity="info">
                            No completed backtests yet. Run a backtest to see results.
                        </Alert>
                    ) : (
                        completedBacktests.map((backtest) => (
                            <Box key={backtest.backtestId} mb={3}>
                                {renderBacktestResults(backtest)}
                            </Box>
                        ))
                    )}
                </Box>
            )}

            {activeTab === 2 && (
                <Card>
                    <CardContent>
                        <Typography variant="h6" gutterBottom>
                            Signal Validation
                        </Typography>

                        <Box mb={3}>
                            <Grid container spacing={2}>
                                <Grid item xs={4}>
                                    <TextField
                                        fullWidth
                                        label="Symbol"
                                        id="val-symbol"
                                    />
                                </Grid>
                                <Grid item xs={3}>
                                    <FormControl fullWidth>
                                        <InputLabel>Action</InputLabel>
                                        <Select id="val-action" defaultValue="BUY">
                                            <MenuItem value="BUY">BUY</MenuItem>
                                            <MenuItem value="SELL">SELL</MenuItem>
                                            <MenuItem value="HOLD">HOLD</MenuItem>
                                        </Select>
                                    </FormControl>
                                </Grid>
                                <Grid item xs={3}>
                                    <TextField
                                        fullWidth
                                        type="number"
                                        label="Confidence"
                                        id="val-confidence"
                                        defaultValue="0.75"
                                        inputProps={{ min: 0, max: 1, step: 0.05 }}
                                    />
                                </Grid>
                                <Grid item xs={2}>
                                    <Button
                                        fullWidth
                                        variant="contained"
                                        onClick={() => {
                                            const symbol = (document.getElementById('val-symbol') as HTMLInputElement).value;
                                            const action = (document.getElementById('val-action') as HTMLInputElement).value;
                                            const confidence = parseFloat((document.getElementById('val-confidence') as HTMLInputElement).value);
                                            if (symbol) {
                                                validateSignal(symbol, action, confidence);
                                            }
                                        }}
                                    >
                                        Validate
                                    </Button>
                                </Grid>
                            </Grid>
                        </Box>

                        {signalValidations.length > 0 && renderSignalValidations()}
                    </CardContent>
                </Card>
            )}

            {activeTab === 3 && (
                <PerformanceMetricsTab />
            )}
        </Box>
    );
};

const PerformanceMetricsTab: React.FC = () => {
    const [metrics, setMetrics] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const response = await axios.get('http://localhost:8001/performance-metrics');
                setMetrics(response.data);
            } catch (err) {
                logger.error('Error fetching metrics:', err);
            } finally {
                setLoading(false);
            }
        };

        fetchMetrics();
    }, []);

    if (loading) return <LinearProgress />;
    if (!metrics) return <Alert severity="error">Failed to load metrics</Alert>;

    return (
        <Grid container spacing={3}>
            <Grid item xs={12} md={3}>
                <Card>
                    <CardContent>
                        <Typography color="textSecondary" gutterBottom>
                            Avg Sharpe Ratio
                        </Typography>
                        <Typography variant="h4">
                            {metrics.aggregate_metrics.average_sharpe_ratio.toFixed(2)}
                        </Typography>
                    </CardContent>
                </Card>
            </Grid>

            <Grid item xs={12} md={3}>
                <Card>
                    <CardContent>
                        <Typography color="textSecondary" gutterBottom>
                            Avg Annual Return
                        </Typography>
                        <Typography variant="h4" color={metrics.aggregate_metrics.average_annual_return > 0 ? 'success.main' : 'error.main'}>
                            {(metrics.aggregate_metrics.average_annual_return * 100).toFixed(1)}%
                        </Typography>
                    </CardContent>
                </Card>
            </Grid>

            <Grid item xs={12} md={3}>
                <Card>
                    <CardContent>
                        <Typography color="textSecondary" gutterBottom>
                            Avg Max Drawdown
                        </Typography>
                        <Typography variant="h4" color="error.main">
                            {(metrics.aggregate_metrics.average_max_drawdown * 100).toFixed(1)}%
                        </Typography>
                    </CardContent>
                </Card>
            </Grid>

            <Grid item xs={12} md={3}>
                <Card>
                    <CardContent>
                        <Typography color="textSecondary" gutterBottom>
                            Model Status
                        </Typography>
                        <Typography variant="h4" color="success.main">
                            {metrics.model_status.ml_models.toUpperCase()}
                        </Typography>
                    </CardContent>
                </Card>
            </Grid>

            <Grid item xs={12}>
                <Card>
                    <CardContent>
                        <Typography variant="h6" gutterBottom>
                            Recommendations
                        </Typography>
                        {metrics.model_status.recommendations.map((rec: string, idx: number) => (
                            <Alert severity="info" sx={{ mb: 1 }} key={idx}>
                                {rec}
                            </Alert>
                        ))}
                    </CardContent>
                </Card>
            </Grid>
        </Grid>
    );
};

export default BacktestingDashboard;
