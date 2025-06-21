import React, { useState, useEffect } from 'react';
import {
    Box,
    Grid,
    Paper,
    Typography,
    Card,
    CardContent,
    Chip,
    LinearProgress,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    IconButton,
    Button,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Tooltip,
    Alert,
    Tabs,
    Tab,
    Badge,
    CircularProgress,
} from '@mui/material';
import {
    TrendingUp as TrendingUpIcon,
    TrendingDown as TrendingDownIcon,
    ShowChart as ChartIcon,
    Speed as SpeedIcon,
    Security as SecurityIcon,
    Psychology as AIIcon,
    Timeline as TimelineIcon,
    Assessment as AssessmentIcon,
    PlayArrow as PlayIcon,
    Pause as PauseIcon,
    Settings as SettingsIcon,
    Info as InfoIcon,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import {
    LineChart,
    Line,
    AreaChart,
    Area,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip as RechartsTooltip,
    ResponsiveContainer,
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    Radar,
} from 'recharts';

interface Strategy {
    name: string;
    active: boolean;
    performance: number;
    trades: number;
    winRate: number;
}

interface Position {
    symbol: string;
    strategy: string;
    direction: 'long' | 'short';
    entryPrice: number;
    currentPrice: number;
    pnl: number;
    pnlPercent: number;
    size: number;
    confidence: number;
    holdingTime: string;
}

interface PerformanceMetric {
    label: string;
    value: number;
    change: number;
    target: number;
}

const AIQuantDashboard: React.FC = () => {
    const theme = useTheme();
    const [selectedTab, setSelectedTab] = useState(0);
    const [isSystemActive, setIsSystemActive] = useState(true);
    const [selectedStrategy, setSelectedStrategy] = useState('all');
    const [timeframe, setTimeframe] = useState('1D');

    // Mock data - would be replaced with real API calls
    const [strategies] = useState<Strategy[]>([
        { name: 'Mean Reversion', active: true, performance: 12.5, trades: 45, winRate: 68 },
        { name: 'Momentum', active: true, performance: 18.3, trades: 32, winRate: 72 },
        { name: 'Statistical Arbitrage', active: false, performance: 8.7, trades: 128, winRate: 58 },
        { name: 'Market Making', active: true, performance: 5.2, trades: 512, winRate: 52 },
        { name: 'ML Ensemble', active: true, performance: 22.1, trades: 67, winRate: 75 },
    ]);

    const [positions] = useState<Position[]>([
        {
            symbol: 'AAPL',
            strategy: 'ML Ensemble',
            direction: 'long',
            entryPrice: 185.50,
            currentPrice: 187.25,
            pnl: 350,
            pnlPercent: 0.94,
            size: 200,
            confidence: 87,
            holdingTime: '2h 15m',
        },
        {
            symbol: 'TSLA',
            strategy: 'Momentum',
            direction: 'long',
            entryPrice: 245.00,
            currentPrice: 248.50,
            pnl: 700,
            pnlPercent: 1.43,
            size: 200,
            confidence: 82,
            holdingTime: '45m',
        },
        {
            symbol: 'SPY',
            strategy: 'Mean Reversion',
            direction: 'short',
            entryPrice: 452.00,
            currentPrice: 451.25,
            pnl: 150,
            pnlPercent: 0.17,
            size: 200,
            confidence: 75,
            holdingTime: '1h 30m',
        },
    ]);

    const performanceMetrics: PerformanceMetric[] = [
        { label: 'Sharpe Ratio', value: 2.34, change: 0.12, target: 2.0 },
        { label: 'Win Rate', value: 68.5, change: 2.3, target: 65 },
        { label: 'Profit Factor', value: 1.82, change: 0.08, target: 1.5 },
        { label: 'Max Drawdown', value: 8.2, change: -1.1, target: 15 },
    ];

    // Mock performance data
    const performanceData = Array.from({ length: 30 }, (_, i) => ({
        day: i + 1,
        pnl: Math.random() * 2000 - 500 + i * 50,
        trades: Math.floor(Math.random() * 50) + 20,
        winRate: 50 + Math.random() * 30,
    }));

    // Strategy performance radar data
    const radarData = strategies.map(s => ({
        strategy: s.name,
        performance: s.performance,
        winRate: s.winRate,
        trades: Math.min(s.trades / 10, 100),
    }));

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setSelectedTab(newValue);
    };

    const toggleSystem = () => {
        setIsSystemActive(!isSystemActive);
    };

    return (
        <Box sx={{ p: 3 }}>
            {/* Header */}
            <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                    <Typography variant="h4" gutterBottom>
                        AI Quant Trading System
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Professional quantitative trading powered by machine learning
                    </Typography>
                </Box>
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                    <Chip
                        icon={isSystemActive ? <PlayIcon /> : <PauseIcon />}
                        label={isSystemActive ? 'System Active' : 'System Paused'}
                        color={isSystemActive ? 'success' : 'default'}
                        onClick={toggleSystem}
                    />
                    <FormControl size="small" sx={{ minWidth: 120 }}>
                        <InputLabel>Timeframe</InputLabel>
                        <Select value={timeframe} onChange={(e) => setTimeframe(e.target.value)}>
                            <MenuItem value="1H">1 Hour</MenuItem>
                            <MenuItem value="1D">1 Day</MenuItem>
                            <MenuItem value="1W">1 Week</MenuItem>
                            <MenuItem value="1M">1 Month</MenuItem>
                        </Select>
                    </FormControl>
                    <IconButton>
                        <SettingsIcon />
                    </IconButton>
                </Box>
            </Box>

            {/* Performance Metrics */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
                {performanceMetrics.map((metric) => (
                    <Grid item xs={12} sm={6} md={3} key={metric.label}>
                        <Card>
                            <CardContent>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                                    <Box>
                                        <Typography color="text.secondary" variant="body2">
                                            {metric.label}
                                        </Typography>
                                        <Typography variant="h4" sx={{ my: 1 }}>
                                            {metric.value}
                                            {metric.label.includes('Rate') || metric.label.includes('Drawdown') ? '%' : ''}
                                        </Typography>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                            {metric.change > 0 ? (
                                                <TrendingUpIcon color="success" fontSize="small" />
                                            ) : (
                                                <TrendingDownIcon color="error" fontSize="small" />
                                            )}
                                            <Typography
                                                variant="body2"
                                                color={metric.change > 0 ? 'success.main' : 'error.main'}
                                            >
                                                {metric.change > 0 ? '+' : ''}{metric.change}
                                            </Typography>
                                        </Box>
                                    </Box>
                                    <Box sx={{ position: 'relative', display: 'inline-flex' }}>
                                        <CircularProgress
                                            variant="determinate"
                                            value={(metric.value / metric.target) * 100}
                                            size={60}
                                            thickness={4}
                                            sx={{
                                                color: metric.value >= metric.target ? 'success.main' : 'warning.main',
                                            }}
                                        />
                                        <Box
                                            sx={{
                                                top: 0,
                                                left: 0,
                                                bottom: 0,
                                                right: 0,
                                                position: 'absolute',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                            }}
                                        >
                                            <Typography variant="caption" component="div" color="text.secondary">
                                                {Math.round((metric.value / metric.target) * 100)}%
                                            </Typography>
                                        </Box>
                                    </Box>
                                </Box>
                            </CardContent>
                        </Card>
                    </Grid>
                ))}
            </Grid>

            {/* Main Content Tabs */}
            <Paper sx={{ mb: 3 }}>
                <Tabs value={selectedTab} onChange={handleTabChange}>
                    <Tab label="Overview" icon={<ChartIcon />} iconPosition="start" />
                    <Tab label="Strategies" icon={<AIIcon />} iconPosition="start" />
                    <Tab label="Positions" icon={<AssessmentIcon />} iconPosition="start" />
                    <Tab label="Risk Analysis" icon={<SecurityIcon />} iconPosition="start" />
                </Tabs>
            </Paper>

            {/* Tab Content */}
            {selectedTab === 0 && (
                <Grid container spacing={3}>
                    {/* P&L Chart */}
                    <Grid item xs={12} lg={8}>
                        <Paper sx={{ p: 3, height: 400 }}>
                            <Typography variant="h6" gutterBottom>
                                Cumulative P&L
                            </Typography>
                            <ResponsiveContainer width="100%" height="90%">
                                <AreaChart data={performanceData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="day" />
                                    <YAxis />
                                    <RechartsTooltip />
                                    <Area
                                        type="monotone"
                                        dataKey="pnl"
                                        stroke={theme.palette.primary.main}
                                        fill={theme.palette.primary.light}
                                        fillOpacity={0.3}
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </Paper>
                    </Grid>

                    {/* Strategy Performance Radar */}
                    <Grid item xs={12} lg={4}>
                        <Paper sx={{ p: 3, height: 400 }}>
                            <Typography variant="h6" gutterBottom>
                                Strategy Performance
                            </Typography>
                            <ResponsiveContainer width="100%" height="90%">
                                <RadarChart data={radarData}>
                                    <PolarGrid />
                                    <PolarAngleAxis dataKey="strategy" />
                                    <PolarRadiusAxis angle={90} domain={[0, 100]} />
                                    <Radar
                                        name="Performance"
                                        dataKey="performance"
                                        stroke={theme.palette.primary.main}
                                        fill={theme.palette.primary.main}
                                        fillOpacity={0.6}
                                    />
                                    <Radar
                                        name="Win Rate"
                                        dataKey="winRate"
                                        stroke={theme.palette.secondary.main}
                                        fill={theme.palette.secondary.main}
                                        fillOpacity={0.6}
                                    />
                                </RadarChart>
                            </ResponsiveContainer>
                        </Paper>
                    </Grid>

                    {/* Active Positions Summary */}
                    <Grid item xs={12}>
                        <Paper sx={{ p: 3 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                                <Typography variant="h6">Active Positions</Typography>
                                <Chip label={`${positions.length} positions`} size="small" />
                            </Box>
                            <TableContainer>
                                <Table>
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Symbol</TableCell>
                                            <TableCell>Strategy</TableCell>
                                            <TableCell>Direction</TableCell>
                                            <TableCell align="right">Entry</TableCell>
                                            <TableCell align="right">Current</TableCell>
                                            <TableCell align="right">P&L</TableCell>
                                            <TableCell align="right">Size</TableCell>
                                            <TableCell align="right">Confidence</TableCell>
                                            <TableCell>Time</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {positions.map((position) => (
                                            <TableRow key={position.symbol}>
                                                <TableCell>
                                                    <Typography variant="body2" fontWeight="bold">
                                                        {position.symbol}
                                                    </Typography>
                                                </TableCell>
                                                <TableCell>
                                                    <Chip label={position.strategy} size="small" />
                                                </TableCell>
                                                <TableCell>
                                                    <Chip
                                                        label={position.direction}
                                                        size="small"
                                                        color={position.direction === 'long' ? 'success' : 'error'}
                                                    />
                                                </TableCell>
                                                <TableCell align="right">${position.entryPrice}</TableCell>
                                                <TableCell align="right">${position.currentPrice}</TableCell>
                                                <TableCell align="right">
                                                    <Typography
                                                        color={position.pnl > 0 ? 'success.main' : 'error.main'}
                                                        fontWeight="bold"
                                                    >
                                                        ${position.pnl} ({position.pnlPercent}%)
                                                    </Typography>
                                                </TableCell>
                                                <TableCell align="right">{position.size}</TableCell>
                                                <TableCell align="right">
                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                        <LinearProgress
                                                            variant="determinate"
                                                            value={position.confidence}
                                                            sx={{ width: 60, height: 6, borderRadius: 3 }}
                                                        />
                                                        <Typography variant="body2">{position.confidence}%</Typography>
                                                    </Box>
                                                </TableCell>
                                                <TableCell>{position.holdingTime}</TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </Paper>
                    </Grid>
                </Grid>
            )}

            {selectedTab === 1 && (
                <Grid container spacing={3}>
                    {/* Strategy Cards */}
                    {strategies.map((strategy) => (
                        <Grid item xs={12} md={6} lg={4} key={strategy.name}>
                            <Card>
                                <CardContent>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                                        <Typography variant="h6">{strategy.name}</Typography>
                                        <Chip
                                            label={strategy.active ? 'Active' : 'Inactive'}
                                            color={strategy.active ? 'success' : 'default'}
                                            size="small"
                                        />
                                    </Box>

                                    <Grid container spacing={2}>
                                        <Grid item xs={4}>
                                            <Typography variant="body2" color="text.secondary">
                                                Performance
                                            </Typography>
                                            <Typography variant="h6" color={strategy.performance > 0 ? 'success.main' : 'error.main'}>
                                                {strategy.performance > 0 ? '+' : ''}{strategy.performance}%
                                            </Typography>
                                        </Grid>
                                        <Grid item xs={4}>
                                            <Typography variant="body2" color="text.secondary">
                                                Trades
                                            </Typography>
                                            <Typography variant="h6">{strategy.trades}</Typography>
                                        </Grid>
                                        <Grid item xs={4}>
                                            <Typography variant="body2" color="text.secondary">
                                                Win Rate
                                            </Typography>
                                            <Typography variant="h6">{strategy.winRate}%</Typography>
                                        </Grid>
                                    </Grid>

                                    <Box sx={{ mt: 2 }}>
                                        <LinearProgress
                                            variant="determinate"
                                            value={strategy.winRate}
                                            sx={{ height: 8, borderRadius: 4 }}
                                            color={strategy.winRate > 60 ? 'success' : 'warning'}
                                        />
                                    </Box>

                                    <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                                        <Button size="small" variant="outlined">
                                            Configure
                                        </Button>
                                        <Button size="small" variant="outlined">
                                            View Details
                                        </Button>
                                    </Box>
                                </CardContent>
                            </Card>
                        </Grid>
                    ))}
                </Grid>
            )}

            {selectedTab === 2 && (
                <Alert severity="info">
                    Detailed position management interface would be implemented here
                </Alert>
            )}

            {selectedTab === 3 && (
                <Alert severity="info">
                    Risk analysis and portfolio optimization tools would be implemented here
                </Alert>
            )}
        </Box>
    );
};

export default AIQuantDashboard; 