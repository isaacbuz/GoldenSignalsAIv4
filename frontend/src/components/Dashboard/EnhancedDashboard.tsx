import React, { useState } from 'react';
import {
    Box,
    Grid,
    Card,
    CardContent,
    Typography,
    Stack,
    Chip,
    IconButton,
    Button,
    LinearProgress,
    Avatar,
    AvatarGroup,
    Tooltip,
    useTheme,
    alpha,
    Paper,
    Skeleton,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    MoreVert,
    ArrowUpward,
    ArrowDownward,
    AccessTime,
    Psychology,
    Speed,
    Security,
    MonetizationOn,
    ShowChart,
    Refresh,
    FilterList,
    DateRange,
    Download,
    Share,
    Fullscreen,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import {
    LineChart,
    Line,
    AreaChart,
    Area,
    BarChart,
    Bar,
    PieChart,
    Pie,
    Cell,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip as RechartsTooltip,
    ResponsiveContainer,
    Legend,
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    Radar,
} from 'recharts';

// Mock data
const performanceData = [
    { date: 'Mon', portfolio: 125000, benchmark: 122000, signals: 15 },
    { date: 'Tue', portfolio: 126500, benchmark: 122500, signals: 18 },
    { date: 'Wed', portfolio: 125800, benchmark: 121800, signals: 12 },
    { date: 'Thu', portfolio: 127200, benchmark: 123000, signals: 22 },
    { date: 'Fri', portfolio: 128500, benchmark: 123500, signals: 25 },
    { date: 'Sat', portfolio: 129000, benchmark: 124000, signals: 8 },
    { date: 'Sun', portfolio: 129500, benchmark: 124200, signals: 10 },
];

const assetAllocation = [
    { name: 'Stocks', value: 45, color: '#8884d8' },
    { name: 'Options', value: 25, color: '#82ca9d' },
    { name: 'Crypto', value: 20, color: '#ffc658' },
    { name: 'Cash', value: 10, color: '#ff7c7c' },
];

const topSignals = [
    { symbol: 'AAPL', type: 'BUY', confidence: 92, profit: '+5.2%', time: '2m ago' },
    { symbol: 'TSLA', type: 'SELL', confidence: 88, profit: '+3.8%', time: '15m ago' },
    { symbol: 'MSFT', type: 'BUY', confidence: 85, profit: '+2.1%', time: '1h ago' },
    { symbol: 'GOOGL', type: 'HOLD', confidence: 78, profit: '+1.5%', time: '2h ago' },
];

const riskMetrics = [
    { metric: 'Sharpe Ratio', value: 2.1, status: 'good' },
    { metric: 'Max Drawdown', value: -8.5, status: 'warning' },
    { metric: 'Win Rate', value: 68, status: 'good' },
    { metric: 'Risk/Reward', value: 1.8, status: 'good' },
];

const aiPerformance = [
    { subject: 'Pattern Recognition', A: 120, B: 110, fullMark: 150 },
    { subject: 'Market Prediction', A: 98, B: 130, fullMark: 150 },
    { subject: 'Risk Assessment', A: 86, B: 130, fullMark: 150 },
    { subject: 'Signal Generation', A: 99, B: 100, fullMark: 150 },
    { subject: 'Sentiment Analysis', A: 85, B: 90, fullMark: 150 },
];

interface MetricCardProps {
    title: string;
    value: string | number;
    change?: number;
    icon: React.ReactNode;
    color?: string;
    loading?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({
    title,
    value,
    change,
    icon,
    color = 'primary.main',
    loading = false,
}) => {
    const theme = useTheme();

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
        >
            <Card
                sx={{
                    height: '100%',
                    background: `linear-gradient(135deg, ${alpha(color, 0.1)} 0%, ${alpha(
                        color,
                        0.05
                    )} 100%)`,
                    border: `1px solid ${alpha(color, 0.2)}`,
                    position: 'relative',
                    overflow: 'hidden',
                }}
            >
                <CardContent>
                    <Stack spacing={2}>
                        <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
                            <Box>
                                <Typography variant="caption" color="text.secondary" gutterBottom>
                                    {title}
                                </Typography>
                                {loading ? (
                                    <Skeleton variant="text" width={120} height={40} />
                                ) : (
                                    <Typography variant="h4" fontWeight={700}>
                                        {value}
                                    </Typography>
                                )}
                            </Box>
                            <Avatar
                                sx={{
                                    backgroundColor: alpha(color, 0.1),
                                    color: color,
                                    width: 48,
                                    height: 48,
                                }}
                            >
                                {icon}
                            </Avatar>
                        </Stack>
                        {change !== undefined && !loading && (
                            <Stack direction="row" alignItems="center" spacing={0.5}>
                                {change >= 0 ? (
                                    <ArrowUpward sx={{ fontSize: 16, color: 'success.main' }} />
                                ) : (
                                    <ArrowDownward sx={{ fontSize: 16, color: 'error.main' }} />
                                )}
                                <Typography
                                    variant="body2"
                                    color={change >= 0 ? 'success.main' : 'error.main'}
                                    fontWeight={600}
                                >
                                    {Math.abs(change)}%
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                    vs last week
                                </Typography>
                            </Stack>
                        )}
                    </Stack>
                </CardContent>
                <Box
                    sx={{
                        position: 'absolute',
                        bottom: -20,
                        right: -20,
                        width: 100,
                        height: 100,
                        borderRadius: '50%',
                        backgroundColor: alpha(color, 0.05),
                    }}
                />
            </Card>
        </motion.div>
    );
};

export const EnhancedDashboard: React.FC = () => {
    const theme = useTheme();
    const [timeRange, setTimeRange] = useState('7d');
    const [loading, setLoading] = useState(false);

    const handleRefresh = () => {
        setLoading(true);
        setTimeout(() => setLoading(false), 1500);
    };

    return (
        <Box sx={{ p: 3 }}>
            {/* Header */}
            <Stack
                direction="row"
                justifyContent="space-between"
                alignItems="center"
                mb={3}
            >
                <Box>
                    <Typography variant="h4" fontWeight={700} gutterBottom>
                        Dashboard
                    </Typography>
                    <Stack direction="row" spacing={2} alignItems="center">
                        <Chip
                            icon={<AccessTime />}
                            label="Live"
                            color="success"
                            size="small"
                            sx={{ fontWeight: 600 }}
                        />
                        <Typography variant="body2" color="text.secondary">
                            Last updated: {new Date().toLocaleTimeString()}
                        </Typography>
                    </Stack>
                </Box>
                <Stack direction="row" spacing={1}>
                    <Button
                        startIcon={<DateRange />}
                        variant="outlined"
                        size="small"
                    >
                        {timeRange === '7d' ? 'Last 7 Days' : 'Custom Range'}
                    </Button>
                    <IconButton onClick={handleRefresh} disabled={loading}>
                        <Refresh />
                    </IconButton>
                    <IconButton>
                        <FilterList />
                    </IconButton>
                    <IconButton>
                        <Download />
                    </IconButton>
                    <IconButton>
                        <Fullscreen />
                    </IconButton>
                </Stack>
            </Stack>

            {/* Key Metrics */}
            <Grid container spacing={3} mb={3}>
                <Grid item xs={12} sm={6} md={3}>
                    <MetricCard
                        title="Portfolio Value"
                        value="$129,500"
                        change={3.6}
                        icon={<MonetizationOn />}
                        color={theme.palette.primary.main}
                        loading={loading}
                    />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                    <MetricCard
                        title="Active Signals"
                        value="25"
                        change={12.5}
                        icon={<ShowChart />}
                        color={theme.palette.success.main}
                        loading={loading}
                    />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                    <MetricCard
                        title="Win Rate"
                        value="68%"
                        change={-2.1}
                        icon={<Speed />}
                        color={theme.palette.warning.main}
                        loading={loading}
                    />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                    <MetricCard
                        title="AI Confidence"
                        value="92%"
                        change={5.3}
                        icon={<Psychology />}
                        color={theme.palette.info.main}
                        loading={loading}
                    />
                </Grid>
            </Grid>

            {/* Main Charts */}
            <Grid container spacing={3}>
                {/* Portfolio Performance */}
                <Grid item xs={12} lg={8}>
                    <Card>
                        <CardContent>
                            <Stack
                                direction="row"
                                justifyContent="space-between"
                                alignItems="center"
                                mb={2}
                            >
                                <Typography variant="h6" fontWeight={600}>
                                    Portfolio Performance
                                </Typography>
                                <Stack direction="row" spacing={1}>
                                    <Chip label="Portfolio" size="small" sx={{ backgroundColor: '#8884d8', color: 'white' }} />
                                    <Chip label="Benchmark" size="small" sx={{ backgroundColor: '#82ca9d', color: 'white' }} />
                                </Stack>
                            </Stack>
                            <ResponsiveContainer width="100%" height={300}>
                                <AreaChart data={performanceData}>
                                    <defs>
                                        <linearGradient id="colorPortfolio" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8} />
                                            <stop offset="95%" stopColor="#8884d8" stopOpacity={0} />
                                        </linearGradient>
                                        <linearGradient id="colorBenchmark" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#82ca9d" stopOpacity={0.8} />
                                            <stop offset="95%" stopColor="#82ca9d" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke={alpha(theme.palette.divider, 0.3)} />
                                    <XAxis dataKey="date" stroke={theme.palette.text.secondary} />
                                    <YAxis stroke={theme.palette.text.secondary} />
                                    <RechartsTooltip
                                        contentStyle={{
                                            backgroundColor: alpha(theme.palette.background.paper, 0.95),
                                            border: `1px solid ${theme.palette.divider}`,
                                            borderRadius: 8,
                                        }}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="portfolio"
                                        stroke="#8884d8"
                                        fillOpacity={1}
                                        fill="url(#colorPortfolio)"
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="benchmark"
                                        stroke="#82ca9d"
                                        fillOpacity={1}
                                        fill="url(#colorBenchmark)"
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </Grid>

                {/* Asset Allocation */}
                <Grid item xs={12} lg={4}>
                    <Card sx={{ height: '100%' }}>
                        <CardContent>
                            <Typography variant="h6" fontWeight={600} gutterBottom>
                                Asset Allocation
                            </Typography>
                            <ResponsiveContainer width="100%" height={250}>
                                <PieChart>
                                    <Pie
                                        data={assetAllocation}
                                        cx="50%"
                                        cy="50%"
                                        labelLine={false}
                                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                        outerRadius={80}
                                        fill="#8884d8"
                                        dataKey="value"
                                    >
                                        {assetAllocation.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Pie>
                                    <RechartsTooltip />
                                </PieChart>
                            </ResponsiveContainer>
                            <Stack spacing={1} mt={2}>
                                {assetAllocation.map((asset) => (
                                    <Stack
                                        key={asset.name}
                                        direction="row"
                                        justifyContent="space-between"
                                        alignItems="center"
                                    >
                                        <Stack direction="row" spacing={1} alignItems="center">
                                            <Box
                                                sx={{
                                                    width: 12,
                                                    height: 12,
                                                    borderRadius: '50%',
                                                    backgroundColor: asset.color,
                                                }}
                                            />
                                            <Typography variant="body2">{asset.name}</Typography>
                                        </Stack>
                                        <Typography variant="body2" fontWeight={600}>
                                            {asset.value}%
                                        </Typography>
                                    </Stack>
                                ))}
                            </Stack>
                        </CardContent>
                    </Card>
                </Grid>

                {/* Top Signals */}
                <Grid item xs={12} md={6}>
                    <Card>
                        <CardContent>
                            <Stack
                                direction="row"
                                justifyContent="space-between"
                                alignItems="center"
                                mb={2}
                            >
                                <Typography variant="h6" fontWeight={600}>
                                    Top Trading Signals
                                </Typography>
                                <Button size="small" endIcon={<ArrowUpward />}>
                                    View All
                                </Button>
                            </Stack>
                            <Stack spacing={2}>
                                {topSignals.map((signal, index) => (
                                    <motion.div
                                        key={signal.symbol}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: index * 0.1 }}
                                    >
                                        <Paper
                                            sx={{
                                                p: 2,
                                                backgroundColor: alpha(
                                                    signal.type === 'BUY'
                                                        ? theme.palette.success.main
                                                        : signal.type === 'SELL'
                                                            ? theme.palette.error.main
                                                            : theme.palette.warning.main,
                                                    0.05
                                                ),
                                                border: `1px solid ${alpha(
                                                    signal.type === 'BUY'
                                                        ? theme.palette.success.main
                                                        : signal.type === 'SELL'
                                                            ? theme.palette.error.main
                                                            : theme.palette.warning.main,
                                                    0.2
                                                )}`,
                                            }}
                                        >
                                            <Stack
                                                direction="row"
                                                justifyContent="space-between"
                                                alignItems="center"
                                            >
                                                <Stack direction="row" spacing={2} alignItems="center">
                                                    <Avatar
                                                        sx={{
                                                            backgroundColor: alpha(
                                                                signal.type === 'BUY'
                                                                    ? theme.palette.success.main
                                                                    : signal.type === 'SELL'
                                                                        ? theme.palette.error.main
                                                                        : theme.palette.warning.main,
                                                                0.1
                                                            ),
                                                            color:
                                                                signal.type === 'BUY'
                                                                    ? theme.palette.success.main
                                                                    : signal.type === 'SELL'
                                                                        ? theme.palette.error.main
                                                                        : theme.palette.warning.main,
                                                        }}
                                                    >
                                                        {signal.type === 'BUY' ? (
                                                            <TrendingUp />
                                                        ) : signal.type === 'SELL' ? (
                                                            <TrendingDown />
                                                        ) : (
                                                            <ShowChart />
                                                        )}
                                                    </Avatar>
                                                    <Box>
                                                        <Stack direction="row" spacing={1} alignItems="center">
                                                            <Typography variant="subtitle1" fontWeight={600}>
                                                                {signal.symbol}
                                                            </Typography>
                                                            <Chip
                                                                label={signal.type}
                                                                size="small"
                                                                color={
                                                                    signal.type === 'BUY'
                                                                        ? 'success'
                                                                        : signal.type === 'SELL'
                                                                            ? 'error'
                                                                            : 'warning'
                                                                }
                                                            />
                                                        </Stack>
                                                        <Typography variant="caption" color="text.secondary">
                                                            {signal.time} • Confidence: {signal.confidence}%
                                                        </Typography>
                                                    </Box>
                                                </Stack>
                                                <Typography
                                                    variant="h6"
                                                    fontWeight={700}
                                                    color={
                                                        signal.profit.startsWith('+')
                                                            ? 'success.main'
                                                            : 'error.main'
                                                    }
                                                >
                                                    {signal.profit}
                                                </Typography>
                                            </Stack>
                                        </Paper>
                                    </motion.div>
                                ))}
                            </Stack>
                        </CardContent>
                    </Card>
                </Grid>

                {/* AI Performance Radar */}
                <Grid item xs={12} md={6}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" fontWeight={600} gutterBottom>
                                AI Model Performance
                            </Typography>
                            <ResponsiveContainer width="100%" height={300}>
                                <RadarChart data={aiPerformance}>
                                    <PolarGrid stroke={alpha(theme.palette.divider, 0.3)} />
                                    <PolarAngleAxis dataKey="subject" />
                                    <PolarRadiusAxis angle={90} domain={[0, 150]} />
                                    <Radar
                                        name="Current"
                                        dataKey="A"
                                        stroke="#8884d8"
                                        fill="#8884d8"
                                        fillOpacity={0.6}
                                    />
                                    <Radar
                                        name="Target"
                                        dataKey="B"
                                        stroke="#82ca9d"
                                        fill="#82ca9d"
                                        fillOpacity={0.6}
                                    />
                                    <Legend />
                                </RadarChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </Grid>

                {/* Risk Metrics */}
                <Grid item xs={12}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" fontWeight={600} gutterBottom>
                                Risk Metrics
                            </Typography>
                            <Grid container spacing={3}>
                                {riskMetrics.map((metric) => (
                                    <Grid item xs={12} sm={6} md={3} key={metric.metric}>
                                        <Stack spacing={1}>
                                            <Stack
                                                direction="row"
                                                justifyContent="space-between"
                                                alignItems="center"
                                            >
                                                <Typography variant="body2" color="text.secondary">
                                                    {metric.metric}
                                                </Typography>
                                                <Chip
                                                    size="small"
                                                    label={metric.status}
                                                    color={
                                                        metric.status === 'good'
                                                            ? 'success'
                                                            : metric.status === 'warning'
                                                                ? 'warning'
                                                                : 'error'
                                                    }
                                                />
                                            </Stack>
                                            <Typography variant="h5" fontWeight={700}>
                                                {metric.value}
                                                {metric.metric.includes('Rate') && '%'}
                                            </Typography>
                                            <LinearProgress
                                                variant="determinate"
                                                value={
                                                    metric.status === 'good'
                                                        ? 80
                                                        : metric.status === 'warning'
                                                            ? 50
                                                            : 20
                                                }
                                                sx={{
                                                    height: 6,
                                                    borderRadius: 3,
                                                    backgroundColor: alpha(theme.palette.divider, 0.1),
                                                    '& .MuiLinearProgress-bar': {
                                                        borderRadius: 3,
                                                        backgroundColor:
                                                            metric.status === 'good'
                                                                ? theme.palette.success.main
                                                                : metric.status === 'warning'
                                                                    ? theme.palette.warning.main
                                                                    : theme.palette.error.main,
                                                    },
                                                }}
                                            />
                                        </Stack>
                                    </Grid>
                                ))}
                            </Grid>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>
        </Box>
    );
}; 