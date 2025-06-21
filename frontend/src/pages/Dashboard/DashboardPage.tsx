import React, { useState } from 'react';
import {
    Box,
    Grid,
    Paper,
    Typography,
    Card,
    CardContent,
    useTheme,
    alpha,
    Stack,
    Chip,
    IconButton,
    ToggleButtonGroup,
    ToggleButton,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    AutoAwesome,
    Psychology,
    Timeline,
    Assessment,
    CalendarToday,
} from '@mui/icons-material';
import { SignalsChart } from '../../components/SignalsChart/SignalsChart';
import { motion } from 'framer-motion';
import { AlertProvider } from '../../contexts/AlertContext';
import { useAlerts } from '../../contexts/AlertContext';

interface MetricCardProps {
    title: string;
    value: string | number;
    change?: number;
    icon: React.ReactNode;
    color?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, change, icon, color }) => {
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
                    background: `linear-gradient(135deg, ${alpha(color || theme.palette.primary.main, 0.1)} 0%, ${alpha(theme.palette.background.paper, 0.8)} 100%)`,
                    backdropFilter: 'blur(10px)',
                    border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                }}
            >
                <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <Box>
                            <Typography variant="caption" color="text.secondary" gutterBottom>
                                {title}
                            </Typography>
                            <Typography variant="h4" fontWeight={700}>
                                {value}
                            </Typography>
                            {change !== undefined && (
                                <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                                    {change > 0 ? (
                                        <TrendingUp sx={{ fontSize: 16, color: theme.palette.success.main, mr: 0.5 }} />
                                    ) : (
                                        <TrendingDown sx={{ fontSize: 16, color: theme.palette.error.main, mr: 0.5 }} />
                                    )}
                                    <Typography
                                        variant="caption"
                                        color={change > 0 ? 'success.main' : 'error.main'}
                                        fontWeight={600}
                                    >
                                        {change > 0 ? '+' : ''}{change}%
                                    </Typography>
                                </Box>
                            )}
                        </Box>
                        <Box
                            sx={{
                                p: 1.5,
                                borderRadius: 2,
                                backgroundColor: alpha(color || theme.palette.primary.main, 0.1),
                                color: color || theme.palette.primary.main,
                            }}
                        >
                            {icon}
                        </Box>
                    </Box>
                </CardContent>
            </Card>
        </motion.div>
    );
};

export const DashboardPage: React.FC = () => {
    const theme = useTheme();
    const [timeframe, setTimeframe] = useState('1D');
    const [selectedStock, setSelectedStock] = useState('AAPL');

    const metrics = [
        {
            title: 'Active Signals',
            value: 23,
            change: 15,
            icon: <AutoAwesome />,
            color: theme.palette.warning.main,
        },
        {
            title: 'Patterns Detected',
            value: 7,
            change: -5,
            icon: <Timeline />,
            color: theme.palette.success.main,
        },
        {
            title: 'AI Agents Active',
            value: '12/15',
            icon: <Psychology />,
            color: theme.palette.secondary.main,
        },
        {
            title: 'Portfolio Value',
            value: '$125.4K',
            change: 3.2,
            icon: <Assessment />,
            color: theme.palette.primary.main,
        },
    ];

    const topSignals = [
        { symbol: 'AAPL', type: 'BUY', confidence: 92, reason: 'Bullish Pattern + Sentiment' },
        { symbol: 'MSFT', type: 'HOLD', confidence: 78, reason: 'Consolidation Phase' },
        { symbol: 'GOOGL', type: 'BUY', confidence: 85, reason: 'Technical Breakout' },
        { symbol: 'TSLA', type: 'SELL', confidence: 71, reason: 'Overbought Conditions' },
    ];

    return (
        <Box>
            {/* Page Header */}
            <Box sx={{ mb: 4 }}>
                <Typography variant="h4" gutterBottom>
                    Dashboard
                </Typography>
                <Typography variant="subtitle1" color="text.secondary">
                    Real-time market insights powered by AI
                </Typography>
            </Box>

            {/* Metrics Grid */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                {metrics.map((metric, index) => (
                    <Grid item xs={12} sm={6} md={3} key={index}>
                        <MetricCard {...metric} />
                    </Grid>
                ))}
            </Grid>

            {/* Main Content Grid */}
            <Grid container spacing={3}>
                {/* Trading Chart - Full Width */}
                <Grid item xs={12}>
                    <TradingChart
                        height={500}
                        onSelectSignal={handleSelectSignal}
                        onSymbolChange={(symbol) => {
                            console.log('Symbol changed to:', symbol);
                            // You can add any additional logic here when symbol changes
                        }}
                    />
                </Grid>

                {/* Top Signals */}
                <Grid item xs={12} lg={4}>
                    <Paper
                        sx={{
                            p: 3,
                            height: 500,
                            backgroundColor: alpha(theme.palette.background.paper, 0.8),
                            backdropFilter: 'blur(10px)',
                            overflow: 'auto',
                        }}
                    >
                        <Typography variant="h6" gutterBottom>
                            Top Signals Today
                        </Typography>
                        <Stack spacing={2}>
                            {topSignals.map((signal, index) => (
                                <motion.div
                                    key={index}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: index * 0.1 }}
                                >
                                    <Box
                                        sx={{
                                            p: 2,
                                            borderRadius: 2,
                                            backgroundColor: alpha(theme.palette.background.default, 0.5),
                                            border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                                            cursor: 'pointer',
                                            transition: 'all 0.2s',
                                            '&:hover': {
                                                backgroundColor: alpha(theme.palette.primary.main, 0.05),
                                                transform: 'translateX(4px)',
                                            },
                                        }}
                                        onClick={() => setSelectedStock(signal.symbol)}
                                    >
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                            <Box>
                                                <Typography variant="subtitle1" fontWeight={600}>
                                                    {signal.symbol}
                                                </Typography>
                                                <Typography variant="caption" color="text.secondary">
                                                    {signal.reason}
                                                </Typography>
                                            </Box>
                                            <Box sx={{ textAlign: 'right' }}>
                                                <Chip
                                                    label={signal.type}
                                                    size="small"
                                                    color={signal.type === 'BUY' ? 'success' : signal.type === 'SELL' ? 'error' : 'warning'}
                                                    sx={{ mb: 0.5 }}
                                                />
                                                <Typography variant="caption" display="block" color="text.secondary">
                                                    {signal.confidence}% confidence
                                                </Typography>
                                            </Box>
                                        </Box>
                                    </Box>
                                </motion.div>
                            ))}
                        </Stack>
                    </Paper>
                </Grid>

                {/* Recent Activity */}
                <Grid item xs={12}>
                    <Paper
                        sx={{
                            p: 3,
                            backgroundColor: alpha(theme.palette.background.paper, 0.8),
                            backdropFilter: 'blur(10px)',
                        }}
                    >
                        <Typography variant="h6" gutterBottom>
                            Recent AI Agent Activity
                        </Typography>
                        <Box sx={{ mt: 2 }}>
                            <Stack direction="row" spacing={2} sx={{ overflowX: 'auto', pb: 1 }}>
                                {[
                                    { agent: 'Technical Scanner', action: 'Found ascending triangle on NVDA', time: '2m ago' },
                                    { agent: 'Sentiment Analyzer', action: 'Detected positive news surge for AAPL', time: '5m ago' },
                                    { agent: 'Risk Monitor', action: 'Portfolio VaR within acceptable range', time: '10m ago' },
                                    { agent: 'Pattern Recognition', action: 'Identified 3 new patterns', time: '15m ago' },
                                ].map((activity, index) => (
                                    <Box
                                        key={index}
                                        sx={{
                                            minWidth: 300,
                                            p: 2,
                                            borderRadius: 2,
                                            backgroundColor: alpha(theme.palette.background.default, 0.5),
                                            border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                                        }}
                                    >
                                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                            <Psychology sx={{ fontSize: 20, mr: 1, color: theme.palette.primary.main }} />
                                            <Typography variant="subtitle2" fontWeight={600}>
                                                {activity.agent}
                                            </Typography>
                                        </Box>
                                        <Typography variant="body2" color="text.secondary" gutterBottom>
                                            {activity.action}
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary">
                                            {activity.time}
                                        </Typography>
                                    </Box>
                                ))}
                            </Stack>
                        </Box>
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
}; 