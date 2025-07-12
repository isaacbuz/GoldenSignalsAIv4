import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
    Box,
    Grid,
    Card,
    CardContent,
    Typography,
    Chip,
    IconButton,
    Divider,
    LinearProgress,
    Stack,
    Paper,
    useTheme,
    alpha,
} from '@mui/material';
import {
    TrendingUp as TrendingUpIcon,
    TrendingDown as TrendingDownIcon,
    Analytics as AnalyticsIcon,
    Psychology as PsychologyIcon,
    Speed as SpeedIcon,
    Refresh as RefreshIcon,
    ShowChart as ShowChartIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { motion } from 'framer-motion';

// Import existing components
import { AdvancedSignalChart } from '../Chart/AdvancedSignalChart';
import AgentConsensusFlow from '../Agents/AgentConsensusFlow';
import SignalList from '../Signals/SignalList';

const DashboardContainer = styled(Box)(({ theme }) => ({
    minHeight: '100vh',
    backgroundColor: theme.palette.background.default,
    padding: theme.spacing(3),
}));

const DashboardCard = styled(Card)(({ theme }) => ({
    backgroundColor: theme.palette.background.paper,
    border: `1px solid ${alpha('#FFD700', 0.2)}`,
    borderRadius: 12,
    height: '100%',
    transition: 'all 0.3s ease',
    '&:hover': {
        borderColor: '#FFD700',
        boxShadow: `0 4px 20px ${alpha('#FFD700', 0.1)}`,
        transform: 'translateY(-2px)',
    },
}));

const MetricCard = styled(Paper)(({ theme }) => ({
    padding: theme.spacing(2),
    backgroundColor: alpha(theme.palette.background.paper, 0.8),
    border: `1px solid ${alpha('#FFD700', 0.2)}`,
    borderRadius: 8,
    textAlign: 'center',
    transition: 'all 0.2s ease',
    '&:hover': {
        borderColor: '#FFD700',
        backgroundColor: alpha('#FFD700', 0.05),
    },
}));

// Mock data
const MARKET_METRICS = [
    { label: 'S&P 500', value: '4,456.78', change: '+1.24%', positive: true },
    { label: 'NASDAQ', value: '13,789.45', change: '+0.87%', positive: true },
    { label: 'VIX', value: '18.45', change: '-2.15%', positive: false },
    { label: 'DXY', value: '103.67', change: '+0.34%', positive: true },
];

const SIGNAL_SUMMARY = {
    totalSignals: 24,
    bullishSignals: 16,
    bearishSignals: 5,
    neutralSignals: 3,
    averageConfidence: 87.4,
    winRate: 73.2,
};

interface UnifiedDashboardProps {
    symbol?: string;
}

export const UnifiedDashboard: React.FC<UnifiedDashboardProps> = ({ symbol = 'SPY' }) => {
    const theme = useTheme();
    const [refreshing, setRefreshing] = useState(false);
    const [selectedTimeframe, setSelectedTimeframe] = useState('1d');

    const handleRefresh = useCallback(async () => {
        setRefreshing(true);
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));
        setRefreshing(false);
    }, []);

    const mockSignals = useMemo(() => [
        {
            id: '1',
            symbol: 'SPY',
            type: 'BUY' as const,
            action: 'BUY' as const,
            confidence: 94,
            price: 456.78,
            timestamp: new Date().toISOString(),
            source: 'AI Prophet',
            reasoning: 'Strong upward momentum with bullish indicators',
            strength: 'STRONG' as const,
            agents: ['technical', 'momentum'],
        },
        {
            id: '2',
            symbol: 'AAPL',
            type: 'HOLD' as const,
            action: 'HOLD' as const,
            confidence: 78,
            price: 189.45,
            timestamp: new Date().toISOString(),
            source: 'Technical',
            reasoning: 'Consolidation phase with mixed signals',
            strength: 'MODERATE' as const,
            agents: ['technical'],
        },
        {
            id: '3',
            symbol: 'MSFT',
            type: 'BUY' as const,
            action: 'BUY' as const,
            confidence: 87,
            price: 378.92,
            timestamp: new Date().toISOString(),
            source: 'Momentum',
            reasoning: 'Breakout pattern with high volume',
            strength: 'STRONG' as const,
            agents: ['momentum', 'volume'],
        },
    ], []);

    return (
        <DashboardContainer>
            {/* Header */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h4" sx={{ fontWeight: 700, color: 'text.primary' }}>
                    Unified Trading Dashboard
                </Typography>
                <IconButton onClick={handleRefresh} disabled={refreshing}>
                    <RefreshIcon />
                </IconButton>
            </Box>

            {/* Market Overview */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
                {MARKET_METRICS.map((metric, index) => (
                    <Grid item xs={12} sm={6} md={3} key={metric.label}>
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.1 }}
                        >
                            <MetricCard elevation={2}>
                                <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'uppercase' }}>
                                    {metric.label}
                                </Typography>
                                <Typography variant="h6" sx={{ fontWeight: 600, my: 0.5 }}>
                                    {metric.value}
                                </Typography>
                                <Chip
                                    label={metric.change}
                                    size="small"
                                    color={metric.positive ? 'success' : 'error'}
                                    icon={metric.positive ? <TrendingUpIcon /> : <TrendingDownIcon />}
                                />
                            </MetricCard>
                        </motion.div>
                    </Grid>
                ))}
            </Grid>

            {/* Main Dashboard Grid */}
            <Grid container spacing={3}>
                {/* Advanced Chart */}
                <Grid item xs={12} lg={8}>
                    <DashboardCard>
                        <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                <ShowChartIcon sx={{ mr: 1, color: 'primary.main' }} />
                                <Typography variant="h6" fontWeight="bold">
                                    Advanced Chart Analysis
                                </Typography>
                            </Box>
                            <AdvancedSignalChart
                                symbol={symbol}
                                height={400}
                                showControls={true}
                                showSignals={true}
                                timeframe={selectedTimeframe as any}
                            />
                        </CardContent>
                    </DashboardCard>
                </Grid>

                {/* Signal Summary */}
                <Grid item xs={12} lg={4}>
                    <DashboardCard>
                        <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                <AnalyticsIcon sx={{ mr: 1, color: 'primary.main' }} />
                                <Typography variant="h6" fontWeight="bold">
                                    Signal Summary
                                </Typography>
                            </Box>

                            <Stack spacing={2}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <Typography variant="body2">Total Signals</Typography>
                                    <Typography variant="h6" fontWeight="bold">
                                        {SIGNAL_SUMMARY.totalSignals}
                                    </Typography>
                                </Box>

                                <Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                        <Typography variant="body2">Bullish</Typography>
                                        <Typography variant="body2" color="success.main" fontWeight="bold">
                                            {SIGNAL_SUMMARY.bullishSignals}
                                        </Typography>
                                    </Box>
                                    <LinearProgress
                                        variant="determinate"
                                        value={(SIGNAL_SUMMARY.bullishSignals / SIGNAL_SUMMARY.totalSignals) * 100}
                                        sx={{
                                            height: 6,
                                            borderRadius: 3,
                                            backgroundColor: alpha(theme.palette.success.main, 0.2),
                                            '& .MuiLinearProgress-bar': {
                                                backgroundColor: theme.palette.success.main,
                                            },
                                        }}
                                    />
                                </Box>

                                <Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                        <Typography variant="body2">Bearish</Typography>
                                        <Typography variant="body2" color="error.main" fontWeight="bold">
                                            {SIGNAL_SUMMARY.bearishSignals}
                                        </Typography>
                                    </Box>
                                    <LinearProgress
                                        variant="determinate"
                                        value={(SIGNAL_SUMMARY.bearishSignals / SIGNAL_SUMMARY.totalSignals) * 100}
                                        sx={{
                                            height: 6,
                                            borderRadius: 3,
                                            backgroundColor: alpha(theme.palette.error.main, 0.2),
                                            '& .MuiLinearProgress-bar': {
                                                backgroundColor: theme.palette.error.main,
                                            },
                                        }}
                                    />
                                </Box>

                                <Divider />

                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <Typography variant="body2">Avg Confidence</Typography>
                                    <Typography variant="h6" fontWeight="bold" color="primary.main">
                                        {SIGNAL_SUMMARY.averageConfidence}%
                                    </Typography>
                                </Box>

                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <Typography variant="body2">Win Rate</Typography>
                                    <Typography variant="h6" fontWeight="bold" color="success.main">
                                        {SIGNAL_SUMMARY.winRate}%
                                    </Typography>
                                </Box>
                            </Stack>
                        </CardContent>
                    </DashboardCard>
                </Grid>

                {/* Agent Consensus */}
                <Grid item xs={12} lg={6}>
                    <DashboardCard>
                        <CardContent sx={{ height: 500, overflow: 'auto' }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                <PsychologyIcon sx={{ mr: 1, color: 'primary.main' }} />
                                <Typography variant="h6" fontWeight="bold">
                                    AI Agent Consensus
                                </Typography>
                            </Box>
                            <AgentConsensusFlow />
                        </CardContent>
                    </DashboardCard>
                </Grid>

                {/* Recent Signals */}
                <Grid item xs={12} lg={6}>
                    <DashboardCard>
                        <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                <SpeedIcon sx={{ mr: 1, color: 'primary.main' }} />
                                <Typography variant="h6" fontWeight="bold">
                                    Recent Signals
                                </Typography>
                            </Box>
                            <SignalList
                                signals={mockSignals}
                                onSignalSelect={(signal) => console.log('Signal clicked:', signal)}
                                height={400}
                                enableFiltering={false}
                                enableSorting={false}
                            />
                        </CardContent>
                    </DashboardCard>
                </Grid>
            </Grid>

            {/* Loading indicator */}
            {refreshing && (
                <LinearProgress
                    sx={{
                        position: 'fixed',
                        top: 0,
                        left: 0,
                        right: 0,
                        zIndex: 1400,
                    }}
                />
            )}
        </DashboardContainer>
    );
};

export default UnifiedDashboard; 