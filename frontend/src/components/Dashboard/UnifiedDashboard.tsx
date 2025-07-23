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
import GridLayout from 'react-grid-layout';

// Import existing components
import SignalCard from '../SignalCard/SignalCard';
import RealTimeFeed from '../RealTimeFeed/RealTimeFeed';
import { FloatingOrbAssistant } from '../FloatingOrbAssistant/FloatingOrbAssistant';
import ProfessionalChart from '../ProfessionalChart/ProfessionalChart';
import { List as VirtualizedList } from 'react-virtualized';
import logger from '../../services/logger';


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
    const [timeframe, setTimeframe] = useState('1d');
    const [chartData, setChartData] = useState<any[]>([]);
    const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
    const [layout, setLayout] = useState([
        { i: 'signals', x: 0, y: 0, w: 6, h: 10 },
        { i: 'professionalChart', x: 6, y: 0, w: 6, h: 8 },
        { i: 'realtime', x: 0, y: 10, w: 12, h: 4 },
    ]);

    const handleRefresh = useCallback(async () => {
        setRefreshing(true);
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));
        setRefreshing(false);
    }, []);

    const handleLayoutChange = (newLayout: any) => {
        setLayout(newLayout);
        localStorage.setItem('dashboardLayout', JSON.stringify(newLayout));
    };

    // Fetch chart data
    useEffect(() => {
        const fetchChartData = async () => {
            try {
                const response = await fetch(`http://localhost:8000/api/v1/market-data/${selectedSymbol}/history?period=30d&interval=1d`);
                if (response.ok) {
                    const data = await response.json();
                    setChartData(data.data || []);
                }
            } catch (error) {
                logger.error('Failed to fetch chart data:', error);
            }
        };

        fetchChartData();
    }, [selectedSymbol, timeframe]);

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
            <GridLayout className="layout" layout={layout} onLayoutChange={handleLayoutChange}>
                <div key="signals">
                    <DashboardCard>
                        <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                <SpeedIcon sx={{ mr: 1, color: 'primary.main' }} />
                                <Typography variant="h6" fontWeight="bold">
                                    Recent Signals
                                </Typography>
                            </Box>
                            <VirtualizedList
                                width={300}
                                height={400}
                                rowCount={mockSignals.length}
                                rowHeight={120}
                                rowRenderer={({ index, key, style }: { index: number, key: string, style: React.CSSProperties }) => (
                                    <div key={key} style={style}>
                                        <SignalCard signal={mockSignals[index]} />
                                    </div>
                                )}
                            />
                        </CardContent>
                    </DashboardCard>
                </div>
                <div key="realtime"><RealTimeFeed /></div>
                <div key="professionalChart">
                    <DashboardCard>
                        <ProfessionalChart
                            symbol={selectedSymbol}
                            timeframe={timeframe}
                        />
                    </DashboardCard>
                </div>
                {/* Add more widgets here */}
            </GridLayout>

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
            <FloatingOrbAssistant onClick={() => logger.info('Open AI Chat')} />
        </DashboardContainer>
    );
};

export default UnifiedDashboard;
