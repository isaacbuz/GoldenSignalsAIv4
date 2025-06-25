import React, { useState, useEffect } from 'react';
import {
    Box,
    Grid,
    Card,
    CardContent,
    Typography,
    Chip,
    LinearProgress,
    IconButton,
    Tooltip,
    Alert,
    Skeleton,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    AutoAwesome,
    Psychology,
    ShowChart,
    Speed,
    Refresh,
    MoreVert,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { utilityClasses } from '../../theme/goldenTheme';

// Styled components
const StyledCard = styled(Card)(({ theme }) => ({
    ...utilityClasses.glassmorphism,
    height: '100%',
    position: 'relative',
    overflow: 'hidden',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '4px',
        background: 'linear-gradient(90deg, #FFD700 0%, #FFA500 100%)',
    },
}));

const MetricCard = styled(Box)(({ theme }) => ({
    padding: theme.spacing(3),
    borderRadius: theme.shape.borderRadius,
    ...utilityClasses.glassmorphism,
    transition: 'all 0.3s ease',
    '&:hover': {
        transform: 'translateY(-4px)',
        boxShadow: '0 8px 24px rgba(255, 215, 0, 0.2)',
    },
}));

const SignalBadge = styled(Chip)(({ theme, signalType }: { theme: any; signalType: string }) => ({
    fontWeight: 700,
    borderRadius: 6,
    ...(signalType === 'BUY' && {
        backgroundColor: 'rgba(76, 175, 80, 0.1)',
        color: '#4CAF50',
        border: '1px solid rgba(76, 175, 80, 0.3)',
    }),
    ...(signalType === 'SELL' && {
        backgroundColor: 'rgba(244, 67, 54, 0.1)',
        color: '#F44336',
        border: '1px solid rgba(244, 67, 54, 0.3)',
    }),
    ...(signalType === 'HOLD' && {
        backgroundColor: 'rgba(255, 165, 0, 0.1)',
        color: '#FFA500',
        border: '1px solid rgba(255, 165, 0, 0.3)',
    }),
}));

const AgentCard = styled(Box)(({ theme, isActive }: { theme: any; isActive: boolean }) => ({
    padding: theme.spacing(2),
    borderRadius: theme.shape.borderRadius,
    border: `1px solid ${isActive ? 'rgba(255, 215, 0, 0.3)' : 'rgba(255, 255, 255, 0.1)'}`,
    backgroundColor: isActive ? 'rgba(255, 215, 0, 0.05)' : 'rgba(255, 255, 255, 0.02)',
    transition: 'all 0.3s ease',
    cursor: 'pointer',
    '&:hover': {
        borderColor: 'rgba(255, 215, 0, 0.5)',
        backgroundColor: 'rgba(255, 215, 0, 0.1)',
    },
}));

// Mock data interfaces
interface SignalMetrics {
    totalSignals: number;
    accuracy: number;
    activeAgents: number;
    consensusStrength: number;
}

interface RecentSignal {
    id: string;
    symbol: string;
    type: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    timestamp: string;
    agents: string[];
}

interface Agent {
    id: string;
    name: string;
    type: string;
    status: 'active' | 'idle' | 'processing';
    accuracy: number;
    lastSignal: string;
}

const AICommandCenter: React.FC = () => {
    const [metrics, setMetrics] = useState<SignalMetrics>({
        totalSignals: 1247,
        accuracy: 94.2,
        activeAgents: 8,
        consensusStrength: 87.5,
    });

    const [recentSignals, setRecentSignals] = useState<RecentSignal[]>([
        {
            id: '1',
            symbol: 'AAPL',
            type: 'BUY',
            confidence: 92.5,
            timestamp: '2 min ago',
            agents: ['sentiment', 'technical', 'flow'],
        },
        {
            id: '2',
            symbol: 'SPY',
            type: 'HOLD',
            confidence: 78.3,
            timestamp: '5 min ago',
            agents: ['risk', 'regime'],
        },
        {
            id: '3',
            symbol: 'TSLA',
            type: 'SELL',
            confidence: 85.7,
            timestamp: '8 min ago',
            agents: ['sentiment', 'technical', 'risk'],
        },
        {
            id: '4',
            symbol: 'NVDA',
            type: 'BUY',
            confidence: 94.1,
            timestamp: '12 min ago',
            agents: ['flow', 'technical', 'sentiment'],
        },
        {
            id: '5',
            symbol: 'BTC',
            type: 'BUY',
            confidence: 88.9,
            timestamp: '15 min ago',
            agents: ['sentiment', 'flow'],
        },
    ]);

    const [agents, setAgents] = useState<Agent[]>([
        { id: '1', name: 'Sentiment AI', type: 'sentiment', status: 'active', accuracy: 92.5, lastSignal: '1m ago' },
        { id: '2', name: 'Technical AI', type: 'technical', status: 'processing', accuracy: 94.8, lastSignal: '2m ago' },
        { id: '3', name: 'Flow AI', type: 'flow', status: 'active', accuracy: 96.2, lastSignal: '30s ago' },
        { id: '4', name: 'Risk AI', type: 'risk', status: 'active', accuracy: 91.7, lastSignal: '3m ago' },
        { id: '5', name: 'Regime AI', type: 'regime', status: 'idle', accuracy: 89.3, lastSignal: '5m ago' },
        { id: '6', name: 'Liquidity AI', type: 'liquidity', status: 'active', accuracy: 93.1, lastSignal: '1m ago' },
    ]);

    const [loading, setLoading] = useState(false);
    const [consensusData, setConsensusData] = useState({
        currentSignal: 'BUY',
        confidence: 87.5,
        agentsAgreeing: 6,
        totalAgents: 8,
        reasoning: 'Strong bullish sentiment with positive flow indicators',
    });

    const handleRefresh = () => {
        setLoading(true);
        setTimeout(() => {
            setLoading(false);
            // Simulate data refresh
            setMetrics(prev => ({
                ...prev,
                totalSignals: prev.totalSignals + Math.floor(Math.random() * 10),
                accuracy: 94.2 + (Math.random() * 2 - 1),
            }));
        }, 1000);
    };

    return (
        <Box>
            {/* Header */}
            <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                    <Typography variant="h4" sx={{ fontWeight: 700, ...utilityClasses.textGradient }}>
                        AI Command Center
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Real-time AI signal intelligence and multi-agent consensus
                    </Typography>
                </Box>
                <Tooltip title="Refresh data">
                    <IconButton onClick={handleRefresh} disabled={loading}>
                        <Refresh sx={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
                    </IconButton>
                </Tooltip>
            </Box>

            {/* Alert for high-confidence signal */}
            <Alert
                severity="success"
                icon={<AutoAwesome />}
                sx={{ mb: 3, ...utilityClasses.glassmorphism }}
            >
                <Typography variant="body2">
                    <strong>High Confidence Signal:</strong> NVDA BUY signal with 94.1% confidence from 3 agents
                </Typography>
            </Alert>

            {/* Metrics Grid */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} sm={6} md={3}>
                    <MetricCard>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Typography variant="overline" color="text.secondary">
                                Total Signals
                            </Typography>
                            <ShowChart sx={{ color: '#FFD700' }} />
                        </Box>
                        <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                            {loading ? <Skeleton width={100} /> : metrics.totalSignals.toLocaleString()}
                        </Typography>
                        <Typography variant="caption" sx={{ color: '#4CAF50' }}>
                            +12.5% from yesterday
                        </Typography>
                    </MetricCard>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <MetricCard>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Typography variant="overline" color="text.secondary">
                                Accuracy Rate
                            </Typography>
                            <Speed sx={{ color: '#FFD700' }} />
                        </Box>
                        <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                            {loading ? <Skeleton width={100} /> : `${metrics.accuracy.toFixed(1)}%`}
                        </Typography>
                        <LinearProgress
                            variant="determinate"
                            value={metrics.accuracy}
                            sx={{ height: 4, borderRadius: 2 }}
                        />
                    </MetricCard>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <MetricCard>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Typography variant="overline" color="text.secondary">
                                Active Agents
                            </Typography>
                            <Psychology sx={{ color: '#FFD700' }} />
                        </Box>
                        <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                            {loading ? <Skeleton width={100} /> : `${metrics.activeAgents}/9`}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            All systems operational
                        </Typography>
                    </MetricCard>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <MetricCard>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Typography variant="overline" color="text.secondary">
                                Consensus Strength
                            </Typography>
                            <AutoAwesome sx={{ color: '#FFD700' }} />
                        </Box>
                        <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                            {loading ? <Skeleton width={100} /> : `${metrics.consensusStrength}%`}
                        </Typography>
                        <Typography variant="caption" sx={{ color: '#4CAF50' }}>
                            Strong agreement
                        </Typography>
                    </MetricCard>
                </Grid>
            </Grid>

            {/* Main Content Grid */}
            <Grid container spacing={3}>
                {/* Live Consensus View */}
                <Grid item xs={12} md={6}>
                    <StyledCard>
                        <CardContent>
                            <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                                Live AI Consensus
                            </Typography>

                            <Box sx={{ textAlign: 'center', py: 3 }}>
                                <Box sx={{ mb: 3 }}>
                                    <SignalBadge
                                        label={consensusData.currentSignal}
                                        signalType={consensusData.currentSignal}
                                        sx={{ fontSize: '1.2rem', px: 3, py: 1 }}
                                    />
                                </Box>

                                <Typography variant="h3" sx={{ fontWeight: 700, mb: 1, ...utilityClasses.textGradient }}>
                                    {consensusData.confidence}%
                                </Typography>

                                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                                    {consensusData.agentsAgreeing} of {consensusData.totalAgents} agents agree
                                </Typography>

                                <Box sx={{
                                    p: 2,
                                    borderRadius: 2,
                                    backgroundColor: 'rgba(255, 215, 0, 0.05)',
                                    border: '1px solid rgba(255, 215, 0, 0.2)'
                                }}>
                                    <Typography variant="body2">
                                        <strong>AI Reasoning:</strong> {consensusData.reasoning}
                                    </Typography>
                                </Box>
                            </Box>
                        </CardContent>
                    </StyledCard>
                </Grid>

                {/* Agent Status Grid */}
                <Grid item xs={12} md={6}>
                    <StyledCard>
                        <CardContent>
                            <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                                Agent Status
                            </Typography>

                            <Grid container spacing={2}>
                                {agents.map((agent) => (
                                    <Grid item xs={6} key={agent.id}>
                                        <AgentCard isActive={agent.status === 'active'}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                                    {agent.name}
                                                </Typography>
                                                <Chip
                                                    label={agent.status}
                                                    size="small"
                                                    sx={{
                                                        height: 20,
                                                        fontSize: '0.7rem',
                                                        backgroundColor: agent.status === 'active' ? 'rgba(76, 175, 80, 0.1)' :
                                                            agent.status === 'processing' ? 'rgba(255, 165, 0, 0.1)' :
                                                                'rgba(255, 255, 255, 0.1)',
                                                        color: agent.status === 'active' ? '#4CAF50' :
                                                            agent.status === 'processing' ? '#FFA500' :
                                                                'rgba(255, 255, 255, 0.5)',
                                                    }}
                                                />
                                            </Box>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                                <Typography variant="caption" color="text.secondary">
                                                    Accuracy: {agent.accuracy}%
                                                </Typography>
                                                <Typography variant="caption" color="text.secondary">
                                                    {agent.lastSignal}
                                                </Typography>
                                            </Box>
                                        </AgentCard>
                                    </Grid>
                                ))}
                            </Grid>
                        </CardContent>
                    </StyledCard>
                </Grid>

                {/* Recent Signals Feed */}
                <Grid item xs={12}>
                    <StyledCard>
                        <CardContent>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
                                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                                    Recent Signals
                                </Typography>
                                <Chip
                                    label="LIVE"
                                    size="small"
                                    sx={{
                                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                                        color: '#4CAF50',
                                        animation: 'pulse 2s infinite'
                                    }}
                                />
                            </Box>

                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                {recentSignals.map((signal) => (
                                    <Box
                                        key={signal.id}
                                        sx={{
                                            p: 2,
                                            borderRadius: 2,
                                            border: '1px solid rgba(255, 255, 255, 0.1)',
                                            transition: 'all 0.3s ease',
                                            '&:hover': {
                                                borderColor: 'rgba(255, 215, 0, 0.3)',
                                                backgroundColor: 'rgba(255, 215, 0, 0.02)',
                                            }
                                        }}
                                    >
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                                                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                                                    {signal.symbol}
                                                </Typography>
                                                <SignalBadge label={signal.type} signalType={signal.type} />
                                                <Chip
                                                    label={`${signal.confidence}%`}
                                                    size="small"
                                                    sx={{ fontWeight: 600 }}
                                                />
                                            </Box>
                                            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                                                <Typography variant="caption" color="text.secondary">
                                                    {signal.agents.join(', ')}
                                                </Typography>
                                                <Typography variant="caption" color="text.secondary">
                                                    {signal.timestamp}
                                                </Typography>
                                                <IconButton size="small">
                                                    <MoreVert fontSize="small" />
                                                </IconButton>
                                            </Box>
                                        </Box>
                                    </Box>
                                ))}
                            </Box>
                        </CardContent>
                    </StyledCard>
                </Grid>
            </Grid>
        </Box>
    );
};

export default AICommandCenter; 