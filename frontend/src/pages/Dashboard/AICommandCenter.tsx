/**
 * AI Command Center - The Brain of GoldenSignalsAI
 * Real-time monitoring and control of all AI agents and trading operations
 */

import React, { useState, useEffect } from 'react';
import {
    Box,
    Container,
    Grid,
    Paper,
    Typography,
    Card,
    CardContent,
    Stack,
    Chip,
    IconButton,
    Button,
    LinearProgress,
    CircularProgress,
    useTheme,
    alpha,
    Tooltip,
    Badge,
    Avatar,
    AvatarGroup,
    Divider,
    Switch,
    FormControlLabel,
    Alert,
    Collapse,
} from '@mui/material';
import {
    Psychology,
    TrendingUp,
    Speed,
    Timeline,
    AutoAwesome,
    Visibility,
    Settings,
    PlayArrow,
    Pause,
    Refresh,
    Warning,
    CheckCircle,
    Error,
    BoltOutlined as Bolt,
    Analytics,
    SmartToy,
    Memory,
    CloudSync,
    Security,
    MonitorHeart,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation } from '@tanstack/react-query';
import { apiClient } from '../../services/api';
import { useWebSocket } from '../../services/websocket';
import { AgentStatusCard } from '../../components/AI/AgentStatusCard';
import { LivePerformanceChart } from '../../components/AI/LivePerformanceChart';
import { SignalFeed } from '../../components/AI/SignalFeed';
import { AIInsightsPanel } from '../../components/AI/AIInsightsPanel';

interface AgentStatus {
    id: string;
    name: string;
    type: 'technical' | 'sentiment' | 'options' | 'risk' | 'ml';
    status: 'active' | 'idle' | 'error' | 'processing';
    performance: {
        accuracy: number;
        signalsGenerated: number;
        profitability: number;
        lastSignal?: Date;
    };
    health: {
        cpu: number;
        memory: number;
        latency: number;
    };
}

interface SystemMetrics {
    totalAgents: number;
    activeAgents: number;
    signalsPerMinute: number;
    avgAccuracy: number;
    systemLoad: number;
    dataLatency: number;
}

export const AICommandCenter: React.FC = () => {
    const theme = useTheme();
    const { subscribeToAgent, requestAgentStatus } = useWebSocket();

    const [autoTrading, setAutoTrading] = useState(false);
    const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [agentStatuses, setAgentStatuses] = useState<Record<string, AgentStatus>>({});

    // Fetch system metrics
    const { data: metrics, refetch: refetchMetrics } = useQuery({
        queryKey: ['system-metrics'],
        queryFn: () => apiClient.getSystemMetrics(),
        refetchInterval: 5000,
    });

    // Fetch agent list
    const { data: agents = [] } = useQuery({
        queryKey: ['agents'],
        queryFn: () => apiClient.getAgents(),
        refetchInterval: 10000,
    });

    // Subscribe to real-time agent updates
    useEffect(() => {
        agents.forEach(agent => {
            subscribeToAgent(agent.id);
        });

        return () => {
            agents.forEach(agent => {
                // Unsubscribe on cleanup
            });
        };
    }, [agents]);

    // Toggle auto-trading
    const toggleAutoTrading = useMutation({
        mutationFn: (enabled: boolean) => apiClient.setAutoTrading(enabled),
        onSuccess: (data, enabled) => {
            setAutoTrading(enabled);
        },
    });

    const getAgentIcon = (type: string) => {
        switch (type) {
            case 'technical': return <Timeline />;
            case 'sentiment': return <Psychology />;
            case 'options': return <TrendingUp />;
            case 'risk': return <Security />;
            case 'ml': return <Memory />;
            default: return <SmartToy />;
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'active': return theme.palette.success.main;
            case 'processing': return theme.palette.info.main;
            case 'idle': return theme.palette.warning.main;
            case 'error': return theme.palette.error.main;
            default: return theme.palette.grey[500];
        }
    };

    return (
        <Container maxWidth={false} sx={{ py: 3 }}>
            {/* Header */}
            <Box sx={{ mb: 4 }}>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                    <Box>
                        <Typography variant="h4" fontWeight="bold" gutterBottom>
                            AI Command Center
                        </Typography>
                        <Typography variant="body1" color="text.secondary">
                            Real-time monitoring and control of your AI trading fleet
                        </Typography>
                    </Box>
                    <Stack direction="row" spacing={2} alignItems="center">
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={autoTrading}
                                    onChange={(e) => setAutoTrading(e.target.checked)}
                                    color="success"
                                />
                            }
                            label={
                                <Stack direction="row" spacing={1} alignItems="center">
                                    <Bolt />
                                    <Typography>Auto Trading</Typography>
                                </Stack>
                            }
                        />
                        <Button
                            variant="outlined"
                            startIcon={showAdvanced ? <Visibility /> : <Settings />}
                            onClick={() => setShowAdvanced(!showAdvanced)}
                        >
                            {showAdvanced ? 'Simple View' : 'Advanced'}
                        </Button>
                        <IconButton onClick={() => requestAgentStatus()}>
                            <Refresh />
                        </IconButton>
                    </Stack>
                </Stack>
            </Box>

            {/* Main Metrics Grid */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} md={3}>
                    <Card sx={{
                        background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.1)} 0%, ${alpha(theme.palette.primary.dark, 0.1)} 100%)`,
                        border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
                    }}>
                        <CardContent>
                            <Stack spacing={1}>
                                <Stack direction="row" justifyContent="space-between" alignItems="center">
                                    <Typography color="text.secondary" variant="body2">
                                        Active Agents
                                    </Typography>
                                    <MonitorHeart sx={{ color: theme.palette.primary.main }} />
                                </Stack>
                                <Typography variant="h3" fontWeight="bold">
                                    3/4
                                </Typography>
                                <LinearProgress
                                    variant="determinate"
                                    value={75}
                                    sx={{ height: 6, borderRadius: 3 }}
                                />
                            </Stack>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} md={3}>
                    <Card sx={{
                        background: `linear-gradient(135deg, ${alpha(theme.palette.success.main, 0.1)} 0%, ${alpha(theme.palette.success.dark, 0.1)} 100%)`,
                        border: `1px solid ${alpha(theme.palette.success.main, 0.2)}`,
                    }}>
                        <CardContent>
                            <Stack spacing={1}>
                                <Stack direction="row" justifyContent="space-between" alignItems="center">
                                    <Typography color="text.secondary" variant="body2">
                                        Signals/Min
                                    </Typography>
                                    <Speed sx={{ color: theme.palette.success.main }} />
                                </Stack>
                                <Typography variant="h3" fontWeight="bold">
                                    12
                                </Typography>
                                <Chip
                                    label="+12% from last hour"
                                    size="small"
                                    color="success"
                                    sx={{ alignSelf: 'flex-start' }}
                                />
                            </Stack>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} md={3}>
                    <Card sx={{
                        background: `linear-gradient(135deg, ${alpha(theme.palette.info.main, 0.1)} 0%, ${alpha(theme.palette.info.dark, 0.1)} 100%)`,
                        border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`,
                    }}>
                        <CardContent>
                            <Stack spacing={1}>
                                <Stack direction="row" justifyContent="space-between" alignItems="center">
                                    <Typography color="text.secondary" variant="body2">
                                        Avg Accuracy
                                    </Typography>
                                    <Analytics sx={{ color: theme.palette.info.main }} />
                                </Stack>
                                <Typography variant="h3" fontWeight="bold">
                                    88.9%
                                </Typography>
                                <Stack direction="row" spacing={0.5}>
                                    {[85, 87, 89, 91, 88, 92].map((val, i) => (
                                        <Box
                                            key={i}
                                            sx={{
                                                width: 4,
                                                height: 20,
                                                bgcolor: val > 88 ? theme.palette.success.main : theme.palette.grey[400],
                                                borderRadius: 1,
                                            }}
                                        />
                                    ))}
                                </Stack>
                            </Stack>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} md={3}>
                    <Card sx={{
                        background: `linear-gradient(135deg, ${alpha(theme.palette.warning.main, 0.1)} 0%, ${alpha(theme.palette.warning.dark, 0.1)} 100%)`,
                        border: `1px solid ${alpha(theme.palette.warning.main, 0.2)}`,
                    }}>
                        <CardContent>
                            <Stack spacing={1}>
                                <Stack direction="row" justifyContent="space-between" alignItems="center">
                                    <Typography color="text.secondary" variant="body2">
                                        Data Latency
                                    </Typography>
                                    <CloudSync sx={{ color: theme.palette.warning.main }} />
                                </Stack>
                                <Typography variant="h3" fontWeight="bold">
                                    47ms
                                </Typography>
                                <Chip
                                    label="Excellent"
                                    size="small"
                                    color="success"
                                    sx={{ alignSelf: 'flex-start' }}
                                />
                            </Stack>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>

            {/* Agent Grid */}
            <Grid container spacing={3}>
                {/* Agent Status Panel */}
                <Grid item xs={12} lg={8}>
                    <Paper sx={{ p: 3, height: '100%' }}>
                        <Typography variant="h6" gutterBottom>
                            Agent Fleet Status
                        </Typography>
                        <Grid container spacing={2}>
                            {agents.map((agent) => (
                                <Grid item xs={12} sm={6} md={6} key={agent.id}>
                                    <Card
                                        sx={{
                                            cursor: 'pointer',
                                            border: selectedAgent === agent.id ? `2px solid ${theme.palette.primary.main}` : '1px solid transparent',
                                            transition: 'all 0.3s',
                                            '&:hover': {
                                                transform: 'translateY(-2px)',
                                                boxShadow: 4,
                                            }
                                        }}
                                        onClick={() => setSelectedAgent(agent.id)}
                                    >
                                        <CardContent>
                                            <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
                                                <Stack direction="row" spacing={1} alignItems="center">
                                                    <Avatar sx={{ bgcolor: alpha(getStatusColor(agent.status), 0.1) }}>
                                                        {getAgentIcon(agent.type)}
                                                    </Avatar>
                                                    <Box>
                                                        <Typography variant="subtitle1" fontWeight="bold">
                                                            {agent.name}
                                                        </Typography>
                                                        <Chip
                                                            label={agent.status}
                                                            size="small"
                                                            sx={{
                                                                bgcolor: alpha(getStatusColor(agent.status), 0.1),
                                                                color: getStatusColor(agent.status),
                                                                fontWeight: 'bold',
                                                            }}
                                                        />
                                                    </Box>
                                                </Stack>
                                                <IconButton size="small">
                                                    {agent.status === 'active' ? <Pause /> : <PlayArrow />}
                                                </IconButton>
                                            </Stack>

                                            <Grid container spacing={1}>
                                                <Grid item xs={4}>
                                                    <Typography variant="caption" color="text.secondary">
                                                        Accuracy
                                                    </Typography>
                                                    <Typography variant="body2" fontWeight="bold">
                                                        {agent.performance.accuracy}%
                                                    </Typography>
                                                </Grid>
                                                <Grid item xs={4}>
                                                    <Typography variant="caption" color="text.secondary">
                                                        Signals
                                                    </Typography>
                                                    <Typography variant="body2" fontWeight="bold">
                                                        {agent.performance.signalsGenerated}
                                                    </Typography>
                                                </Grid>
                                                <Grid item xs={4}>
                                                    <Typography variant="caption" color="text.secondary">
                                                        CPU
                                                    </Typography>
                                                    <Typography variant="body2" fontWeight="bold">
                                                        {agent.health.cpu}%
                                                    </Typography>
                                                </Grid>
                                            </Grid>
                                        </CardContent>
                                    </Card>
                                </Grid>
                            ))}
                        </Grid>
                    </Paper>
                </Grid>

                {/* Live Signal Feed */}
                <Grid item xs={12} lg={4}>
                    <Paper sx={{ p: 3, height: '100%', maxHeight: 600, overflow: 'auto' }}>
                        <Typography variant="h6" gutterBottom>
                            Live Signal Feed
                        </Typography>
                        <Stack spacing={2}>
                            {[
                                { symbol: 'AAPL', type: 'CALL', confidence: 92, time: '2m ago' },
                                { symbol: 'TSLA', type: 'PUT', confidence: 87, time: '5m ago' },
                                { symbol: 'SPY', type: 'CALL', confidence: 94, time: '8m ago' },
                                { symbol: 'NVDA', type: 'CALL', confidence: 89, time: '12m ago' },
                            ].map((signal, index) => (
                                <Card key={index} variant="outlined">
                                    <CardContent sx={{ py: 1.5 }}>
                                        <Stack direction="row" justifyContent="space-between" alignItems="center">
                                            <Stack direction="row" spacing={1} alignItems="center">
                                                <Chip
                                                    label={signal.symbol}
                                                    size="small"
                                                    sx={{ fontWeight: 'bold' }}
                                                />
                                                <Chip
                                                    label={signal.type}
                                                    size="small"
                                                    color={signal.type === 'CALL' ? 'success' : 'error'}
                                                />
                                            </Stack>
                                            <Stack alignItems="flex-end">
                                                <Typography variant="body2" fontWeight="bold">
                                                    {signal.confidence}%
                                                </Typography>
                                                <Typography variant="caption" color="text.secondary">
                                                    {signal.time}
                                                </Typography>
                                            </Stack>
                                        </Stack>
                                    </CardContent>
                                </Card>
                            ))}
                        </Stack>
                    </Paper>
                </Grid>
            </Grid>
        </Container>
    );
};

export default AICommandCenter; 