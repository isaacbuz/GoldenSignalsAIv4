import React, { useState, useEffect } from 'react';
import {
    Box,
    Card,
    CardContent,
    Typography,
    Grid,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    Chip,
    LinearProgress,
    IconButton,
    Tooltip,
    Tabs,
    Tab,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Button,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    CircularProgress,
    Alert,
    Badge,
    Divider,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    Remove,
    Refresh,
    Assessment,
    Speed,
    CheckCircle,
    Cancel,
    ExpandMore,
    Info,
    Timeline,
    Psychology,
    Analytics,
    BubbleChart,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { Line, Bar, Doughnut, Radar } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    RadialLinearScale,
    Title,
    Tooltip as ChartTooltip,
    Legend,
    Filler,
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    RadialLinearScale,
    Title,
    ChartTooltip,
    Legend,
    Filler
);

interface AgentSignal {
    agent_name: string;
    action: 'BUY' | 'SELL' | 'HOLD' | 'NEUTRAL';
    confidence: number;
    reasoning: string;
}

interface Signal {
    symbol: string;
    action: string;
    confidence: number;
    timestamp: string;
    metadata: {
        reasoning: string;
        agent_breakdown: Record<string, AgentSignal>;
        total_agents: number;
        consensus_details?: {
            buy_weight: number;
            sell_weight: number;
            hold_weight: number;
            agreement_score: number;
        };
    };
}

interface AgentPerformance {
    agent_name: string;
    total_signals: number;
    correct_signals: number;
    accuracy: number;
    avg_confidence: number;
    avg_execution_time?: number;
    last_updated: string;
    signal_history?: number[];
}

interface PerformanceMetrics {
    agents: Record<string, AgentPerformance>;
    summary: {
        total_agents: number;
        total_signals: number;
        avg_accuracy: number;
        total_correct: number;
        phase_1_agents?: number;
        phase_2_agents?: number;
        phase_3_agents?: number;
        phase_4_agents?: number;
    };
    timestamp: string;
}

const AgentPerformanceDashboard: React.FC = () => {
    const theme = useTheme();
    const [activeTab, setActiveTab] = useState(0);
    const [signals, setSignals] = useState<Signal[]>([]);
    const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics | null>(null);
    const [selectedTimeframe, setSelectedTimeframe] = useState('24h');
    const [selectedPhase, setSelectedPhase] = useState('all');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [expandedSignal, setExpandedSignal] = useState<string | null>(null);

    // Agent phases mapping
    const agentPhases: Record<string, string[]> = {
        phase_1: ['rsi', 'macd', 'volume_spike', 'ma_crossover'],
        phase_2: ['bollinger', 'stochastic', 'ema', 'atr', 'vwap'],
        phase_3: ['ichimoku', 'fibonacci', 'adx', 'parabolic_sar', 'std_dev'],
        phase_4: ['volume_profile', 'market_profile', 'order_flow', 'sentiment', 'options_flow'],
    };

    // Fetch signals and performance data
    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);

                // Fetch signals
                const signalsResponse = await fetch('/api/v1/signals/all');
                const signalsData = await signalsResponse.json();
                setSignals(signalsData.signals || []);

                // Fetch performance metrics
                const performanceResponse = await fetch('/api/v1/agents/performance');
                const performanceData = await performanceResponse.json();
                setPerformanceMetrics(performanceData);

                setError(null);
            } catch (err) {
                setError('Failed to fetch data');
                console.error('Error fetching data:', err);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 30000); // Refresh every 30 seconds

        return () => clearInterval(interval);
    }, []);

    const getActionIcon = (action: string) => {
        switch (action.toUpperCase()) {
            case 'BUY':
                return <TrendingUp sx={{ color: theme.palette.success.main }} />;
            case 'SELL':
                return <TrendingDown sx={{ color: theme.palette.error.main }} />;
            default:
                return <Remove sx={{ color: theme.palette.text.secondary }} />;
        }
    };

    const getActionColor = (action: string) => {
        switch (action.toUpperCase()) {
            case 'BUY':
                return 'success';
            case 'SELL':
                return 'error';
            default:
                return 'default';
        }
    };

    const getPhaseColor = (phase: string) => {
        const colors = {
            phase_1: theme.palette.info.main,
            phase_2: theme.palette.success.main,
            phase_3: theme.palette.warning.main,
            phase_4: theme.palette.error.main,
        };
        return colors[phase] || theme.palette.grey[500];
    };

    const getAgentPhase = (agentName: string): string => {
        for (const [phase, agents] of Object.entries(agentPhases)) {
            if (agents.includes(agentName.toLowerCase())) {
                return phase;
            }
        }
        return 'unknown';
    };

    const filterAgentsByPhase = (agents: Record<string, AgentPerformance>) => {
        if (selectedPhase === 'all') return agents;

        const filtered: Record<string, AgentPerformance> = {};
        for (const [name, data] of Object.entries(agents)) {
            if (getAgentPhase(name) === selectedPhase) {
                filtered[name] = data;
            }
        }
        return filtered;
    };

    // Chart data for performance overview
    const performanceChartData = {
        labels: performanceMetrics ? Object.keys(filterAgentsByPhase(performanceMetrics.agents)) : [],
        datasets: [
            {
                label: 'Accuracy (%)',
                data: performanceMetrics
                    ? Object.values(filterAgentsByPhase(performanceMetrics.agents)).map(a => a.accuracy)
                    : [],
                backgroundColor: theme.palette.primary.main,
                borderColor: theme.palette.primary.dark,
                borderWidth: 1,
            },
        ],
    };

    // Radar chart for agent comparison
    const radarChartData = {
        labels: ['Accuracy', 'Confidence', 'Signal Count', 'Success Rate', 'Speed'],
        datasets: performanceMetrics
            ? Object.entries(filterAgentsByPhase(performanceMetrics.agents)).slice(0, 5).map(([name, data], index) => ({
                label: name,
                data: [
                    data.accuracy,
                    data.avg_confidence * 100,
                    Math.min(100, (data.total_signals / 10)), // Normalized
                    (data.correct_signals / Math.max(1, data.total_signals)) * 100,
                    data.avg_execution_time ? Math.max(0, 100 - data.avg_execution_time * 10) : 80,
                ],
                backgroundColor: `rgba(${index * 50}, ${100 + index * 30}, ${200 - index * 30}, 0.2)`,
                borderColor: `rgba(${index * 50}, ${100 + index * 30}, ${200 - index * 30}, 1)`,
                borderWidth: 2,
            }))
            : [],
    };

    // Signal distribution chart
    const signalDistributionData = {
        labels: ['BUY', 'SELL', 'HOLD/NEUTRAL'],
        datasets: [{
            data: signals.reduce((acc, signal) => {
                const action = signal.action.toUpperCase();
                if (action === 'BUY') acc[0]++;
                else if (action === 'SELL') acc[1]++;
                else acc[2]++;
                return acc;
            }, [0, 0, 0]),
            backgroundColor: [
                theme.palette.success.main,
                theme.palette.error.main,
                theme.palette.grey[500],
            ],
        }],
    };

    if (loading) {
        return (
            <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
                <CircularProgress />
            </Box>
        );
    }

    if (error) {
        return (
            <Alert severity="error" sx={{ m: 2 }}>
                {error}
            </Alert>
        );
    }

    return (
        <Box sx={{ p: 3 }}>
            {/* Header */}
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
                <Typography variant="h4" component="h1">
                    Agent Performance & Signal Analysis
                </Typography>
                <Box display="flex" gap={2}>
                    <FormControl size="small" sx={{ minWidth: 120 }}>
                        <InputLabel>Timeframe</InputLabel>
                        <Select
                            value={selectedTimeframe}
                            onChange={(e) => setSelectedTimeframe(e.target.value)}
                            label="Timeframe"
                        >
                            <MenuItem value="1h">1 Hour</MenuItem>
                            <MenuItem value="24h">24 Hours</MenuItem>
                            <MenuItem value="7d">7 Days</MenuItem>
                            <MenuItem value="30d">30 Days</MenuItem>
                        </Select>
                    </FormControl>
                    <FormControl size="small" sx={{ minWidth: 120 }}>
                        <InputLabel>Phase</InputLabel>
                        <Select
                            value={selectedPhase}
                            onChange={(e) => setSelectedPhase(e.target.value)}
                            label="Phase"
                        >
                            <MenuItem value="all">All Phases</MenuItem>
                            <MenuItem value="phase_1">Phase 1</MenuItem>
                            <MenuItem value="phase_2">Phase 2</MenuItem>
                            <MenuItem value="phase_3">Phase 3</MenuItem>
                            <MenuItem value="phase_4">Phase 4</MenuItem>
                        </Select>
                    </FormControl>
                    <Tooltip title="Refresh data">
                        <IconButton onClick={() => window.location.reload()}>
                            <Refresh />
                        </IconButton>
                    </Tooltip>
                </Box>
            </Box>

            {/* Summary Cards */}
            <Grid container spacing={3} mb={3}>
                <Grid item xs={12} sm={6} md={3}>
                    <Card>
                        <CardContent>
                            <Box display="flex" alignItems="center" justifyContent="space-between">
                                <Box>
                                    <Typography color="textSecondary" gutterBottom>
                                        Total Agents
                                    </Typography>
                                    <Typography variant="h4">
                                        {performanceMetrics?.summary.total_agents || 0}
                                    </Typography>
                                    <Typography variant="body2" color="textSecondary">
                                        Across 4 phases
                                    </Typography>
                                </Box>
                                <Psychology sx={{ fontSize: 40, color: theme.palette.primary.main }} />
                            </Box>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <Card>
                        <CardContent>
                            <Box display="flex" alignItems="center" justifyContent="space-between">
                                <Box>
                                    <Typography color="textSecondary" gutterBottom>
                                        Total Signals
                                    </Typography>
                                    <Typography variant="h4">
                                        {performanceMetrics?.summary.total_signals || 0}
                                    </Typography>
                                    <Typography variant="body2" color="textSecondary">
                                        {selectedTimeframe} timeframe
                                    </Typography>
                                </Box>
                                <Timeline sx={{ fontSize: 40, color: theme.palette.success.main }} />
                            </Box>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <Card>
                        <CardContent>
                            <Box display="flex" alignItems="center" justifyContent="space-between">
                                <Box>
                                    <Typography color="textSecondary" gutterBottom>
                                        Avg Accuracy
                                    </Typography>
                                    <Typography variant="h4">
                                        {performanceMetrics?.summary.avg_accuracy?.toFixed(1) || 0}%
                                    </Typography>
                                    <Typography variant="body2" color="textSecondary">
                                        All agents
                                    </Typography>
                                </Box>
                                <Assessment sx={{ fontSize: 40, color: theme.palette.warning.main }} />
                            </Box>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <Card>
                        <CardContent>
                            <Box display="flex" alignItems="center" justifyContent="space-between">
                                <Box>
                                    <Typography color="textSecondary" gutterBottom>
                                        Success Rate
                                    </Typography>
                                    <Typography variant="h4">
                                        {performanceMetrics?.summary.total_correct && performanceMetrics?.summary.total_signals
                                            ? ((performanceMetrics.summary.total_correct / performanceMetrics.summary.total_signals) * 100).toFixed(1)
                                            : 0}%
                                    </Typography>
                                    <Typography variant="body2" color="textSecondary">
                                        Correct predictions
                                    </Typography>
                                </Box>
                                <CheckCircle sx={{ fontSize: 40, color: theme.palette.success.main }} />
                            </Box>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>

            {/* Tabs */}
            <Paper sx={{ mb: 3 }}>
                <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
                    <Tab label="Agent Performance" />
                    <Tab label="Recent Signals" />
                    <Tab label="Analytics" />
                    <Tab label="Phase Breakdown" />
                </Tabs>
            </Paper>

            {/* Tab Content */}
            {activeTab === 0 && (
                <Grid container spacing={3}>
                    {/* Performance Table */}
                    <Grid item xs={12}>
                        <Card>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    Agent Performance Metrics
                                </Typography>
                                <TableContainer>
                                    <Table size="small">
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Agent</TableCell>
                                                <TableCell>Phase</TableCell>
                                                <TableCell align="right">Signals</TableCell>
                                                <TableCell align="right">Correct</TableCell>
                                                <TableCell align="right">Accuracy</TableCell>
                                                <TableCell align="right">Avg Confidence</TableCell>
                                                <TableCell align="right">Status</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {performanceMetrics && Object.entries(filterAgentsByPhase(performanceMetrics.agents)).map(([name, data]) => {
                                                const phase = getAgentPhase(name);
                                                return (
                                                    <TableRow key={name}>
                                                        <TableCell>
                                                            <Typography variant="body2" fontWeight="medium">
                                                                {name}
                                                            </Typography>
                                                        </TableCell>
                                                        <TableCell>
                                                            <Chip
                                                                label={phase.replace('_', ' ').toUpperCase()}
                                                                size="small"
                                                                sx={{
                                                                    backgroundColor: getPhaseColor(phase),
                                                                    color: 'white',
                                                                }}
                                                            />
                                                        </TableCell>
                                                        <TableCell align="right">{data.total_signals}</TableCell>
                                                        <TableCell align="right">{data.correct_signals}</TableCell>
                                                        <TableCell align="right">
                                                            <Box display="flex" alignItems="center" justifyContent="flex-end">
                                                                <Typography variant="body2" mr={1}>
                                                                    {data.accuracy.toFixed(1)}%
                                                                </Typography>
                                                                <LinearProgress
                                                                    variant="determinate"
                                                                    value={data.accuracy}
                                                                    sx={{ width: 60, height: 6, borderRadius: 3 }}
                                                                />
                                                            </Box>
                                                        </TableCell>
                                                        <TableCell align="right">
                                                            {(data.avg_confidence * 100).toFixed(1)}%
                                                        </TableCell>
                                                        <TableCell align="right">
                                                            <Chip
                                                                icon={<CheckCircle />}
                                                                label="Active"
                                                                size="small"
                                                                color="success"
                                                                variant="outlined"
                                                            />
                                                        </TableCell>
                                                    </TableRow>
                                                );
                                            })}
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                            </CardContent>
                        </Card>
                    </Grid>

                    {/* Performance Chart */}
                    <Grid item xs={12} md={6}>
                        <Card>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    Accuracy by Agent
                                </Typography>
                                <Box height={300}>
                                    <Bar
                                        data={performanceChartData}
                                        options={{
                                            responsive: true,
                                            maintainAspectRatio: false,
                                            scales: {
                                                y: {
                                                    beginAtZero: true,
                                                    max: 100,
                                                },
                                            },
                                        }}
                                    />
                                </Box>
                            </CardContent>
                        </Card>
                    </Grid>

                    {/* Radar Chart */}
                    <Grid item xs={12} md={6}>
                        <Card>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    Agent Comparison
                                </Typography>
                                <Box height={300}>
                                    <Radar
                                        data={radarChartData}
                                        options={{
                                            responsive: true,
                                            maintainAspectRatio: false,
                                            scales: {
                                                r: {
                                                    beginAtZero: true,
                                                    max: 100,
                                                },
                                            },
                                        }}
                                    />
                                </Box>
                            </CardContent>
                        </Card>
                    </Grid>
                </Grid>
            )}

            {activeTab === 1 && (
                <Grid container spacing={3}>
                    {/* Recent Signals */}
                    <Grid item xs={12}>
                        <Card>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    Recent Trading Signals
                                </Typography>
                                {signals.map((signal, index) => (
                                    <Accordion
                                        key={`${signal.symbol}-${index}`}
                                        expanded={expandedSignal === `${signal.symbol}-${index}`}
                                        onChange={() => setExpandedSignal(
                                            expandedSignal === `${signal.symbol}-${index}` ? null : `${signal.symbol}-${index}`
                                        )}
                                    >
                                        <AccordionSummary expandIcon={<ExpandMore />}>
                                            <Box display="flex" alignItems="center" width="100%" gap={2}>
                                                {getActionIcon(signal.action)}
                                                <Typography variant="h6">{signal.symbol}</Typography>
                                                <Chip
                                                    label={signal.action}
                                                    color={getActionColor(signal.action) as any}
                                                    size="small"
                                                />
                                                <Box flexGrow={1} />
                                                <Typography variant="body2" color="textSecondary">
                                                    Confidence: {(signal.confidence * 100).toFixed(1)}%
                                                </Typography>
                                                <Typography variant="caption" color="textSecondary">
                                                    {new Date(signal.timestamp).toLocaleString()}
                                                </Typography>
                                            </Box>
                                        </AccordionSummary>
                                        <AccordionDetails>
                                            <Box>
                                                <Typography variant="body2" paragraph>
                                                    <strong>Reasoning:</strong> {signal.metadata.reasoning}
                                                </Typography>

                                                {signal.metadata.consensus_details && (
                                                    <Box mb={2}>
                                                        <Typography variant="subtitle2" gutterBottom>
                                                            Consensus Details
                                                        </Typography>
                                                        <Grid container spacing={2}>
                                                            <Grid item xs={3}>
                                                                <Typography variant="caption" color="textSecondary">
                                                                    Buy Weight
                                                                </Typography>
                                                                <Typography variant="body2">
                                                                    {signal.metadata.consensus_details.buy_weight.toFixed(2)}
                                                                </Typography>
                                                            </Grid>
                                                            <Grid item xs={3}>
                                                                <Typography variant="caption" color="textSecondary">
                                                                    Sell Weight
                                                                </Typography>
                                                                <Typography variant="body2">
                                                                    {signal.metadata.consensus_details.sell_weight.toFixed(2)}
                                                                </Typography>
                                                            </Grid>
                                                            <Grid item xs={3}>
                                                                <Typography variant="caption" color="textSecondary">
                                                                    Hold Weight
                                                                </Typography>
                                                                <Typography variant="body2">
                                                                    {signal.metadata.consensus_details.hold_weight.toFixed(2)}
                                                                </Typography>
                                                            </Grid>
                                                            <Grid item xs={3}>
                                                                <Typography variant="caption" color="textSecondary">
                                                                    Agreement Score
                                                                </Typography>
                                                                <Typography variant="body2">
                                                                    {(signal.metadata.consensus_details.agreement_score * 100).toFixed(1)}%
                                                                </Typography>
                                                            </Grid>
                                                        </Grid>
                                                    </Box>
                                                )}

                                                <Divider sx={{ my: 2 }} />

                                                <Typography variant="subtitle2" gutterBottom>
                                                    Agent Breakdown ({Object.keys(signal.metadata.agent_breakdown).length} agents)
                                                </Typography>
                                                <Grid container spacing={1}>
                                                    {Object.entries(signal.metadata.agent_breakdown).map(([agentName, agentData]) => (
                                                        <Grid item xs={12} sm={6} md={4} key={agentName}>
                                                            <Paper variant="outlined" sx={{ p: 1 }}>
                                                                <Box display="flex" alignItems="center" justifyContent="space-between">
                                                                    <Typography variant="caption" fontWeight="medium">
                                                                        {agentName}
                                                                    </Typography>
                                                                    <Chip
                                                                        label={getAgentPhase(agentName).replace('_', ' ')}
                                                                        size="small"
                                                                        sx={{
                                                                            backgroundColor: getPhaseColor(getAgentPhase(agentName)),
                                                                            color: 'white',
                                                                            fontSize: '0.65rem',
                                                                            height: 20,
                                                                        }}
                                                                    />
                                                                </Box>
                                                                <Box display="flex" alignItems="center" gap={1} mt={0.5}>
                                                                    {getActionIcon(agentData.action)}
                                                                    <Typography variant="body2">
                                                                        {agentData.action}
                                                                    </Typography>
                                                                    <Typography variant="caption" color="textSecondary">
                                                                        ({(agentData.confidence * 100).toFixed(0)}%)
                                                                    </Typography>
                                                                </Box>
                                                                {agentData.reasoning && (
                                                                    <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mt: 0.5 }}>
                                                                        {agentData.reasoning.substring(0, 50)}...
                                                                    </Typography>
                                                                )}
                                                            </Paper>
                                                        </Grid>
                                                    ))}
                                                </Grid>
                                            </Box>
                                        </AccordionDetails>
                                    </Accordion>
                                ))}
                            </CardContent>
                        </Card>
                    </Grid>
                </Grid>
            )}

            {activeTab === 2 && (
                <Grid container spacing={3}>
                    {/* Signal Distribution */}
                    <Grid item xs={12} md={4}>
                        <Card>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    Signal Distribution
                                </Typography>
                                <Box height={300}>
                                    <Doughnut
                                        data={signalDistributionData}
                                        options={{
                                            responsive: true,
                                            maintainAspectRatio: false,
                                            plugins: {
                                                legend: {
                                                    position: 'bottom',
                                                },
                                            },
                                        }}
                                    />
                                </Box>
                            </CardContent>
                        </Card>
                    </Grid>

                    {/* Performance Timeline */}
                    <Grid item xs={12} md={8}>
                        <Card>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    Performance Timeline
                                </Typography>
                                <Box height={300}>
                                    <Line
                                        data={{
                                            labels: ['1h ago', '2h ago', '3h ago', '4h ago', '5h ago', '6h ago'],
                                            datasets: [
                                                {
                                                    label: 'Accuracy',
                                                    data: [75, 78, 76, 79, 77, 80],
                                                    borderColor: theme.palette.primary.main,
                                                    backgroundColor: theme.palette.primary.light,
                                                    tension: 0.4,
                                                },
                                                {
                                                    label: 'Signal Count',
                                                    data: [12, 15, 14, 18, 16, 20],
                                                    borderColor: theme.palette.secondary.main,
                                                    backgroundColor: theme.palette.secondary.light,
                                                    tension: 0.4,
                                                    yAxisID: 'y1',
                                                },
                                            ],
                                        }}
                                        options={{
                                            responsive: true,
                                            maintainAspectRatio: false,
                                            scales: {
                                                y: {
                                                    type: 'linear',
                                                    display: true,
                                                    position: 'left',
                                                    title: {
                                                        display: true,
                                                        text: 'Accuracy (%)',
                                                    },
                                                },
                                                y1: {
                                                    type: 'linear',
                                                    display: true,
                                                    position: 'right',
                                                    title: {
                                                        display: true,
                                                        text: 'Signal Count',
                                                    },
                                                    grid: {
                                                        drawOnChartArea: false,
                                                    },
                                                },
                                            },
                                        }}
                                    />
                                </Box>
                            </CardContent>
                        </Card>
                    </Grid>

                    {/* Top Performing Agents */}
                    <Grid item xs={12}>
                        <Card>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    Top Performing Agents
                                </Typography>
                                <Grid container spacing={2}>
                                    {performanceMetrics && Object.entries(performanceMetrics.agents)
                                        .sort((a, b) => b[1].accuracy - a[1].accuracy)
                                        .slice(0, 5)
                                        .map(([name, data], index) => (
                                            <Grid item xs={12} sm={6} md={2.4} key={name}>
                                                <Paper
                                                    variant="outlined"
                                                    sx={{
                                                        p: 2,
                                                        textAlign: 'center',
                                                        borderColor: index === 0 ? theme.palette.warning.main : undefined,
                                                        borderWidth: index === 0 ? 2 : 1,
                                                    }}
                                                >
                                                    {index === 0 && (
                                                        <Badge
                                                            badgeContent="👑"
                                                            sx={{ mb: 1 }}
                                                        >
                                                            <Typography variant="h6">
                                                                {name}
                                                            </Typography>
                                                        </Badge>
                                                    )}
                                                    {index !== 0 && (
                                                        <Typography variant="h6">
                                                            {name}
                                                        </Typography>
                                                    )}
                                                    <Typography variant="h4" color="primary">
                                                        {data.accuracy.toFixed(1)}%
                                                    </Typography>
                                                    <Typography variant="caption" color="textSecondary">
                                                        {data.total_signals} signals
                                                    </Typography>
                                                </Paper>
                                            </Grid>
                                        ))}
                                </Grid>
                            </CardContent>
                        </Card>
                    </Grid>
                </Grid>
            )}

            {activeTab === 3 && (
                <Grid container spacing={3}>
                    {/* Phase Breakdown */}
                    {['phase_1', 'phase_2', 'phase_3', 'phase_4'].map((phase) => {
                        const phaseAgents = performanceMetrics
                            ? Object.entries(performanceMetrics.agents).filter(([name]) => getAgentPhase(name) === phase)
                            : [];

                        const phaseStats = phaseAgents.reduce(
                            (acc, [_, data]) => ({
                                totalSignals: acc.totalSignals + data.total_signals,
                                totalCorrect: acc.totalCorrect + data.correct_signals,
                                avgAccuracy: acc.avgAccuracy + data.accuracy,
                                avgConfidence: acc.avgConfidence + data.avg_confidence,
                            }),
                            { totalSignals: 0, totalCorrect: 0, avgAccuracy: 0, avgConfidence: 0 }
                        );

                        if (phaseAgents.length > 0) {
                            phaseStats.avgAccuracy /= phaseAgents.length;
                            phaseStats.avgConfidence /= phaseAgents.length;
                        }

                        return (
                            <Grid item xs={12} md={6} key={phase}>
                                <Card>
                                    <CardContent>
                                        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                                            <Typography variant="h6">
                                                {phase.replace('_', ' ').toUpperCase()}
                                            </Typography>
                                            <Chip
                                                label={`${phaseAgents.length} agents`}
                                                sx={{
                                                    backgroundColor: getPhaseColor(phase),
                                                    color: 'white',
                                                }}
                                            />
                                        </Box>

                                        <Grid container spacing={2} mb={2}>
                                            <Grid item xs={6}>
                                                <Typography variant="caption" color="textSecondary">
                                                    Total Signals
                                                </Typography>
                                                <Typography variant="h6">
                                                    {phaseStats.totalSignals}
                                                </Typography>
                                            </Grid>
                                            <Grid item xs={6}>
                                                <Typography variant="caption" color="textSecondary">
                                                    Avg Accuracy
                                                </Typography>
                                                <Typography variant="h6">
                                                    {phaseStats.avgAccuracy.toFixed(1)}%
                                                </Typography>
                                            </Grid>
                                        </Grid>

                                        <Divider sx={{ my: 1 }} />

                                        <Typography variant="subtitle2" gutterBottom>
                                            Agents
                                        </Typography>
                                        {phaseAgents.map(([name, data]) => (
                                            <Box key={name} display="flex" alignItems="center" justifyContent="space-between" py={0.5}>
                                                <Typography variant="body2">{name}</Typography>
                                                <Box display="flex" alignItems="center" gap={1}>
                                                    <Typography variant="caption" color="textSecondary">
                                                        {data.accuracy.toFixed(1)}%
                                                    </Typography>
                                                    <LinearProgress
                                                        variant="determinate"
                                                        value={data.accuracy}
                                                        sx={{ width: 60, height: 4 }}
                                                    />
                                                </Box>
                                            </Box>
                                        ))}
                                    </CardContent>
                                </Card>
                            </Grid>
                        );
                    })}
                </Grid>
            )}
        </Box>
    );
};

export default AgentPerformanceDashboard; 