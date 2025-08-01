import React, { useState, useEffect } from 'react';
import {
    Box,
    Card,
    CardContent,
    Typography,
    Grid,
    LinearProgress,
    Chip,
    Avatar,
    List,
    ListItem,
    ListItemAvatar,
    ListItemText,
    IconButton,
    Tooltip,
} from '@mui/material';
import {
    TrendingUp as TrendingUpIcon,
    TrendingDown as TrendingDownIcon,
    Psychology as PsychologyIcon,
    Analytics as AnalyticsIcon,
    Speed as SpeedIcon,
    CheckCircle as CheckCircleIcon,
    Warning as WarningIcon,
    Error as ErrorIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

const ConsensusCard = styled(Card)(({ theme }) => ({
    background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
    border: '1px solid #FFD700',
    borderRadius: '12px',
    boxShadow: '0 8px 32px rgba(255, 215, 0, 0.1)',
    '&:hover': {
        boxShadow: '0 12px 48px rgba(255, 215, 0, 0.2)',
        transform: 'translateY(-2px)',
    },
    transition: 'all 0.3s ease',
}));

const AgentAvatar = styled(Avatar)(({ theme }) => ({
    background: 'linear-gradient(135deg, #FFD700 0%, #FFA500 100%)',
    color: '#000',
    fontWeight: 'bold',
}));

const ConsensusProgress = styled(LinearProgress)(({ theme }) => ({
    height: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    '& .MuiLinearProgress-bar': {
        background: 'linear-gradient(90deg, #FFD700 0%, #FFA500 100%)',
        borderRadius: 4,
    },
}));

interface AgentData {
    id: string;
    name: string;
    type: 'sentiment' | 'technical' | 'fundamental' | 'risk' | 'flow';
    confidence: number;
    signal: 'bullish' | 'bearish' | 'neutral';
    lastUpdate: string;
    performance: number;
    status: 'active' | 'warning' | 'error';
}

const mockAgents: AgentData[] = [
    {
        id: '1',
        name: 'Sentiment Prophet',
        type: 'sentiment',
        confidence: 87,
        signal: 'bullish',
        lastUpdate: '2 min ago',
        performance: 94.2,
        status: 'active',
    },
    {
        id: '2',
        name: 'Technical Wizard',
        type: 'technical',
        confidence: 92,
        signal: 'bullish',
        lastUpdate: '1 min ago',
        performance: 89.7,
        status: 'active',
    },
    {
        id: '3',
        name: 'Flow Master',
        type: 'flow',
        confidence: 78,
        signal: 'neutral',
        lastUpdate: '3 min ago',
        performance: 85.3,
        status: 'warning',
    },
    {
        id: '4',
        name: 'Risk Guardian',
        type: 'risk',
        confidence: 95,
        signal: 'bearish',
        lastUpdate: '1 min ago',
        performance: 91.8,
        status: 'active',
    },
    {
        id: '5',
        name: 'Fundamental Sage',
        type: 'fundamental',
        confidence: 83,
        signal: 'bullish',
        lastUpdate: '5 min ago',
        performance: 88.1,
        status: 'active',
    },
];

const AgentConsensusFlow: React.FC = () => {
    const [agents, setAgents] = useState<AgentData[]>(mockAgents);
    const [consensus, setConsensus] = useState<{
        overall: 'bullish' | 'bearish' | 'neutral';
        confidence: number;
        bullishCount: number;
        bearishCount: number;
        neutralCount: number;
    }>({
        overall: 'bullish',
        confidence: 0,
        bullishCount: 0,
        bearishCount: 0,
        neutralCount: 0,
    });

    useEffect(() => {
        // Calculate consensus
        const bullishCount = agents.filter(a => a.signal === 'bullish').length;
        const bearishCount = agents.filter(a => a.signal === 'bearish').length;
        const neutralCount = agents.filter(a => a.signal === 'neutral').length;

        const totalConfidence = agents.reduce((sum, agent) => sum + agent.confidence, 0);
        const avgConfidence = totalConfidence / agents.length;

        let overall: 'bullish' | 'bearish' | 'neutral' = 'neutral';
        if (bullishCount > bearishCount && bullishCount > neutralCount) {
            overall = 'bullish';
        } else if (bearishCount > bullishCount && bearishCount > neutralCount) {
            overall = 'bearish';
        }

        setConsensus({
            overall,
            confidence: avgConfidence,
            bullishCount,
            bearishCount,
            neutralCount,
        });
    }, [agents]);

    const getSignalColor = (signal: string) => {
        switch (signal) {
            case 'bullish': return '#00D4AA';
            case 'bearish': return '#FF4757';
            default: return '#94A3B8';
        }
    };

    const getSignalIcon = (signal: string) => {
        switch (signal) {
            case 'bullish': return <TrendingUpIcon />;
            case 'bearish': return <TrendingDownIcon />;
            default: return <AnalyticsIcon />;
        }
    };

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'active': return <CheckCircleIcon sx={{ color: '#00D4AA' }} />;
            case 'warning': return <WarningIcon sx={{ color: '#FFA500' }} />;
            case 'error': return <ErrorIcon sx={{ color: '#FF4757' }} />;
            default: return <CheckCircleIcon />;
        }
    };

    const getAgentTypeIcon = (type: string) => {
        switch (type) {
            case 'sentiment': return <PsychologyIcon />;
            case 'technical': return <AnalyticsIcon />;
            case 'flow': return <SpeedIcon />;
            default: return <AnalyticsIcon />;
        }
    };

    return (
        <Box sx={{ p: 3 }}>
            <Typography variant="h4" sx={{ color: '#FFD700', mb: 3, fontWeight: 'bold' }}>
                🤖 Agent Consensus Flow
            </Typography>

            {/* Overall Consensus */}
            <ConsensusCard sx={{ mb: 3 }}>
                <CardContent>
                    <Grid container spacing={3} alignItems="center">
                        <Grid item xs={12} md={4}>
                            <Box sx={{ textAlign: 'center' }}>
                                <Typography variant="h6" sx={{ color: '#E2E8F0', mb: 1 }}>
                                    Overall Consensus
                                </Typography>
                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                                    {getSignalIcon(consensus.overall)}
                                    <Typography
                                        variant="h4"
                                        sx={{
                                            color: getSignalColor(consensus.overall),
                                            fontWeight: 'bold',
                                            textTransform: 'uppercase'
                                        }}
                                    >
                                        {consensus.overall}
                                    </Typography>
                                </Box>
                                <Typography variant="body2" sx={{ color: '#94A3B8', mt: 1 }}>
                                    {consensus.confidence.toFixed(1)}% Confidence
                                </Typography>
                            </Box>
                        </Grid>

                        <Grid item xs={12} md={8}>
                            <Box sx={{ mb: 2 }}>
                                <Typography variant="body2" sx={{ color: '#94A3B8', mb: 1 }}>
                                    Agent Distribution
                                </Typography>
                                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                                    <Chip
                                        label={`${consensus.bullishCount} Bullish`}
                                        sx={{
                                            backgroundColor: 'rgba(0, 212, 170, 0.1)',
                                            color: '#00D4AA',
                                            border: '1px solid #00D4AA'
                                        }}
                                    />
                                    <Chip
                                        label={`${consensus.bearishCount} Bearish`}
                                        sx={{
                                            backgroundColor: 'rgba(255, 71, 87, 0.1)',
                                            color: '#FF4757',
                                            border: '1px solid #FF4757'
                                        }}
                                    />
                                    <Chip
                                        label={`${consensus.neutralCount} Neutral`}
                                        sx={{
                                            backgroundColor: 'rgba(148, 163, 184, 0.1)',
                                            color: '#94A3B8',
                                            border: '1px solid #94A3B8'
                                        }}
                                    />
                                </Box>
                            </Box>

                            <ConsensusProgress variant="determinate" value={consensus.confidence} />
                        </Grid>
                    </Grid>
                </CardContent>
            </ConsensusCard>

            {/* Individual Agents */}
            <ConsensusCard>
                <CardContent>
                    <Typography variant="h6" sx={{ color: '#E2E8F0', mb: 2 }}>
                        Individual Agent Signals
                    </Typography>

                    <List sx={{ maxHeight: 400, overflow: 'auto' }}>
                        {agents.map((agent) => (
                            <ListItem
                                key={agent.id}
                                sx={{
                                    backgroundColor: 'rgba(255, 255, 255, 0.02)',
                                    borderRadius: '8px',
                                    mb: 1,
                                    border: '1px solid rgba(255, 255, 255, 0.1)',
                                }}
                            >
                                <ListItemAvatar>
                                    <AgentAvatar>
                                        {getAgentTypeIcon(agent.type)}
                                    </AgentAvatar>
                                </ListItemAvatar>

                                <ListItemText
                                    primary={
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            <Typography variant="subtitle1" sx={{ color: '#E2E8F0', fontWeight: 'bold' }}>
                                                {agent.name}
                                            </Typography>
                                            <Chip
                                                label={agent.signal.toUpperCase()}
                                                size="small"
                                                sx={{
                                                    backgroundColor: `${getSignalColor(agent.signal)}20`,
                                                    color: getSignalColor(agent.signal),
                                                    border: `1px solid ${getSignalColor(agent.signal)}`,
                                                    fontSize: '0.7rem',
                                                }}
                                            />
                                        </Box>
                                    }
                                    secondary={
                                        <Box sx={{ mt: 1 }}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                                <Typography variant="body2" sx={{ color: '#94A3B8' }}>
                                                    Confidence: {agent.confidence}%
                                                </Typography>
                                                <Typography variant="body2" sx={{ color: '#94A3B8' }}>
                                                    Performance: {agent.performance}%
                                                </Typography>
                                            </Box>
                                            <LinearProgress
                                                variant="determinate"
                                                value={agent.confidence}
                                                sx={{
                                                    height: 4,
                                                    borderRadius: 2,
                                                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                                                    '& .MuiLinearProgress-bar': {
                                                        backgroundColor: getSignalColor(agent.signal),
                                                        borderRadius: 2,
                                                    },
                                                }}
                                            />
                                            <Typography variant="caption" sx={{ color: '#64748B', mt: 0.5 }}>
                                                Last update: {agent.lastUpdate}
                                            </Typography>
                                        </Box>
                                    }
                                />

                                <Tooltip title={`Status: ${agent.status}`}>
                                    <IconButton size="small">
                                        {getStatusIcon(agent.status)}
                                    </IconButton>
                                </Tooltip>
                            </ListItem>
                        ))}
                    </List>
                </CardContent>
            </ConsensusCard>
        </Box>
    );
};

export default AgentConsensusFlow;
