import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Chip,
    LinearProgress,
    Divider,
    Card,
    CardContent,
    Collapse,
    IconButton,
    Alert,
} from '@mui/material';
import {
    Timeline as TimelineIcon,
    TrendingUp as TrendingUpIcon,
    TrendingDown as TrendingDownIcon,
    Analytics as AnalyticsIcon,
    Psychology as PsychologyIcon,
    CheckCircle as CheckIcon,
    RadioButtonUnchecked as PendingIcon,
    ExpandMore as ExpandMoreIcon,
    ExpandLess as ExpandLessIcon,
    Speed as SpeedIcon,
    Security as SecurityIcon,
    AttachMoney as MoneyIcon,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface AIActivity {
    currentAction: string;
    confidence: number;
    stage: 'scanning' | 'analyzing' | 'drawing' | 'confirming';
    lastSignal: any;
}

interface Trade {
    id: number;
    symbol: string;
    type: string;
    strike: number;
    expiry: string;
    entry: number;
    status: string;
    pnl: string;
    confidence: number;
}

interface AIThoughtProcessProps {
    activity: AIActivity;
    recentTrades: Trade[];
}

interface ThoughtStep {
    id: string;
    timestamp: Date;
    type: 'scan' | 'pattern' | 'analysis' | 'decision' | 'execution';
    title: string;
    description: string;
    confidence?: number;
    status: 'completed' | 'active' | 'pending';
    details?: string[];
}

const AIThoughtProcess: React.FC<AIThoughtProcessProps> = ({ activity, recentTrades }) => {
    const theme = useTheme();
    const [thoughtSteps, setThoughtSteps] = useState<ThoughtStep[]>([]);
    const [expandedStep, setExpandedStep] = useState<string | null>(null);
    const [currentThought, setCurrentThought] = useState<string>('');

    useEffect(() => {
        // Simulate AI thought process
        const interval = setInterval(() => {
            generateNewThought();
        }, 3000);

        return () => clearInterval(interval);
    }, [activity]);

    const generateNewThought = () => {
        const thoughts = [
            {
                type: 'scan' as const,
                title: 'Market Scan Complete',
                description: 'Analyzed 500+ symbols for potential setups',
                details: [
                    'Identified 23 symbols with unusual volume',
                    'Found 8 symbols near key support levels',
                    '5 symbols showing bullish divergence',
                ],
            },
            {
                type: 'pattern' as const,
                title: 'Pattern Recognition',
                description: 'Detected ascending triangle on AAPL 5m chart',
                details: [
                    'Pattern confidence: 87%',
                    'Breakout target: $185.50',
                    'Volume confirmation: Positive',
                ],
            },
            {
                type: 'analysis' as const,
                title: 'Technical Analysis',
                description: 'Running multi-timeframe analysis',
                details: [
                    'RSI: Oversold on 1m, neutral on 5m',
                    'MACD: Bullish crossover imminent',
                    'Support level: $180.25 (strong)',
                ],
            },
            {
                type: 'decision' as const,
                title: 'Trade Decision',
                description: 'Evaluating risk/reward for CALL option',
                details: [
                    'Strike: $185, Expiry: Jan 26',
                    'Risk: $250, Reward: $750',
                    'Win probability: 68%',
                ],
            },
        ];

        const randomThought = thoughts[Math.floor(Math.random() * thoughts.length)];
        const newStep: ThoughtStep = {
            id: `thought-${Date.now()}`,
            timestamp: new Date(),
            ...randomThought,
            confidence: 70 + Math.random() * 30,
            status: 'active',
        };

        setThoughtSteps(prev => {
            const updated = prev.map(step => ({
                ...step,
                status: step.status === 'active' ? 'completed' : step.status,
            }));
            return [newStep, ...updated].slice(0, 10);
        });

        setCurrentThought(randomThought.description);
    };

    const getStepIcon = (type: string, status: string) => {
        const iconProps = {
            sx: {
                color: status === 'completed'
                    ? theme.palette.success.main
                    : status === 'active'
                        ? theme.palette.primary.main
                        : theme.palette.text.disabled,
            },
        };

        switch (type) {
            case 'scan':
                return <TimelineIcon {...iconProps} />;
            case 'pattern':
                return <AnalyticsIcon {...iconProps} />;
            case 'analysis':
                return <PsychologyIcon {...iconProps} />;
            case 'decision':
                return <SpeedIcon {...iconProps} />;
            case 'execution':
                return <CheckIcon {...iconProps} />;
            default:
                return <PendingIcon {...iconProps} />;
        }
    };

    const getStageProgress = () => {
        switch (activity.stage) {
            case 'scanning':
                return 25;
            case 'analyzing':
                return 50;
            case 'drawing':
                return 75;
            case 'confirming':
                return 100;
            default:
                return 0;
        }
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Current Activity */}
            <Card sx={{ mb: 2 }}>
                <CardContent>
                    <Typography variant="h6" gutterBottom>
                        AI Trading Brain
                    </Typography>

                    <Box sx={{ mb: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Typography variant="body2" color="text.secondary">
                                {activity.currentAction}
                            </Typography>
                            <Chip
                                label={`${activity.confidence}%`}
                                size="small"
                                color="primary"
                            />
                        </Box>
                        <LinearProgress
                            variant="determinate"
                            value={getStageProgress()}
                            sx={{ height: 8, borderRadius: 4 }}
                        />
                    </Box>

                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        <Chip
                            icon={<TimelineIcon />}
                            label={activity.stage}
                            size="small"
                            variant="outlined"
                        />
                        <Chip
                            icon={<SpeedIcon />}
                            label="High Speed"
                            size="small"
                            color="success"
                        />
                        <Chip
                            icon={<SecurityIcon />}
                            label="Risk Managed"
                            size="small"
                            color="info"
                        />
                    </Box>
                </CardContent>
            </Card>

            {/* Current Thought */}
            {currentThought && (
                <Alert severity="info" sx={{ mb: 2 }}>
                    <Typography variant="body2">
                        <strong>Current Analysis:</strong> {currentThought}
                    </Typography>
                </Alert>
            )}

            {/* Thought Process Timeline */}
            <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                AI Thought Process
            </Typography>

            <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
                <List>
                    {thoughtSteps.map((step, index) => (
                        <React.Fragment key={step.id}>
                            <ListItem
                                sx={{
                                    backgroundColor: step.status === 'active'
                                        ? theme.palette.action.selected
                                        : 'transparent',
                                    borderRadius: 1,
                                    mb: 1,
                                }}
                            >
                                <ListItemIcon>
                                    {getStepIcon(step.type, step.status)}
                                </ListItemIcon>
                                <ListItemText
                                    primary={
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            <Typography variant="body1">
                                                {step.title}
                                            </Typography>
                                            {step.confidence && (
                                                <Chip
                                                    label={`${Math.round(step.confidence)}%`}
                                                    size="small"
                                                    color={step.confidence > 80 ? 'success' : 'default'}
                                                />
                                            )}
                                        </Box>
                                    }
                                    secondary={
                                        <>
                                            <Typography variant="body2" color="text.secondary">
                                                {step.description}
                                            </Typography>
                                            <Typography variant="caption" color="text.disabled">
                                                {new Date(step.timestamp).toLocaleTimeString()}
                                            </Typography>
                                        </>
                                    }
                                />
                                {step.details && (
                                    <IconButton
                                        size="small"
                                        onClick={() => setExpandedStep(
                                            expandedStep === step.id ? null : step.id
                                        )}
                                    >
                                        {expandedStep === step.id ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                                    </IconButton>
                                )}
                            </ListItem>

                            {step.details && (
                                <Collapse in={expandedStep === step.id}>
                                    <Box sx={{ pl: 7, pr: 2, pb: 2 }}>
                                        {step.details.map((detail, idx) => (
                                            <Typography
                                                key={idx}
                                                variant="body2"
                                                color="text.secondary"
                                                sx={{ mb: 0.5 }}
                                            >
                                                • {detail}
                                            </Typography>
                                        ))}
                                    </Box>
                                </Collapse>
                            )}

                            {index < thoughtSteps.length - 1 && <Divider variant="inset" />}
                        </React.Fragment>
                    ))}
                </List>
            </Box>

            {/* Performance Summary */}
            <Card sx={{ mt: 2 }}>
                <CardContent>
                    <Typography variant="subtitle2" gutterBottom>
                        Today's AI Performance
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-around' }}>
                        <Box sx={{ textAlign: 'center' }}>
                            <Typography variant="h6" color="success.main">
                                12
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                                Signals
                            </Typography>
                        </Box>
                        <Box sx={{ textAlign: 'center' }}>
                            <Typography variant="h6" color="primary.main">
                                68%
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                                Win Rate
                            </Typography>
                        </Box>
                        <Box sx={{ textAlign: 'center' }}>
                            <Typography variant="h6" color="success.main">
                                +$2,450
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                                P&L
                            </Typography>
                        </Box>
                    </Box>
                </CardContent>
            </Card>
        </Box>
    );
};

export default AIThoughtProcess; 