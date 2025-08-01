import React, { useState } from 'react';
import {
    Box,
    Card,
    CardContent,
    Typography,
    Chip,
    Button,
    Divider,
    LinearProgress,
    Alert,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    IconButton,
    Tooltip,
    useTheme,
    alpha,
} from '@mui/material';
import {
    ExpandMore,
    Psychology,
    TrendingUp,
    TrendingDown,
    GpsFixed,
    Security,
    ContentCopy,
    PlayArrow,
    CheckCircle,
    Warning,
    Info,
    Timeline,
    Speed,
    Insights,
    AutoAwesome,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { motion, AnimatePresence } from 'framer-motion';

// Professional styled components
const GuidanceContainer = styled(motion.div)(({ theme }) => ({
    height: '100%',
    backgroundColor: alpha(theme.palette.background.paper, 0.95),
    backdropFilter: 'blur(10px)',
    border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
    borderRadius: '12px',
    overflow: 'hidden',
}));

const ConfidenceBar = styled(Box)(({ theme, confidence }) => ({
    position: 'relative',
    height: '8px',
    backgroundColor: alpha(theme.palette.grey[300], 0.3),
    borderRadius: '4px',
    overflow: 'hidden',
    '&::after': {
        content: '""',
        position: 'absolute',
        left: 0,
        top: 0,
        height: '100%',
        width: `${confidence}%`,
        backgroundColor: confidence >= 90 ? theme.palette.success.main :
            confidence >= 70 ? theme.palette.warning.main :
                theme.palette.error.main,
        borderRadius: '4px',
        transition: 'width 1s ease-in-out',
    },
}));

const ActionButton = styled(Button)(({ theme, variant: buttonVariant }) => ({
    borderRadius: '8px',
    textTransform: 'none',
    fontWeight: 'bold',
    padding: '12px 24px',
    ...(buttonVariant === 'primary' && {
        background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
        '&:hover': {
            background: `linear-gradient(135deg, ${theme.palette.primary.dark} 0%, ${theme.palette.primary.main} 100%)`,
        },
    }),
}));

const AgentCard = styled(Box)(({ theme }) => ({
    padding: '12px',
    backgroundColor: alpha(theme.palette.primary.main, 0.05),
    border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
    borderRadius: '8px',
    margin: '8px 0',
}));

interface Signal {
    id: string;
    symbol: string;
    type: 'buy' | 'sell';
    action: string;
    price: number;
    confidence: number;
    timestamp: number;
    reason: string;
    timeframe: string;
    targetPrice?: number;
    stopLoss?: number;
    riskReward?: number;
    agentAnalysis?: {
        technical: string;
        sentiment: string;
        aiPredictor: string;
    };
    historicalPerformance?: {
        successRate: number;
        todayTrades: number;
        weeklyPerformance: number;
    };
}

interface TradeGuidancePanelProps {
    signal: Signal | null;
    onCopyTradeInfo: (signal: Signal) => void;
    onSimulateTrade?: (signal: Signal) => void;
    onDismiss: () => void;
}

const TradeGuidancePanel: React.FC<TradeGuidancePanelProps> = ({
    signal,
    onCopyTradeInfo,
    onSimulateTrade,
    onDismiss,
}) => {
    const theme = useTheme();
    const [expandedSection, setExpandedSection] = useState<string>('explanation');

    if (!signal) {
        return (
            <GuidanceContainer
                initial={{ opacity: 0, x: 300 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 300 }}
            >
                <CardContent sx={{ textAlign: 'center', py: 8 }}>
                    <Psychology sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="h6" color="text.secondary" gutterBottom>
                        Select a signal to view guidance
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Click on any signal marker or alert to see detailed analysis and trading recommendations.
                    </Typography>
                </CardContent>
            </GuidanceContainer>
        );
    }

    const getConfidenceLevel = (confidence: number) => {
        if (confidence >= 90) return { level: 'High', color: 'success' };
        if (confidence >= 70) return { level: 'Medium', color: 'warning' };
        return { level: 'Low', color: 'error' };
    };

    const confidenceInfo = getConfidenceLevel(signal.confidence);

    return (
        <GuidanceContainer
            initial={{ opacity: 0, x: 300 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 300 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
        >
            <CardContent sx={{ p: 0, height: '100%' }}>
                {/* Header */}
                <Box sx={{ p: 3, borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}` }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                            <Box sx={{
                                width: 48,
                                height: 48,
                                borderRadius: '50%',
                                backgroundColor: signal.type === 'buy' ?
                                    alpha(theme.palette.success.main, 0.1) :
                                    alpha(theme.palette.error.main, 0.1),
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                            }}>
                                {signal.type === 'buy' ?
                                    <TrendingUp sx={{ color: theme.palette.success.main, fontSize: 24 }} /> :
                                    <TrendingDown sx={{ color: theme.palette.error.main, fontSize: 24 }} />
                                }
                            </Box>
                            <Box>
                                <Typography variant="h6" fontWeight="bold">
                                    {signal.symbol} Signal
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    {signal.type === 'buy' ? 'Bullish' : 'Bearish'} • {signal.timeframe}
                                </Typography>
                            </Box>
                        </Box>
                        <IconButton onClick={onDismiss} size="small">
                            ✕
                        </IconButton>
                    </Box>

                    {/* Confidence Indicator */}
                    <Box sx={{ mb: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                            <Typography variant="body2" fontWeight="medium">
                                Confidence Level
                            </Typography>
                            <Chip
                                label={`${signal.confidence}% ${confidenceInfo.level}`}
                                color={confidenceInfo.color as any}
                                size="small"
                                variant="outlined"
                            />
                        </Box>
                        <ConfidenceBar confidence={signal.confidence} />
                    </Box>

                    {/* Current Price */}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Typography variant="h4" fontWeight="bold" color="primary">
                            ${signal.price.toFixed(2)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                            Entry Price
                        </Typography>
                    </Box>
                </Box>

                {/* Scrollable Content */}
                <Box sx={{ flex: 1, overflow: 'auto', p: 3 }}>
                    {/* AI Explanation */}
                    <Accordion
                        expanded={expandedSection === 'explanation'}
                        onChange={() => setExpandedSection(expandedSection === 'explanation' ? '' : 'explanation')}
                        sx={{ mb: 2, boxShadow: 'none', border: `1px solid ${alpha(theme.palette.divider, 0.1)}` }}
                    >
                        <AccordionSummary expandIcon={<ExpandMore />}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <AutoAwesome color="primary" />
                                <Typography variant="subtitle1" fontWeight="bold">
                                    AI Analysis & Reasoning
                                </Typography>
                            </Box>
                        </AccordionSummary>
                        <AccordionDetails>
                            <Typography variant="body2" sx={{ mb: 2, lineHeight: 1.6 }}>
                                {signal.reason}
                            </Typography>

                            {/* Agent Contributions */}
                            {signal.agentAnalysis && (
                                <Box sx={{ mt: 2 }}>
                                    <Typography variant="subtitle2" fontWeight="bold" sx={{ mb: 1 }}>
                                        Agent Contributions:
                                    </Typography>

                                    <AgentCard>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                            <Timeline color="primary" sx={{ fontSize: 16 }} />
                                            <Typography variant="body2" fontWeight="medium">
                                                Technical Agent
                                            </Typography>
                                        </Box>
                                        <Typography variant="caption" color="text.secondary">
                                            {signal.agentAnalysis.technical}
                                        </Typography>
                                    </AgentCard>

                                    <AgentCard>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                            <Psychology color="primary" sx={{ fontSize: 16 }} />
                                            <Typography variant="body2" fontWeight="medium">
                                                Sentiment Agent
                                            </Typography>
                                        </Box>
                                        <Typography variant="caption" color="text.secondary">
                                            {signal.agentAnalysis.sentiment}
                                        </Typography>
                                    </AgentCard>

                                    <AgentCard>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                            <Insights color="primary" sx={{ fontSize: 16 }} />
                                            <Typography variant="body2" fontWeight="medium">
                                                AI Predictor
                                            </Typography>
                                        </Box>
                                        <Typography variant="caption" color="text.secondary">
                                            {signal.agentAnalysis.aiPredictor}
                                        </Typography>
                                    </AgentCard>
                                </Box>
                            )}
                        </AccordionDetails>
                    </Accordion>

                    {/* Recommended Action */}
                    <Accordion
                        expanded={expandedSection === 'action'}
                        onChange={() => setExpandedSection(expandedSection === 'action' ? '' : 'action')}
                        sx={{ mb: 2, boxShadow: 'none', border: `1px solid ${alpha(theme.palette.divider, 0.1)}` }}
                    >
                        <AccordionSummary expandIcon={<ExpandMore />}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <GpsFixed color="primary" />
                                <Typography variant="subtitle1" fontWeight="bold">
                                    Recommended Trade Action
                                </Typography>
                            </Box>
                        </AccordionSummary>
                        <AccordionDetails>
                            <Alert
                                severity={signal.type === 'buy' ? 'success' : 'error'}
                                sx={{ mb: 2 }}
                                icon={signal.type === 'buy' ? <TrendingUp /> : <TrendingDown />}
                            >
                                <Typography variant="body2" fontWeight="bold">
                                    {signal.action}
                                </Typography>
                            </Alert>

                            <List dense>
                                <ListItem>
                                    <ListItemIcon>
                                        <GpsFixed sx={{ fontSize: 20 }} />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Entry Price"
                                        secondary={`$${signal.price.toFixed(2)}`}
                                    />
                                </ListItem>

                                {signal.targetPrice && (
                                    <ListItem>
                                        <ListItemIcon>
                                            <CheckCircle sx={{ fontSize: 20, color: theme.palette.success.main }} />
                                        </ListItemIcon>
                                        <ListItemText
                                            primary="Target Price"
                                            secondary={`$${signal.targetPrice.toFixed(2)} (+${((signal.targetPrice - signal.price) / signal.price * 100).toFixed(1)}%)`}
                                        />
                                    </ListItem>
                                )}

                                {signal.stopLoss && (
                                    <ListItem>
                                        <ListItemIcon>
                                            <Security sx={{ fontSize: 20, color: theme.palette.error.main }} />
                                        </ListItemIcon>
                                        <ListItemText
                                            primary="Stop Loss"
                                            secondary={`$${signal.stopLoss.toFixed(2)} (${((signal.stopLoss - signal.price) / signal.price * 100).toFixed(1)}%)`}
                                        />
                                    </ListItem>
                                )}

                                {signal.riskReward && (
                                    <ListItem>
                                        <ListItemIcon>
                                            <Speed sx={{ fontSize: 20, color: theme.palette.primary.main }} />
                                        </ListItemIcon>
                                        <ListItemText
                                            primary="Risk/Reward Ratio"
                                            secondary={`1:${signal.riskReward.toFixed(1)}`}
                                        />
                                    </ListItem>
                                )}
                            </List>
                        </AccordionDetails>
                    </Accordion>

                    {/* Performance Stats */}
                    {signal.historicalPerformance && (
                        <Accordion
                            expanded={expandedSection === 'performance'}
                            onChange={() => setExpandedSection(expandedSection === 'performance' ? '' : 'performance')}
                            sx={{ mb: 2, boxShadow: 'none', border: `1px solid ${alpha(theme.palette.divider, 0.1)}` }}
                        >
                            <AccordionSummary expandIcon={<ExpandMore />}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <Timeline color="primary" />
                                    <Typography variant="subtitle1" fontWeight="bold">
                                        Historical Performance
                                    </Typography>
                                </Box>
                            </AccordionSummary>
                            <AccordionDetails>
                                <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 2 }}>
                                    <Box sx={{ textAlign: 'center' }}>
                                        <Typography variant="h6" color="primary" fontWeight="bold">
                                            {signal.historicalPerformance.successRate}%
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary">
                                            Success Rate
                                        </Typography>
                                    </Box>
                                    <Box sx={{ textAlign: 'center' }}>
                                        <Typography variant="h6" color="primary" fontWeight="bold">
                                            {signal.historicalPerformance.todayTrades}
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary">
                                            Today's Trades
                                        </Typography>
                                    </Box>
                                    <Box sx={{ textAlign: 'center' }}>
                                        <Typography variant="h6" color="primary" fontWeight="bold">
                                            +{signal.historicalPerformance.weeklyPerformance}%
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary">
                                            Weekly Performance
                                        </Typography>
                                    </Box>
                                </Box>

                                <Alert severity="info" sx={{ mt: 2 }}>
                                    <Typography variant="caption">
                                        This strategy has been performing well with a {signal.historicalPerformance.successRate}% success rate over the past week.
                                    </Typography>
                                </Alert>
                            </AccordionDetails>
                        </Accordion>
                    )}
                </Box>

                {/* Action Buttons */}
                <Box sx={{ p: 3, borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}` }}>
                    <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                        <ActionButton
                            variant="primary"
                            fullWidth
                            startIcon={<ContentCopy />}
                            onClick={() => onCopyTradeInfo(signal)}
                        >
                            Copy Trade Info
                        </ActionButton>

                        {onSimulateTrade && (
                            <ActionButton
                                variant="outlined"
                                fullWidth
                                startIcon={<PlayArrow />}
                                onClick={() => onSimulateTrade(signal)}
                            >
                                Simulate Trade
                            </ActionButton>
                        )}
                    </Box>

                    <Button
                        variant="text"
                        fullWidth
                        onClick={onDismiss}
                        sx={{ textTransform: 'none' }}
                    >
                        Got it, dismiss
                    </Button>
                </Box>
            </CardContent>
        </GuidanceContainer>
    );
};

export default TradeGuidancePanel;
