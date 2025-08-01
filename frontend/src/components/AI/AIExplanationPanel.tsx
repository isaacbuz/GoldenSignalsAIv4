import React, { useEffect, useState } from 'react';
import {
    Box,
    Typography,
    Paper,
    Chip,
    Stack,
    LinearProgress,
    useTheme,
    alpha,
    Divider,
    IconButton,
    Tooltip,
    Collapse,
    Avatar,
} from '@mui/material';
import {
    Psychology,
    TrendingUp,
    TrendingDown,
    Warning,
    CheckCircle,
    Info,
    ExpandMore,
    ExpandLess,
    AutoAwesome,
    Lightbulb,
    Timeline,
    Assessment,
    Speed,
    Timer,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { PreciseOptionsSignal } from '../../types/signals';

interface AIExplanationPanelProps {
    signal: PreciseOptionsSignal | null;
    insights?: any;
    symbol: string;
    timeframe: string;
    expanded?: boolean;
}

interface AIAgent {
    name: string;
    icon: React.ReactNode;
    color: string;
    analysis: string;
    confidence: number;
    details?: string[];
}

const AIExplanationPanel: React.FC<AIExplanationPanelProps> = ({
    signal,
    insights,
    symbol,
    timeframe,
    expanded = false,
}) => {
    const theme = useTheme();
    const [isExpanded, setIsExpanded] = useState(true);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [currentAnalysis, setCurrentAnalysis] = useState<string>('');
    const [agents, setAgents] = useState<AIAgent[]>([]);

    // Simulate AI analysis when signal changes
    useEffect(() => {
        if (signal) {
            setIsAnalyzing(true);

            // Simulate different AI agents analyzing
            const analysisSteps = [
                {
                    name: 'Technical Agent',
                    icon: <Timeline />,
                    color: theme.palette.info.main,
                    analysis: `Detected ${signal.type || 'signal'} setup on ${symbol}. RSI showing ${signal.type === 'CALL' ? 'oversold' : 'overbought'} conditions with bullish divergence. MACD histogram turning ${signal.type === 'CALL' ? 'positive' : 'negative'}.`,
                    confidence: 85,
                    details: expanded ? [
                        `RSI: ${signal.type === 'CALL' ? '28' : '72'} (${signal.type === 'CALL' ? 'Oversold' : 'Overbought'})`,
                        `MACD: ${signal.type === 'CALL' ? 'Bullish' : 'Bearish'} crossover confirmed`,
                        `Support/Resistance: Testing key ${signal.type === 'CALL' ? 'support' : 'resistance'} at $${signal.entry_price || 0}`,
                        `Volume: ${signal.type === 'CALL' ? 'Above' : 'Below'} 20-day average`,
                    ] : [],
                },
                {
                    name: 'Sentiment Agent',
                    icon: <Psychology />,
                    color: theme.palette.warning.main,
                    analysis: `Market sentiment is ${signal.type === 'CALL' ? 'shifting bullish' : 'turning bearish'}. Social media mentions increased 45% with ${signal.type === 'CALL' ? 'positive' : 'negative'} tone. Options flow shows institutional ${signal.type === 'CALL' ? 'buying' : 'selling'}.`,
                    confidence: 78,
                    details: expanded ? [
                        `Social Sentiment: ${signal.type === 'CALL' ? '+68%' : '-52%'} (24h change)`,
                        `News Sentiment: ${signal.type === 'CALL' ? 'Positive' : 'Negative'} (3 major catalysts)`,
                        `Analyst Ratings: ${signal.type === 'CALL' ? '8 upgrades' : '5 downgrades'} this week`,
                        `Retail Interest: ${signal.type === 'CALL' ? 'Increasing' : 'Decreasing'} rapidly`,
                    ] : [],
                },
                {
                    name: 'Volume Agent',
                    icon: <Assessment />,
                    color: theme.palette.success.main,
                    analysis: `Unusual options activity detected. Volume is 3.2x average with ${signal.type === 'CALL' ? 'call' : 'put'} bias. Order flow indicates ${signal.type === 'CALL' ? 'accumulation' : 'distribution'} at current levels.`,
                    confidence: 92,
                    details: expanded ? [
                        `Options Volume: 3.2x daily average`,
                        `Put/Call Ratio: ${signal.type === 'CALL' ? '0.45' : '2.1'} (${signal.type === 'CALL' ? 'Bullish' : 'Bearish'})`,
                        `Large Orders: ${signal.type === 'CALL' ? '12 sweeps' : '8 blocks'} detected`,
                        `Dark Pool: $${signal.type === 'CALL' ? '4.2M' : '3.8M'} in ${signal.type === 'CALL' ? 'buys' : 'sells'}`,
                    ] : [],
                },
                {
                    name: 'Risk Agent',
                    icon: <Warning />,
                    color: theme.palette.error.main,
                    analysis: `Risk/Reward ratio: ${signal.risk_reward_ratio || 'N/A'}. Max loss capped at $${signal.max_loss || 0}. Volatility expected to ${signal.type === 'CALL' ? 'decrease' : 'increase'} based on implied volatility skew.`,
                    confidence: 88,
                    details: expanded ? [
                        `IV Rank: ${signal.type === 'CALL' ? '25th' : '75th'} percentile`,
                        `Expected Move: ±${signal.type === 'CALL' ? '2.8%' : '3.5%'} by expiration`,
                        `Greeks: Delta ${signal.type === 'CALL' ? '0.55' : '-0.45'}, Theta -0.08`,
                        `Probability ITM: ${signal.type === 'CALL' ? '62%' : '58%'}`,
                    ] : [],
                },
                {
                    name: 'Pattern Recognition',
                    icon: <AutoAwesome />,
                    color: theme.palette.secondary.main,
                    analysis: `Identified ${signal.type === 'CALL' ? 'Bull Flag' : 'Bear Flag'} pattern with 78% historical win rate. Previous similar setups resulted in ${signal.type === 'CALL' ? '+4.2%' : '-3.8%'} average move within 3 days.`,
                    confidence: 81,
                    details: expanded ? [
                        `Pattern: ${signal.type === 'CALL' ? 'Bull Flag' : 'Bear Flag'} (Confirmed)`,
                        `Historical Win Rate: 78% (142 occurrences)`,
                        `Average Move: ${signal.type === 'CALL' ? '+4.2%' : '-3.8%'} in 3 days`,
                        `Pattern Strength: 8.5/10`,
                    ] : [],
                },
            ];

            // Animate agents appearing one by one
            analysisSteps.forEach((agent, index) => {
                setTimeout(() => {
                    setAgents(prev => [...prev.slice(0, index), agent]);
                }, index * 500);
            });

            // Generate final analysis
            setTimeout(() => {
                const finalAnalysis = signal.type === 'CALL'
                    ? `Strong bullish signal detected for ${symbol}. Multiple technical indicators align with positive sentiment and unusual call buying. Entry at $${signal.entry_price || 0} offers favorable risk/reward with tight stop at $${signal.stop_loss || 0}. Target $${signal.take_profit || 0} based on resistance levels and options flow.`
                    : `Bearish setup identified for ${symbol}. Technical breakdown confirmed by negative sentiment and heavy put buying. Short entry at $${signal.entry_price || 0} with stop at $${signal.stop_loss || 0}. Downside target $${signal.take_profit || 0} supported by volume analysis.`;

                setCurrentAnalysis(finalAnalysis);
                setIsAnalyzing(false);
            }, 2500);
        }

        return () => {
            setAgents([]);
            setCurrentAnalysis('');
        };
    }, [signal, symbol, theme]);

    if (!signal) {
        return (
            <Box p={3} textAlign="center">
                <Psychology sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" color="text.secondary">
                    Select a signal to view AI analysis
                </Typography>
                <Typography variant="body2" color="text.secondary" mt={1}>
                    Our AI agents will provide real-time insights and explanations
                </Typography>
            </Box>
        );
    }

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Header */}
            <Box
                sx={{
                    p: 2,
                    borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                }}
            >
                <Stack direction="row" spacing={1} alignItems="center">
                    <AutoAwesome sx={{ color: theme.palette.primary.main }} />
                    <Typography variant="h6" fontWeight="bold">
                        AI Signal Analysis
                    </Typography>
                    <Chip
                        size="small"
                        label={`${timeframe} timeframe`}
                        sx={{ ml: 1 }}
                    />
                </Stack>
                <IconButton size="small" onClick={() => setIsExpanded(!isExpanded)}>
                    {isExpanded ? <ExpandLess /> : <ExpandMore />}
                </IconButton>
            </Box>

            {/* Content */}
            <Collapse in={isExpanded} sx={{ flex: 1, overflow: 'auto' }}>
                <Box
                    p={2}
                    sx={{
                        overflowY: 'auto',
                        '&::-webkit-scrollbar': {
                            width: '8px',
                        },
                        '&::-webkit-scrollbar-track': {
                            background: alpha(theme.palette.background.paper, 0.1),
                            borderRadius: '4px',
                        },
                        '&::-webkit-scrollbar-thumb': {
                            background: alpha(theme.palette.primary.main, 0.3),
                            borderRadius: '4px',
                            '&:hover': {
                                background: alpha(theme.palette.primary.main, 0.5),
                            },
                        },
                    }}
                >
                    {isAnalyzing ? (
                        <Box>
                            <LinearProgress sx={{ mb: 2 }} />
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                                AI agents analyzing signal...
                            </Typography>
                        </Box>
                    ) : null}

                    {/* Agent Analysis */}
                    <AnimatePresence>
                        {agents.map((agent, index) => (
                            <motion.div
                                key={agent.name}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: index * 0.1 }}
                            >
                                <Paper
                                    elevation={0}
                                    sx={{
                                        p: 1.5,
                                        mb: 1.5,
                                        background: alpha(agent.color, 0.05),
                                        border: `1px solid ${alpha(agent.color, 0.2)}`,
                                    }}
                                >
                                    <Stack direction="row" spacing={1} alignItems="flex-start">
                                        <Avatar
                                            sx={{
                                                width: 32,
                                                height: 32,
                                                bgcolor: alpha(agent.color, 0.1),
                                                color: agent.color,
                                            }}
                                        >
                                            {agent.icon}
                                        </Avatar>
                                        <Box flex={1}>
                                            <Stack direction="row" justifyContent="space-between" alignItems="center" mb={0.5}>
                                                <Typography variant="subtitle2" fontWeight="bold">
                                                    {agent.name}
                                                </Typography>
                                                <Chip
                                                    size="small"
                                                    label={`${agent.confidence}%`}
                                                    sx={{
                                                        bgcolor: alpha(agent.color, 0.1),
                                                        color: agent.color,
                                                        fontWeight: 'bold',
                                                    }}
                                                />
                                            </Stack>
                                            <Typography variant="body2" color="text.secondary">
                                                {agent.analysis}
                                            </Typography>
                                            {agent.details && agent.details.length > 0 && (
                                                <Box mt={1}>
                                                    <Stack spacing={0.5}>
                                                        {agent.details.map((detail, idx) => (
                                                            <Typography
                                                                key={idx}
                                                                variant="caption"
                                                                sx={{
                                                                    display: 'flex',
                                                                    alignItems: 'center',
                                                                    color: alpha(theme.palette.text.primary, 0.7),
                                                                    '&:before': {
                                                                        content: '"•"',
                                                                        mr: 1,
                                                                        color: agent.color,
                                                                    },
                                                                }}
                                                            >
                                                                {detail}
                                                            </Typography>
                                                        ))}
                                                    </Stack>
                                                </Box>
                                            )}
                                        </Box>
                                    </Stack>
                                </Paper>
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {/* Final Analysis */}
                    {currentAnalysis && !isAnalyzing && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.5 }}
                        >
                            <Paper
                                elevation={0}
                                sx={{
                                    p: 2,
                                    mt: 2,
                                    background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.05)}, ${alpha(theme.palette.secondary.main, 0.05)})`,
                                    border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
                                }}
                            >
                                <Stack direction="row" spacing={1} alignItems="center" mb={1}>
                                    <Lightbulb sx={{ color: theme.palette.warning.main }} />
                                    <Typography variant="subtitle1" fontWeight="bold">
                                        Consensus Analysis
                                    </Typography>
                                </Stack>
                                <Typography variant="body2" sx={{ lineHeight: 1.6 }}>
                                    {currentAnalysis}
                                </Typography>

                                {/* Key Metrics */}
                                <Stack direction="row" spacing={2} mt={2}>
                                    <Chip
                                        icon={<Speed />}
                                        label={`${signal.confidence || 0}% Confidence`}
                                        size="small"
                                        color="primary"
                                    />
                                    <Chip
                                        icon={<Timer />}
                                        label={signal.entry_window?.end_time
                                            ? `Execute within ${Math.floor((new Date(signal.entry_window?.end_time).getTime() - new Date().getTime()) / 60000)} min`
                                            : 'Execute soon'
                                        }
                                        size="small"
                                        color="warning"
                                    />
                                    <Chip
                                        icon={signal.type === 'CALL' ? <TrendingUp /> : <TrendingDown />}
                                        label={`${signal.type || 'SIGNAL'} ${signal.strike_price || ''}`}
                                        size="small"
                                        color={signal.type === 'CALL' ? 'success' : 'error'}
                                    />
                                </Stack>
                            </Paper>
                        </motion.div>
                    )}
                </Box>
            </Collapse>
        </Box>
    );
};

export default AIExplanationPanel;
