import React from 'react';
import {
    Box,
    Paper,
    Typography,
    Grid,
    Stack,
    useTheme,
    alpha,
    LinearProgress,
} from '@mui/material';
import {
    TrendingUp,
    Speed,
    CheckCircle,
    Timer,
    AttachMoney,
    ShowChart,
} from '@mui/icons-material';
import { PreciseOptionsSignal } from '../../types/signals';

interface QuickStatsProps {
    signals: PreciseOptionsSignal[];
}

interface StatCardProps {
    icon: React.ReactNode;
    label: string;
    value: string | number;
    subValue?: string;
    color?: string;
    progress?: number;
}

const StatCard: React.FC<StatCardProps> = ({
    icon,
    label,
    value,
    subValue,
    color = 'primary',
    progress,
}) => {
    const theme = useTheme();

    return (
        <Paper
            elevation={0}
            sx={{
                p: 2,
                height: '100%',
                background: alpha(theme.palette.background.paper, 0.5),
                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                position: 'relative',
                overflow: 'hidden',
            }}
        >
            <Stack spacing={1}>
                <Stack direction="row" alignItems="center" justifyContent="space-between">
                    <Box
                        sx={{
                            p: 1,
                            borderRadius: 1,
                            background: alpha((theme.palette as any)[color]?.main || theme.palette.primary.main, 0.1),
                            color: (theme.palette as any)[color]?.main || theme.palette.primary.main,
                        }}
                    >
                        {icon}
                    </Box>
                    <Typography variant="h5" fontWeight="bold">
                        {value}
                    </Typography>
                </Stack>

                <Typography variant="body2" color="text.secondary">
                    {label}
                </Typography>

                {subValue && (
                    <Typography variant="caption" color="text.secondary">
                        {subValue}
                    </Typography>
                )}

                {progress !== undefined && (
                    <LinearProgress
                        variant="determinate"
                        value={progress}
                        sx={{
                            height: 4,
                            borderRadius: 2,
                            backgroundColor: alpha((theme.palette as any)[color]?.main || theme.palette.primary.main, 0.1),
                            '& .MuiLinearProgress-bar': {
                                backgroundColor: (theme.palette as any)[color]?.main || theme.palette.primary.main,
                            },
                        }}
                    />
                )}
            </Stack>

            {/* Background decoration */}
            <Box
                sx={{
                    position: 'absolute',
                    top: -20,
                    right: -20,
                    width: 80,
                    height: 80,
                    borderRadius: '50%',
                    background: alpha((theme.palette as any)[color]?.main || theme.palette.primary.main, 0.05),
                }}
            />
        </Paper>
    );
};

const QuickStats: React.FC<QuickStatsProps> = ({ signals }) => {
    const theme = useTheme();

    // Calculate statistics
    const activeSignals = signals.filter(s => {
        const now = new Date();
        const signalTime = new Date(s.timestamp);
        return signalTime > now;
    }).length;

    const highConfidenceSignals = signals.filter(s => s.confidence >= 80).length;
    const avgConfidence = signals.length > 0
        ? Math.round(signals.reduce((sum, s) => sum + s.confidence, 0) / signals.length)
        : 0;

    const callSignals = signals.filter(s => s.type === 'CALL').length;
    const putSignals = signals.filter(s => s.type === 'PUT').length;

    const avgRiskReward = signals.length > 0
        ? (signals.reduce((sum, s) => sum + s.risk_reward_ratio, 0) / signals.length).toFixed(1)
        : '0.0';

    return (
        <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
                <StatCard
                    icon={<Speed />}
                    label="Active Signals"
                    value={activeSignals}
                    subValue={`${signals.length} total`}
                    color="primary"
                    progress={(activeSignals / Math.max(signals.length, 1)) * 100}
                />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
                <StatCard
                    icon={<CheckCircle />}
                    label="Avg Confidence"
                    value={`${avgConfidence}%`}
                    subValue={`${highConfidenceSignals} high confidence`}
                    color="success"
                    progress={avgConfidence}
                />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
                <StatCard
                    icon={<ShowChart />}
                    label="Signal Distribution"
                    value={`${callSignals}/${putSignals}`}
                    subValue="Calls / Puts"
                    color="info"
                />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
                <StatCard
                    icon={<AttachMoney />}
                    label="Avg Risk/Reward"
                    value={`1:${avgRiskReward}`}
                    subValue="Risk to reward ratio"
                    color="warning"
                />
            </Grid>
        </Grid>
    );
};

export default QuickStats;
