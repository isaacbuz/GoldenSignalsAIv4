import React from 'react';
import {
    Box,
    Paper,
    Typography,
    Stack,
    useTheme,
    alpha,
    CircularProgress,
    Chip,
    Divider,
    LinearProgress,
    Tooltip,
} from '@mui/material';
import {
    Warning,
    Shield,
    TrendingUp,
    TrendingDown,
    AttachMoney,
    Timer,
    Assessment,
} from '@mui/icons-material';
import { PreciseOptionsSignal } from '../../types/signals';

interface RiskMonitorProps {
    signals: PreciseOptionsSignal[];
}

const RiskMonitor: React.FC<RiskMonitorProps> = ({ signals }) => {
    const theme = useTheme();

    // Calculate risk metrics
    const totalRisk = signals.reduce((sum, signal) => sum + signal.max_loss, 0);
    const maxRiskLimit = 10000; // This should come from user settings
    const riskUtilization = (totalRisk / maxRiskLimit) * 100;

    const activePositions = signals.filter(s => {
        const now = new Date();
        const signalTime = new Date(s.timestamp);
        return signalTime <= now;
    }).length;

    const callRisk = signals
        .filter(s => s.type === 'CALL')
        .reduce((sum, s) => sum + s.max_loss, 0);

    const putRisk = signals
        .filter(s => s.type === 'PUT')
        .reduce((sum, s) => sum + s.max_loss, 0);

    const avgStopLoss = signals.length > 0
        ? signals.reduce((sum, s) => sum + s.stop_loss_pct, 0) / signals.length
        : 0;

    // Get risk color based on utilization
    const getRiskColor = (utilization: number) => {
        if (utilization >= 80) return theme.palette.error.main;
        if (utilization >= 60) return theme.palette.warning.main;
        return theme.palette.success.main;
    };

    const riskColor = getRiskColor(riskUtilization);

    return (
        <Paper
            elevation={0}
            sx={{
                p: 3,
                height: '100%',
                background: alpha(theme.palette.background.paper, 0.5),
                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                display: 'flex',
                flexDirection: 'column',
            }}
        >
            {/* Header */}
            <Stack direction="row" alignItems="center" justifyContent="space-between" mb={3}>
                <Typography variant="h6" fontWeight="bold" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Shield sx={{ color: riskColor }} />
                    Risk Monitor
                </Typography>
                <Chip
                    size="small"
                    label={activePositions > 0 ? 'ACTIVE' : 'MONITORING'}
                    color={activePositions > 0 ? 'success' : 'default'}
                    sx={{ fontWeight: 'bold' }}
                />
            </Stack>

            {/* Risk Utilization Circle */}
            <Box sx={{ position: 'relative', display: 'inline-flex', mx: 'auto', mb: 3 }}>
                <CircularProgress
                    variant="determinate"
                    value={riskUtilization}
                    size={120}
                    thickness={8}
                    sx={{
                        color: riskColor,
                        '& .MuiCircularProgress-circle': {
                            strokeLinecap: 'round',
                        },
                    }}
                />
                <CircularProgress
                    variant="determinate"
                    value={100}
                    size={120}
                    thickness={8}
                    sx={{
                        color: alpha(theme.palette.divider, 0.1),
                        position: 'absolute',
                        left: 0,
                    }}
                />
                <Box
                    sx={{
                        top: 0,
                        left: 0,
                        bottom: 0,
                        right: 0,
                        position: 'absolute',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexDirection: 'column',
                    }}
                >
                    <Typography variant="h4" fontWeight="bold" color={riskColor}>
                        {Math.round(riskUtilization)}%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                        Risk Used
                    </Typography>
                </Box>
            </Box>

            {/* Risk Breakdown */}
            <Stack spacing={2} flex={1}>
                <Box>
                    <Stack direction="row" justifyContent="space-between" mb={1}>
                        <Typography variant="body2" color="text.secondary">
                            Total Risk Exposure
                        </Typography>
                        <Typography variant="body2" fontWeight="bold">
                            ${totalRisk.toFixed(0)} / ${maxRiskLimit}
                        </Typography>
                    </Stack>
                    <LinearProgress
                        variant="determinate"
                        value={riskUtilization}
                        sx={{
                            height: 8,
                            borderRadius: 4,
                            backgroundColor: alpha(theme.palette.divider, 0.1),
                            '& .MuiLinearProgress-bar': {
                                backgroundColor: riskColor,
                                borderRadius: 4,
                            },
                        }}
                    />
                </Box>

                <Divider />

                {/* Position Breakdown */}
                <Stack spacing={1.5}>
                    <Stack direction="row" alignItems="center" justifyContent="space-between">
                        <Stack direction="row" alignItems="center" spacing={1}>
                            <TrendingUp sx={{ fontSize: 16, color: theme.palette.success.main }} />
                            <Typography variant="body2">Call Risk</Typography>
                        </Stack>
                        <Typography variant="body2" fontWeight="medium">
                            ${callRisk.toFixed(0)}
                        </Typography>
                    </Stack>

                    <Stack direction="row" alignItems="center" justifyContent="space-between">
                        <Stack direction="row" alignItems="center" spacing={1}>
                            <TrendingDown sx={{ fontSize: 16, color: theme.palette.error.main }} />
                            <Typography variant="body2">Put Risk</Typography>
                        </Stack>
                        <Typography variant="body2" fontWeight="medium">
                            ${putRisk.toFixed(0)}
                        </Typography>
                    </Stack>

                    <Stack direction="row" alignItems="center" justifyContent="space-between">
                        <Stack direction="row" alignItems="center" spacing={1}>
                            <Assessment sx={{ fontSize: 16, color: theme.palette.info.main }} />
                            <Typography variant="body2">Active Positions</Typography>
                        </Stack>
                        <Typography variant="body2" fontWeight="medium">
                            {activePositions}
                        </Typography>
                    </Stack>

                    <Stack direction="row" alignItems="center" justifyContent="space-between">
                        <Stack direction="row" alignItems="center" spacing={1}>
                            <Warning sx={{ fontSize: 16, color: theme.palette.warning.main }} />
                            <Typography variant="body2">Avg Stop Loss</Typography>
                        </Stack>
                        <Typography variant="body2" fontWeight="medium">
                            {avgStopLoss.toFixed(1)}%
                        </Typography>
                    </Stack>
                </Stack>

                <Divider />

                {/* Risk Alerts */}
                {riskUtilization >= 80 && (
                    <Paper
                        sx={{
                            p: 2,
                            background: alpha(theme.palette.error.main, 0.1),
                            border: `1px solid ${alpha(theme.palette.error.main, 0.3)}`,
                        }}
                    >
                        <Stack direction="row" spacing={1} alignItems="center">
                            <Warning sx={{ color: theme.palette.error.main }} />
                            <Box>
                                <Typography variant="subtitle2" color="error" fontWeight="bold">
                                    High Risk Warning
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                    Consider reducing position sizes
                                </Typography>
                            </Box>
                        </Stack>
                    </Paper>
                )}

                {riskUtilization < 30 && activePositions < 3 && (
                    <Paper
                        sx={{
                            p: 2,
                            background: alpha(theme.palette.info.main, 0.1),
                            border: `1px solid ${alpha(theme.palette.info.main, 0.3)}`,
                        }}
                    >
                        <Stack direction="row" spacing={1} alignItems="center">
                            <AttachMoney sx={{ color: theme.palette.info.main }} />
                            <Box>
                                <Typography variant="subtitle2" color="info" fontWeight="bold">
                                    Risk Capacity Available
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                    Room for additional positions
                                </Typography>
                            </Box>
                        </Stack>
                    </Paper>
                )}
            </Stack>
        </Paper>
    );
};

export default RiskMonitor; 