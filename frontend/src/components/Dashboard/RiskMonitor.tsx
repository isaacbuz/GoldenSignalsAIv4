/**
 * RiskMonitor Component - Real-time Risk Management Display
 * 
 * Shows current risk exposure, position limits, and risk metrics
 */

import React from 'react';
import {
    Card,
    CardContent,
    Stack,
    Typography,
    Box,
    LinearProgress,
    Chip,
    Alert,
    useTheme,
    alpha,
    Tooltip,
    IconButton,
} from '@mui/material';
import {
    Warning,
    Shield,
    TrendingUp,
    TrendingDown,
    Info,
    AttachMoney,
    Speed,
} from '@mui/icons-material';
import { RiskMetrics, PreciseOptionsSignal } from '../../types/signals';

interface RiskMonitorProps {
    metrics?: RiskMetrics;
    activeSignals?: PreciseOptionsSignal[];
}

const RiskMonitor: React.FC<RiskMonitorProps> = ({ metrics, activeSignals = [] }) => {
    const theme = useTheme();

    // Default metrics if not provided
    const riskData = metrics || {
        activePositions: 0,
        totalExposure: 0,
        maxDrawdown: 0,
        sharpeRatio: 0,
        currentRisk: 0,
        riskLimit: 10000,
        utilizationPct: 0,
    };

    const riskLevel = riskData.utilizationPct > 80 ? 'high' :
        riskData.utilizationPct > 50 ? 'medium' : 'low';

    const riskColor = riskLevel === 'high' ? theme.palette.error.main :
        riskLevel === 'medium' ? theme.palette.warning.main :
            theme.palette.success.main;

    // Calculate position breakdown
    const callPositions = activeSignals.filter(s => s.signal_type === 'BUY_CALL').length;
    const putPositions = activeSignals.filter(s => s.signal_type === 'BUY_PUT').length;

    return (
        <Card sx={{
            background: alpha(theme.palette.background.paper, 0.6),
            backdropFilter: 'blur(10px)',
            border: `1px solid ${alpha(riskColor, 0.3)}`,
        }}>
            <CardContent>
                <Stack direction="row" alignItems="center" justifyContent="space-between" mb={2}>
                    <Stack direction="row" alignItems="center" spacing={1}>
                        <Shield sx={{ color: riskColor }} />
                        <Typography variant="h6" fontWeight="bold">
                            Risk Monitor
                        </Typography>
                    </Stack>
                    <Tooltip title="Risk management overview">
                        <IconButton size="small">
                            <Info fontSize="small" />
                        </IconButton>
                    </Tooltip>
                </Stack>

                {/* Risk Utilization Bar */}
                <Box mb={3}>
                    <Stack direction="row" justifyContent="space-between" mb={1}>
                        <Typography variant="body2" color="text.secondary">
                            Risk Utilization
                        </Typography>
                        <Typography variant="body2" fontWeight="bold" sx={{ color: riskColor }}>
                            {riskData.utilizationPct.toFixed(1)}%
                        </Typography>
                    </Stack>
                    <LinearProgress
                        variant="determinate"
                        value={riskData.utilizationPct}
                        sx={{
                            height: 8,
                            borderRadius: 4,
                            bgcolor: alpha(theme.palette.divider, 0.1),
                            '& .MuiLinearProgress-bar': {
                                bgcolor: riskColor,
                                borderRadius: 4,
                            },
                        }}
                    />
                    <Stack direction="row" justifyContent="space-between" mt={0.5}>
                        <Typography variant="caption" color="text.secondary">
                            ${riskData.currentRisk.toLocaleString()}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            ${riskData.riskLimit.toLocaleString()} limit
                        </Typography>
                    </Stack>
                </Box>

                {/* Risk Alert */}
                {riskLevel === 'high' && (
                    <Alert
                        severity="warning"
                        icon={<Warning />}
                        sx={{ mb: 2 }}
                    >
                        High risk utilization. Consider reducing position sizes.
                    </Alert>
                )}

                {/* Position Breakdown */}
                <Box mb={2}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                        Position Breakdown
                    </Typography>
                    <Stack direction="row" spacing={1}>
                        <Chip
                            icon={<TrendingUp />}
                            label={`${callPositions} Calls`}
                            size="small"
                            sx={{
                                bgcolor: alpha(theme.palette.success.main, 0.1),
                                color: theme.palette.success.main,
                            }}
                        />
                        <Chip
                            icon={<TrendingDown />}
                            label={`${putPositions} Puts`}
                            size="small"
                            sx={{
                                bgcolor: alpha(theme.palette.error.main, 0.1),
                                color: theme.palette.error.main,
                            }}
                        />
                    </Stack>
                </Box>

                {/* Key Metrics */}
                <Stack spacing={1.5}>
                    <Stack direction="row" justifyContent="space-between">
                        <Typography variant="body2" color="text.secondary">
                            Total Exposure
                        </Typography>
                        <Typography variant="body2" fontWeight="bold">
                            ${riskData.totalExposure.toLocaleString()}
                        </Typography>
                    </Stack>

                    <Stack direction="row" justifyContent="space-between">
                        <Typography variant="body2" color="text.secondary">
                            Max Drawdown
                        </Typography>
                        <Typography
                            variant="body2"
                            fontWeight="bold"
                            color={riskData.maxDrawdown > 10 ? 'error.main' : 'text.primary'}
                        >
                            -{riskData.maxDrawdown.toFixed(1)}%
                        </Typography>
                    </Stack>

                    <Stack direction="row" justifyContent="space-between">
                        <Typography variant="body2" color="text.secondary">
                            Sharpe Ratio
                        </Typography>
                        <Typography
                            variant="body2"
                            fontWeight="bold"
                            color={riskData.sharpeRatio > 1 ? 'success.main' : 'text.primary'}
                        >
                            {riskData.sharpeRatio.toFixed(2)}
                        </Typography>
                    </Stack>
                </Stack>

                {/* Risk Status */}
                <Box mt={2} p={1.5} sx={{
                    bgcolor: alpha(riskColor, 0.1),
                    borderRadius: 1,
                    textAlign: 'center',
                }}>
                    <Typography variant="body2" fontWeight="bold" sx={{ color: riskColor }}>
                        Risk Status: {riskLevel.toUpperCase()}
                    </Typography>
                </Box>
            </CardContent>
        </Card>
    );
};

export default RiskMonitor; 