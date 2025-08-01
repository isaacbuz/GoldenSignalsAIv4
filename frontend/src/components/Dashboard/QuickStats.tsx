/**
 * QuickStats Component - Key Metrics Display
 *
 * Shows important trading metrics in a horizontal bar format
 */

import React from 'react';
import {
    Box,
    Card,
    Stack,
    Typography,
    Chip,
    useTheme,
    alpha,
    Tooltip,
    LinearProgress,
} from '@mui/material';
import {
    TrendingUp,
    Speed,
    CheckCircle,
    AccountBalance,
    ShowChart,
    Timer,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

interface QuickStatsProps {
    totalSignals: number;
    winRate: number;
    avgReturn: number;
    activePositions: number;
}

const QuickStats: React.FC<QuickStatsProps> = ({
    totalSignals,
    winRate,
    avgReturn,
    activePositions,
}) => {
    const theme = useTheme();

    const stats = [
        {
            label: 'Active Signals',
            value: totalSignals,
            icon: <Speed />,
            color: theme.palette.primary.main,
            suffix: '',
        },
        {
            label: 'Win Rate',
            value: winRate,
            icon: <CheckCircle />,
            color: theme.palette.success.main,
            suffix: '%',
            showProgress: true,
        },
        {
            label: 'Avg Return',
            value: avgReturn,
            icon: <TrendingUp />,
            color: avgReturn >= 0 ? theme.palette.success.main : theme.palette.error.main,
            suffix: '%',
            prefix: avgReturn >= 0 ? '+' : '',
        },
        {
            label: 'Open Positions',
            value: activePositions,
            icon: <ShowChart />,
            color: theme.palette.info.main,
            suffix: '',
        },
    ];

    return (
        <Card sx={{
            background: alpha(theme.palette.background.paper, 0.6),
            backdropFilter: 'blur(10px)',
            p: 2,
        }}>
            <Stack
                direction={{ xs: 'column', sm: 'row' }}
                spacing={3}
                divider={
                    <Box sx={{
                        width: { xs: '100%', sm: '1px' },
                        height: { xs: '1px', sm: '40px' },
                        bgcolor: alpha(theme.palette.divider, 0.2),
                    }} />
                }
            >
                {stats.map((stat, index) => (
                    <motion.div
                        key={stat.label}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        style={{ flex: 1 }}
                    >
                        <Stack spacing={1}>
                            <Stack direction="row" alignItems="center" spacing={1}>
                                <Box sx={{ color: stat.color }}>
                                    {stat.icon}
                                </Box>
                                <Typography variant="body2" color="text.secondary">
                                    {stat.label}
                                </Typography>
                            </Stack>

                            <Typography variant="h4" fontWeight="bold" sx={{ color: stat.color }}>
                                {stat.prefix}{stat.value}{stat.suffix}
                            </Typography>

                            {stat.showProgress && (
                                <LinearProgress
                                    variant="determinate"
                                    value={stat.value}
                                    sx={{
                                        height: 4,
                                        borderRadius: 2,
                                        bgcolor: alpha(stat.color, 0.1),
                                        '& .MuiLinearProgress-bar': {
                                            bgcolor: stat.color,
                                        },
                                    }}
                                />
                            )}
                        </Stack>
                    </motion.div>
                ))}
            </Stack>
        </Card>
    );
};

export default QuickStats;
