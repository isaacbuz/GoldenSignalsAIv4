/**
 * SignalCard Component - Professional Options Signal Display
 * 
 * Displays a single options signal in a card format with all critical information
 * Designed for quick scanning and decision making
 */

import React from 'react';
import {
    Card,
    CardContent,
    Stack,
    Typography,
    Chip,
    Box,
    IconButton,
    Tooltip,
    LinearProgress,
    Divider,
    useTheme,
    alpha,
    Grid,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    Timer,
    AttachMoney,
    ShowChart,
    Notifications,
    ContentCopy,
    MoreVert,
    Speed,
    Warning,
    CheckCircle,
    Schedule,
    TrendingFlat,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { PreciseOptionsSignal } from '../../types/signals';

interface SignalCardProps {
    signal: PreciseOptionsSignal;
    onClick: () => void;
    onQuickAction?: (action: string) => void;
    selected?: boolean;
    compact?: boolean;
}

const SignalCard: React.FC<SignalCardProps> = ({ signal, onClick, onQuickAction, selected = false, compact = false }) => {
    const theme = useTheme();
    const isCall = signal.type === 'CALL';
    const signalColor = isCall ? theme.palette.success.main : theme.palette.error.main;

    // Calculate time until entry with defensive checks
    const now = new Date();
    const entryTimeString = signal.entry_window?.start_time || signal.timestamp;
    const entryTime = new Date(entryTimeString);
    const hoursUntilEntry = Math.max(0, (entryTime.getTime() - now.getTime()) / (1000 * 60 * 60));

    // Determine urgency
    const urgencyLevel = hoursUntilEntry < 1 ? 'urgent' : hoursUntilEntry < 24 ? 'today' : 'upcoming';
    const urgencyColor = urgencyLevel === 'urgent' ? 'error' : urgencyLevel === 'today' ? 'warning' : 'info';

    // Compact version for sidebar
    if (compact) {
        return (
            <Card
                sx={{
                    background: selected
                        ? alpha(signalColor, 0.1)
                        : alpha(theme.palette.background.paper, 0.9),
                    backdropFilter: 'blur(10px)',
                    border: `1px solid ${selected ? signalColor : alpha(theme.palette.divider, 0.2)}`,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    '&:hover': {
                        borderColor: signalColor,
                        transform: 'translateX(4px)',
                    },
                }}
                onClick={onClick}
            >
                <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                    <Stack spacing={1}>
                        {/* Header */}
                        <Stack direction="row" alignItems="center" justifyContent="space-between">
                            <Stack direction="row" alignItems="center" spacing={1}>
                                <Typography variant="subtitle1" fontWeight="bold">
                                    {signal.symbol}
                                </Typography>
                                {isCall ? (
                                    <TrendingUp sx={{ fontSize: 16, color: signalColor }} />
                                ) : (
                                    <TrendingDown sx={{ fontSize: 16, color: signalColor }} />
                                )}
                            </Stack>
                            <Chip
                                label={`${signal.confidence}%`}
                                size="small"
                                color={signal.confidence >= 80 ? 'success' : 'warning'}
                            />
                        </Stack>

                        {/* Strike & Entry */}
                        <Stack direction="row" justifyContent="space-between">
                            <Typography variant="body2" color="text.secondary">
                                ${signal.strike_price || 0} {signal.type}
                            </Typography>
                            <Typography variant="body2" fontWeight="medium">
                                ${(signal.entry_price || 0).toFixed(2)}
                            </Typography>
                        </Stack>

                        {/* Timing */}
                        <Chip
                            icon={<Timer />}
                            label={
                                urgencyLevel === 'urgent'
                                    ? 'Now!'
                                    : urgencyLevel === 'today'
                                        ? `${Math.floor(hoursUntilEntry)}h`
                                        : signal.entry_window?.date || 'Upcoming'
                            }
                            size="small"
                            color={urgencyColor}
                            sx={{ width: '100%' }}
                        />
                    </Stack>
                </CardContent>
            </Card>
        );
    }

    // Full version
    return (
        <motion.div
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
        >
            <Card
                sx={{
                    background: alpha(theme.palette.background.paper, 0.9),
                    backdropFilter: 'blur(10px)',
                    border: `1px solid ${alpha(signalColor, 0.3)}`,
                    borderLeft: `4px solid ${signalColor}`,
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                        boxShadow: `0 8px 32px ${alpha(signalColor, 0.2)}`,
                        borderColor: alpha(signalColor, 0.5),
                    },
                }}
                onClick={onClick}
            >
                <CardContent>
                    <Grid container spacing={2}>
                        {/* Left Section - Symbol & Type */}
                        <Grid item xs={12} sm={3}>
                            <Stack spacing={1}>
                                <Stack direction="row" alignItems="center" spacing={1}>
                                    <Typography variant="h5" fontWeight="bold">
                                        {signal.symbol}
                                    </Typography>
                                    {isCall ? (
                                        <TrendingUp sx={{ color: signalColor }} />
                                    ) : (
                                        <TrendingDown sx={{ color: signalColor }} />
                                    )}
                                </Stack>

                                <Chip
                                    label={signal.type}
                                    size="small"
                                    sx={{
                                        backgroundColor: alpha(signalColor, 0.1),
                                        color: signalColor,
                                        fontWeight: 'bold',
                                    }}
                                />

                                <Stack direction="row" spacing={0.5}>
                                    <Chip
                                        icon={<Speed />}
                                        label={`${signal.confidence}%`}
                                        size="small"
                                        color={signal.confidence >= 80 ? 'success' : 'warning'}
                                        variant="outlined"
                                    />
                                    <Chip
                                        label={signal.priority}
                                        size="small"
                                        color={signal.priority === 'HIGH' ? 'error' : 'default'}
                                        variant="outlined"
                                    />
                                </Stack>
                            </Stack>
                        </Grid>

                        {/* Middle Section - Entry & Contract Details */}
                        <Grid item xs={12} sm={5}>
                            <Stack spacing={1.5}>
                                {/* Entry Details */}
                                <Box>
                                    <Typography variant="caption" color="text.secondary">
                                        ENTRY DETAILS
                                    </Typography>
                                    <Stack direction="row" spacing={2}>
                                        <Box>
                                            <Typography variant="body2" color="text.secondary">
                                                Entry
                                            </Typography>
                                            <Typography variant="h6" fontWeight="bold">
                                                ${(signal.entry_price || 0).toFixed(2)}
                                            </Typography>
                                        </Box>
                                        <Box>
                                            <Typography variant="body2" color="text.secondary">
                                                Strike
                                            </Typography>
                                            <Typography variant="h6" fontWeight="bold">
                                                ${signal.strike_price || 0}
                                            </Typography>
                                        </Box>
                                        <Box>
                                            <Typography variant="body2" color="text.secondary">
                                                Expiry
                                            </Typography>
                                            <Typography variant="h6" fontWeight="bold">
                                                {new Date(signal.expiration_date).toLocaleDateString('en-US', {
                                                    month: 'short',
                                                    day: 'numeric'
                                                })}
                                            </Typography>
                                        </Box>
                                    </Stack>
                                </Box>

                                {/* Risk/Reward */}
                                <Box>
                                    <Typography variant="caption" color="text.secondary">
                                        RISK / REWARD
                                    </Typography>
                                    <Stack direction="row" spacing={2} alignItems="center">
                                        <Box>
                                            <Typography variant="body2" color="error.main">
                                                Stop: ${(signal.stop_loss || 0).toFixed(2)}
                                            </Typography>
                                        </Box>
                                        <TrendingFlat sx={{ color: 'text.secondary' }} />
                                        <Box>
                                            <Typography variant="body2" color="success.main">
                                                Target: ${(signal.targets?.[0]?.price || 0).toFixed(2)}
                                            </Typography>
                                        </Box>
                                        <Chip
                                            label={`${signal.risk_reward_ratio || 0}:1 R:R`}
                                            size="small"
                                            color="primary"
                                            variant="outlined"
                                        />
                                    </Stack>
                                </Box>
                            </Stack>
                        </Grid>

                        {/* Right Section - Timing & Actions */}
                        <Grid item xs={12} sm={4}>
                            <Stack spacing={1} alignItems="flex-end">
                                {/* Timing Alert */}
                                <Chip
                                    icon={<Timer />}
                                    label={
                                        urgencyLevel === 'urgent'
                                            ? 'Enter Now!'
                                            : urgencyLevel === 'today'
                                                ? `In ${Math.floor(hoursUntilEntry)}h`
                                                : signal.entry_window?.date || 'Upcoming'
                                    }
                                    color={urgencyColor}
                                    size="small"
                                />

                                {/* Entry Window */}
                                {signal.entry_window && (
                                    <Typography variant="caption" color="text.secondary" textAlign="right">
                                        {signal.entry_window?.start_time} - {signal.entry_window?.end_time}
                                    </Typography>
                                )}

                                {/* Quick Actions */}
                                <Stack direction="row" spacing={0.5}>
                                    <Tooltip title="Set Alert">
                                        <IconButton
                                            size="small"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                onQuickAction?.('setAlert');
                                            }}
                                        >
                                            <Notifications fontSize="small" />
                                        </IconButton>
                                    </Tooltip>
                                    <Tooltip title="Copy Trade">
                                        <IconButton
                                            size="small"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                onQuickAction?.('copyTrade');
                                            }}
                                        >
                                            <ContentCopy fontSize="small" />
                                        </IconButton>
                                    </Tooltip>
                                    <Tooltip title="View Chart">
                                        <IconButton
                                            size="small"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                onQuickAction?.('analyze');
                                            }}
                                        >
                                            <ShowChart fontSize="small" />
                                        </IconButton>
                                    </Tooltip>
                                </Stack>

                                {/* Setup Name */}
                                {signal.setup_name && (
                                    <Chip
                                        label={signal.setup_name}
                                        size="small"
                                        variant="outlined"
                                        sx={{ mt: 1 }}
                                    />
                                )}
                            </Stack>
                        </Grid>
                    </Grid>

                    {/* Bottom Section - Key Indicators */}
                    {signal.key_indicators && (
                        <>
                            <Divider sx={{ my: 2 }} />
                            <Stack direction="row" spacing={1} flexWrap="wrap">
                                {Object.entries(signal.key_indicators).slice(0, 4).map(([key, value]) => (
                                    <Chip
                                        key={key}
                                        label={`${key}: ${value}`}
                                        size="small"
                                        variant="outlined"
                                        sx={{
                                            borderColor: alpha(theme.palette.divider, 0.3),
                                            fontSize: '0.75rem',
                                        }}
                                    />
                                ))}
                            </Stack>
                        </>
                    )}
                </CardContent>
            </Card>
        </motion.div>
    );
};

export default SignalCard; 