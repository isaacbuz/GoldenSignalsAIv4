import React, { useState } from 'react';
import { Card, CardContent, Box, Typography, LinearProgress, Chip, CircularProgress, useTheme, alpha, IconButton, Tooltip, Stack } from '@mui/material';
import { TrendingUp, TrendingDown, Timer, ContentCopy, Share, InfoOutlined, Psychology } from '@mui/icons-material';
import {
    ArrowUpIcon,
    ArrowDownIcon,
    ChartBarIcon,
    ClockIcon,
    ShieldCheckIcon,
    SparklesIcon,
    InformationCircleIcon,
    ChevronDownIcon,
    ChevronUpIcon
} from '@heroicons/react/24/outline';
import { Button } from '../Core/Button';
import { clsx } from 'clsx';

interface Target {
    price: number;
    percentage: number;
}

interface SignalLevels {
    entry: {
        price: number;
        type: 'MARKET' | 'LIMIT';
        zone?: [number, number];
    };
    targets: Target[];
    stopLoss: {
        price: number;
        type: 'STOP' | 'STOP_LIMIT';
        trailingOptions?: {
            activateAt: number;
            trailBy: number;
        };
    };
}

interface Signal {
    id: string;
    symbol: string;
    type: string;
    signal_type?: string;
    confidence: number;
    strike_price: number;
    entry_price?: number;
    expiration_date: string | Date;
    timestamp: string | Date;
    reasoning?: string;
    targets?: number[];
    stop_loss?: number;
    ai_confidence?: number;
    priority?: 'high' | 'medium' | 'low';
}

interface ConfidenceIndicatorProps {
    value: number;
}

const ConfidenceIndicator: React.FC<ConfidenceIndicatorProps> = ({ value }) => {
    const theme = useTheme();

    const getColor = () => {
        if (value >= 80) return theme.palette.success.main;
        if (value >= 60) return theme.palette.warning.main;
        return theme.palette.error.main;
    };

    return (
        <Box sx={{ position: 'relative', display: 'inline-flex' }}>
            <CircularProgress
                variant="determinate"
                value={value}
                size={60}
                thickness={6}
                sx={{
                    color: getColor(),
                    '& .MuiCircularProgress-circle': {
                        strokeLinecap: 'round',
                    },
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
                }}
            >
                <Typography
                    variant="caption"
                    component="div"
                    sx={{
                        fontWeight: 'bold',
                        color: theme.palette.text.primary
                    }}
                >
                    {`${Math.round(value)}%`}
                </Typography>
            </Box>
        </Box>
    );
};

interface TimeDecayBarProps {
    expirationDate: string | Date;
}

const TimeDecayBar: React.FC<TimeDecayBarProps> = ({ expirationDate }) => {
    const theme = useTheme();
    const now = new Date();
    const expiry = new Date(expirationDate);
    const totalTime = expiry.getTime() - now.getTime();
    const daysLeft = Math.ceil(totalTime / (1000 * 60 * 60 * 24));
    const percentLeft = Math.max(0, Math.min(100, (daysLeft / 30) * 100));

    const getColor = () => {
        if (percentLeft > 50) return theme.palette.success.main;
        if (percentLeft > 20) return theme.palette.warning.main;
        return theme.palette.error.main;
    };

    return (
        <Box sx={{ mt: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Timer fontSize="small" sx={{ color: theme.palette.text.secondary }} />
                    <Typography variant="caption" color="text.secondary">
                        Time to Expiry
                    </Typography>
                </Box>
                <Typography
                    variant="caption"
                    sx={{
                        fontWeight: 'medium',
                        color: getColor()
                    }}
                >
                    {daysLeft} days
                </Typography>
            </Box>
            <LinearProgress
                variant="determinate"
                value={percentLeft}
                sx={{
                    height: 6,
                    borderRadius: 3,
                    backgroundColor: alpha(theme.palette.grey[300], 0.3),
                    '& .MuiLinearProgress-bar': {
                        backgroundColor: getColor(),
                        borderRadius: 3,
                    }
                }}
            />
        </Box>
    );
};

interface EnhancedSignalCardProps {
    signal: Signal;
    onCopy?: () => void;
    onShare?: () => void;
    onInfo?: () => void;
}

/**
 * Enhanced SignalCard - Displays complete trade suggestions
 * 
 * Shows:
 * - Entry price/zone
 * - Multiple exit targets
 * - Stop loss level
 * - Risk/reward visualization
 * - AI confidence and reasoning
 * 
 * This is NOT for execution - just detailed suggestions!
 */
export const EnhancedSignalCard: React.FC<EnhancedSignalCardProps> = ({
    signal,
    onCopy,
    onShare,
    onInfo
}) => {
    const theme = useTheme();
    const signalType = signal.signal_type || signal.type;
    const isCall = signalType.includes('CALL');

    const getPriorityColor = () => {
        switch (signal.priority) {
            case 'high': return theme.palette.error.main;
            case 'medium': return theme.palette.warning.main;
            case 'low': return theme.palette.info.main;
            default: return theme.palette.grey[500];
        }
    };

    return (
        <Card
            sx={{
                position: 'relative',
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                borderLeft: `4px solid ${isCall ? theme.palette.success.main : theme.palette.error.main}`,
                '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: theme.shadows[8]
                },
                background: `linear-gradient(135deg, ${alpha(
                    isCall ? theme.palette.success.main : theme.palette.error.main,
                    0.05
                )} 0%, ${theme.palette.background.paper} 100%)`
            }}
        >
            {signal.priority && (
                <Box
                    sx={{
                        position: 'absolute',
                        top: 0,
                        right: 0,
                        width: 0,
                        height: 0,
                        borderStyle: 'solid',
                        borderWidth: '0 40px 40px 0',
                        borderColor: `transparent ${getPriorityColor()} transparent transparent`,
                    }}
                />
            )}

            <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                    <Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                            <Typography variant="h6" fontWeight="bold">
                                {signal.symbol}
                            </Typography>
                            <Chip
                                icon={isCall ? <TrendingUp fontSize="small" /> : <TrendingDown fontSize="small" />}
                                label={signalType}
                                color={isCall ? 'success' : 'error'}
                                size="small"
                                sx={{ fontWeight: 'medium' }}
                            />
                            {signal.ai_confidence && (
                                <Chip
                                    icon={<Psychology fontSize="small" />}
                                    label="AI"
                                    size="small"
                                    color="primary"
                                    variant="outlined"
                                />
                            )}
                        </Box>

                        <Stack spacing={0.5}>
                            <Typography variant="body2" color="text.secondary">
                                Strike: <strong>${signal.strike_price}</strong>
                            </Typography>
                            {signal.entry_price && (
                                <Typography variant="body2" color="text.secondary">
                                    Entry: <strong>${signal.entry_price}</strong>
                                </Typography>
                            )}
                            {signal.targets && signal.targets.length > 0 && (
                                <Typography variant="body2" color="text.secondary">
                                    Target 1: <strong>${signal.targets[0]}</strong>
                                </Typography>
                            )}
                            {signal.stop_loss && (
                                <Typography variant="body2" color="text.secondary">
                                    Stop Loss: <strong>${signal.stop_loss}</strong>
                                </Typography>
                            )}
                        </Stack>
                    </Box>

                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                        <ConfidenceIndicator value={signal.confidence} />
                        <Stack direction="row" spacing={0.5}>
                            <Tooltip title="Copy trade details">
                                <IconButton size="small" onClick={onCopy}>
                                    <ContentCopy fontSize="small" />
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="Share signal">
                                <IconButton size="small" onClick={onShare}>
                                    <Share fontSize="small" />
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="More info">
                                <IconButton size="small" onClick={onInfo}>
                                    <InfoOutlined fontSize="small" />
                                </IconButton>
                            </Tooltip>
                        </Stack>
                    </Box>
                </Box>

                <TimeDecayBar expirationDate={signal.expiration_date} />

                {signal.reasoning && (
                    <Box sx={{ mt: 2, p: 1.5, bgcolor: alpha(theme.palette.primary.main, 0.05), borderRadius: 1 }}>
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                            AI Reasoning:
                        </Typography>
                        <Typography variant="caption" sx={{
                            display: '-webkit-box',
                            WebkitLineClamp: 2,
                            WebkitBoxOrient: 'vertical',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis'
                        }}>
                            {signal.reasoning}
                        </Typography>
                    </Box>
                )}
            </CardContent>
        </Card>
    );
};

export default EnhancedSignalCard; 