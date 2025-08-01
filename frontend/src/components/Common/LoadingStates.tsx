import React from 'react';
import {
    Box,
    CircularProgress,
    Skeleton,
    Typography,
    Stack,
    Card,
    LinearProgress,
    useTheme,
    alpha,
} from '@mui/material';
import { Psychology, TrendingUp, Analytics } from '@mui/icons-material';

interface LoadingStateProps {
    type?: 'signal' | 'chart' | 'agent' | 'data' | 'ai-analysis';
    message?: string;
    progress?: number;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
    type = 'data',
    message,
    progress
}) => {
    const theme = useTheme();

    const getIcon = () => {
        switch (type) {
            case 'signal':
                return <TrendingUp sx={{ fontSize: 48, color: theme.palette.primary.main }} />;
            case 'agent':
                return <Psychology sx={{ fontSize: 48, color: theme.palette.primary.main }} />;
            case 'ai-analysis':
                return <Analytics sx={{ fontSize: 48, color: theme.palette.primary.main }} />;
            default:
                return null;
        }
    };

    const getMessage = () => {
        if (message) return message;

        switch (type) {
            case 'signal':
                return 'AI agents analyzing market patterns...';
            case 'chart':
                return 'Loading chart data...';
            case 'agent':
                return 'Initializing AI agents...';
            case 'ai-analysis':
                return 'Running AI analysis...';
            default:
                return 'Loading...';
        }
    };

    return (
        <Box
            sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: 200,
                p: 3,
            }}
        >
            <div
                style={{
                    opacity: 0,
                    transform: 'scale(0.9)',
                    transition: 'opacity 0.3s, transform 0.3s',
                }}
            >
                <Stack spacing={3} alignItems="center">
                    {getIcon()}

                    <Box sx={{ position: 'relative' }}>
                        <CircularProgress
                            size={60}
                            thickness={2}
                            sx={{
                                color: theme.palette.primary.main,
                            }}
                        />
                        {progress !== undefined && (
                            <Box
                                sx={{
                                    position: 'absolute',
                                    top: '50%',
                                    left: '50%',
                                    transform: 'translate(-50%, -50%)',
                                }}
                            >
                                <Typography variant="caption" fontWeight="bold">
                                    {Math.round(progress)}%
                                </Typography>
                            </Box>
                        )}
                    </Box>

                    <Typography variant="body1" color="text.secondary">
                        {getMessage()}
                    </Typography>

                    {progress !== undefined && (
                        <Box sx={{ width: 300 }}>
                            <LinearProgress
                                variant="determinate"
                                value={progress}
                                sx={{
                                    height: 6,
                                    borderRadius: 3,
                                    backgroundColor: alpha(theme.palette.primary.main, 0.1),
                                    '& .MuiLinearProgress-bar': {
                                        borderRadius: 3,
                                        background: `linear-gradient(90deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
                                    },
                                }}
                            />
                        </Box>
                    )}
                </Stack>
            </div>
        </Box>
    );
};

// Signal Card Skeleton
export const SignalCardSkeleton: React.FC = () => {
    const theme = useTheme();

    return (
        <Card sx={{ p: 3 }}>
            <Stack spacing={2}>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                    <Stack direction="row" spacing={1} alignItems="center">
                        <Skeleton variant="circular" width={40} height={40} />
                        <Box>
                            <Skeleton variant="text" width={80} height={24} />
                            <Skeleton variant="text" width={60} height={20} />
                        </Box>
                    </Stack>
                    <Skeleton variant="rectangular" width={80} height={32} sx={{ borderRadius: 1 }} />
                </Stack>

                <Skeleton variant="text" width="100%" height={20} />
                <Skeleton variant="text" width="80%" height={20} />

                <Stack direction="row" spacing={1}>
                    <Skeleton variant="rectangular" width={60} height={24} sx={{ borderRadius: 3 }} />
                    <Skeleton variant="rectangular" width={60} height={24} sx={{ borderRadius: 3 }} />
                    <Skeleton variant="rectangular" width={60} height={24} sx={{ borderRadius: 3 }} />
                </Stack>
            </Stack>
        </Card>
    );
};

// Chart Skeleton
export const ChartSkeleton: React.FC<{ height?: number }> = ({ height = 400 }) => {
    return (
        <Box sx={{ width: '100%', height }}>
            <Stack spacing={2} sx={{ height: '100%' }}>
                <Stack direction="row" justifyContent="space-between">
                    <Skeleton variant="text" width={150} height={32} />
                    <Stack direction="row" spacing={1}>
                        <Skeleton variant="rectangular" width={60} height={32} sx={{ borderRadius: 1 }} />
                        <Skeleton variant="rectangular" width={60} height={32} sx={{ borderRadius: 1 }} />
                        <Skeleton variant="rectangular" width={60} height={32} sx={{ borderRadius: 1 }} />
                    </Stack>
                </Stack>
                <Skeleton
                    variant="rectangular"
                    sx={{
                        flex: 1,
                        borderRadius: 2,
                        background: `linear-gradient(180deg, ${alpha('#f5f5f5', 0.8)} 0%, ${alpha('#e0e0e0', 0.8)} 100%)`,
                    }}
                />
            </Stack>
        </Box>
    );
};

// Agent Card Skeleton
export const AgentCardSkeleton: React.FC = () => {
    return (
        <Card sx={{ p: 2 }}>
            <Stack spacing={2}>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                    <Stack direction="row" spacing={1} alignItems="center">
                        <Skeleton variant="circular" width={48} height={48} />
                        <Box>
                            <Skeleton variant="text" width={120} height={24} />
                            <Skeleton variant="rectangular" width={60} height={20} sx={{ borderRadius: 3, mt: 0.5 }} />
                        </Box>
                    </Stack>
                    <Skeleton variant="circular" width={32} height={32} />
                </Stack>

                <Box>
                    <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                        <Skeleton variant="text" width={60} />
                        <Skeleton variant="text" width={40} />
                    </Stack>
                    <Skeleton variant="rectangular" height={4} sx={{ borderRadius: 2 }} />
                </Box>
            </Stack>
        </Card>
    );
};

// Table Skeleton
export const TableSkeleton: React.FC<{ rows?: number }> = ({ rows = 5 }) => {
    return (
        <Box>
            <Skeleton variant="rectangular" height={56} sx={{ mb: 1 }} />
            {Array.from({ length: rows }).map((_, index) => (
                <Skeleton
                    key={index}
                    variant="rectangular"
                    height={52}
                    sx={{ mb: 0.5, opacity: 1 - (index * 0.1) }}
                />
            ))}
        </Box>
    );
};

export default LoadingState;
