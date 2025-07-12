import React from 'react';
import {
    Skeleton,
    Card,
    CardContent,
    Box,
    Stack,
    useTheme,
    alpha,
} from '@mui/material';

export const SignalCardSkeleton: React.FC = () => {
    const theme = useTheme();

    return (
        <Card
            sx={{
                borderLeft: `4px solid ${alpha(theme.palette.divider, 0.3)}`,
                background: alpha(theme.palette.background.paper, 0.8),
            }}
        >
            <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                    <Box sx={{ flex: 1 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                            <Skeleton variant="text" width={60} height={28} />
                            <Skeleton variant="rectangular" width={80} height={24} sx={{ borderRadius: 1 }} />
                            <Skeleton variant="rectangular" width={40} height={24} sx={{ borderRadius: 1 }} />
                        </Box>

                        <Stack spacing={0.5}>
                            <Skeleton variant="text" width="60%" />
                            <Skeleton variant="text" width="50%" />
                            <Skeleton variant="text" width="55%" />
                            <Skeleton variant="text" width="45%" />
                        </Stack>
                    </Box>

                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                        <Skeleton variant="circular" width={60} height={60} />
                        <Skeleton variant="rectangular" width={100} height={32} sx={{ borderRadius: 1 }} />
                    </Box>
                </Box>

                <Box sx={{ mt: 2 }}>
                    <Skeleton variant="rectangular" width="100%" height={6} sx={{ borderRadius: 3 }} />
                </Box>

                <Box sx={{ mt: 2, p: 1.5, borderRadius: 1 }}>
                    <Skeleton variant="text" width="30%" height={16} />
                    <Skeleton variant="text" width="100%" />
                    <Skeleton variant="text" width="90%" />
                </Box>
            </CardContent>
        </Card>
    );
};

export const PerformanceMetricSkeleton: React.FC = () => {
    const theme = useTheme();

    return (
        <Card sx={{ height: '100%' }}>
            <CardContent>
                <Skeleton variant="text" width="60%" height={24} />
                <Skeleton variant="text" width="40%" height={40} sx={{ mt: 1 }} />
                <Skeleton variant="rectangular" height={60} sx={{ mt: 2, borderRadius: 1 }} />
            </CardContent>
        </Card>
    );
};

export const ChartSkeleton: React.FC<{ height?: number }> = ({ height = 400 }) => {
    const theme = useTheme();

    return (
        <Box
            sx={{
                width: '100%',
                height,
                background: alpha(theme.palette.background.paper, 0.8),
                borderRadius: 2,
                p: 3,
                display: 'flex',
                flexDirection: 'column',
            }}
        >
            {/* Header */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
                <Box>
                    <Skeleton variant="text" width={120} height={32} />
                    <Skeleton variant="text" width={200} height={20} />
                </Box>
                <Box sx={{ display: 'flex', gap: 1 }}>
                    {[1, 2, 3, 4].map((i) => (
                        <Skeleton key={i} variant="rectangular" width={40} height={32} sx={{ borderRadius: 1 }} />
                    ))}
                </Box>
            </Box>

            {/* Chart Area */}
            <Box sx={{ flex: 1, position: 'relative' }}>
                <Skeleton
                    variant="rectangular"
                    width="100%"
                    height="100%"
                    sx={{
                        borderRadius: 1,
                        '&::after': {
                            background: `linear-gradient(90deg, transparent, ${alpha(theme.palette.primary.main, 0.1)}, transparent)`,
                        }
                    }}
                />
            </Box>

            {/* Footer */}
            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 3 }}>
                {[1, 2, 3].map((i) => (
                    <Box key={i} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Skeleton variant="circular" width={8} height={8} />
                        <Skeleton variant="text" width={60} />
                    </Box>
                ))}
            </Box>
        </Box>
    );
};

export const TableSkeleton: React.FC<{ rows?: number }> = ({ rows = 5 }) => {
    const theme = useTheme();

    return (
        <Box>
            {/* Table Header */}
            <Box
                sx={{
                    display: 'flex',
                    p: 2,
                    borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                    background: alpha(theme.palette.background.paper, 0.5),
                }}
            >
                {[100, 150, 120, 100, 80].map((width, i) => (
                    <Skeleton key={i} variant="text" width={width} height={20} sx={{ mr: 3 }} />
                ))}
            </Box>

            {/* Table Rows */}
            {Array.from({ length: rows }).map((_, rowIndex) => (
                <Box
                    key={rowIndex}
                    sx={{
                        display: 'flex',
                        p: 2,
                        borderBottom: `1px solid ${alpha(theme.palette.divider, 0.05)}`,
                        '&:hover': {
                            background: alpha(theme.palette.action.hover, 0.02),
                        }
                    }}
                >
                    {[100, 150, 120, 100, 80].map((width, i) => (
                        <Skeleton key={i} variant="text" width={width} height={20} sx={{ mr: 3 }} />
                    ))}
                </Box>
            ))}
        </Box>
    );
};

export const NewsSkeleton: React.FC = () => {
    return (
        <Stack spacing={2}>
            {[1, 2, 3].map((i) => (
                <Box key={i} sx={{ pb: 1.5 }}>
                    <Skeleton variant="text" width="90%" height={20} />
                    <Skeleton variant="text" width="100%" height={16} />
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                        <Skeleton variant="text" width={60} height={14} />
                        <Skeleton variant="circular" width={3} height={3} />
                        <Skeleton variant="text" width={40} height={14} />
                        <Skeleton variant="circular" width={3} height={3} />
                        <Skeleton variant="rectangular" width={50} height={16} sx={{ borderRadius: 0.5 }} />
                    </Box>
                </Box>
            ))}
        </Stack>
    );
};

export const FormSkeleton: React.FC<{ fields?: number }> = ({ fields = 4 }) => {
    return (
        <Stack spacing={3}>
            {Array.from({ length: fields }).map((_, i) => (
                <Box key={i}>
                    <Skeleton variant="text" width={120} height={20} sx={{ mb: 1 }} />
                    <Skeleton variant="rectangular" width="100%" height={40} sx={{ borderRadius: 1 }} />
                </Box>
            ))}
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end', mt: 2 }}>
                <Skeleton variant="rectangular" width={100} height={36} sx={{ borderRadius: 1 }} />
                <Skeleton variant="rectangular" width={120} height={36} sx={{ borderRadius: 1 }} />
            </Box>
        </Stack>
    );
};

export const DashboardSkeleton: React.FC = () => {
    return (
        <Box>
            {/* Header */}
            <Box sx={{ mb: 4 }}>
                <Skeleton variant="text" width={300} height={40} />
                <Skeleton variant="text" width={500} height={24} />
            </Box>

            {/* Metrics */}
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 2, mb: 4 }}>
                {[1, 2, 3, 4].map((i) => (
                    <PerformanceMetricSkeleton key={i} />
                ))}
            </Box>

            {/* Main Content */}
            <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3 }}>
                <ChartSkeleton />
                <Box>
                    <Skeleton variant="text" width={150} height={28} sx={{ mb: 2 }} />
                    <Stack spacing={2}>
                        {[1, 2, 3].map((i) => (
                            <SignalCardSkeleton key={i} />
                        ))}
                    </Stack>
                </Box>
            </Box>
        </Box>
    );
};

// Shimmer effect component
export const ShimmerSkeleton: React.FC<{ width?: string | number; height?: string | number }> = ({
    width = '100%',
    height = 20
}) => {
    const theme = useTheme();

    return (
        <Box
            sx={{
                width,
                height,
                background: `linear-gradient(90deg, 
          ${alpha(theme.palette.action.hover, 0.05)} 0%, 
          ${alpha(theme.palette.action.hover, 0.1)} 50%, 
          ${alpha(theme.palette.action.hover, 0.05)} 100%)`,
                backgroundSize: '200% 100%',
                animation: 'shimmer 1.5s infinite',
                borderRadius: 1,
                '@keyframes shimmer': {
                    '0%': { backgroundPosition: '200% 0' },
                    '100%': { backgroundPosition: '-200% 0' },
                },
            }}
        />
    );
};

// Export LoadingStates as a collection of all skeleton components
export const LoadingStates = {
    SignalCard: SignalCardSkeleton,
    PerformanceMetric: PerformanceMetricSkeleton,
    Chart: ChartSkeleton,
    Table: TableSkeleton,
    News: NewsSkeleton,
    Form: FormSkeleton,
    Dashboard: DashboardSkeleton,
    Shimmer: ShimmerSkeleton,
};
