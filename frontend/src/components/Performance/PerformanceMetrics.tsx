import React from 'react';
import { Grid, Card, CardContent, Typography, Box, useTheme, alpha, Skeleton } from '@mui/material';
import { TrendingUp, TrendingDown, ShowChart, Timeline } from '@mui/icons-material';

interface MetricCardProps {
    title: string;
    value: string | number;
    change?: number | null;
    color?: 'primary' | 'success' | 'error' | 'warning' | 'info';
    icon?: React.ReactNode;
    loading?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({
    title,
    value,
    change,
    color = 'primary',
    icon,
    loading = false
}) => {
    const theme = useTheme();

    if (loading) {
        return (
            <Card sx={{ height: '100%' }}>
                <CardContent>
                    <Skeleton variant="text" width="60%" height={24} />
                    <Skeleton variant="text" width="40%" height={40} sx={{ mt: 1 }} />
                    <Skeleton variant="rectangular" height={60} sx={{ mt: 2 }} />
                </CardContent>
            </Card>
        );
    }

    return (
        <Card
            sx={{
                height: '100%',
                background: `linear-gradient(135deg, ${alpha(theme.palette[color].main, 0.05)} 0%, ${alpha(theme.palette.background.paper, 0.8)} 100%)`,
                borderTop: `3px solid ${theme.palette[color].main}`,
                transition: 'all 0.3s ease',
                '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: theme.shadows[8]
                }
            }}
        >
            <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                    <Typography
                        color="text.secondary"
                        variant="body2"
                        sx={{ fontWeight: 'medium' }}
                    >
                        {title}
                    </Typography>
                    {icon && (
                        <Box sx={{ color: theme.palette[color].main }}>
                            {icon}
                        </Box>
                    )}
                </Box>

                <Typography
                    variant="h4"
                    fontWeight="bold"
                    sx={{
                        color: theme.palette.text.primary,
                        mb: 1
                    }}
                >
                    {value}
                </Typography>

                {change !== null && change !== undefined && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        {change > 0 ? (
                            <TrendingUp fontSize="small" sx={{ color: theme.palette.success.main }} />
                        ) : (
                            <TrendingDown fontSize="small" sx={{ color: theme.palette.error.main }} />
                        )}
                        <Typography
                            variant="body2"
                            sx={{
                                color: change > 0 ? theme.palette.success.main : theme.palette.error.main,
                                fontWeight: 'medium'
                            }}
                        >
                            {change > 0 ? '+' : ''}{change}%
                        </Typography>
                    </Box>
                )}
            </CardContent>
        </Card>
    );
};

const PerformanceMetrics: React.FC = () => {
    return (
        <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                    title="Win Rate (30D)"
                    value="73.5%"
                    change={2.3}
                    icon={<ShowChart />}
                    color="success"
                />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                    title="Avg Return"
                    value="+12.4%"
                    change={0.8}
                    color="primary"
                    icon={<TrendingUp />}
                />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                    title="Sharpe Ratio"
                    value="2.31"
                    change={0.15}
                    color="info"
                    icon={<Timeline />}
                />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
                <MetricCard
                    title="Active Signals"
                    value="8/10"
                    change={null}
                    color="warning"
                />
            </Grid>
        </Grid>
    );
};

export default PerformanceMetrics;
