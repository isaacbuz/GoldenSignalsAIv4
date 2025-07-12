/**
 * Live Data Status Component
 * 
 * Shows real-time connection status, data flow statistics, and system health
 */

import React, { useState, useEffect } from 'react';
import {
    Box,
    Card,
    CardContent,
    Typography,
    Stack,
    Chip,
    IconButton,
    Tooltip,
    LinearProgress,
    Grid,
    Divider,
    Badge,
    useTheme,
    alpha,
} from '@mui/material';
import {
    Wifi,
    WifiOff,
    Speed,
    Timeline,
    Analytics,
    Refresh,
    Info,
    Warning,
    CheckCircle,
    Error as ErrorIcon,
    TrendingUp,
    ShowChart,
    Psychology,
    BarChart,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { liveDataService, LiveMetrics } from '../../services/LiveDataService';

interface LiveDataStatusProps {
    compact?: boolean;
    showDetails?: boolean;
}

export const LiveDataStatus: React.FC<LiveDataStatusProps> = ({
    compact = false,
    showDetails = true,
}) => {
    const theme = useTheme();
    const [isConnected, setIsConnected] = useState(false);
    const [stats, setStats] = useState<any>(null);
    const [metrics, setMetrics] = useState<LiveMetrics | null>(null);
    const [lastUpdate, setLastUpdate] = useState<string | null>(null);

    useEffect(() => {
        // Subscribe to connection status
        const unsubscribeConnection = liveDataService.subscribe('connected', () => {
            setIsConnected(true);
        });

        const unsubscribeDisconnection = liveDataService.subscribe('disconnected', () => {
            setIsConnected(false);
        });

        const unsubscribeMetrics = liveDataService.subscribe('metrics', (newMetrics: LiveMetrics) => {
            setMetrics(newMetrics);
        });

        const unsubscribeUpdate = liveDataService.subscribe('update', (update) => {
            setLastUpdate(update.timestamp);
        });

        // Update stats periodically
        const updateStats = () => {
            const currentStats = liveDataService.getStats();
            setStats(currentStats);
            setIsConnected(currentStats.isConnected);
        };

        updateStats();
        const interval = setInterval(updateStats, 5000);

        return () => {
            unsubscribeConnection();
            unsubscribeDisconnection();
            unsubscribeMetrics();
            unsubscribeUpdate();
            clearInterval(interval);
        };
    }, []);

    const getHealthColor = (health?: string) => {
        switch (health) {
            case 'healthy': return theme.palette.success.main;
            case 'warning': return theme.palette.warning.main;
            case 'error': return theme.palette.error.main;
            default: return theme.palette.text.secondary;
        }
    };

    const getHealthIcon = (health?: string) => {
        switch (health) {
            case 'healthy': return <CheckCircle sx={{ color: theme.palette.success.main }} />;
            case 'warning': return <Warning sx={{ color: theme.palette.warning.main }} />;
            case 'error': return <ErrorIcon sx={{ color: theme.palette.error.main }} />;
            default: return <Info sx={{ color: theme.palette.text.secondary }} />;
        }
    };

    const formatUptime = (uptime: number) => {
        const seconds = Math.floor(uptime / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);

        if (hours > 0) {
            return `${hours}h ${minutes % 60}m`;
        } else if (minutes > 0) {
            return `${minutes}m ${seconds % 60}s`;
        } else {
            return `${seconds}s`;
        }
    };

    if (compact) {
        return (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    bgcolor: isConnected ? theme.palette.success.main : theme.palette.error.main,
                    animation: isConnected ? 'pulse 2s infinite' : 'none',
                }} />
                <Typography variant="caption" sx={{
                    color: isConnected ? theme.palette.success.main : theme.palette.error.main,
                    fontWeight: 600
                }}>
                    {isConnected ? 'LIVE' : 'OFFLINE'}
                </Typography>
                {stats && (
                    <Typography variant="caption" sx={{ color: theme.palette.text.secondary }}>
                        {stats.totalUpdates} updates
                    </Typography>
                )}
            </Box>
        );
    }

    return (
        <Card sx={{
            background: alpha(theme.palette.background.paper, 0.8),
            backdropFilter: 'blur(10px)',
            border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        }}>
            <CardContent sx={{ p: 2 }}>
                <Stack spacing={2}>
                    {/* Header */}
                    <Stack direction="row" alignItems="center" justifyContent="space-between">
                        <Stack direction="row" alignItems="center" spacing={1}>
                            {isConnected ? (
                                <Wifi sx={{ color: theme.palette.success.main }} />
                            ) : (
                                <WifiOff sx={{ color: theme.palette.error.main }} />
                            )}
                            <Typography variant="h6" fontWeight={600}>
                                Live Data Status
                            </Typography>
                        </Stack>

                        <Stack direction="row" spacing={1}>
                            {metrics && (
                                <Tooltip title={`System Health: ${metrics.system_health}`}>
                                    <Box>{getHealthIcon(metrics.system_health)}</Box>
                                </Tooltip>
                            )}
                            <Tooltip title="Refresh Data">
                                <IconButton
                                    size="small"
                                    onClick={() => liveDataService.refreshAll()}
                                >
                                    <Refresh />
                                </IconButton>
                            </Tooltip>
                        </Stack>
                    </Stack>

                    {/* Connection Status */}
                    <Stack direction="row" alignItems="center" spacing={2}>
                        <Chip
                            label={isConnected ? 'CONNECTED' : 'DISCONNECTED'}
                            color={isConnected ? 'success' : 'error'}
                            size="small"
                            icon={isConnected ? <CheckCircle /> : <ErrorIcon />}
                        />
                        {lastUpdate && (
                            <Typography variant="caption" sx={{ color: theme.palette.text.secondary }}>
                                Last update: {new Date(lastUpdate).toLocaleTimeString()}
                            </Typography>
                        )}
                    </Stack>

                    {showDetails && stats && (
                        <>
                            <Divider />

                            {/* Statistics Grid */}
                            <Grid container spacing={2}>
                                <Grid item xs={6}>
                                    <Stack spacing={1}>
                                        <Typography variant="caption" color="text.secondary">
                                            Total Updates
                                        </Typography>
                                        <Stack direction="row" alignItems="center" spacing={1}>
                                            <Timeline sx={{ fontSize: 16, color: theme.palette.primary.main }} />
                                            <Typography variant="body2" fontWeight={600}>
                                                {stats.totalUpdates.toLocaleString()}
                                            </Typography>
                                        </Stack>
                                    </Stack>
                                </Grid>

                                <Grid item xs={6}>
                                    <Stack spacing={1}>
                                        <Typography variant="caption" color="text.secondary">
                                            Signal Updates
                                        </Typography>
                                        <Stack direction="row" alignItems="center" spacing={1}>
                                            <TrendingUp sx={{ fontSize: 16, color: theme.palette.success.main }} />
                                            <Typography variant="body2" fontWeight={600}>
                                                {stats.signalUpdates.toLocaleString()}
                                            </Typography>
                                        </Stack>
                                    </Stack>
                                </Grid>

                                <Grid item xs={6}>
                                    <Stack spacing={1}>
                                        <Typography variant="caption" color="text.secondary">
                                            Market Updates
                                        </Typography>
                                        <Stack direction="row" alignItems="center" spacing={1}>
                                            <ShowChart sx={{ fontSize: 16, color: theme.palette.info.main }} />
                                            <Typography variant="body2" fontWeight={600}>
                                                {stats.marketUpdates.toLocaleString()}
                                            </Typography>
                                        </Stack>
                                    </Stack>
                                </Grid>

                                <Grid item xs={6}>
                                    <Stack spacing={1}>
                                        <Typography variant="caption" color="text.secondary">
                                            Agent Updates
                                        </Typography>
                                        <Stack direction="row" alignItems="center" spacing={1}>
                                            <Psychology sx={{ fontSize: 16, color: theme.palette.secondary.main }} />
                                            <Typography variant="body2" fontWeight={600}>
                                                {stats.agentUpdates.toLocaleString()}
                                            </Typography>
                                        </Stack>
                                    </Stack>
                                </Grid>
                            </Grid>

                            {/* System Metrics */}
                            {metrics && (
                                <>
                                    <Divider />
                                    <Grid container spacing={2}>
                                        <Grid item xs={12}>
                                            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                                                System Performance
                                            </Typography>
                                        </Grid>

                                        <Grid item xs={6}>
                                            <Stack spacing={1}>
                                                <Typography variant="caption" color="text.secondary">
                                                    CPU Usage
                                                </Typography>
                                                <Box>
                                                    <LinearProgress
                                                        variant="determinate"
                                                        value={metrics.cpu_usage}
                                                        sx={{ height: 6, borderRadius: 3 }}
                                                    />
                                                    <Typography variant="caption">
                                                        {metrics.cpu_usage.toFixed(1)}%
                                                    </Typography>
                                                </Box>
                                            </Stack>
                                        </Grid>

                                        <Grid item xs={6}>
                                            <Stack spacing={1}>
                                                <Typography variant="caption" color="text.secondary">
                                                    Memory Usage
                                                </Typography>
                                                <Box>
                                                    <LinearProgress
                                                        variant="determinate"
                                                        value={metrics.memory_usage}
                                                        sx={{ height: 6, borderRadius: 3 }}
                                                    />
                                                    <Typography variant="caption">
                                                        {metrics.memory_usage.toFixed(1)}%
                                                    </Typography>
                                                </Box>
                                            </Stack>
                                        </Grid>

                                        <Grid item xs={6}>
                                            <Stack spacing={1}>
                                                <Typography variant="caption" color="text.secondary">
                                                    Active Connections
                                                </Typography>
                                                <Typography variant="body2" fontWeight={600}>
                                                    {metrics.active_connections}
                                                </Typography>
                                            </Stack>
                                        </Grid>

                                        <Grid item xs={6}>
                                            <Stack spacing={1}>
                                                <Typography variant="caption" color="text.secondary">
                                                    Signals Today
                                                </Typography>
                                                <Typography variant="body2" fontWeight={600}>
                                                    {metrics.signals_generated_today}
                                                </Typography>
                                            </Stack>
                                        </Grid>
                                    </Grid>
                                </>
                            )}

                            {/* Cache Statistics */}
                            {stats.cacheSize && (
                                <>
                                    <Divider />
                                    <Stack spacing={1}>
                                        <Typography variant="caption" color="text.secondary">
                                            Cache Statistics
                                        </Typography>
                                        <Stack direction="row" spacing={2}>
                                            <Chip
                                                label={`Signals: ${stats.cacheSize.signals}`}
                                                size="small"
                                                variant="outlined"
                                            />
                                            <Chip
                                                label={`Market: ${stats.cacheSize.market}`}
                                                size="small"
                                                variant="outlined"
                                            />
                                            <Chip
                                                label={`Agents: ${stats.cacheSize.agents}`}
                                                size="small"
                                                variant="outlined"
                                            />
                                        </Stack>
                                    </Stack>
                                </>
                            )}

                            {/* Uptime */}
                            {stats.uptime && (
                                <>
                                    <Divider />
                                    <Stack direction="row" alignItems="center" spacing={1}>
                                        <Speed sx={{ fontSize: 16, color: theme.palette.text.secondary }} />
                                        <Typography variant="caption" color="text.secondary">
                                            Uptime: {formatUptime(stats.uptime)}
                                        </Typography>
                                    </Stack>
                                </>
                            )}
                        </>
                    )}
                </Stack>
            </CardContent>
        </Card>
    );
};

export default LiveDataStatus; 