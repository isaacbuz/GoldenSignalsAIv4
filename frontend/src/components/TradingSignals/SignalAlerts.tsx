import React, { useState, useEffect } from 'react';
import {
    Box,
    Card,
    CardContent,
    Typography,
    Chip,
    IconButton,
    Badge,
    Collapse,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    ListItemSecondaryAction,
    Divider,
    Alert,
    Slide,
    Fade,
    useTheme,
    alpha,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    NotificationsActive,
    Close,
    ExpandMore,
    ExpandLess,
    AccessTime,
    Speed,
    Psychology,
    ShowChart,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { motion, AnimatePresence } from 'framer-motion';

// Professional styled components
const AlertBanner = styled(motion.div)(({ theme, alertType }) => ({
    position: 'fixed',
    top: '20px',
    right: '20px',
    zIndex: 1000,
    minWidth: '400px',
    maxWidth: '500px',
    backgroundColor: alertType === 'buy' ?
        alpha(theme.palette.success.main, 0.95) :
        alpha(theme.palette.error.main, 0.95),
    color: theme.palette.common.white,
    borderRadius: '12px',
    padding: '16px',
    boxShadow: `0 8px 32px ${alpha(theme.palette.common.black, 0.3)}`,
    backdropFilter: 'blur(10px)',
    border: `1px solid ${alpha(theme.palette.common.white, 0.2)}`,
}));

const SignalsSidebar = styled(Card)(({ theme }) => ({
    height: '100%',
    backgroundColor: alpha(theme.palette.background.paper, 0.95),
    backdropFilter: 'blur(10px)',
    border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
    borderRadius: '12px',
}));

const SignalCard = styled(motion.div)(({ theme, signalType }) => ({
    padding: '12px',
    margin: '8px 0',
    backgroundColor: alpha(
        signalType === 'buy' ? theme.palette.success.main : theme.palette.error.main,
        0.1
    ),
    border: `1px solid ${alpha(
        signalType === 'buy' ? theme.palette.success.main : theme.palette.error.main,
        0.3
    )}`,
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    '&:hover': {
        backgroundColor: alpha(
            signalType === 'buy' ? theme.palette.success.main : theme.palette.error.main,
            0.2
        ),
        transform: 'translateY(-2px)',
        boxShadow: `0 4px 20px ${alpha(
            signalType === 'buy' ? theme.palette.success.main : theme.palette.error.main,
            0.3
        )}`,
    },
}));

const ConfidenceGauge = styled(Box)(({ theme, confidence }) => ({
    position: 'relative',
    width: '40px',
    height: '40px',
    borderRadius: '50%',
    background: `conic-gradient(
        ${confidence >= 90 ? theme.palette.success.main :
            confidence >= 70 ? theme.palette.warning.main :
                theme.palette.error.main} ${confidence * 3.6}deg,
        ${alpha(theme.palette.grey[300], 0.3)} 0deg
    )`,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    '&::before': {
        content: '""',
        position: 'absolute',
        width: '30px',
        height: '30px',
        borderRadius: '50%',
        backgroundColor: theme.palette.background.paper,
    },
}));

const ConfidenceText = styled(Typography)(({ theme }) => ({
    position: 'relative',
    zIndex: 1,
    fontSize: '10px',
    fontWeight: 'bold',
}));

interface Signal {
    id: string;
    symbol: string;
    type: 'buy' | 'sell';
    action: string;
    price: number;
    confidence: number;
    timestamp: number;
    reason: string;
    timeframe: string;
    status: 'active' | 'executed' | 'expired';
}

interface SignalAlertsProps {
    signals: Signal[];
    onSignalClick: (signal: Signal) => void;
    onDismissAlert: (signalId: string) => void;
}

const SignalAlerts: React.FC<SignalAlertsProps> = ({
    signals,
    onSignalClick,
    onDismissAlert,
}) => {
    const theme = useTheme();
    const [activeAlert, setActiveAlert] = useState<Signal | null>(null);
    const [sidebarExpanded, setSidebarExpanded] = useState(true);
    const [groupedSignals, setGroupedSignals] = useState<{
        active: Signal[];
        recent: Signal[];
        watching: Signal[];
    }>({
        active: [],
        recent: [],
        watching: [],
    });

    // Group signals by status
    useEffect(() => {
        const now = Date.now();
        const grouped = {
            active: signals.filter(s => s.status === 'active' && (now - s.timestamp) < 300000), // 5 minutes
            recent: signals.filter(s => s.status === 'executed' && (now - s.timestamp) < 3600000), // 1 hour
            watching: signals.filter(s => s.status === 'active' && (now - s.timestamp) >= 300000), // Older than 5 minutes
        };
        setGroupedSignals(grouped);
    }, [signals]);

    // Show alert for new signals
    useEffect(() => {
        const latestSignal = signals
            .filter(s => s.status === 'active')
            .sort((a, b) => b.timestamp - a.timestamp)[0];

        if (latestSignal && (!activeAlert || latestSignal.id !== activeAlert.id)) {
            setActiveAlert(latestSignal);

            // Auto-dismiss after 10 seconds
            const timer = setTimeout(() => {
                setActiveAlert(null);
            }, 10000);

            return () => clearTimeout(timer);
        }
    }, [signals, activeAlert]);

    const formatTimeAgo = (timestamp: number) => {
        const seconds = Math.floor((Date.now() - timestamp) / 1000);
        if (seconds < 60) return 'just now';
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        return `${Math.floor(seconds / 86400)}d ago`;
    };

    const getSignalIcon = (signal: Signal) => {
        return signal.type === 'buy' ? (
            <TrendingUp sx={{ color: theme.palette.success.main }} />
        ) : (
            <TrendingDown sx={{ color: theme.palette.error.main }} />
        );
    };

    const getConfidenceColor = (confidence: number) => {
        if (confidence >= 90) return theme.palette.success.main;
        if (confidence >= 70) return theme.palette.warning.main;
        return theme.palette.error.main;
    };

    return (
        <>
            {/* Alert Banner */}
            <AnimatePresence>
                {activeAlert && (
                    <AlertBanner
                        alertType={activeAlert.type}
                        initial={{ x: 400, opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        exit={{ x: 400, opacity: 0 }}
                        transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    >
                        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                            <Box sx={{
                                backgroundColor: alpha(theme.palette.common.white, 0.2),
                                borderRadius: '50%',
                                p: 1,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                            }}>
                                {getSignalIcon(activeAlert)}
                            </Box>

                            <Box sx={{ flex: 1 }}>
                                <Typography variant="h6" fontWeight="bold">
                                    {activeAlert.type === 'buy' ? 'Bullish Signal' : 'Bearish Signal'}
                                </Typography>
                                <Typography variant="body2" sx={{ opacity: 0.9, mb: 1 }}>
                                    {activeAlert.symbol} – {activeAlert.action}
                                </Typography>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <Chip
                                        label={`${activeAlert.confidence}% Confidence`}
                                        size="small"
                                        sx={{
                                            backgroundColor: alpha(theme.palette.common.white, 0.2),
                                            color: theme.palette.common.white,
                                            fontWeight: 'bold',
                                        }}
                                    />
                                    <Typography variant="caption">
                                        @ ${activeAlert.price.toFixed(2)}
                                    </Typography>
                                </Box>
                            </Box>

                            <IconButton
                                size="small"
                                onClick={() => {
                                    setActiveAlert(null);
                                    onDismissAlert(activeAlert.id);
                                }}
                                sx={{ color: theme.palette.common.white }}
                            >
                                <Close />
                            </IconButton>
                        </Box>
                    </AlertBanner>
                )}
            </AnimatePresence>

            {/* Signals Sidebar */}
            <SignalsSidebar>
                <CardContent sx={{ p: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Badge badgeContent={groupedSignals.active.length} color="primary">
                                <NotificationsActive color="primary" />
                            </Badge>
                            <Typography variant="h6" fontWeight="bold">
                                Trading Signals
                            </Typography>
                        </Box>
                        <IconButton
                            size="small"
                            onClick={() => setSidebarExpanded(!sidebarExpanded)}
                        >
                            {sidebarExpanded ? <ExpandLess /> : <ExpandMore />}
                        </IconButton>
                    </Box>

                    <Collapse in={sidebarExpanded}>
                        {/* Active Signals */}
                        {groupedSignals.active.length > 0 && (
                            <>
                                <Typography variant="subtitle2" color="primary" fontWeight="bold" sx={{ mb: 1 }}>
                                    🔥 Active Now ({groupedSignals.active.length})
                                </Typography>
                                {groupedSignals.active.map((signal) => (
                                    <SignalCard
                                        key={signal.id}
                                        signalType={signal.type}
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                        onClick={() => onSignalClick(signal)}
                                    >
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                            <ConfidenceGauge confidence={signal.confidence}>
                                                <ConfidenceText>
                                                    {signal.confidence}%
                                                </ConfidenceText>
                                            </ConfidenceGauge>

                                            <Box sx={{ flex: 1 }}>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                                                    <Typography variant="subtitle2" fontWeight="bold">
                                                        {signal.symbol}
                                                    </Typography>
                                                    <Chip
                                                        icon={getSignalIcon(signal)}
                                                        label={signal.action}
                                                        size="small"
                                                        color={signal.type === 'buy' ? 'success' : 'error'}
                                                        variant="outlined"
                                                    />
                                                </Box>
                                                <Typography variant="caption" color="text.secondary">
                                                    ${signal.price.toFixed(2)} • {formatTimeAgo(signal.timestamp)}
                                                </Typography>
                                            </Box>

                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                                <AccessTime sx={{ fontSize: 16, color: 'text.secondary' }} />
                                                <Typography variant="caption" color="text.secondary">
                                                    {signal.timeframe}
                                                </Typography>
                                            </Box>
                                        </Box>
                                    </SignalCard>
                                ))}
                                <Divider sx={{ my: 2 }} />
                            </>
                        )}

                        {/* Watching Signals */}
                        {groupedSignals.watching.length > 0 && (
                            <>
                                <Typography variant="subtitle2" color="warning.main" fontWeight="bold" sx={{ mb: 1 }}>
                                    👁️ Watching ({groupedSignals.watching.length})
                                </Typography>
                                {groupedSignals.watching.slice(0, 3).map((signal) => (
                                    <SignalCard
                                        key={signal.id}
                                        signalType={signal.type}
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        onClick={() => onSignalClick(signal)}
                                        style={{ opacity: 0.7 }}
                                    >
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                            <Box sx={{
                                                width: 8,
                                                height: 8,
                                                borderRadius: '50%',
                                                backgroundColor: getConfidenceColor(signal.confidence),
                                            }} />
                                            <Box sx={{ flex: 1 }}>
                                                <Typography variant="body2" fontWeight="medium">
                                                    {signal.symbol} • {signal.action}
                                                </Typography>
                                                <Typography variant="caption" color="text.secondary">
                                                    {formatTimeAgo(signal.timestamp)}
                                                </Typography>
                                            </Box>
                                            <Typography variant="caption" color="text.secondary">
                                                {signal.confidence}%
                                            </Typography>
                                        </Box>
                                    </SignalCard>
                                ))}
                                <Divider sx={{ my: 2 }} />
                            </>
                        )}

                        {/* Recent Executed */}
                        {groupedSignals.recent.length > 0 && (
                            <>
                                <Typography variant="subtitle2" color="text.secondary" fontWeight="bold" sx={{ mb: 1 }}>
                                    ✅ Recent ({groupedSignals.recent.length})
                                </Typography>
                                {groupedSignals.recent.slice(0, 2).map((signal) => (
                                    <Box
                                        key={signal.id}
                                        sx={{
                                            p: 1,
                                            mb: 1,
                                            backgroundColor: alpha(theme.palette.success.main, 0.05),
                                            borderRadius: '6px',
                                            border: `1px solid ${alpha(theme.palette.success.main, 0.1)}`,
                                        }}
                                    >
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            <Typography variant="body2">
                                                {signal.symbol} {signal.action}
                                            </Typography>
                                            <Chip label="Executed" size="small" color="success" variant="outlined" />
                                        </Box>
                                        <Typography variant="caption" color="text.secondary">
                                            {formatTimeAgo(signal.timestamp)}
                                        </Typography>
                                    </Box>
                                ))}
                            </>
                        )}

                        {/* Empty State */}
                        {signals.length === 0 && (
                            <Box sx={{ textAlign: 'center', py: 4 }}>
                                <ShowChart sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                                <Typography variant="body2" color="text.secondary">
                                    No signals detected yet
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                    Monitoring markets for opportunities...
                                </Typography>
                            </Box>
                        )}
                    </Collapse>
                </CardContent>
            </SignalsSidebar>
        </>
    );
};

export default SignalAlerts; 