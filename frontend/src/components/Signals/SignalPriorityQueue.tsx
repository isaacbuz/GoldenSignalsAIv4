import React, { useState, useMemo } from 'react';
import {
    Box,
    Card,
    CardContent,
    Typography,
    Chip,
    Stack,
    IconButton,
    Tooltip,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Badge,
    Button,
    Collapse,
    Alert,
    LinearProgress,
    useTheme,
    alpha,
    Menu,
    Divider,
} from '@mui/material';
import {
    DragIndicator,
    Star,
    StarBorder,
    Timer,
    TrendingUp,
    Warning,
    CheckCircle,
    Cancel,
    ExpandMore,
    ExpandLess,
    FilterList,
    SortByAlpha,
    AccessTime,
    LocalFireDepartment,
    Psychology,
    PriorityHigh,
    AutoAwesome,
    Schedule,
    Sort,
    Refresh,
} from '@mui/icons-material';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import { motion, AnimatePresence } from 'framer-motion';

interface Signal {
    id: string;
    symbol: string;
    type: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    priority: 'high' | 'medium' | 'low';
    timeUntilExpiry: number; // minutes
    potentialReturn: number;
    riskScore: number;
    timestamp: Date;
    reason: string;
}

interface QueueMetrics {
    totalSignals: number;
    criticalSignals: number;
    avgConfidence: number;
    successRate: number;
}

const mockSignals: Signal[] = [
    {
        id: '1',
        symbol: 'AAPL',
        type: 'BUY',
        confidence: 92,
        priority: 'high',
        timeUntilExpiry: 15,
        potentialReturn: 5,
        riskScore: 85,
        timestamp: new Date(),
        reason: 'Strong bullish momentum with volume breakout',
    },
    {
        id: '2',
        symbol: 'TSLA',
        type: 'SELL',
        confidence: 87,
        priority: 'medium',
        timeUntilExpiry: 45,
        potentialReturn: -3,
        riskScore: 78,
        timestamp: new Date(),
        reason: 'Bearish divergence on RSI',
    },
    // Add more mock signals...
];

interface SignalPriorityQueueProps {
    signals: Signal[];
    onSignalClick?: (signal: Signal) => void;
    onRefresh?: () => void;
}

export const SignalPriorityQueue: React.FC<SignalPriorityQueueProps> = ({
    signals,
    onSignalClick,
    onRefresh,
}) => {
    const theme = useTheme();
    const [expanded, setExpanded] = useState(true);
    const [filterAnchorEl, setFilterAnchorEl] = useState<null | HTMLElement>(null);
    const [sortBy, setSortBy] = useState<'priority' | 'confidence' | 'return' | 'time'>('priority');
    const [filterPriority, setFilterPriority] = useState<'all' | 'high' | 'medium' | 'low'>('all');

    // Calculate priority score for sorting
    const calculatePriorityScore = (signal: Signal): number => {
        const priorityWeight = signal.priority === 'high' ? 3 : signal.priority === 'medium' ? 2 : 1;
        const confidenceWeight = signal.confidence / 100;
        const returnWeight = signal.potentialReturn / 100;
        const timeWeight = Math.max(0, 1 - (signal.timeUntilExpiry / 1440)); // 1440 = 24 hours

        return (priorityWeight * 0.4) + (confidenceWeight * 0.3) + (returnWeight * 0.2) + (timeWeight * 0.1);
    };

    // Sort and filter signals
    const sortedSignals = useMemo(() => {
        let filtered = signals;

        if (filterPriority !== 'all') {
            filtered = signals.filter(s => s.priority === filterPriority);
        }

        return [...filtered].sort((a, b) => {
            switch (sortBy) {
                case 'priority':
                    return calculatePriorityScore(b) - calculatePriorityScore(a);
                case 'confidence':
                    return b.confidence - a.confidence;
                case 'return':
                    return b.potentialReturn - a.potentialReturn;
                case 'time':
                    return a.timeUntilExpiry - b.timeUntilExpiry;
                default:
                    return 0;
            }
        });
    }, [signals, sortBy, filterPriority]);

    const getPriorityColor = (priority: string) => {
        switch (priority) {
            case 'high':
                return theme.palette.error.main;
            case 'medium':
                return theme.palette.warning.main;
            case 'low':
                return theme.palette.info.main;
            default:
                return theme.palette.grey[500];
        }
    };

    const getTimeUrgency = (minutes: number) => {
        if (minutes < 60) return { color: 'error', label: `${minutes}m`, urgent: true };
        if (minutes < 240) return { color: 'warning', label: `${Math.floor(minutes / 60)}h`, urgent: false };
        return { color: 'info', label: `${Math.floor(minutes / 60)}h`, urgent: false };
    };

    const SignalItem: React.FC<{ signal: Signal; index: number }> = ({ signal, index }) => {
        const timeInfo = getTimeUrgency(signal.timeUntilExpiry);

        return (
            <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ delay: index * 0.05 }}
            >
                <Box
                    sx={{
                        p: 2,
                        borderRadius: 2,
                        background: alpha(theme.palette.background.paper, 0.8),
                        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                        borderLeft: `4px solid ${getPriorityColor(signal.priority)}`,
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                        '&:hover': {
                            background: alpha(theme.palette.primary.main, 0.05),
                            transform: 'translateX(4px)',
                        },
                    }}
                    onClick={() => onSignalClick?.(signal)}
                >
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                        <Box sx={{ flex: 1 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                <Typography variant="h6" fontWeight="bold">
                                    {signal.symbol}
                                </Typography>
                                <Chip
                                    label={signal.type}
                                    size="small"
                                    color={signal.type === 'BUY' ? 'success' : signal.type === 'SELL' ? 'error' : 'default'}
                                />
                                <Badge
                                    badgeContent={`${signal.confidence}%`}
                                    color="primary"
                                    sx={{ '& .MuiBadge-badge': { position: 'static', transform: 'none' } }}
                                />
                            </Box>

                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                {signal.reason}
                            </Typography>

                            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                                <Chip
                                    icon={<TrendingUp />}
                                    label={`+${signal.potentialReturn}%`}
                                    size="small"
                                    variant="outlined"
                                    color="success"
                                />
                                <Chip
                                    icon={<Timer />}
                                    label={timeInfo.label}
                                    size="small"
                                    variant={timeInfo.urgent ? 'filled' : 'outlined'}
                                    color={timeInfo.color as any}
                                />
                                <Chip
                                    icon={<PriorityHigh />}
                                    label={signal.priority.toUpperCase()}
                                    size="small"
                                    sx={{
                                        background: alpha(getPriorityColor(signal.priority), 0.1),
                                        color: getPriorityColor(signal.priority),
                                        fontWeight: 'bold',
                                    }}
                                />
                            </Box>
                        </Box>

                        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                            <Box
                                sx={{
                                    width: 60,
                                    height: 60,
                                    borderRadius: '50%',
                                    background: `conic-gradient(${theme.palette.primary.main} ${signal.confidence * 3.6}deg, ${alpha(theme.palette.divider, 0.2)} 0deg)`,
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    position: 'relative',
                                }}
                            >
                                <Box
                                    sx={{
                                        width: 50,
                                        height: 50,
                                        borderRadius: '50%',
                                        background: theme.palette.background.paper,
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                    }}
                                >
                                    <Typography variant="body2" fontWeight="bold">
                                        {signal.confidence}%
                                    </Typography>
                                </Box>
                            </Box>
                            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
                                AI Score
                            </Typography>
                        </Box>
                    </Box>
                </Box>
            </motion.div>
        );
    };

    return (
        <Card
            sx={{
                background: alpha(theme.palette.background.paper, 0.8),
                backdropFilter: 'blur(10px)',
                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
            }}
        >
            <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <AutoAwesome sx={{ color: theme.palette.primary.main }} />
                        <Typography variant="h6" fontWeight="bold">
                            Priority Queue
                        </Typography>
                        <Chip
                            label={sortedSignals.length}
                            size="small"
                            color="primary"
                            sx={{ fontWeight: 'bold' }}
                        />
                    </Box>

                    <Box sx={{ display: 'flex', gap: 1 }}>
                        <Tooltip title="Filter">
                            <IconButton
                                size="small"
                                onClick={(e) => setFilterAnchorEl(e.currentTarget)}
                            >
                                <FilterList />
                            </IconButton>
                        </Tooltip>

                        <Tooltip title="Refresh">
                            <IconButton size="small" onClick={onRefresh}>
                                <Refresh />
                            </IconButton>
                        </Tooltip>

                        <IconButton
                            size="small"
                            onClick={() => setExpanded(!expanded)}
                        >
                            {expanded ? <ExpandLess /> : <ExpandMore />}
                        </IconButton>
                    </Box>
                </Box>

                <Menu
                    anchorEl={filterAnchorEl}
                    open={Boolean(filterAnchorEl)}
                    onClose={() => setFilterAnchorEl(null)}
                >
                    <MenuItem dense disabled>
                        <Typography variant="caption" fontWeight="bold">SORT BY</Typography>
                    </MenuItem>
                    <MenuItem onClick={() => { setSortBy('priority'); setFilterAnchorEl(null); }}>
                        <Sort fontSize="small" sx={{ mr: 1 }} /> Priority Score
                    </MenuItem>
                    <MenuItem onClick={() => { setSortBy('confidence'); setFilterAnchorEl(null); }}>
                        <AutoAwesome fontSize="small" sx={{ mr: 1 }} /> AI Confidence
                    </MenuItem>
                    <MenuItem onClick={() => { setSortBy('return'); setFilterAnchorEl(null); }}>
                        <TrendingUp fontSize="small" sx={{ mr: 1 }} /> Potential Return
                    </MenuItem>
                    <MenuItem onClick={() => { setSortBy('time'); setFilterAnchorEl(null); }}>
                        <Timer fontSize="small" sx={{ mr: 1 }} /> Time Remaining
                    </MenuItem>

                    <Divider />

                    <MenuItem dense disabled>
                        <Typography variant="caption" fontWeight="bold">FILTER PRIORITY</Typography>
                    </MenuItem>
                    <MenuItem onClick={() => { setFilterPriority('all'); setFilterAnchorEl(null); }}>
                        All Priorities
                    </MenuItem>
                    <MenuItem onClick={() => { setFilterPriority('high'); setFilterAnchorEl(null); }}>
                        <PriorityHigh fontSize="small" sx={{ mr: 1, color: theme.palette.error.main }} /> High Only
                    </MenuItem>
                    <MenuItem onClick={() => { setFilterPriority('medium'); setFilterAnchorEl(null); }}>
                        <Warning fontSize="small" sx={{ mr: 1, color: theme.palette.warning.main }} /> Medium Only
                    </MenuItem>
                    <MenuItem onClick={() => { setFilterPriority('low'); setFilterAnchorEl(null); }}>
                        <Schedule fontSize="small" sx={{ mr: 1, color: theme.palette.info.main }} /> Low Only
                    </MenuItem>
                </Menu>

                <Collapse in={expanded}>
                    <Stack spacing={2}>
                        {sortedSignals.length === 0 ? (
                            <Box sx={{ textAlign: 'center', py: 4 }}>
                                <Typography variant="body2" color="text.secondary">
                                    No signals match your criteria
                                </Typography>
                            </Box>
                        ) : (
                            <AnimatePresence>
                                {sortedSignals.slice(0, 10).map((signal, index) => (
                                    <SignalItem key={signal.id} signal={signal} index={index} />
                                ))}
                            </AnimatePresence>
                        )}

                        {sortedSignals.length > 10 && (
                            <Box sx={{ textAlign: 'center' }}>
                                <Button size="small" variant="text">
                                    View All {sortedSignals.length} Signals
                                </Button>
                            </Box>
                        )}
                    </Stack>
                </Collapse>
            </CardContent>
        </Card>
    );
};

export default SignalPriorityQueue;
