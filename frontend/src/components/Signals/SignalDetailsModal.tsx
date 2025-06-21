/**
 * SignalDetailsModal - Comprehensive Signal Information Display
 * 
 * Shows detailed information about a signal including:
 * - Full entry/exit parameters
 * - Technical chart with overlays
 * - AI reasoning and confidence breakdown
 * - Execution checklist
 * - Risk management details
 */

import React, { useState } from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    IconButton,
    Stack,
    Typography,
    Box,
    Chip,
    Grid,
    Divider,
    Button,
    Card,
    CardContent,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Checkbox,
    Tab,
    Tabs,
    Alert,
    LinearProgress,
    useTheme,
    alpha,
} from '@mui/material';
import {
    Close,
    TrendingUp,
    TrendingDown,
    ContentCopy,
    Notifications,
    ShowChart,
    Psychology,
    CheckCircle,
    Warning,
    Timer,
    AttachMoney,
    Speed,
    Assessment,
    Timeline,
    Rule,
    PlayArrow,
} from '@mui/icons-material';
import { PreciseOptionsSignal } from '../../types/signals';
import MiniChart from '../Chart/MiniChart';

interface SignalDetailsModalProps {
    signal: PreciseOptionsSignal;
    open: boolean;
    onClose: () => void;
}

const SignalDetailsModal: React.FC<SignalDetailsModalProps> = ({ signal, open, onClose }) => {
    const theme = useTheme();
    const [activeTab, setActiveTab] = useState(0);
    const [checkedItems, setCheckedItems] = useState<string[]>([]);

    if (!signal) {
        return null;
    }

    const isCall = signal.signal_type === 'BUY_CALL';
    const signalColor = isCall ? theme.palette.success.main : theme.palette.error.main;

    const handleChecklistToggle = (item: string) => {
        setCheckedItems(prev =>
            prev.includes(item)
                ? prev.filter(i => i !== item)
                : [...prev, item]
        );
    };

    const allChecked = checkedItems.length === (signal.pre_entry_checklist?.length || 0);

    return (
        <Dialog
            open={open}
            onClose={onClose}
            maxWidth="lg"
            fullWidth
            PaperProps={{
                sx: {
                    background: theme.palette.background.paper,
                    backgroundImage: 'none',
                }
            }}
        >
            <DialogTitle>
                <Stack direction="row" alignItems="center" justifyContent="space-between">
                    <Stack direction="row" alignItems="center" spacing={2}>
                        <Typography variant="h5" fontWeight="bold">
                            {signal.symbol || 'Unknown'}
                        </Typography>
                        <Chip
                            icon={isCall ? <TrendingUp /> : <TrendingDown />}
                            label={signal.signal_type?.replace('BUY_', '') || 'SIGNAL'}
                            sx={{
                                backgroundColor: alpha(signalColor, 0.1),
                                color: signalColor,
                                fontWeight: 'bold',
                            }}
                        />
                        <Chip
                            icon={<Speed />}
                            label={`${signal.confidence || 0}% Confidence`}
                            color={signal.confidence >= 80 ? 'success' : 'warning'}
                        />
                    </Stack>
                    <IconButton onClick={onClose}>
                        <Close />
                    </IconButton>
                </Stack>
            </DialogTitle>

            <DialogContent>
                <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ mb: 3 }}>
                    <Tab label="Overview" icon={<Assessment />} iconPosition="start" />
                    <Tab label="Technical Analysis" icon={<ShowChart />} iconPosition="start" />
                    <Tab label="AI Insights" icon={<Psychology />} iconPosition="start" />
                    <Tab label="Execution" icon={<PlayArrow />} iconPosition="start" />
                </Tabs>

                {/* Overview Tab */}
                {activeTab === 0 && (
                    <Grid container spacing={3}>
                        {/* Entry Parameters */}
                        <Grid item xs={12} md={6}>
                            <Card sx={{ height: '100%' }}>
                                <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                        Entry Parameters
                                    </Typography>
                                    <Stack spacing={2}>
                                        <Box>
                                            <Typography variant="body2" color="text.secondary">
                                                Entry Trigger Price
                                            </Typography>
                                            <Typography variant="h5" fontWeight="bold">
                                                ${(signal.entry_trigger || 0).toFixed(2)}
                                            </Typography>
                                            <Typography variant="caption" color="text.secondary">
                                                Zone: ${(signal.entry_zone?.[0] || 0).toFixed(2)} - ${(signal.entry_zone?.[1] || 0).toFixed(2)}
                                            </Typography>
                                        </Box>

                                        <Divider />

                                        <Grid container spacing={2}>
                                            <Grid item xs={6}>
                                                <Typography variant="body2" color="text.secondary">
                                                    Strike Price
                                                </Typography>
                                                <Typography variant="h6">
                                                    ${signal.strike_price}
                                                </Typography>
                                            </Grid>
                                            <Grid item xs={6}>
                                                <Typography variant="body2" color="text.secondary">
                                                    Expiration
                                                </Typography>
                                                <Typography variant="h6">
                                                    {new Date(signal.expiration_date).toLocaleDateString()}
                                                </Typography>
                                            </Grid>
                                            <Grid item xs={6}>
                                                <Typography variant="body2" color="text.secondary">
                                                    Max Premium
                                                </Typography>
                                                <Typography variant="h6">
                                                    ${(signal.max_premium || 0).toFixed(2)}
                                                </Typography>
                                            </Grid>
                                            <Grid item xs={6}>
                                                <Typography variant="body2" color="text.secondary">
                                                    Position Size
                                                </Typography>
                                                <Typography variant="h6">
                                                    {signal.position_size} contracts
                                                </Typography>
                                            </Grid>
                                        </Grid>
                                    </Stack>
                                </CardContent>
                            </Card>
                        </Grid>

                        {/* Risk & Targets */}
                        <Grid item xs={12} md={6}>
                            <Card sx={{ height: '100%' }}>
                                <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                        Risk Management & Targets
                                    </Typography>
                                    <Stack spacing={2}>
                                        <Box>
                                            <Typography variant="body2" color="text.secondary">
                                                Stop Loss
                                            </Typography>
                                            <Typography variant="h5" color="error">
                                                ${(signal.stop_loss || 0).toFixed(2)} ({signal.stop_loss_pct || 0}%)
                                            </Typography>
                                        </Box>

                                        <Divider />

                                        <Box>
                                            <Typography variant="body2" color="text.secondary" gutterBottom>
                                                Profit Targets
                                            </Typography>
                                            {(signal.targets || []).map((target, idx) => (
                                                <Stack key={idx} direction="row" justifyContent="space-between" mb={1}>
                                                    <Typography color="success.main">
                                                        Target {idx + 1}: ${(target.price || 0).toFixed(2)}
                                                    </Typography>
                                                    <Chip
                                                        label={`Exit ${target.exit_pct || 0}%`}
                                                        size="small"
                                                        color="success"
                                                        variant="outlined"
                                                    />
                                                </Stack>
                                            ))}
                                        </Box>

                                        <Box>
                                            <Typography variant="body2" color="text.secondary">
                                                Risk/Reward Ratio
                                            </Typography>
                                            <Typography variant="h6" color="primary">
                                                {signal.risk_reward_ratio}:1
                                            </Typography>
                                        </Box>

                                        <Box>
                                            <Typography variant="body2" color="text.secondary">
                                                Max Risk
                                            </Typography>
                                            <Typography variant="h6">
                                                ${signal.max_risk_dollars}
                                            </Typography>
                                        </Box>
                                    </Stack>
                                </CardContent>
                            </Card>
                        </Grid>

                        {/* Timing */}
                        <Grid item xs={12}>
                            <Card>
                                <CardContent>
                                    <Stack direction="row" alignItems="center" spacing={1} mb={2}>
                                        <Timer />
                                        <Typography variant="h6">
                                            Timing Details
                                        </Typography>
                                    </Stack>

                                    <Grid container spacing={3}>
                                        <Grid item xs={12} sm={4}>
                                            <Typography variant="body2" color="text.secondary">
                                                Entry Window
                                            </Typography>
                                            <Typography variant="body1" fontWeight="bold">
                                                {signal.entry_window?.date || 'N/A'}
                                            </Typography>
                                            <Typography variant="body2">
                                                {signal.entry_window?.start_time || 'N/A'} - {signal.entry_window?.end_time || 'N/A'}
                                            </Typography>
                                        </Grid>
                                        <Grid item xs={12} sm={4}>
                                            <Typography variant="body2" color="text.secondary">
                                                Hold Duration
                                            </Typography>
                                            <Typography variant="body1" fontWeight="bold">
                                                {signal.hold_duration}
                                            </Typography>
                                        </Grid>
                                        <Grid item xs={12} sm={4}>
                                            <Typography variant="body2" color="text.secondary">
                                                Exit Warning
                                            </Typography>
                                            <Typography variant="body1" fontWeight="bold" color="warning.main">
                                                {signal.expiration_warning}
                                            </Typography>
                                        </Grid>
                                    </Grid>
                                </CardContent>
                            </Card>
                        </Grid>
                    </Grid>
                )}

                {/* Technical Analysis Tab */}
                {activeTab === 1 && (
                    <Stack spacing={3}>
                        <Card>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    Price Chart & Entry Points
                                </Typography>
                                <Box sx={{ height: 400, mt: 2 }}>
                                    <MiniChart
                                        symbol={signal.symbol}
                                        entryPrice={signal.entry_trigger || 0}
                                        stopLoss={signal.stop_loss || 0}
                                        targets={(signal.targets || []).map(t => t.price || 0)}
                                        signalType={signal.signal_type}
                                    />
                                </Box>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    Technical Indicators
                                </Typography>
                                <Grid container spacing={2} mt={1}>
                                    {Object.entries(signal.key_indicators || {}).map(([key, value]) => (
                                        <Grid item xs={6} sm={3} key={key}>
                                            <Box sx={{
                                                p: 2,
                                                border: `1px solid ${alpha(theme.palette.divider, 0.2)}`,
                                                borderRadius: 1,
                                                textAlign: 'center',
                                            }}>
                                                <Typography variant="caption" color="text.secondary">
                                                    {key}
                                                </Typography>
                                                <Typography variant="h6">
                                                    {value}
                                                </Typography>
                                            </Box>
                                        </Grid>
                                    ))}
                                </Grid>
                            </CardContent>
                        </Card>
                    </Stack>
                )}

                {/* AI Insights Tab */}
                {activeTab === 2 && (
                    <Stack spacing={3}>
                        <Card>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    AI Analysis & Reasoning
                                </Typography>
                                <Stack spacing={2} mt={2}>
                                    <Alert severity="info" icon={<Psychology />}>
                                        <Typography variant="body2">
                                            <strong>Setup Pattern:</strong> {signal.setup_name}
                                        </Typography>
                                    </Alert>

                                    <Box>
                                        <Typography variant="subtitle2" gutterBottom>
                                            Detected Patterns:
                                        </Typography>
                                        <Stack direction="row" spacing={1} flexWrap="wrap">
                                            {(signal.chart_patterns || []).map((pattern) => (
                                                <Chip
                                                    key={pattern}
                                                    label={pattern}
                                                    size="small"
                                                    variant="outlined"
                                                    icon={<Timeline />}
                                                />
                                            ))}
                                        </Stack>
                                    </Box>

                                    <Box>
                                        <Typography variant="subtitle2" gutterBottom>
                                            Confidence Breakdown:
                                        </Typography>
                                        <Stack spacing={1}>
                                            <Box>
                                                <Stack direction="row" justifyContent="space-between" mb={0.5}>
                                                    <Typography variant="body2">Technical Analysis</Typography>
                                                    <Typography variant="body2">85%</Typography>
                                                </Stack>
                                                <LinearProgress variant="determinate" value={85} />
                                            </Box>
                                            <Box>
                                                <Stack direction="row" justifyContent="space-between" mb={0.5}>
                                                    <Typography variant="body2">Pattern Recognition</Typography>
                                                    <Typography variant="body2">78%</Typography>
                                                </Stack>
                                                <LinearProgress variant="determinate" value={78} />
                                            </Box>
                                            <Box>
                                                <Stack direction="row" justifyContent="space-between" mb={0.5}>
                                                    <Typography variant="body2">Market Conditions</Typography>
                                                    <Typography variant="body2">92%</Typography>
                                                </Stack>
                                                <LinearProgress variant="determinate" value={92} />
                                            </Box>
                                        </Stack>
                                    </Box>
                                </Stack>
                            </CardContent>
                        </Card>
                    </Stack>
                )}

                {/* Execution Tab */}
                {activeTab === 3 && (
                    <Stack spacing={3}>
                        <Card>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    Pre-Entry Checklist
                                </Typography>
                                <List>
                                    {(signal.pre_entry_checklist || []).map((item) => (
                                        <ListItem key={item} dense>
                                            <ListItemIcon>
                                                <Checkbox
                                                    edge="start"
                                                    checked={checkedItems.includes(item)}
                                                    onChange={() => handleChecklistToggle(item)}
                                                />
                                            </ListItemIcon>
                                            <ListItemText primary={item} />
                                        </ListItem>
                                    ))}
                                </List>

                                {allChecked && (
                                    <Alert severity="success" sx={{ mt: 2 }}>
                                        All checks complete! Ready to execute trade.
                                    </Alert>
                                )}
                            </CardContent>
                        </Card>

                        <Card>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    Exit Rules
                                </Typography>
                                <List dense>
                                    {(signal.exit_rules || []).map((rule, idx) => (
                                        <ListItem key={idx}>
                                            <ListItemIcon>
                                                <Rule />
                                            </ListItemIcon>
                                            <ListItemText primary={rule} />
                                        </ListItem>
                                    ))}
                                </List>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    Quick Actions
                                </Typography>
                                <Stack spacing={2} mt={2}>
                                    <Button
                                        fullWidth
                                        variant="contained"
                                        startIcon={<ContentCopy />}
                                        onClick={() => {
                                            const tradeText = `${signal.signal_type} ${signal.symbol} ${signal.strike_price} ${signal.expiration_date}`;
                                            navigator.clipboard.writeText(tradeText);
                                        }}
                                    >
                                        Copy Trade Details
                                    </Button>
                                    <Button
                                        fullWidth
                                        variant="outlined"
                                        startIcon={<Notifications />}
                                    >
                                        Set All Alerts
                                    </Button>
                                    <Button
                                        fullWidth
                                        variant="outlined"
                                        startIcon={<ShowChart />}
                                    >
                                        Open Full Chart
                                    </Button>
                                </Stack>
                            </CardContent>
                        </Card>
                    </Stack>
                )}
            </DialogContent>
        </Dialog>
    );
};

export default SignalDetailsModal; 