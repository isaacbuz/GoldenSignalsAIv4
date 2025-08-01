import React, { useState, useEffect } from 'react';
import {
    Card,
    CardContent,
    Typography,
    TextField,
    Box,
    Slider,
    Chip,
    Alert,
    Button,
    Stack,
    Divider,
    InputAdornment,
    Tooltip,
    IconButton,
    useTheme,
    alpha,
} from '@mui/material';
import {
    Calculate,
    Info,
    TrendingUp,
    TrendingDown,
    AttachMoney,
    Warning,
    ContentCopy,
    Refresh,
} from '@mui/icons-material';

interface CalculatorResult {
    positionSize: number;
    shares: number;
    risk: number;
    potentialProfit: number;
    riskRewardRatio: number;
}

export const PositionSizeCalculator: React.FC = () => {
    const theme = useTheme();

    // Input states
    const [accountBalance, setAccountBalance] = useState(25000);
    const [riskPercentage, setRiskPercentage] = useState(2);
    const [entryPrice, setEntryPrice] = useState(150);
    const [stopLoss, setStopLoss] = useState(145);
    const [targetPrice, setTargetPrice] = useState(160);

    // Calculation result
    const [result, setResult] = useState<CalculatorResult | null>(null);

    // Calculate position size
    useEffect(() => {
        if (accountBalance > 0 && riskPercentage > 0 && entryPrice > 0 && stopLoss > 0) {
            const riskAmount = accountBalance * (riskPercentage / 100);
            const riskPerShare = Math.abs(entryPrice - stopLoss);
            const shares = Math.floor(riskAmount / riskPerShare);
            const positionSize = shares * entryPrice;
            const potentialProfit = shares * Math.abs(targetPrice - entryPrice);
            const riskRewardRatio = potentialProfit / riskAmount;

            setResult({
                positionSize,
                shares,
                risk: riskAmount,
                potentialProfit,
                riskRewardRatio,
            });
        }
    }, [accountBalance, riskPercentage, entryPrice, stopLoss, targetPrice]);

    const handleCopy = () => {
        if (result) {
            const text = `Position Size Calculator:
Account: $${accountBalance.toLocaleString()}
Risk: ${riskPercentage}% ($${result.risk.toFixed(2)})
Entry: $${entryPrice}
Stop Loss: $${stopLoss}
Target: $${targetPrice}
Shares: ${result.shares}
Position Size: $${result.positionSize.toFixed(2)}
R:R Ratio: 1:${result.riskRewardRatio.toFixed(2)}`;

            navigator.clipboard.writeText(text);
        }
    };

    const InfoTooltip: React.FC<{ text: string }> = ({ text }) => (
        <Tooltip title={text}>
            <IconButton size="small" sx={{ ml: 0.5 }}>
                <Info fontSize="small" />
            </IconButton>
        </Tooltip>
    );

    return (
        <Card
            sx={{
                maxWidth: 500,
                background: alpha(theme.palette.background.paper, 0.8),
                backdropFilter: 'blur(10px)',
                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
            }}
        >
            <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                    <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Calculate sx={{ color: theme.palette.primary.main }} />
                        Position Size Calculator
                    </Typography>
                    <Box>
                        <Tooltip title="Copy results">
                            <IconButton size="small" onClick={handleCopy} disabled={!result}>
                                <ContentCopy fontSize="small" />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Reset">
                            <IconButton
                                size="small"
                                onClick={() => {
                                    setAccountBalance(25000);
                                    setRiskPercentage(2);
                                    setEntryPrice(150);
                                    setStopLoss(145);
                                    setTargetPrice(160);
                                }}
                            >
                                <Refresh fontSize="small" />
                            </IconButton>
                        </Tooltip>
                    </Box>
                </Box>

                <Stack spacing={3}>
                    {/* Account Balance */}
                    <Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <Typography variant="body2" color="text.secondary">
                                Account Balance
                            </Typography>
                            <InfoTooltip text="Your total trading account balance" />
                        </Box>
                        <TextField
                            fullWidth
                            type="number"
                            value={accountBalance}
                            onChange={(e) => setAccountBalance(parseFloat(e.target.value) || 0)}
                            InputProps={{
                                startAdornment: <InputAdornment position="start">$</InputAdornment>,
                            }}
                            size="small"
                        />
                    </Box>

                    {/* Risk Percentage */}
                    <Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <Typography variant="body2" color="text.secondary">
                                Risk Per Trade: {riskPercentage}%
                            </Typography>
                            <InfoTooltip text="Percentage of account to risk on this trade (1-3% recommended)" />
                        </Box>
                        <Slider
                            value={riskPercentage}
                            onChange={(_, value) => setRiskPercentage(value as number)}
                            min={0.5}
                            max={5}
                            step={0.5}
                            marks={[
                                { value: 1, label: '1%' },
                                { value: 2, label: '2%' },
                                { value: 3, label: '3%' },
                                { value: 5, label: '5%' },
                            ]}
                            sx={{
                                '& .MuiSlider-markLabel': {
                                    fontSize: '0.75rem',
                                },
                            }}
                        />
                        {riskPercentage > 3 && (
                            <Alert severity="warning" sx={{ mt: 1 }} icon={<Warning />}>
                                <Typography variant="caption">
                                    Risking more than 3% per trade is considered aggressive
                                </Typography>
                            </Alert>
                        )}
                    </Box>

                    {/* Trade Details */}
                    <Box>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            Trade Details
                        </Typography>
                        <Stack spacing={2}>
                            <TextField
                                fullWidth
                                label="Entry Price"
                                type="number"
                                value={entryPrice}
                                onChange={(e) => setEntryPrice(parseFloat(e.target.value) || 0)}
                                InputProps={{
                                    startAdornment: <InputAdornment position="start">$</InputAdornment>,
                                }}
                                size="small"
                            />
                            <TextField
                                fullWidth
                                label="Stop Loss"
                                type="number"
                                value={stopLoss}
                                onChange={(e) => setStopLoss(parseFloat(e.target.value) || 0)}
                                InputProps={{
                                    startAdornment: <InputAdornment position="start">$</InputAdornment>,
                                    endAdornment: stopLoss < entryPrice ?
                                        <TrendingDown fontSize="small" color="error" /> :
                                        <TrendingUp fontSize="small" color="success" />,
                                }}
                                size="small"
                                error={stopLoss === entryPrice}
                                helperText={stopLoss === entryPrice ? "Stop loss cannot equal entry price" : ""}
                            />
                            <TextField
                                fullWidth
                                label="Target Price"
                                type="number"
                                value={targetPrice}
                                onChange={(e) => setTargetPrice(parseFloat(e.target.value) || 0)}
                                InputProps={{
                                    startAdornment: <InputAdornment position="start">$</InputAdornment>,
                                    endAdornment: targetPrice > entryPrice ?
                                        <TrendingUp fontSize="small" color="success" /> :
                                        <TrendingDown fontSize="small" color="error" />,
                                }}
                                size="small"
                            />
                        </Stack>
                    </Box>

                    <Divider />

                    {/* Results */}
                    {result && (
                        <Box>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                Calculation Results
                            </Typography>

                            <Stack spacing={2}>
                                <Box
                                    sx={{
                                        p: 2,
                                        borderRadius: 2,
                                        background: alpha(theme.palette.primary.main, 0.1),
                                        border: `1px solid ${alpha(theme.palette.primary.main, 0.3)}`,
                                    }}
                                >
                                    <Typography variant="body2" color="text.secondary">
                                        Position Size
                                    </Typography>
                                    <Typography variant="h5" fontWeight="bold" color="primary">
                                        ${result.positionSize.toFixed(2)}
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary">
                                        {result.shares} shares @ ${entryPrice}
                                    </Typography>
                                </Box>

                                <Stack direction="row" spacing={2}>
                                    <Box sx={{ flex: 1 }}>
                                        <Typography variant="body2" color="text.secondary">
                                            Risk Amount
                                        </Typography>
                                        <Typography variant="h6" color="error">
                                            ${result.risk.toFixed(2)}
                                        </Typography>
                                    </Box>
                                    <Box sx={{ flex: 1 }}>
                                        <Typography variant="body2" color="text.secondary">
                                            Potential Profit
                                        </Typography>
                                        <Typography variant="h6" color="success.main">
                                            ${result.potentialProfit.toFixed(2)}
                                        </Typography>
                                    </Box>
                                </Stack>

                                <Box
                                    sx={{
                                        p: 1.5,
                                        borderRadius: 1,
                                        background: alpha(
                                            result.riskRewardRatio >= 2
                                                ? theme.palette.success.main
                                                : result.riskRewardRatio >= 1.5
                                                    ? theme.palette.warning.main
                                                    : theme.palette.error.main,
                                            0.1
                                        ),
                                        textAlign: 'center',
                                    }}
                                >
                                    <Typography variant="body2" color="text.secondary">
                                        Risk/Reward Ratio
                                    </Typography>
                                    <Typography
                                        variant="h6"
                                        fontWeight="bold"
                                        color={
                                            result.riskRewardRatio >= 2
                                                ? 'success.main'
                                                : result.riskRewardRatio >= 1.5
                                                    ? 'warning.main'
                                                    : 'error.main'
                                        }
                                    >
                                        1:{result.riskRewardRatio.toFixed(2)}
                                    </Typography>
                                    {result.riskRewardRatio < 1.5 && (
                                        <Typography variant="caption" color="text.secondary">
                                            Consider a better risk/reward setup
                                        </Typography>
                                    )}
                                </Box>

                                {/* Quick Summary */}
                                <Alert
                                    severity="info"
                                    sx={{
                                        '& .MuiAlert-message': { width: '100%' }
                                    }}
                                >
                                    <Stack spacing={0.5}>
                                        <Typography variant="caption">
                                            <strong>Entry:</strong> ${entryPrice} |
                                            <strong> Stop:</strong> ${stopLoss} ({Math.abs(((stopLoss - entryPrice) / entryPrice) * 100).toFixed(1)}%) |
                                            <strong> Target:</strong> ${targetPrice} ({Math.abs(((targetPrice - entryPrice) / entryPrice) * 100).toFixed(1)}%)
                                        </Typography>
                                        <Typography variant="caption">
                                            <strong>Max Loss:</strong> ${result.risk.toFixed(2)} |
                                            <strong> Potential Gain:</strong> ${result.potentialProfit.toFixed(2)}
                                        </Typography>
                                    </Stack>
                                </Alert>
                            </Stack>
                        </Box>
                    )}
                </Stack>
            </CardContent>
        </Card>
    );
};

export default PositionSizeCalculator;
