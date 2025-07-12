import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
    Box,
    Card,
    CardContent,
    Typography,
    Chip,
    Stack,
    IconButton,
    Tooltip,
    useTheme,
    alpha,
    Grid,
    Paper,
    Divider,
    Switch,
    FormControlLabel,
    ButtonGroup,
    Button,
    Slider,
} from '@mui/material';
import {
    TrendingUp,
    TrendingDown,
    ShowChart,
    Timeline,
    Refresh,
    Fullscreen,
    Settings,
    CallMade,
    CallReceived,
    Speed,
    Analytics,
    Visibility,
    VisibilityOff,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

const MassiveChartContainer = styled(Box)(({ theme }) => ({
    width: '100%',
    height: 800,
    position: 'relative',
    background: `linear-gradient(135deg, ${alpha('#0A0E1A', 0.95)} 0%, ${alpha('#131A2A', 0.9)} 100%)`,
    borderRadius: theme.spacing(2),
    border: `2px solid ${alpha('#FFD700', 0.3)}`,
    overflow: 'hidden',
    boxShadow: `0 20px 60px ${alpha('#FFD700', 0.1)}`,
}));

const OptionsFlowPanel = styled(Paper)(({ theme }) => ({
    background: `linear-gradient(135deg, ${alpha('#1E293B', 0.8)} 0%, ${alpha('#334155', 0.6)} 100%)`,
    border: `1px solid ${alpha('#FFD700', 0.2)}`,
    borderRadius: theme.spacing(1),
    padding: theme.spacing(2),
    height: '100%',
}));

const GammaBar = styled(Box)<{ intensity: number }>(({ theme, intensity }) => ({
    height: `${intensity}%`,
    background: intensity > 70
        ? `linear-gradient(to top, #FF4757, #FF6B7A)`
        : intensity > 40
            ? `linear-gradient(to top, #FFA500, #FFB84D)`
            : `linear-gradient(to top, #00D4AA, #4DFFDF)`,
    borderRadius: '2px 2px 0 0',
    position: 'relative',
    minHeight: '10px',
    transition: 'all 0.3s ease',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '2px',
        background: alpha('#FFD700', 0.8),
        borderRadius: '2px',
    }
}));

interface OptionsData {
    strike: number;
    callVolume: number;
    putVolume: number;
    callOI: number;
    putOI: number;
    gamma: number;
    delta: number;
    iv: number;
}

interface MassiveOptionsChartProps {
    symbol?: string;
    height?: number;
    showOptionsFlow?: boolean;
    showGammaExposure?: boolean;
    showDarkPool?: boolean;
}

export const MassiveOptionsChart: React.FC<MassiveOptionsChartProps> = ({
    symbol = 'SPY',
    height = 800,
    showOptionsFlow = true,
    showGammaExposure = true,
    showDarkPool = true,
}) => {
    const theme = useTheme();
    const [currentPrice, setCurrentPrice] = useState(458.32);
    const [timeframe, setTimeframe] = useState('1D');
    const [showHeatmap, setShowHeatmap] = useState(true);
    const [gammaExposure, setGammaExposure] = useState(2.4);
    const [darkPoolIndex, setDarkPoolIndex] = useState(67.8);

    // Generate realistic options data
    const optionsData = useMemo(() => {
        const strikes: OptionsData[] = [];
        const baseStrike = Math.floor(currentPrice / 5) * 5;

        for (let i = -10; i <= 10; i++) {
            const strike = baseStrike + (i * 5);
            const distanceFromMoney = Math.abs(strike - currentPrice);
            const isITM = (i < 0 && strike < currentPrice) || (i > 0 && strike > currentPrice);

            strikes.push({
                strike,
                callVolume: Math.floor(Math.random() * 5000 + 1000) * (isITM ? 0.7 : 1.3),
                putVolume: Math.floor(Math.random() * 3000 + 500) * (isITM ? 1.3 : 0.7),
                callOI: Math.floor(Math.random() * 15000 + 2000),
                putOI: Math.floor(Math.random() * 12000 + 1500),
                gamma: Math.max(0.1, 1 - (distanceFromMoney / 50)),
                delta: strike < currentPrice ? 0.8 - (distanceFromMoney / 100) : 0.2 + (distanceFromMoney / 100),
                iv: 0.2 + (Math.random() * 0.3) + (distanceFromMoney / 1000),
            });
        }
        return strikes;
    }, [currentPrice]);

    // Simulate real-time updates
    useEffect(() => {
        const interval = setInterval(() => {
            const change = (Math.random() - 0.5) * 2;
            setCurrentPrice(prev => Math.max(400, Math.min(500, prev + change)));
            setGammaExposure(prev => Math.max(0, Math.min(5, prev + (Math.random() - 0.5) * 0.2)));
            setDarkPoolIndex(prev => Math.max(0, Math.min(100, prev + (Math.random() - 0.5) * 5)));
        }, 3000);

        return () => clearInterval(interval);
    }, []);

    const maxVolume = Math.max(...optionsData.map(d => Math.max(d.callVolume, d.putVolume)));

    return (
        <Card
            elevation={8}
            sx={{
                background: 'linear-gradient(135deg, rgba(10, 14, 26, 0.95) 0%, rgba(19, 26, 42, 0.9) 100%)',
                border: '2px solid rgba(255, 215, 0, 0.3)',
                borderRadius: 4,
                overflow: 'visible',
                position: 'relative',
                '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: -2,
                    left: -2,
                    right: -2,
                    bottom: -2,
                    background: 'linear-gradient(45deg, #FFD700, #FFA500, #FFD700)',
                    borderRadius: 4,
                    zIndex: -1,
                    opacity: 0.3,
                },
            }}
        >
            <CardContent sx={{ p: 0 }}>
                {/* Header */}
                <Box sx={{ p: 3, borderBottom: '1px solid rgba(255, 215, 0, 0.2)' }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                        <Box>
                            <Typography
                                variant="h4"
                                fontWeight="bold"
                                sx={{
                                    background: 'linear-gradient(45deg, #FFD700, #FFA500)',
                                    backgroundClip: 'text',
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 2
                                }}
                            >
                                <Analytics sx={{ color: '#FFD700', fontSize: 40 }} />
                                {symbol} MASSIVE OPTIONS FLOW
                            </Typography>
                            <Stack direction="row" spacing={3} sx={{ mt: 2 }}>
                                <Typography variant="h3" fontWeight="bold" color="#FFD700">
                                    ${currentPrice.toFixed(2)}
                                </Typography>
                                <Box>
                                    <Typography variant="caption" color="text.secondary">GAMMA EXPOSURE</Typography>
                                    <Typography variant="h6" fontWeight="bold" color="#00D4AA">
                                        {gammaExposure.toFixed(1)}B
                                    </Typography>
                                </Box>
                                <Box>
                                    <Typography variant="caption" color="text.secondary">DARK POOL INDEX</Typography>
                                    <Typography variant="h6" fontWeight="bold" color="#FF6B7A">
                                        {darkPoolIndex.toFixed(1)}%
                                    </Typography>
                                </Box>
                            </Stack>
                        </Box>

                        <Stack spacing={2}>
                            <ButtonGroup variant="outlined" size="small">
                                {['1D', '1W', '1M', '3M'].map((tf) => (
                                    <Button
                                        key={tf}
                                        variant={timeframe === tf ? 'contained' : 'outlined'}
                                        onClick={() => setTimeframe(tf)}
                                        sx={{
                                            color: timeframe === tf ? '#000' : '#FFD700',
                                            borderColor: '#FFD700',
                                            '&.Mui-selected': { background: '#FFD700' }
                                        }}
                                    >
                                        {tf}
                                    </Button>
                                ))}
                            </ButtonGroup>
                            <Stack direction="row" spacing={1}>
                                <FormControlLabel
                                    control={<Switch checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} />}
                                    label="Heatmap"
                                    sx={{ color: 'text.secondary' }}
                                />
                            </Stack>
                        </Stack>
                    </Box>
                </Box>

                {/* Main Chart Area */}
                <MassiveChartContainer sx={{ height }}>
                    <Grid container sx={{ height: '100%' }}>
                        {/* Options Chain Heatmap */}
                        <Grid item xs={8}>
                            <Box sx={{ p: 2, height: '100%' }}>
                                <Typography variant="h6" color="#FFD700" gutterBottom>
                                    OPTIONS CHAIN HEATMAP
                                </Typography>

                                {/* Options Strikes */}
                                <Box sx={{ display: 'flex', flexDirection: 'column', height: 'calc(100% - 40px)' }}>
                                    {optionsData.map((option, index) => (
                                        <Box
                                            key={option.strike}
                                            sx={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                minHeight: 35,
                                                borderBottom: '1px solid rgba(255, 215, 0, 0.1)',
                                                background: Math.abs(option.strike - currentPrice) < 2.5
                                                    ? alpha('#FFD700', 0.1)
                                                    : 'transparent',
                                                px: 1,
                                            }}
                                        >
                                            {/* Put Volume Bar */}
                                            <Box sx={{ width: '25%', display: 'flex', justifyContent: 'flex-end', pr: 1 }}>
                                                <Box
                                                    sx={{
                                                        width: `${(option.putVolume / maxVolume) * 100}%`,
                                                        height: 20,
                                                        background: 'linear-gradient(to left, #FF4757, #FF6B7A)',
                                                        borderRadius: '0 2px 2px 0',
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        justifyContent: 'flex-start',
                                                        pl: 1,
                                                    }}
                                                >
                                                    <Typography variant="caption" color="white" fontWeight="bold">
                                                        {option.putVolume > 1000 ? `${(option.putVolume / 1000).toFixed(1)}K` : option.putVolume}
                                                    </Typography>
                                                </Box>
                                            </Box>

                                            {/* Strike Price */}
                                            <Box sx={{ width: '20%', textAlign: 'center' }}>
                                                <Typography
                                                    variant="body2"
                                                    fontWeight="bold"
                                                    color={Math.abs(option.strike - currentPrice) < 2.5 ? '#FFD700' : 'text.primary'}
                                                >
                                                    ${option.strike}
                                                </Typography>
                                                <Typography variant="caption" color="text.secondary">
                                                    Γ{option.gamma.toFixed(2)}
                                                </Typography>
                                            </Box>

                                            {/* Call Volume Bar */}
                                            <Box sx={{ width: '25%', display: 'flex', pl: 1 }}>
                                                <Box
                                                    sx={{
                                                        width: `${(option.callVolume / maxVolume) * 100}%`,
                                                        height: 20,
                                                        background: 'linear-gradient(to right, #00D4AA, #4DFFDF)',
                                                        borderRadius: '2px 0 0 2px',
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        justifyContent: 'flex-end',
                                                        pr: 1,
                                                    }}
                                                >
                                                    <Typography variant="caption" color="white" fontWeight="bold">
                                                        {option.callVolume > 1000 ? `${(option.callVolume / 1000).toFixed(1)}K` : option.callVolume}
                                                    </Typography>
                                                </Box>
                                            </Box>

                                            {/* IV & Greeks */}
                                            <Box sx={{ width: '30%', pl: 2 }}>
                                                <Stack direction="row" spacing={2}>
                                                    <Box>
                                                        <Typography variant="caption" color="text.secondary">IV</Typography>
                                                        <Typography variant="body2" fontWeight="bold">
                                                            {(option.iv * 100).toFixed(1)}%
                                                        </Typography>
                                                    </Box>
                                                    <Box>
                                                        <Typography variant="caption" color="text.secondary">Δ</Typography>
                                                        <Typography variant="body2" fontWeight="bold">
                                                            {option.delta.toFixed(2)}
                                                        </Typography>
                                                    </Box>
                                                </Stack>
                                            </Box>
                                        </Box>
                                    ))}
                                </Box>
                            </Box>
                        </Grid>

                        {/* Side Panels */}
                        <Grid item xs={4}>
                            <Box sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column', gap: 2 }}>
                                {/* Gamma Exposure */}
                                {showGammaExposure && (
                                    <OptionsFlowPanel sx={{ flex: 1 }}>
                                        <Typography variant="h6" color="#FFD700" gutterBottom>
                                            GAMMA EXPOSURE
                                        </Typography>
                                        <Box sx={{ display: 'flex', alignItems: 'end', height: 150, gap: 1 }}>
                                            {optionsData.slice(5, 15).map((option, index) => (
                                                <Box key={index} sx={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                                                    <GammaBar intensity={option.gamma * 100} />
                                                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, transform: 'rotate(-45deg)' }}>
                                                        {option.strike}
                                                    </Typography>
                                                </Box>
                                            ))}
                                        </Box>
                                        <Typography variant="body2" color="#00D4AA" sx={{ mt: 2 }}>
                                            Total Exposure: ${gammaExposure.toFixed(1)}B
                                        </Typography>
                                    </OptionsFlowPanel>
                                )}

                                {/* Options Flow */}
                                {showOptionsFlow && (
                                    <OptionsFlowPanel sx={{ flex: 1 }}>
                                        <Typography variant="h6" color="#FFD700" gutterBottom>
                                            LIVE OPTIONS FLOW
                                        </Typography>
                                        <Stack spacing={1}>
                                            {[
                                                { type: 'CALL', strike: 460, size: '50K', premium: '$2.45M', color: '#00D4AA' },
                                                { type: 'PUT', strike: 455, size: '75K', premium: '$1.89M', color: '#FF4757' },
                                                { type: 'CALL', strike: 465, size: '25K', premium: '$890K', color: '#00D4AA' },
                                                { type: 'PUT', strike: 450, size: '100K', premium: '$3.2M', color: '#FF4757' },
                                            ].map((flow, index) => (
                                                <Box
                                                    key={index}
                                                    sx={{
                                                        display: 'flex',
                                                        justifyContent: 'space-between',
                                                        alignItems: 'center',
                                                        p: 1,
                                                        background: alpha(flow.color, 0.1),
                                                        borderRadius: 1,
                                                        border: `1px solid ${alpha(flow.color, 0.3)}`,
                                                    }}
                                                >
                                                    <Box>
                                                        <Typography variant="body2" fontWeight="bold" color={flow.color}>
                                                            {flow.type} ${flow.strike}
                                                        </Typography>
                                                        <Typography variant="caption" color="text.secondary">
                                                            {flow.size} contracts
                                                        </Typography>
                                                    </Box>
                                                    <Typography variant="body2" fontWeight="bold">
                                                        {flow.premium}
                                                    </Typography>
                                                </Box>
                                            ))}
                                        </Stack>
                                    </OptionsFlowPanel>
                                )}

                                {/* Dark Pool Activity */}
                                {showDarkPool && (
                                    <OptionsFlowPanel sx={{ flex: 1 }}>
                                        <Typography variant="h6" color="#FFD700" gutterBottom>
                                            DARK POOL ACTIVITY
                                        </Typography>
                                        <Box sx={{ textAlign: 'center', mb: 2 }}>
                                            <Typography variant="h4" fontWeight="bold" color="#FF6B7A">
                                                {darkPoolIndex.toFixed(1)}%
                                            </Typography>
                                            <Typography variant="caption" color="text.secondary">
                                                Dark Pool Index
                                            </Typography>
                                        </Box>
                                        <Stack spacing={1}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                                <Typography variant="body2">Block Trades</Typography>
                                                <Typography variant="body2" fontWeight="bold">847</Typography>
                                            </Box>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                                <Typography variant="body2">Volume</Typography>
                                                <Typography variant="body2" fontWeight="bold">2.4M</Typography>
                                            </Box>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                                <Typography variant="body2">Avg Size</Typography>
                                                <Typography variant="body2" fontWeight="bold">2,834</Typography>
                                            </Box>
                                        </Stack>
                                    </OptionsFlowPanel>
                                )}
                            </Box>
                        </Grid>
                    </Grid>
                </MassiveChartContainer>

                {/* Footer Stats */}
                <Box sx={{ p: 2, borderTop: '1px solid rgba(255, 215, 0, 0.2)' }}>
                    <Grid container spacing={4}>
                        <Grid item xs={3}>
                            <Box sx={{ textAlign: 'center' }}>
                                <Typography variant="h6" fontWeight="bold" color="#00D4AA">
                                    {optionsData.reduce((sum, opt) => sum + opt.callVolume, 0).toLocaleString()}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">Call Volume</Typography>
                            </Box>
                        </Grid>
                        <Grid item xs={3}>
                            <Box sx={{ textAlign: 'center' }}>
                                <Typography variant="h6" fontWeight="bold" color="#FF4757">
                                    {optionsData.reduce((sum, opt) => sum + opt.putVolume, 0).toLocaleString()}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">Put Volume</Typography>
                            </Box>
                        </Grid>
                        <Grid item xs={3}>
                            <Box sx={{ textAlign: 'center' }}>
                                <Typography variant="h6" fontWeight="bold" color="#FFA500">
                                    {(optionsData.reduce((sum, opt) => sum + opt.callVolume, 0) / optionsData.reduce((sum, opt) => sum + opt.putVolume, 0)).toFixed(2)}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">Call/Put Ratio</Typography>
                            </Box>
                        </Grid>
                        <Grid item xs={3}>
                            <Box sx={{ textAlign: 'center' }}>
                                <Typography variant="h6" fontWeight="bold" color="#FFD700">
                                    {(optionsData.reduce((sum, opt) => sum + opt.iv, 0) / optionsData.length * 100).toFixed(1)}%
                                </Typography>
                                <Typography variant="caption" color="text.secondary">Avg IV</Typography>
                            </Box>
                        </Grid>
                    </Grid>
                </Box>
            </CardContent>
        </Card>
    );
}; 