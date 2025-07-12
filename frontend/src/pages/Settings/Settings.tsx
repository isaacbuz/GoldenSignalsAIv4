import React, { useState } from 'react';
import {
    Box,
    Grid,
    Card,
    CardContent,
    Typography,
    Switch,
    FormControlLabel,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Button,
    Stack,
    Alert,
    Chip,
    alpha,
    useTheme,
} from '@mui/material';
import {
    Notifications,
    Language,
    Speed,
    Storage,
    Save,
    RestartAlt,
    AutoAwesome,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

const StyledCard = styled(Card)(({ theme }) => ({
    background: alpha(theme.palette.background.paper, 0.8),
    backdropFilter: 'blur(20px)',
    border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
    transition: 'all 0.3s ease',
    '&:hover': {
        border: `1px solid ${alpha('#FFD700', 0.2)}`,
        boxShadow: `0 4px 20px ${alpha('#FFD700', 0.05)}`,
    },
}));

const SectionTitle = styled(Typography)(({ theme }) => ({
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(1),
    marginBottom: theme.spacing(2),
    fontWeight: 600,
}));

const GoldButton = styled(Button)(({ theme }) => ({
    background: 'linear-gradient(135deg, #FFD700 0%, #FFA500 100%)',
    color: '#0a0a0a',
    fontWeight: 600,
    '&:hover': {
        background: 'linear-gradient(135deg, #FFD700 0%, #FF8C00 100%)',
        boxShadow: '0 4px 20px rgba(255, 215, 0, 0.3)',
    },
}));

const Settings: React.FC = () => {
    const theme = useTheme();
    const [settings, setSettings] = useState({
        // Notifications
        emailNotifications: true,
        pushNotifications: true,
        signalAlerts: true,
        priceAlerts: false,

        // Display
        language: 'en',
        timeZone: 'America/New_York',

        // Trading
        defaultTimeframe: '1D',
        autoRefresh: true,
        refreshInterval: 30,

        // Performance
        enableAnimations: true,
        enableWebSocket: true,
        cacheEnabled: true,
    });

    const [saved, setSaved] = useState(false);

    const handleChange = (key: string, value: any) => {
        setSettings(prev => ({ ...prev, [key]: value }));
        setSaved(false);
    };

    const handleSave = () => {
        // Save settings logic here
        setSaved(true);
        setTimeout(() => setSaved(false), 3000);
    };

    const handleReset = () => {
        // Reset to defaults
        setSettings({
            emailNotifications: true,
            pushNotifications: true,
            signalAlerts: true,
            priceAlerts: false,
            language: 'en',
            timeZone: 'America/New_York',
            defaultTimeframe: '1D',
            autoRefresh: true,
            refreshInterval: 30,
            enableAnimations: true,
            enableWebSocket: true,
            cacheEnabled: true,
        });
    };

    return (
        <Box>
            {/* Header */}
            <Box sx={{ mb: 3 }}>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                    Settings
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    Customize your trading experience
                </Typography>
            </Box>

            {saved && (
                <Alert
                    severity="success"
                    icon={<AutoAwesome />}
                    sx={{ mb: 3, bgcolor: alpha('#4CAF50', 0.1) }}
                >
                    Settings saved successfully!
                </Alert>
            )}

            <Grid container spacing={3}>
                {/* Notifications */}
                <Grid item xs={12} md={6}>
                    <StyledCard>
                        <CardContent>
                            <SectionTitle variant="h6">
                                <Notifications sx={{ color: '#FFD700' }} />
                                Notifications
                            </SectionTitle>

                            <Stack spacing={2}>
                                <FormControlLabel
                                    control={
                                        <Switch
                                            checked={settings.emailNotifications}
                                            onChange={(e) => handleChange('emailNotifications', e.target.checked)}
                                            sx={{
                                                '& .MuiSwitch-switchBase.Mui-checked': {
                                                    color: '#FFD700',
                                                },
                                                '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                                                    backgroundColor: '#FFD700',
                                                },
                                            }}
                                        />
                                    }
                                    label="Email Notifications"
                                />

                                <FormControlLabel
                                    control={
                                        <Switch
                                            checked={settings.pushNotifications}
                                            onChange={(e) => handleChange('pushNotifications', e.target.checked)}
                                            sx={{
                                                '& .MuiSwitch-switchBase.Mui-checked': {
                                                    color: '#FFD700',
                                                },
                                                '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                                                    backgroundColor: '#FFD700',
                                                },
                                            }}
                                        />
                                    }
                                    label="Push Notifications"
                                />

                                <FormControlLabel
                                    control={
                                        <Switch
                                            checked={settings.signalAlerts}
                                            onChange={(e) => handleChange('signalAlerts', e.target.checked)}
                                            sx={{
                                                '& .MuiSwitch-switchBase.Mui-checked': {
                                                    color: '#FFD700',
                                                },
                                                '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                                                    backgroundColor: '#FFD700',
                                                },
                                            }}
                                        />
                                    }
                                    label="AI Signal Alerts"
                                />

                                <FormControlLabel
                                    control={
                                        <Switch
                                            checked={settings.priceAlerts}
                                            onChange={(e) => handleChange('priceAlerts', e.target.checked)}
                                            sx={{
                                                '& .MuiSwitch-switchBase.Mui-checked': {
                                                    color: '#FFD700',
                                                },
                                                '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                                                    backgroundColor: '#FFD700',
                                                },
                                            }}
                                        />
                                    }
                                    label="Price Movement Alerts"
                                />
                            </Stack>
                        </CardContent>
                    </StyledCard>
                </Grid>

                {/* Localization */}
                <Grid item xs={12} md={6}>
                    <StyledCard>
                        <CardContent>
                            <SectionTitle variant="h6">
                                <Language sx={{ color: '#FFD700' }} />
                                Localization
                            </SectionTitle>

                            <Stack spacing={3}>
                                <FormControl fullWidth size="small">
                                    <InputLabel>Language</InputLabel>
                                    <Select
                                        value={settings.language}
                                        label="Language"
                                        onChange={(e) => handleChange('language', e.target.value)}
                                    >
                                        <MenuItem value="en">English</MenuItem>
                                        <MenuItem value="es">Español</MenuItem>
                                        <MenuItem value="fr">Français</MenuItem>
                                        <MenuItem value="de">Deutsch</MenuItem>
                                    </Select>
                                </FormControl>

                                <FormControl fullWidth size="small">
                                    <InputLabel>Time Zone</InputLabel>
                                    <Select
                                        value={settings.timeZone}
                                        label="Time Zone"
                                        onChange={(e) => handleChange('timeZone', e.target.value)}
                                    >
                                        <MenuItem value="America/New_York">Eastern (NYSE)</MenuItem>
                                        <MenuItem value="America/Chicago">Central</MenuItem>
                                        <MenuItem value="America/Los_Angeles">Pacific</MenuItem>
                                        <MenuItem value="Europe/London">London</MenuItem>
                                        <MenuItem value="Asia/Tokyo">Tokyo</MenuItem>
                                    </Select>
                                </FormControl>
                            </Stack>
                        </CardContent>
                    </StyledCard>
                </Grid>

                {/* Trading Preferences */}
                <Grid item xs={12} md={6}>
                    <StyledCard>
                        <CardContent>
                            <SectionTitle variant="h6">
                                <AutoAwesome sx={{ color: '#FFD700' }} />
                                Trading Preferences
                            </SectionTitle>

                            <Stack spacing={3}>
                                <FormControl fullWidth size="small">
                                    <InputLabel>Default Timeframe</InputLabel>
                                    <Select
                                        value={settings.defaultTimeframe}
                                        label="Default Timeframe"
                                        onChange={(e) => handleChange('defaultTimeframe', e.target.value)}
                                    >
                                        <MenuItem value="1D">1 Day</MenuItem>
                                        <MenuItem value="1W">1 Week</MenuItem>
                                        <MenuItem value="1M">1 Month</MenuItem>
                                        <MenuItem value="3M">3 Months</MenuItem>
                                        <MenuItem value="1Y">1 Year</MenuItem>
                                    </Select>
                                </FormControl>

                                <FormControlLabel
                                    control={
                                        <Switch
                                            checked={settings.autoRefresh}
                                            onChange={(e) => handleChange('autoRefresh', e.target.checked)}
                                            sx={{
                                                '& .MuiSwitch-switchBase.Mui-checked': {
                                                    color: '#FFD700',
                                                },
                                                '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                                                    backgroundColor: '#FFD700',
                                                },
                                            }}
                                        />
                                    }
                                    label="Auto-refresh Data"
                                />

                                {settings.autoRefresh && (
                                    <FormControl fullWidth size="small">
                                        <InputLabel>Refresh Interval</InputLabel>
                                        <Select
                                            value={settings.refreshInterval}
                                            label="Refresh Interval"
                                            onChange={(e) => handleChange('refreshInterval', e.target.value)}
                                        >
                                            <MenuItem value={10}>10 seconds</MenuItem>
                                            <MenuItem value={30}>30 seconds</MenuItem>
                                            <MenuItem value={60}>1 minute</MenuItem>
                                            <MenuItem value={300}>5 minutes</MenuItem>
                                        </Select>
                                    </FormControl>
                                )}
                            </Stack>
                        </CardContent>
                    </StyledCard>
                </Grid>

                {/* Performance */}
                <Grid item xs={12} md={6}>
                    <StyledCard>
                        <CardContent>
                            <SectionTitle variant="h6">
                                <Speed sx={{ color: '#FFD700' }} />
                                Performance
                            </SectionTitle>

                            <Stack spacing={2}>
                                <FormControlLabel
                                    control={
                                        <Switch
                                            checked={settings.enableAnimations}
                                            onChange={(e) => handleChange('enableAnimations', e.target.checked)}
                                            sx={{
                                                '& .MuiSwitch-switchBase.Mui-checked': {
                                                    color: '#FFD700',
                                                },
                                                '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                                                    backgroundColor: '#FFD700',
                                                },
                                            }}
                                        />
                                    }
                                    label="Enable Animations"
                                />

                                <FormControlLabel
                                    control={
                                        <Switch
                                            checked={settings.enableWebSocket}
                                            onChange={(e) => handleChange('enableWebSocket', e.target.checked)}
                                            sx={{
                                                '& .MuiSwitch-switchBase.Mui-checked': {
                                                    color: '#FFD700',
                                                },
                                                '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                                                    backgroundColor: '#FFD700',
                                                },
                                            }}
                                        />
                                    }
                                    label="Real-time Updates"
                                />

                                <FormControlLabel
                                    control={
                                        <Switch
                                            checked={settings.cacheEnabled}
                                            onChange={(e) => handleChange('cacheEnabled', e.target.checked)}
                                            sx={{
                                                '& .MuiSwitch-switchBase.Mui-checked': {
                                                    color: '#FFD700',
                                                },
                                                '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                                                    backgroundColor: '#FFD700',
                                                },
                                            }}
                                        />
                                    }
                                    label="Enable Cache"
                                />

                                <Box sx={{ mt: 2 }}>
                                    <Chip
                                        icon={<Storage />}
                                        label="Cache Size: 24.5 MB"
                                        size="small"
                                        sx={{ mr: 1 }}
                                    />
                                    <Button size="small" startIcon={<RestartAlt />}>
                                        Clear Cache
                                    </Button>
                                </Box>
                            </Stack>
                        </CardContent>
                    </StyledCard>
                </Grid>
            </Grid>

            {/* Actions */}
            <Box sx={{ mt: 4, display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
                <Button
                    variant="outlined"
                    startIcon={<RestartAlt />}
                    onClick={handleReset}
                >
                    Reset to Defaults
                </Button>
                <GoldButton
                    variant="contained"
                    startIcon={<Save />}
                    onClick={handleSave}
                >
                    Save Changes
                </GoldButton>
            </Box>
        </Box>
    );
};

export default Settings; 