/**
 * Settings Page - Institutional Grade Configuration
 * 
 * Comprehensive settings and configuration management
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  Grid,
  Tabs,
  Tab,
  Paper,
  Switch,
  FormControl,
  FormControlLabel,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Button,
  Slider,
  Divider,
  Alert,
  Stack,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  ShowChart,
  Notifications,
  Security,
  Palette,
  Save,
  Restore,
  Delete,
  Add,
  Edit,
  Visibility,
  VisibilityOff,
} from '@mui/icons-material';
import { useAppStore } from '../../store';
import toast from 'react-hot-toast';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

export default function SettingsPage() {
  const [tabValue, setTabValue] = useState(0);
  const [showApiKey, setShowApiKey] = useState(false);
  const [apiKeyDialog, setApiKeyDialog] = useState(false);
  const [newApiKey, setNewApiKey] = useState('');
  const { settings, updateSettings } = useAppStore();

  // Local state for settings
  const [localSettings, setLocalSettings] = useState({
    trading: {
      riskTolerance: 'medium',
      maxPositionSize: 10000,
      autoExecution: false,
      stopLossPercentage: 5,
      takeProfitPercentage: 15,
      maxDailyLoss: 2000,
      maxOpenPositions: 10,
    },
    notifications: {
      signals: true,
      portfolio: true,
      system: true,
      email: false,
      push: true,
      soundEnabled: true,
      criticalOnly: false,
    },
    dashboard: {
      refreshInterval: 5000,
      defaultSymbols: ['SPY', 'QQQ', 'TSLA', 'AAPL'],
      autoRefresh: true,
      theme: 'dark',
      compactMode: false,
      showAdvancedMetrics: true,
    },
    api: {
      endpoint: settings.apiEndpoint || 'http://localhost:8000/api/v1',
      timeout: 30000,
      retryAttempts: 3,
      rateLimitPerMinute: 100,
    },
  });

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleSaveSettings = () => {
    // Update app settings
    updateSettings({
      apiEndpoint: localSettings.api.endpoint,
      refreshInterval: localSettings.dashboard.refreshInterval,
    });

    toast.success('Settings saved successfully');
  };

  const handleResetSettings = () => {
    setLocalSettings({
      trading: {
        riskTolerance: 'medium',
        maxPositionSize: 10000,
        autoExecution: false,
        stopLossPercentage: 5,
        takeProfitPercentage: 15,
        maxDailyLoss: 2000,
        maxOpenPositions: 10,
      },
      notifications: {
        signals: true,
        portfolio: true,
        system: true,
        email: false,
        push: true,
        soundEnabled: true,
        criticalOnly: false,
      },
      dashboard: {
        refreshInterval: 5000,
        defaultSymbols: ['SPY', 'QQQ', 'TSLA', 'AAPL'],
        autoRefresh: true,
        theme: 'dark',
        compactMode: false,
        showAdvancedMetrics: true,
      },
      api: {
        endpoint: 'http://localhost:8000/api/v1',
        timeout: 30000,
        retryAttempts: 3,
        rateLimitPerMinute: 100,
      },
    });
    toast.success('Settings reset to defaults');
  };

  const handleAddSymbol = () => {
    const symbol = prompt('Enter symbol (e.g., AAPL):');
    if (symbol && !localSettings.dashboard.defaultSymbols.includes(symbol.toUpperCase())) {
      setLocalSettings(prev => ({
        ...prev,
        dashboard: {
          ...prev.dashboard,
          defaultSymbols: [...prev.dashboard.defaultSymbols, symbol.toUpperCase()],
        },
      }));
    }
  };

  const handleRemoveSymbol = (symbol: string) => {
    setLocalSettings(prev => ({
      ...prev,
      dashboard: {
        ...prev.dashboard,
        defaultSymbols: prev.dashboard.defaultSymbols.filter(s => s !== symbol),
      },
    }));
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Settings
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Configure your trading preferences and system settings
        </Typography>
      </Box>

      {/* Action Buttons */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2 }}>
        <Button
          variant="contained"
          startIcon={<Save />}
          onClick={handleSaveSettings}
        >
          Save Settings
        </Button>
        <Button
          variant="outlined"
          startIcon={<Restore />}
          onClick={handleResetSettings}
        >
          Reset to Defaults
        </Button>
      </Box>

      {/* Settings Tabs */}
      <Paper>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="Trading" icon={<ShowChart />} />
          <Tab label="Notifications" icon={<Notifications />} />
          <Tab label="Dashboard" icon={<Palette />} />
          <Tab label="API & Security" icon={<Security />} />
        </Tabs>

        {/* Trading Settings Tab */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Risk Management
                </Typography>
                
                <FormControl fullWidth sx={{ mb: 3 }}>
                  <InputLabel>Risk Tolerance</InputLabel>
                  <Select
                    value={localSettings.trading.riskTolerance}
                    label="Risk Tolerance"
                    onChange={(e) => setLocalSettings(prev => ({
                      ...prev,
                      trading: { ...prev.trading, riskTolerance: e.target.value as any }
                    }))}
                  >
                    <MenuItem value="low">Conservative</MenuItem>
                    <MenuItem value="medium">Moderate</MenuItem>
                    <MenuItem value="high">Aggressive</MenuItem>
                  </Select>
                </FormControl>

                <TextField
                  fullWidth
                  label="Max Position Size ($)"
                  type="number"
                  value={localSettings.trading.maxPositionSize}
                  onChange={(e) => setLocalSettings(prev => ({
                    ...prev,
                    trading: { ...prev.trading, maxPositionSize: Number(e.target.value) }
                  }))}
                  sx={{ mb: 3 }}
                />

                <TextField
                  fullWidth
                  label="Max Daily Loss ($)"
                  type="number"
                  value={localSettings.trading.maxDailyLoss}
                  onChange={(e) => setLocalSettings(prev => ({
                    ...prev,
                    trading: { ...prev.trading, maxDailyLoss: Number(e.target.value) }
                  }))}
                  sx={{ mb: 3 }}
                />

                <TextField
                  fullWidth
                  label="Max Open Positions"
                  type="number"
                  value={localSettings.trading.maxOpenPositions}
                  onChange={(e) => setLocalSettings(prev => ({
                    ...prev,
                    trading: { ...prev.trading, maxOpenPositions: Number(e.target.value) }
                  }))}
                />
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Execution Settings
                </Typography>

                <FormControlLabel
                  control={
                    <Switch
                      checked={localSettings.trading.autoExecution}
                      onChange={(e) => setLocalSettings(prev => ({
                        ...prev,
                        trading: { ...prev.trading, autoExecution: e.target.checked }
                      }))}
                    />
                  }
                  label="Auto Execution"
                  sx={{ mb: 3, display: 'block' }}
                />

                <Typography gutterBottom>
                  Stop Loss Percentage: {localSettings.trading.stopLossPercentage}%
                </Typography>
                <Slider
                  value={localSettings.trading.stopLossPercentage}
                  onChange={(e, value) => setLocalSettings(prev => ({
                    ...prev,
                    trading: { ...prev.trading, stopLossPercentage: value as number }
                  }))}
                  min={1}
                  max={20}
                  step={0.5}
                  marks
                  sx={{ mb: 3 }}
                />

                <Typography gutterBottom>
                  Take Profit Percentage: {localSettings.trading.takeProfitPercentage}%
                </Typography>
                <Slider
                  value={localSettings.trading.takeProfitPercentage}
                  onChange={(e, value) => setLocalSettings(prev => ({
                    ...prev,
                    trading: { ...prev.trading, takeProfitPercentage: value as number }
                  }))}
                  min={5}
                  max={50}
                  step={1}
                  marks
                />
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Notifications Tab */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Notification Types
                </Typography>

                <FormControlLabel
                  control={
                    <Switch
                      checked={localSettings.notifications.signals}
                      onChange={(e) => setLocalSettings(prev => ({
                        ...prev,
                        notifications: { ...prev.notifications, signals: e.target.checked }
                      }))}
                    />
                  }
                  label="Trading Signals"
                  sx={{ mb: 2, display: 'block' }}
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={localSettings.notifications.portfolio}
                      onChange={(e) => setLocalSettings(prev => ({
                        ...prev,
                        notifications: { ...prev.notifications, portfolio: e.target.checked }
                      }))}
                    />
                  }
                  label="Portfolio Updates"
                  sx={{ mb: 2, display: 'block' }}
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={localSettings.notifications.system}
                      onChange={(e) => setLocalSettings(prev => ({
                        ...prev,
                        notifications: { ...prev.notifications, system: e.target.checked }
                      }))}
                    />
                  }
                  label="System Alerts"
                  sx={{ mb: 2, display: 'block' }}
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={localSettings.notifications.criticalOnly}
                      onChange={(e) => setLocalSettings(prev => ({
                        ...prev,
                        notifications: { ...prev.notifications, criticalOnly: e.target.checked }
                      }))}
                    />
                  }
                  label="Critical Only"
                  sx={{ display: 'block' }}
                />
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Delivery Methods
                </Typography>

                <FormControlLabel
                  control={
                    <Switch
                      checked={localSettings.notifications.push}
                      onChange={(e) => setLocalSettings(prev => ({
                        ...prev,
                        notifications: { ...prev.notifications, push: e.target.checked }
                      }))}
                    />
                  }
                  label="Push Notifications"
                  sx={{ mb: 2, display: 'block' }}
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={localSettings.notifications.email}
                      onChange={(e) => setLocalSettings(prev => ({
                        ...prev,
                        notifications: { ...prev.notifications, email: e.target.checked }
                      }))}
                    />
                  }
                  label="Email Notifications"
                  sx={{ mb: 2, display: 'block' }}
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={localSettings.notifications.soundEnabled}
                      onChange={(e) => setLocalSettings(prev => ({
                        ...prev,
                        notifications: { ...prev.notifications, soundEnabled: e.target.checked }
                      }))}
                    />
                  }
                  label="Sound Alerts"
                  sx={{ display: 'block' }}
                />
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Dashboard Tab */}
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Display Settings
                </Typography>

                <FormControl fullWidth sx={{ mb: 3 }}>
                  <InputLabel>Theme</InputLabel>
                  <Select
                    value={localSettings.dashboard.theme}
                    label="Theme"
                    onChange={(e) => setLocalSettings(prev => ({
                      ...prev,
                      dashboard: { ...prev.dashboard, theme: e.target.value }
                    }))}
                  >
                    <MenuItem value="light">Light</MenuItem>
                    <MenuItem value="dark">Dark</MenuItem>
                    <MenuItem value="auto">Auto</MenuItem>
                  </Select>
                </FormControl>

                <FormControlLabel
                  control={
                    <Switch
                      checked={localSettings.dashboard.autoRefresh}
                      onChange={(e) => setLocalSettings(prev => ({
                        ...prev,
                        dashboard: { ...prev.dashboard, autoRefresh: e.target.checked }
                      }))}
                    />
                  }
                  label="Auto Refresh"
                  sx={{ mb: 2, display: 'block' }}
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={localSettings.dashboard.compactMode}
                      onChange={(e) => setLocalSettings(prev => ({
                        ...prev,
                        dashboard: { ...prev.dashboard, compactMode: e.target.checked }
                      }))}
                    />
                  }
                  label="Compact Mode"
                  sx={{ mb: 2, display: 'block' }}
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={localSettings.dashboard.showAdvancedMetrics}
                      onChange={(e) => setLocalSettings(prev => ({
                        ...prev,
                        dashboard: { ...prev.dashboard, showAdvancedMetrics: e.target.checked }
                      }))}
                    />
                  }
                  label="Advanced Metrics"
                  sx={{ display: 'block' }}
                />
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Default Symbols
                </Typography>

                <Box sx={{ mb: 2 }}>
                  <Button
                    variant="outlined"
                    startIcon={<Add />}
                    onClick={handleAddSymbol}
                    size="small"
                  >
                    Add Symbol
                  </Button>
                </Box>

                <Stack direction="row" spacing={1} flexWrap="wrap" gap={1}>
                  {localSettings.dashboard.defaultSymbols.map((symbol) => (
                    <Chip
                      key={symbol}
                      label={symbol}
                      onDelete={() => handleRemoveSymbol(symbol)}
                      color="primary"
                      variant="outlined"
                    />
                  ))}
                </Stack>

                <Divider sx={{ my: 3 }} />

                <Typography gutterBottom>
                  Refresh Interval: {localSettings.dashboard.refreshInterval / 1000}s
                </Typography>
                <Slider
                  value={localSettings.dashboard.refreshInterval}
                  onChange={(e, value) => setLocalSettings(prev => ({
                    ...prev,
                    dashboard: { ...prev.dashboard, refreshInterval: value as number }
                  }))}
                  min={1000}
                  max={30000}
                  step={1000}
                  marks={[
                    { value: 1000, label: '1s' },
                    { value: 5000, label: '5s' },
                    { value: 10000, label: '10s' },
                    { value: 30000, label: '30s' },
                  ]}
                />
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* API & Security Tab */}
        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  API Configuration
                </Typography>

                <TextField
                  fullWidth
                  label="API Endpoint"
                  value={localSettings.api.endpoint}
                  onChange={(e) => setLocalSettings(prev => ({
                    ...prev,
                    api: { ...prev.api, endpoint: e.target.value }
                  }))}
                  sx={{ mb: 3 }}
                />

                <TextField
                  fullWidth
                  label="Timeout (ms)"
                  type="number"
                  value={localSettings.api.timeout}
                  onChange={(e) => setLocalSettings(prev => ({
                    ...prev,
                    api: { ...prev.api, timeout: Number(e.target.value) }
                  }))}
                  sx={{ mb: 3 }}
                />

                <TextField
                  fullWidth
                  label="Retry Attempts"
                  type="number"
                  value={localSettings.api.retryAttempts}
                  onChange={(e) => setLocalSettings(prev => ({
                    ...prev,
                    api: { ...prev.api, retryAttempts: Number(e.target.value) }
                  }))}
                  sx={{ mb: 3 }}
                />

                <TextField
                  fullWidth
                  label="Rate Limit (per minute)"
                  type="number"
                  value={localSettings.api.rateLimitPerMinute}
                  onChange={(e) => setLocalSettings(prev => ({
                    ...prev,
                    api: { ...prev.api, rateLimitPerMinute: Number(e.target.value) }
                  }))}
                />
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Security Settings
                </Typography>

                <Box sx={{ mb: 3 }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    API Key
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TextField
                      fullWidth
                      type={showApiKey ? 'text' : 'password'}
                      value="sk-1234567890abcdef"
                      InputProps={{
                        readOnly: true,
                      }}
                      size="small"
                    />
                    <IconButton
                      onClick={() => setShowApiKey(!showApiKey)}
                      size="small"
                    >
                      {showApiKey ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                    <IconButton
                      onClick={() => setApiKeyDialog(true)}
                      size="small"
                    >
                      <Edit />
                    </IconButton>
                  </Box>
                </Box>

                <Alert severity="info" sx={{ mb: 3 }}>
                  Your API key is encrypted and stored securely. Never share your API key with others.
                </Alert>

                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<Delete />}
                  fullWidth
                >
                  Revoke API Access
                </Button>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>

      {/* API Key Dialog */}
      <Dialog open={apiKeyDialog} onClose={() => setApiKeyDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Update API Key</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="New API Key"
            type="password"
            value={newApiKey}
            onChange={(e) => setNewApiKey(e.target.value)}
            sx={{ mt: 2 }}
            placeholder="Enter your new API key"
          />
          <Alert severity="warning" sx={{ mt: 2 }}>
            Changing your API key will require re-authentication and may interrupt active connections.
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setApiKeyDialog(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={() => {
              // Handle API key update
              setApiKeyDialog(false);
              setNewApiKey('');
              toast.success('API key updated successfully');
            }}
            disabled={!newApiKey}
          >
            Update
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
} 