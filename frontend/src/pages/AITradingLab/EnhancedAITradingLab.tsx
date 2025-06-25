/**
 * Enhanced AI Trading Lab - Consolidated View
 * This is a mockup showing how all AI features will be integrated
 */

import React, { useState } from 'react';
import {
    Box,
    Tabs,
    Tab,
    Paper,
    Typography,
    Badge,
    Chip,
    useTheme,
    alpha,
    Stack,
    IconButton,
    Tooltip,
} from '@mui/material';
import {
    // Tab Icons
    AutoAwesome,        // Autonomous Trading
    Psychology,         // Signal Prophet
    Groups,            // Agent Fleet
    Speed,             // Command Center
    Architecture,      // Strategy Builder
    ShowChart,         // Backtesting
    CurrencyExchange,  // Paper Trading
    ModelTraining,     // ML Training
    Analytics,         // Performance
    Settings,          // Settings/Config

    // Status Icons
    FiberManualRecord,
    Notifications,
} from '@mui/icons-material';

interface TabConfig {
    id: string;
    label: string;
    icon: React.ReactNode;
    badge?: string | number;
    color?: string;
    description?: string;
}

const EnhancedAITradingLab: React.FC = () => {
    const theme = useTheme();
    const [activeTab, setActiveTab] = useState('command-center');

    // Tab configuration with all consolidated features
    const tabs: TabConfig[] = [
        {
            id: 'command-center',
            label: 'Command Center',
            icon: <Speed />,
            badge: '3',
            color: theme.palette.primary.main,
            description: 'Real-time monitoring and control of all AI agents'
        },
        {
            id: 'autonomous',
            label: 'Autonomous Trading',
            icon: <AutoAwesome />,
            color: theme.palette.success.main,
            description: 'AI-powered autonomous trading with live chart analysis'
        },
        {
            id: 'signal-prophet',
            label: 'Signal Prophet',
            icon: <Psychology />,
            badge: 'NEW',
            color: theme.palette.secondary.main,
            description: 'Advanced AI signal generation and prediction'
        },
        {
            id: 'agent-fleet',
            label: 'Agent Fleet',
            icon: <Groups />,
            color: theme.palette.info.main,
            description: 'Manage and monitor your AI agent fleet'
        },
        {
            id: 'strategy-builder',
            label: 'Strategy Builder',
            icon: <Architecture />,
            badge: 'BETA',
            color: theme.palette.warning.main,
            description: 'Visual and code-based strategy creation'
        },
        {
            id: 'backtesting',
            label: 'Backtesting',
            icon: <ShowChart />,
            color: theme.palette.error.main,
            description: 'Test strategies with historical data'
        },
        {
            id: 'paper-trading',
            label: 'Paper Trading',
            icon: <CurrencyExchange />,
            color: theme.palette.success.light,
            description: 'Risk-free live market simulation'
        },
        {
            id: 'ml-training',
            label: 'ML Training',
            icon: <ModelTraining />,
            color: theme.palette.purple[500],
            description: 'Train custom machine learning models'
        },
        {
            id: 'performance',
            label: 'Performance',
            icon: <Analytics />,
            color: theme.palette.orange[500],
            description: 'Comprehensive performance analytics'
        },
    ];

    const handleTabChange = (event: React.SyntheticEvent, newValue: string) => {
        setActiveTab(newValue);
    };

    // Enhanced tab label with badges and styling
    const TabLabel: React.FC<{ tab: TabConfig }> = ({ tab }) => (
        <Stack direction="row" spacing={1} alignItems="center">
            <Box sx={{ color: tab.color }}>{tab.icon}</Box>
            <Typography variant="body2" sx={{ textTransform: 'none' }}>
                {tab.label}
            </Typography>
            {tab.badge && (
                <Chip
                    label={tab.badge}
                    size="small"
                    sx={{
                        height: 18,
                        fontSize: '0.7rem',
                        bgcolor: typeof tab.badge === 'number'
                            ? theme.palette.error.main
                            : theme.palette.primary.main,
                        color: 'white',
                    }}
                />
            )}
        </Stack>
    );

    return (
        <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', bgcolor: '#0a0e1a' }}>
            {/* Header */}
            <Paper sx={{
                px: 3,
                py: 2,
                borderRadius: 0,
                bgcolor: '#131722',
                borderBottom: '1px solid #1e222d'
            }}>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                    <Stack direction="row" spacing={2} alignItems="center">
                        <Typography variant="h5" fontWeight="bold" color="white">
                            AI Trading Lab
                        </Typography>
                        <Chip
                            icon={<FiberManualRecord sx={{ fontSize: 12 }} />}
                            label="Live"
                            size="small"
                            sx={{
                                bgcolor: alpha(theme.palette.success.main, 0.1),
                                color: theme.palette.success.main,
                                '& .MuiChip-icon': { color: theme.palette.success.main }
                            }}
                        />
                    </Stack>
                    <Stack direction="row" spacing={1}>
                        <Tooltip title="Notifications">
                            <IconButton size="small">
                                <Badge badgeContent={5} color="error">
                                    <Notifications sx={{ color: '#787b86' }} />
                                </Badge>
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Lab Settings">
                            <IconButton size="small">
                                <Settings sx={{ color: '#787b86' }} />
                            </IconButton>
                        </Tooltip>
                    </Stack>
                </Stack>
            </Paper>

            {/* Enhanced Tabs */}
            <Paper sx={{ bgcolor: '#131722', borderRadius: 0, borderBottom: '1px solid #1e222d' }}>
                <Tabs
                    value={activeTab}
                    onChange={handleTabChange}
                    variant="scrollable"
                    scrollButtons="auto"
                    sx={{
                        '& .MuiTab-root': {
                            color: '#787b86',
                            minHeight: 64,
                            textTransform: 'none',
                            fontSize: '0.875rem',
                            '&.Mui-selected': {
                                color: '#fff',
                                bgcolor: alpha(theme.palette.primary.main, 0.08),
                            },
                            '&:hover': {
                                bgcolor: alpha(theme.palette.primary.main, 0.04),
                            },
                        },
                        '& .MuiTabs-indicator': {
                            height: 3,
                            backgroundColor: theme.palette.primary.main,
                        },
                    }}
                >
                    {tabs.map((tab) => (
                        <Tab
                            key={tab.id}
                            value={tab.id}
                            label={<TabLabel tab={tab} />}
                        />
                    ))}
                </Tabs>
            </Paper>

            {/* Tab Content Area */}
            <Box sx={{ flex: 1, overflow: 'hidden', bgcolor: '#0a0e1a', p: 3 }}>
                {/* Active Tab Info */}
                <Paper sx={{
                    p: 2,
                    mb: 3,
                    bgcolor: '#131722',
                    border: `1px solid ${alpha(tabs.find(t => t.id === activeTab)?.color || theme.palette.primary.main, 0.3)}`
                }}>
                    <Stack direction="row" spacing={2} alignItems="center">
                        <Box sx={{
                            color: tabs.find(t => t.id === activeTab)?.color,
                            fontSize: 32
                        }}>
                            {tabs.find(t => t.id === activeTab)?.icon}
                        </Box>
                        <Box>
                            <Typography variant="h6" color="white">
                                {tabs.find(t => t.id === activeTab)?.label}
                            </Typography>
                            <Typography variant="body2" color="#787b86">
                                {tabs.find(t => t.id === activeTab)?.description}
                            </Typography>
                        </Box>
                    </Stack>
                </Paper>

                {/* Dynamic Content Based on Active Tab */}
                <Box sx={{ color: 'white' }}>
                    {activeTab === 'command-center' && (
                        <Typography>Command Center: Monitor all AI agents, system metrics, and live signals in real-time.</Typography>
                    )}
                    {activeTab === 'autonomous' && (
                        <Typography>Autonomous Trading: AI analyzes charts and executes trades automatically.</Typography>
                    )}
                    {activeTab === 'signal-prophet' && (
                        <Typography>Signal Prophet: Advanced signal generation with confidence scores and predictions.</Typography>
                    )}
                    {activeTab === 'agent-fleet' && (
                        <Typography>Agent Fleet: Manage, configure, and monitor individual AI agents.</Typography>
                    )}
                    {activeTab === 'strategy-builder' && (
                        <Typography>Strategy Builder: Create custom strategies with visual flow editor or code.</Typography>
                    )}
                    {activeTab === 'backtesting' && (
                        <Typography>Backtesting: Test your strategies against historical data with detailed analytics.</Typography>
                    )}
                    {activeTab === 'paper-trading' && (
                        <Typography>Paper Trading: Practice with virtual money in real market conditions.</Typography>
                    )}
                    {activeTab === 'ml-training' && (
                        <Typography>ML Training: Build and train custom machine learning models for trading.</Typography>
                    )}
                    {activeTab === 'performance' && (
                        <Typography>Performance Analytics: Comprehensive analysis of all strategies and agents.</Typography>
                    )}
                </Box>
            </Box>
        </Box>
    );
};

export default EnhancedAITradingLab; 