/**
 * Unified Professional Trading Application
 * 
 * This is the main application that brings together all existing components
 * and pages using the Bloomberg professional theme for a cohesive experience.
 * 
 * Key Features:
 * - Reuses all existing components
 * - Professional Bloomberg-inspired design
 * - Unified navigation and layout
 * - Real-time data integration
 * - Multi-page support with seamless transitions
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
    Box,
    AppBar,
    Toolbar,
    Typography,
    IconButton,
    Drawer,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    ListItemButton,
    Badge,
    Avatar,
    Divider,
    useTheme,
    alpha,
    Chip,
    Stack,
    Button,
    Container,
    Grid,
    Card,
    CardContent,
    Tooltip,
} from '@mui/material';
import {
    Menu as MenuIcon,
    Dashboard,
    ShowChart,
    Psychology,
    Analytics,
    Assessment,
    AutoGraph,
    Timeline,
    AccountBalance,
    Settings,
    Notifications,
    Search,
    TrendingUp,
    Speed,
    Insights,
    School,
    Science,
    SmartToy,
} from '@mui/icons-material';
import { Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

// Import existing components
import UnifiedSearchBar from '../../components/Common/UnifiedSearchBar';
import { FloatingAIProphetWidget } from '../../components/AI/FloatingAIProphetWidget';
import AgentConsensusFlow from '../../components/Agents/AgentConsensusFlow';

// Import existing pages
import ProfessionalDashboard from './ProfessionalDashboard';
import { UnifiedDashboard } from '../../components/Dashboard/UnifiedDashboard';
import { ProfessionalTradingDashboard } from '../../components/Dashboard/ProfessionalTradingDashboard';
import TradingSignalsApp from '../TradingSignals/TradingSignalsApp';
import { BacktestingInterface } from '../../components/Backtesting/BacktestingInterface';
import { AIInsightsPanel } from '../../components/AI/AIInsightsPanel';
import { UnifiedChart } from '../../components/Chart/UnifiedChart';
import { ExplodedHeatMap } from '../../components/Market/ExplodedHeatMap';
import { VoiceSignalAssistant } from '../../components/AI/VoiceSignalAssistant';
import { SignalList } from '../../components/Signals/SignalList';

// Navigation items with existing pages
const navigationItems = [
    {
        id: 'dashboard',
        label: 'Professional Dashboard',
        icon: <Dashboard />,
        path: '/professional/dashboard',
        component: ProfessionalDashboard,
        description: 'Bloomberg-style professional dashboard'
    },
    {
        id: 'unified',
        label: 'Unified Dashboard',
        icon: <ShowChart />,
        path: '/professional/unified',
        component: UnifiedDashboard,
        description: 'Comprehensive trading dashboard'
    },
    {
        id: 'trading',
        label: 'Trading Dashboard',
        icon: <TrendingUp />,
        path: '/professional/trading',
        component: ProfessionalTradingDashboard,
        description: 'Advanced trading interface'
    },
    {
        id: 'signals',
        label: 'Trading Signals',
        icon: <Psychology />,
        path: '/professional/signals',
        component: TradingSignalsApp,
        description: 'Real-time trading signals'
    },
    {
        id: 'analytics',
        label: 'Market Analytics',
        icon: <Analytics />,
        path: '/professional/analytics',
        component: () => (
            <Grid container spacing={3}>
                <Grid item xs={12}>
                    <UnifiedChart symbol="SPY" height={600} showAdvancedFeatures={true} />
                </Grid>
                <Grid item xs={12}>
                    <ExplodedHeatMap height={400} />
                </Grid>
            </Grid>
        ),
        description: 'Advanced market analysis'
    },
    {
        id: 'backtesting',
        label: 'Backtesting',
        icon: <Assessment />,
        path: '/professional/backtesting',
        component: BacktestingInterface,
        description: 'Strategy backtesting suite'
    },
    {
        id: 'ai-lab',
        label: 'AI Laboratory',
        icon: <Science />,
        path: '/professional/ai-lab',
        component: () => (
            <Grid container spacing={3}>
                <Grid item xs={12} md={8}>
                    <AIInsightsPanel
                        symbol="SPY"
                        signals={[]}
                        maxInsights={10}
                        enableLiveUpdates={true}
                        showFilters={true}
                        showPriorityBadges={true}
                        autoRefresh={true}
                        refreshInterval={30000}
                        height={600}
                    />
                </Grid>
                <Grid item xs={12} md={4}>
                    <VoiceSignalAssistant />
                </Grid>
            </Grid>
        ),
        description: 'AI-powered market intelligence'
    },
    {
        id: 'signals-list',
        label: 'Signal Explorer',
        icon: <AutoGraph />,
        path: '/professional/signals-list',
        component: () => (
            <SignalList
                signals={[]}
                onSignalClick={() => { }}
                height={700}
                enableVirtualization={true}
                showFilters={true}
                showSearch={true}
                groupBy="confidence"
                sortBy="timestamp"
            />
        ),
        description: 'Comprehensive signal analysis'
    },
    {
        id: 'agents',
        label: 'Agent Consensus',
        icon: <SmartToy />,
        path: '/professional/agents',
        component: () => (
            <Box sx={{ p: 3 }}>
                <AgentConsensusFlow />
            </Box>
        ),
        description: 'Multi-agent consensus view'
    },
];

interface UnifiedProfessionalAppProps {
    // Props can be added here for customization
}

const UnifiedProfessionalApp: React.FC<UnifiedProfessionalAppProps> = () => {
    const theme = useTheme();
    const location = useLocation();
    const navigate = useNavigate();
    const [drawerOpen, setDrawerOpen] = useState(false);
    const [selectedSymbol, setSelectedSymbol] = useState('SPY');
    const [aiWidgetVisible, setAiWidgetVisible] = useState(true);

    // Find current navigation item
    const currentItem = useMemo(() => {
        return navigationItems.find(item => location.pathname.startsWith(item.path)) || navigationItems[0];
    }, [location.pathname]);

    // Handle navigation
    const handleNavigation = useCallback((item: typeof navigationItems[0]) => {
        navigate(item.path);
        setDrawerOpen(false);
    }, [navigate]);

    // Handle search selection
    const handleSearchSelect = useCallback((item: any) => {
        if (item.category === 'symbols') {
            setSelectedSymbol(item.label);
        } else if (item.category === 'pages') {
            const navItem = navigationItems.find(nav => nav.id === item.id);
            if (navItem) {
                handleNavigation(navItem);
            }
        }
    }, [handleNavigation]);

    // Drawer toggle
    const toggleDrawer = useCallback(() => {
        setDrawerOpen(!drawerOpen);
    }, [drawerOpen]);

    return (
        <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
            {/* Professional App Bar */}
            <AppBar
                position="fixed"
                sx={{
                    zIndex: theme.zIndex.drawer + 1,
                    bgcolor: alpha('#0A0E1A', 0.95),
                    backdropFilter: 'blur(20px)',
                    borderBottom: `1px solid ${alpha('#FFD700', 0.1)}`,
                }}
            >
                <Toolbar>
                    <IconButton
                        color="inherit"
                        aria-label="open drawer"
                        onClick={toggleDrawer}
                        edge="start"
                        sx={{ mr: 2 }}
                    >
                        <MenuIcon />
                    </IconButton>

                    <Typography
                        variant="h6"
                        noWrap
                        component="div"
                        sx={{
                            flexGrow: 0,
                            mr: 3,
                            background: 'linear-gradient(135deg, #FFD700 0%, #FFA500 100%)',
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                            fontWeight: 700,
                        }}
                    >
                        GoldenSignals AI
                    </Typography>

                    <Chip
                        label={currentItem.label}
                        size="small"
                        sx={{
                            mr: 3,
                            bgcolor: alpha('#FFD700', 0.1),
                            color: '#FFD700',
                            border: `1px solid ${alpha('#FFD700', 0.3)}`,
                        }}
                    />

                    <Box sx={{ flexGrow: 1, mx: 3, maxWidth: 600 }}>
                        <UnifiedSearchBar
                            onSelect={handleSearchSelect}
                            placeholder="Search symbols, pages, or ask AI..."
                        />
                    </Box>

                    <Stack direction="row" spacing={1} alignItems="center">
                        <Tooltip title="Notifications">
                            <IconButton color="inherit">
                                <Badge badgeContent={3} color="error">
                                    <Notifications />
                                </Badge>
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Settings">
                            <IconButton color="inherit">
                                <Settings />
                            </IconButton>
                        </Tooltip>
                        <Avatar
                            sx={{
                                width: 32,
                                height: 32,
                                bgcolor: '#FFD700',
                                color: '#0A0E1A',
                                fontWeight: 600,
                            }}
                        >
                            AI
                        </Avatar>
                    </Stack>
                </Toolbar>
            </AppBar>

            {/* Navigation Drawer */}
            <Drawer
                variant="temporary"
                open={drawerOpen}
                onClose={toggleDrawer}
                sx={{
                    '& .MuiDrawer-paper': {
                        width: 300,
                        bgcolor: alpha('#131A2A', 0.95),
                        backdropFilter: 'blur(20px)',
                        borderRight: `1px solid ${alpha('#FFD700', 0.1)}`,
                        mt: '64px',
                    },
                }}
            >
                <Box sx={{ p: 2 }}>
                    <Typography
                        variant="subtitle2"
                        sx={{
                            color: 'text.secondary',
                            textTransform: 'uppercase',
                            letterSpacing: 1,
                            mb: 2,
                        }}
                    >
                        Professional Suite
                    </Typography>

                    <List>
                        {navigationItems.map((item) => (
                            <ListItemButton
                                key={item.id}
                                onClick={() => handleNavigation(item)}
                                selected={currentItem.id === item.id}
                                sx={{
                                    borderRadius: 2,
                                    mb: 1,
                                    '&.Mui-selected': {
                                        bgcolor: alpha('#FFD700', 0.1),
                                        borderLeft: `3px solid #FFD700`,
                                    },
                                    '&:hover': {
                                        bgcolor: alpha('#FFD700', 0.05),
                                    },
                                }}
                            >
                                <ListItemIcon
                                    sx={{
                                        color: currentItem.id === item.id ? '#FFD700' : 'text.secondary',
                                        minWidth: 40,
                                    }}
                                >
                                    {item.icon}
                                </ListItemIcon>
                                <ListItemText
                                    primary={item.label}
                                    secondary={item.description}
                                    primaryTypographyProps={{
                                        fontWeight: currentItem.id === item.id ? 600 : 400,
                                        color: currentItem.id === item.id ? '#FFD700' : 'text.primary',
                                    }}
                                    secondaryTypographyProps={{
                                        fontSize: '0.75rem',
                                        color: 'text.secondary',
                                    }}
                                />
                            </ListItemButton>
                        ))}
                    </List>

                    <Divider sx={{ my: 2, borderColor: alpha('#FFD700', 0.1) }} />

                    <Typography
                        variant="subtitle2"
                        sx={{
                            color: 'text.secondary',
                            textTransform: 'uppercase',
                            letterSpacing: 1,
                            mb: 2,
                        }}
                    >
                        Quick Stats
                    </Typography>

                    <Grid container spacing={1}>
                        <Grid item xs={6}>
                            <Card
                                sx={{
                                    bgcolor: alpha('#00D4AA', 0.1),
                                    border: `1px solid ${alpha('#00D4AA', 0.3)}`,
                                }}
                            >
                                <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
                                    <Typography variant="caption" color="text.secondary">
                                        Active Signals
                                    </Typography>
                                    <Typography variant="h6" sx={{ color: '#00D4AA', fontWeight: 600 }}>
                                        12
                                    </Typography>
                                </CardContent>
                            </Card>
                        </Grid>
                        <Grid item xs={6}>
                            <Card
                                sx={{
                                    bgcolor: alpha('#FFD700', 0.1),
                                    border: `1px solid ${alpha('#FFD700', 0.3)}`,
                                }}
                            >
                                <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
                                    <Typography variant="caption" color="text.secondary">
                                        AI Confidence
                                    </Typography>
                                    <Typography variant="h6" sx={{ color: '#FFD700', fontWeight: 600 }}>
                                        94%
                                    </Typography>
                                </CardContent>
                            </Card>
                        </Grid>
                    </Grid>
                </Box>
            </Drawer>

            {/* Main Content */}
            <Box
                component="main"
                sx={{
                    flexGrow: 1,
                    pt: '64px',
                    minHeight: '100vh',
                    bgcolor: 'background.default',
                }}
            >
                <Container maxWidth={false} sx={{ py: 3, px: 3 }}>
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={currentItem.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                        >
                            <Routes>
                                {navigationItems.map((item) => (
                                    <Route
                                        key={item.id}
                                        path={item.path.replace('/professional', '')}
                                        element={<item.component />}
                                    />
                                ))}
                                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                                <Route path="*" element={<Navigate to="/dashboard" replace />} />
                            </Routes>
                        </motion.div>
                    </AnimatePresence>
                </Container>
            </Box>

            {/* Golden Eye AI Prophet Widget */}
            {aiWidgetVisible && (
                <FloatingAIProphetWidget
                    onClick={() => setAiWidgetVisible(!aiWidgetVisible)}
                    isVisible={aiWidgetVisible}
                />
            )}
        </Box>
    );
};

export default UnifiedProfessionalApp; 