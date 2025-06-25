import React, { useState } from 'react';
import {
    Box,
    Drawer,
    AppBar,
    Toolbar,
    Typography,
    IconButton,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    useTheme,
    useMediaQuery,
    Chip,
    Badge,
    Tooltip,
    LinearProgress,
} from '@mui/material';
import {
    Menu as MenuIcon,
    Dashboard as DashboardIcon,
    ShowChart as SignalIcon,
    SmartToy as AIIcon,
    Analytics as AnalyticsIcon,
    ModelTraining as ModelIcon,
    Psychology as IntelligenceIcon,
    History as HistoryIcon,
    Settings as SettingsIcon,
    AutoAwesome as AutoAwesomeIcon,
    Notifications as NotificationsIcon,
    WifiTethering as LiveIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { styled } from '@mui/material/styles';
import NotificationCenter from '../Notifications/NotificationCenter';

const drawerWidth = 260;

const StyledDrawer = styled(Drawer)(({ theme }) => ({
    '& .MuiDrawer-paper': {
        width: drawerWidth,
        backgroundColor: '#0A0E27',
        borderRight: '1px solid rgba(255, 215, 0, 0.1)',
        backgroundImage: 'linear-gradient(180deg, #0A0E27 0%, #0D1117 100%)',
    },
}));

const StyledAppBar = styled(AppBar)(({ theme }) => ({
    backgroundColor: '#0A0E27',
    borderBottom: '1px solid rgba(255, 215, 0, 0.1)',
    boxShadow: '0 2px 10px rgba(255, 215, 0, 0.1)',
}));

const LogoBox = styled(Box)({
    display: 'flex',
    alignItems: 'center',
    padding: '20px',
    borderBottom: '1px solid rgba(255, 215, 0, 0.1)',
    background: 'linear-gradient(45deg, #FFD700 30%, #FFA500 90%)',
    '-webkit-background-clip': 'text',
    '-webkit-text-fill-color': 'transparent',
});

const NavItem = styled(ListItem)(({ theme, selected }: { theme: any; selected: boolean }) => ({
    margin: '4px 12px',
    borderRadius: '12px',
    transition: 'all 0.3s ease',
    backgroundColor: selected ? 'rgba(255, 215, 0, 0.1)' : 'transparent',
    '&:hover': {
        backgroundColor: 'rgba(255, 215, 0, 0.05)',
        transform: 'translateX(4px)',
    },
    '& .MuiListItemIcon-root': {
        color: selected ? '#FFD700' : 'rgba(255, 255, 255, 0.7)',
    },
    '& .MuiListItemText-primary': {
        color: selected ? '#FFD700' : 'rgba(255, 255, 255, 0.9)',
        fontWeight: selected ? 600 : 400,
    },
}));

const StatusChip = styled(Chip)(({ theme }) => ({
    backgroundColor: 'rgba(76, 175, 80, 0.1)',
    color: '#4CAF50',
    border: '1px solid rgba(76, 175, 80, 0.3)',
    '& .MuiChip-icon': {
        color: '#4CAF50',
    },
}));

interface MainLayoutProps {
    children: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
    const [mobileOpen, setMobileOpen] = useState(false);
    const [aiProcessing, setAiProcessing] = useState(false);
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('md'));
    const navigate = useNavigate();
    const location = useLocation();

    const handleDrawerToggle = () => {
        setMobileOpen(!mobileOpen);
    };

    const menuItems = [
        {
            text: 'AI Command Center',
            icon: <DashboardIcon />,
            path: '/command-center',
            badge: 'LIVE',
            badgeColor: 'success',
        },
        {
            text: 'Signal Stream',
            icon: <SignalIcon />,
            path: '/signals',
            badge: '12',
            badgeColor: 'error',
        },
        {
            text: 'AI Assistant',
            icon: <AIIcon />,
            path: '/ai-assistant',
            badge: 'NEW',
            badgeColor: 'primary',
        },
        {
            text: 'Signal Analytics',
            icon: <AnalyticsIcon />,
            path: '/analytics',
        },
        {
            text: 'Model Dashboard',
            icon: <ModelIcon />,
            path: '/models',
        },
        {
            text: 'Market Intelligence',
            icon: <IntelligenceIcon />,
            path: '/intelligence',
        },
        {
            text: 'Signal History',
            icon: <HistoryIcon />,
            path: '/history',
        },
        {
            text: 'Settings',
            icon: <SettingsIcon />,
            path: '/settings',
        },
    ];

    const drawer = (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <LogoBox>
                <AutoAwesomeIcon sx={{ mr: 1, fontSize: 32, color: '#FFD700' }} />
                <Box>
                    <Typography variant="h5" sx={{ fontWeight: 'bold', color: '#FFD700' }}>
                        GoldenSignals
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#FFA500', opacity: 0.8 }}>
                        AI Signal Intelligence
                    </Typography>
                </Box>
            </LogoBox>

            <Box sx={{ p: 2, borderBottom: '1px solid rgba(255, 215, 0, 0.1)' }}>
                <StatusChip
                    icon={<LiveIcon sx={{ animation: 'pulse 2s infinite' }} />}
                    label="AI Models Active"
                    size="small"
                />
            </Box>

            <List sx={{ flex: 1, pt: 2 }}>
                {menuItems.map((item) => (
                    <NavItem
                        button
                        key={item.path}
                        selected={location.pathname === item.path}
                        onClick={() => navigate(item.path)}
                        theme={theme}
                    >
                        <ListItemIcon>
                            {item.badge ? (
                                <Badge badgeContent={item.badge} color={item.badgeColor as any}>
                                    {item.icon}
                                </Badge>
                            ) : (
                                item.icon
                            )}
                        </ListItemIcon>
                        <ListItemText primary={item.text} />
                    </NavItem>
                ))}
            </List>

            <Box sx={{ p: 2, borderTop: '1px solid rgba(255, 215, 0, 0.1)' }}>
                <Box sx={{ mb: 1, display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="caption" color="text.secondary">
                        Model Accuracy
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#4CAF50' }}>
                        94.2%
                    </Typography>
                </Box>
                <LinearProgress
                    variant="determinate"
                    value={94.2}
                    sx={{
                        height: 6,
                        borderRadius: 3,
                        backgroundColor: 'rgba(255, 255, 255, 0.1)',
                        '& .MuiLinearProgress-bar': {
                            backgroundColor: '#4CAF50',
                            borderRadius: 3,
                        },
                    }}
                />
            </Box>
        </Box>
    );

    return (
        <Box sx={{ display: 'flex', height: '100vh', backgroundColor: '#0D1117' }}>
            <StyledAppBar
                position="fixed"
                sx={{
                    width: { md: `calc(100% - ${drawerWidth}px)` },
                    ml: { md: `${drawerWidth}px` },
                }}
            >
                <Toolbar>
                    <IconButton
                        color="inherit"
                        edge="start"
                        onClick={handleDrawerToggle}
                        sx={{ mr: 2, display: { md: 'none' } }}
                    >
                        <MenuIcon />
                    </IconButton>

                    <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center' }}>
                        <Typography variant="h6" noWrap sx={{ fontWeight: 600 }}>
                            {menuItems.find(item => item.path === location.pathname)?.text || 'AI Command Center'}
                        </Typography>
                        {aiProcessing && (
                            <Chip
                                label="AI Processing..."
                                size="small"
                                sx={{ ml: 2, backgroundColor: 'rgba(255, 215, 0, 0.1)', color: '#FFD700' }}
                                icon={<AutoAwesomeIcon sx={{ animation: 'spin 2s linear infinite' }} />}
                            />
                        )}
                    </Box>

                    <Box sx={{ display: 'flex', gap: 1 }}>
                        <NotificationCenter />
                    </Box>
                </Toolbar>
                {aiProcessing && (
                    <LinearProgress
                        sx={{
                            position: 'absolute',
                            bottom: 0,
                            left: 0,
                            right: 0,
                            height: 2,
                            '& .MuiLinearProgress-bar': {
                                backgroundColor: '#FFD700',
                            },
                        }}
                    />
                )}
            </StyledAppBar>

            <Box
                component="nav"
                sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
            >
                <StyledDrawer
                    variant={isMobile ? 'temporary' : 'permanent'}
                    anchor="left"
                    open={isMobile ? mobileOpen : true}
                    onClose={handleDrawerToggle}
                    ModalProps={{
                        keepMounted: true, // Better open performance on mobile.
                    }}
                >
                    {drawer}
                </StyledDrawer>
            </Box>

            <Box
                component="main"
                sx={{
                    flexGrow: 1,
                    p: 3,
                    width: { md: `calc(100% - ${drawerWidth}px)` },
                    mt: '64px',
                    backgroundColor: '#0D1117',
                    minHeight: 'calc(100vh - 64px)',
                    overflowY: 'auto',
                }}
            >
                {children}
            </Box>
        </Box>
    );
};

export default MainLayout; 