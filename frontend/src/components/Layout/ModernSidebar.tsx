import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
    Drawer,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    IconButton,
    Box,
    Typography,
    Chip,
    Divider,
    Tooltip,
    useTheme,
    alpha,
} from '@mui/material';
import {
    Dashboard,
    AutoAwesome,
    Analytics,
    Settings,
    ChevronLeft,
    ChevronRight,
    TrendingUp,
    Psychology,
    School,
    Help,
} from '@mui/icons-material';
// import { colors } from '../../theme/modernTradingTheme';

interface NavigationItem {
    id: string;
    label: string;
    path: string;
    icon: React.ElementType;
    badge?: string;
    badgeColor?: 'primary' | 'secondary' | 'success' | 'error' | 'warning';
}

interface ModernSidebarProps {
    collapsed: boolean;
    onToggle: () => void;
    width: number;
}

export const ModernSidebar: React.FC<ModernSidebarProps> = ({
    collapsed,
    onToggle,
    width,
}) => {
    const theme = useTheme();
    const location = useLocation();
    const navigate = useNavigate();

    // Navigation items - focused on signals and AI
    const navigationItems: NavigationItem[] = [
        {
            id: 'dashboard',
            label: 'Trading Dashboard',
            path: '/dashboard',
            icon: Dashboard,
        },
        {
            id: 'signals',
            label: 'AI Signals',
            path: '/signals',
            icon: AutoAwesome,
            badge: '24',
            badgeColor: 'success',
        },
        {
            id: 'analytics',
            label: 'Market Analytics',
            path: '/analytics',
            icon: Analytics,
        },
        {
            id: 'ai-lab',
            label: 'AI Laboratory',
            path: '/ai-lab',
            icon: Psychology,
            badge: 'NEW',
            badgeColor: 'secondary',
        },
        {
            id: 'education',
            label: 'Trading Education',
            path: '/education',
            icon: School,
        },
    ];

    const bottomItems: NavigationItem[] = [
        {
            id: 'settings',
            label: 'Settings',
            path: '/settings',
            icon: Settings,
        },
        {
            id: 'help',
            label: 'Help & Support',
            path: '/help',
            icon: Help,
        },
    ];

    const isActive = (path: string) => location.pathname === path;

    const handleNavigate = (path: string) => {
        navigate(path);
    };

    const NavItemContent: React.FC<{ item: NavigationItem }> = ({ item }) => {
        const active = isActive(item.path);
        const Icon = item.icon;

        const content = (
            <ListItemButton
                onClick={() => handleNavigate(item.path)}
                sx={{
                    mx: 1,
                    mb: 0.5,
                    borderRadius: 2,
                    minHeight: 48,
                    bgcolor: active ? theme.palette.primary.main : 'transparent',
                    color: active ? 'white' : theme.palette.text.secondary,
                    transition: 'all 0.2s ease',
                    '&:hover': {
                        bgcolor: active ? theme.palette.primary.main : alpha(theme.palette.primary.main, 0.1),
                        color: active ? 'white' : theme.palette.text.primary,
                        transform: 'translateX(4px)',
                    },
                }}
            >
                <ListItemIcon
                    sx={{
                        minWidth: collapsed ? 0 : 40,
                        justifyContent: 'center',
                        color: 'inherit',
                    }}
                >
                    <Icon sx={{ fontSize: 22 }} />
                </ListItemIcon>

                {!collapsed && (
                    <>
                        <ListItemText
                            primary={item.label}
                            primaryTypographyProps={{
                                fontSize: '0.875rem',
                                fontWeight: active ? 600 : 500,
                            }}
                            sx={{ my: 0 }}
                        />

                        {item.badge && (
                            <Chip
                                label={item.badge}
                                size="small"
                                color={item.badgeColor || 'primary'}
                                sx={{
                                    height: 20,
                                    fontSize: '0.6875rem',
                                    fontWeight: 600,
                                }}
                            />
                        )}
                    </>
                )}
            </ListItemButton>
        );

        if (collapsed) {
            return (
                <Tooltip title={item.label} placement="right" arrow>
                    {content}
                </Tooltip>
            );
        }

        return content;
    };

    return (
        <Drawer
            variant="permanent"
            sx={{
                width,
                flexShrink: 0,
                '& .MuiDrawer-paper': {
                    width,
                    boxSizing: 'border-box',
                    bgcolor: theme.palette.background.paper,
                    borderRight: `1px solid ${theme.palette.divider}`,
                    transition: 'width 0.3s ease',
                    overflow: 'hidden',
                },
            }}
        >
            {/* Header */}
            <Box
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: collapsed ? 'center' : 'space-between',
                    px: collapsed ? 1 : 2,
                    py: 2,
                    borderBottom: `1px solid ${theme.palette.divider}`,
                    minHeight: 72,
                }}
            >
                {!collapsed && (
                    <Box>
                        <Typography variant="h6" sx={{ color: theme.palette.text.primary, fontWeight: 600 }}>
                            Navigation
                        </Typography>
                        <Typography variant="caption" sx={{ color: theme.palette.text.secondary }}>
                            AI Trading Platform
                        </Typography>
                    </Box>
                )}

                <IconButton
                    onClick={onToggle}
                    sx={{
                        color: theme.palette.text.secondary,
                        bgcolor: theme.palette.background.default,
                        border: `1px solid ${theme.palette.divider}`,
                        width: 32,
                        height: 32,
                        '&:hover': {
                            bgcolor: alpha(theme.palette.primary.main, 0.1),
                            color: theme.palette.text.primary,
                        },
                    }}
                >
                    {collapsed ? <ChevronRight /> : <ChevronLeft />}
                </IconButton>
            </Box>

            {/* Main Navigation */}
            <Box sx={{ flex: 1, overflow: 'auto' }}>
                <List sx={{ px: 1, py: 2 }}>
                    {navigationItems.map((item) => (
                        <ListItem key={item.id} disablePadding>
                            <NavItemContent item={item} />
                        </ListItem>
                    ))}
                </List>

                {/* Performance Summary (when expanded) */}
                {!collapsed && (
                    <Box sx={{ mx: 2, my: 2 }}>
                        <Divider sx={{ borderColor: theme.palette.divider, mb: 2 }} />
                        <Typography variant="caption" sx={{ color: theme.palette.text.secondary, mb: 1, display: 'block' }}>
                            Today's Performance
                        </Typography>
                        <Box
                            sx={{
                                p: 2,
                                bgcolor: theme.palette.background.default,
                                borderRadius: 2,
                                border: `1px solid ${theme.palette.divider}`,
                            }}
                        >
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                <TrendingUp sx={{ color: theme.palette.success.main, fontSize: 16, mr: 1 }} />
                                <Typography variant="body2" sx={{ color: theme.palette.text.primary, fontWeight: 600 }}>
                                    87.3% Accuracy
                                </Typography>
                            </Box>
                            <Typography variant="caption" sx={{ color: theme.palette.text.secondary }}>
                                24 active signals
                            </Typography>
                        </Box>
                    </Box>
                )}
            </Box>

            {/* Bottom Navigation */}
            <Box sx={{ borderTop: `1px solid ${theme.palette.divider}` }}>
                <List sx={{ px: 1, py: 1 }}>
                    {bottomItems.map((item) => (
                        <ListItem key={item.id} disablePadding>
                            <NavItemContent item={item} />
                        </ListItem>
                    ))}
                </List>
            </Box>
        </Drawer>
    );
}; 