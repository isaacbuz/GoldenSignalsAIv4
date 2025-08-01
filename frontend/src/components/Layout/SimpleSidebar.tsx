import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
    Drawer,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Box,
    Typography,
    useTheme,
} from '@mui/material';
import {
    Dashboard,
    AutoAwesome,
    Analytics,
    Settings,
} from '@mui/icons-material';

interface SimpleSidebarProps {
    collapsed: boolean;
    onToggle: () => void;
    width: number;
}

export const SimpleSidebar: React.FC<SimpleSidebarProps> = ({
    collapsed,
    onToggle,
    width,
}) => {
    const theme = useTheme();
    const location = useLocation();
    const navigate = useNavigate();

    const navigationItems = [
        {
            id: 'dashboard',
            label: 'Dashboard',
            path: '/dashboard',
            icon: Dashboard,
        },
        {
            id: 'signals',
            label: 'Signals',
            path: '/signals',
            icon: AutoAwesome,
        },
        {
            id: 'analytics',
            label: 'Analytics',
            path: '/analytics',
            icon: Analytics,
        },
        {
            id: 'settings',
            label: 'Settings',
            path: '/settings',
            icon: Settings,
        },
    ];

    const isActive = (path: string) => location.pathname === path;

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
                },
            }}
        >
            <Box sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}` }}>
                <Typography variant="h6" sx={{ color: theme.palette.text.primary }}>
                    Navigation
                </Typography>
            </Box>

            <List sx={{ p: 1 }}>
                {navigationItems.map((item) => {
                    const active = isActive(item.path);
                    const Icon = item.icon;

                    return (
                        <ListItem key={item.id} disablePadding>
                            <ListItemButton
                                onClick={() => navigate(item.path)}
                                sx={{
                                    borderRadius: 1,
                                    mb: 0.5,
                                    bgcolor: active ? theme.palette.primary.main : 'transparent',
                                    color: active ? 'white' : theme.palette.text.primary,
                                    '&:hover': {
                                        bgcolor: active ? theme.palette.primary.dark : theme.palette.action.hover,
                                    },
                                }}
                            >
                                <ListItemIcon sx={{ color: 'inherit' }}>
                                    <Icon />
                                </ListItemIcon>
                                {!collapsed && <ListItemText primary={item.label} />}
                            </ListItemButton>
                        </ListItem>
                    );
                })}
            </List>
        </Drawer>
    );
};
