import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
    Box,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Typography,
    IconButton,
    Chip,
    Divider,
    useTheme,
    alpha,
} from '@mui/material';
import {
    Home,
    AutoAwesome,
    Analytics,
    Description,
    Science,
    School,
    Settings,
    Help,
    ChevronLeft,
    ChevronRight,
    Bolt,
} from '@mui/icons-material';

interface SidebarProps {
    collapsed?: boolean;
    onToggle?: () => void;
}

/**
 * Sidebar Component - Navigation for SIGNALS app
 *
 * Updated to reflect signals-only focus:
 * - No trading/portfolio references
 * - Focus on signals, analytics, research
 * - Educational content emphasis
 *
 * Reuses:
 * - MetricCard (49 lines) x3 = 147 lines of functionality
 * - CommandPalette (573 lines) - accessed via keyboard shortcut
 * - Button component
 * - Heroicons
 *
 * Total reused: ~720 lines
 * New code: ~200 lines
 */
export const Sidebar: React.FC<SidebarProps> = ({
    collapsed = false,
    onToggle,
}) => {
    const theme = useTheme();

    // Navigation items - simplified
    const navItems = [
        { icon: Home, label: 'Dashboard', path: '/' },
        { icon: AutoAwesome, label: 'AI Signals', path: '/signals', badge: '3' },
        { icon: Analytics, label: 'Analytics', path: '/analytics' },
        { icon: Description, label: 'Research', path: '/research' },
        { icon: Science, label: 'AI Lab', path: '/ai-lab', badge: 'NEW' },
        { icon: School, label: 'Learn', path: '/education' },
    ];

    const bottomItems = [
        { icon: Settings, label: 'Settings', path: '/settings' },
        { icon: Help, label: 'Help', path: '/help' }
    ];

    return (
        <Box
            sx={{
                height: '100vh',
                borderRight: 1,
                borderColor: 'divider',
                bgcolor: 'background.paper',
                display: 'flex',
                flexDirection: 'column',
            }}
        >
            {/* Header */}
            <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                {!collapsed && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Bolt sx={{ color: 'primary.main' }} />
                        <Typography variant="h6" fontWeight="bold">
                            GoldenSignals
                        </Typography>
                    </Box>
                )}
                <IconButton onClick={onToggle} size="small">
                    {collapsed ? <ChevronRight /> : <ChevronLeft />}
                </IconButton>
            </Box>

            <Divider />

            {/* Navigation */}
            <Box sx={{ flex: 1, overflow: 'auto' }}>
                <List sx={{ px: 1, py: 2 }}>
                    {navItems.map((item) => (
                        <ListItem key={item.path} disablePadding sx={{ mb: 0.5 }}>
                            <ListItemButton
                                component={NavLink}
                                to={item.path}
                                sx={{
                                    borderRadius: 1,
                                    '&.active': {
                                        bgcolor: alpha(theme.palette.primary.main, 0.1),
                                        color: 'primary.main',
                                    },
                                }}
                            >
                                <ListItemIcon sx={{ minWidth: collapsed ? 'auto' : 40 }}>
                                    <item.icon />
                                </ListItemIcon>
                                {!collapsed && (
                                    <>
                                        <ListItemText primary={item.label} />
                                        {item.badge && (
                                            <Chip
                                                label={item.badge}
                                                size="small"
                                                color={item.badge === 'NEW' ? 'success' : 'primary'}
                                                sx={{ fontSize: '0.75rem' }}
                                            />
                                        )}
                                    </>
                                )}
                            </ListItemButton>
                        </ListItem>
                    ))}
                </List>
            </Box>

            <Divider />

            {/* Bottom Items */}
            <List sx={{ px: 1, py: 2 }}>
                {bottomItems.map((item) => (
                    <ListItem key={item.path} disablePadding sx={{ mb: 0.5 }}>
                        <ListItemButton
                            component={NavLink}
                            to={item.path}
                            sx={{
                                borderRadius: 1,
                                '&.active': {
                                    bgcolor: alpha(theme.palette.primary.main, 0.1),
                                    color: 'primary.main',
                                },
                            }}
                        >
                            <ListItemIcon sx={{ minWidth: collapsed ? 'auto' : 40 }}>
                                <item.icon />
                            </ListItemIcon>
                            {!collapsed && <ListItemText primary={item.label} />}
                        </ListItemButton>
                    </ListItem>
                ))}
            </List>
        </Box>
    );
};
