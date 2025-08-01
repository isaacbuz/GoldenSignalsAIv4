import React from 'react';
import { useLocation, Link as RouterLink } from 'react-router-dom';
import {
    Breadcrumbs as MuiBreadcrumbs,
    Link,
    Typography,
    Box,
    Chip,
    Stack,
} from '@mui/material';
import {
    ChevronRight,
    Home,
    Dashboard,
    Analytics,
    ShowChart,
    AccountBalance,
    Settings,
    TrendingUp,
} from '@mui/icons-material';

interface BreadcrumbItem {
    label: string;
    path: string;
    icon?: React.ReactNode;
    badge?: string | number;
}

const routeConfig: Record<string, BreadcrumbItem> = {
    '/': { label: 'Home', path: '/', icon: <Home fontSize="small" /> },
    '/dashboard': { label: 'Dashboard', path: '/dashboard', icon: <Dashboard fontSize="small" /> },
    '/signals': { label: 'Signals', path: '/signals', icon: <ShowChart fontSize="small" /> },
    '/signals-dashboard': { label: 'Signals Dashboard', path: '/signals-dashboard', icon: <TrendingUp fontSize="small" /> },
    '/analytics': { label: 'Analytics', path: '/analytics', icon: <Analytics fontSize="small" /> },
    '/portfolio': { label: 'Portfolio', path: '/portfolio', icon: <AccountBalance fontSize="small" /> },
    '/settings': { label: 'Settings', path: '/settings', icon: <Settings fontSize="small" /> },
};

export const Breadcrumbs: React.FC = () => {
    const location = useLocation();
    const pathnames = location.pathname.split('/').filter((x) => x);

    // Build breadcrumb items
    const breadcrumbItems: BreadcrumbItem[] = [routeConfig['/']];

    let currentPath = '';
    pathnames.forEach((segment) => {
        currentPath += `/${segment}`;
        const config = routeConfig[currentPath];
        if (config) {
            breadcrumbItems.push(config);
        } else {
            // Handle dynamic routes
            breadcrumbItems.push({
                label: segment.charAt(0).toUpperCase() + segment.slice(1).replace(/-/g, ' '),
                path: currentPath,
            });
        }
    });

    return (
        <Box
            sx={{
                py: 1,
                px: 2,
                backgroundColor: 'background.paper',
                borderBottom: 1,
                borderColor: 'divider',
            }}
        >
            <MuiBreadcrumbs
                separator={<ChevronRight fontSize="small" sx={{ color: 'text.secondary' }} />}
                aria-label="breadcrumb"
                sx={{
                    '& .MuiBreadcrumbs-ol': {
                        alignItems: 'center',
                    },
                }}
            >
                {breadcrumbItems.map((item, index) => {
                    const isLast = index === breadcrumbItems.length - 1;

                    return isLast ? (
                        <Stack key={item.path} direction="row" alignItems="center" spacing={1}>
                            {item.icon}
                            <Typography color="text.primary" fontWeight={600}>
                                {item.label}
                            </Typography>
                            {item.badge && (
                                <Chip
                                    label={item.badge}
                                    size="small"
                                    color="primary"
                                    sx={{ height: 20, fontSize: '0.75rem' }}
                                />
                            )}
                        </Stack>
                    ) : (
                        <Link
                            key={item.path}
                            component={RouterLink}
                            to={item.path}
                            underline="hover"
                            color="text.secondary"
                            sx={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 0.5,
                                transition: 'all 0.2s',
                                '&:hover': {
                                    color: 'primary.main',
                                    transform: 'translateY(-1px)',
                                },
                            }}
                        >
                            {item.icon}
                            {item.label}
                        </Link>
                    );
                })}
            </MuiBreadcrumbs>
        </Box>
    );
};
