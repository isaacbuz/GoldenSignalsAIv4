import React from 'react';
import {
    Box,
    Button,
    ButtonGroup,
    Chip,
    Stack,
    Typography,
    Card,
    CardContent,
    alpha,
    useTheme
} from '@mui/material';
import {
    Dashboard,
    Psychology,
    AutoGraph,
    Palette,
    NewReleases
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

const DesignSwitcher: React.FC = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const theme = useTheme();

    const designs = [
        {
            path: '/dashboard',
            name: 'Professional',
            description: 'Bloomberg Terminal Style',
            icon: <Dashboard />,
            color: theme.palette.primary.main,
            features: ['Advanced Charts', 'Real-time Data', 'Professional Layout']
        },
        {
            path: '/signals',
            name: 'Enhanced',
            description: 'Command Center Style',
            icon: <Psychology />,
            color: theme.palette.secondary.main,
            features: ['AI Prophet', 'File Upload', 'Market Intelligence']
        },
        {
            path: '/modern',
            name: 'Modern',
            description: 'Card-Based Design',
            icon: <Palette />,
            color: theme.palette.success.main,
            features: ['Card Layout', 'Glassmorphism', 'Smooth Animations'],
            isNew: true
        },
    ];

    const currentDesign = designs.find(design => design.path === location.pathname);

    return (
        <Card
            sx={{
                mb: 3,
                background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.05)} 0%, ${alpha(theme.palette.background.paper, 0.8)} 100%)`,
                backdropFilter: 'blur(10px)',
                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
            }}
        >
            <CardContent>
                <Stack direction="row" alignItems="center" spacing={2} mb={3}>
                    <AutoGraph sx={{ color: theme.palette.primary.main }} />
                    <Typography variant="h6" fontWeight="bold">
                        UI Design Showcase
                    </Typography>
                    <Chip
                        label="3 Designs Available"
                        size="small"
                        color="primary"
                        variant="outlined"
                    />
                </Stack>

                <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
                    {designs.map((design) => (
                        <Card
                            key={design.path}
                            sx={{
                                flex: 1,
                                cursor: 'pointer',
                                border: currentDesign?.path === design.path
                                    ? `2px solid ${design.color}`
                                    : `1px solid ${alpha(theme.palette.divider, 0.2)}`,
                                background: currentDesign?.path === design.path
                                    ? `linear-gradient(135deg, ${alpha(design.color, 0.1)} 0%, ${alpha(design.color, 0.05)} 100%)`
                                    : 'transparent',
                                transition: 'all 0.3s ease',
                                '&:hover': {
                                    transform: 'translateY(-2px)',
                                    boxShadow: `0 8px 16px ${alpha(design.color, 0.15)}`,
                                    border: `1px solid ${alpha(design.color, 0.3)}`,
                                }
                            }}
                            onClick={() => navigate(design.path)}
                        >
                            <CardContent sx={{ p: 2 }}>
                                <Stack direction="row" alignItems="center" spacing={2} mb={2}>
                                    <Box
                                        sx={{
                                            p: 1,
                                            borderRadius: 2,
                                            bgcolor: alpha(design.color, 0.1),
                                            color: design.color,
                                        }}
                                    >
                                        {design.icon}
                                    </Box>
                                    <Box>
                                        <Stack direction="row" alignItems="center" spacing={1}>
                                            <Typography variant="h6" fontWeight="bold">
                                                {design.name}
                                            </Typography>
                                            {design.isNew && (
                                                <Chip
                                                    label="NEW"
                                                    size="small"
                                                    color="success"
                                                    icon={<NewReleases />}
                                                />
                                            )}
                                        </Stack>
                                        <Typography variant="body2" color="text.secondary">
                                            {design.description}
                                        </Typography>
                                    </Box>
                                </Stack>

                                <Stack spacing={0.5}>
                                    {design.features.map((feature, index) => (
                                        <Typography
                                            key={index}
                                            variant="caption"
                                            color="text.secondary"
                                            sx={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                '&:before': {
                                                    content: '"•"',
                                                    color: design.color,
                                                    fontWeight: 'bold',
                                                    marginRight: 1,
                                                }
                                            }}
                                        >
                                            {feature}
                                        </Typography>
                                    ))}
                                </Stack>

                                {currentDesign?.path === design.path && (
                                    <Chip
                                        label="Current"
                                        size="small"
                                        sx={{
                                            mt: 2,
                                            bgcolor: design.color,
                                            color: 'white',
                                            fontWeight: 'bold',
                                        }}
                                    />
                                )}
                            </CardContent>
                        </Card>
                    ))}
                </Stack>

                <Box mt={3} p={2} sx={{
                    bgcolor: alpha(theme.palette.info.main, 0.1),
                    borderRadius: 2,
                    border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`,
                }}>
                    <Typography variant="body2" color="info.main" fontWeight="bold" mb={1}>
                        💡 Design Philosophy
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Each design showcases different approaches to financial UI/UX:
                        Professional focuses on data density, Enhanced emphasizes AI capabilities,
                        and Modern prioritizes visual appeal and user experience.
                    </Typography>
                </Box>
            </CardContent>
        </Card>
    );
};

export default DesignSwitcher; 