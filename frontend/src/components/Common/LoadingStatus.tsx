import React from 'react';
import {
    Box,
    CircularProgress,
    Typography,
    Alert,
    Paper,
    Stack,
    useTheme,
    alpha,
} from '@mui/material';
import { CheckCircle, Warning, Error } from '@mui/icons-material';

interface LoadingStatusProps {
    loading?: boolean;
    error?: string | null;
    success?: boolean;
    message?: string;
    variant?: 'minimal' | 'detailed';
}

export const LoadingStatus: React.FC<LoadingStatusProps> = ({
    loading = false,
    error = null,
    success = false,
    message = '',
    variant = 'minimal',
}) => {
    const theme = useTheme();

    if (!loading && !error && !success) {
        return null;
    }

    const getStatusColor = () => {
        if (error) return theme.palette.error.main;
        if (success) return theme.palette.success.main;
        return theme.palette.primary.main;
    };

    const getStatusIcon = () => {
        if (error) return <Error sx={{ color: theme.palette.error.main }} />;
        if (success) return <CheckCircle sx={{ color: theme.palette.success.main }} />;
        return <CircularProgress size={20} sx={{ color: theme.palette.primary.main }} />;
    };

    const getStatusMessage = () => {
        if (error) return error;
        if (success) return message || 'Success!';
        return message || 'Loading...';
    };

    if (variant === 'minimal') {
        return (
            <Box
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    p: 1,
                    borderRadius: 1,
                    bgcolor: alpha(getStatusColor(), 0.1),
                    border: `1px solid ${alpha(getStatusColor(), 0.2)}`,
                }}
            >
                {getStatusIcon()}
                <Typography variant="body2" sx={{ color: getStatusColor() }}>
                    {getStatusMessage()}
                </Typography>
            </Box>
        );
    }

    return (
        <Paper
            elevation={2}
            sx={{
                p: 3,
                borderRadius: 2,
                bgcolor: alpha(theme.palette.background.paper, 0.95),
                backdropFilter: 'blur(10px)',
                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
            }}
        >
            <Stack spacing={2} alignItems="center">
                {getStatusIcon()}
                <Typography variant="h6" sx={{ color: getStatusColor() }}>
                    {getStatusMessage()}
                </Typography>
                {error && (
                    <Alert severity="error" sx={{ width: '100%' }}>
                        {error}
                    </Alert>
                )}
                {success && message && (
                    <Alert severity="success" sx={{ width: '100%' }}>
                        {message}
                    </Alert>
                )}
            </Stack>
        </Paper>
    );
};
