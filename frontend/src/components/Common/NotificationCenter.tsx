import React, { useState } from 'react';
import {
    IconButton,
    Badge,
    Popover,
    Box,
    Typography,
    List,
    ListItem,
    ListItemText,
    ListItemIcon,
    Divider,
    Button,
    useTheme,
    alpha,
    Chip,
} from '@mui/material';
import {
    Notifications,
    NotificationsNone,
    CheckCircle,
    Error,
    Warning,
    Info,
    Clear,
    MarkEmailRead,
} from '@mui/icons-material';
import { useNotifications } from '../Notifications/NotificationProvider';
import { formatDistanceToNow } from 'date-fns';

/**
 * NotificationCenter - Placeholder component
 * 
 * This is a stub component to fix import errors.
 * In a real implementation, this would handle notifications.
 */
export const NotificationCenter: React.FC = () => {
    const theme = useTheme();
    const { notifications, removeNotification, clearAll } = useNotifications();
    const [anchorEl, setAnchorEl] = useState<HTMLButtonElement | null>(null);

    const unreadCount = notifications.filter(n => !n.read).length;

    const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
        setAnchorEl(event.currentTarget);
    };

    const handleClose = () => {
        setAnchorEl(null);
    };

    const open = Boolean(anchorEl);
    const id = open ? 'notification-popover' : undefined;

    const getIcon = (type: string) => {
        switch (type) {
            case 'success':
                return <CheckCircle sx={{ color: theme.palette.success.main }} />;
            case 'error':
                return <Error sx={{ color: theme.palette.error.main }} />;
            case 'warning':
                return <Warning sx={{ color: theme.palette.warning.main }} />;
            default:
                return <Info sx={{ color: theme.palette.info.main }} />;
        }
    };

    return (
        <>
            <IconButton
                aria-describedby={id}
                onClick={handleClick}
                color="inherit"
                sx={{
                    '&:hover': {
                        background: alpha(theme.palette.primary.main, 0.1),
                    },
                }}
            >
                <Badge badgeContent={unreadCount} color="error">
                    {open ? <Notifications /> : <NotificationsNone />}
                </Badge>
            </IconButton>

            <Popover
                id={id}
                open={open}
                anchorEl={anchorEl}
                onClose={handleClose}
                anchorOrigin={{
                    vertical: 'bottom',
                    horizontal: 'right',
                }}
                transformOrigin={{
                    vertical: 'top',
                    horizontal: 'right',
                }}
                PaperProps={{
                    sx: {
                        width: 400,
                        maxHeight: 500,
                        overflow: 'hidden',
                        display: 'flex',
                        flexDirection: 'column',
                    },
                }}
            >
                {/* Header */}
                <Box
                    sx={{
                        p: 2,
                        borderBottom: `1px solid ${theme.palette.divider}`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                    }}
                >
                    <Typography variant="h6">Notifications</Typography>
                    {notifications.length > 0 && (
                        <Button
                            size="small"
                            onClick={clearAll}
                            startIcon={<Clear />}
                        >
                            Clear All
                        </Button>
                    )}
                </Box>

                {/* Notification List */}
                <Box sx={{ flex: 1, overflow: 'auto' }}>
                    {notifications.length === 0 ? (
                        <Box
                            sx={{
                                p: 4,
                                textAlign: 'center',
                                color: theme.palette.text.secondary,
                            }}
                        >
                            <NotificationsNone sx={{ fontSize: 48, mb: 1, opacity: 0.5 }} />
                            <Typography variant="body2">No notifications</Typography>
                        </Box>
                    ) : (
                        <List sx={{ py: 0 }}>
                            {notifications.map((notification, index) => (
                                <React.Fragment key={notification.id}>
                                    {index > 0 && <Divider />}
                                    <ListItem
                                        sx={{
                                            py: 2,
                                            pr: 1,
                                            backgroundColor: !notification.read
                                                ? alpha(theme.palette.primary.main, 0.05)
                                                : 'transparent',
                                            '&:hover': {
                                                backgroundColor: alpha(theme.palette.action.hover, 0.05),
                                            },
                                        }}
                                    >
                                        <ListItemIcon sx={{ minWidth: 40 }}>
                                            {getIcon(notification.type)}
                                        </ListItemIcon>
                                        <ListItemText
                                            primary={
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <Typography variant="subtitle2">
                                                        {notification.title || 'Notification'}
                                                    </Typography>
                                                    {notification.data?.symbol && (
                                                        <Chip
                                                            label={notification.data.symbol}
                                                            size="small"
                                                            sx={{ height: 20 }}
                                                        />
                                                    )}
                                                </Box>
                                            }
                                            secondary={
                                                <>
                                                    <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                                                        {notification.message}
                                                    </Typography>
                                                    <Typography variant="caption" color="text.secondary">
                                                        {formatDistanceToNow(notification.timestamp, { addSuffix: true })}
                                                    </Typography>
                                                </>
                                            }
                                        />
                                        <IconButton
                                            size="small"
                                            onClick={() => removeNotification(notification.id)}
                                            sx={{ ml: 1 }}
                                        >
                                            <Clear fontSize="small" />
                                        </IconButton>
                                    </ListItem>
                                </React.Fragment>
                            ))}
                        </List>
                    )}
                </Box>

                {/* Footer */}
                {notifications.length > 0 && (
                    <Box
                        sx={{
                            p: 1,
                            borderTop: `1px solid ${theme.palette.divider}`,
                            textAlign: 'center',
                        }}
                    >
                        <Button
                            size="small"
                            fullWidth
                            startIcon={<MarkEmailRead />}
                        >
                            Mark All as Read
                        </Button>
                    </Box>
                )}
            </Popover>
        </>
    );
};

export default NotificationCenter; 