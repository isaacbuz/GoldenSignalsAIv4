/**
 * AI Chat Button Component
 * 
 * A floating action button that opens the AI Trading Assistant chat
 */

import React, { useState } from 'react';
import {
    Fab,
    Dialog,
    DialogContent,
    Badge,
    Zoom,
    useTheme,
    alpha,
    Box,
} from '@mui/material';
import {
    SmartToy,
    Close,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { AIChat } from './AIChat';

interface AIChatButtonProps {
    onClick?: () => void;
    initialContext?: {
        symbol?: string;
        timeframe?: string;
    };
}

const AIChatButton: React.FC<AIChatButtonProps> = ({ onClick, initialContext }) => {
    const theme = useTheme();
    const [open, setOpen] = useState(false);
    const [hasNewMessage, setHasNewMessage] = useState(false);

    const handleOpen = () => {
        if (onClick) {
            onClick();
        } else {
            setOpen(true);
            setHasNewMessage(false);
        }
    };

    const handleClose = () => {
        setOpen(false);
    };

    return (
        <>
            {/* Floating Action Button */}
            <Zoom in={!open} timeout={300}>
                <Fab
                    color="primary"
                    aria-label="AI Assistant"
                    onClick={handleOpen}
                    sx={{
                        position: 'fixed',
                        bottom: 24,
                        right: 24,
                        width: 64,
                        height: 64,
                        background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main} 100%)`,
                        boxShadow: `0 4px 20px ${alpha(theme.palette.primary.main, 0.4)}`,
                        '&:hover': {
                            transform: 'scale(1.1)',
                            boxShadow: `0 6px 30px ${alpha(theme.palette.primary.main, 0.6)}`,
                        },
                        transition: 'all 0.3s ease',
                    }}
                >
                    <Badge
                        color="error"
                        variant="dot"
                        invisible={!hasNewMessage}
                        sx={{
                            '& .MuiBadge-dot': {
                                width: 12,
                                height: 12,
                                border: '2px solid white',
                            },
                        }}
                    >
                        <SmartToy sx={{ fontSize: 32 }} />
                    </Badge>
                </Fab>
            </Zoom>

            {/* Chat Dialog */}
            <Dialog
                open={open}
                onClose={handleClose}
                maxWidth="md"
                fullWidth
                PaperProps={{
                    sx: {
                        height: '80vh',
                        maxHeight: 800,
                        borderRadius: 3,
                        overflow: 'hidden',
                    },
                }}
                TransitionComponent={Zoom}
                TransitionProps={{
                    timeout: 400,
                }}
            >
                <DialogContent sx={{ p: 0, height: '100%' }}>
                    <AIChat
                        open={true}
                        onClose={handleClose}
                        initialQuery={initialContext?.symbol ? `Analyze ${initialContext.symbol}` : undefined}
                    />
                </DialogContent>
            </Dialog>

            {/* Pulse animation when closed */}
            {!open && (
                <Box
                    sx={{
                        position: 'fixed',
                        bottom: 24,
                        right: 24,
                        width: 64,
                        height: 64,
                        borderRadius: '50%',
                        pointerEvents: 'none',
                    }}
                >
                    <motion.div
                        animate={{
                            scale: [1, 1.5, 1.5, 1, 1],
                            opacity: [1, 0.5, 0.5, 0.5, 0],
                        }}
                        transition={{
                            duration: 2,
                            repeat: Infinity,
                            repeatDelay: 3,
                        }}
                        style={{
                            width: '100%',
                            height: '100%',
                            borderRadius: '50%',
                            border: `2px solid ${theme.palette.primary.main}`,
                        }}
                    />
                </Box>
            )}
        </>
    );
};

export default AIChatButton; 