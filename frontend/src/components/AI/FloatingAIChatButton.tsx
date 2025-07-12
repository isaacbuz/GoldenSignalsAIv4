/**
 * Floating AI Chat Button Component
 * 
 * A modern floating action button that toggles the AI chat assistant
 * Features:
 * - Fixed position bottom-right
 * - Smooth animations
 * - Gold accent glow effect
 * - Notification badge for new messages
 * - Minimized state indicator
 */

import React, { useState, useEffect } from 'react';
import {
    Box,
    Fab,
    Badge,
    Zoom,
    Tooltip,
    IconButton,
    useTheme,
    alpha,
    keyframes,
} from '@mui/material';
import {
    AutoAwesome,
    Chat,
    Close,
    SmartToy,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

// Animations
const pulse = keyframes`
  0% {
    box-shadow: 0 0 0 0 rgba(255, 215, 0, 0.7);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(255, 215, 0, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(255, 215, 0, 0);
  }
`;

const float = keyframes`
  0% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
  100% { transform: translateY(0px); }
`;

const sparkle = keyframes`
  0% { opacity: 0; transform: scale(0) rotate(0deg); }
  50% { opacity: 1; transform: scale(1) rotate(180deg); }
  100% { opacity: 0; transform: scale(0) rotate(360deg); }
`;

// Styled components
const FloatingButton = styled(Fab)(({ theme }) => ({
    position: 'fixed',
    bottom: theme.spacing(3),
    right: theme.spacing(3),
    background: `linear-gradient(135deg, #FFD700 0%, #FFA000 100%)`,
    color: '#0a0a0a',
    width: 64,
    height: 64,
    boxShadow: '0 4px 20px rgba(255, 215, 0, 0.4)',
    animation: `${float} 3s ease-in-out infinite`,
    transition: 'all 0.3s ease',
    '&:hover': {
        background: `linear-gradient(135deg, #FFD700 0%, #FF8C00 100%)`,
        boxShadow: '0 6px 30px rgba(255, 215, 0, 0.6)',
        animation: `${pulse} 1.5s infinite`,
    },
    '&.MuiFab-extended': {
        paddingLeft: theme.spacing(3),
        paddingRight: theme.spacing(3),
        animation: 'none',
    },
}));

const SparkleEffect = styled(Box)({
    position: 'absolute',
    width: '100%',
    height: '100%',
    pointerEvents: 'none',
    '& .sparkle': {
        position: 'absolute',
        width: 4,
        height: 4,
        backgroundColor: '#FFD700',
        borderRadius: '50%',
        animation: `${sparkle} 1.5s linear infinite`,
    },
});

const NotificationDot = styled(Box)(({ theme }) => ({
    position: 'absolute',
    top: 8,
    right: 8,
    width: 12,
    height: 12,
    backgroundColor: '#FF4444',
    borderRadius: '50%',
    border: `2px solid ${theme.palette.background.paper}`,
    animation: `${pulse} 2s infinite`,
}));

interface FloatingAIChatButtonProps {
    onToggle: () => void;
    isOpen?: boolean;
    isMinimized?: boolean;
    hasNewMessages?: boolean;
    messageCount?: number;
}

export const FloatingAIChatButton: React.FC<FloatingAIChatButtonProps> = ({
    onToggle,
    isOpen = false,
    hasNewMessages = false,
    messageCount = 0,
}) => {
    const theme = useTheme();
    const [showSparkles, setShowSparkles] = useState(false);
    const [isHovered, setIsHovered] = useState(false);

    useEffect(() => {
        // Show sparkles periodically
        const interval = setInterval(() => {
            setShowSparkles(true);
            setTimeout(() => setShowSparkles(false), 1500);
        }, 5000);

        return () => clearInterval(interval);
    }, []);

    const generateSparkles = () => {
        const sparkles = [];
        for (let i = 0; i < 6; i++) {
            const delay = i * 0.2;
            const angle = (i * 60) * Math.PI / 180;
            const distance = 30;
            const x = Math.cos(angle) * distance + 50;
            const y = Math.sin(angle) * distance + 50;

            sparkles.push(
                <Box
                    key={i}
                    className="sparkle"
                    sx={{
                        left: `${x}%`,
                        top: `${y}%`,
                        animationDelay: `${delay}s`,
                    }}
                />
            );
        }
        return sparkles;
    };

    return (
        <Box
            sx={{
                position: 'fixed',
                bottom: 24,
                right: 24,
                zIndex: 1300,
            }}
        >
            <Badge badgeContent={hasNewMessages ? messageCount : 0} color="error">
                <Fab
                    color="secondary"
                    onClick={onToggle}
                    sx={{
                        bgcolor: 'secondary.main',
                        '&:hover': {
                            bgcolor: 'secondary.dark',
                        },
                    }}
                >
                    {isOpen ? <Chat /> : <AutoAwesome />}
                </Fab>
            </Badge>
        </Box>
    );
}; 