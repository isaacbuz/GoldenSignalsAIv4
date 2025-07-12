/**
 * Floating AI Prophet Widget
 * 
 * A mystical and engaging AI assistant interface inspired by fortune-telling and prophecy
 * Features:
 * - Orb-like design with ethereal glow
 * - Mystical animations and particle effects
 * - Expanding tooltip with prophet message
 * - Smooth open/close transitions
 * - Floating crystal ball aesthetic
 */

import React, { useState, useEffect } from 'react';
import {
    Box,
    Fab,
    Typography,
    Zoom,
    Fade,
    Paper,
    useTheme,
    alpha,
    keyframes,
    IconButton,
    Collapse,
} from '@mui/material';
import {
    AutoAwesome,
    Psychology,
    Close,
    RemoveRedEye,
    Lens,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

// Mystical animations
const etherealGlow = keyframes`
  0% {
    box-shadow: 
      0 0 20px ${alpha('#FFD700', 0.4)},
      0 0 40px ${alpha('#FFD700', 0.2)},
      0 0 60px ${alpha('#FF00FF', 0.1)};
  }
  50% {
    box-shadow: 
      0 0 30px ${alpha('#FFD700', 0.6)},
      0 0 50px ${alpha('#FFD700', 0.3)},
      0 0 80px ${alpha('#FF00FF', 0.2)};
  }
  100% {
    box-shadow: 
      0 0 20px ${alpha('#FFD700', 0.4)},
      0 0 40px ${alpha('#FFD700', 0.2)},
      0 0 60px ${alpha('#FF00FF', 0.1)};
  }
`;

const levitate = keyframes`
  0% { transform: translateY(0px) rotate(0deg); }
  33% { transform: translateY(-8px) rotate(1deg); }
  66% { transform: translateY(-5px) rotate(-1deg); }
  100% { transform: translateY(0px) rotate(0deg); }
`;



const orbPulse = keyframes`
  0%, 100% { transform: scale(1); opacity: 0.8; }
  50% { transform: scale(1.1); opacity: 1; }
`;

// Styled components
const ProphetOrb = styled(Box)(({ theme }) => ({
    position: 'fixed',
    bottom: theme.spacing(4),
    right: theme.spacing(4),
    zIndex: 1400,
}));

const MysticalButton = styled(Fab)(({ theme }) => ({
    width: 70,
    height: 70,
    background: `radial-gradient(circle at 30% 30%, ${alpha('#FFD700', 0.9)}, ${alpha('#FF6B00', 0.8)}, ${alpha('#8B00FF', 0.7)})`,
    backdropFilter: 'blur(10px)',
    border: `2px solid ${alpha('#FFD700', 0.3)}`,
    animation: `${etherealGlow} 3s ease-in-out infinite, ${levitate} 4s ease-in-out infinite`,
    transition: 'all 0.3s ease',
    overflow: 'visible',
    '&:hover': {
        transform: 'scale(1.1)',
        background: `radial-gradient(circle at 30% 30%, ${alpha('#FFD700', 1)}, ${alpha('#FF6B00', 0.9)}, ${alpha('#8B00FF', 0.8)})`,
    },
    '&::before': {
        content: '""',
        position: 'absolute',
        width: '100%',
        height: '100%',
        borderRadius: '50%',
        background: `radial-gradient(circle, ${alpha('#FFD700', 0.2)}, transparent 70%)`,
        animation: `${orbPulse} 2s ease-in-out infinite`,
    },
}));

const ProphetEye = styled(RemoveRedEye)(({ theme }) => ({
    fontSize: 32,
    color: '#FFFFFF',
    filter: 'drop-shadow(0 0 8px rgba(255, 255, 255, 0.8))',
    '& path': {
        fill: 'url(#eyeGradient)',
    },
}));

const MessageBubble = styled(Paper)(({ theme }) => ({
    position: 'absolute',
    bottom: 80,
    right: 0,
    padding: theme.spacing(2),
    minWidth: 220,
    maxWidth: 280,
    background: theme.palette.background.paper,
    // Removed transparent background for better visibility
    border: `1px solid ${alpha('#FFD700', 0.3)}`,
    boxShadow: `0 4px 20px ${alpha('#FFD700', 0.2)}`,
    '&::after': {
        content: '""',
        position: 'absolute',
        bottom: -8,
        right: 30,
        width: 16,
        height: 16,
        background: theme.palette.background.paper,
        border: `1px solid ${alpha('#FFD700', 0.3)}`,
        borderTop: 'none',
        borderLeft: 'none',
        transform: 'rotate(45deg)',
    },
}));



interface FloatingAIProphetWidgetProps {
    onClick: () => void;
    isVisible?: boolean;
}

const PROPHET_MESSAGES = [
    "The markets speak to those who listen...",
    "I sense opportunity in your future...",
    "The patterns reveal hidden truths...",
    "Ask, and the oracle shall answer...",
    "Your financial destiny awaits...",
];

export const FloatingAIProphetWidget: React.FC<FloatingAIProphetWidgetProps> = ({ onClick, isVisible = true }) => {
    const theme = useTheme();
    const [isHovered, setIsHovered] = useState(false);
    const [messageIndex, setMessageIndex] = useState(0);

    useEffect(() => {
        const messageInterval = setInterval(() => {
            setMessageIndex((prev) => (prev + 1) % PROPHET_MESSAGES.length);
        }, 5000);

        return () => {
            clearInterval(messageInterval);
        };
    }, []);

    if (!isVisible) return null;

    return (
        <ProphetOrb
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            {/* Prophet message */}
            <Zoom in={isHovered} timeout={300}>
                <MessageBubble elevation={8}>
                    <Typography variant="body2" sx={{
                        fontWeight: 500,
                        color: theme.palette.text.primary,
                        textAlign: 'center',
                        fontStyle: 'italic',
                    }}>
                        {PROPHET_MESSAGES[messageIndex]}
                    </Typography>
                    <Typography variant="caption" sx={{
                        display: 'block',
                        textAlign: 'center',
                        mt: 1,
                        color: alpha(theme.palette.primary.main, 0.8),
                        fontWeight: 600,
                    }}>
                        AI Prophet
                    </Typography>
                </MessageBubble>
            </Zoom>

            {/* Main orb button */}
            <MysticalButton
                onClick={onClick}
                color="primary"
                aria-label="Open AI Prophet"
            >
                {/* SVG gradient definition */}
                <svg width="0" height="0" style={{ position: 'absolute' }}>
                    <defs>
                        <linearGradient id="eyeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stopColor="#FFFFFF" />
                            <stop offset="100%" stopColor="#FFD700" />
                        </linearGradient>
                    </defs>
                </svg>

                {/* Animated eye icon */}
                <Box sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    position: 'relative',
                }}>
                    <ProphetEye />
                    {/* Inner glow */}
                    <Lens sx={{
                        position: 'absolute',
                        fontSize: 8,
                        color: '#FFFFFF',
                        animation: `${orbPulse} 1.5s ease-in-out infinite`,
                    }} />
                </Box>
            </MysticalButton>

            {/* Outer ring effect */}
            <Box
                sx={{
                    position: 'absolute',
                    width: 90,
                    height: 90,
                    borderRadius: '50%',
                    border: `2px solid ${alpha('#FFD700', 0.2)}`,
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    pointerEvents: 'none',
                    animation: `${orbPulse} 3s ease-in-out infinite`,
                    animationDelay: '0.5s',
                }}
            />
        </ProphetOrb>
    );
}; 