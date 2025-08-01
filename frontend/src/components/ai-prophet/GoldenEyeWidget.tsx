/**
 * Golden Eye AI Prophet Widget
 *
 * Simplified floating mystical orb with preserved animations.
 * The heart of the Golden Eye AI Prophet experience.
 */

import React, { useState, useEffect } from 'react';
import {
    Box,
    Fab,
    Typography,
    Zoom,
    Paper,
    alpha
} from '@mui/material';
import { RemoveRedEye, Lens } from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import {
    goldenEyeColors,
    goldenEyeAnimations,
    goldenEyeGradients,
    prophetMessages
} from '../../theme/goldenEye';
import { useAppStore } from '../../store/appStore';

// === STYLED COMPONENTS ===
const ProphetOrb = styled(Box)({
    position: 'fixed',
    bottom: 24,
    right: 24,
    zIndex: 1400,
});

const MysticalButton = styled(Fab)({
    width: 70,
    height: 70,
    background: goldenEyeGradients.mystical,
    backdropFilter: 'blur(10px)',
    border: `2px solid ${alpha(goldenEyeColors.primary, 0.3)}`,
    animation: `${goldenEyeAnimations.etherealGlow} 3s ease-in-out infinite, ${goldenEyeAnimations.levitate} 4s ease-in-out infinite`,
    transition: 'all 0.3s ease',
    overflow: 'visible',
    '&:hover': {
        transform: 'scale(1.1)',
        background: goldenEyeGradients.mystical,
    },
    '&::before': {
        content: '""',
        position: 'absolute',
        width: '100%',
        height: '100%',
        borderRadius: '50%',
        background: `radial-gradient(circle, ${alpha(goldenEyeColors.primary, 0.2)}, transparent 70%)`,
        animation: `${goldenEyeAnimations.orbPulse} 2s ease-in-out infinite`,
    },
});

const ProphetEye = styled(RemoveRedEye)({
    fontSize: 32,
    color: '#FFFFFF',
    filter: 'drop-shadow(0 0 8px rgba(255, 255, 255, 0.8))',
    '& path': {
        fill: 'url(#eyeGradient)',
    },
});

const MessageBubble = styled(Paper)(({ theme }) => ({
    position: 'absolute',
    bottom: 80,
    right: 0,
    padding: theme.spacing(2),
    minWidth: 220,
    maxWidth: 280,
    background: theme.palette.background.paper,
    border: `1px solid ${alpha(goldenEyeColors.primary, 0.3)}`,
    boxShadow: `0 4px 20px ${alpha(goldenEyeColors.primary, 0.2)}`,
    borderRadius: 12,
    '&::after': {
        content: '""',
        position: 'absolute',
        bottom: -8,
        right: 30,
        width: 16,
        height: 16,
        background: theme.palette.background.paper,
        border: `1px solid ${alpha(goldenEyeColors.primary, 0.3)}`,
        borderTop: 'none',
        borderLeft: 'none',
        transform: 'rotate(45deg)',
    },
}));

const OuterRing = styled(Box)({
    position: 'absolute',
    width: 90,
    height: 90,
    borderRadius: '50%',
    border: `2px solid ${alpha(goldenEyeColors.primary, 0.2)}`,
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    pointerEvents: 'none',
    animation: `${goldenEyeAnimations.orbPulse} 3s ease-in-out infinite`,
    animationDelay: '0.5s',
});

// === COMPONENT ===
interface GoldenEyeWidgetProps {
    isVisible?: boolean;
}

export const GoldenEyeWidget: React.FC<GoldenEyeWidgetProps> = ({
    isVisible = true
}) => {
    const [isHovered, setIsHovered] = useState(false);
    const [messageIndex, setMessageIndex] = useState(0);
    const toggleAIChat = useAppStore(state => state.toggleAIChat);

    // Rotate prophet messages
    useEffect(() => {
        const messageInterval = setInterval(() => {
            setMessageIndex((prev) => (prev + 1) % prophetMessages.length);
        }, 5000);

        return () => clearInterval(messageInterval);
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
                    <Typography
                        variant="body2"
                        sx={{
                            fontWeight: 500,
                            color: 'text.primary',
                            textAlign: 'center',
                            fontStyle: 'italic',
                        }}
                    >
                        {prophetMessages[messageIndex]}
                    </Typography>
                    <Typography
                        variant="caption"
                        sx={{
                            display: 'block',
                            textAlign: 'center',
                            mt: 1,
                            color: goldenEyeColors.primary,
                            fontWeight: 600,
                        }}
                    >
                        Golden Eye AI Prophet
                    </Typography>
                </MessageBubble>
            </Zoom>

            {/* Main orb button */}
            <MysticalButton
                onClick={toggleAIChat}
                color="primary"
                aria-label="Open Golden Eye AI Prophet"
            >
                {/* SVG gradient definition */}
                <svg width="0" height="0" style={{ position: 'absolute' }}>
                    <defs>
                        <linearGradient id="eyeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stopColor="#FFFFFF" />
                            <stop offset="100%" stopColor={goldenEyeColors.primary} />
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
                        animation: `${goldenEyeAnimations.orbPulse} 1.5s ease-in-out infinite`,
                    }} />
                </Box>
            </MysticalButton>

            {/* Outer ring effect */}
            <OuterRing />
        </ProphetOrb>
    );
};

export default GoldenEyeWidget;
