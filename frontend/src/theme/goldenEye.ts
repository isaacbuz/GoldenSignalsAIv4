/**
 * Golden Eye AI Prophet Theme
 *
 * Mystical theme for the AI Prophet experience.
 * Preserves the magical golden eye aesthetic.
 */

import { createTheme, alpha, keyframes } from '@mui/material/styles';

// === GOLDEN EYE COLORS ===
export const goldenEyeColors = {
    // Core golden colors
    primary: '#FFD700',        // Pure gold
    secondary: '#FF6B00',      // Orange accent
    mystical: '#8B00FF',       // Purple mystical

    // Background colors
    background: '#0A0E1A',     // Deep navy
    surface: '#131A2A',        // Midnight blue
    elevated: '#1E293B',       // Dark slate

    // State colors
    bullish: '#00D4AA',        // Mint green
    bearish: '#FF4757',        // Coral red
    neutral: '#94A3B8',        // Gray

    // Text colors
    textPrimary: '#E2E8F0',    // Light gray
    textSecondary: '#94A3B8',  // Medium gray
    textGolden: '#FFD700',     // Golden text
};

// === MYSTICAL ANIMATIONS ===
export const goldenEyeAnimations = {
    // Floating orb animations
    etherealGlow: keyframes`
    0% {
      box-shadow:
        0 0 20px ${alpha(goldenEyeColors.primary, 0.4)},
        0 0 40px ${alpha(goldenEyeColors.primary, 0.2)},
        0 0 60px ${alpha(goldenEyeColors.mystical, 0.1)};
    }
    50% {
      box-shadow:
        0 0 30px ${alpha(goldenEyeColors.primary, 0.6)},
        0 0 50px ${alpha(goldenEyeColors.primary, 0.3)},
        0 0 80px ${alpha(goldenEyeColors.mystical, 0.2)};
    }
    100% {
      box-shadow:
        0 0 20px ${alpha(goldenEyeColors.primary, 0.4)},
        0 0 40px ${alpha(goldenEyeColors.primary, 0.2)},
        0 0 60px ${alpha(goldenEyeColors.mystical, 0.1)};
    }
  `,

    levitate: keyframes`
    0% { transform: translateY(0px) rotate(0deg); }
    33% { transform: translateY(-8px) rotate(1deg); }
    66% { transform: translateY(-5px) rotate(-1deg); }
    100% { transform: translateY(0px) rotate(0deg); }
  `,

    orbPulse: keyframes`
    0%, 100% { transform: scale(1); opacity: 0.8; }
    50% { transform: scale(1.1); opacity: 1; }
  `,

    shimmer: keyframes`
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
  `,

    fadeInUp: keyframes`
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
  `,
};

// === GOLDEN EYE GRADIENTS ===
export const goldenEyeGradients = {
    primary: `linear-gradient(45deg, ${goldenEyeColors.primary} 30%, ${goldenEyeColors.secondary} 90%)`,
    mystical: `radial-gradient(circle at 30% 30%, ${alpha(goldenEyeColors.primary, 0.9)}, ${alpha(goldenEyeColors.secondary, 0.8)}, ${alpha(goldenEyeColors.mystical, 0.7)})`,
    surface: `linear-gradient(135deg, ${goldenEyeColors.surface} 0%, ${alpha(goldenEyeColors.primary, 0.02)} 100%)`,
    glassmorphism: `rgba(255, 215, 0, 0.1)`,
};

// === GOLDEN EYE THEME ===
export const goldenEyeTheme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: goldenEyeColors.primary,
            dark: '#B8860B',
            light: '#FFFF00',
            contrastText: goldenEyeColors.background,
        },
        secondary: {
            main: goldenEyeColors.secondary,
            dark: '#CC5500',
            light: '#FF8533',
            contrastText: goldenEyeColors.textPrimary,
        },
        background: {
            default: goldenEyeColors.background,
            paper: goldenEyeColors.surface,
        },
        text: {
            primary: goldenEyeColors.textPrimary,
            secondary: goldenEyeColors.textSecondary,
        },
        success: {
            main: goldenEyeColors.bullish,
        },
        error: {
            main: goldenEyeColors.bearish,
        },
        warning: {
            main: goldenEyeColors.secondary,
        },
        info: {
            main: goldenEyeColors.mystical,
        },
    },
    typography: {
        fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
        h1: {
            fontWeight: 700,
            fontSize: '2.5rem',
        },
        h2: {
            fontWeight: 600,
            fontSize: '2rem',
        },
        h3: {
            fontWeight: 600,
            fontSize: '1.5rem',
        },
        h4: {
            fontWeight: 600,
            fontSize: '1.25rem',
        },
        h5: {
            fontWeight: 600,
            fontSize: '1.125rem',
        },
        h6: {
            fontWeight: 600,
            fontSize: '1rem',
        },
    },
    shape: {
        borderRadius: 12,
    },
    components: {
        MuiCard: {
            styleOverrides: {
                root: {
                    backgroundColor: alpha(goldenEyeColors.surface, 0.8),
                    backdropFilter: 'blur(10px)',
                    border: `1px solid ${alpha(goldenEyeColors.primary, 0.2)}`,
                    transition: 'all 0.3s ease',
                    '&:hover': {
                        borderColor: goldenEyeColors.primary,
                        boxShadow: `0 4px 20px ${alpha(goldenEyeColors.primary, 0.1)}`,
                    },
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    textTransform: 'none',
                    fontWeight: 600,
                    borderRadius: 8,
                },
                contained: {
                    background: goldenEyeGradients.primary,
                    '&:hover': {
                        background: `linear-gradient(45deg, ${goldenEyeColors.secondary} 30%, ${goldenEyeColors.primary} 90%)`,
                    },
                },
            },
        },
        MuiChip: {
            styleOverrides: {
                root: {
                    fontWeight: 600,
                },
            },
        },
        MuiFab: {
            styleOverrides: {
                root: {
                    background: goldenEyeGradients.mystical,
                    '&:hover': {
                        background: goldenEyeGradients.mystical,
                        transform: 'scale(1.05)',
                    },
                },
            },
        },
    },
});

// === UTILITY FUNCTIONS ===
export const getSignalColor = (type: 'BUY' | 'SELL' | 'HOLD') => {
    switch (type) {
        case 'BUY':
            return goldenEyeColors.bullish;
        case 'SELL':
            return goldenEyeColors.bearish;
        case 'HOLD':
            return goldenEyeColors.neutral;
        default:
            return goldenEyeColors.neutral;
    }
};

export const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return goldenEyeColors.bullish;
    if (confidence >= 60) return goldenEyeColors.secondary;
    return goldenEyeColors.bearish;
};

export const getGlassmorphismStyle = (opacity = 0.1) => ({
    backgroundColor: alpha(goldenEyeColors.surface, 0.8),
    backdropFilter: 'blur(10px)',
    border: `1px solid ${alpha(goldenEyeColors.primary, opacity)}`,
});

// === PROPHET MESSAGES ===
export const prophetMessages = [
    "The markets speak to those who listen...",
    "I sense opportunity in your future...",
    "The patterns reveal hidden truths...",
    "Ask, and the oracle shall answer...",
    "Your financial destiny awaits...",
    "The golden signals illuminate the path...",
    "Wisdom flows through the digital realm...",
    "The eye sees what others cannot...",
];

export default goldenEyeTheme;
