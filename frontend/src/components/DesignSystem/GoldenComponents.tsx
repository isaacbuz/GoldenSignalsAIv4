import React from 'react';
import {
  Button,
  ButtonProps,
  styled,
  alpha,
  keyframes,
  useTheme
} from '@mui/material';

// Define custom variants that extend the base variants
type CustomVariant = 'glow' | 'gradient';
type GoldenButtonVariant = ButtonProps['variant'] | CustomVariant;

interface GoldenButtonProps extends Omit<ButtonProps, 'variant'> {
  variant?: GoldenButtonVariant;
  glowColor?: string;
  gradientFrom?: string;
  gradientTo?: string;
}

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

const StyledButton = styled(Button)<GoldenButtonProps>(({ theme, variant, glowColor, gradientFrom, gradientTo }) => {
  const baseStyles = {
    fontWeight: 600,
    textTransform: 'none' as const,
    borderRadius: theme.spacing(1),
    transition: 'all 0.3s ease',
    position: 'relative' as const,
    overflow: 'hidden' as const,

    '&::before': {
      content: '""',
      position: 'absolute' as const,
      top: 0,
      left: '-100%',
      width: '100%',
      height: '100%',
      background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent)',
      transition: 'left 0.5s',
    },

    '&:hover::before': {
      left: '100%',
    },
  };

  if ((variant as string) === 'glow') {
    return {
      ...baseStyles,
      background: `linear-gradient(45deg, ${glowColor || theme.palette.primary.main}, ${alpha(glowColor || theme.palette.primary.main, 0.8)})`,
      color: theme.palette.primary.contrastText,
      boxShadow: `0 0 20px ${alpha(glowColor || theme.palette.primary.main, 0.5)}`,
      animation: `${pulse} 2s infinite`,

      '&:hover': {
        boxShadow: `0 0 30px ${alpha(glowColor || theme.palette.primary.main, 0.8)}`,
        transform: 'translateY(-2px)',
      },
    };
  }

  if ((variant as string) === 'gradient') {
    return {
      ...baseStyles,
      background: `linear-gradient(45deg, ${gradientFrom || theme.palette.primary.main}, ${gradientTo || theme.palette.secondary.main})`,
      color: theme.palette.primary.contrastText,

      '&:hover': {
        background: `linear-gradient(45deg, ${gradientTo || theme.palette.secondary.main}, ${gradientFrom || theme.palette.primary.main})`,
        transform: 'translateY(-2px)',
      },
    };
  }

  return baseStyles;
});

export const GoldenButton: React.FC<GoldenButtonProps> = ({
  variant = 'contained',
  glowColor,
  gradientFrom,
  gradientTo,
  children,
  ...props
}) => {
  const theme = useTheme();
  const customVariant = variant as GoldenButtonVariant;

  // Handle custom variants
  if (customVariant === 'glow' || customVariant === 'gradient') {
    return (
      <StyledButton
        variant="contained" // Use base variant for Material-UI
        {...props}
        glowColor={glowColor}
        gradientFrom={gradientFrom}
        gradientTo={gradientTo}
        sx={{
          ...((customVariant === 'glow') && {
            background: `linear-gradient(45deg, ${glowColor || theme.palette.primary.main}, ${alpha(glowColor || theme.palette.primary.main, 0.8)})`,
            boxShadow: `0 0 20px ${alpha(glowColor || theme.palette.primary.main, 0.5)}`,
            animation: `${pulse} 2s infinite`,
            '&:hover': {
              boxShadow: `0 0 30px ${alpha(glowColor || theme.palette.primary.main, 0.8)}`,
              transform: 'translateY(-2px)',
            },
          }),
          ...((customVariant === 'gradient') && {
            background: `linear-gradient(45deg, ${gradientFrom || theme.palette.primary.main}, ${gradientTo || theme.palette.secondary.main})`,
            '&:hover': {
              background: `linear-gradient(45deg, ${gradientTo || theme.palette.secondary.main}, ${gradientFrom || theme.palette.primary.main})`,
              transform: 'translateY(-2px)',
            },
          }),
        }}
      >
        {children}
      </StyledButton>
    );
  }

  // Use standard Material-UI button for standard variants
  return (
    <Button variant={variant as ButtonProps['variant']} {...props}>
      {children}
    </Button>
  );
};

// Card with hover effects
export const GoldenCard = styled('div')(({ theme }) => ({
  background: 'rgba(255, 255, 255, 0.02)',
  backdropFilter: 'blur(10px)',
  borderRadius: theme.shape.borderRadius * 2,
  border: '1px solid rgba(255, 215, 0, 0.1)',
  padding: theme.spacing(3),
  position: 'relative',
  overflow: 'hidden',
  transition: 'all 0.3s ease',

  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'radial-gradient(circle at var(--mouse-x) var(--mouse-y), rgba(255, 215, 0, 0.1) 0%, transparent 50%)',
    opacity: 0,
    transition: 'opacity 0.3s ease',
    pointerEvents: 'none',
  },

  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: '0 10px 40px rgba(255, 215, 0, 0.2)',
    border: '1px solid rgba(255, 215, 0, 0.3)',

    '&::before': {
      opacity: 1,
    },
  },
}));

// Animated badge
export const GoldenBadge = styled('span')(({ theme }) => ({
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: '4px 12px',
  borderRadius: '20px',
  fontSize: '0.75rem',
  fontWeight: 600,
  background: 'linear-gradient(45deg, #FFD700, #FFA500)',
  color: '#0A0E27',
  position: 'relative',

  '&::after': {
    content: '""',
    position: 'absolute',
    top: -2,
    left: -2,
    right: -2,
    bottom: -2,
    background: 'linear-gradient(45deg, #FFD700, #FFA500)',
    borderRadius: '20px',
    opacity: 0.3,
    filter: 'blur(8px)',
    zIndex: -1,
  },
}));

// Input with golden focus
export const GoldenInput = styled('input')(({ theme }) => ({
  width: '100%',
  padding: '12px 16px',
  background: 'rgba(255, 255, 255, 0.05)',
  border: '2px solid rgba(255, 215, 0, 0.2)',
  borderRadius: theme.shape.borderRadius,
  color: '#fff',
  fontSize: '1rem',
  outline: 'none',
  transition: 'all 0.3s ease',

  '&:focus': {
    border: '2px solid #FFD700',
    background: 'rgba(255, 255, 255, 0.08)',
    boxShadow: '0 0 0 3px rgba(255, 215, 0, 0.1)',
  },

  '&::placeholder': {
    color: 'rgba(255, 255, 255, 0.5)',
  },
}));

// Loading skeleton with shimmer
export const GoldenSkeleton = styled('div')(({ theme }) => ({
  background: 'linear-gradient(90deg, rgba(255, 215, 0, 0.1) 0%, rgba(255, 215, 0, 0.2) 50%, rgba(255, 215, 0, 0.1) 100%)',
  backgroundSize: '200% 100%',
  animation: `${pulse} 1.5s infinite`,
  borderRadius: theme.shape.borderRadius,
  height: '100%',
  width: '100%',
}));
