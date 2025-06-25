import React from 'react';
import { Button as MuiButton, ButtonProps, styled } from '@mui/material';
import { keyframes } from '@mui/system';

const shimmer = keyframes`
  0% {
    background-position: -200% center;
  }
  100% {
    background-position: 200% center;
  }
`;

const glow = keyframes`
  0% {
    box-shadow: 0 0 5px rgba(255, 215, 0, 0.5);
  }
  50% {
    box-shadow: 0 0 20px rgba(255, 215, 0, 0.8), 0 0 30px rgba(255, 215, 0, 0.6);
  }
  100% {
    box-shadow: 0 0 5px rgba(255, 215, 0, 0.5);
  }
`;

interface GoldenButtonProps extends ButtonProps {
  variant?: 'contained' | 'outlined' | 'text' | 'gradient' | 'glow';
  loading?: boolean;
}

const StyledButton = styled(MuiButton)<{ variant?: string; loading?: boolean }>(({ theme, variant, loading }) => ({
  position: 'relative',
  textTransform: 'none',
  fontWeight: 600,
  borderRadius: theme.shape.borderRadius * 1.5,
  padding: '10px 24px',
  transition: 'all 0.3s ease',
  
  ...(variant === 'gradient' && {
    background: 'linear-gradient(45deg, #FFD700 30%, #FFA500 90%)',
    color: '#0A0E27',
    '&:hover': {
      background: 'linear-gradient(45deg, #FFA500 30%, #FFD700 90%)',
      transform: 'translateY(-2px)',
      boxShadow: '0 6px 20px rgba(255, 215, 0, 0.4)',
    },
  }),
  
  ...(variant === 'glow' && {
    background: 'rgba(255, 215, 0, 0.1)',
    color: '#FFD700',
    border: '2px solid #FFD700',
    animation: `${glow} 2s ease-in-out infinite`,
    '&:hover': {
      background: 'rgba(255, 215, 0, 0.2)',
    },
  }),
  
  ...(loading && {
    color: 'transparent',
    '&::after': {
      content: '""',
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent)',
      backgroundSize: '200% 100%',
      animation: `${shimmer} 1.5s infinite`,
    },
  }),
}));

export const GoldenButton: React.FC<GoldenButtonProps> = ({ 
  children, 
  variant = 'contained', 
  loading = false,
  disabled,
  ...props 
}) => {
  return (
    <StyledButton
      variant={variant as any}
      loading={loading}
      disabled={disabled || loading}
      {...props}
    >
      {children}
    </StyledButton>
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
  animation: `${shimmer} 1.5s infinite`,
  borderRadius: theme.shape.borderRadius,
  height: '100%',
  width: '100%',
}));
