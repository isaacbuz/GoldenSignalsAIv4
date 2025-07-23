import React, { useEffect, useState } from 'react';
import { Box, Typography, Fade, Zoom } from '@mui/material';
import { styled, keyframes } from '@mui/material/styles';
import {
  ShowChart as PatternIcon,
  TrendingUp as BullishIcon,
  TrendingDown as BearishIcon,
  CheckCircle as ConfirmIcon,
} from '@mui/icons-material';

const flashAnimation = keyframes`
  0% {
    opacity: 0;
    transform: scale(0.8);
  }
  20% {
    opacity: 1;
    transform: scale(1.1);
  }
  40% {
    opacity: 0.8;
    transform: scale(1);
  }
  60% {
    opacity: 1;
    transform: scale(1.05);
  }
  80% {
    opacity: 0.9;
    transform: scale(1);
  }
  100% {
    opacity: 1;
    transform: scale(1);
  }
`;

const rippleAnimation = keyframes`
  0% {
    transform: scale(0);
    opacity: 1;
  }
  100% {
    transform: scale(4);
    opacity: 0;
  }
`;

const PatternContainer = styled(Box)(({ theme }) => ({
  position: 'absolute',
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.paper,
  border: `2px solid ${theme.palette.warning.main}`,
  borderRadius: theme.spacing(1),
  boxShadow: `0 0 20px ${theme.palette.warning.main}`,
  animation: `${flashAnimation} 1s ease-out`,
  zIndex: 30,
  minWidth: 200,
  '&::before': {
    content: '""',
    position: 'absolute',
    top: '50%',
    left: '50%',
    width: '100%',
    height: '100%',
    transform: 'translate(-50%, -50%)',
    borderRadius: theme.spacing(1),
    border: `2px solid ${theme.palette.warning.main}`,
    animation: `${rippleAnimation} 1.5s ease-out`,
  },
}));

const PatternHeader = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  marginBottom: theme.spacing(1),
  '& .MuiSvgIcon-root': {
    color: theme.palette.warning.main,
    fontSize: '1.5rem',
  },
}));

interface PatternConfirmationProps {
  pattern: {
    type: string;
    name: string;
    confidence: number;
    direction: 'bullish' | 'bearish' | 'neutral';
    target?: number;
  };
  position: { x: number; y: number };
  onClose?: () => void;
}

export const PatternConfirmation: React.FC<PatternConfirmationProps> = ({
  pattern,
  position,
  onClose,
}) => {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(false);
      onClose?.();
    }, 5000); // Hide after 5 seconds

    return () => clearTimeout(timer);
  }, [onClose]);

  const getDirectionIcon = () => {
    switch (pattern.direction) {
      case 'bullish':
        return <BullishIcon color="success" />;
      case 'bearish':
        return <BearishIcon color="error" />;
      default:
        return <PatternIcon />;
    }
  };

  return (
    <Fade in={visible}>
      <PatternContainer
        style={{
          left: position.x,
          top: position.y,
          transform: 'translate(-50%, -100%)',
        }}
      >
        <PatternHeader>
          <ConfirmIcon color="warning" />
          <Typography variant="subtitle1" fontWeight="bold">
            Pattern Confirmed!
          </Typography>
        </PatternHeader>

        <Box>
          <Typography variant="body2" fontWeight="bold" gutterBottom>
            {pattern.name}
          </Typography>

          <Box display="flex" alignItems="center" gap={1} mb={1}>
            {getDirectionIcon()}
            <Typography variant="body2" color="text.secondary">
              {pattern.direction.charAt(0).toUpperCase() + pattern.direction.slice(1)} Signal
            </Typography>
          </Box>

          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant="caption" color="text.secondary">
              Confidence:
            </Typography>
            <Typography variant="caption" fontWeight="bold" color="warning.main">
              {(pattern.confidence * 100).toFixed(0)}%
            </Typography>
          </Box>

          {pattern.target && (
            <Box display="flex" justifyContent="space-between" alignItems="center" mt={0.5}>
              <Typography variant="caption" color="text.secondary">
                Target:
              </Typography>
              <Typography variant="caption" fontWeight="bold">
                ${pattern.target.toFixed(2)}
              </Typography>
            </Box>
          )}
        </Box>

        <Box
          sx={{
            position: 'absolute',
            bottom: -10,
            left: '50%',
            transform: 'translateX(-50%)',
            width: 0,
            height: 0,
            borderLeft: '10px solid transparent',
            borderRight: '10px solid transparent',
            borderTop: `10px solid ${(theme) => theme.palette.warning.main}`,
          }}
        />
      </PatternContainer>
    </Fade>
  );
};
