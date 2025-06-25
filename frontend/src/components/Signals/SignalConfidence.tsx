import React from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  LinearProgress,
  Chip,
  Tooltip,
} from '@mui/material';
import { Speed, TrendingUp, Warning } from '@mui/icons-material';
import { styled } from '@mui/material/styles';

const ConfidenceRing = styled(Box)(({ theme }) => ({
  position: 'relative',
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
}));

const ConfidenceText = styled(Typography)(({ theme }) => ({
  position: 'absolute',
  fontWeight: 700,
  background: 'linear-gradient(45deg, #FFD700 30%, #FFA500 90%)',
  WebkitBackgroundClip: 'text',
  WebkitTextFillColor: 'transparent',
}));

interface SignalConfidenceProps {
  confidence: number;
  variant?: 'circular' | 'linear' | 'compact';
  size?: 'small' | 'medium' | 'large';
  showLabel?: boolean;
  historicalAccuracy?: number;
}

const SignalConfidence: React.FC<SignalConfidenceProps> = ({
  confidence,
  variant = 'circular',
  size = 'medium',
  showLabel = true,
  historicalAccuracy,
}) => {
  const getColor = () => {
    if (confidence >= 90) return '#4CAF50';
    if (confidence >= 80) return '#FFD700';
    if (confidence >= 70) return '#FFA500';
    return '#F44336';
  };

  const getSizeProps = () => {
    switch (size) {
      case 'small':
        return { circularSize: 60, fontSize: 'h6' };
      case 'large':
        return { circularSize: 120, fontSize: 'h3' };
      default:
        return { circularSize: 80, fontSize: 'h5' };
    }
  };

  const { circularSize, fontSize } = getSizeProps();

  if (variant === 'circular') {
    return (
      <Box sx={{ textAlign: 'center' }}>
        <ConfidenceRing>
          <CircularProgress
            variant="determinate"
            value={confidence}
            size={circularSize}
            thickness={4}
            sx={{
              color: getColor(),
              '& .MuiCircularProgress-circle': {
                strokeLinecap: 'round',
              },
            }}
          />
          <ConfidenceText variant={fontSize as any}>
            {confidence.toFixed(0)}%
          </ConfidenceText>
        </ConfidenceRing>
        {showLabel && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            Confidence Score
          </Typography>
        )}
        {historicalAccuracy && (
          <Tooltip title="Historical accuracy for similar signals">
            <Chip
              label={`Hist: ${historicalAccuracy.toFixed(1)}%`}
              size="small"
              sx={{ mt: 1 }}
              icon={<TrendingUp sx={{ fontSize: 16 }} />}
            />
          </Tooltip>
        )}
      </Box>
    );
  }

  if (variant === 'linear') {
    return (
      <Box>
        {showLabel && (
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="caption" color="text.secondary">
              Confidence Score
            </Typography>
            <Typography variant="caption" sx={{ fontWeight: 600, color: getColor() }}>
              {confidence.toFixed(1)}%
            </Typography>
          </Box>
        )}
        <LinearProgress
          variant="determinate"
          value={confidence}
          sx={{
            height: size === 'small' ? 4 : size === 'large' ? 12 : 8,
            borderRadius: 2,
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            '& .MuiLinearProgress-bar': {
              backgroundColor: getColor(),
              borderRadius: 2,
            },
          }}
        />
        {confidence < 70 && (
          <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
            <Warning sx={{ fontSize: 14, color: '#F44336', mr: 0.5 }} />
            <Typography variant="caption" color="error">
              Low confidence signal
            </Typography>
          </Box>
        )}
      </Box>
    );
  }

  // Compact variant
  return (
    <Chip
      label={`${confidence.toFixed(1)}%`}
      size={size === 'small' ? 'small' : 'medium'}
      icon={<Speed />}
      sx={{
        fontWeight: 600,
        backgroundColor: 'rgba(255, 215, 0, 0.1)',
        color: getColor(),
        border: `1px solid ${getColor()}33`,
      }}
    />
  );
};

export default SignalConfidence;
