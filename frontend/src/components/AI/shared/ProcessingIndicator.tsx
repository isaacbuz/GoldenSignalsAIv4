import React from 'react';
import { Box, Typography, LinearProgress, Fade } from '@mui/material';
import { AutoAwesome } from '@mui/icons-material';
import { styled, keyframes } from '@mui/material/styles';

const shimmer = keyframes`
  0% {
    background-position: -1000px 0;
  }
  100% {
    background-position: 1000px 0;
  }
`;

const ProcessingContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(3),
  borderRadius: theme.spacing(2),
  background: 'rgba(10, 14, 39, 0.8)',
  border: '1px solid rgba(255, 215, 0, 0.2)',
  backdropFilter: 'blur(10px)',
}));

const ShimmerText = styled(Typography)(({ theme }) => ({
  background: 'linear-gradient(90deg, #FFD700 0%, #FFA500 50%, #FFD700 100%)',
  backgroundSize: '1000px 100%',
  WebkitBackgroundClip: 'text',
  WebkitTextFillColor: 'transparent',
  animation: `${shimmer} 3s infinite linear`,
}));

interface ProcessingIndicatorProps {
  message?: string;
  subMessage?: string;
  progress?: number;
  variant?: 'default' | 'minimal' | 'detailed';
}

const ProcessingIndicator: React.FC<ProcessingIndicatorProps> = ({
  message = 'AI is processing...',
  subMessage,
  progress,
  variant = 'default',
}) => {
  if (variant === 'minimal') {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <AutoAwesome sx={{ fontSize: 20, color: '#FFD700', animation: 'spin 2s linear infinite' }} />
        <Typography variant="body2" color="text.secondary">
          {message}
        </Typography>
      </Box>
    );
  }

  return (
    <Fade in>
      <ProcessingContainer>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <AutoAwesome
            sx={{
              fontSize: 32,
              color: '#FFD700',
              mr: 2,
              animation: 'spin 2s linear infinite',
            }}
          />
          <Box>
            <ShimmerText variant="h6">{message}</ShimmerText>
            {subMessage && (
              <Typography variant="caption" color="text.secondary">
                {subMessage}
              </Typography>
            )}
          </Box>
        </Box>

        {progress !== undefined ? (
          <LinearProgress
            variant="determinate"
            value={progress}
            sx={{
              height: 6,
              borderRadius: 3,
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
              '& .MuiLinearProgress-bar': {
                backgroundColor: '#FFD700',
                borderRadius: 3,
              },
            }}
          />
        ) : (
          <LinearProgress
            sx={{
              height: 6,
              borderRadius: 3,
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
              '& .MuiLinearProgress-bar': {
                backgroundColor: '#FFD700',
                borderRadius: 3,
              },
            }}
          />
        )}

        {variant === 'detailed' && (
          <Box sx={{ mt: 2, display: 'flex', gap: 3 }}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Models Active
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                8/9
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Confidence Building
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                {progress ? `${progress}%` : 'Calculating...'}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Est. Time
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                ~2s
              </Typography>
            </Box>
          </Box>
        )}
      </ProcessingContainer>
    </Fade>
  );
};

export default ProcessingIndicator;
