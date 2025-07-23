/**
 * LoadingOverlay Component
 *
 * Displays a loading state overlay on the chart during data fetching or analysis.
 * Provides visual feedback to users during async operations.
 *
 * Features:
 * - Semi-transparent backdrop
 * - Animated loading spinner
 * - Customizable loading messages
 * - Progress indicator support
 * - Smooth fade transitions
 */

import React from 'react';
import {
  Box,
  CircularProgress,
  Typography,
  Fade,
  LinearProgress,
  useTheme,
  alpha,
} from '@mui/material';
import { keyframes } from '@mui/material/styles';

interface LoadingOverlayProps {
  /**
   * Whether the loading overlay is visible
   */
  visible: boolean;

  /**
   * Primary loading message to display
   */
  message?: string;

  /**
   * Secondary message for additional context
   */
  subMessage?: string;

  /**
   * Progress percentage (0-100) for determinate loading
   */
  progress?: number;

  /**
   * Type of loading indicator
   */
  variant?: 'circular' | 'linear' | 'both';

  /**
   * Whether to blur the background content
   */
  blur?: boolean;

  /**
   * Custom height for the overlay container
   */
  height?: string | number;
}

/**
 * Pulse animation for the loading container
 */
const pulseAnimation = keyframes`
  0% {
    transform: scale(1);
    opacity: 0.8;
  }
  50% {
    transform: scale(1.05);
    opacity: 1;
  }
  100% {
    transform: scale(1);
    opacity: 0.8;
  }
`;

/**
 * Default loading messages for different contexts
 */
const DEFAULT_MESSAGES = {
  data: 'Loading market data...',
  analysis: 'Running AI analysis...',
  connecting: 'Connecting to real-time feed...',
  processing: 'Processing your request...',
};

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  visible,
  message = DEFAULT_MESSAGES.data,
  subMessage,
  progress,
  variant = 'circular',
  blur = true,
  height = '100%',
}) => {
  const theme = useTheme();
  const showProgress = progress !== undefined && progress >= 0 && progress <= 100;
  const showLinear = variant === 'linear' || variant === 'both';
  const showCircular = variant === 'circular' || variant === 'both';

  return (
    <Fade in={visible} timeout={300}>
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: alpha(theme.palette.background.default, 0.7),
          backdropFilter: blur ? 'blur(4px)' : 'none',
          zIndex: 1000,
          height,
          // Prevent interaction with underlying content
          pointerEvents: visible ? 'auto' : 'none',
        }}
      >
        {/* Loading container */}
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 2,
            p: 4,
            borderRadius: 2,
            backgroundColor: alpha(theme.palette.background.paper, 0.9),
            boxShadow: `0 8px 32px ${alpha(theme.palette.common.black, 0.2)}`,
            animation: `${pulseAnimation} 2s ease-in-out infinite`,
            minWidth: 200,
            maxWidth: 400,
          }}
        >
          {/* Circular progress */}
          {showCircular && (
            <CircularProgress
              size={48}
              thickness={4}
              variant={showProgress ? 'determinate' : 'indeterminate'}
              value={progress}
              sx={{
                color: theme.palette.primary.main,
                '& .MuiCircularProgress-circle': {
                  strokeLinecap: 'round',
                },
              }}
            />
          )}

          {/* Messages */}
          <Box sx={{ textAlign: 'center' }}>
            <Typography
              variant="body1"
              color="text.primary"
              fontWeight={500}
              gutterBottom={!!subMessage}
            >
              {message}
            </Typography>

            {subMessage && (
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ display: 'block' }}
              >
                {subMessage}
              </Typography>
            )}
          </Box>

          {/* Linear progress */}
          {showLinear && (
            <Box sx={{ width: '100%', mt: 1 }}>
              <LinearProgress
                variant={showProgress ? 'determinate' : 'indeterminate'}
                value={progress}
                sx={{
                  height: 4,
                  borderRadius: 2,
                  backgroundColor: alpha(theme.palette.primary.main, 0.1),
                  '& .MuiLinearProgress-bar': {
                    borderRadius: 2,
                    background: `linear-gradient(90deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
                  },
                }}
              />

              {showProgress && (
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ display: 'block', textAlign: 'center', mt: 0.5 }}
                >
                  {progress}%
                </Typography>
              )}
            </Box>
          )}
        </Box>

        {/* Loading tips (optional) */}
        {visible && (
          <Fade in timeout={1000}>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{
                position: 'absolute',
                bottom: 20,
                textAlign: 'center',
                fontStyle: 'italic',
              }}
            >
              Tip: Premium subscribers enjoy faster analysis with priority processing
            </Typography>
          </Fade>
        )}
      </Box>
    </Fade>
  );
};
