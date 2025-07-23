/**
 * AgentErrorAlert Component
 *
 * Displays user-friendly error messages when agent analysis fails.
 * Provides context-specific error information and recovery options.
 *
 * Features:
 * - Clear error descriptions
 * - Retry functionality
 * - Contact support option for persistent errors
 * - Auto-dismiss for transient errors
 */

import React, { useEffect } from 'react';
import {
  Alert,
  AlertTitle,
  Button,
  Box,
  Typography,
  Collapse,
  IconButton,
} from '@mui/material';
import {
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  Close as CloseIcon,
  HelpOutline as HelpIcon,
} from '@mui/icons-material';

interface AgentErrorAlertProps {
  /**
   * The error object containing error details
   */
  error: Error | null;

  /**
   * Number of retry attempts made
   */
  retryCount?: number;

  /**
   * Maximum retry attempts before showing contact support
   */
  maxRetries?: number;

  /**
   * Callback to retry the failed operation
   */
  onRetry?: () => void;

  /**
   * Callback to dismiss the error alert
   */
  onDismiss?: () => void;

  /**
   * Auto-dismiss the alert after specified milliseconds
   */
  autoDismissMs?: number;
}

/**
 * Maps error messages to user-friendly descriptions
 */
const getErrorDescription = (error: Error): string => {
  const errorMap: Record<string, string> = {
    'Network Error': 'Unable to connect to the analysis service. Please check your internet connection.',
    'Analysis cancelled': 'The analysis was cancelled. Click retry to start again.',
    'No analysis result received': 'The analysis completed but no results were returned. This may indicate a server issue.',
    'Timeout': 'The analysis is taking longer than expected. Please try again.',
    '403': 'You don\'t have permission to analyze this symbol. Please check your subscription.',
    '429': 'Too many requests. Please wait a moment before trying again.',
    '500': 'Server error occurred. Our team has been notified.',
  };

  // Check for mapped errors
  for (const [key, description] of Object.entries(errorMap)) {
    if (error.message.includes(key)) {
      return description;
    }
  }

  // Default error message
  return 'An unexpected error occurred during analysis. Please try again.';
};

/**
 * Determines error severity based on error type
 */
const getErrorSeverity = (error: Error): 'error' | 'warning' | 'info' => {
  if (error.message.includes('Network') || error.message.includes('500')) {
    return 'error';
  }
  if (error.message.includes('cancelled') || error.message.includes('429')) {
    return 'warning';
  }
  return 'info';
};

export const AgentErrorAlert: React.FC<AgentErrorAlertProps> = ({
  error,
  retryCount = 0,
  maxRetries = 3,
  onRetry,
  onDismiss,
  autoDismissMs,
}) => {
  // Auto-dismiss effect
  useEffect(() => {
    if (error && autoDismissMs && onDismiss) {
      const timer = setTimeout(() => {
        onDismiss();
      }, autoDismissMs);

      return () => clearTimeout(timer);
    }
  }, [error, autoDismissMs, onDismiss]);

  // Don't render if no error
  if (!error) {
    return null;
  }

  const severity = getErrorSeverity(error);
  const description = getErrorDescription(error);
  const showContactSupport = retryCount >= maxRetries;

  return (
    <Collapse in={!!error}>
      <Alert
        severity={severity}
        icon={<ErrorIcon />}
        action={
          <IconButton
            aria-label="close"
            color="inherit"
            size="small"
            onClick={onDismiss}
          >
            <CloseIcon fontSize="inherit" />
          </IconButton>
        }
        sx={{
          mb: 2,
          '& .MuiAlert-icon': {
            fontSize: '28px',
          },
        }}
      >
        <AlertTitle sx={{ fontWeight: 600 }}>
          Agent Analysis Failed
        </AlertTitle>

        <Typography variant="body2" sx={{ mb: 2 }}>
          {description}
        </Typography>

        {/* Error details for debugging (only in development) */}
        {process.env.NODE_ENV === 'development' && (
          <Typography
            variant="caption"
            sx={{
              display: 'block',
              fontFamily: 'monospace',
              color: 'text.secondary',
              mb: 1
            }}
          >
            {error.message}
          </Typography>
        )}

        <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
          {/* Retry button */}
          {onRetry && !showContactSupport && (
            <Button
              size="small"
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={onRetry}
              sx={{ textTransform: 'none' }}
            >
              Retry Analysis
              {retryCount > 0 && ` (${retryCount}/${maxRetries})`}
            </Button>
          )}

          {/* Contact support after max retries */}
          {showContactSupport && (
            <Button
              size="small"
              variant="outlined"
              startIcon={<HelpIcon />}
              onClick={() => {
                // In production, this would open a support ticket
                window.open('/support', '_blank');
              }}
              sx={{ textTransform: 'none' }}
            >
              Contact Support
            </Button>
          )}
        </Box>

        {/* Additional help text for specific errors */}
        {error.message.includes('429') && (
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
            Tip: Our premium plans offer higher rate limits for frequent analysis.
          </Typography>
        )}
      </Alert>
    </Collapse>
  );
};
