import React from 'react';
import {
  Card,
  CardContent,
  Box,
  Typography,
  Chip,
  IconButton,
  LinearProgress,
  Tooltip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Remove,
  MoreVert,
  Speed,
  Psychology,
  Timeline,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { utilityClasses } from '../../theme/goldenTheme';

const StyledCard = styled(Card)<{ highlight?: boolean }>(({ theme, highlight }) => ({
  ...utilityClasses.glassmorphism,
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  position: 'relative',
  overflow: 'hidden',
  ...(highlight && {
    '&::before': {
      content: '""',
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      height: '3px',
      background: 'linear-gradient(90deg, #FFD700 0%, #FFA500 100%)',
    },
  }),
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: '0 8px 24px rgba(255, 215, 0, 0.2)',
    borderColor: 'rgba(255, 215, 0, 0.3)',
  },
}));

const SignalBadge = styled(Chip)<{ signalType: string }>(({ signalType }) => ({
  fontWeight: 700,
  ...(signalType === 'BUY' && {
    backgroundColor: 'rgba(76, 175, 80, 0.1)',
    color: '#4CAF50',
    border: '1px solid rgba(76, 175, 80, 0.3)',
  }),
  ...(signalType === 'SELL' && {
    backgroundColor: 'rgba(244, 67, 54, 0.1)',
    color: '#F44336',
    border: '1px solid rgba(244, 67, 54, 0.3)',
  }),
  ...(signalType === 'HOLD' && {
    backgroundColor: 'rgba(255, 165, 0, 0.1)',
    color: '#FFA500',
    border: '1px solid rgba(255, 165, 0, 0.3)',
  }),
}));

export interface SignalData {
  id: string;
  symbol: string;
  type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  timestamp: string;
  agents: string[];
  reasoning?: string;
  impact?: 'HIGH' | 'MEDIUM' | 'LOW';
  historicalAccuracy?: number;
  expectedMove?: number;
  metadata?: any;
}

interface SignalCardProps {
  signal: SignalData;
  variant?: 'compact' | 'detailed';
  highlight?: boolean;
  onClick?: (signal: SignalData) => void;
  onAction?: (action: string, signal: SignalData) => void;
}

const SignalCard: React.FC<SignalCardProps> = ({
  signal,
  variant = 'compact',
  highlight = false,
  onClick,
  onAction,
}) => {
  const getSignalIcon = () => {
    switch (signal.type) {
      case 'BUY':
        return <TrendingUp />;
      case 'SELL':
        return <TrendingDown />;
      default:
        return <Remove />;
    }
  };

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onClick?.(signal);
  };

  return (
    <StyledCard highlight={highlight} onClick={handleClick}>
      <CardContent sx={{ p: variant === 'compact' ? 2 : 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <Box sx={{ flex: 1 }}>
            {/* Header */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: variant === 'compact' ? 1 : 2 }}>
              <Typography variant={variant === 'compact' ? 'body1' : 'h6'} sx={{ fontWeight: 600 }}>
                {signal.symbol}
              </Typography>
              <SignalBadge
                label={signal.type}
                icon={getSignalIcon()}
                size={variant === 'compact' ? 'small' : 'medium'}
                signalType={signal.type}
              />
              <Chip
                label={`${signal.confidence.toFixed(1)}%`}
                size={variant === 'compact' ? 'small' : 'medium'}
                icon={<Speed />}
                sx={{ fontWeight: 600 }}
              />
              {signal.impact === 'HIGH' && (
                <Chip
                  label="HIGH IMPACT"
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(255, 215, 0, 0.1)',
                    color: '#FFD700',
                    border: '1px solid rgba(255, 215, 0, 0.3)',
                  }}
                />
              )}
            </Box>

            {/* Reasoning - only in detailed view */}
            {variant === 'detailed' && signal.reasoning && (
              <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
                {signal.reasoning}
              </Typography>
            )}

            {/* Metadata */}
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
              <Tooltip title="Contributing Agents">
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Psychology sx={{ fontSize: 16, color: 'text.secondary' }} />
                  <Typography variant="caption" color="text.secondary">
                    {signal.agents.join(', ')}
                  </Typography>
                </Box>
              </Tooltip>
              
              {signal.historicalAccuracy && (
                <Tooltip title="Historical Accuracy">
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Timeline sx={{ fontSize: 16, color: 'text.secondary' }} />
                    <Typography variant="caption" color="text.secondary">
                      {signal.historicalAccuracy.toFixed(1)}%
                    </Typography>
                  </Box>
                </Tooltip>
              )}
              
              <Typography variant="caption" color="text.secondary">
                {signal.timestamp}
              </Typography>
            </Box>

            {/* Confidence Bar - detailed view only */}
            {variant === 'detailed' && (
              <Box sx={{ mt: 2 }}>
                <LinearProgress
                  variant="determinate"
                  value={signal.confidence}
                  sx={{
                    height: 6,
                    borderRadius: 3,
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    '& .MuiLinearProgress-bar': {
                      backgroundColor:
                        signal.confidence > 90 ? '#4CAF50' :
                        signal.confidence > 80 ? '#FFD700' : '#FFA500',
                      borderRadius: 3,
                    },
                  }}
                />
              </Box>
            )}
          </Box>

          {/* Action Menu */}
          <IconButton
            size="small"
            onClick={(e) => {
              e.stopPropagation();
              onAction?.('menu', signal);
            }}
          >
            <MoreVert />
          </IconButton>
        </Box>
      </CardContent>
    </StyledCard>
  );
};

export default SignalCard;
