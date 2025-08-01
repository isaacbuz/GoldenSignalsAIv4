import React, { useState } from 'react';
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
import { Button } from '../Core/Button';
import {
  ArrowUpIcon,
  ArrowDownIcon,
  ChartBarIcon,
  ClockIcon,
  ShieldCheckIcon,
  SparklesIcon,
  InformationCircleIcon,
  ChevronDownIcon,
  ChevronUpIcon
} from '@heroicons/react/24/outline';
import { clsx } from 'clsx';

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

interface Target {
  price: number;
  percentage: number;
}

interface SignalLevels {
  entry: {
    price: number;
    type: 'MARKET' | 'LIMIT';
    zone?: [number, number];
  };
  targets: Target[];
  stopLoss: {
    price: number;
    type: 'STOP' | 'STOP_LIMIT';
    trailingOptions?: {
      activateAt: number;
      trailBy: number;
    };
  };
}

interface SignalCardProps {
  signal: SignalData;
  variant?: 'compact' | 'detailed';
  highlight?: boolean;
  onClick?: (signal: SignalData) => void;
  onAction?: (action: string, signal: SignalData) => void;
  currentPrice: number;
  levels: SignalLevels;
  riskReward: string;
  performance?: {
    hitTarget?: boolean;
    returnPct?: number;
  };
  onViewChart?: () => void;
  onShare?: () => void;
}

/**
 * Enhanced SignalCard - Displays complete trade suggestions
 *
 * Shows:
 * - Entry price/zone
 * - Multiple exit targets
 * - Stop loss level
 * - Risk/reward visualization
 * - AI confidence and reasoning
 *
 * This is NOT for execution - just detailed suggestions!
 */
export const SignalCard: React.FC<SignalCardProps> = ({
  signal,
  variant = 'compact',
  highlight = false,
  onClick,
  onAction,
  currentPrice,
  levels,
  riskReward,
  performance,
  onViewChart,
  onShare
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

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

  // Calculate distances from current price
  const entryDistance = ((levels.entry.price - currentPrice) / currentPrice * 100).toFixed(2);
  const stopDistance = ((levels.stopLoss.price - currentPrice) / currentPrice * 100).toFixed(2);

  // Determine colors based on action
  const actionColors = {
    BUY: {
      bg: 'bg-green-50 dark:bg-green-900/20',
      text: 'text-green-700 dark:text-green-400',
      border: 'border-green-200 dark:border-green-800'
    },
    SELL: {
      bg: 'bg-red-50 dark:bg-red-900/20',
      text: 'text-red-700 dark:text-red-400',
      border: 'border-red-200 dark:border-red-800'
    },
    HOLD: {
      bg: 'bg-blue-50 dark:bg-blue-900/20',
      text: 'text-blue-700 dark:text-blue-400',
      border: 'border-blue-200 dark:border-blue-800'
    }
  };

  const colors = actionColors[signal.type];

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

            {/* Price Levels Grid */}
            <div className="grid grid-cols-3 gap-4 mb-4">
              {/* Entry */}
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">Entry Price</div>
                <div className="text-lg font-bold text-gray-900 dark:text-white">
                  ${levels.entry.price.toFixed(2)}
                </div>
                {levels.entry.zone && (
                  <div className="text-xs text-gray-500">
                    Zone: ${levels.entry.zone[0]} - ${levels.entry.zone[1]}
                  </div>
                )}
                <div className={`text-xs mt-1 ${parseFloat(entryDistance) > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {entryDistance}% from current
                </div>
              </div>

              {/* Targets */}
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">Targets</div>
                <div className="space-y-1">
                  {levels.targets.slice(0, 2).map((target, idx) => (
                    <div key={idx} className="flex justify-between items-center">
                      <span className="text-sm font-medium">${target.price.toFixed(2)}</span>
                      <span className="text-xs text-gray-500">{target.percentage}%</span>
                    </div>
                  ))}
                  {levels.targets.length > 2 && (
                    <div className="text-xs text-gray-500">+{levels.targets.length - 2} more</div>
                  )}
                </div>
              </div>

              {/* Stop Loss */}
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">Stop Loss</div>
                <div className="text-lg font-bold text-red-600">
                  ${levels.stopLoss.price.toFixed(2)}
                </div>
                {levels.stopLoss.trailingOptions && (
                  <div className="text-xs text-gray-500">Trailing available</div>
                )}
                <div className="text-xs mt-1 text-red-600">
                  {Math.abs(parseFloat(stopDistance))}% risk
                </div>
              </div>
            </div>

            {/* Risk/Reward Bar */}
            <div className="mb-4">
              <div className="flex justify-between items-center mb-1">
                <span className="text-sm text-gray-600">Risk/Reward Ratio</span>
                <span className="text-sm font-bold text-blue-600">{riskReward}</span>
              </div>
              <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div className="h-full flex">
                  <div className="bg-red-500 w-1/4"></div>
                  <div className="bg-green-500 flex-1"></div>
                </div>
              </div>
            </div>

            {/* Current Price Reference */}
            <div className="flex items-center justify-between p-3 bg-gray-100 dark:bg-gray-800 rounded-lg mb-4">
              <span className="text-sm text-gray-600">Current Price</span>
              <span className="text-lg font-bold">${currentPrice.toFixed(2)}</span>
            </div>

            {/* Expandable Reasoning */}
            {signal.reasoning && (
              <div className="mb-4">
                <button
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
                >
                  <InformationCircleIcon className="w-4 h-4" />
                  <span>AI Reasoning</span>
                  {isExpanded ? (
                    <ChevronUpIcon className="w-4 h-4" />
                  ) : (
                    <ChevronDownIcon className="w-4 h-4" />
                  )}
                </button>

                {isExpanded && (
                  <div className="mt-2 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <p className="text-sm text-gray-700 dark:text-gray-300">{signal.reasoning}</p>
                  </div>
                )}
              </div>
            )}

            {/* Performance Badge (if available) */}
            {performance && (
              <div className="flex items-center gap-2 mb-4">
                <span className="text-sm text-gray-600">Performance:</span>
                <span className={clsx(
                  'px-2 py-1 text-xs rounded-full font-medium',
                  performance.hitTarget
                    ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                    : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                )}>
                  {performance.hitTarget ? 'Target Hit' : 'Stopped Out'}
                  {performance.returnPct && ` (${performance.returnPct > 0 ? '+' : ''}${performance.returnPct}%)`}
                </span>
              </div>
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

        {/* Actions */}
        <div className="flex gap-3 mt-4">
          <Button
            variant="primary"
            onClick={onViewChart}
            className="flex-1"
          >
            <ChartBarIcon className="w-4 h-4 mr-2" />
            View on Chart
          </Button>
          <Button
            variant="secondary"
            onClick={onShare}
          >
            Share
          </Button>
        </div>

        {/* Timestamp */}
        <div className="flex items-center gap-1 mt-4 text-xs text-gray-500">
          <ClockIcon className="w-3 h-3" />
          <span>{new Date(signal.timestamp).toLocaleString()}</span>
        </div>

        {/* Disclaimer */}
        <div className="mt-4 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded text-xs text-yellow-800 dark:text-yellow-200">
          ⚠️ This is a trade suggestion only. Not financial advice. Execute at your own discretion.
        </div>
      </CardContent>
    </StyledCard>
  );
};

export default SignalCard;
