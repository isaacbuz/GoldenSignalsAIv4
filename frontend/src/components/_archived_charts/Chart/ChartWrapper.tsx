import React from 'react';
import { Box } from '@mui/material';
import ProfessionalChart from '../ProfessionalChart/ProfessionalChart';
import MiniChart from './MiniChart';

export interface ChartWrapperProps {
  // Core props
  symbol: string;
  variant?: 'full' | 'mini' | 'compact';

  // Display options
  height?: string | number;
  showIndicators?: boolean;
  showVolume?: boolean;
  showPrediction?: boolean;
  showSignals?: boolean;

  // Data and callbacks
  onSymbolChange?: (symbol: string) => void;
  onTimeframeChange?: (timeframe: string) => void;

  // Additional features
  enableWebSocket?: boolean;
  showWatermark?: boolean;
  theme?: 'dark' | 'light';
}

/**
 * Unified chart wrapper component that provides a consistent interface
 * for all chart usage across the application
 */
export const ChartWrapper: React.FC<ChartWrapperProps> = ({
  symbol,
  variant = 'full',
  height = '100%',
  showIndicators = true,
  showVolume = true,
  showPrediction = true,
  showSignals = true,
  onSymbolChange,
  onTimeframeChange,
  enableWebSocket = true,
  showWatermark = false,
  theme = 'dark',
}) => {
  // For mini variant, use the lightweight MiniChart component
  if (variant === 'mini') {
    return (
      <MiniChart
        symbol={symbol}
        height={height as number || 100}
        showTooltip={false}
      />
    );
  }

  // For compact variant, show ProfessionalChart with minimal features
  if (variant === 'compact') {
    return (
      <Box sx={{ height, width: '100%' }}>
        <ProfessionalChart
          symbol={symbol}
          // Disable most features for compact view
          initialIndicators={showVolume ? ['volume'] : []}
        />
      </Box>
    );
  }

  // Default full variant with all features
  return (
    <Box sx={{ height, width: '100%' }}>
      <ProfessionalChart
        symbol={symbol}
        // Pass through all configuration
        initialIndicators={
          [
            ...(showPrediction ? ['prediction'] : []),
            ...(showSignals ? ['signals'] : []),
            ...(showVolume ? ['volume'] : []),
            ...(showIndicators ? ['patterns'] : []),
          ]
        }
      />
    </Box>
  );
};

// Export convenience components for common use cases
export const FullChart: React.FC<Omit<ChartWrapperProps, 'variant'>> = (props) => (
  <ChartWrapper {...props} variant="full" />
);

export const CompactChart: React.FC<Omit<ChartWrapperProps, 'variant'>> = (props) => (
  <ChartWrapper {...props} variant="compact" />
);

export const MiniChartWrapper: React.FC<Omit<ChartWrapperProps, 'variant'>> = (props) => (
  <ChartWrapper {...props} variant="mini" />
);

export default ChartWrapper;
