import React from 'react';
import { Box, Typography } from '@mui/material';
import { useAppSelector } from '../../store/store';
import { selectUserTheme } from '../../store/selectors';

interface MarketInsightsProps {
  // Define props here
  className?: string;
  children?: React.ReactNode;
}

export const MarketInsights: React.FC<MarketInsightsProps> = ({
  className,
  children,
  ...props
}) => {
  const theme = useAppSelector(selectUserTheme);

  return (
    <Box
      data-testid="marketinsights"
      className={className}
      sx={{
        // Add styles here
      }}
      {...props}
    >
      <Typography variant="h6">
        MarketInsights Component
      </Typography>
      {children}
    </Box>
  );
};

export default MarketInsights;