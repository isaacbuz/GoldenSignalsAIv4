import React from 'react';
import { Box, Typography } from '@mui/material';
import { useAppSelector } from '../../store/store';
import { selectUserTheme } from '../../store/selectors';

interface FloatingOrbProps {
  // Define props here
  className?: string;
  children?: React.ReactNode;
}

export const FloatingOrb: React.FC<FloatingOrbProps> = ({
  className,
  children,
  ...props
}) => {
  const theme = useAppSelector(selectUserTheme);

  return (
    <Box
      data-testid="floatingorb"
      className={className}
      sx={{
        // Add styles here
      }}
      {...props}
    >
      <Typography variant="h6">
        FloatingOrb Component
      </Typography>
      {children}
    </Box>
  );
};

export default FloatingOrb;