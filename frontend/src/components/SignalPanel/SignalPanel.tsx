import React from 'react';
import { Box, Typography } from '@mui/material';
import { useAppSelector } from '../../store/store';
import { selectUserTheme } from '../../store/selectors';

interface SignalPanelProps {
  // Define props here
  className?: string;
  children?: React.ReactNode;
}

export const SignalPanel: React.FC<SignalPanelProps> = ({
  className,
  children,
  ...props
}) => {
  const theme = useAppSelector(selectUserTheme);

  return (
    <Box
      data-testid="signalpanel"
      className={className}
      sx={{
        // Add styles here
      }}
      {...props}
    >
      <Typography variant="h6">
        SignalPanel Component
      </Typography>
      {children}
    </Box>
  );
};

export default SignalPanel;