import React from 'react';
import { Box, Typography } from '@mui/material';
import { useAppSelector } from '../../store/store';
import { selectUserTheme } from '../../store/selectors';

interface OptionsChainTableProps {
  // Define props here
  className?: string;
  children?: React.ReactNode;
}

export const OptionsChainTable: React.FC<OptionsChainTableProps> = ({
  className,
  children,
  ...props
}) => {
  const theme = useAppSelector(selectUserTheme);

  return (
    <Box
      data-testid="optionschaintable"
      className={className}
      sx={{
        // Add styles here
      }}
      {...props}
    >
      <Typography variant="h6">
        OptionsChainTable Component
      </Typography>
      {children}
    </Box>
  );
};

export default OptionsChainTable;
