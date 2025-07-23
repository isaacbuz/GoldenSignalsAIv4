import React, { useEffect, useRef, useState } from 'react';
import { Box } from '@mui/material';

interface ChartWrapperProps {
  children: React.ReactElement;
}

/**
 * Wrapper component to handle Chart.js cleanup in React StrictMode
 * Prevents "Canvas is already in use" errors
 */
export const ChartWrapper: React.FC<ChartWrapperProps> = ({ children }) => {
  const [key, setKey] = useState(0);
  const isMountedRef = useRef(false);

  useEffect(() => {
    // Force re-render on mount to avoid StrictMode double render issues
    if (!isMountedRef.current) {
      isMountedRef.current = true;
      setKey(prev => prev + 1);
    }

    return () => {
      isMountedRef.current = false;
    };
  }, []);

  return (
    <Box key={key} sx={{ width: '100%', height: '100%', position: 'relative' }}>
      {children}
    </Box>
  );
};

export default ChartWrapper;
