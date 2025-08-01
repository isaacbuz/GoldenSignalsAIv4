/**
 * Signals Page
 *
 * Trading signals management with filtering, sorting, and real-time updates
 */

import React, { useState } from 'react';
import { Box, Container } from '@mui/material';
import { useParams } from 'react-router-dom';
import { AITradingChart } from '../../components/AIChart/AITradingChart';

export const SignalsPage: React.FC = () => {
  const { symbol } = useParams<{ symbol?: string }>();
  const [selectedSymbol, setSelectedSymbol] = useState(symbol || 'AAPL');

  return (
    <Box sx={{ 
      height: '100%',
      display: 'flex', 
      flexDirection: 'column',
      p: 0 
    }}>
      <AITradingChart 
        symbol={selectedSymbol}
        onSymbolChange={setSelectedSymbol}
      />
    </Box>
  );
};
