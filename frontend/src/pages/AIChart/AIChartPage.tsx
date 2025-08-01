/**
 * AI Chart Page
 *
 * Displays the advanced AI Trading Chart with multi-agent analysis
 */

import React from 'react';
import { Container, Box } from '@mui/material';
import { AITradingChart } from '../../components/AIChart/AITradingChart';

export const AIChartPage: React.FC = () => {
  return (
    <Container maxWidth={false} sx={{ p: 0, height: '100vh' }}>
      <Box sx={{ height: 'calc(100vh - 64px)', position: 'relative' }}>
        <AITradingChart
          height="100%"
          autoAnalyze={true}
        />
      </Box>
    </Container>
  );
};

export default AIChartPage;
