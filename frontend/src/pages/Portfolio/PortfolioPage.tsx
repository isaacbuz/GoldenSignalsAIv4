/**
 * Portfolio Page
 *
 * Portfolio tracking with performance metrics, positions, and P&L analysis
 */

import React from 'react';
import { Box, Typography, Container } from '@mui/material';

export const PortfolioPage: React.FC = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom>
        Portfolio Tracking
      </Typography>
      <Box sx={{ mt: 4 }}>
        <Typography variant="body1" color="text.secondary">
          Track your signal performance and theoretical portfolio here.
        </Typography>
      </Box>
    </Container>
  );
};

export default PortfolioPage;
