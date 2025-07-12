/**
 * Signals Page
 * 
 * Trading signals management with filtering, sorting, and real-time updates
 */

import React from 'react';
import { Box, Typography, Container } from '@mui/material';

export const SignalsPage: React.FC = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom>
        Trading Signals
      </Typography>
      <Box sx={{ mt: 4 }}>
        <Typography variant="body1" color="text.secondary">
          All trading signals are displayed on the main dashboard.
        </Typography>
      </Box>
    </Container>
  );
}; 