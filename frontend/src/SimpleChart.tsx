import React from 'react';
import { Box, Typography } from '@mui/material';

const SimpleChart: React.FC = () => {
  console.log('SimpleChart rendering');

  return (
    <Box sx={{
      width: '100vw',
      height: '100vh',
      backgroundColor: '#000',
      color: '#fff',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center'
    }}>
      <Typography variant="h2" sx={{ color: '#FFD700', mb: 4 }}>
        GoldenSignalsAI
      </Typography>
      <Box sx={{
        width: '80%',
        height: '60%',
        backgroundColor: '#111',
        border: '1px solid #333',
        borderRadius: 2,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <Typography variant="h4">
          Chart will appear here
        </Typography>
      </Box>
      <Typography sx={{ mt: 2, color: '#666' }}>
        Debug: {new Date().toLocaleTimeString()}
      </Typography>
    </Box>
  );
};

export default SimpleChart;
