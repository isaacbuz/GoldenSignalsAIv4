import React, { useEffect, useState } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import logger from '../services/logger';


export const TestChart: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Test backend connection
    fetch('http://localhost:8000/api/v1/market-data/AAPL/history?interval=5m&period=1d')
      .then(res => res.json())
      .then(result => {
        logger.info('Backend data:', result);
        setData(result);
        setLoading(false);
      })
      .catch(err => {
        logger.error('Backend error:', err);
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <Box sx={{ p: 4, bgcolor: 'background.paper', borderRadius: 2 }}>
      <Typography variant="h5" gutterBottom>
        Chart Component Test
      </Typography>

      {loading && <CircularProgress />}

      {error && (
        <Typography color="error">
          Error: {error}
        </Typography>
      )}

      {data && (
        <Box>
          <Typography variant="h6">Backend is working!</Typography>
          <Typography>Symbol: {data.symbol}</Typography>
          <Typography>Data points: {data.data?.length || 0}</Typography>
          <Typography>
            Latest price: ${data.data?.[0]?.close || 'N/A'}
          </Typography>
        </Box>
      )}
    </Box>
  );
};
