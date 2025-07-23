import React, { useEffect, useState } from 'react';
import { Box, Typography, Chip, Paper } from '@mui/material';
import { fetchHistoricalData } from '../services/backendMarketDataService';
import logger from '../services/logger';


export const LiveDataMonitor: React.FC = () => {
  const [dataStatus, setDataStatus] = useState({
    lastUpdate: null as Date | null,
    dataPoints: 0,
    latestPrice: 0,
    isLive: false,
    error: null as string | null,
  });

  useEffect(() => {
    const checkData = async () => {
      try {
        const response = await fetchHistoricalData('AAPL', '5m');
        if (response.data && response.data.length > 0) {
          const latest = response.data[response.data.length - 1];
          setDataStatus({
            lastUpdate: new Date(),
            dataPoints: response.data.length,
            latestPrice: latest.close,
            isLive: true,
            error: null,
          });
          logger.info('Live data received:', {
            symbol: 'AAPL',
            dataPoints: response.data.length,
            latestPrice: latest.close,
            latestTime: new Date(latest.time * 1000).toLocaleTimeString(),
          });
        }
      } catch (error) {
        setDataStatus(prev => ({
          ...prev,
          isLive: false,
          error: error.message,
        }));
      }
    };

    // Initial check
    checkData();

    // Check every 5 seconds
    const interval = setInterval(checkData, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <Paper
      sx={{
        position: 'fixed',
        bottom: 20,
        right: 20,
        p: 2,
        backgroundColor: 'background.paper',
        zIndex: 9999,
        minWidth: 300,
      }}
    >
      <Typography variant="h6" gutterBottom>
        Live Data Monitor
      </Typography>

      <Box display="flex" alignItems="center" gap={1} mb={1}>
        <Chip
          label={dataStatus.isLive ? 'LIVE' : 'OFFLINE'}
          color={dataStatus.isLive ? 'success' : 'error'}
          size="small"
        />
        {dataStatus.lastUpdate && (
          <Typography variant="caption">
            Updated: {dataStatus.lastUpdate.toLocaleTimeString()}
          </Typography>
        )}
      </Box>

      {dataStatus.isLive && (
        <>
          <Typography variant="body2">
            Data Points: {dataStatus.dataPoints}
          </Typography>
          <Typography variant="body2">
            Latest Price: ${dataStatus.latestPrice.toFixed(2)}
          </Typography>
        </>
      )}

      {dataStatus.error && (
        <Typography variant="body2" color="error">
          Error: {dataStatus.error}
        </Typography>
      )}
    </Paper>
  );
};
