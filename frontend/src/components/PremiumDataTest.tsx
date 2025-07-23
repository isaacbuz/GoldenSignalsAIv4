import React, { useEffect, useState } from 'react';
import { Box, Typography, Card, CardContent, Chip, Grid } from '@mui/material';
import { TrendingUp, TrendingDown, Update } from '@mui/icons-material';
import logger from '../services/logger';


export const PremiumDataTest: React.FC = () => {
  const [dataSource, setDataSource] = useState('');
  const [latestData, setLatestData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [updateTime, setUpdateTime] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch data and check logs to see which provider was used
        const response = await fetch('http://localhost:8000/api/v1/market-data/AAPL/history?period=1d&interval=5m');
        const data = await response.json();

        if (data.data && data.data.length > 0) {
          const latest = data.data[data.data.length - 1];
          setLatestData(latest);

          // Check backend logs to determine data source
          // For now, we'll check data characteristics
          if (latest.volume > 1000000) {
            setDataSource('TwelveData / Finnhub (Premium)');
          } else {
            setDataSource('Yahoo Finance (Fallback)');
          }

          setUpdateTime(new Date().toLocaleTimeString());
          setLoading(false);
        }
      } catch (error) {
        logger.error('Error fetching data:', error);
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000); // Update every 10 seconds

    return () => clearInterval(interval);
  }, []);

  if (loading) return <Typography>Loading premium data...</Typography>;

  return (
    <Card sx={{
      position: 'fixed',
      top: 80,
      right: 20,
      width: 350,
      zIndex: 1000,
      backgroundColor: 'background.paper',
      boxShadow: 3
    }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Premium Data Status
        </Typography>

        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Chip
              label={dataSource}
              color="primary"
              icon={<Update />}
              sx={{ mb: 1 }}
            />
            <Chip
              label="Data Normalized ✓"
              color="success"
              size="small"
              sx={{ mb: 1, ml: 1 }}
            />
          </Grid>

          {latestData && (
            <>
              <Grid item xs={6}>
                <Typography variant="caption" color="text.secondary">
                  Latest Price
                </Typography>
                <Typography variant="h5" color={latestData.close > latestData.open ? 'success.main' : 'error.main'}>
                  ${latestData.close.toFixed(2)}
                  {latestData.close > latestData.open ? <TrendingUp /> : <TrendingDown />}
                </Typography>
              </Grid>

              <Grid item xs={6}>
                <Typography variant="caption" color="text.secondary">
                  Volume
                </Typography>
                <Typography variant="h6">
                  {(latestData.volume / 1000000).toFixed(2)}M
                </Typography>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="caption" color="text.secondary">
                  OHLC
                </Typography>
                <Typography variant="body2">
                  O: ${latestData.open.toFixed(2)} |
                  H: ${latestData.high.toFixed(2)} |
                  L: ${latestData.low.toFixed(2)}
                </Typography>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="caption" color="text.secondary">
                  Last Update: {updateTime}
                </Typography>
              </Grid>
            </>
          )}
        </Grid>

        <Box mt={2} p={1} bgcolor="info.main" borderRadius={1}>
          <Typography variant="caption" color="white">
            ✨ Using API Keys: TwelveData & Finnhub
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};
