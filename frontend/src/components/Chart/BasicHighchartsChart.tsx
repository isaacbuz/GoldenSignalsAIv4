/**
 * BasicHighchartsChart - Most basic Highcharts implementation
 *
 * Simple working chart to test Highcharts integration
 */

import React, { useEffect, useState } from 'react';
import Highcharts from 'highcharts';
import HighchartsReact from 'highcharts-react-official';
import { Box, Typography, Paper } from '@mui/material';

// Mock data for testing
const generateMockData = () => {
  const data = [];
  const now = Date.now();
  for (let i = 100; i >= 0; i--) {
    data.push({
      x: now - i * 60000, // 1 minute intervals
      y: 150 + Math.random() * 10 - 5 + Math.sin(i / 10) * 5
    });
  }
  return data;
};

export const BasicHighchartsChart: React.FC = () => {
  const [chartOptions, setChartOptions] = useState<Highcharts.Options>({
    title: {
      text: 'AAPL Stock Price',
      style: {
        color: '#ffffff'
      }
    },
    chart: {
      backgroundColor: '#16181d',
      style: {
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
      }
    },
    xAxis: {
      type: 'datetime',
      gridLineColor: '#2a2d37',
      labels: {
        style: {
          color: '#8b92a8'
        }
      }
    },
    yAxis: {
      title: {
        text: 'Price ($)',
        style: {
          color: '#8b92a8'
        }
      },
      gridLineColor: '#2a2d37',
      labels: {
        style: {
          color: '#8b92a8'
        }
      }
    },
    series: [{
      type: 'line',
      name: 'Price',
      data: generateMockData(),
      color: '#00d4ff',
      lineWidth: 2
    }],
    credits: {
      enabled: false
    },
    legend: {
      itemStyle: {
        color: '#8b92a8'
      }
    }
  });

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setChartOptions(prevOptions => ({
        ...prevOptions,
        series: [{
          type: 'line',
          name: 'Price',
          data: generateMockData(),
          color: '#00d4ff',
          lineWidth: 2
        }]
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <Box sx={{
      width: '100%',
      height: '100vh',
      backgroundColor: '#0e0f14',
      display: 'flex',
      flexDirection: 'column'
    }}>
      <Paper sx={{
        m: 2,
        p: 2,
        backgroundColor: '#16181d',
        borderRadius: 2,
        boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
      }}>
        <Typography variant="h5" sx={{ color: '#ffffff', mb: 2 }}>
          Basic Highcharts Test
        </Typography>
      </Paper>

      <Box sx={{
        flex: 1,
        m: 2,
        backgroundColor: '#16181d',
        borderRadius: 2,
        overflow: 'hidden'
      }}>
        <HighchartsReact
          highcharts={Highcharts}
          options={chartOptions}
          containerProps={{ style: { width: '100%', height: '100%' } }}
        />
      </Box>
    </Box>
  );
};
