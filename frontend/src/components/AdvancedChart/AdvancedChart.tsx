import React, { useEffect, useRef } from 'react';
import Chart from 'react-apexcharts';
import { Box, Select, MenuItem } from '@mui/material';

import { ChartData } from '../../../types/charts';

interface AdvancedChartProps {
  data: ChartData[];
  signals: any[];
  timeframe: string;
  onTimeframeChange: (tf: string) => void;
}

export const AdvancedChart: React.FC<AdvancedChartProps> = ({ data, signals, timeframe, onTimeframeChange }) => {
  const chartRef = useRef(null);

  const config = {
    series: [{
      name: 'candlestick',
      type: 'candlestick',
      data: data.map(d => ({ x: d.time, y: [d.open, d.high, d.low, d.close] }))
    }, {
      name: 'volume',
      type: 'column',
      data: data.map(d => ({ x: d.time, y: d.volume }))
    }],
    options: {
      chart: { type: 'candlestick', height: 400 },
      xaxis: { type: 'datetime' },
      yaxis: [{ title: { text: 'Price' } }, { opposite: true, title: { text: 'Volume' } }],
      markers: signals.map(s => ({ x: s.timestamp, y: s.price, label: s.action })),
    }
  };

  return (
    <Box>
      <Select value={timeframe} onChange={(e) => onTimeframeChange(e.target.value)}>
        <MenuItem value="1m">1m</MenuItem>
        <MenuItem value="5m">5m</MenuItem>
        <MenuItem value="1h">1h</MenuItem>
        <MenuItem value="1d">1d</MenuItem>
      </Select>
      <Chart options={config.options} series={config.series} type="candlestick" height={400} />
    </Box>
  );
};

export default AdvancedChart;