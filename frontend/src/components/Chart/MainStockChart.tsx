import React, { useEffect, useState } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Line,
  Bar,
  Area,
  ReferenceLine,
  ReferenceArea,
  Label,
  Scatter
} from 'recharts';
import { timeParse } from 'd3-time-format';

interface ChartProps {
  symbol: string;
  timeframe: string;
  chartStyle: 'candlestick' | 'line';
}

// Mock OHLCV data for demonstration
const MOCK_DATA = [
  { date: '2024-06-10T09:30:00.000Z', open: 180, high: 185, low: 178, close: 184, volume: 1200000 },
  { date: '2024-06-11T09:30:00.000Z', open: 184, high: 188, low: 182, close: 187, volume: 1100000 },
  { date: '2024-06-12T09:30:00.000Z', open: 187, high: 190, low: 185, close: 189, volume: 1300000 },
  { date: '2024-06-13T09:30:00.000Z', open: 189, high: 192, low: 188, close: 191, volume: 1400000 },
  { date: '2024-06-14T09:30:00.000Z', open: 191, high: 195, low: 190, close: 194, volume: 1500000 },
];

const parseDate = timeParse('%Y-%m-%dT%H:%M:%S.%LZ');

// Explicitly type the custom candlestick shape props
const CandlestickShape = (props: any) => {
  const { x, y, width, height, payload } = props;
  const lowY = y + height;
  const highY = y;
  const openY = y + ((payload.high - payload.open) / (payload.high - payload.low)) * height;
  const closeY = y + ((payload.high - payload.close) / (payload.high - payload.low)) * height;
  const color = payload.close > payload.open ? '#00C49F' : '#FF4C4C';
  return (
    <g>
      {/* Wick */}
      <line x1={x + width / 2} x2={x + width / 2} y1={highY} y2={lowY} stroke={color} strokeWidth={2} />
      {/* Body */}
      <rect
        x={x + width / 4}
        y={Math.min(openY, closeY)}
        width={width / 2}
        height={Math.abs(closeY - openY) || 2}
        fill={color}
        stroke={color}
      />
    </g>
  );
};

export const MainStockChart: React.FC<ChartProps> = ({ symbol, timeframe, chartStyle }) => {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    setTimeout(() => {
      setData(
        MOCK_DATA.map((d) => ({
          ...d,
          date: parseDate(d.date),
          dateStr: d.date.slice(0, 10),
        }))
      );
      setLoading(false);
    }, 500);
  }, [symbol, timeframe]);

  if (loading) return <div>Loading chart...</div>;

  return (
    <ResponsiveContainer width="100%" height={500}>
      <ComposedChart data={data} margin={{ top: 20, right: 40, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="dateStr" tickFormatter={(d) => d} />
        <YAxis domain={[(dataMin: any) => Math.floor(dataMin * 0.98), (dataMax: any) => Math.ceil(dataMax * 1.02)]} />
        <Tooltip
          labelFormatter={(label) => `Date: ${label}`}
          formatter={(value: any, name: string) => [`$${value}`, name]}
        />
        {/* Candlestick style */}
        {chartStyle === 'candlestick' && (
          <Bar
            dataKey="high"
            fill="#8884d8"
            shape={CandlestickShape}
            barSize={16}
          />
        )}
        {/* Line style */}
        {chartStyle === 'line' && <Line type="monotone" dataKey="close" stroke="#8884d8" dot={false} strokeWidth={2} />}
        {/* Overlays: Entry/Exit, TP/SL, Trendline, Signals (placeholders) */}
        {/* <ReferenceLine x="2024-06-12" label="Entry" stroke="#00C49F" strokeDasharray="3 3" /> */}
        {/* <ReferenceArea x1="2024-06-13" x2="2024-06-14" label="TP Zone" stroke="#00C49F" fill="#00C49F" fillOpacity={0.1} /> */}
        {/* <Line dataKey="trendline" stroke="#FFD700" dot={false} strokeWidth={2} /> */}
        {/* <Scatter data={signalMarkers} shape={<CustomSignalIcon />} /> */}
      </ComposedChart>
    </ResponsiveContainer>
  );
}; 