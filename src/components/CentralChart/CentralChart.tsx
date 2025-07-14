import React, { useEffect, useRef, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  ChartData,
  ChartOptions,
  Point
} from 'chart.js';
import { Chart } from 'react-chartjs-2';
import annotationPlugin, { AnnotationOptions } from 'chartjs-plugin-annotation';
import 'chartjs-adapter-date-fns';
import { Box, CircularProgress, Alert, Paper } from '@mui/material';
import TradeSearch from '../TradeSearch/TradeSearch';
import ProphetOrb from '../ProphetOrb/ProphetOrb';
import OptionsPanel from '../OptionsPanel/OptionsPanel';
import AnalysisLegend from '../AnalysisLegend/AnalysisLegend';
import { apiClient } from '../../services/api';
import './CentralChart.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  annotationPlugin
);

interface CandlestickData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface PredictionData {
  time: string;
  value: number;
  confidence?: number;
}

interface AIAnalysis {
  entries: Array<{
    price: number;
    time?: string;
    optionDetails: string;
    confidence?: number;
  }>;
  profitZones: Array<{
    start: number;
    end: number;
    color: string;
    alpha?: number;
  }>;
  stopLoss: number;
  takeProfit: number;
  rationale: string;
  riskRewardRatio?: number;
  confidence?: number;
}

interface CentralChartProps {
  symbol?: string;
  timeframe?: string;
  onSymbolChange?: (symbol: string) => void;
  onTimeframeChange?: (timeframe: string) => void;
}

const CentralChart: React.FC<CentralChartProps> = ({
  symbol = 'AAPL',
  timeframe = '1d',
  onSymbolChange,
  onTimeframeChange
}) => {
  const chartRef = useRef<ChartJS>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<CandlestickData[]>([]);
  const [predictions, setPredictions] = useState<PredictionData[]>([]);
  const [aiAnalysis, setAiAnalysis] = useState<AIAnalysis | null>(null);

  // Fetch data when symbol or timeframe changes
  useEffect(() => {
    fetchChartData();
  }, [symbol, timeframe]);

  const fetchChartData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Fetch historical data
      const response = await fetch(`/api/v1/market-data/${symbol}/historical?period=${timeframe}`);
      if (!response.ok) throw new Error('Failed to fetch market data');
      
      const marketData = await response.json();
      setData(marketData.data || []);
      
      // Fetch AI analysis
      const analysisResponse = await fetch(`/api/v1/signals/${symbol}/insights`);
      if (analysisResponse.ok) {
        const analysis = await analysisResponse.json();
        setAiAnalysis(analysis.aiAnalysis);
        setPredictions(analysis.predictions || []);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
      // Use mock data as fallback
      setData(generateMockData());
      setAiAnalysis(generateMockAnalysis());
    } finally {
      setLoading(false);
    }
  };

  const generateMockData = (): CandlestickData[] => {
    const now = new Date();
    const data: CandlestickData[] = [];
    let basePrice = 150;
    
    for (let i = 30; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      
      const change = (Math.random() - 0.5) * 5;
      basePrice += change;
      
      data.push({
        time: date.toISOString(),
        open: basePrice + (Math.random() - 0.5) * 2,
        high: basePrice + Math.random() * 3,
        low: basePrice - Math.random() * 3,
        close: basePrice + (Math.random() - 0.5) * 2,
        volume: Math.floor(Math.random() * 1000000)
      });
    }
    
    return data;
  };

  const generateMockAnalysis = (): AIAnalysis => ({
    entries: [
      { price: 152.5, optionDetails: 'Call $155 Strike', confidence: 0.85 },
      { price: 148.0, optionDetails: 'Put $145 Strike', confidence: 0.72 }
    ],
    profitZones: [
      { start: 155, end: 160, color: '#4CAF50', alpha: 0.2 },
      { start: 145, end: 140, color: '#FF5252', alpha: 0.2 }
    ],
    stopLoss: 145,
    takeProfit: 160,
    rationale: 'Bullish momentum with strong support at $145',
    riskRewardRatio: 3.0,
    confidence: 0.82
  });

  const chartData: ChartData<'line' | 'bar'> = {
    labels: data.map(d => d.time),
    datasets: [
      {
        type: 'bar' as const,
        label: 'Price Range',
        data: data.map(d => [d.low, d.high]),
        backgroundColor: data.map(d => d.close > d.open ? '#4CAF5080' : '#FF525280'),
        borderColor: data.map(d => d.close > d.open ? '#4CAF50' : '#FF5252'),
        borderWidth: 1,
      },
      {
        type: 'line' as const,
        label: 'Close Price',
        data: data.map(d => ({ x: d.time, y: d.close })),
        borderColor: '#2196F3',
        backgroundColor: '#2196F380',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1
      }
    ]
  };

  // Add predictions if available
  if (predictions.length > 0) {
    chartData.datasets.push({
      type: 'line' as const,
      label: 'AI Predictions',
      data: predictions.map(p => ({ x: p.time, y: p.value })),
      borderColor: '#FFD700',
      backgroundColor: '#FFD70080',
      borderWidth: 2,
      borderDash: [5, 5],
      pointRadius: 3,
      tension: 0.2
    });
  }

  const chartOptions: ChartOptions<'line' | 'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: `${symbol} - ${timeframe} Chart`,
      },
      annotation: {
        annotations: aiAnalysis ? {
          stopLoss: {
            type: 'line',
            yMin: aiAnalysis.stopLoss,
            yMax: aiAnalysis.stopLoss,
            borderColor: '#FF5252',
            borderWidth: 2,
            borderDash: [6, 6],
            label: {
              display: true,
              content: `Stop Loss: $${aiAnalysis.stopLoss}`,
              position: 'end'
            }
          },
          takeProfit: {
            type: 'line',
            yMin: aiAnalysis.takeProfit,
            yMax: aiAnalysis.takeProfit,
            borderColor: '#4CAF50',
            borderWidth: 2,
            borderDash: [6, 6],
            label: {
              display: true,
              content: `Take Profit: $${aiAnalysis.takeProfit}`,
              position: 'end'
            }
          },
          ...aiAnalysis.profitZones.reduce((acc, zone, idx) => ({
            ...acc,
            [`zone${idx}`]: {
              type: 'box',
              yMin: zone.start,
              yMax: zone.end,
              backgroundColor: zone.color + '40',
              borderColor: zone.color,
              borderWidth: 1,
            }
          }), {}),
          ...aiAnalysis.entries.reduce((acc, entry, idx) => ({
            ...acc,
            [`entry${idx}`]: {
              type: 'point',
              xValue: entry.time || data[data.length - 1]?.time,
              yValue: entry.price,
              backgroundColor: '#2196F3',
              borderColor: '#fff',
              borderWidth: 2,
              radius: 6,
              label: {
                display: true,
                content: entry.optionDetails,
                position: 'top'
              }
            }
          }), {})
        } as Record<string, AnnotationOptions> : {}
      }
    },
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'day',
          displayFormats: {
            day: 'MMM dd'
          }
        },
        title: {
          display: true,
          text: 'Date'
        }
      },
      y: {
        title: {
          display: true,
          text: 'Price ($)'
        },
        position: 'right'
      }
    }
  };

  const handleTradeSubmit = async (symbol: string, options: any) => {
    onSymbolChange?.(symbol);
    await fetchChartData();
  };

  const handleProphetAction = (action: string) => {
    console.log('Prophet action:', action);
    // Implement prophet actions
  };

  const handleOptionsUpdate = (options: any) => {
    console.log('Options update:', options);
    // Implement options panel updates
  };

  return (
    <Box className="central-chart-container" sx={{ position: 'relative', height: '100%', p: 2 }}>
      <Paper elevation={3} sx={{ height: '100%', p: 2, position: 'relative' }}>
        {loading && (
          <Box sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>
            <CircularProgress />
          </Box>
        )}
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <Box sx={{ height: 'calc(100% - 100px)' }}>
          <Chart ref={chartRef} type='bar' data={chartData} options={chartOptions} />
        </Box>
        
        <TradeSearch onSubmit={handleTradeSubmit} />
        <ProphetOrb 
          onAction={handleProphetAction}
          analysis={aiAnalysis}
          confidence={aiAnalysis?.confidence}
        />
        <OptionsPanel 
          onUpdate={handleOptionsUpdate}
          currentAnalysis={aiAnalysis}
        />
        {aiAnalysis && (
          <AnalysisLegend 
            analysis={aiAnalysis}
            symbol={symbol}
          />
        )}
      </Paper>
    </Box>
  );
};

export default CentralChart;