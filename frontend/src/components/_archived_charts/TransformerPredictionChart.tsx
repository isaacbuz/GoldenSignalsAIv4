/**
 * Transformer Prediction Chart - Visualizes Transformer model's market predictions
 * Implements the architecture from the tutorial video
 */

import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartOptions,
  ChartData
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import annotationPlugin from 'chartjs-plugin-annotation';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  annotationPlugin
);

interface TransformerPredictionData {
  symbol: string;
  currentPrice: number;
  prediction: {
    targetPrice: number;
    confidence: number;
    timeframe: string;
  };
  historicalData: {
    timestamps: string[];
    prices: number[];
    predictions: number[];
  };
  technicalIndicators: {
    rsi: number[];
    bollingerHigh: number[];
    bollingerLow: number[];
    ma20: number[];
    ma20Slope: number[];
  };
}

interface TransformerPredictionChartProps {
  data: TransformerPredictionData;
  variant?: 'detailed' | 'compact';
  showAnimation?: boolean;
}

const TransformerPredictionChart: React.FC<TransformerPredictionChartProps> = ({
  data,
  variant = 'detailed',
  showAnimation = true
}) => {
  const chartRef = useRef<any>(null);

  // Calculate prediction direction and percentage
  const priceChange = data.prediction.targetPrice - data.currentPrice;
  const priceChangePercent = (priceChange / data.currentPrice) * 100;
  const isUpward = priceChange > 0;

  // Price chart data
  const priceChartData: ChartData<'line'> = {
    labels: data.historicalData.timestamps,
    datasets: [
      {
        label: 'Actual Price',
        data: data.historicalData.prices,
        borderColor: '#3B82F6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.4,
        fill: true
      },
      {
        label: 'Transformer Predictions',
        data: data.historicalData.predictions,
        borderColor: '#10B981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        borderWidth: 2,
        borderDash: [5, 5],
        pointRadius: 0,
        tension: 0.4,
        fill: false
      },
      {
        label: 'Bollinger High',
        data: data.technicalIndicators.bollingerHigh,
        borderColor: '#EF4444',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        borderWidth: 1,
        borderDash: [3, 3],
        pointRadius: 0,
        tension: 0.4,
        fill: false
      },
      {
        label: 'Bollinger Low',
        data: data.technicalIndicators.bollingerLow,
        borderColor: '#10B981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        borderWidth: 1,
        borderDash: [3, 3],
        pointRadius: 0,
        tension: 0.4,
        fill: false
      },
      {
        label: 'MA20',
        data: data.technicalIndicators.ma20,
        borderColor: '#FBBF24',
        backgroundColor: 'rgba(251, 191, 36, 0.1)',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.4,
        fill: false
      }
    ]
  };

  // Price chart options
  const priceChartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: variant === 'detailed',
        position: 'top' as const,
        labels: {
          color: '#fff',
          usePointStyle: true,
          pointStyle: 'circle'
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: '#333',
        borderWidth: 1
      },
      annotation: {
        annotations: {
          targetPrice: {
            type: 'line',
            yMin: data.prediction.targetPrice,
            yMax: data.prediction.targetPrice,
            borderColor: isUpward ? '#10B981' : '#EF4444',
            borderWidth: 2,
            label: {
              content: `Target: $${data.prediction.targetPrice.toFixed(2)}`,
              display: true,
              position: 'end'
            }
          }
        }
      }
    },
    scales: {
      x: {
        display: variant === 'detailed',
        grid: {
          display: false
        }
      },
      y: {
        display: variant === 'detailed',
        position: 'right',
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      }
    },
    animation: showAnimation ? {
      duration: 2000,
      easing: 'easeInOutQuart'
    } : false
  };

  if (variant === 'compact') {
    return (
      <motion.div
        className="bg-gray-800 rounded-lg p-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-semibold text-white">{data.symbol}</h3>
          <span className={`text-sm font-medium ${isUpward ? 'text-green-400' : 'text-red-400'}`}>
            {isUpward ? '↑' : '↓'} {Math.abs(priceChangePercent).toFixed(2)}%
          </span>
        </div>

        <div className="h-32">
          <Line data={priceChartData} options={priceChartOptions} />
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      className="bg-gray-800 rounded-lg p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-xl font-semibold text-white">{data.symbol}</h3>
          <p className="text-gray-400 text-sm">Transformer Model Prediction</p>
        </div>
        <div className="text-right">
          <p className="text-white text-lg">${data.currentPrice.toFixed(2)}</p>
          <span className={`text-sm font-medium ${isUpward ? 'text-green-400' : 'text-red-400'}`}>
            {isUpward ? '↑' : '↓'} {Math.abs(priceChangePercent).toFixed(2)}%
          </span>
        </div>
      </div>

      <div className="h-96 mb-4">
        <Line ref={chartRef} data={priceChartData} options={priceChartOptions} />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-700 rounded-lg p-4">
          <h4 className="text-white font-medium mb-2">Prediction Details</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Target Price</span>
              <span className="text-white">${data.prediction.targetPrice.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Confidence</span>
              <span className="text-white">{data.prediction.confidence.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Timeframe</span>
              <span className="text-white">{data.prediction.timeframe}</span>
            </div>
          </div>
        </div>

        <div className="bg-gray-700 rounded-lg p-4">
          <h4 className="text-white font-medium mb-2">Technical Indicators</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">RSI</span>
              <span className="text-white">{data.technicalIndicators.rsi[data.technicalIndicators.rsi.length - 1].toFixed(1)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">MA20 Slope</span>
              <span className="text-white">{data.technicalIndicators.ma20Slope[data.technicalIndicators.ma20Slope.length - 1].toFixed(4)}</span>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default TransformerPredictionChart;
