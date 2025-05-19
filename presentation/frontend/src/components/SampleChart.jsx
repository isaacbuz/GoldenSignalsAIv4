import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const data = {
  labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
  datasets: [
    {
      label: 'Price',
      data: [150, 153, 149, 155, 158, 162, 160],
      fill: false,
      backgroundColor: '#FFD700',
      borderColor: '#FFD700',
      pointBackgroundColor: '#232323',
      pointBorderColor: '#FFD700',
      pointRadius: 5,
      tension: 0.4,
      borderWidth: 3,
      shadowOffsetX: 2,
      shadowOffsetY: 2,
      shadowBlur: 8,
      shadowColor: '#C9A100',
    }
  ]
};

const options = {
  responsive: true,
  plugins: {
    legend: {
      display: false
    },
    title: {
      display: true,
      text: 'Sample Price Chart',
      color: '#FFD700',
      font: { size: 20, weight: 'bold' }
    },
    tooltip: {
      backgroundColor: '#232323',
      titleColor: '#FFD700',
      bodyColor: '#F2E9C9',
      borderColor: '#FFD700',
      borderWidth: 1
    }
  },
  scales: {
    x: {
      grid: { color: '#333' },
      ticks: { color: '#FFD700', font: { weight: 'bold' } }
    },
    y: {
      grid: { color: '#333' },
      ticks: { color: '#FFD700', font: { weight: 'bold' } }
    }
  }
};

export default function SampleChart() {
  return (
    <div style={{ background: 'none', borderRadius: 16, boxShadow: '0 2px 12px #FFD70033', padding: 12 }}>
      <Line data={data} options={options} />
    </div>
  );
}
