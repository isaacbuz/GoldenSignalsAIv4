// Import Chart.js components
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
  Filler,
  TimeScale,
  TimeSeriesScale,
  ArcElement,
  RadialLinearScale,
} from 'chart.js';

// IMPORTANT: Import date adapter BEFORE registering components
import './chartDateAdapter';

// Import plugins
import zoomPlugin from 'chartjs-plugin-zoom';
import annotationPlugin from 'chartjs-plugin-annotation';

// Register all Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  TimeScale,
  TimeSeriesScale,
  ArcElement,
  RadialLinearScale,
  zoomPlugin,
  annotationPlugin
);

// Set default options
ChartJS.defaults.responsive = true;
ChartJS.defaults.maintainAspectRatio = false;
ChartJS.defaults.plugins.legend.display = true;
ChartJS.defaults.plugins.legend.position = 'top';

// Export configured ChartJS
export { ChartJS };

// Export common chart options
export const defaultChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: {
    mode: 'index' as const,
    intersect: false,
  },
  plugins: {
    legend: {
      display: true,
      position: 'top' as const,
    },
    tooltip: {
      enabled: true,
    },
  },
};

// Export time scale options
export const timeScaleOptions = {
  type: 'time' as const,
  time: {
    displayFormats: {
      hour: 'HH:mm',
      day: 'MMM dd',
      week: 'MMM dd',
      month: 'MMM yyyy',
    },
  },
  ticks: {
    source: 'auto' as const,
    autoSkip: true,
    maxRotation: 0,
  },
  adapters: {
    date: {
      locale: 'en-US',
    },
  },
};
