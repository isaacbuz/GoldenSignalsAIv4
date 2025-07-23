// Initialize Chart.js with all required components
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

// Import date-fns adapter
import 'chartjs-adapter-date-fns';
import logger from '../services/logger';


// Register components
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
  RadialLinearScale
);

// Test that the adapter is loaded
logger.info('Chart.js adapters:', ChartJS.adapters);

export { ChartJS };
