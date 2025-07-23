// Custom date adapter setup for Chart.js
import 'chartjs-adapter-date-fns';
import { enUS } from 'date-fns/locale';

// Force locale configuration
const dateAdapterOptions = {
  locale: enUS,
};

// Export the configuration
export { dateAdapterOptions };
