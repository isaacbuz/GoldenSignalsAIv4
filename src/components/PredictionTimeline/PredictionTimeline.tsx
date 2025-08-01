import { Line } from 'react-chartjs-2';
import 'chart.js/auto';
// Add Chart.js line chart with datasets for predicted and actual.
const chartData = {
    labels: ['Jan', 'Feb' /* ... */],
    datasets: [
        { label: 'Predicted', data: [150, 155 /* ... */], borderColor: '#FFD700' },
        { label: 'Actual', data: [152, 154 /* ... */], borderColor: '#FFFFFF' },
    ],
};
const PredictionTimeline = () => {
    return <Line data={chartData} />;
};
export default PredictionTimeline;
