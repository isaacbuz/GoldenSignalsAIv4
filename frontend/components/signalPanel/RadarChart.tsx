import {
  Radar as ReRadar,
  RadarChart as ReRadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer
} from 'recharts';
import { motion } from 'framer-motion';

const data = [
  { factor: 'Sentiment', score: 70 },
  { factor: 'Volatility', score: 60 },
  { factor: 'Forecast', score: 85 },
  { factor: 'Technical', score: 80 },
  { factor: 'Volume', score: 75 },
];

export default function RadarChart() {
  return (
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="bg-bgPanel p-5 rounded-xl shadow-neon font-sans border border-accentBlue text-white mb-2"
    >
      <h3 className="text-xl font-bold text-accentBlue mb-1">Signal Factors</h3>
      <ResponsiveContainer width="100%" height={250}>
        <ReRadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
          <PolarGrid />
          <PolarAngleAxis dataKey="factor" stroke="#94a3b8" />
          <PolarRadiusAxis angle={30} domain={[0, 100]} />
          <ReRadar name="Score" dataKey="score" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.6} />
        </ReRadarChart>
      </ResponsiveContainer>
      <div className="text-xs text-gray-500 mt-3 italic">Radar powered by AI analytics</div>
    </motion.div>
  );
}
