import React from 'react';
import { motion } from 'framer-motion';

interface SentimentGaugeProps {
  sentiment: 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  showMomentum?: boolean;
  momentum?: number;
}

const SentimentGauge: React.FC<SentimentGaugeProps> = ({
  sentiment,
  confidence,
  showMomentum = false,
  momentum = 0
}) => {
  const getRotation = () => {
    if (sentiment === 'bearish') return -45;
    if (sentiment === 'bullish') return 45;
    return 0;
  };

  const getSentimentColor = () => {
    if (sentiment === 'bearish') return '#EF4444';
    if (sentiment === 'bullish') return '#10B981';
    return '#6B7280';
  };

  return (
    <div className="flex flex-col items-center">
      {/* Gauge */}
      <div className="relative w-40 h-20 mb-4">
        <svg viewBox="0 0 200 100" className="w-full h-full">
          {/* Background arc */}
          <path
            d="M 20 80 A 60 60 0 0 1 180 80"
            fill="none"
            stroke="#1E293B"
            strokeWidth="20"
          />
          
          {/* Colored segments */}
          <path
            d="M 20 80 A 60 60 0 0 1 70 40"
            fill="none"
            stroke="#EF4444"
            strokeWidth="20"
            opacity="0.3"
          />
          <path
            d="M 70 40 A 60 60 0 0 1 130 40"
            fill="none"
            stroke="#6B7280"
            strokeWidth="20"
            opacity="0.3"
          />
          <path
            d="M 130 40 A 60 60 0 0 1 180 80"
            fill="none"
            stroke="#10B981"
            strokeWidth="20"
            opacity="0.3"
          />
          
          {/* Needle */}
          <motion.g
            initial={{ rotate: 0 }}
            animate={{ rotate: getRotation() }}
            transition={{ type: "spring", stiffness: 50 }}
            style={{ transformOrigin: '100px 80px' }}
          >
            <line
              x1="100"
              y1="80"
              x2="100"
              y2="30"
              stroke={getSentimentColor()}
              strokeWidth="4"
            />
            <circle cx="100" cy="80" r="6" fill={getSentimentColor()} />
          </motion.g>
        </svg>
        
        {/* Labels */}
        <span className="absolute left-0 bottom-0 text-xs text-red-500">🔴</span>
        <span className="absolute left-1/2 -translate-x-1/2 top-0 text-xs text-gray-500">⚪</span>
        <span className="absolute right-0 bottom-0 text-xs text-green-500">🟢</span>
      </div>
      
      {/* Sentiment Text */}
      <div className="text-center">
        <p className={`text-lg font-bold ${
          sentiment === 'bullish' ? 'text-green-500' :
          sentiment === 'bearish' ? 'text-red-500' :
          'text-gray-500'
        }`}>
          {sentiment.toUpperCase()} {Math.round(confidence * 100)}%
        </p>
      </div>
      
      {/* Momentum Indicator */}
      {showMomentum && momentum !== undefined && (
        <div className="mt-2 flex items-center gap-1">
          <span className="text-xs text-text-secondary">Momentum:</span>
          {Array.from({ length: Math.abs(momentum) }).map((_, i) => (
            <motion.span
              key={i}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: i * 0.1 }}
              className={momentum > 0 ? 'text-green-500' : 'text-red-500'}
            >
              {momentum > 0 ? '↗️' : '↘️'}
            </motion.span>
          ))}
        </div>
      )}
      
      {/* Confidence Bar */}
      <div className="w-full mt-2">
        <div className="text-xs text-text-secondary mb-1">Confidence</div>
        <div className="w-full bg-surface-light rounded-full h-2">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${confidence * 100}%` }}
            transition={{ duration: 0.5 }}
            className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-cyan-400"
          />
        </div>
      </div>
    </div>
  );
};

export default SentimentGauge; 