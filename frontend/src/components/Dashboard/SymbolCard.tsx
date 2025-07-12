import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Pause, AlertCircle, CheckCircle, Zap } from 'lucide-react';

interface SymbolCardProps {
  symbol: string;
  price: number;
  change: number;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  divergenceCount: number;
  consensus: 'strong' | 'moderate' | 'weak';
  onClick: () => void;
}

const SymbolCard: React.FC<SymbolCardProps> = ({
  symbol,
  price,
  change,
  signal,
  confidence,
  divergenceCount,
  consensus,
  onClick
}) => {
  const getSignalIcon = () => {
    switch (signal) {
      case 'BUY': return <TrendingUp className="w-5 h-5" />;
      case 'SELL': return <TrendingDown className="w-5 h-5" />;
      case 'HOLD': return <Pause className="w-5 h-5" />;
    }
  };

  const getSignalColor = () => {
    switch (signal) {
      case 'BUY': return 'text-green-500';
      case 'SELL': return 'text-red-500';
      case 'HOLD': return 'text-yellow-500';
    }
  };

  const getConsensusIcon = () => {
    if (divergenceCount > 2) {
      return (
        <motion.div
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ repeat: Infinity, duration: 2 }}
          className="flex items-center gap-1 text-purple-400"
        >
          <AlertCircle className="w-4 h-4" />
          <span className="text-xs">High Uncertainty</span>
        </motion.div>
      );
    } else if (divergenceCount > 0) {
      return (
        <div className="flex items-center gap-1 text-cyan-400">
          <Zap className="w-4 h-4" />
          <span className="text-xs">{divergenceCount} Divergences</span>
        </div>
      );
    } else if (consensus === 'strong') {
      return (
        <div className="flex items-center gap-1 text-green-400">
          <CheckCircle className="w-4 h-4" />
          <span className="text-xs">Strong Consensus</span>
        </div>
      );
    }
    return null;
  };

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className="bg-surface border border-surface-light rounded-lg p-4 cursor-pointer transition-all hover:border-cyan-500/50"
    >
      <div className="flex justify-between items-start mb-2">
        <h3 className="text-lg font-bold text-text-primary">{symbol}</h3>
        <div className={`flex items-center gap-1 ${getSignalColor()}`}>
          {getSignalIcon()}
          <span className="font-semibold">{signal}</span>
        </div>
      </div>

      <div className="space-y-2">
        <div>
          <p className="text-xl font-semibold text-text-primary">
            ${price.toFixed(2)}
          </p>
          <p className={`text-sm ${change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {change >= 0 ? '+' : ''}{change.toFixed(2)}%
          </p>
        </div>

        <div className="relative">
          <div className="w-full bg-surface-light rounded-full h-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${confidence}%` }}
              transition={{ duration: 0.5 }}
              className={`h-full rounded-full ${
                confidence > 70 ? 'bg-green-500' : 
                confidence > 50 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
            />
          </div>
          <p className="text-xs text-text-secondary mt-1">{confidence}% confidence</p>
        </div>

        {getConsensusIcon()}
      </div>
    </motion.div>
  );
};

export default SymbolCard; 