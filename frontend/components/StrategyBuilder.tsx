import { useState } from 'react';

import { motion } from 'framer-motion';

export default function StrategyBuilder({ onUpdate }: { onUpdate: (strategy: any) => void }) {
  const [rsiThreshold, setRsiThreshold] = useState(30);
  const [macdEnabled, setMacdEnabled] = useState(true);

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      className="p-5 bg-bgPanel rounded-xl shadow font-sans border border-borderSoft text-white space-y-4 animate-fadeIn"
      aria-label="Strategy builder panel"
    >
      {/*
        StrategyBuilder lets users adjust strategy parameters (RSI, MACD) and apply them.
      */}
      <h3 className="text-lg font-bold text-accentGreen mb-2">Strategy Settings</h3>
      <label className="block mb-2">
        <span className="text-sm text-gray-400">RSI Threshold:</span>
        <input
          type="number"
          value={rsiThreshold}
          onChange={(e) => setRsiThreshold(+e.target.value)}
          className="w-full mt-1 rounded-lg p-2 bg-bgDark border border-borderSoft text-white focus:outline-none focus:ring-2 focus:ring-accentBlue"
        />
      </label>
      <label className="flex items-center gap-2 mb-2">
        <input type="checkbox" checked={macdEnabled} onChange={() => setMacdEnabled(!macdEnabled)} />
        <span className="text-sm text-gray-400">Enable MACD Filter</span>
      </label>
      <button
        onClick={() => onUpdate({ rsiThreshold, macdEnabled })}
        className="bg-accentPink w-full rounded-lg py-2 font-bold shadow-neon hover:shadow-glow transition"
      >
        Apply Strategy
      </button>
    </motion.div>
  );
}
