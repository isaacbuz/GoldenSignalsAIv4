import React from 'react';
import { getAdaptiveOscillatorScore } from '@/indicators/AdaptiveOscillator';

export function AIScoreDisplay({ data }: { data: { price: number }[] }) {
  // Simulate indicator arrays for demo
  const prices = data?.map(d => d.price) ?? [];
  const oscillatorScore = getAdaptiveOscillatorScore([
    prices.slice(-14), // RSI
    prices.slice(-14), // MACD
    prices.slice(-14), // CCI
  ]);
  const confidence = oscillatorScore > 70
    ? '++'
    : oscillatorScore > 50
    ? '+'
    : oscillatorScore > 30
    ? '~'
    : '-';
  return (
    <div className="mt-3 flex items-center gap-2">
      <span className="text-xs text-gray-400">AI Oscillator Score:</span>
      <span className="font-bold text-accentBlue">{oscillatorScore}</span>
      <span className="ml-2 text-xs text-gray-400">Confidence:</span>
      <span className="font-bold text-accentGreen">{confidence}</span>
    </div>
  );
}
