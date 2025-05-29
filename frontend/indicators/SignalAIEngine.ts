/**
 * AI Signal Engine: Combines multiple ML indicators into unified signal object
 */
import { knnClassifyTrend } from '@/indicators/KNNTrendClassifier';
import { calculateTrendChannel } from '@/indicators/MLTrendChannel';
import { getSupportResistanceLevels } from '@/indicators/ClusteredSupportResistance';
import { getAdaptiveOscillatorScore } from '@/indicators/AdaptiveOscillator';

export function generateAISignal(data: { time: string, price: number }[]) {
  const prices = data.map(d => d.price);
  const channel = calculateTrendChannel(data);
  const trend = knnClassifyTrend(prices);
  const levels = getSupportResistanceLevels(prices);

  const oscillatorScore = getAdaptiveOscillatorScore([
    prices.slice(-14), // simulate RSI input
    prices.slice(-14), // simulate MACD input
    prices.slice(-14)  // simulate CCI input
  ]);

  const confidence = oscillatorScore > 70
    ? '++'
    : oscillatorScore > 50
    ? '+'
    : oscillatorScore > 30
    ? '~'
    : '-';

  return {
    trend,
    channel,
    supportResistance: levels,
    oscillatorScore,
    confidence
  };
}
