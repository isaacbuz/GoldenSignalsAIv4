/**
 * Adaptive Oscillator: Combines RSI, MACD, CCI into a weighted ensemble
 */
export function getAdaptiveOscillatorScore(values: number[][]): number {
  const [rsi, macd, cci] = values;
  if (!rsi || !macd || !cci) return 0;

  const rsiScore = rsi[rsi.length - 1];
  const macdScore = macd[macd.length - 1];
  const cciScore = cci[cci.length - 1];

  const weightedAverage = (rsiScore * 0.3 + macdScore * 0.4 + cciScore * 0.3);
  return Math.min(Math.max(weightedAverage, -100), 100);
}
