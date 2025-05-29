/**
 * K-means simulated price clustering for support/resistance
 */
export function getSupportResistanceLevels(prices: number[]): number[] {
  if (prices.length < 10) return [];

  const sorted = [...prices].sort((a, b) => a - b);
  const clusters = [sorted[5], sorted[Math.floor(sorted.length / 2)], sorted[sorted.length - 5]];

  return clusters;
}
