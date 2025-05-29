/**
 * KNN Classifier for trend direction prediction
 */
export function knnClassifyTrend(prices: number[], k = 5): 'up' | 'down' | 'neutral' {
  const n = prices.length;
  if (n < k + 1) return 'neutral';

  const recent = prices[n - 1];
  const diffs = prices.slice(n - k - 1, n - 1).map(p => Math.abs(p - recent));

  const avg = diffs.reduce((a, b) => a + b, 0) / k;
  if (recent > prices[n - 2] + avg) return 'up';
  if (recent < prices[n - 2] - avg) return 'down';
  return 'neutral';
}
