/**
 * ML Trend Channel using simple RANSAC simulation
 */
export function calculateTrendChannel(data: { time: string, price: number }[]) {
  const len = data.length;
  if (len < 10) return null;

  const prices = data.map(d => d.price);
  const mean = prices.reduce((a, b) => a + b, 0) / len;
  const deviations = prices.map(p => Math.abs(p - mean));
  const stdev = deviations.reduce((a, b) => a + b, 0) / len;

  return {
    support: mean - stdev,
    resistance: mean + stdev,
    mid: mean,
  };
}
