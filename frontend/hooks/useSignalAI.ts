import { RSI, MACD } from 'technicalindicators';

export function useSignalAI(data: { time: string, price: number }[]) {
  if (!Array.isArray(data) || data.length < 30) return null;

  const prices = data.map((d) => d.price);
  const latest = prices[prices.length - 1];

  // RSI Logic
  const rsi = RSI.calculate({ period: 14, values: prices });
  const lastRSI = rsi[rsi.length - 1];

  // MACD Logic
  const macd = MACD.calculate({
    values: prices,
    fastPeriod: 12,
    slowPeriod: 26,
    signalPeriod: 9,
    SimpleMAOscillator: false,
    SimpleMASignal: false,
  });
  const lastMACD = macd[macd.length - 1];

  if (!lastRSI || !lastMACD) return null;

  const signal = {
    type: '',
    confidence: 0,
    entry: latest,
    explanation: '',
  };

  if (lastRSI < 30 && lastMACD.MACD > lastMACD.signal) {
    signal.type = 'Bullish RSI + MACD';
    signal.confidence = 85;
    signal.explanation = `RSI is oversold (${lastRSI.toFixed(1)}) & MACD is crossing bullish.`;
  } else if (lastRSI > 70 && lastMACD.MACD < lastMACD.signal) {
    signal.type = 'Bearish RSI + MACD';
    signal.confidence = 82;
    signal.explanation = `RSI is overbought (${lastRSI.toFixed(1)}) & MACD is crossing bearish.`;
  }

  return signal.type ? signal : null;
}
