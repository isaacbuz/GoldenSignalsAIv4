import { RSI, MACD } from 'technicalindicators';

export function simulateStrategy(data: number[], logic: any) {
  const trades = [];
  const rsi = RSI.calculate({ values: data, period: 14 });
  const macd = MACD.calculate({
    values: data,
    fastPeriod: 12,
    slowPeriod: 26,
    signalPeriod: 9,
    SimpleMAOscillator: false,
    SimpleMASignal: false,
  });

  for (let i = 26; i < data.length; i++) {
    const currentRSI = rsi[i - 14];
    const currentMACD = macd[i - 26];

    if (
      logic.conditions?.['RSI.under'] &&
      currentRSI < logic.conditions['RSI.under'] &&
      logic.conditions['MACD.crossesAboveSignal'] &&
      currentMACD?.MACD > currentMACD?.signal
    ) {
      trades.push({ index: i, price: data[i], action: 'BUY' });
    }
  }

  return trades;
}
