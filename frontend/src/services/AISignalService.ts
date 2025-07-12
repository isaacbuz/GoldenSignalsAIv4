import { AISignal } from '../types/TradingViewTypes';
import { marketDataService, MarketBar } from './MarketDataService';

export class AISignalService {
    private static instance: AISignalService;
    private activeSignals: Map<string, AISignal> = new Map();
    private signalSubscribers: Set<(signals: AISignal[]) => void> = new Set();

    public static getInstance(): AISignalService {
        if (!AISignalService.instance) {
            AISignalService.instance = new AISignalService();
        }
        return AISignalService.instance;
    }

    /**
     * Generate AI signals for a symbol
     */
    async generateSignalsForSymbol(symbol: string): Promise<AISignal[]> {
        try {
            console.log(`🤖 Generating AI signals for ${symbol}`);

            // Get historical data
            const to = Math.floor(Date.now() / 1000);
            const from = to - (24 * 60 * 60 * 7); // 7 days

            const data = await marketDataService.getHistoricalData(symbol, '15', from, to);

            if (data.length < 50) {
                return [];
            }

            // Calculate technical indicators
            const rsi = this.calculateRSI(data.map(d => d.close), 14);
            const ema21 = this.calculateEMA(data.map(d => d.close), 21);
            const ema50 = this.calculateEMA(data.map(d => d.close), 50);

            const currentPrice = data[data.length - 1].close;
            const currentRSI = rsi[rsi.length - 1] || 50;
            const currentEMA21 = ema21[ema21.length - 1] || currentPrice;
            const currentEMA50 = ema50[ema50.length - 1] || currentPrice;

            // Generate signals based on technical analysis
            const signals: AISignal[] = [];

            // RSI Oversold Signal
            if (currentRSI < 30 && currentEMA21 > currentEMA50) {
                signals.push({
                    id: `${symbol}_rsi_buy_${Date.now()}`,
                    symbol,
                    type: 'BUY',
                    confidence: 75,
                    timestamp: Math.floor(Date.now() / 1000),
                    price: currentPrice,
                    reasoning: 'RSI oversold with bullish EMA alignment',
                    targets: [currentPrice * 1.02, currentPrice * 1.05],
                    stopLoss: currentPrice * 0.98,
                    timeframe: '15m',
                    pattern: 'RSI_OVERSOLD',
                    indicators: {
                        rsi: currentRSI,
                        ema21: currentEMA21,
                        ema50: currentEMA50,
                    },
                });
            }

            // RSI Overbought Signal
            if (currentRSI > 70 && currentEMA21 < currentEMA50) {
                signals.push({
                    id: `${symbol}_rsi_sell_${Date.now()}`,
                    symbol,
                    type: 'SELL',
                    confidence: 75,
                    timestamp: Math.floor(Date.now() / 1000),
                    price: currentPrice,
                    reasoning: 'RSI overbought with bearish EMA alignment',
                    targets: [currentPrice * 0.98, currentPrice * 0.95],
                    stopLoss: currentPrice * 1.02,
                    timeframe: '15m',
                    pattern: 'RSI_OVERBOUGHT',
                    indicators: {
                        rsi: currentRSI,
                        ema21: currentEMA21,
                        ema50: currentEMA50,
                    },
                });
            }

            // EMA Crossover Signal
            if (this.detectEMACrossover(ema21, ema50)) {
                const isBullish = currentEMA21 > currentEMA50;
                signals.push({
                    id: `${symbol}_ema_${isBullish ? 'buy' : 'sell'}_${Date.now()}`,
                    symbol,
                    type: isBullish ? 'BUY' : 'SELL',
                    confidence: 80,
                    timestamp: Math.floor(Date.now() / 1000),
                    price: currentPrice,
                    reasoning: `EMA ${isBullish ? 'bullish' : 'bearish'} crossover detected`,
                    targets: isBullish
                        ? [currentPrice * 1.03, currentPrice * 1.06]
                        : [currentPrice * 0.97, currentPrice * 0.94],
                    stopLoss: isBullish ? currentPrice * 0.97 : currentPrice * 1.03,
                    timeframe: '15m',
                    pattern: 'EMA_CROSSOVER',
                    indicators: {
                        ema21: currentEMA21,
                        ema50: currentEMA50,
                        rsi: currentRSI,
                    },
                });
            }

            return signals;
        } catch (error) {
            console.error(`Error generating signals for ${symbol}:`, error);
            return [];
        }
    }

    /**
     * Subscribe to signal updates
     */
    subscribeToSignals(callback: (signals: AISignal[]) => void): string {
        this.signalSubscribers.add(callback);
        callback(Array.from(this.activeSignals.values()));
        return Date.now().toString();
    }

    /**
     * Calculate EMA
     */
    private calculateEMA(prices: number[], period: number): number[] {
        const ema = [];
        const multiplier = 2 / (period + 1);
        ema[0] = prices[0];

        for (let i = 1; i < prices.length; i++) {
            ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
        }

        return ema;
    }

    /**
     * Calculate RSI
     */
    private calculateRSI(prices: number[], period: number): number[] {
        const rsi = [];
        const gains = [];
        const losses = [];

        for (let i = 1; i < prices.length; i++) {
            const change = prices[i] - prices[i - 1];
            gains.push(change > 0 ? change : 0);
            losses.push(change < 0 ? -change : 0);
        }

        for (let i = period - 1; i < gains.length; i++) {
            const avgGain = gains.slice(i - period + 1, i + 1).reduce((a, b) => a + b) / period;
            const avgLoss = losses.slice(i - period + 1, i + 1).reduce((a, b) => a + b) / period;

            if (avgLoss === 0) {
                rsi.push(100);
            } else {
                const rs = avgGain / avgLoss;
                rsi.push(100 - (100 / (1 + rs)));
            }
        }

        return rsi;
    }

    /**
     * Detect EMA crossover
     */
    private detectEMACrossover(ema21: number[], ema50: number[]): boolean {
        if (ema21.length < 2 || ema50.length < 2) return false;

        const prev21 = ema21[ema21.length - 2];
        const curr21 = ema21[ema21.length - 1];
        const prev50 = ema50[ema50.length - 2];
        const curr50 = ema50[ema50.length - 1];

        // Crossover occurred if the relationship changed
        return (prev21 <= prev50 && curr21 > curr50) || (prev21 >= prev50 && curr21 < curr50);
    }
}

export const aiSignalService = AISignalService.getInstance(); 