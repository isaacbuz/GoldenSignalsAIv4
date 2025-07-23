/**
 * AI Prediction Service
 * Handles accurate AI predictions based on real market analysis
 */

import { backendMarketDataService } from './backendMarketDataService';
import logger from './logger';


export interface AIPredictionResult {
  predictions: PredictionPoint[];
  confidence: number;
  reasoning: string[];
  supportLevel: number;
  resistanceLevel: number;
  trendDirection: 'bullish' | 'bearish' | 'neutral';
  timeframe: string;
}

export interface PredictionPoint {
  time: number;
  price: number;
  upperBound: number;
  lowerBound: number;
  confidence: number;
}

export interface TechnicalIndicators {
  rsi: number;
  macd: { value: number; signal: number; histogram: number };
  sma20: number;
  sma50: number;
  ema12: number;
  ema26: number;
  atr: number;
  bollingerBands: { upper: number; middle: number; lower: number };
}

class AIPredictionService {
  private cache: Map<string, { data: AIPredictionResult; timestamp: number }> = new Map();
  private cacheTimeout = 60000; // 1 minute cache

  /**
   * Get AI predictions from backend
   */
  async getAIPredictions(
    symbol: string,
    timeframe: string,
    historicalData: any[]
  ): Promise<AIPredictionResult> {
    const cacheKey = `${symbol}-${timeframe}`;
    const cached = this.cache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }

    try {
      // Call advanced AI prediction endpoint
      const response = await fetch(
        `${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/api/v1/ai/predict/${symbol}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ timeframe }),
        }
      );

      if (!response.ok) {
        throw new Error(`AI prediction failed: ${response.statusText}`);
      }

      const aiPrediction = await response.json();

      // Use advanced predictions from backend
      if (aiPrediction.predictions && aiPrediction.predictions.length > 0) {
        const result: AIPredictionResult = {
          predictions: aiPrediction.predictions.map((p: any) => ({
            time: new Date(p.time).getTime() / 1000,
            price: p.price,
            upperBound: p.upper_bound,
            lowerBound: p.lower_bound,
            confidence: p.confidence,
          })),
          confidence: aiPrediction.confidence,
          reasoning: aiPrediction.reasoning,
          supportLevel: aiPrediction.support_level,
          resistanceLevel: aiPrediction.resistance_level,
          trendDirection: aiPrediction.trend_direction,
          timeframe,
        };

        // Cache the result
        this.cache.set(cacheKey, { data: result, timestamp: Date.now() });
        return result;
      }

      // Fallback to local analysis if backend doesn't return predictions
      const aiAnalysis = aiPrediction;

      // Extract technical analysis from response
      const indicators = this.calculateTechnicalIndicators(historicalData);

      // Generate realistic predictions based on AI analysis
      const predictions = this.generateRealisticPredictions(
        historicalData,
        indicators,
        aiAnalysis
      );

      const result: AIPredictionResult = {
        predictions,
        confidence: aiAnalysis.confidence || 0.7,
        reasoning: aiAnalysis.reasoning || ['Technical analysis indicates potential movement'],
        supportLevel: this.calculateSupportLevel(historicalData),
        resistanceLevel: this.calculateResistanceLevel(historicalData),
        trendDirection: this.determineTrend(historicalData, indicators),
        timeframe,
      };

      // Cache the result
      this.cache.set(cacheKey, { data: result, timestamp: Date.now() });

      return result;
    } catch (error) {
      logger.error('AI prediction failed:', error);

      // Fallback to technical analysis only
      const indicators = this.calculateTechnicalIndicators(historicalData);
      return this.generateFallbackPrediction(historicalData, indicators, timeframe);
    }
  }

  /**
   * Calculate technical indicators for analysis
   */
  private calculateTechnicalIndicators(data: any[]): TechnicalIndicators {
    const prices = data.map(d => d.close);

    return {
      rsi: this.calculateRSI(prices),
      macd: this.calculateMACD(prices),
      sma20: this.calculateSMA(prices, 20),
      sma50: this.calculateSMA(prices, 50),
      ema12: this.calculateEMA(prices, 12),
      ema26: this.calculateEMA(prices, 26),
      atr: this.calculateATR(data),
      bollingerBands: this.calculateBollingerBands(prices),
    };
  }

  /**
   * Generate realistic predictions based on analysis
   */
  private generateRealisticPredictions(
    data: any[],
    indicators: TechnicalIndicators,
    aiAnalysis?: any
  ): PredictionPoint[] {
    const predictions: PredictionPoint[] = [];
    const lastCandle = data[data.length - 1];
    const lastPrice = lastCandle.close;
    const lastTime = lastCandle.time;

    // Determine prediction parameters
    const volatility = indicators.atr / lastPrice;
    const trend = this.calculateTrendStrength(data, indicators);
    const momentum = this.calculateMomentum(data);

    // Generate predictions for next 5-10 candles only
    const numPredictions = 5;
    let predictedPrice = lastPrice;

    for (let i = 1; i <= numPredictions; i++) {
      // Calculate time based on timeframe
      const timeIncrement = this.getTimeIncrement(data);
      const predictionTime = lastTime + (i * timeIncrement);

      // Price movement based on multiple factors
      let priceChange = 0;

      // Trend component (40% weight)
      priceChange += trend * volatility * 0.4;

      // Mean reversion component (30% weight)
      const deviation = (predictedPrice - indicators.sma20) / indicators.sma20;
      priceChange -= deviation * volatility * 0.3;

      // RSI component (20% weight)
      if (indicators.rsi > 70) {
        priceChange -= volatility * 0.2;
      } else if (indicators.rsi < 30) {
        priceChange += volatility * 0.2;
      }

      // Momentum component (10% weight)
      priceChange += momentum * volatility * 0.1;

      // Apply AI adjustment if available
      if (aiAnalysis?.direction) {
        const aiDirection = aiAnalysis.direction === 'up' ? 1 : -1;
        priceChange = priceChange * 0.7 + (aiDirection * volatility * 0.3);
      }

      // Update predicted price
      predictedPrice = predictedPrice * (1 + priceChange);

      // Calculate confidence bounds
      const confidence = Math.max(0.5, Math.min(0.9, 0.8 - (i * 0.05)));
      const boundWidth = volatility * Math.sqrt(i) * 2;

      predictions.push({
        time: predictionTime,
        price: predictedPrice,
        upperBound: predictedPrice * (1 + boundWidth),
        lowerBound: predictedPrice * (1 - boundWidth),
        confidence,
      });
    }

    return predictions;
  }

  /**
   * Generate quality trading signals
   */
  generateTradingSignals(
    data: any[],
    indicators: TechnicalIndicators,
    predictions: PredictionPoint[]
  ): any[] {
    const signals = [];
    const lastCandle = data[data.length - 1];
    const lastPrice = lastCandle.close;

    // Only generate signal if conditions are strong
    const signalStrength = this.calculateSignalStrength(indicators, predictions);

    if (signalStrength > 0.7) {
      const direction = predictions[0].price > lastPrice ? 'buy' : 'sell';
      const reason = this.generateSignalReason(indicators, direction);

      signals.push({
        time: lastCandle.time,
        type: direction,
        price: lastPrice,
        confidence: signalStrength,
        reason,
        target: direction === 'buy'
          ? lastPrice * (1 + indicators.atr / lastPrice * 2)
          : lastPrice * (1 - indicators.atr / lastPrice * 2),
        stopLoss: direction === 'buy'
          ? lastPrice * (1 - indicators.atr / lastPrice)
          : lastPrice * (1 + indicators.atr / lastPrice),
      });
    }

    return signals;
  }

  /**
   * Detect high-quality patterns only
   */
  detectPatterns(data: any[]): any[] {
    const patterns = [];

    // Only return the most significant pattern if found
    const headAndShoulders = this.detectHeadAndShoulders(data);
    if (headAndShoulders && headAndShoulders.confidence > 0.7) {
      patterns.push(headAndShoulders);
    }

    const triangle = this.detectTriangle(data);
    if (triangle && triangle.confidence > 0.7 && patterns.length === 0) {
      patterns.push(triangle);
    }

    // Maximum 2 patterns for clarity
    return patterns.slice(0, 2);
  }

  // Helper methods
  private calculateRSI(prices: number[], period = 14): number {
    if (prices.length < period) return 50;

    let gains = 0;
    let losses = 0;

    for (let i = prices.length - period; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      if (change > 0) gains += change;
      else losses -= change;
    }

    const avgGain = gains / period;
    const avgLoss = losses / period;
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;

    return 100 - (100 / (1 + rs));
  }

  private calculateSMA(prices: number[], period: number): number {
    if (prices.length < period) return prices[prices.length - 1];
    const slice = prices.slice(-period);
    return slice.reduce((a, b) => a + b, 0) / period;
  }

  private calculateEMA(prices: number[], period: number): number {
    if (prices.length < period) return prices[prices.length - 1];

    const k = 2 / (period + 1);
    let ema = prices[0];

    for (let i = 1; i < prices.length; i++) {
      ema = prices[i] * k + ema * (1 - k);
    }

    return ema;
  }

  private calculateMACD(prices: number[]): any {
    const ema12 = this.calculateEMA(prices, 12);
    const ema26 = this.calculateEMA(prices, 26);
    const macdLine = ema12 - ema26;
    const signal = this.calculateEMA([macdLine], 9);

    return {
      value: macdLine,
      signal,
      histogram: macdLine - signal,
    };
  }

  private calculateATR(data: any[], period = 14): number {
    if (data.length < period) return 0;

    const trueRanges = [];
    for (let i = 1; i < data.length; i++) {
      const high = data[i].high;
      const low = data[i].low;
      const prevClose = data[i - 1].close;

      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );

      trueRanges.push(tr);
    }

    const recentTR = trueRanges.slice(-period);
    return recentTR.reduce((a, b) => a + b, 0) / period;
  }

  private calculateBollingerBands(prices: number[], period = 20, stdDev = 2): any {
    const sma = this.calculateSMA(prices, period);
    const slice = prices.slice(-period);
    const variance = slice.reduce((sum, price) => sum + Math.pow(price - sma, 2), 0) / period;
    const std = Math.sqrt(variance);

    return {
      upper: sma + std * stdDev,
      middle: sma,
      lower: sma - std * stdDev,
    };
  }

  private calculateSupportLevel(data: any[]): number {
    const recentLows = data.slice(-50).map(d => d.low);
    return Math.min(...recentLows) * 1.01; // 1% above lowest
  }

  private calculateResistanceLevel(data: any[]): number {
    const recentHighs = data.slice(-50).map(d => d.high);
    return Math.max(...recentHighs) * 0.99; // 1% below highest
  }

  private determineTrend(data: any[], indicators: TechnicalIndicators): 'bullish' | 'bearish' | 'neutral' {
    const smaDirection = indicators.sma20 > indicators.sma50 ? 1 : -1;
    const pricePosition = data[data.length - 1].close > indicators.sma20 ? 1 : -1;
    const macdDirection = indicators.macd.value > indicators.macd.signal ? 1 : -1;

    const trendScore = smaDirection + pricePosition + macdDirection;

    if (trendScore >= 2) return 'bullish';
    if (trendScore <= -2) return 'bearish';
    return 'neutral';
  }

  private calculateTrendStrength(data: any[], indicators: TechnicalIndicators): number {
    const prices = data.map(d => d.close);
    const recentPrices = prices.slice(-20);

    // Linear regression slope
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for (let i = 0; i < recentPrices.length; i++) {
      sumX += i;
      sumY += recentPrices[i];
      sumXY += i * recentPrices[i];
      sumX2 += i * i;
    }

    const n = recentPrices.length;
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const avgPrice = sumY / n;

    // Normalize slope
    return Math.max(-1, Math.min(1, slope / avgPrice * 100));
  }

  private calculateMomentum(data: any[]): number {
    if (data.length < 10) return 0;

    const current = data[data.length - 1].close;
    const past = data[data.length - 10].close;

    return (current - past) / past;
  }

  private getTimeIncrement(data: any[]): number {
    if (data.length < 2) return 300; // Default 5 minutes
    return data[data.length - 1].time - data[data.length - 2].time;
  }

  private calculateSignalStrength(indicators: TechnicalIndicators, predictions: PredictionPoint[]): number {
    let strength = 0;
    let factors = 0;

    // RSI signal
    if (indicators.rsi < 30) {
      strength += 0.8;
      factors++;
    } else if (indicators.rsi > 70) {
      strength += 0.8;
      factors++;
    }

    // MACD signal
    if (Math.abs(indicators.macd.histogram) > 0) {
      strength += 0.7;
      factors++;
    }

    // Bollinger Bands signal
    const lastPrice = predictions[0]?.price || 0;
    if (lastPrice <= indicators.bollingerBands.lower) {
      strength += 0.7;
      factors++;
    } else if (lastPrice >= indicators.bollingerBands.upper) {
      strength += 0.7;
      factors++;
    }

    return factors > 0 ? strength / factors : 0;
  }

  private generateSignalReason(indicators: TechnicalIndicators, direction: string): string {
    const reasons = [];

    if (indicators.rsi < 30) reasons.push('RSI oversold');
    else if (indicators.rsi > 70) reasons.push('RSI overbought');

    if (indicators.macd.histogram > 0 && direction === 'buy') reasons.push('MACD bullish');
    else if (indicators.macd.histogram < 0 && direction === 'sell') reasons.push('MACD bearish');

    if (reasons.length === 0) reasons.push('Technical setup');

    return reasons.join(', ');
  }

  private detectHeadAndShoulders(data: any[]): any | null {
    // Implement proper head and shoulders detection
    // This is a placeholder - would need proper implementation
    return null;
  }

  private detectTriangle(data: any[]): any | null {
    // Implement proper triangle pattern detection
    // This is a placeholder - would need proper implementation
    return null;
  }

  private generateFallbackPrediction(
    data: any[],
    indicators: TechnicalIndicators,
    timeframe: string
  ): AIPredictionResult {
    return {
      predictions: this.generateRealisticPredictions(data, indicators),
      confidence: 0.6,
      reasoning: ['Technical analysis based prediction'],
      supportLevel: this.calculateSupportLevel(data),
      resistanceLevel: this.calculateResistanceLevel(data),
      trendDirection: this.determineTrend(data, indicators),
      timeframe,
    };
  }
}

// Export singleton instance
export const aiPredictionService = new AIPredictionService();
