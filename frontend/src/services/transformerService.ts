import axios from 'axios';

interface TechnicalIndicators {
  rsi: number[];
  bollingerHigh: number[];
  bollingerLow: number[];
  ma20: number[];
  ma20Slope: number[];
}

interface HistoricalData {
  timestamps: string[];
  prices: number[];
  predictions: number[];
}

interface TransformerPrediction {
  targetPrice: number;
  confidence: number;
  timeframe: string;
}

interface TransformerPredictionData {
  symbol: string;
  currentPrice: number;
  prediction: TransformerPrediction;
  historicalData: HistoricalData;
  technicalIndicators: TechnicalIndicators;
}

class TransformerService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  }

  async getPrediction(symbol: string, timeframe: string = '1h'): Promise<TransformerPredictionData> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/transformer/predict`, {
        params: {
          symbol,
          timeframe
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching transformer prediction:', error);
      throw error;
    }
  }

  async getHistoricalPredictions(symbol: string, timeframe: string = '1h'): Promise<HistoricalData> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/transformer/history`, {
        params: {
          symbol,
          timeframe
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching historical predictions:', error);
      throw error;
    }
  }

  async getTechnicalIndicators(symbol: string, timeframe: string = '1h'): Promise<TechnicalIndicators> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/transformer/indicators`, {
        params: {
          symbol,
          timeframe
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching technical indicators:', error);
      throw error;
    }
  }

  // Helper function to calculate RSI
  calculateRSI(prices: number[], period: number = 14): number[] {
    const rsi: number[] = [];
    let gains = 0;
    let losses = 0;

    // Calculate initial average gain and loss
    for (let i = 1; i < period + 1; i++) {
      const difference = prices[i] - prices[i - 1];
      if (difference >= 0) {
        gains += difference;
      } else {
        losses -= difference;
      }
    }

    let avgGain = gains / period;
    let avgLoss = losses / period;

    // Calculate first RSI
    let rs = avgGain / avgLoss;
    rsi.push(100 - (100 / (1 + rs)));

    // Calculate remaining RSI values
    for (let i = period + 1; i < prices.length; i++) {
      const difference = prices[i] - prices[i - 1];
      avgGain = ((avgGain * (period - 1)) + (difference > 0 ? difference : 0)) / period;
      avgLoss = ((avgLoss * (period - 1)) + (difference < 0 ? -difference : 0)) / period;

      rs = avgGain / avgLoss;
      rsi.push(100 - (100 / (1 + rs)));
    }

    return rsi;
  }

  // Helper function to calculate Bollinger Bands
  calculateBollingerBands(prices: number[], period: number = 20, stdDev: number = 2): { upper: number[], lower: number[] } {
    const sma = this.calculateSMA(prices, period);
    const upper: number[] = [];
    const lower: number[] = [];

    for (let i = period - 1; i < prices.length; i++) {
      const slice = prices.slice(i - period + 1, i + 1);
      const std = this.calculateStandardDeviation(slice, sma[i - period + 1]);
      upper.push(sma[i - period + 1] + (stdDev * std));
      lower.push(sma[i - period + 1] - (stdDev * std));
    }

    return { upper, lower };
  }

  // Helper function to calculate Simple Moving Average
  private calculateSMA(prices: number[], period: number): number[] {
    const sma: number[] = [];
    for (let i = period - 1; i < prices.length; i++) {
      const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      sma.push(sum / period);
    }
    return sma;
  }

  // Helper function to calculate Standard Deviation
  private calculateStandardDeviation(values: number[], mean: number): number {
    const squareDiffs = values.map(value => {
      const diff = value - mean;
      return diff * diff;
    });
    const avgSquareDiff = squareDiffs.reduce((a, b) => a + b, 0) / values.length;
    return Math.sqrt(avgSquareDiff);
  }

  // Helper function to calculate MA20 Slope
  calculateMA20Slope(ma20: number[]): number[] {
    const slope: number[] = [];
    for (let i = 1; i < ma20.length; i++) {
      slope.push(ma20[i] - ma20[i - 1]);
    }
    return slope;
  }
}

export const transformerService = new TransformerService();
export type { TransformerPredictionData, TechnicalIndicators, HistoricalData, TransformerPrediction }; 