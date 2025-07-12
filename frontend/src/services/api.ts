// Simple API client for the trading platform
const API_BASE_URL = 'http://localhost:8000/api/v1';

interface MarketData {
    symbol: string;
    price: number;
    change: number;
    changePercent: number;
    volume: number;
    high: number;
    low: number;
}

interface AIInsight {
    symbol: string;
    signal: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    reasoning: string;
    sentiment?: string | { overall: string };
    summary?: string;
}

interface PreciseOptionsSignal {
    id: string;
    signal_id?: string;
    symbol: string;
    signal_type: 'BUY_CALL' | 'BUY_PUT';
    type?: string;
    strike_price: number;
    expiration_date: string;
    confidence: number;
    timestamp: string;
    reasoning?: string;
}

interface MarketNews {
    id: string;
    title: string;
    source: string;
    timestamp: string;
    url?: string;
    impact?: 'HIGH' | 'MEDIUM' | 'LOW';
}

class ApiClient {
    async getMarketData(symbol: string): Promise<MarketData> {
        try {
            const response = await fetch(`${API_BASE_URL}/market-data/${symbol}`);
            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.warn('API call failed, using mock data:', error);
        }

        // Mock data fallback
        return {
            symbol,
            price: 195.89 + (Math.random() - 0.5) * 10,
            change: (Math.random() - 0.5) * 5,
            changePercent: (Math.random() - 0.5) * 3,
            volume: Math.floor(Math.random() * 10000000),
            high: 196.38,
            low: 193.67,
        };
    }

    async getAIInsights(signalId: string): Promise<AIInsight> {
        try {
            const response = await fetch(`${API_BASE_URL}/ai-insights/${signalId}`);
            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.warn('API call failed, using mock data:', error);
        }

        // Mock data fallback
        return {
            symbol: 'AAPL',
            signal: 'BUY',
            confidence: 92,
            reasoning: 'Strong technical indicators suggest upward momentum.',
            sentiment: 'bullish',
            summary: 'The AI analysis indicates a strong bullish sentiment based on technical patterns and market momentum.'
        };
    }

    async getPreciseOptionsSignals(symbol: string, timeframe: string): Promise<PreciseOptionsSignal[]> {
        try {
            const response = await fetch(`${API_BASE_URL}/signals?symbol=${symbol}&timeframe=${timeframe}`);
            if (response.ok) {
                const data = await response.json();
                // Transform backend signals to PreciseOptionsSignal format
                return (data.signals || []).map((signal: any, index: number) => ({
                    id: signal.signal_id || `sig_${index}`,
                    signal_id: signal.signal_id,
                    symbol: signal.symbol,
                    signal_type: signal.signal_type === 'BUY' ? 'BUY_CALL' : 'BUY_PUT',
                    type: signal.signal_type,
                    strike_price: signal.signal_details?.entry_price || 100 + Math.random() * 50,
                    expiration_date: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(), // 7 days from now
                    confidence: signal.confidence_score || signal.confidence || 85,
                    timestamp: signal.timestamp,
                    reasoning: signal.signal_details?.reason || 'AI-driven pattern recognition'
                }));
            }
        } catch (error) {
            console.warn('API call failed, using mock data:', error);
        }

        // Mock data fallback for options signals
        const now = new Date();
        return [
            {
                id: 'opt_1',
                signal_id: 'opt_1',
                symbol,
                signal_type: 'BUY_CALL',
                strike_price: 200,
                expiration_date: new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000).toISOString(),
                confidence: 88,
                timestamp: now.toISOString(),
                reasoning: 'Strong bullish momentum detected'
            },
            {
                id: 'opt_2',
                signal_id: 'opt_2',
                symbol,
                signal_type: 'BUY_PUT',
                strike_price: 185,
                expiration_date: new Date(now.getTime() + 14 * 24 * 60 * 60 * 1000).toISOString(),
                confidence: 75,
                timestamp: new Date(now.getTime() - 3600000).toISOString(),
                reasoning: 'Potential reversal pattern forming'
            }
        ];
    }

    async generateSignalsForSymbol(symbol: string, timeframe: string): Promise<void> {
        try {
            const response = await fetch(`${API_BASE_URL}/signals/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ symbol, timeframe }),
            });
            if (!response.ok) {
                throw new Error('Failed to generate signals');
            }
        } catch (error) {
            console.warn('Failed to generate signals:', error);
        }
    }

    async getMarketNews(): Promise<MarketNews[]> {
        try {
            const response = await fetch(`${API_BASE_URL}/market-news`);
            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.warn('API call failed, using mock data:', error);
        }

        // Mock news data
        return [
            {
                id: 'news_1',
                title: 'Fed Signals Potential Rate Cut in Coming Months',
                source: 'Reuters',
                timestamp: new Date().toISOString(),
                url: '#',
                impact: 'HIGH'
            },
            {
                id: 'news_2',
                title: 'Tech Stocks Rally on Strong Earnings Reports',
                source: 'Bloomberg',
                timestamp: new Date(Date.now() - 3600000).toISOString(),
                url: '#',
                impact: 'MEDIUM'
            },
            {
                id: 'news_3',
                title: 'Oil Prices Stabilize After Volatile Week',
                source: 'CNBC',
                timestamp: new Date(Date.now() - 7200000).toISOString(),
                url: '#',
                impact: 'LOW'
            }
        ];
    }
}

export const apiClient = new ApiClient();

// Legacy exports for backward compatibility
export const fetchMarketData = (symbol: string) => apiClient.getMarketData(symbol);
export const fetchAIInsights = (symbol: string) => apiClient.getAIInsights(symbol); 