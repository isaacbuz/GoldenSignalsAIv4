/**
 * Mock Data System
 * Provides realistic mock data for development and testing
 */

import { faker } from '@faker-js/faker';

// Types
export interface MarketData {
    symbol: string;
    price: number;
    change: number;
    changePercent: number;
    volume: number;
    high: number;
    low: number;
    open: number;
    previousClose: number;
    timestamp: string;
}

export interface Signal {
    id: string;
    symbol: string;
    action: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    price: number;
    timestamp: string;
    indicators: {
        RSI: number;
        MACD: number;
        BB_position: number;
    };
    riskLevel: 'low' | 'medium' | 'high';
    entryPrice: number;
    stopLoss: number;
    takeProfit: number;
}

export interface Agent {
    id: string;
    name: string;
    type: 'technical' | 'sentiment' | 'risk' | 'options';
    accuracy: number;
    signalsGenerated: number;
    profitLoss: number;
    status: 'active' | 'idle' | 'error';
    lastUpdate: string;
}

export interface Portfolio {
    totalValue: number;
    cash: number;
    positions: Position[];
    dayChange: number;
    dayChangePercent: number;
    totalReturn: number;
    totalReturnPercent: number;
}

export interface Position {
    id: string;
    symbol: string;
    quantity: number;
    averagePrice: number;
    currentPrice: number;
    value: number;
    dayChange: number;
    totalReturn: number;
    totalReturnPercent: number;
}

// Mock data generators
export class MockDataGenerator {
    private static symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ'];

    static generateMarketData(symbol?: string): MarketData {
        const basePrice = faker.number.float({ min: 50, max: 500, fractionDigits: 2 });
        const change = faker.number.float({ min: -10, max: 10, fractionDigits: 2 });
        const changePercent = (change / basePrice) * 100;

        return {
            symbol: symbol || faker.helpers.arrayElement(this.symbols),
            price: basePrice + change,
            change,
            changePercent,
            volume: faker.number.int({ min: 1000000, max: 50000000 }),
            high: basePrice + faker.number.float({ min: 0, max: 5, fractionDigits: 2 }),
            low: basePrice - faker.number.float({ min: 0, max: 5, fractionDigits: 2 }),
            open: basePrice,
            previousClose: basePrice,
            timestamp: new Date().toISOString()
        };
    }

    static generateSignal(symbol?: string): Signal {
        const action = faker.helpers.arrayElement(['BUY', 'SELL', 'HOLD']);
        const price = faker.number.float({ min: 50, max: 500, fractionDigits: 2 });
        const riskMultiplier = action === 'BUY' ? 0.02 : -0.02;

        return {
            id: faker.string.uuid(),
            symbol: symbol || faker.helpers.arrayElement(this.symbols),
            action,
            confidence: faker.number.float({ min: 0.65, max: 0.95, fractionDigits: 2 }),
            price,
            timestamp: faker.date.recent({ days: 1 }).toISOString(),
            indicators: {
                RSI: faker.number.float({ min: 20, max: 80, fractionDigits: 1 }),
                MACD: faker.number.float({ min: -2, max: 2, fractionDigits: 2 }),
                BB_position: faker.number.float({ min: 0, max: 1, fractionDigits: 2 })
            },
            riskLevel: faker.helpers.arrayElement(['low', 'medium', 'high']),
            entryPrice: price,
            stopLoss: price * (1 - Math.abs(riskMultiplier)),
            takeProfit: price * (1 + Math.abs(riskMultiplier) * 2)
        };
    }

    static generateAgent(): Agent {
        return {
            id: faker.string.uuid(),
            name: faker.helpers.arrayElement([
                'Momentum Analyzer',
                'Sentiment Scanner',
                'Risk Monitor',
                'Options Flow Tracker',
                'Pattern Recognition',
                'Volume Analyzer'
            ]),
            type: faker.helpers.arrayElement(['technical', 'sentiment', 'risk', 'options']),
            accuracy: faker.number.float({ min: 0.65, max: 0.92, fractionDigits: 2 }),
            signalsGenerated: faker.number.int({ min: 10, max: 1000 }),
            profitLoss: faker.number.float({ min: -10000, max: 50000, fractionDigits: 2 }),
            status: faker.helpers.arrayElement(['active', 'idle', 'error']),
            lastUpdate: faker.date.recent({ days: 1 }).toISOString()
        };
    }

    static generatePosition(symbol?: string): Position {
        const quantity = faker.number.int({ min: 1, max: 1000 });
        const averagePrice = faker.number.float({ min: 50, max: 500, fractionDigits: 2 });
        const currentPrice = averagePrice * faker.number.float({ min: 0.8, max: 1.2, fractionDigits: 2 });
        const value = quantity * currentPrice;
        const totalReturn = (currentPrice - averagePrice) * quantity;
        const totalReturnPercent = ((currentPrice - averagePrice) / averagePrice) * 100;

        return {
            id: faker.string.uuid(),
            symbol: symbol || faker.helpers.arrayElement(this.symbols),
            quantity,
            averagePrice,
            currentPrice,
            value,
            dayChange: faker.number.float({ min: -1000, max: 1000, fractionDigits: 2 }),
            totalReturn,
            totalReturnPercent
        };
    }

    static generatePortfolio(): Portfolio {
        const positions = Array.from({ length: faker.number.int({ min: 5, max: 15 }) }, () =>
            this.generatePosition()
        );

        const totalValue = positions.reduce((sum, pos) => sum + pos.value, 0);
        const cash = faker.number.float({ min: 10000, max: 100000, fractionDigits: 2 });
        const dayChange = positions.reduce((sum, pos) => sum + pos.dayChange, 0);

        return {
            totalValue: totalValue + cash,
            cash,
            positions,
            dayChange,
            dayChangePercent: (dayChange / (totalValue + cash)) * 100,
            totalReturn: positions.reduce((sum, pos) => sum + pos.totalReturn, 0),
            totalReturnPercent: faker.number.float({ min: -20, max: 50, fractionDigits: 2 })
        };
    }

    static generateHistoricalData(symbol: string, days: number = 30): Array<{
        date: string;
        open: number;
        high: number;
        low: number;
        close: number;
        volume: number;
    }> {
        const data = [];
        let basePrice = faker.number.float({ min: 100, max: 200, fractionDigits: 2 });

        for (let i = days; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);

            const dayChange = faker.number.float({ min: -5, max: 5, fractionDigits: 2 });
            const open = basePrice;
            const close = basePrice + dayChange;
            const high = Math.max(open, close) + faker.number.float({ min: 0, max: 2, fractionDigits: 2 });
            const low = Math.min(open, close) - faker.number.float({ min: 0, max: 2, fractionDigits: 2 });

            data.push({
                date: date.toISOString().split('T')[0],
                open,
                high,
                low,
                close,
                volume: faker.number.int({ min: 1000000, max: 10000000 })
            });

            basePrice = close;
        }

        return data;
    }
}

// Mock API responses
export const mockResponses = {
    signals: {
        latest: () => ({
            signals: Array.from({ length: 10 }, () => MockDataGenerator.generateSignal()),
            timestamp: new Date().toISOString(),
            count: 10
        }),

        bySymbol: (symbol: string) => ({
            symbol,
            signals: Array.from({ length: 5 }, () => MockDataGenerator.generateSignal(symbol)),
            timestamp: new Date().toISOString()
        })
    },

    market: {
        quote: (symbol: string) => MockDataGenerator.generateMarketData(symbol),

        summary: () => ({
            indices: {
                SP500: { value: 4500.21, change: 15.42, changePercent: 0.34 },
                NASDAQ: { value: 14123.45, change: -23.67, changePercent: -0.17 },
                DOW: { value: 35678.90, change: 123.45, changePercent: 0.35 }
            },
            topGainers: Array.from({ length: 5 }, () => MockDataGenerator.generateMarketData()),
            topLosers: Array.from({ length: 5 }, () => MockDataGenerator.generateMarketData()),
            mostActive: Array.from({ length: 5 }, () => MockDataGenerator.generateMarketData()),
            timestamp: new Date().toISOString()
        })
    },

    agents: {
        performance: () => ({
            agents: Array.from({ length: 6 }, () => MockDataGenerator.generateAgent()),
            summary: {
                totalAgents: 6,
                activeAgents: 4,
                totalSignals: faker.number.int({ min: 1000, max: 5000 }),
                averageAccuracy: 0.78,
                totalProfitLoss: faker.number.float({ min: 10000, max: 100000, fractionDigits: 2 })
            },
            timestamp: new Date().toISOString()
        })
    },

    portfolio: {
        summary: () => MockDataGenerator.generatePortfolio(),

        positions: () => ({
            positions: Array.from({ length: 10 }, () => MockDataGenerator.generatePosition()),
            timestamp: new Date().toISOString()
        })
    }
};

// Utility functions for mock data
export const mockDataUtils = {
    /**
     * Simulate real-time data updates
     */
    streamMarketData: (symbol: string, callback: (data: MarketData) => void) => {
        const interval = setInterval(() => {
            callback(MockDataGenerator.generateMarketData(symbol));
        }, 1000);

        return () => clearInterval(interval);
    },

    /**
     * Simulate WebSocket messages
     */
    simulateWebSocket: (onMessage: (data: any) => void) => {
        const messageTypes = ['signal_update', 'market_update', 'agent_status'];

        const interval = setInterval(() => {
            const type = faker.helpers.arrayElement(messageTypes);
            let data;

            switch (type) {
                case 'signal_update':
                    data = {
                        type,
                        data: MockDataGenerator.generateSignal()
                    };
                    break;
                case 'market_update':
                    data = {
                        type,
                        data: MockDataGenerator.generateMarketData()
                    };
                    break;
                case 'agent_status':
                    data = {
                        type,
                        data: MockDataGenerator.generateAgent()
                    };
                    break;
            }

            onMessage(data);
        }, faker.number.int({ min: 1000, max: 5000 }));

        return () => clearInterval(interval);
    },

    /**
     * Generate mock error responses
     */
    generateError: (status: number = 500) => ({
        error: faker.helpers.arrayElement([
            'Internal Server Error',
            'Service Unavailable',
            'Bad Request',
            'Unauthorized'
        ]),
        message: faker.lorem.sentence(),
        status,
        timestamp: new Date().toISOString()
    })
};

// Export for use in MSW handlers
export const mockData = {
    generator: MockDataGenerator,
    responses: mockResponses,
    utils: mockDataUtils
};
