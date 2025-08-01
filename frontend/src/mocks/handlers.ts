import { http, HttpResponse } from 'msw';
import { faker } from '@faker-js/faker';

export const handlers = [
    // Mock signals endpoint
    http.get('/api/signals/:symbol', ({ params }) => {
        const signals = Array.from({ length: 10 }, () => ({
            id: faker.string.uuid(),
            symbol: params.symbol as string,
            type: faker.helpers.arrayElement(['BUY_CALL', 'BUY_PUT']),
            signal_type: faker.helpers.arrayElement(['BUY_CALL', 'BUY_PUT']),
            confidence: faker.number.int({ min: 65, max: 95 }),
            strike_price: faker.number.float({ min: 100, max: 500, precision: 0.01 }),
            entry_price: faker.number.float({ min: 100, max: 500, precision: 0.01 }),
            expiration_date: faker.date.future({ days: 30 }),
            timestamp: faker.date.recent(),
            reasoning: faker.lorem.paragraph(),
            targets: [
                faker.number.float({ min: 105, max: 600, precision: 0.01 }),
                faker.number.float({ min: 110, max: 650, precision: 0.01 })
            ],
            stop_loss: faker.number.float({ min: 80, max: 95, precision: 0.01 }),
            ai_confidence: faker.number.int({ min: 70, max: 95 }),
            priority: faker.helpers.arrayElement(['high', 'medium', 'low'])
        }));

        return HttpResponse.json(signals);
    }),

    // Mock market data endpoint
    http.get('/api/market-data/:symbol', ({ params }) => {
        return HttpResponse.json({
            symbol: params.symbol,
            price: faker.number.float({ min: 100, max: 500, precision: 0.01 }),
            change: faker.number.float({ min: -5, max: 5, precision: 0.01 }),
            changePercent: faker.number.float({ min: -2, max: 2, precision: 0.01 }),
            volume: faker.number.int({ min: 1000000, max: 50000000 }),
            open: faker.number.float({ min: 100, max: 500, precision: 0.01 }),
            high: faker.number.float({ min: 100, max: 500, precision: 0.01 }),
            low: faker.number.float({ min: 100, max: 500, precision: 0.01 }),
            marketCap: faker.number.int({ min: 1000000000, max: 3000000000000 })
        });
    }),

    // Mock performance metrics endpoint
    http.get('/api/performance-metrics', () => {
        return HttpResponse.json({
            winRate: faker.number.float({ min: 65, max: 85, precision: 0.1 }),
            avgReturn: faker.number.float({ min: 8, max: 25, precision: 0.1 }),
            sharpeRatio: faker.number.float({ min: 1.5, max: 3.5, precision: 0.2 }),
            maxDrawdown: faker.number.float({ min: -15, max: -5, precision: 0.1 }),
            totalSignals: faker.number.int({ min: 100, max: 500 }),
            successfulSignals: faker.number.int({ min: 70, max: 400 }),
            avgHoldingPeriod: faker.number.int({ min: 1, max: 10 }),
            profitFactor: faker.number.float({ min: 1.5, max: 3.0, precision: 0.1 })
        });
    }),

    // Mock news endpoint
    http.get('/api/market-news', () => {
        const news = Array.from({ length: 5 }, () => ({
            id: faker.string.uuid(),
            title: faker.company.catchPhrase() + ' ' + faker.helpers.arrayElement(['Surges', 'Drops', 'Announces', 'Reports']),
            summary: faker.lorem.sentences(2),
            url: faker.internet.url(),
            source: faker.helpers.arrayElement(['Reuters', 'Bloomberg', 'CNBC', 'WSJ']),
            publishedAt: faker.date.recent({ days: 2 }),
            sentiment: faker.helpers.arrayElement(['positive', 'negative', 'neutral']),
            relevanceScore: faker.number.float({ min: 0.5, max: 1.0, precision: 0.1 })
        }));

        return HttpResponse.json(news);
    }),

    // Mock AI insights endpoint
    http.get('/api/ai-insights/:signalId', ({ params }) => {
        return HttpResponse.json({
            signalId: params.signalId,
            analysis: faker.lorem.paragraphs(2),
            keyFactors: Array.from({ length: 3 }, () => ({
                factor: faker.helpers.arrayElement(['Technical', 'Sentiment', 'Volume', 'Options Flow']),
                description: faker.lorem.sentence(),
                impact: faker.helpers.arrayElement(['High', 'Medium', 'Low'])
            })),
            risks: Array.from({ length: 2 }, () => ({
                type: faker.helpers.arrayElement(['Market Risk', 'Volatility Risk', 'Event Risk']),
                description: faker.lorem.sentence(),
                mitigation: faker.lorem.sentence()
            }))
        });
    })
];
