import { http, HttpResponse } from 'msw'

// Mock data
const mockSignals = [
  {
    id: '1',
    symbol: 'AAPL',
    action: 'BUY',
    confidence: 0.85,
    price: 150.00,
    timestamp: new Date().toISOString(),
    source: 'ML_ENSEMBLE',
    metadata: {
      technical_score: 0.8,
      sentiment_score: 0.9,
      volume_analysis: 'BULLISH'
    }
  },
  {
    id: '2',
    symbol: 'GOOGL',
    action: 'HOLD',
    confidence: 0.65,
    price: 2800.00,
    timestamp: new Date().toISOString(),
    source: 'TECHNICAL',
    metadata: {
      technical_score: 0.6,
      sentiment_score: 0.7,
      volume_analysis: 'NEUTRAL'
    }
  }
]

const mockMarketData = {
  symbol: 'AAPL',
  price: 150.00,
  change: 2.5,
  changePercent: 1.69,
  volume: 75000000,
  high: 151.50,
  low: 148.00,
  open: 148.50,
  previousClose: 147.50,
  timestamp: new Date().toISOString()
}

const mockOpportunities = [
  {
    id: '1',
    symbol: 'TSLA',
    type: 'BREAKOUT',
    confidence: 0.78,
    potentialReturn: 5.2,
    risk: 'MEDIUM',
    description: 'Potential breakout above resistance level'
  },
  {
    id: '2',
    symbol: 'MSFT',
    type: 'MOMENTUM',
    confidence: 0.82,
    potentialReturn: 3.8,
    risk: 'LOW',
    description: 'Strong momentum with increasing volume'
  }
]

// Request handlers
export const handlers = [
  // Signals endpoints
  http.get('/api/v1/signals', () => {
    return HttpResponse.json(mockSignals)
  }),

  http.get('/api/v1/signals/:symbol/insights', ({ params }) => {
    return HttpResponse.json({
      symbol: params.symbol,
      insights: {
        trend: 'BULLISH',
        support: 145.00,
        resistance: 155.00,
        recommendation: 'BUY',
        confidence: 0.75
      }
    })
  }),

  // Market data endpoints
  http.get('/api/v1/market-data/:symbol', ({ params }) => {
    return HttpResponse.json({
      ...mockMarketData,
      symbol: params.symbol as string
    })
  }),

  http.get('/api/v1/market-data/:symbol/historical', ({ request, params }) => {
    const url = new URL(request.url)
    const period = url.searchParams.get('period') || '1d'
    const interval = url.searchParams.get('interval') || '5m'
    
    // Generate mock historical data
    const dataPoints = []
    const now = Date.now()
    const intervalMs = interval === '5m' ? 5 * 60 * 1000 : 60 * 60 * 1000
    
    for (let i = 0; i < 100; i++) {
      dataPoints.push({
        timestamp: new Date(now - i * intervalMs).toISOString(),
        open: 150 + Math.random() * 5,
        high: 152 + Math.random() * 5,
        low: 148 + Math.random() * 5,
        close: 150 + Math.random() * 5,
        volume: Math.floor(Math.random() * 1000000)
      })
    }
    
    return HttpResponse.json({
      symbol: params.symbol,
      period,
      interval,
      data: dataPoints
    })
  }),

  // Market opportunities
  http.get('/api/v1/market/opportunities', () => {
    return HttpResponse.json(mockOpportunities)
  }),

  // WebSocket mock (for testing WebSocket connections)
  http.get('/ws', () => {
    return new HttpResponse(null, {
      status: 101,
      headers: {
        'Upgrade': 'websocket',
        'Connection': 'Upgrade',
      },
    })
  }),

  // Performance endpoint
  http.get('/api/v1/performance', () => {
    return HttpResponse.json({
      totalSignals: 150,
      successRate: 0.72,
      averageReturn: 2.5,
      sharpeRatio: 1.8,
      maxDrawdown: -0.15,
      winRate: 0.68
    })
  }),

  // Error handling examples
  http.get('/api/v1/error/500', () => {
    return new HttpResponse(null, { status: 500 })
  }),

  http.get('/api/v1/error/404', () => {
    return new HttpResponse(null, { status: 404 })
  }),
]
