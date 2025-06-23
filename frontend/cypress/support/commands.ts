/// <reference types="cypress" />

// Custom commands for GoldenSignalsAI testing

// Login command
Cypress.Commands.add('login', (email = 'test@example.com', password = 'password123') => {
  cy.visit('/login')
  cy.get('[data-testid="email-input"]').type(email)
  cy.get('[data-testid="password-input"]').type(password)
  cy.get('[data-testid="login-button"]').click()
  cy.url().should('include', '/dashboard')
})

// Mock API responses
Cypress.Commands.add('mockSignals', (signals) => {
  cy.intercept('GET', '/api/v1/signals', {
    statusCode: 200,
    body: signals || [
      {
        id: '1',
        symbol: 'AAPL',
        action: 'BUY',
        confidence: 0.85,
        price: 150.00,
        timestamp: new Date().toISOString(),
        source: 'ML_ENSEMBLE'
      }
    ]
  }).as('getSignals')
})

// Wait for chart to load
Cypress.Commands.add('waitForChart', () => {
  cy.get('[data-testid="trading-chart"]', { timeout: 10000 }).should('be.visible')
  cy.wait(1000) // Additional wait for chart rendering
})

// Check WebSocket connection
Cypress.Commands.add('checkWebSocketConnection', () => {
  cy.get('[data-testid="websocket-status"]').should('contain', 'Connected')
})

// Navigate to specific page
Cypress.Commands.add('navigateTo', (page: string) => {
  const routes: Record<string, string> = {
    dashboard: '/dashboard',
    signals: '/signals',
    portfolio: '/portfolio',
    analytics: '/analytics',
    settings: '/settings'
  }
  cy.visit(routes[page] || page)
})

// Declare types for TypeScript
declare global {
  namespace Cypress {
    interface Chainable {
      login(email?: string, password?: string): Chainable<void>
      mockSignals(signals?: any[]): Chainable<void>
      waitForChart(): Chainable<void>
      checkWebSocketConnection(): Chainable<void>
      navigateTo(page: string): Chainable<void>
    }
  }
}

export {}
