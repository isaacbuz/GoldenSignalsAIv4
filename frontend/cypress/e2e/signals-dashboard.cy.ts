describe('Signals Dashboard E2E Tests', () => {
  beforeEach(() => {
    // Mock API responses
    cy.intercept('GET', '/api/v1/signals', { fixture: 'signals.json' }).as('getSignals')
    cy.intercept('GET', '/api/v1/market-data/*', { fixture: 'market-data.json' }).as('getMarketData')
    cy.intercept('GET', '/api/v1/market/opportunities', { fixture: 'opportunities.json' }).as('getOpportunities')
    
    // Visit signals dashboard
    cy.visit('/signals')
  })

  it('displays signals dashboard correctly', () => {
    // Check page title
    cy.contains('h1', 'AI Trading Signals').should('be.visible')
    
    // Wait for signals to load
    cy.wait('@getSignals')
    
    // Check if signals are displayed
    cy.get('[data-testid="signal-card"]').should('have.length.at.least', 1)
  })

  it('filters signals by symbol', () => {
    cy.wait('@getSignals')
    
    // Type in search bar
    cy.get('[data-testid="symbol-search"]').type('AAPL')
    
    // Check filtered results
    cy.get('[data-testid="signal-card"]').each(($card) => {
      cy.wrap($card).should('contain', 'AAPL')
    })
  })

  it('filters signals by action type', () => {
    cy.wait('@getSignals')
    
    // Click BUY filter
    cy.get('[data-testid="filter-buy"]').click()
    
    // Check filtered results
    cy.get('[data-testid="signal-card"]').each(($card) => {
      cy.wrap($card).find('[data-testid="signal-action"]').should('contain', 'BUY')
    })
  })

  it('displays signal details on click', () => {
    cy.wait('@getSignals')
    
    // Click first signal card
    cy.get('[data-testid="signal-card"]').first().click()
    
    // Check if details modal opens
    cy.get('[data-testid="signal-details-modal"]').should('be.visible')
    cy.get('[data-testid="signal-details-modal"]').should('contain', 'Technical Analysis')
  })

  it('handles error states gracefully', () => {
    // Mock error response
    cy.intercept('GET', '/api/v1/signals', {
      statusCode: 500,
      body: { error: 'Internal Server Error' }
    }).as('getSignalsError')
    
    cy.visit('/signals')
    cy.wait('@getSignalsError')
    
    // Check error message
    cy.get('[data-testid="error-message"]').should('be.visible')
    cy.get('[data-testid="error-message"]').should('contain', 'Failed to load signals')
  })
})
