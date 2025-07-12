describe('Signals Dashboard E2E Tests', () => {
  beforeEach(() => {
    // Mock API responses
    cy.intercept('GET', '/api/v1/precise-options-signals/SPY/15m', { fixture: 'signals.json' }).as('getPreciseOptionsSignals')
    cy.intercept('GET', '/api/v1/market-data/SPY', { fixture: 'market-data.json' }).as('getMarketData')
    cy.intercept('GET', '/api/v1/market/opportunities', { fixture: 'opportunities.json' }).as('getOpportunities')

    // Visit signals dashboard
    cy.visit('/signals')
  })

  it('displays signals dashboard correctly', () => {
    // Check page title
    cy.contains('h4', 'Signals Dashboard').should('be.visible')

    // Wait for signals to load
    cy.wait('@getPreciseOptionsSignals')

    // Check if signals are displayed
    cy.get('[data-testid="signal-card"]').should('have.length.at.least', 1)
  })

  it('filters signals by symbol', () => {
    cy.wait('@getPreciseOptionsSignals')

    // Type in search bar
    cy.get('[data-testid="symbol-search"]').type('AAPL')

    // Check filtered results
    cy.get('[data-testid="signal-card"]').each(($card) => {
      cy.wrap($card).should('contain', 'AAPL')
    })
  })

  it('filters signals by action type', () => {
    cy.wait('@getPreciseOptionsSignals')

    // Click BUY filter
    cy.get('[data-testid="filter-buy"]').click()

    // Check filtered results
    cy.get('[data-testid="signal-card"]').each(($card) => {
      cy.wrap($card).find('[data-testid="signal-action"]').should('contain', 'BUY')
    })
  })

  it('displays signal details on click', () => {
    cy.wait('@getPreciseOptionsSignals')

    // Click first signal card
    cy.get('[data-testid="signal-card"]').first().click()

    // Check if details modal opens
    cy.get('[data-testid="signal-details-modal"]').should('be.visible')
    cy.get('[data-testid="signal-details-modal"]').should('contain', 'Technical Analysis')
  })

  it('renders the trading chart', () => {
    cy.wait('@getMarketData');
    cy.get('[data-testid="trading-chart"]').should('be.visible');
    // More specific chart assertions can be added here
  });

  it('opens and interacts with the AI chat', () => {
    // Mock the AI chat response
    cy.intercept('POST', '/api/v1/ai/chat', {
      body: {
        content: "This is a mocked AI response.",
        metadata: {}
      }
    }).as('postAIChat');

    // Click the AI Assistant button
    cy.get('[data-testid="ai-assistant-button"]').click();

    // Check if the chat drawer is open
    cy.get('[data-testid="ai-chat-drawer"]').should('be.visible');

    // Type a message and send it
    cy.get('[data-testid="ai-chat-input"]').type('Hello, AI!');
    cy.get('[data-testid="ai-chat-send-button"]').click();

    // Wait for the mocked response and check if it's displayed
    cy.wait('@postAIChat');
    cy.get('[data-testid="ai-chat-messages"]').should('contain', 'This is a mocked AI response.');
  });

  it('handles error states gracefully', () => {
    // Mock error response
    cy.intercept('GET', '/api/v1/precise-options-signals/SPY/15m', {
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
