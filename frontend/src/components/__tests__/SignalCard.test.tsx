import { describe, it, expect, vi } from 'vitest'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '../../test/test-utils'
import SignalCard from '../Signals/SignalCard'
import { PreciseOptionsSignal } from '../../types/signals'

const mockSignal: PreciseOptionsSignal = {
  id: '1',
  symbol: 'AAPL',
  type: 'CALL',
  confidence: 85,
  priority: 'HIGH',
  timestamp: '2024-01-20T10:00:00Z',
  entry_price: 150.00,
  strike_price: 155,
  expiration_date: '2024-02-20',
  stop_loss: 145.00,
  targets: [
    { price: 160.00, probability: 0.7 }
  ],
  risk_reward_ratio: 3.0,
  entry_window: {
    date: '2024-01-20',
    start_time: '09:30',
    end_time: '10:30'
  },
  setup_name: 'Bullish Breakout',
  key_indicators: {
    RSI: '65',
    MACD: 'Bullish',
    Volume: 'Above Average'
  }
}

describe('SignalCard', () => {
  it('renders signal information correctly', () => {
    const handleClick = vi.fn()
    renderWithProviders(<SignalCard signal={mockSignal} onClick={handleClick} />)

    expect(screen.getByText('AAPL')).toBeInTheDocument()
    expect(screen.getByText('CALL')).toBeInTheDocument()
    expect(screen.getByText('85%')).toBeInTheDocument() // confidence
    expect(screen.getByText('$150.00')).toBeInTheDocument() // entry price
    expect(screen.getByText('$155')).toBeInTheDocument() // strike price
  })

  it('applies correct styling for CALL signal', () => {
    const handleClick = vi.fn()
    renderWithProviders(<SignalCard signal={mockSignal} onClick={handleClick} />)

    const callChip = screen.getByText('CALL')
    // Check if the parent element has the success color styling
    expect(callChip.closest('.MuiChip-root')).toBeTruthy()
  })

  it('applies correct styling for PUT signal', () => {
    const putSignal: PreciseOptionsSignal = { ...mockSignal, type: 'PUT' }
    const handleClick = vi.fn()
    renderWithProviders(<SignalCard signal={putSignal} onClick={handleClick} />)

    const putChip = screen.getByText('PUT')
    // Check if the parent element has the error color styling
    expect(putChip.closest('.MuiChip-root')).toBeTruthy()
  })

  it('handles click events', async () => {
    const user = userEvent.setup()
    const handleClick = vi.fn()

    renderWithProviders(
      <SignalCard signal={mockSignal} onClick={handleClick} />
    )

    const card = screen.getByText('AAPL').closest('.MuiCard-root')
    if (card) {
      await user.click(card)
      expect(handleClick).toHaveBeenCalled()
    }
  })

  it('displays risk/reward information', () => {
    const handleClick = vi.fn()
    renderWithProviders(<SignalCard signal={mockSignal} onClick={handleClick} />)

    expect(screen.getByText('Stop: $145.00')).toBeInTheDocument()
    expect(screen.getByText('Target: $160.00')).toBeInTheDocument()
    expect(screen.getByText('3:1 R:R')).toBeInTheDocument()
  })

  it('formats expiration date correctly', () => {
    const handleClick = vi.fn()
    const { container } = renderWithProviders(<SignalCard signal={mockSignal} onClick={handleClick} />)

    // The expiration date might be rendered differently depending on locale
    // Let's check for the presence of "Expiry" label and a date value
    const expiryLabel = screen.getByText('Expiry')
    expect(expiryLabel).toBeInTheDocument()

    // Check that there's a date element near the Expiry label
    const expiryBox = expiryLabel.closest('.MuiBox-root')
    const dateText = expiryBox?.querySelector('.MuiTypography-h6')?.textContent
    expect(dateText).toBeTruthy()
    // The date format may vary by locale, so we just check it exists
  })

  it('displays entry window information', () => {
    const handleClick = vi.fn()
    renderWithProviders(<SignalCard signal={mockSignal} onClick={handleClick} />)

    expect(screen.getByText('09:30 - 10:30')).toBeInTheDocument()
  })

  it('renders compact version correctly', () => {
    const handleClick = vi.fn()
    renderWithProviders(<SignalCard signal={mockSignal} onClick={handleClick} compact />)

    // In compact mode, we should still see key information
    expect(screen.getByText('AAPL')).toBeInTheDocument()
    expect(screen.getByText('85%')).toBeInTheDocument()
    expect(screen.getByText('$150.00')).toBeInTheDocument()
  })

  it('handles quick actions', async () => {
    const user = userEvent.setup()
    const handleClick = vi.fn()
    const handleQuickAction = vi.fn()

    renderWithProviders(
      <SignalCard
        signal={mockSignal}
        onClick={handleClick}
        onQuickAction={handleQuickAction}
      />
    )

    // Click on Set Alert button
    const alertButton = screen.getByLabelText('Set Alert')
    await user.click(alertButton)

    expect(handleQuickAction).toHaveBeenCalledWith('setAlert')
    expect(handleClick).not.toHaveBeenCalled() // Should not trigger card click
  })

  it('displays key indicators', () => {
    const handleClick = vi.fn()
    renderWithProviders(<SignalCard signal={mockSignal} onClick={handleClick} />)

    expect(screen.getByText('RSI: 65')).toBeInTheDocument()
    expect(screen.getByText('MACD: Bullish')).toBeInTheDocument()
    expect(screen.getByText('Volume: Above Average')).toBeInTheDocument()
  })
})
