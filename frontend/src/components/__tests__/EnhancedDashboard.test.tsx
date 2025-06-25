import React from 'react';
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import EnhancedDashboard from '../Dashboard/EnhancedDashboard';
import { store } from '../../store';
import { WebSocketService } from '../../services/websocket';

// Mock WebSocket
jest.mock('../../services/websocket');

// Mock chart components
jest.mock('react-chartjs-2', () => ({
    Line: () => <div data-testid="line-chart">Line Chart</div>,
    Bar: () => <div data-testid="bar-chart">Bar Chart</div>,
    Doughnut: () => <div data-testid="doughnut-chart">Doughnut Chart</div>,
}));

// Mock API calls
jest.mock('../../services/api', () => ({
    getMarketData: jest.fn().mockResolvedValue({
        data: {
            symbols: ['AAPL', 'GOOGL', 'MSFT'],
            prices: { AAPL: 150.25, GOOGL: 2750.50, MSFT: 325.75 },
            changes: { AAPL: 2.5, GOOGL: -1.2, MSFT: 0.8 }
        }
    }),
    getSignals: jest.fn().mockResolvedValue({
        data: [
            {
                id: '1',
                symbol: 'AAPL',
                type: 'BUY',
                confidence: 0.85,
                price: 150.00,
                timestamp: new Date().toISOString()
            },
            {
                id: '2',
                symbol: 'GOOGL',
                type: 'SELL',
                confidence: 0.72,
                price: 2760.00,
                timestamp: new Date().toISOString()
            }
        ]
    }),
    getPortfolioMetrics: jest.fn().mockResolvedValue({
        data: {
            totalValue: 100000,
            dayChange: 1250,
            dayChangePercent: 1.25,
            positions: [
                { symbol: 'AAPL', shares: 100, value: 15025 },
                { symbol: 'MSFT', shares: 50, value: 16287.50 }
            ]
        }
    })
}));

const queryClient = new QueryClient({
    defaultOptions: {
        queries: { retry: false },
    },
});

const theme = createTheme();

const renderWithProviders = (component: React.ReactElement) => {
    return render(
        <QueryClientProvider client={queryClient}>
            <Provider store={store}>
                <ThemeProvider theme={theme}>
                    <BrowserRouter>
                        {component}
                    </BrowserRouter>
                </ThemeProvider>
            </Provider>
        </QueryClientProvider>
    );
};

describe('EnhancedDashboard', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        // Reset WebSocket mock
        (WebSocketService as jest.Mock).mockImplementation(() => ({
            connect: jest.fn(),
            disconnect: jest.fn(),
            subscribe: jest.fn(),
            unsubscribe: jest.fn(),
        }));
    });

    afterEach(() => {
        queryClient.clear();
    });

    test('renders dashboard with all main sections', async () => {
        renderWithProviders(<EnhancedDashboard />);

        // Check for main sections
        expect(screen.getByText(/Market Overview/i)).toBeInTheDocument();
        expect(screen.getByText(/Portfolio Performance/i)).toBeInTheDocument();
        expect(screen.getByText(/Active Signals/i)).toBeInTheDocument();
        expect(screen.getByText(/Agent Consensus/i)).toBeInTheDocument();
    });

    test('displays loading state initially', () => {
        renderWithProviders(<EnhancedDashboard />);

        expect(screen.getByTestId('dashboard-skeleton')).toBeInTheDocument();
    });

    test('displays market data after loading', async () => {
        renderWithProviders(<EnhancedDashboard />);

        await waitFor(() => {
            expect(screen.getByText('AAPL')).toBeInTheDocument();
            expect(screen.getByText('$150.25')).toBeInTheDocument();
            expect(screen.getByText('+2.50%')).toBeInTheDocument();
        });

        expect(screen.getByText('GOOGL')).toBeInTheDocument();
        expect(screen.getByText('$2,750.50')).toBeInTheDocument();
        expect(screen.getByText('-1.20%')).toBeInTheDocument();
    });

    test('displays portfolio metrics correctly', async () => {
        renderWithProviders(<EnhancedDashboard />);

        await waitFor(() => {
            expect(screen.getByText('$100,000')).toBeInTheDocument();
            expect(screen.getByText('+$1,250')).toBeInTheDocument();
            expect(screen.getByText('+1.25%')).toBeInTheDocument();
        });
    });

    test('shows active signals with correct formatting', async () => {
        renderWithProviders(<EnhancedDashboard />);

        await waitFor(() => {
            const signalCards = screen.getAllByTestId('signal-card');
            expect(signalCards).toHaveLength(2);
        });

        // Check buy signal
        const buySignal = screen.getByTestId('signal-1');
        expect(within(buySignal).getByText('BUY')).toBeInTheDocument();
        expect(within(buySignal).getByText('AAPL')).toBeInTheDocument();
        expect(within(buySignal).getByText('85%')).toBeInTheDocument();

        // Check sell signal
        const sellSignal = screen.getByTestId('signal-2');
        expect(within(sellSignal).getByText('SELL')).toBeInTheDocument();
        expect(within(sellSignal).getByText('GOOGL')).toBeInTheDocument();
        expect(within(sellSignal).getByText('72%')).toBeInTheDocument();
    });

    test('handles WebSocket connection', async () => {
        const mockWs = {
            connect: jest.fn(),
            disconnect: jest.fn(),
            subscribe: jest.fn(),
            unsubscribe: jest.fn(),
        };
        (WebSocketService as jest.Mock).mockImplementation(() => mockWs);

        renderWithProviders(<EnhancedDashboard />);

        await waitFor(() => {
            expect(mockWs.connect).toHaveBeenCalled();
            expect(mockWs.subscribe).toHaveBeenCalledWith('market-updates', expect.any(Function));
            expect(mockWs.subscribe).toHaveBeenCalledWith('signals', expect.any(Function));
        });
    });

    test('handles error states gracefully', async () => {
        const consoleError = jest.spyOn(console, 'error').mockImplementation();
        const api = require('../../services/api');
        api.getMarketData.mockRejectedValueOnce(new Error('Network error'));

        renderWithProviders(<EnhancedDashboard />);

        await waitFor(() => {
            expect(screen.getByText(/Error loading market data/i)).toBeInTheDocument();
        });

        consoleError.mockRestore();
    });

    test('refreshes data on interval', async () => {
        jest.useFakeTimers();
        const api = require('../../services/api');

        renderWithProviders(<EnhancedDashboard />);

        await waitFor(() => {
            expect(api.getMarketData).toHaveBeenCalledTimes(1);
        });

        // Fast-forward 5 minutes
        jest.advanceTimersByTime(5 * 60 * 1000);

        await waitFor(() => {
            expect(api.getMarketData).toHaveBeenCalledTimes(2);
        });

        jest.useRealTimers();
    });

    test('handles user interactions - changing time period', async () => {
        renderWithProviders(<EnhancedDashboard />);

        await waitFor(() => {
            expect(screen.getByRole('button', { name: /1D/i })).toBeInTheDocument();
        });

        // Click on 1W button
        const weekButton = screen.getByRole('button', { name: /1W/i });
        fireEvent.click(weekButton);

        expect(weekButton).toHaveClass('Mui-selected');
    });

    test('displays agent consensus visualization', async () => {
        renderWithProviders(<EnhancedDashboard />);

        await waitFor(() => {
            expect(screen.getByTestId('doughnut-chart')).toBeInTheDocument();
        });

        // Check for agent labels
        expect(screen.getByText(/Technical Analysis/i)).toBeInTheDocument();
        expect(screen.getByText(/Sentiment Analysis/i)).toBeInTheDocument();
        expect(screen.getByText(/Options Flow/i)).toBeInTheDocument();
    });

    test('handles signal filtering', async () => {
        renderWithProviders(<EnhancedDashboard />);

        await waitFor(() => {
            expect(screen.getAllByTestId('signal-card')).toHaveLength(2);
        });

        // Click on filter button
        const filterButton = screen.getByRole('button', { name: /Filter/i });
        fireEvent.click(filterButton);

        // Select only BUY signals
        const buyCheckbox = screen.getByRole('checkbox', { name: /Buy Signals/i });
        fireEvent.click(buyCheckbox);

        await waitFor(() => {
            expect(screen.getAllByTestId('signal-card')).toHaveLength(1);
            expect(screen.getByText('BUY')).toBeInTheDocument();
            expect(screen.queryByText('SELL')).not.toBeInTheDocument();
        });
    });

    test('opens signal detail modal on click', async () => {
        renderWithProviders(<EnhancedDashboard />);

        await waitFor(() => {
            expect(screen.getByTestId('signal-1')).toBeInTheDocument();
        });

        fireEvent.click(screen.getByTestId('signal-1'));

        await waitFor(() => {
            expect(screen.getByRole('dialog')).toBeInTheDocument();
            expect(screen.getByText(/Signal Details/i)).toBeInTheDocument();
            expect(screen.getByText(/Confidence: 85%/i)).toBeInTheDocument();
        });
    });

    test('handles portfolio position updates via WebSocket', async () => {
        const mockWs = {
            connect: jest.fn(),
            disconnect: jest.fn(),
            subscribe: jest.fn((channel, callback) => {
                if (channel === 'portfolio-updates') {
                    // Simulate WebSocket message
                    setTimeout(() => {
                        callback({
                            type: 'position_update',
                            data: {
                                symbol: 'AAPL',
                                shares: 150,
                                value: 22537.50
                            }
                        });
                    }, 100);
                }
            }),
            unsubscribe: jest.fn(),
        };
        (WebSocketService as jest.Mock).mockImplementation(() => mockWs);

        renderWithProviders(<EnhancedDashboard />);

        await waitFor(() => {
            expect(screen.getByText('150 shares')).toBeInTheDocument();
            expect(screen.getByText('$22,537.50')).toBeInTheDocument();
        });
    });

    test('exports data when export button is clicked', async () => {
        global.URL.createObjectURL = jest.fn();
        global.URL.revokeObjectURL = jest.fn();
        const mockClick = jest.fn();
        HTMLAnchorElement.prototype.click = mockClick;

        renderWithProviders(<EnhancedDashboard />);

        await waitFor(() => {
            expect(screen.getByRole('button', { name: /Export/i })).toBeInTheDocument();
        });

        fireEvent.click(screen.getByRole('button', { name: /Export/i }));

        expect(mockClick).toHaveBeenCalled();
        expect(global.URL.createObjectURL).toHaveBeenCalled();
    });

    test('handles dark mode toggle', async () => {
        renderWithProviders(<EnhancedDashboard />);

        const themeToggle = screen.getByRole('button', { name: /Toggle theme/i });
        fireEvent.click(themeToggle);

        expect(document.body).toHaveClass('dark-mode');
    });

    test('cleans up on unmount', async () => {
        const mockWs = {
            connect: jest.fn(),
            disconnect: jest.fn(),
            subscribe: jest.fn(),
            unsubscribe: jest.fn(),
        };
        (WebSocketService as jest.Mock).mockImplementation(() => mockWs);

        const { unmount } = renderWithProviders(<EnhancedDashboard />);

        await waitFor(() => {
            expect(mockWs.connect).toHaveBeenCalled();
        });

        unmount();

        expect(mockWs.unsubscribe).toHaveBeenCalledWith('market-updates');
        expect(mockWs.unsubscribe).toHaveBeenCalledWith('signals');
        expect(mockWs.disconnect).toHaveBeenCalled();
    });
});

describe('EnhancedDashboard Accessibility', () => {
    test('has proper ARIA labels', async () => {
        renderWithProviders(<EnhancedDashboard />);

        await waitFor(() => {
            expect(screen.getByRole('main')).toHaveAttribute('aria-label', 'Trading Dashboard');
            expect(screen.getByRole('region', { name: /Market Overview/i })).toBeInTheDocument();
            expect(screen.getByRole('region', { name: /Portfolio Performance/i })).toBeInTheDocument();
        });
    });

    test('supports keyboard navigation', async () => {
        const user = userEvent.setup();
        renderWithProviders(<EnhancedDashboard />);

        await waitFor(() => {
            expect(screen.getByRole('button', { name: /1D/i })).toBeInTheDocument();
        });

        // Tab through interactive elements
        await user.tab();
        expect(screen.getByRole('button', { name: /1D/i })).toHaveFocus();

        await user.tab();
        expect(screen.getByRole('button', { name: /1W/i })).toHaveFocus();

        // Activate with Enter key
        await user.keyboard('{Enter}');
        expect(screen.getByRole('button', { name: /1W/i })).toHaveClass('Mui-selected');
    });
}); 