/**
 * Test Utilities
 * Standard testing helpers and data-testid conventions
 */

import React from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../../theme';

// Standard data-testid convention
export const testIds = {
    // Navigation
    nav: {
        main: 'nav-main',
        dashboard: 'nav-dashboard',
        aiLab: 'nav-ai-lab',
        signals: 'nav-signals',
        portfolio: 'nav-portfolio',
        analytics: 'nav-analytics',
        settings: 'nav-settings',
        admin: 'nav-admin',
    },

    // Common components
    button: (name: string) => `btn-${name}`,
    input: (name: string) => `input-${name}`,
    select: (name: string) => `select-${name}`,
    modal: (name: string) => `modal-${name}`,
    dialog: (name: string) => `dialog-${name}`,
    tab: (name: string) => `tab-${name}`,
    table: (name: string) => `table-${name}`,
    form: (name: string) => `form-${name}`,

    // Layout components
    header: 'layout-header',
    sidebar: 'layout-sidebar',
    main: 'layout-main',
    footer: 'layout-footer',

    // Features
    signals: {
        dashboard: 'signals-dashboard',
        list: 'signals-list',
        card: (id: string) => `signal-card-${id}`,
        filter: 'signals-filter',
        search: 'signals-search',
        timeframe: 'signals-timeframe',
        details: (id: string) => `signal-details-${id}`,
    },

    aiLab: {
        container: 'ai-lab-container',
        tabs: 'ai-lab-tabs',
        signalGeneration: 'ai-lab-signal-generation',
        agentFleet: 'ai-lab-agent-fleet',
        backtesting: 'ai-lab-backtesting',
        strategyBuilder: 'ai-lab-strategy-builder',
        modelTraining: 'ai-lab-model-training',
        paperTrading: 'ai-lab-paper-trading',
    },

    chart: {
        container: (id: string) => `chart-${id}`,
        canvas: (id: string) => `chart-canvas-${id}`,
        controls: (id: string) => `chart-controls-${id}`,
        timeframe: (id: string) => `chart-timeframe-${id}`,
        indicators: (id: string) => `chart-indicators-${id}`,
    },

    portfolio: {
        overview: 'portfolio-overview',
        positions: 'portfolio-positions',
        performance: 'portfolio-performance',
        allocation: 'portfolio-allocation',
    },

    // State indicators
    loading: (component: string) => `${component}-loading`,
    error: (component: string) => `${component}-error`,
    empty: (component: string) => `${component}-empty`,
    success: (component: string) => `${component}-success`,
};

// Mock providers for testing
interface MockProvidersProps {
    children: React.ReactNode;
    initialRoute?: string;
}

export function MockProviders({ children, initialRoute = '/' }: MockProvidersProps) {
    const queryClient = new QueryClient({
        defaultOptions: {
            queries: {
                retry: false,
                gcTime: 0,
            },
        },
    });

    return (
        <QueryClientProvider client={queryClient}>
            <BrowserRouter>
                <ThemeProvider theme={theme}>
                    {children}
                </ThemeProvider>
            </BrowserRouter>
        </QueryClientProvider>
    );
}

// Custom render function
export function renderWithProviders(
    ui: React.ReactElement,
    options?: Omit<RenderOptions, 'wrapper'>
) {
    return render(ui, { wrapper: MockProviders, ...options });
}

// Mock WebSocket for testing
export class MockWebSocket {
    url: string;
    readyState: number = WebSocket.CONNECTING;
    onopen?: (event: Event) => void;
    onclose?: (event: CloseEvent) => void;
    onerror?: (event: Event) => void;
    onmessage?: (event: MessageEvent) => void;

    constructor(url: string) {
        this.url = url;
        setTimeout(() => {
            this.readyState = WebSocket.OPEN;
            if (this.onopen) {
                this.onopen(new Event('open'));
            }
        }, 0);
    }

    send(data: string) {
        // Mock send implementation
    }

    close() {
        this.readyState = WebSocket.CLOSED;
        if (this.onclose) {
            this.onclose(new CloseEvent('close'));
        }
    }

    // Helper to simulate incoming messages
    simulateMessage(data: any) {
        if (this.onmessage) {
            this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }));
        }
    }
}

// Mock API responses
export const mockApiResponses = {
    signals: {
        success: [
            {
                id: 'sig-1',
                symbol: 'AAPL',
                type: 'BUY',
                confidence: 0.85,
                timestamp: new Date().toISOString(),
                agents: ['technical', 'sentiment', 'ml'],
            },
            {
                id: 'sig-2',
                symbol: 'GOOGL',
                type: 'SELL',
                confidence: 0.72,
                timestamp: new Date().toISOString(),
                agents: ['technical', 'risk'],
            },
        ],
        error: {
            error: 'Failed to fetch signals',
            message: 'Network error',
        },
    },

    marketData: {
        success: {
            symbol: 'AAPL',
            price: 185.50,
            change: 2.35,
            changePercent: 1.28,
            volume: 45000000,
            high: 186.20,
            low: 183.10,
        },
        error: {
            error: 'Symbol not found',
            message: 'Invalid symbol',
        },
    },
};

// Helper to wait for async updates
export const waitForAsync = () => new Promise(resolve => setTimeout(resolve, 0));

// Helper to mock console methods during tests
export function mockConsole() {
    const originalConsole = { ...console };

    beforeAll(() => {
        console.error = jest.fn();
        console.warn = jest.fn();
        console.log = jest.fn();
    });

    afterAll(() => {
        console.error = originalConsole.error;
        console.warn = originalConsole.warn;
        console.log = originalConsole.log;
    });

    return {
        getErrors: () => (console.error as jest.Mock).mock.calls,
        getWarnings: () => (console.warn as jest.Mock).mock.calls,
        getLogs: () => (console.log as jest.Mock).mock.calls,
    };
}

// Export everything from testing library for convenience
export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event'; 