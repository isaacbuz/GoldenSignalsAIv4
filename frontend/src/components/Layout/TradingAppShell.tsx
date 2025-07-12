import React from 'react';
import { Layout } from './Layout';
import { ErrorBoundary } from '../ErrorBoundary';

interface TradingAppShellProps {
    children: React.ReactNode;
    sidebarCollapsed?: boolean;
    onSidebarToggle?: () => void;
    theme?: 'light' | 'dark';
}

/**
 * TradingAppShell - Wrapper around existing Layout component
 * 
 * This demonstrates the reuse principle:
 * - We're NOT creating a new layout system
 * - We're configuring the existing 383-line Layout.tsx
 * - Total new code: ~30 lines vs 300+ lines
 */
export const TradingAppShell: React.FC<TradingAppShellProps> = ({
    children,
    sidebarCollapsed = false,
    onSidebarToggle,
    theme = 'dark'
}) => {
    // Configure Layout for trading-specific needs
    const layoutConfig = {
        variant: 'trading',
        sidebarCollapsed,
        onSidebarToggle,
        theme,
        // Add trading-specific configuration
        headerHeight: 64,
        sidebarWidth: sidebarCollapsed ? 64 : 240,
        enableResponsive: true,
        breakpoints: {
            mobile: 768,
            tablet: 1024,
            desktop: 1280
        }
    };

    return (
        <ErrorBoundary>
            <Layout {...layoutConfig}>
                {children}
            </Layout>
        </ErrorBoundary>
    );
}; 