/**
 * Golden Signals AI - Simplified App
 *
 * New simplified, stable architecture that replaces the complex previous version.
 * Single entry point with Golden Eye AI Prophet theme preserved.
 */

import React from 'react';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { Toaster } from 'react-hot-toast';
import { ErrorBoundary } from 'react-error-boundary';

// New simplified architecture
import { goldenEyeTheme } from '../theme/goldenEye';
import GoldenSignalsDashboard from './GoldenSignalsDashboard';

// Error fallback component
const ErrorFallback: React.FC<{ error: Error }> = ({ error }) => (
    <div style={{
        padding: '20px',
        textAlign: 'center',
        backgroundColor: '#0A0E1A',
        color: '#E2E8F0',
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column',
    }}>
        <h2 style={{ color: '#FFD700' }}>⚠️ Something went wrong</h2>
        <p style={{ marginBottom: '20px' }}>
            The Golden Eye AI Prophet has encountered an error:
        </p>
        <pre style={{
            backgroundColor: '#131A2A',
            padding: '15px',
            borderRadius: '8px',
            maxWidth: '80%',
            overflow: 'auto',
            fontSize: '14px',
        }}>
            {error.message}
        </pre>
        <button
            onClick={() => window.location.reload()}
            style={{
                marginTop: '20px',
                padding: '10px 20px',
                backgroundColor: '#FFD700',
                color: '#0A0E1A',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: 'bold',
            }}
        >
            Reload Application
        </button>
    </div>
);

// Main App component
const App: React.FC = () => {
    return (
        <ErrorBoundary FallbackComponent={ErrorFallback}>
            <ThemeProvider theme={goldenEyeTheme}>
                <CssBaseline />

                {/* Toast notifications */}
                <Toaster
                    position="top-right"
                    toastOptions={{
                        style: {
                            background: '#131A2A',
                            color: '#E2E8F0',
                            border: '1px solid rgba(255, 215, 0, 0.2)',
                        },
                        success: {
                            iconTheme: {
                                primary: '#00D4AA',
                                secondary: '#0A0E1A',
                            },
                        },
                        error: {
                            iconTheme: {
                                primary: '#FF4757',
                                secondary: '#0A0E1A',
                            },
                        },
                    }}
                />

                {/* Main Dashboard */}
                <GoldenSignalsDashboard />
            </ThemeProvider>
        </ErrorBoundary>
    );
};

export default App;
