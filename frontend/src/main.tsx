/**
 * Golden Signals AI - Main Entry Point
 * 
 * Fixed entry point with proper App import and error handling.
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import { ErrorBoundary } from 'react-error-boundary';
import App from './App';
import './index.css';

// Error fallback component
const ErrorFallback: React.FC<{ error: Error; resetErrorBoundary: () => void }> = ({
    error,
    resetErrorBoundary
}) => (
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
        <h2 style={{ color: '#FFD700' }}>⚠️ Golden Eye AI Error</h2>
        <p style={{ marginBottom: '20px' }}>Something went wrong:</p>
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
            onClick={resetErrorBoundary}
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
            Try Again
        </button>
    </div>
);

// Clean, stable startup
ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <ErrorBoundary
            FallbackComponent={ErrorFallback}
            onError={(error, errorInfo) => {
                console.error('App Error:', error, errorInfo);
            }}
        >
            <App />
        </ErrorBoundary>
    </React.StrictMode>
); 