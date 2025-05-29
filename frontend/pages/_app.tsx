/* @jsxRuntime automatic */
import '../styles/globals.css';
import type { AppProps } from 'next/app';

import { TickerProvider } from '../context/TickerContext';
import { ErrorBoundary } from '../components/ErrorBoundary';
import { AgentOrchestrator } from '../components/agent/AgentOrchestrator';
// import { Toaster } from 'react-hot-toast';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <ErrorBoundary>
      <AgentOrchestrator>
        <TickerProvider>
          <Component {...pageProps} />
        </TickerProvider>
      </AgentOrchestrator>
    </ErrorBoundary>
  );
}
