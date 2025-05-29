import React from 'react';

interface ErrorBoundaryState {
  hasError: boolean;
}

export class ErrorBoundary extends React.Component<{ children: React.ReactNode }, ErrorBoundaryState> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: any, info: any) {
    // Optionally log error to an external service
    console.error('ErrorBoundary caught an error:', error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center min-h-screen text-red-400 bg-bgDark">
          <div>
            <h1 className="text-2xl font-bold mb-2">Something went wrong.</h1>
            <p>We're unable to display the dashboard. Please try refreshing or check your network connection.</p>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
