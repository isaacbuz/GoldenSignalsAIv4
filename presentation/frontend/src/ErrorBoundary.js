// ErrorBoundary.js
// Purpose: Provides an error boundary component for React. Catches and displays errors in child components, preventing the entire app from crashing. Useful for robust error handling and user feedback in GoldenSignalsAI frontend.

import React from "react";

class ErrorBoundary extends React.Component {
  // Initialize error state
  constructor(props) {
    super(props);
    // Initialize state to track if an error has occurred and store the error
    this.state = { hasError: false, error: null };
  }

  // Update state so the next render will show the fallback UI
  // React lifecycle: update state so the next render shows the fallback UI
  static getDerivedStateFromError(error) {
    // Update state to reflect the error and trigger fallback UI
    return { hasError: true, error };
  }

  // Optional: Log error details to an external service
  // React lifecycle: log error info if needed
  componentDidCatch(error, errorInfo) {
    // You can also log the error to an error reporting service
    // logErrorToService(error, errorInfo);
  }

  // Render fallback UI if an error was caught, otherwise render children
  render() {
    // Check if an error occurred and render fallback UI if so
    if (this.state.hasError) {
      // Render fallback UI with error message
      return (
        <div style={{ padding: 32, color: "#ff5252", background: "#292929", borderRadius: 8 }}>
          <h2>Something went wrong in the Admin Panel.</h2>
          <pre>{this.state.error && this.state.error.toString()}</pre>
        </div>
      );
    }
    // If no error occurred, render children as usual
    return this.props.children;
  }
}

export default ErrorBoundary;
