import React, { Component, ErrorInfo, ReactNode } from 'react';
import clsx from 'clsx';
import { logger as frontendLogger } from '../../services/logging/logger';
import { Button } from '../Core/Button/Button';
import { Card } from '../Card/Card';
import styles from './Error.module.css';
import { Typography } from '@mui/material';

// Error types
export type ErrorType = 'inline' | 'page' | 'card' | 'toast';
export type ErrorSeverity = 'error' | 'warning' | 'info';

export interface ErrorProps {
    /** Error type/variant */
    type?: ErrorType;
    /** Error severity */
    severity?: ErrorSeverity;
    /** Error title */
    title?: string;
    /** Error message/description */
    message?: ReactNode;
    /** Error details (technical info) */
    details?: string | Error;
    /** Whether to show details */
    showDetails?: boolean;
    /** Action buttons */
    actions?: ReactNode;
    /** Icon to display */
    icon?: ReactNode;
    /** Whether error is dismissible */
    dismissible?: boolean;
    /** Callback when dismissed */
    onDismiss?: () => void;
    /** Callback for retry action */
    onRetry?: () => void;
    /** Custom className */
    className?: string;
    /** Test ID */
    'data-testid'?: string;
}

// Error display component
export const Error: React.FC<ErrorProps> = ({
    type = 'inline',
    severity = 'error',
    title = 'Something went wrong',
    message,
    details,
    showDetails = false,
    actions,
    icon,
    dismissible = false,
    onDismiss,
    onRetry,
    className,
    'data-testid': testId,
}) => {
    const [detailsExpanded, setDetailsExpanded] = React.useState(showDetails);

    React.useEffect(() => {
        frontendLogger.error('Error component rendered', {
            type,
            severity,
            title,
            message,
            details: details instanceof Error ? details.message : details,
        });
    }, [type, severity, title, message, details]);

    const containerClasses = clsx(
        styles.container,
        styles[type],
        styles[severity],
        className
    );

    const renderIcon = () => {
        if (icon) return icon;

        return (
            <svg className={styles.defaultIcon} viewBox="0 0 24 24" fill="none">
                {severity === 'error' && (
                    <path
                        d="M12 8V12M12 16H12.01M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    />
                )}
                {severity === 'warning' && (
                    <path
                        d="M12 9V13M12 17H12.01M5.07183 19H18.9282C20.4678 19 21.4301 17.3333 20.6603 16L13.7321 4C12.9623 2.66667 11.0377 2.66667 10.2679 4L3.33975 16C2.56995 17.3333 3.53223 19 5.07183 19Z"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    />
                )}
                {severity === 'info' && (
                    <path
                        d="M13 16H12V12H11M12 8H12.01M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    />
                )}
            </svg>
        );
    };

    const renderActions = () => {
        if (actions) return actions;

        return (
            <div className={styles.actions}>
                {onRetry && (
                    <Button
                        variant={severity === 'error' ? 'danger' : 'primary'}
                        size="small"
                        onClick={onRetry}
                    >
                        Try Again
                    </Button>
                )}
                {dismissible && onDismiss && type !== 'page' && (
                    <Button variant="ghost" size="small" onClick={onDismiss}>
                        Dismiss
                    </Button>
                )}
            </div>
        );
    };

    const renderDetails = () => {
        if (!details) return null;

        const detailsText = details instanceof Error
            ? `${details.name}: ${details.message}\n${details.stack}`
            : details;

        return (
            <div className={styles.details}>
                <button
                    className={styles.detailsToggle}
                    onClick={() => setDetailsExpanded(!detailsExpanded)}
                >
                    {detailsExpanded ? 'Hide' : 'Show'} details
                    <svg
                        className={clsx(styles.chevron, { [styles.expanded]: detailsExpanded })}
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                    >
                        <path
                            d="M19 9L12 16L5 9"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        />
                    </svg>
                </button>
                {detailsExpanded && (
                    <pre className={styles.detailsContent}>{detailsText}</pre>
                )}
            </div>
        );
    };

    if (type === 'page') {
        return (
            <div className={containerClasses} data-testid={testId}>
                <div className={styles.pageContent}>
                    <div className={styles.iconWrapper}>{renderIcon()}</div>
                    <h1 className={styles.title}>{title}</h1>
                    {message && <p className={styles.message}>{message}</p>}
                    {renderActions()}
                    {renderDetails()}
                </div>
            </div>
        );
    }

    if (type === 'card') {
        return (
            <Card className={containerClasses} data-testid={testId}>
                <div className={styles.cardContent}>
                    <div className={styles.header}>
                        <div className={styles.iconWrapper}>{renderIcon()}</div>
                        <div className={styles.textWrapper}>
                            <h3 className={styles.title}>{title}</h3>
                            {message && <p className={styles.message}>{message}</p>}
                        </div>
                        {dismissible && onDismiss && (
                            <button className={styles.dismissButton} onClick={onDismiss}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                                    <path
                                        d="M18 6L6 18M6 6L18 18"
                                        stroke="currentColor"
                                        strokeWidth="2"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                    />
                                </svg>
                            </button>
                        )}
                    </div>
                    {renderDetails()}
                    {(actions || onRetry) && renderActions()}
                </div>
            </Card>
        );
    }

    // Default inline error
    return (
        <div className={containerClasses} role="alert" data-testid={testId}>
            <div className={styles.iconWrapper}>{renderIcon()}</div>
            <div className={styles.content}>
                <div className={styles.textWrapper}>
                    {title && <strong className={styles.title}>{title}</strong>}
                    {message && <span className={styles.message}>{message}</span>}
                </div>
                {renderDetails()}
            </div>
            {(actions || onRetry || (dismissible && onDismiss)) && renderActions()}
        </div>
    );
};

// Error Boundary Component
interface ErrorBoundaryProps {
    children: ReactNode;
    fallback?: (error: Error, reset: () => void) => ReactNode;
    onError?: (error: Error, errorInfo: ErrorInfo) => void;
    className?: string;
}

interface ErrorBoundaryState {
    hasError: boolean;
    error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
    constructor(props: ErrorBoundaryProps) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error: Error): ErrorBoundaryState {
        return { hasError: true, error };
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        frontendLogger.error('ErrorBoundary caught error', {
            error: error.message,
            stack: error.stack,
            componentStack: errorInfo.componentStack,
        });

        this.props.onError?.(error, errorInfo);
    }

    handleReset = () => {
        this.setState({ hasError: false, error: null });
    };

    render() {
        if (this.state.hasError && this.state.error) {
            if (this.props.fallback) {
                return this.props.fallback(this.state.error, this.handleReset);
            }

            return (
                <Error
                    type="card"
                    severity="error"
                    title="Application Error"
                    message="An unexpected error occurred. The error has been logged and we'll look into it."
                    details={this.state.error}
                    onRetry={this.handleReset}
                    className={this.props.className}
                />
            );
        }

        return this.props.children;
    }
}

// Common error pages
export const NotFoundError: React.FC<{ onGoHome?: () => void }> = ({ onGoHome }) => (
    <Error
        type="page"
        severity="warning"
        title="404 - Page Not Found"
        message="The page you're looking for doesn't exist or has been moved."
        actions={
            <Button variant="primary" onClick={onGoHome}>
                Go to Homepage
            </Button>
        }
        data-testid="404-error"
    />
);

export const ServerError: React.FC<{ onRetry?: () => void }> = ({ onRetry }) => (
    <Error
        type="page"
        severity="error"
        title="500 - Server Error"
        message="Something went wrong on our end. Please try again later."
        onRetry={onRetry}
        data-testid="500-error"
    />
);

export const NetworkError: React.FC<{ onRetry?: () => void }> = ({ onRetry }) => (
    <Error
        type="card"
        severity="error"
        title="Network Error"
        message="Unable to connect to the server. Please check your internet connection."
        onRetry={onRetry}
        data-testid="network-error"
    />
);

export const ValidationError: React.FC<{ errors: string[] }> = ({ errors }) => (
    <Error
        type="inline"
        severity="error"
        title="Validation Error"
        message={
            <ul className={styles.errorList}>
                {errors.map((error, index) => (
                    <li key={index}>{error}</li>
                ))}
            </ul>
        }
        data-testid="validation-error"
    />
);

export const ErrorDetails: React.FC<{ error: Error | string }> = ({ error }) => (
    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
        {typeof error === 'string' ? error : error.message}
    </Typography>
);
