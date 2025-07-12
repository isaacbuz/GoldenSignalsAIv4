import React from 'react';
import clsx from 'clsx';
import { logger as frontendLogger } from '../../services/logging/logger';
import styles from './Loading.module.css';

export interface LoadingProps {
    /** Type of loading indicator */
    type?: 'spinner' | 'dots' | 'bars' | 'skeleton' | 'progress' | 'pulse';
    /** Size of the loading indicator */
    size?: 'small' | 'medium' | 'large';
    /** Color variant */
    color?: 'primary' | 'secondary' | 'success' | 'danger' | 'warning' | 'info';
    /** Loading text to display */
    text?: string;
    /** Whether to show as overlay */
    overlay?: boolean;
    /** Whether to center the loading indicator */
    center?: boolean;
    /** Progress value (0-100) for progress type */
    progress?: number;
    /** Whether to show progress text */
    showProgressText?: boolean;
    /** Custom className */
    className?: string;
    /** Test ID */
    'data-testid'?: string;
}

export const Loading: React.FC<LoadingProps> = ({
    type = 'spinner',
    size = 'medium',
    color = 'primary',
    text,
    overlay = false,
    center = false,
    progress = 0,
    showProgressText = true,
    className,
    'data-testid': testId,
}) => {
    React.useEffect(() => {
        frontendLogger.debug('Loading component mounted', { type, size, color });
    }, [type, size, color]);

    const containerClasses = clsx(
        styles.container,
        {
            [styles.overlay]: overlay,
            [styles.center]: center,
        },
        className
    );

    const loaderClasses = clsx(
        styles.loader,
        styles[type],
        styles[size],
        styles[color]
    );

    const renderLoader = () => {
        switch (type) {
            case 'spinner':
                return (
                    <div className={loaderClasses} data-testid={`${testId}-spinner`}>
                        <svg viewBox="0 0 50 50" className={styles.spinnerSvg}>
                            <circle
                                cx="25"
                                cy="25"
                                r="20"
                                fill="none"
                                strokeWidth="4"
                                className={styles.spinnerCircle}
                            />
                        </svg>
                    </div>
                );

            case 'dots':
                return (
                    <div className={loaderClasses} data-testid={`${testId}-dots`}>
                        <span className={styles.dot} />
                        <span className={styles.dot} />
                        <span className={styles.dot} />
                    </div>
                );

            case 'bars':
                return (
                    <div className={loaderClasses} data-testid={`${testId}-bars`}>
                        <span className={styles.bar} />
                        <span className={styles.bar} />
                        <span className={styles.bar} />
                        <span className={styles.bar} />
                    </div>
                );

            case 'skeleton':
                return (
                    <div className={loaderClasses} data-testid={`${testId}-skeleton`}>
                        <div className={styles.skeletonHeader} />
                        <div className={styles.skeletonContent}>
                            <div className={styles.skeletonLine} />
                            <div className={styles.skeletonLine} style={{ width: '80%' }} />
                            <div className={styles.skeletonLine} style={{ width: '60%' }} />
                        </div>
                    </div>
                );

            case 'progress':
                return (
                    <div className={loaderClasses} data-testid={`${testId}-progress`}>
                        <div className={styles.progressBar}>
                            <div
                                className={styles.progressFill}
                                style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
                            />
                        </div>
                        {showProgressText && (
                            <span className={styles.progressText}>
                                {Math.round(progress)}%
                            </span>
                        )}
                    </div>
                );

            case 'pulse':
                return (
                    <div className={loaderClasses} data-testid={`${testId}-pulse`}>
                        <div className={styles.pulse} />
                    </div>
                );

            default:
                return null;
        }
    };

    return (
        <div className={containerClasses} data-testid={testId}>
            {renderLoader()}
            {text && <p className={styles.text}>{text}</p>}
        </div>
    );
};

// Skeleton sub-components for custom layouts
export const SkeletonText: React.FC<{
    lines?: number;
    width?: string | number;
    className?: string;
}> = ({ lines = 1, width, className }) => (
    <div className={clsx(styles.skeletonText, className)}>
        {Array.from({ length: lines }).map((_, i) => (
            <div
                key={i}
                className={styles.skeletonLine}
                style={{
                    width: width || (i === lines - 1 ? '60%' : '100%'),
                }}
            />
        ))}
    </div>
);

export const SkeletonAvatar: React.FC<{
    size?: 'small' | 'medium' | 'large';
    shape?: 'circle' | 'square';
    className?: string;
}> = ({ size = 'medium', shape = 'circle', className }) => (
    <div
        className={clsx(
            styles.skeletonAvatar,
            styles[size],
            styles[shape],
            className
        )}
    />
);

export const SkeletonButton: React.FC<{
    size?: 'small' | 'medium' | 'large';
    width?: string | number;
    className?: string;
}> = ({ size = 'medium', width, className }) => (
    <div
        className={clsx(styles.skeletonButton, styles[size], className)}
        style={{ width }}
    />
);

export const SkeletonCard: React.FC<{
    showImage?: boolean;
    showAvatar?: boolean;
    lines?: number;
    className?: string;
}> = ({ showImage = true, showAvatar = false, lines = 3, className }) => (
    <div className={clsx(styles.skeletonCard, className)}>
        {showImage && <div className={styles.skeletonImage} />}
        <div className={styles.skeletonCardContent}>
            {showAvatar && (
                <div className={styles.skeletonCardHeader}>
                    <SkeletonAvatar />
                    <div className={styles.skeletonCardHeaderText}>
                        <div className={styles.skeletonLine} style={{ width: '60%' }} />
                        <div className={styles.skeletonLine} style={{ width: '40%', height: '12px' }} />
                    </div>
                </div>
            )}
            <SkeletonText lines={lines} />
        </div>
    </div>
);

// Loading container for wrapping content
export const LoadingContainer: React.FC<{
    loading?: boolean;
    children: React.ReactNode;
    loadingProps?: LoadingProps;
    className?: string;
}> = ({ loading = false, children, loadingProps = {}, className }) => {
    if (loading) {
        return (
            <div className={clsx(styles.loadingContainer, className)}>
                <Loading overlay center {...loadingProps} />
            </div>
        );
    }

    return <>{children}</>;
}; 