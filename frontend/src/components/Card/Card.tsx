import React, { ReactNode } from 'react';
import clsx from 'clsx';
import { logger as frontendLogger } from '../../services/logging/logger';
import styles from './Card.module.css';

export interface CardProps {
    /** Card content */
    children?: ReactNode;
    /** Card variant */
    variant?: 'elevated' | 'outlined' | 'contained';
    /** Additional CSS classes */
    className?: string;
    /** Header content */
    header?: ReactNode;
    /** Footer content */
    footer?: ReactNode;
    /** Media content (image, chart, etc.) */
    media?: ReactNode;
    /** Media position */
    mediaPosition?: 'top' | 'bottom' | 'left' | 'right';
    /** Action buttons */
    actions?: ReactNode;
    /** Whether the card is clickable */
    clickable?: boolean;
    /** Click handler */
    onClick?: () => void;
    /** Whether the card is loading */
    loading?: boolean;
    /** Whether to show a skeleton loader */
    skeleton?: boolean;
    /** Padding size */
    padding?: 'none' | 'small' | 'medium' | 'large';
    /** Whether the card is selected/active */
    selected?: boolean;
    /** Whether the card is disabled */
    disabled?: boolean;
    /** Border radius */
    borderRadius?: 'none' | 'small' | 'medium' | 'large';
    /** Shadow depth for elevated variant */
    elevation?: 0 | 1 | 2 | 3 | 4;
    /** Background color */
    backgroundColor?: string;
    /** Test ID for testing */
    'data-testid'?: string;
}

export const Card: React.FC<CardProps> = ({
    children,
    variant = 'elevated',
    className,
    header,
    footer,
    media,
    mediaPosition = 'top',
    actions,
    clickable = false,
    onClick,
    loading = false,
    skeleton = false,
    padding = 'medium',
    selected = false,
    disabled = false,
    borderRadius = 'medium',
    elevation = 1,
    backgroundColor,
    'data-testid': testId,
}) => {
    const handleClick = () => {
        if (disabled || !clickable || !onClick) return;

        frontendLogger.debug('Card clicked', { testId });
        onClick();
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (disabled || !clickable || !onClick) return;

        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            handleClick();
        }
    };

    const cardClasses = clsx(
        styles.card,
        styles[variant],
        styles[`padding-${padding}`],
        styles[`radius-${borderRadius}`],
        {
            [styles.clickable]: clickable && !disabled,
            [styles.loading]: loading,
            [styles.skeleton]: skeleton,
            [styles.selected]: selected,
            [styles.disabled]: disabled,
            [styles[`elevation-${elevation}`]]: variant === 'elevated',
            [styles[`media-${mediaPosition}`]]: media,
        },
        className
    );

    const cardStyle = backgroundColor ? { backgroundColor } : undefined;

    // Skeleton loader
    if (skeleton) {
        return (
            <div
                className={clsx(cardClasses, styles.skeletonCard)}
                data-testid={testId}
                aria-busy="true"
                aria-label="Loading content"
            >
                <div className={styles.skeletonHeader} />
                <div className={styles.skeletonContent}>
                    <div className={styles.skeletonLine} />
                    <div className={styles.skeletonLine} style={{ width: '80%' }} />
                    <div className={styles.skeletonLine} style={{ width: '60%' }} />
                </div>
            </div>
        );
    }

    const cardContent = (
        <>
            {loading && (
                <div className={styles.loadingOverlay}>
                    <div className={styles.spinner} />
                </div>
            )}

            {media && mediaPosition === 'top' && (
                <div className={styles.media}>{media}</div>
            )}

            {header && (
                <div className={styles.header}>{header}</div>
            )}

            {media && mediaPosition === 'left' && (
                <div className={styles.mediaWrapper}>
                    <div className={styles.media}>{media}</div>
                    <div className={styles.contentWrapper}>
                        {children && <div className={styles.content}>{children}</div>}
                        {actions && <div className={styles.actions}>{actions}</div>}
                    </div>
                </div>
            )}

            {media && mediaPosition === 'right' && (
                <div className={styles.mediaWrapper}>
                    <div className={styles.contentWrapper}>
                        {children && <div className={styles.content}>{children}</div>}
                        {actions && <div className={styles.actions}>{actions}</div>}
                    </div>
                    <div className={styles.media}>{media}</div>
                </div>
            )}

            {(!media || (media && (mediaPosition === 'top' || mediaPosition === 'bottom'))) && (
                <>
                    {children && <div className={styles.content}>{children}</div>}
                    {actions && <div className={styles.actions}>{actions}</div>}
                </>
            )}

            {media && mediaPosition === 'bottom' && (
                <div className={styles.media}>{media}</div>
            )}

            {footer && (
                <div className={styles.footer}>{footer}</div>
            )}
        </>
    );

    // Clickable card
    if (clickable && !disabled) {
        return (
            <div
                className={cardClasses}
                style={cardStyle}
                onClick={handleClick}
                onKeyDown={handleKeyDown}
                role="button"
                tabIndex={0}
                aria-pressed={selected}
                aria-disabled={disabled}
                data-testid={testId}
            >
                {cardContent}
            </div>
        );
    }

    // Regular card
    return (
        <div
            className={cardClasses}
            style={cardStyle}
            aria-disabled={disabled}
            data-testid={testId}
        >
            {cardContent}
        </div>
    );
};

// Sub-components for better composition
export const CardHeader: React.FC<{
    title: string;
    subtitle?: string;
    avatar?: ReactNode;
    action?: ReactNode;
    className?: string;
}> = ({ title, subtitle, avatar, action, className }) => (
    <div className={clsx(styles.cardHeader, className)}>
        {avatar && <div className={styles.avatar}>{avatar}</div>}
        <div className={styles.headerText}>
            <h3 className={styles.title}>{title}</h3>
            {subtitle && <p className={styles.subtitle}>{subtitle}</p>}
        </div>
        {action && <div className={styles.headerAction}>{action}</div>}
    </div>
);

export const CardActions: React.FC<{
    children: ReactNode;
    align?: 'left' | 'right' | 'center' | 'space-between';
    className?: string;
}> = ({ children, align = 'right', className }) => (
    <div className={clsx(styles.cardActions, styles[`align-${align}`], className)}>
        {children}
    </div>
);

export const CardMedia: React.FC<{
    src?: string;
    alt?: string;
    height?: number | string;
    children?: ReactNode;
    className?: string;
}> = ({ src, alt, height = 200, children, className }) => {
    if (src) {
        return (
            <img
                src={src}
                alt={alt || ''}
                className={clsx(styles.cardMediaImage, className)}
                style={{ height }}
                onError={(e) => {
                    frontendLogger.error('Card media failed to load', { src, alt });
                    (e.target as HTMLImageElement).style.display = 'none';
                }}
            />
        );
    }

    return (
        <div
            className={clsx(styles.cardMediaContent, className)}
            style={{ height }}
        >
            {children}
        </div>
    );
};
