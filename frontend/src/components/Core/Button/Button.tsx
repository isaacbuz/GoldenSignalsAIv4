import React, { forwardRef, ButtonHTMLAttributes, ReactNode } from 'react';
import { logger } from '@/services/logging/logger';
import styles from './Button.module.css';

export interface ButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'size'> {
    /**
     * The visual style variant of the button
     */
    variant?: 'primary' | 'secondary' | 'ghost' | 'danger';

    /**
     * The size of the button
     */
    size?: 'small' | 'medium' | 'large';

    /**
     * Whether the button should take full width
     */
    fullWidth?: boolean;

    /**
     * Loading state - shows spinner and disables interaction
     */
    loading?: boolean;

    /**
     * Icon to display before the text
     */
    startIcon?: React.ReactNode;

    /**
     * Icon to display after the text
     */
    endIcon?: React.ReactNode;

    /**
     * Ripple effect on click
     */
    ripple?: boolean;

    children: ReactNode;

    'data-testid'?: string;
}

/**
 * Core Button component with multiple variants and states
 *
 * @example
 * ```tsx
 * <Button variant="primary" size="medium" onClick={handleClick}>
 *   Click me
 * </Button>
 * ```
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
    (
        {
            variant = 'primary',
            size = 'medium',
            fullWidth = false,
            loading = false,
            startIcon,
            endIcon,
            ripple = true,
            className = '',
            children,
            disabled,
            onClick,
            'data-testid': testId,
            ...props
        },
        ref
    ) => {
        const [ripples, setRipples] = React.useState<Array<{ x: number; y: number; id: number }>>([]);

        const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
            if (ripple && !disabled && !loading) {
                const rect = event.currentTarget.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;
                const id = Date.now();

                setRipples(prev => [...prev, { x, y, id }]);

                // Remove ripple after animation
                setTimeout(() => {
                    setRipples(prev => prev.filter(r => r.id !== id));
                }, 600);
            }

            if (onClick && !disabled && !loading) {
                logger.debug('Button clicked', {
                    variant,
                    size,
                    testId: testId || 'unknown'
                });
                onClick(event);
            }
        };

        const buttonClasses = [
            styles.button,
            styles[variant],
            styles[size],
            fullWidth && styles.fullWidth,
            loading && styles.loading,
            disabled && styles.disabled,
            className
        ].filter(Boolean).join(' ');

        return (
            <button
                ref={ref}
                className={buttonClasses}
                disabled={disabled || loading}
                onClick={handleClick}
                data-testid={testId || `button-${variant}`}
                {...props}
            >
                {loading && (
                    <span className={styles.spinner} data-testid="button-spinner">
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <circle
                                cx="12"
                                cy="12"
                                r="10"
                                stroke="currentColor"
                                strokeWidth="3"
                                strokeLinecap="round"
                                strokeDasharray="31.4 31.4"
                                transform="rotate(-90 12 12)"
                                className={styles.spinnerCircle}
                            />
                        </svg>
                    </span>
                )}

                {startIcon && !loading && (
                    <span className={styles.startIcon} data-testid="button-start-icon">
                        {startIcon}
                    </span>
                )}

                <span className={styles.label}>{children}</span>

                {endIcon && !loading && (
                    <span className={styles.endIcon} data-testid="button-end-icon">
                        {endIcon}
                    </span>
                )}

                {ripple && ripples.map(({ x, y, id }) => (
                    <span
                        key={id}
                        className={styles.ripple}
                        style={{
                            left: x,
                            top: y,
                        }}
                    />
                ))}
            </button>
        );
    }
);

Button.displayName = 'Button';
