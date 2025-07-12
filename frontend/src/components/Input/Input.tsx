import React, { useState, useRef, forwardRef, InputHTMLAttributes } from 'react';
import clsx from 'clsx';
import { logger as frontendLogger } from '../../services/logging/logger';
import styles from './Input.module.css';

export interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
    /** Label for the input field */
    label?: string;
    /** Input variant */
    variant?: 'outlined' | 'filled' | 'standard';
    /** Size of the input */
    size?: 'small' | 'medium' | 'large';
    /** Error state and message */
    error?: boolean | string;
    /** Helper text to display below the input */
    helperText?: string;
    /** Icon to display at the start of the input */
    startIcon?: React.ReactNode;
    /** Icon to display at the end of the input */
    endIcon?: React.ReactNode;
    /** Whether the input should take full width */
    fullWidth?: boolean;
    /** Whether to show character count */
    showCount?: boolean;
    /** Maximum character length */
    maxLength?: number;
    /** Callback when input loses focus */
    onBlur?: (e: React.FocusEvent<HTMLInputElement>) => void;
    /** Callback when input gains focus */
    onFocus?: (e: React.FocusEvent<HTMLInputElement>) => void;
    /** Test ID for testing */
    'data-testid'?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(({
    label,
    variant = 'outlined',
    size = 'medium',
    error = false,
    helperText,
    startIcon,
    endIcon,
    fullWidth = false,
    showCount = false,
    maxLength,
    className,
    disabled,
    type = 'text',
    value,
    onChange,
    onBlur,
    onFocus,
    'data-testid': testId,
    ...props
}, ref) => {
    const [isFocused, setIsFocused] = useState(false);
    const [inputValue, setInputValue] = useState(value || '');
    const inputRef = useRef<HTMLInputElement>(null);

    // Combine refs
    React.useImperativeHandle(ref, () => inputRef.current!);

    const handleFocus = (e: React.FocusEvent<HTMLInputElement>) => {
        setIsFocused(true);
        onFocus?.(e);
        frontendLogger.debug('Input focused', {
            label,
            value: e.target.value,
            testId
        });
    };

    const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
        setIsFocused(false);
        onBlur?.(e);
        frontendLogger.debug('Input blurred', {
            label,
            value: e.target.value,
            testId
        });
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newValue = e.target.value;

        if (maxLength && newValue.length > maxLength) {
            frontendLogger.warn('Input max length exceeded', {
                label,
                maxLength,
                attemptedLength: newValue.length
            });
            return;
        }

        setInputValue(newValue);
        onChange?.(e);
    };

    const errorMessage = typeof error === 'string' ? error : helperText;
    const hasError = !!error;
    const characterCount = String(inputValue).length;

    const inputClasses = clsx(
        styles.input,
        styles[variant],
        styles[size],
        {
            [styles.error]: hasError,
            [styles.disabled]: disabled,
            [styles.focused]: isFocused,
            [styles.hasStartIcon]: !!startIcon,
            [styles.hasEndIcon]: !!endIcon,
            [styles.fullWidth]: fullWidth,
        },
        className
    );

    const containerClasses = clsx(
        styles.container,
        {
            [styles.fullWidth]: fullWidth,
        }
    );

    const handleContainerClick = () => {
        inputRef.current?.focus();
    };

    return (
        <div className={containerClasses} data-testid={`${testId}-container`}>
            {label && (
                <label
                    className={clsx(
                        styles.label,
                        {
                            [styles.focused]: isFocused || !!inputValue,
                            [styles.error]: hasError,
                            [styles.disabled]: disabled,
                            [styles[size]]: size,
                        }
                    )}
                    htmlFor={props.id}
                >
                    {label}
                    {props.required && <span className={styles.required}>*</span>}
                </label>
            )}

            <div
                className={clsx(styles.inputWrapper, styles[variant])}
                onClick={handleContainerClick}
            >
                {startIcon && (
                    <span className={clsx(styles.icon, styles.startIcon)}>
                        {startIcon}
                    </span>
                )}

                <input
                    ref={inputRef}
                    type={type}
                    value={inputValue}
                    onChange={handleChange}
                    onFocus={handleFocus}
                    onBlur={handleBlur}
                    disabled={disabled}
                    className={inputClasses}
                    data-testid={testId}
                    aria-invalid={hasError}
                    aria-describedby={errorMessage ? `${props.id}-error` : undefined}
                    {...props}
                />

                {endIcon && (
                    <span className={clsx(styles.icon, styles.endIcon)}>
                        {endIcon}
                    </span>
                )}
            </div>

            {(errorMessage || (showCount && maxLength)) && (
                <div className={styles.footer}>
                    {errorMessage && (
                        <span
                            id={`${props.id}-error`}
                            className={clsx(
                                styles.helperText,
                                { [styles.error]: hasError }
                            )}
                            role={hasError ? 'alert' : undefined}
                        >
                            {errorMessage}
                        </span>
                    )}

                    {showCount && maxLength && (
                        <span className={styles.count}>
                            {characterCount}/{maxLength}
                        </span>
                    )}
                </div>
            )}
        </div>
    );
});

Input.displayName = 'Input'; 