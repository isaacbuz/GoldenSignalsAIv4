import React, { useEffect, useRef, useCallback, ReactNode } from 'react';
import { createPortal } from 'react-dom';
import clsx from 'clsx';
import { logger as frontendLogger } from '../../services/logging/logger';
import { Button } from '../Core/Button/Button';
import styles from './Modal.module.css';

export interface ModalProps {
    /** Whether the modal is open */
    isOpen: boolean;
    /** Callback when modal should close */
    onClose: () => void;
    /** Modal title */
    title?: string;
    /** Modal content */
    children: ReactNode;
    /** Modal size */
    size?: 'small' | 'medium' | 'large' | 'fullscreen';
    /** Whether to show close button */
    showCloseButton?: boolean;
    /** Whether clicking backdrop closes modal */
    closeOnBackdropClick?: boolean;
    /** Whether pressing Escape closes modal */
    closeOnEscape?: boolean;
    /** Custom footer content */
    footer?: ReactNode;
    /** Custom className for modal */
    className?: string;
    /** Custom className for content */
    contentClassName?: string;
    /** Z-index for modal */
    zIndex?: number;
    /** Animation type */
    animation?: 'fade' | 'slide' | 'scale' | 'none';
    /** Test ID */
    'data-testid'?: string;
}

export const Modal: React.FC<ModalProps> = ({
    isOpen,
    onClose,
    title,
    children,
    size = 'medium',
    showCloseButton = true,
    closeOnBackdropClick = true,
    closeOnEscape = true,
    footer,
    className,
    contentClassName,
    zIndex = 1000,
    animation = 'scale',
    'data-testid': testId,
}) => {
    const modalRef = useRef<HTMLDivElement>(null);
    const previousActiveElement = useRef<HTMLElement | null>(null);

    useEffect(() => {
        frontendLogger.debug('Modal state changed', { isOpen, title });
    }, [isOpen, title]);

    // Handle escape key
    useEffect(() => {
        if (!isOpen || !closeOnEscape) return;

        const handleEscape = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                onClose();
            }
        };

        document.addEventListener('keydown', handleEscape);
        return () => document.removeEventListener('keydown', handleEscape);
    }, [isOpen, closeOnEscape, onClose]);

    // Focus management
    useEffect(() => {
        if (isOpen) {
            previousActiveElement.current = document.activeElement as HTMLElement;

            // Focus first focusable element in modal
            setTimeout(() => {
                const focusableElements = modalRef.current?.querySelectorAll(
                    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
                );
                if (focusableElements?.length) {
                    (focusableElements[0] as HTMLElement).focus();
                }
            }, 100);
        } else {
            // Restore focus to previous element
            previousActiveElement.current?.focus();
        }
    }, [isOpen]);

    // Prevent body scroll when modal is open
    useEffect(() => {
        if (isOpen) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }

        return () => {
            document.body.style.overflow = '';
        };
    }, [isOpen]);

    const handleBackdropClick = useCallback((e: React.MouseEvent) => {
        if (closeOnBackdropClick && e.target === e.currentTarget) {
            onClose();
        }
    }, [closeOnBackdropClick, onClose]);

    if (!isOpen) return null;

    const modalContent = (
        <div
            className={clsx(
                styles.backdrop,
                styles[`animation-${animation}`],
                className
            )}
            onClick={handleBackdropClick}
            style={{ zIndex }}
            data-testid={`${testId}-backdrop`}
        >
            <div
                ref={modalRef}
                className={clsx(
                    styles.modal,
                    styles[size],
                    styles[`animation-${animation}`],
                    contentClassName
                )}
                role="dialog"
                aria-modal="true"
                aria-labelledby={title ? 'modal-title' : undefined}
                data-testid={testId}
            >
                {(title || showCloseButton) && (
                    <div className={styles.header}>
                        {title && (
                            <h2 id="modal-title" className={styles.title}>
                                {title}
                            </h2>
                        )}
                        {showCloseButton && (
                            <button
                                className={styles.closeButton}
                                onClick={onClose}
                                aria-label="Close modal"
                                data-testid={`${testId}-close`}
                            >
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
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
                )}

                <div className={styles.content}>{children}</div>

                {footer && <div className={styles.footer}>{footer}</div>}
            </div>
        </div>
    );

    return createPortal(modalContent, document.body);
};

// Confirmation modal component
export interface ConfirmModalProps {
    isOpen: boolean;
    onClose: () => void;
    onConfirm: () => void;
    title?: string;
    message: string;
    confirmText?: string;
    cancelText?: string;
    variant?: 'danger' | 'warning' | 'info';
    loading?: boolean;
}

export const ConfirmModal: React.FC<ConfirmModalProps> = ({
    isOpen,
    onClose,
    onConfirm,
    title = 'Confirm Action',
    message,
    confirmText = 'Confirm',
    cancelText = 'Cancel',
    variant = 'info',
    loading = false,
}) => {
    const handleConfirm = () => {
        frontendLogger.info('Confirm modal action confirmed', { title });
        onConfirm();
    };

    return (
        <Modal
            isOpen={isOpen}
            onClose={onClose}
            title={title}
            size="small"
            data-testid="confirm-modal"
        >
            <p className={styles.confirmMessage}>{message}</p>
            <div className={styles.confirmActions}>
                <Button
                    variant="ghost"
                    onClick={onClose}
                    disabled={loading}
                >
                    {cancelText}
                </Button>
                <Button
                    variant={variant === 'danger' ? 'danger' : 'primary'}
                    onClick={handleConfirm}
                    loading={loading}
                >
                    {confirmText}
                </Button>
            </div>
        </Modal>
    );
};

// Alert modal component
export interface AlertModalProps {
    isOpen: boolean;
    onClose: () => void;
    title?: string;
    message: string;
    variant?: 'success' | 'error' | 'warning' | 'info';
    closeText?: string;
}

export const AlertModal: React.FC<AlertModalProps> = ({
    isOpen,
    onClose,
    title = 'Alert',
    message,
    variant = 'info',
    closeText = 'OK',
}) => {
    const getIcon = () => {
        switch (variant) {
            case 'success':
                return (
                    <svg className={styles.alertIconSuccess} viewBox="0 0 24 24" fill="none">
                        <path
                            d="M9 12L11 14L15 10M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        />
                    </svg>
                );
            case 'error':
                return (
                    <svg className={styles.alertIconError} viewBox="0 0 24 24" fill="none">
                        <path
                            d="M12 8V12M12 16H12.01M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        />
                    </svg>
                );
            case 'warning':
                return (
                    <svg className={styles.alertIconWarning} viewBox="0 0 24 24" fill="none">
                        <path
                            d="M12 9V13M12 17H12.01M5.07183 19H18.9282C20.4678 19 21.4301 17.3333 20.6603 16L13.7321 4C12.9623 2.66667 11.0377 2.66667 10.2679 4L3.33975 16C2.56995 17.3333 3.53223 19 5.07183 19Z"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        />
                    </svg>
                );
            default:
                return (
                    <svg className={styles.alertIconInfo} viewBox="0 0 24 24" fill="none">
                        <path
                            d="M13 16H12V12H11M12 8H12.01M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        />
                    </svg>
                );
        }
    };

    return (
        <Modal
            isOpen={isOpen}
            onClose={onClose}
            size="small"
            showCloseButton={false}
            data-testid="alert-modal"
        >
            <div className={styles.alertContent}>
                <div className={styles.alertIcon}>{getIcon()}</div>
                <h3 className={styles.alertTitle}>{title}</h3>
                <p className={styles.alertMessage}>{message}</p>
                <Button
                    variant={variant === 'error' ? 'danger' : 'primary'}
                    onClick={onClose}
                    fullWidth
                >
                    {closeText}
                </Button>
            </div>
        </Modal>
    );
};
