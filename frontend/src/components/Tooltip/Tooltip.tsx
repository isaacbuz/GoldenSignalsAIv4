import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';

export interface TooltipProps {
    content: React.ReactNode;
    children: React.ReactElement;
    position?: 'top' | 'bottom' | 'left' | 'right';
    delay?: number;
    offset?: number;
    className?: string;
    disabled?: boolean;
    arrow?: boolean;
    maxWidth?: number;
}

interface TooltipPosition {
    top: number;
    left: number;
    arrowTop?: number;
    arrowLeft?: number;
}

/**
 * Tooltip Component
 * 
 * A lightweight, accessible tooltip that appears on hover/focus.
 * Reuses patterns from other components:
 * - Portal rendering (like Alert)
 * - Framer Motion animations (like Modal/Alert)
 * - Positioning logic similar to dropdown menus
 */
export const Tooltip: React.FC<TooltipProps> = ({
    content,
    children,
    position = 'top',
    delay = 500,
    offset = 8,
    className = '',
    disabled = false,
    arrow = true,
    maxWidth = 200
}) => {
    const [isVisible, setIsVisible] = useState(false);
    const [tooltipPosition, setTooltipPosition] = useState<TooltipPosition>({ top: 0, left: 0 });
    const triggerRef = useRef<HTMLElement>(null);
    const tooltipRef = useRef<HTMLDivElement>(null);
    const timeoutRef = useRef<NodeJS.Timeout>();

    const calculatePosition = () => {
        if (!triggerRef.current || !tooltipRef.current) return;

        const triggerRect = triggerRef.current.getBoundingClientRect();
        const tooltipRect = tooltipRef.current.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        let top = 0;
        let left = 0;
        let arrowTop;
        let arrowLeft;

        // Calculate position based on preferred position
        switch (position) {
            case 'top':
                top = triggerRect.top - tooltipRect.height - offset;
                left = triggerRect.left + (triggerRect.width - tooltipRect.width) / 2;
                arrowTop = tooltipRect.height;
                arrowLeft = tooltipRect.width / 2;
                break;
            case 'bottom':
                top = triggerRect.bottom + offset;
                left = triggerRect.left + (triggerRect.width - tooltipRect.width) / 2;
                arrowTop = -4;
                arrowLeft = tooltipRect.width / 2;
                break;
            case 'left':
                top = triggerRect.top + (triggerRect.height - tooltipRect.height) / 2;
                left = triggerRect.left - tooltipRect.width - offset;
                arrowTop = tooltipRect.height / 2;
                arrowLeft = tooltipRect.width;
                break;
            case 'right':
                top = triggerRect.top + (triggerRect.height - tooltipRect.height) / 2;
                left = triggerRect.right + offset;
                arrowTop = tooltipRect.height / 2;
                arrowLeft = -4;
                break;
        }

        // Adjust if tooltip goes outside viewport
        if (left < 8) left = 8;
        if (left + tooltipRect.width > viewportWidth - 8) {
            left = viewportWidth - tooltipRect.width - 8;
        }
        if (top < 8) top = 8;
        if (top + tooltipRect.height > viewportHeight - 8) {
            top = viewportHeight - tooltipRect.height - 8;
        }

        setTooltipPosition({ top, left, arrowTop, arrowLeft });
    };

    const showTooltip = () => {
        if (disabled) return;

        clearTimeout(timeoutRef.current);
        timeoutRef.current = setTimeout(() => {
            setIsVisible(true);
        }, delay);
    };

    const hideTooltip = () => {
        clearTimeout(timeoutRef.current);
        setIsVisible(false);
    };

    useEffect(() => {
        if (isVisible) {
            calculatePosition();
            window.addEventListener('scroll', calculatePosition);
            window.addEventListener('resize', calculatePosition);

            return () => {
                window.removeEventListener('scroll', calculatePosition);
                window.removeEventListener('resize', calculatePosition);
            };
        }
    }, [isVisible]);

    useEffect(() => {
        return () => {
            clearTimeout(timeoutRef.current);
        };
    }, []);

    if (disabled || !content) {
        return children;
    }

    // Clone child element and add event handlers
    const trigger = React.cloneElement(children, {
        ref: triggerRef,
        onMouseEnter: (e: React.MouseEvent) => {
            children.props.onMouseEnter?.(e);
            showTooltip();
        },
        onMouseLeave: (e: React.MouseEvent) => {
            children.props.onMouseLeave?.(e);
            hideTooltip();
        },
        onFocus: (e: React.FocusEvent) => {
            children.props.onFocus?.(e);
            showTooltip();
        },
        onBlur: (e: React.FocusEvent) => {
            children.props.onBlur?.(e);
            hideTooltip();
        },
        'aria-describedby': isVisible ? 'tooltip' : undefined
    });

    const arrowClasses = {
        top: 'bottom-0 left-1/2 -translate-x-1/2 translate-y-full border-t-gray-900 dark:border-t-gray-100 border-x-transparent border-b-transparent',
        bottom: 'top-0 left-1/2 -translate-x-1/2 -translate-y-full border-b-gray-900 dark:border-b-gray-100 border-x-transparent border-t-transparent',
        left: 'right-0 top-1/2 -translate-y-1/2 translate-x-full border-l-gray-900 dark:border-l-gray-100 border-y-transparent border-r-transparent',
        right: 'left-0 top-1/2 -translate-y-1/2 -translate-x-full border-r-gray-900 dark:border-r-gray-100 border-y-transparent border-l-transparent'
    };

    return (
        <>
            {trigger}
            {createPortal(
                <AnimatePresence>
                    {isVisible && (
                        <motion.div
                            ref={tooltipRef}
                            id="tooltip"
                            role="tooltip"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            transition={{ duration: 0.1 }}
                            style={{
                                position: 'fixed',
                                top: tooltipPosition.top,
                                left: tooltipPosition.left,
                                maxWidth,
                                zIndex: 9999,
                            }}
                            className={`
                px-3 py-2 text-sm font-medium text-gray-100 dark:text-gray-900
                bg-gray-900 dark:bg-gray-100 rounded-lg shadow-lg
                pointer-events-none select-none
                ${className}
              `}
                        >
                            {content}
                            {arrow && (
                                <div
                                    className={`absolute w-0 h-0 border-4 ${arrowClasses[position]}`}
                                    style={{
                                        top: tooltipPosition.arrowTop,
                                        left: tooltipPosition.arrowLeft,
                                    }}
                                />
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>,
                document.body
            )}
        </>
    );
};

// Convenience wrapper for simple text tooltips
export const TooltipText: React.FC<{
    text: string;
    children: React.ReactElement;
    position?: TooltipProps['position'];
}> = ({ text, children, position }) => (
    <Tooltip content={text} position={position}>
        {children}
    </Tooltip>
);

// Wrapper for icon tooltips with reduced delay
export const IconTooltip: React.FC<{
    content: React.ReactNode;
    children: React.ReactElement;
    position?: TooltipProps['position'];
}> = ({ content, children, position }) => (
    <Tooltip content={content} position={position} delay={200}>
        {children}
    </Tooltip>
);

// Rich content tooltip wrapper
export const RichTooltip: React.FC<{
    title?: string;
    description?: string;
    children: React.ReactElement;
    position?: TooltipProps['position'];
}> = ({ title, description, children, position }) => (
    <Tooltip
        content={
            <div>
                {title && <div className="font-semibold mb-1">{title}</div>}
                {description && <div className="text-xs opacity-90">{description}</div>}
            </div>
        }
        position={position}
        maxWidth={300}
    >
        {children}
    </Tooltip>
); 