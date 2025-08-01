/**
 * Responsive Typography System
 * Provides responsive font sizes that scale across different screen sizes
 * with slightly reduced base sizes for better information density
 */

import { TypographyOptions } from '@mui/material/styles/createTypography';
import { Breakpoint } from '@mui/material/styles';

// Responsive font size helper
export const responsiveFontSizes = {
    // Headings - reduced by additional 15-20%
    h1: {
        xs: '1.75rem',   // 28px on mobile (was 32px)
        sm: '2rem',      // 32px on tablet
        md: '2.25rem',   // 36px on desktop
        lg: '2.5rem',    // 40px on large screens
        xl: '2.75rem',   // 44px on extra large
    },
    h2: {
        xs: '1.25rem',   // 20px (was 24px)
        sm: '1.5rem',    // 24px
        md: '1.75rem',   // 28px
        lg: '2rem',      // 32px
        xl: '2.25rem',   // 36px
    },
    h3: {
        xs: '1.125rem',  // 18px (was 20px)
        sm: '1.25rem',   // 20px
        md: '1.375rem',  // 22px
        lg: '1.5rem',    // 24px
        xl: '1.75rem',   // 28px
    },
    h4: {
        xs: '1rem',      // 16px (was 18px)
        sm: '1.125rem',  // 18px
        md: '1.25rem',   // 20px
        lg: '1.375rem',  // 22px
        xl: '1.5rem',    // 24px
    },
    h5: {
        xs: '0.875rem',  // 14px (was 16px)
        sm: '1rem',      // 16px
        md: '1.0625rem', // 17px
        lg: '1.125rem',  // 18px
        xl: '1.25rem',   // 20px
    },
    h6: {
        xs: '0.8125rem', // 13px (was 14px)
        sm: '0.875rem',  // 14px
        md: '0.9375rem', // 15px
        lg: '1rem',      // 16px
        xl: '1.0625rem', // 17px
    },
    // Body text - reduced by additional 15%
    body1: {
        xs: '0.75rem',   // 12px (was 13px)
        sm: '0.8125rem', // 13px
        md: '0.875rem',  // 14px
        lg: '0.9375rem', // 15px
        xl: '0.9375rem', // 15px
    },
    body2: {
        xs: '0.6875rem', // 11px (was 12px)
        sm: '0.75rem',   // 12px
        md: '0.8125rem', // 13px
        lg: '0.875rem',  // 14px
        xl: '0.875rem',  // 14px
    },
    subtitle1: {
        xs: '0.8125rem', // 13px (was 14px)
        sm: '0.875rem',  // 14px
        md: '0.9375rem', // 15px
        lg: '1rem',      // 16px
        xl: '1.0625rem', // 17px
    },
    subtitle2: {
        xs: '0.75rem',   // 12px (was 13px)
        sm: '0.8125rem', // 13px
        md: '0.875rem',  // 14px
        lg: '0.9375rem', // 15px
        xl: '0.9375rem', // 15px
    },
    caption: {
        xs: '0.625rem',  // 10px (was 11px)
        sm: '0.6875rem', // 11px
        md: '0.6875rem', // 11px
        lg: '0.75rem',   // 12px
        xl: '0.75rem',   // 12px
    },
    button: {
        xs: '0.6875rem', // 11px (was 12px)
        sm: '0.75rem',   // 12px
        md: '0.8125rem', // 13px
        lg: '0.8125rem', // 13px
        xl: '0.875rem',  // 14px
    },
    overline: {
        xs: '0.5625rem', // 9px (was 10px)
        sm: '0.625rem',  // 10px
        md: '0.6875rem', // 11px
        lg: '0.6875rem', // 11px
        xl: '0.75rem',   // 12px
    },
};

// Helper to create responsive typography variant
export const createResponsiveTypography = (
    sizes: Record<Breakpoint, string>
): any => ({
    fontSize: sizes.md, // Default to medium size
    '@media (max-width:600px)': {
        fontSize: sizes.xs,
    },
    '@media (min-width:600px) and (max-width:900px)': {
        fontSize: sizes.sm,
    },
    '@media (min-width:900px) and (max-width:1200px)': {
        fontSize: sizes.md,
    },
    '@media (min-width:1200px) and (max-width:1536px)': {
        fontSize: sizes.lg,
    },
    '@media (min-width:1536px)': {
        fontSize: sizes.xl,
    },
});

// Create responsive typography configuration
export const createResponsiveTypographyOptions = (): TypographyOptions => ({
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    h1: {
        ...createResponsiveTypography(responsiveFontSizes.h1),
        fontWeight: 700,
        lineHeight: 1.2,
        letterSpacing: '-0.02em',
    },
    h2: {
        ...createResponsiveTypography(responsiveFontSizes.h2),
        fontWeight: 600,
        lineHeight: 1.3,
        letterSpacing: '-0.01em',
    },
    h3: {
        ...createResponsiveTypography(responsiveFontSizes.h3),
        fontWeight: 600,
        lineHeight: 1.4,
        letterSpacing: '-0.005em',
    },
    h4: {
        ...createResponsiveTypography(responsiveFontSizes.h4),
        fontWeight: 600,
        lineHeight: 1.4,
    },
    h5: {
        ...createResponsiveTypography(responsiveFontSizes.h5),
        fontWeight: 600,
        lineHeight: 1.5,
    },
    h6: {
        ...createResponsiveTypography(responsiveFontSizes.h6),
        fontWeight: 600,
        lineHeight: 1.5,
    },
    body1: {
        ...createResponsiveTypography(responsiveFontSizes.body1),
        fontWeight: 400,
        lineHeight: 1.6,
    },
    body2: {
        ...createResponsiveTypography(responsiveFontSizes.body2),
        fontWeight: 400,
        lineHeight: 1.5,
    },
    subtitle1: {
        ...createResponsiveTypography(responsiveFontSizes.subtitle1),
        fontWeight: 500,
        lineHeight: 1.5,
    },
    subtitle2: {
        ...createResponsiveTypography(responsiveFontSizes.subtitle2),
        fontWeight: 500,
        lineHeight: 1.4,
    },
    caption: {
        ...createResponsiveTypography(responsiveFontSizes.caption),
        fontWeight: 400,
        lineHeight: 1.4,
        letterSpacing: '0.01em',
    },
    button: {
        ...createResponsiveTypography(responsiveFontSizes.button),
        fontWeight: 600,
        lineHeight: 1.5,
        textTransform: 'none',
        letterSpacing: '0.02em',
    },
    overline: {
        ...createResponsiveTypography(responsiveFontSizes.overline),
        fontWeight: 600,
        lineHeight: 1.5,
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
    },
});

// Responsive spacing helper
export const responsiveSpacing = {
    xs: (factor: number) => factor * 0.75,  // 75% on mobile
    sm: (factor: number) => factor * 0.875, // 87.5% on tablet
    md: (factor: number) => factor,         // 100% on desktop
    lg: (factor: number) => factor * 1.125, // 112.5% on large
    xl: (factor: number) => factor * 1.25,  // 125% on extra large
};

// Component-specific responsive sizes
export const componentSizes = {
    // Button sizes - reduced further
    button: {
        small: {
            xs: { padding: '3px 6px', fontSize: '0.6875rem' }, // Reduced
            sm: { padding: '3px 8px', fontSize: '0.75rem' },
            md: { padding: '3px 10px', fontSize: '0.75rem' },
        },
        medium: {
            xs: { padding: '4px 8px', fontSize: '0.75rem' }, // Reduced
            sm: { padding: '4px 10px', fontSize: '0.8125rem' },
            md: { padding: '4px 12px', fontSize: '0.8125rem' },
        },
        large: {
            xs: { padding: '6px 12px', fontSize: '0.8125rem' }, // Reduced
            sm: { padding: '6px 16px', fontSize: '0.875rem' },
            md: { padding: '6px 20px', fontSize: '0.9375rem' },
        },
    },
    // Icon sizes - reduced
    icon: {
        small: { xs: '1rem', sm: '1.125rem', md: '1.25rem' }, // Reduced
        medium: { xs: '1.125rem', sm: '1.25rem', md: '1.375rem' },
        large: { xs: '1.375rem', sm: '1.5rem', md: '1.75rem' },
    },
    // Card padding - reduced
    card: {
        xs: '8px', // Reduced from 12px
        sm: '12px', // Reduced from 16px
        md: '16px', // Reduced from 20px
        lg: '20px', // Reduced from 24px
    },
    // Chip sizes - new addition
    chip: {
        small: {
            xs: { height: '16px', fontSize: '0.625rem' },
            sm: { height: '18px', fontSize: '0.6875rem' },
            md: { height: '20px', fontSize: '0.75rem' },
        },
        medium: {
            xs: { height: '20px', fontSize: '0.6875rem' },
            sm: { height: '22px', fontSize: '0.75rem' },
            md: { height: '24px', fontSize: '0.8125rem' },
        },
    },
    // Form control sizes - new addition
    formControl: {
        small: {
            xs: { height: '24px', fontSize: '0.6875rem' },
            sm: { height: '26px', fontSize: '0.75rem' },
            md: { height: '28px', fontSize: '0.75rem' },
        },
        medium: {
            xs: { height: '28px', fontSize: '0.75rem' },
            sm: { height: '32px', fontSize: '0.8125rem' },
            md: { height: '36px', fontSize: '0.875rem' },
        },
    },
};

// Export utility function for responsive values
export const getResponsiveValue = (
    breakpoint: 'xs' | 'sm' | 'md' | 'lg' | 'xl',
    values: Record<string, any>
) => values[breakpoint] || values.md;
