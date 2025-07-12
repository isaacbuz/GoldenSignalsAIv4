/**
 * Responsive Layout Utilities
 * Provides helpers for creating responsive components and layouts
 */

import { SxProps, Theme } from '@mui/material/styles';

// Breakpoint values
export const breakpoints = {
    xs: 0,
    sm: 600,
    md: 900,
    lg: 1200,
    xl: 1536,
};

// Responsive padding/margin helper
export const responsivePadding = (base: number): SxProps<Theme> => ({
    p: {
        xs: base * 0.75,  // 75% on mobile
        sm: base * 0.875, // 87.5% on small
        md: base,         // 100% on medium
        lg: base * 1.1,   // 110% on large
    },
});

export const responsiveMargin = (base: number): SxProps<Theme> => ({
    m: {
        xs: base * 0.75,
        sm: base * 0.875,
        md: base,
        lg: base * 1.1,
    },
});

// Responsive container widths
export const responsiveContainer: SxProps<Theme> = {
    width: '100%',
    maxWidth: {
        xs: '100%',
        sm: '100%',
        md: '90%',
        lg: '1200px',
        xl: '1400px',
    },
    mx: 'auto',
    px: {
        xs: 2,
        sm: 3,
        md: 4,
        lg: 5,
    },
};

// Responsive grid spacing
export const responsiveGridSpacing = {
    xs: 1,
    sm: 1.5,
    md: 2,
    lg: 2.5,
    xl: 3,
};

// Responsive card styles
export const responsiveCard: SxProps<Theme> = {
    p: {
        xs: 1.5,
        sm: 2,
        md: 2.5,
        lg: 3,
    },
    borderRadius: {
        xs: 1,
        sm: 1.5,
        md: 2,
    },
};

// Responsive button sizes
export const responsiveButton = {
    small: {
        py: { xs: 0.5, sm: 0.75, md: 1 },
        px: { xs: 1, sm: 1.5, md: 2 },
        fontSize: { xs: '0.75rem', sm: '0.8125rem', md: '0.875rem' },
    },
    medium: {
        py: { xs: 0.75, sm: 1, md: 1.25 },
        px: { xs: 1.5, sm: 2, md: 2.5 },
        fontSize: { xs: '0.8125rem', sm: '0.875rem', md: '0.9375rem' },
    },
    large: {
        py: { xs: 1, sm: 1.25, md: 1.5 },
        px: { xs: 2, sm: 2.5, md: 3 },
        fontSize: { xs: '0.875rem', sm: '0.9375rem', md: '1rem' },
    },
};

// Responsive icon sizes
export const responsiveIcon = {
    small: {
        fontSize: { xs: '1rem', sm: '1.125rem', md: '1.25rem' },
    },
    medium: {
        fontSize: { xs: '1.25rem', sm: '1.375rem', md: '1.5rem' },
    },
    large: {
        fontSize: { xs: '1.5rem', sm: '1.75rem', md: '2rem' },
    },
};

// Responsive dialog/modal sizes
export const responsiveDialog = {
    small: {
        width: { xs: '90vw', sm: '70vw', md: '50vw', lg: '40vw' },
        maxWidth: { xs: '100%', sm: 400, md: 500 },
    },
    medium: {
        width: { xs: '95vw', sm: '80vw', md: '60vw', lg: '50vw' },
        maxWidth: { xs: '100%', sm: 600, md: 800 },
    },
    large: {
        width: { xs: '100vw', sm: '90vw', md: '80vw', lg: '70vw' },
        maxWidth: { xs: '100%', sm: 900, md: 1200 },
    },
};

// Responsive chart heights
export const responsiveChartHeight = {
    small: { xs: 200, sm: 250, md: 300, lg: 350 },
    medium: { xs: 300, sm: 350, md: 400, lg: 450 },
    large: { xs: 400, sm: 450, md: 500, lg: 600 },
};

// Hide on specific breakpoints
export const hideOn = {
    xs: { display: { xs: 'none', sm: 'block' } },
    sm: { display: { xs: 'block', sm: 'none', md: 'block' } },
    md: { display: { xs: 'block', md: 'none', lg: 'block' } },
    lg: { display: { xs: 'block', lg: 'none' } },
};

// Show only on specific breakpoints
export const showOnlyOn = {
    xs: { display: { xs: 'block', sm: 'none' } },
    sm: { display: { xs: 'none', sm: 'block', md: 'none' } },
    md: { display: { xs: 'none', md: 'block', lg: 'none' } },
    lg: { display: { xs: 'none', lg: 'block', xl: 'none' } },
    xl: { display: { xs: 'none', xl: 'block' } },
};

// Responsive flex direction
export const responsiveFlexDirection = {
    columnToRow: {
        flexDirection: { xs: 'column', sm: 'row' },
    },
    rowToColumn: {
        flexDirection: { xs: 'row', sm: 'column' },
    },
};

// Responsive stack spacing
export const responsiveStackSpacing = {
    small: { xs: 1, sm: 1.5, md: 2 },
    medium: { xs: 1.5, sm: 2, md: 2.5 },
    large: { xs: 2, sm: 2.5, md: 3 },
};

// Helper to create responsive sx props
export const createResponsiveSx = (
    mobile: SxProps<Theme>,
    tablet?: SxProps<Theme>,
    desktop?: SxProps<Theme>,
    large?: SxProps<Theme>
): SxProps<Theme> => ({
    ...mobile,
    '@media (min-width:600px)': tablet || mobile,
    '@media (min-width:900px)': desktop || tablet || mobile,
    '@media (min-width:1200px)': large || desktop || tablet || mobile,
});

// Responsive table cell padding
export const responsiveTableCell = {
    py: { xs: 0.5, sm: 1, md: 1.5 },
    px: { xs: 1, sm: 1.5, md: 2 },
    fontSize: { xs: '0.75rem', sm: '0.8125rem', md: '0.875rem' },
};

// Responsive app bar height
export const responsiveAppBarHeight = {
    xs: 56,
    sm: 60,
    md: 64,
    lg: 68,
};

// Export all utilities as a single object for convenience
export const responsive = {
    padding: responsivePadding,
    margin: responsiveMargin,
    container: responsiveContainer,
    gridSpacing: responsiveGridSpacing,
    card: responsiveCard,
    button: responsiveButton,
    icon: responsiveIcon,
    dialog: responsiveDialog,
    chartHeight: responsiveChartHeight,
    hideOn,
    showOnlyOn,
    flexDirection: responsiveFlexDirection,
    stackSpacing: responsiveStackSpacing,
    tableCell: responsiveTableCell,
    appBarHeight: responsiveAppBarHeight,
    createSx: createResponsiveSx,
}; 