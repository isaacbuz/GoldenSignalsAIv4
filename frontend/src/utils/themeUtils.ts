import { alpha } from '@mui/material/styles';
import logger from '../services/logger';


/**
 * Validates if a color string is in a valid format for MUI
 */
export const validateColor = (color: string): string => {
    // Check if it's a valid hex color
    if (/^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/.test(color)) {
        return color;
    }

    // Check if it's a valid rgb/rgba color
    if (/^rgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*(,\s*[\d.]+)?\s*\)$/.test(color)) {
        return color;
    }

    // Check if it's a valid hsl/hsla color
    if (/^hsla?\(\s*\d+\s*,\s*\d+%\s*,\s*\d+%\s*(,\s*[\d.]+)?\s*\)$/.test(color)) {
        return color;
    }

    // Fallback to a default color if invalid
    logger.warn(`Invalid color format: ${color}, falling back to default`);
    return '#000000';
};

/**
 * Safe alpha function that validates inputs and handles errors
 */
export const safeAlpha = (color: string, opacity: number): string => {
    try {
        // Validate inputs
        if (typeof color !== 'string' || typeof opacity !== 'number') {
            logger.warn('Invalid alpha function inputs:', { color, opacity });
            return 'rgba(0, 0, 0, 0.1)';
        }

        // Ensure opacity is between 0 and 1
        const validOpacity = Math.max(0, Math.min(1, opacity));

        // Use MUI's alpha function with validation
        const result = alpha(validateColor(color), validOpacity);

        // Validate result
        if (typeof result !== 'string') {
            logger.warn('Alpha function returned invalid result:', result);
            return 'rgba(0, 0, 0, 0.1)';
        }

        return result;
    } catch (error) {
        logger.warn('Error in alpha function:', error);
        return 'rgba(0, 0, 0, 0.1)';
    }
};

/**
 * Safe color getter from theme palette
 */
export const getThemeColor = (theme: any, colorPath: string): string => {
    try {
        const parts = colorPath.split('.');
        let value = theme.palette;

        for (const part of parts) {
            if (value && typeof value === 'object' && part in value) {
                value = value[part];
            } else {
                throw new Error(`Color path ${colorPath} not found in theme`);
            }
        }

        if (typeof value === 'string') {
            return validateColor(value);
        }

        throw new Error(`Color path ${colorPath} does not resolve to a string`);
    } catch (error) {
        logger.warn(`Error getting theme color ${colorPath}:`, error);
        return '#000000';
    }
};

/**
 * Create a safe alpha color from theme palette
 */
export const getAlphaColor = (theme: any, colorPath: string, opacity: number): string => {
    const color = getThemeColor(theme, colorPath);
    return safeAlpha(color, opacity);
};
