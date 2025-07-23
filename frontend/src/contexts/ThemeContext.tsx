import React, { createContext, useContext, useState, ReactNode } from 'react';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { tradingTheme } from '../theme/tradingTheme';
import logger from '../services/logger';


interface ThemeContextType {
    isDarkMode: boolean;
    toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error('useTheme must be used within a ThemeProvider');
    }
    return context;
};

interface CustomThemeProviderProps {
    children: ReactNode;
}

export const CustomThemeProvider: React.FC<CustomThemeProviderProps> = ({ children }) => {
    // Always use dark theme for professional trading look
    const [isDarkMode] = useState(true);

    const toggleTheme = () => {
        // Theme toggle disabled - always use professional dark theme
        logger.info('Professional trading theme is always dark');
    };

    return (
        <ThemeContext.Provider value={{ isDarkMode, toggleTheme }}>
            <ThemeProvider theme={tradingTheme}>
                <CssBaseline />
                {children}
            </ThemeProvider>
        </ThemeContext.Provider>
    );
};
