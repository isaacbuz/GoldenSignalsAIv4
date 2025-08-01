import { createTheme } from '@mui/material/styles';

// Sublime Text Inspired Theme - Softer, more balanced dark theme
export const sublimeTheme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#66D9EF', // Sublime's signature cyan
            light: '#A6E3F0',
            dark: '#4FA5C7',
            contrastText: '#272822',
        },
        secondary: {
            main: '#A6E22E', // Sublime's green
            light: '#C3E88D',
            dark: '#7CB342',
            contrastText: '#272822',
        },
        background: {
            default: '#272822', // Sublime's background - softer than pure black
            paper: '#3E3D32', // Slightly lighter for cards
        },
        text: {
            primary: '#F8F8F2', // Sublime's text color - warm white
            secondary: '#75715E', // Sublime's comment color
        },
        success: {
            main: '#A6E22E', // Green
            light: '#C3E88D',
            dark: '#7CB342',
        },
        error: {
            main: '#F92672', // Sublime's pink/red
            light: '#FF6B9D',
            dark: '#C74E39',
        },
        warning: {
            main: '#FD971F', // Sublime's orange
            light: '#FFB74D',
            dark: '#F57C00',
        },
        info: {
            main: '#66D9EF', // Cyan
            light: '#81E6F7',
            dark: '#4FC3F7',
        },
        divider: 'rgba(248, 248, 242, 0.12)',
        grey: {
            50: '#FAFAFA',
            100: '#F5F5F5',
            200: '#EEEEEE',
            300: '#E0E0E0',
            400: '#BDBDBD',
            500: '#9E9E9E',
            600: '#757575',
            700: '#616161',
            800: '#424242',
            900: '#212121',
        },
    },
    typography: {
        fontFamily: [
            'Consolas',
            'Monaco',
            'Courier New',
            'monospace',
            '-apple-system',
            'BlinkMacSystemFont',
            'Segoe UI',
            'Roboto',
            'sans-serif',
        ].join(','),
        h1: {
            fontSize: '3rem',
            fontWeight: 600,
            letterSpacing: '-0.01em',
            lineHeight: 1.2,
        },
        h2: {
            fontSize: '2.5rem',
            fontWeight: 600,
            letterSpacing: '-0.008em',
            lineHeight: 1.3,
        },
        h3: {
            fontSize: '2rem',
            fontWeight: 500,
            letterSpacing: '-0.006em',
            lineHeight: 1.4,
        },
        h4: {
            fontSize: '1.5rem',
            fontWeight: 500,
            letterSpacing: '-0.004em',
            lineHeight: 1.5,
        },
        h5: {
            fontSize: '1.25rem',
            fontWeight: 500,
            lineHeight: 1.6,
        },
        h6: {
            fontSize: '1.1rem',
            fontWeight: 500,
            lineHeight: 1.7,
        },
        subtitle1: {
            fontSize: '1rem',
            fontWeight: 400,
            letterSpacing: '0.008em',
            color: '#75715E',
        },
        subtitle2: {
            fontSize: '0.875rem',
            fontWeight: 400,
            letterSpacing: '0.01em',
            color: '#75715E',
        },
        body1: {
            fontSize: '1rem',
            fontWeight: 400,
            letterSpacing: '0.01em',
            lineHeight: 1.8,
        },
        body2: {
            fontSize: '0.875rem',
            fontWeight: 400,
            letterSpacing: '0.01em',
            lineHeight: 1.7,
        },
        caption: {
            fontSize: '0.75rem',
            fontWeight: 400,
            letterSpacing: '0.03em',
            color: '#75715E',
        },
        overline: {
            fontSize: '0.75rem',
            fontWeight: 500,
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            color: '#75715E',
        },
        button: {
            textTransform: 'none',
            fontWeight: 500,
            letterSpacing: '0.02em',
        },
    },
    shape: {
        borderRadius: 8,
    },
    shadows: [
        'none',
        '0px 2px 4px rgba(0, 0, 0, 0.2)',
        '0px 4px 8px rgba(0, 0, 0, 0.2)',
        '0px 8px 16px rgba(0, 0, 0, 0.2)',
        '0px 12px 24px rgba(0, 0, 0, 0.2)',
        '0px 16px 32px rgba(0, 0, 0, 0.2)',
        '0px 20px 40px rgba(0, 0, 0, 0.2)',
        '0px 24px 48px rgba(0, 0, 0, 0.2)',
        '0px 28px 56px rgba(0, 0, 0, 0.2)',
        '0px 32px 64px rgba(0, 0, 0, 0.2)',
        '0px 36px 72px rgba(0, 0, 0, 0.2)',
        '0px 40px 80px rgba(0, 0, 0, 0.2)',
        '0px 44px 88px rgba(0, 0, 0, 0.2)',
        '0px 48px 96px rgba(0, 0, 0, 0.2)',
        '0px 52px 104px rgba(0, 0, 0, 0.2)',
        '0px 56px 112px rgba(0, 0, 0, 0.2)',
        '0px 60px 120px rgba(0, 0, 0, 0.2)',
        '0px 64px 128px rgba(0, 0, 0, 0.2)',
        '0px 68px 136px rgba(0, 0, 0, 0.2)',
        '0px 72px 144px rgba(0, 0, 0, 0.2)',
        '0px 76px 152px rgba(0, 0, 0, 0.2)',
        '0px 80px 160px rgba(0, 0, 0, 0.2)',
        '0px 84px 168px rgba(0, 0, 0, 0.2)',
        '0px 88px 176px rgba(0, 0, 0, 0.2)',
        '0px 92px 184px rgba(0, 0, 0, 0.2)',
    ],
    components: {
        MuiCssBaseline: {
            styleOverrides: {
                body: {
                    scrollbarColor: '#75715E #3E3D32',
                    '&::-webkit-scrollbar, & *::-webkit-scrollbar': {
                        width: 10,
                        height: 10,
                    },
                    '&::-webkit-scrollbar-thumb, & *::-webkit-scrollbar-thumb': {
                        borderRadius: 5,
                        backgroundColor: '#75715E',
                        '&:hover': {
                            backgroundColor: '#8F8A7A',
                        },
                    },
                    '&::-webkit-scrollbar-track, & *::-webkit-scrollbar-track': {
                        backgroundColor: '#3E3D32',
                    },
                },
            },
        },
        MuiPaper: {
            defaultProps: {
                elevation: 0,
            },
            styleOverrides: {
                root: {
                    backgroundImage: 'none',
                    backgroundColor: '#3E3D32',
                    border: '1px solid rgba(248, 248, 242, 0.08)',
                },
                elevation1: {
                    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.15)',
                },
                elevation2: {
                    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.15)',
                },
                elevation3: {
                    boxShadow: '0 8px 16px rgba(0, 0, 0, 0.15)',
                },
            },
        },
        MuiCard: {
            defaultProps: {
                elevation: 0,
            },
            styleOverrides: {
                root: {
                    backgroundColor: '#3E3D32',
                    border: '1px solid rgba(248, 248, 242, 0.08)',
                    borderRadius: 8,
                    transition: 'all 0.2s ease',
                    '&:hover': {
                        transform: 'translateY(-1px)',
                        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
                        borderColor: 'rgba(248, 248, 242, 0.12)',
                    },
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 6,
                    padding: '8px 16px',
                    fontSize: '0.875rem',
                    fontWeight: 500,
                    transition: 'all 0.2s ease',
                    textTransform: 'none',
                },
                containedPrimary: {
                    backgroundColor: '#66D9EF',
                    color: '#272822',
                    boxShadow: 'none',
                    '&:hover': {
                        backgroundColor: '#A6E3F0',
                        boxShadow: '0 2px 8px rgba(102, 217, 239, 0.3)',
                    },
                    '&:active': {
                        backgroundColor: '#4FA5C7',
                    },
                },
                containedSecondary: {
                    backgroundColor: '#A6E22E',
                    color: '#272822',
                    boxShadow: 'none',
                    '&:hover': {
                        backgroundColor: '#C3E88D',
                        boxShadow: '0 2px 8px rgba(166, 226, 46, 0.3)',
                    },
                },
                outlined: {
                    borderColor: 'rgba(248, 248, 242, 0.3)',
                    color: '#F8F8F2',
                    '&:hover': {
                        borderColor: 'rgba(248, 248, 242, 0.5)',
                        backgroundColor: 'rgba(248, 248, 242, 0.05)',
                    },
                },
                text: {
                    color: '#F8F8F2',
                    '&:hover': {
                        backgroundColor: 'rgba(248, 248, 242, 0.05)',
                    },
                },
            },
        },
        MuiTextField: {
            defaultProps: {
                variant: 'outlined',
            },
            styleOverrides: {
                root: {
                    '& .MuiOutlinedInput-root': {
                        backgroundColor: 'rgba(0, 0, 0, 0.2)',
                        borderRadius: 6,
                        transition: 'all 0.2s ease',
                        '& fieldset': {
                            borderColor: 'rgba(248, 248, 242, 0.2)',
                            transition: 'all 0.2s ease',
                        },
                        '&:hover fieldset': {
                            borderColor: 'rgba(248, 248, 242, 0.3)',
                        },
                        '&.Mui-focused fieldset': {
                            borderColor: '#66D9EF',
                            borderWidth: '1px',
                        },
                        '&.Mui-focused': {
                            backgroundColor: 'rgba(102, 217, 239, 0.05)',
                        },
                    },
                    '& .MuiInputLabel-root': {
                        color: '#75715E',
                        '&.Mui-focused': {
                            color: '#66D9EF',
                        },
                    },
                },
            },
        },
        MuiChip: {
            styleOverrides: {
                root: {
                    fontWeight: 400,
                    borderRadius: 4,
                    backgroundColor: 'rgba(248, 248, 242, 0.1)',
                    '&:hover': {
                        backgroundColor: 'rgba(248, 248, 242, 0.15)',
                    },
                },
                colorPrimary: {
                    backgroundColor: 'rgba(102, 217, 239, 0.2)',
                    color: '#66D9EF',
                    '&:hover': {
                        backgroundColor: 'rgba(102, 217, 239, 0.3)',
                    },
                },
                colorSecondary: {
                    backgroundColor: 'rgba(166, 226, 46, 0.2)',
                    color: '#A6E22E',
                    '&:hover': {
                        backgroundColor: 'rgba(166, 226, 46, 0.3)',
                    },
                },
            },
        },
        MuiTooltip: {
            styleOverrides: {
                tooltip: {
                    backgroundColor: '#49483E',
                    color: '#F8F8F2',
                    fontSize: '0.75rem',
                    fontWeight: 400,
                    padding: '6px 10px',
                    borderRadius: 4,
                    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)',
                },
                arrow: {
                    color: '#49483E',
                },
            },
        },
        MuiDivider: {
            styleOverrides: {
                root: {
                    borderColor: 'rgba(248, 248, 242, 0.08)',
                },
            },
        },
        MuiSwitch: {
            styleOverrides: {
                root: {
                    width: 42,
                    height: 26,
                    padding: 0,
                    '& .MuiSwitch-switchBase': {
                        padding: 0,
                        margin: 2,
                        transitionDuration: '300ms',
                        '&.Mui-checked': {
                            transform: 'translateX(16px)',
                            color: '#fff',
                            '& + .MuiSwitch-track': {
                                backgroundColor: '#66D9EF',
                                opacity: 1,
                                border: 0,
                            },
                        },
                    },
                    '& .MuiSwitch-thumb': {
                        boxSizing: 'border-box',
                        width: 22,
                        height: 22,
                    },
                    '& .MuiSwitch-track': {
                        borderRadius: 26 / 2,
                        backgroundColor: '#75715E',
                        opacity: 1,
                        transition: 'background-color 300ms',
                    },
                },
            },
        },
        MuiAlert: {
            styleOverrides: {
                root: {
                    borderRadius: 6,
                    fontWeight: 400,
                },
                standardSuccess: {
                    backgroundColor: 'rgba(166, 226, 46, 0.15)',
                    color: '#A6E22E',
                    '& .MuiAlert-icon': {
                        color: '#A6E22E',
                    },
                },
                standardError: {
                    backgroundColor: 'rgba(249, 38, 114, 0.15)',
                    color: '#F92672',
                    '& .MuiAlert-icon': {
                        color: '#F92672',
                    },
                },
                standardWarning: {
                    backgroundColor: 'rgba(253, 151, 31, 0.15)',
                    color: '#FD971F',
                    '& .MuiAlert-icon': {
                        color: '#FD971F',
                    },
                },
                standardInfo: {
                    backgroundColor: 'rgba(102, 217, 239, 0.15)',
                    color: '#66D9EF',
                    '& .MuiAlert-icon': {
                        color: '#66D9EF',
                    },
                },
            },
        },
        MuiTableCell: {
            styleOverrides: {
                root: {
                    borderBottom: '1px solid rgba(248, 248, 242, 0.08)',
                },
                head: {
                    backgroundColor: '#272822',
                    fontWeight: 500,
                    color: '#75715E',
                    fontSize: '0.75rem',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                },
            },
        },
        MuiTableRow: {
            styleOverrides: {
                root: {
                    '&:hover': {
                        backgroundColor: 'rgba(248, 248, 242, 0.03)',
                    },
                },
            },
        },
    },
});

// Sublime Text color palette for charts and data visualization
export const sublimeColors = {
    // Syntax highlighting colors
    keyword: '#F92672',      // Pink
    string: '#E6DB74',       // Yellow
    function: '#66D9EF',     // Cyan
    variable: '#FD971F',     // Orange
    comment: '#75715E',      // Gray
    constant: '#AE81FF',     // Purple

    // Chart colors
    primary: ['#66D9EF', '#A6E3F0', '#4FA5C7'],
    success: ['#A6E22E', '#C3E88D', '#7CB342'],
    error: ['#F92672', '#FF6B9D', '#C74E39'],
    warning: ['#FD971F', '#FFB74D', '#F57C00'],
    info: ['#66D9EF', '#81E6F7', '#4FC3F7'],
    purple: ['#AE81FF', '#C7A4FF', '#9966FF'],

    // Gradients
    gradient: {
        cyan: 'linear-gradient(135deg, #66D9EF 0%, #4FA5C7 100%)',
        green: 'linear-gradient(135deg, #A6E22E 0%, #7CB342 100%)',
        pink: 'linear-gradient(135deg, #F92672 0%, #C74E39 100%)',
        orange: 'linear-gradient(135deg, #FD971F 0%, #F57C00 100%)',
        purple: 'linear-gradient(135deg, #AE81FF 0%, #9966FF 100%)',
    },
};

// Utility styles for Sublime theme
export const sublimeStyles = {
    codeBlock: {
        backgroundColor: '#272822',
        color: '#F8F8F2',
        padding: '16px',
        borderRadius: '6px',
        fontFamily: 'Consolas, Monaco, "Courier New", monospace',
        fontSize: '14px',
        lineHeight: '1.6',
        overflow: 'auto',
    },
    syntaxHighlight: {
        keyword: { color: '#F92672' },
        string: { color: '#E6DB74' },
        function: { color: '#66D9EF' },
        variable: { color: '#FD971F' },
        comment: { color: '#75715E', fontStyle: 'italic' },
        constant: { color: '#AE81FF' },
    },
    card: {
        background: '#3E3D32',
        border: '1px solid rgba(248, 248, 242, 0.08)',
        borderRadius: '8px',
    },
    hover: {
        background: 'rgba(248, 248, 242, 0.05)',
        cursor: 'pointer',
    },
};
