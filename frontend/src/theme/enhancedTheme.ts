import { createTheme, alpha } from '@mui/material/styles';

// Central theme configuration for the enhanced trading dashboard
const enhancedTheme = createTheme({
  // Global typography settings - reduced font sizes
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    fontSize: 13, // Base font size (down from 14)

    h1: {
      fontSize: '1.75rem', // 28px
      fontWeight: 700,
    },
    h2: {
      fontSize: '1.5rem', // 24px
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.25rem', // 20px
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.125rem', // 18px
      fontWeight: 600,
    },
    h5: {
      fontSize: '1rem', // 16px
      fontWeight: 600,
    },
    h6: {
      fontSize: '0.875rem', // 14px
      fontWeight: 600,
    },
    subtitle1: {
      fontSize: '0.875rem', // 14px
      fontWeight: 500,
    },
    subtitle2: {
      fontSize: '0.8125rem', // 13px
      fontWeight: 500,
    },
    body1: {
      fontSize: '0.8125rem', // 13px
      fontWeight: 400,
    },
    body2: {
      fontSize: '0.75rem', // 12px
      fontWeight: 400,
    },
    button: {
      fontSize: '0.8125rem', // 13px
      fontWeight: 500,
      textTransform: 'none',
    },
    caption: {
      fontSize: '0.6875rem', // 11px
      fontWeight: 400,
    },
    overline: {
      fontSize: '0.625rem', // 10px
      fontWeight: 500,
      textTransform: 'uppercase',
    },
  },

  // Color palette
  palette: {
    mode: 'dark',
    primary: {
      main: '#FFD700', // Gold
      light: '#FFE55C',
      dark: '#C5A300',
    },
    secondary: {
      main: '#1E88E5',
      light: '#64B5F6',
      dark: '#1565C0',
    },
    success: {
      main: '#4CAF50',
      light: '#81C784',
      dark: '#388E3C',
    },
    error: {
      main: '#F44336',
      light: '#E57373',
      dark: '#D32F2F',
    },
    warning: {
      main: '#FF9800',
      light: '#FFB74D',
      dark: '#F57C00',
    },
    info: {
      main: '#2196F3',
      light: '#64B5F6',
      dark: '#1976D2',
    },
    background: {
      default: '#0A0E1A',
      paper: '#1A1F2E',
    },
    text: {
      primary: '#E0E0E0',
      secondary: '#B0B0B0',
    },
  },

  // Spacing - slightly reduced
  spacing: 8, // Base spacing unit (default is 8)

  // Component defaults
  components: {
    // Global component overrides
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          fontSize: '0.8125rem',
          scrollbarColor: '#FFD700 #1A1F2E',
          '&::-webkit-scrollbar': {
            width: '8px',
            height: '8px',
          },
          '&::-webkit-scrollbar-track': {
            backgroundColor: '#1A1F2E',
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: '#FFD700',
            borderRadius: '4px',
          },
        },
      },
    },

    // Paper components
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
      defaultProps: {
        elevation: 0,
      },
    },

    // Cards
    MuiCard: {
      styleOverrides: {
        root: ({ theme }) => ({
          borderRadius: theme.spacing(1),
          border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          backgroundColor: theme.palette.background.paper,
        }),
      },
    },

    MuiCardContent: {
      styleOverrides: {
        root: ({ theme }) => ({
          padding: theme.spacing(1.5),
          '&:last-child': {
            paddingBottom: theme.spacing(1.5),
          },
        }),
      },
    },

    // Buttons
    MuiButton: {
      styleOverrides: {
        root: ({ theme }) => ({
          borderRadius: theme.spacing(0.75),
          padding: theme.spacing(0.75, 2),
          fontSize: '0.8125rem',
        }),
        sizeSmall: ({ theme }) => ({
          padding: theme.spacing(0.5, 1.5),
          fontSize: '0.75rem',
        }),
        sizeLarge: ({ theme }) => ({
          padding: theme.spacing(1, 2.5),
          fontSize: '0.875rem',
        }),
      },
    },

    // Chips
    MuiChip: {
      styleOverrides: {
        root: ({ theme }) => ({
          borderRadius: theme.spacing(0.75),
          fontSize: '0.6875rem',
          height: 24,
        }),
        sizeSmall: {
          fontSize: '0.625rem',
          height: 20,
        },
      },
    },

    // Text fields
    MuiTextField: {
      defaultProps: {
        size: 'small',
      },
    },

    MuiInputBase: {
      styleOverrides: {
        root: {
          fontSize: '0.8125rem',
        },
        sizeSmall: {
          fontSize: '0.75rem',
        },
      },
    },

    // Typography
    MuiTypography: {
      styleOverrides: {
        gutterBottom: ({ theme }) => ({
          marginBottom: theme.spacing(0.75),
        }),
      },
    },

    // Dividers
    MuiDivider: {
      styleOverrides: {
        root: ({ theme }) => ({
          borderColor: alpha(theme.palette.divider, 0.1),
        }),
      },
    },

    // Icons
    MuiSvgIcon: {
      styleOverrides: {
        fontSizeSmall: {
          fontSize: '1rem',
        },
        fontSizeMedium: {
          fontSize: '1.25rem',
        },
        fontSizeLarge: {
          fontSize: '1.5rem',
        },
      },
    },

    // Tooltips
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          fontSize: '0.6875rem',
        },
      },
    },
  },

  // Custom theme extensions
  shape: {
    borderRadius: 8,
  },

  shadows: [
    'none',
    '0px 2px 4px rgba(0,0,0,0.1)',
    '0px 4px 8px rgba(0,0,0,0.1)',
    '0px 8px 16px rgba(0,0,0,0.1)',
    '0px 16px 32px rgba(0,0,0,0.1)',
    ...Array(20).fill('0px 16px 32px rgba(0,0,0,0.1)'),
  ] as any,
});

// Custom style utilities
export const customStyles = {
  // Glass effect for cards
  glassEffect: {
    background: alpha('#1A1F2E', 0.8),
    backdropFilter: 'blur(10px)',
    border: `1px solid ${alpha('#FFD700', 0.1)}`,
  },

  // Gold gradient
  goldGradient: {
    background: `linear-gradient(135deg, ${alpha('#FFD700', 0.1)} 0%, ${alpha('#FFD700', 0.05)} 100%)`,
  },

  // Success gradient
  successGradient: {
    background: `linear-gradient(135deg, ${alpha('#4CAF50', 0.1)} 0%, ${alpha('#4CAF50', 0.05)} 100%)`,
  },

  // Info gradient
  infoGradient: {
    background: `linear-gradient(135deg, ${alpha('#2196F3', 0.1)} 0%, ${alpha('#2196F3', 0.05)} 100%)`,
  },

  // Common card styles
  card: {
    borderRadius: enhancedTheme.spacing(1),
    padding: enhancedTheme.spacing(1.5),
    backgroundColor: enhancedTheme.palette.background.paper,
    border: `1px solid ${alpha(enhancedTheme.palette.divider, 0.1)}`,
  },

  // Common header styles
  sectionHeader: {
    fontSize: '0.875rem',
    fontWeight: 600,
    marginBottom: enhancedTheme.spacing(1),
  },

  // Compact spacing
  compactSpacing: {
    padding: enhancedTheme.spacing(1),
    gap: enhancedTheme.spacing(1),
  },

  // Standard spacing
  standardSpacing: {
    padding: enhancedTheme.spacing(1.5),
    gap: enhancedTheme.spacing(1.5),
  },
};

export default enhancedTheme;
