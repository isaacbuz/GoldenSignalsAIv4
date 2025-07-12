import React from 'react';
import type { Preview } from '@storybook/react';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { goldenTheme } from '../src/theme/goldenTheme';
import '../src/index.css';

// Create a client for each story
const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            retry: false,
            staleTime: Infinity,
        },
    },
});

const withProviders = (Story: any) => (
    <QueryClientProvider client={queryClient}>
        <BrowserRouter>
            <ThemeProvider theme={goldenTheme}>
                <CssBaseline />
                <Story />
            </ThemeProvider>
        </BrowserRouter>
    </QueryClientProvider>
);

const preview: Preview = {
    parameters: {
        actions: { argTypesRegex: '^on[A-Z].*' },
        controls: {
            matchers: {
                color: /(background|color)$/i,
                date: /Date$/i,
            },
        },
        docs: {
            toc: true,
        },
        backgrounds: {
            default: 'dark',
            values: [
                {
                    name: 'dark',
                    value: '#121212',
                },
                {
                    name: 'light',
                    value: '#ffffff',
                },
            ],
        },
    },
    decorators: [withProviders],
    globalTypes: {
        theme: {
            name: 'Theme',
            description: 'Global theme for components',
            defaultValue: 'dark',
            toolbar: {
                icon: 'paintbrush',
                items: ['light', 'dark'],
                showName: true,
            },
        },
    },
};

export default preview; 