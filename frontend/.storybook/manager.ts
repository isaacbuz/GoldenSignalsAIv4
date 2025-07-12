import { addons } from '@storybook/manager-api';
import { themes } from '@storybook/theming';

addons.setConfig({
    theme: {
        ...themes.dark,
        brandTitle: 'GoldenSignalsAI Components',
        brandUrl: 'https://goldensignalsai.com',
        brandImage: undefined, // Add your logo path here if you have one

        // UI colors
        appBg: '#1e1e1e',
        appContentBg: '#252525',
        appBorderColor: '#3e3e3e',
        appBorderRadius: 4,

        // Typography
        fontBase: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        fontCode: '"Fira Code", "Monaco", "Consolas", monospace',

        // Text colors
        textColor: '#ffffff',
        textInverseColor: '#1e1e1e',

        // Toolbar colors
        barTextColor: '#9e9e9e',
        barSelectedColor: '#1976d2',
        barBg: '#2d2d2d',

        // Form colors
        inputBg: '#3e3e3e',
        inputBorder: '#4e4e4e',
        inputTextColor: '#ffffff',
        inputBorderRadius: 4,
    },
    sidebar: {
        showRoots: true,
    },
}); 