import React from 'react';
import { IconButton, Tooltip } from '@mui/material';
import { Brightness4, Brightness7 } from '@mui/icons-material';

interface ThemeSwitcherProps {
    isDarkPro?: boolean;
    onToggle?: () => void;
}

export const ThemeSwitcher: React.FC<ThemeSwitcherProps> = ({
    isDarkPro = true,
    onToggle
}) => {
    return (
        <Tooltip title={isDarkPro ? "Switch to Light Theme" : "Switch to Dark Pro Theme"}>
            <IconButton
                onClick={onToggle}
                sx={{
                    color: 'text.secondary',
                    '&:hover': {
                        color: 'primary.main',
                        backgroundColor: 'rgba(255, 255, 255, 0.05)',
                    },
                }}
            >
                {isDarkPro ? <Brightness7 /> : <Brightness4 />}
            </IconButton>
        </Tooltip>
    );
}; 