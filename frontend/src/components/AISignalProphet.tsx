import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Psychology } from '@mui/icons-material';

export interface AISignalProphetProps {
    onSignalGenerated?: (signal: any) => void;
}

const AISignalProphet: React.FC<AISignalProphetProps> = ({ onSignalGenerated }) => {
    return (
        <Paper sx={{ p: 3, textAlign: 'center' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1, mb: 2 }}>
                <Psychology color="primary" />
                <Typography variant="h5">AI Signal Prophet</Typography>
            </Box>
            <Typography variant="body1" color="text.secondary">
                AI Signal Prophet component is currently being upgraded. Please check back soon.
            </Typography>
        </Paper>
    );
};

export default AISignalProphet;
export { AISignalProphet };
