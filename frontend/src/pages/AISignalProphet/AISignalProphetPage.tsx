import React from 'react';
import { Box, Container } from '@mui/material';
import { AISignalProphet } from '../../components/AISignalProphet';
import { useAlerts } from '../../contexts/AlertContext';

const AISignalProphetPage: React.FC = () => {
    const { showAlert } = useAlert();

    const handleSignalGenerated = (signal: any) => {
        showAlert(
            `New ${signal.type} signal generated for ${signal.symbol} with ${signal.confidence}% confidence!`,
            'success'
        );
    };

    return (
        <Container maxWidth={false} sx={{ py: 3 }}>
            <AISignalProphet onSignalGenerated={handleSignalGenerated} />
        </Container>
    );
};

export default AISignalProphetPage; 