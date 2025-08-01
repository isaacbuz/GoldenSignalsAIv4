import React from 'react';
import { Box, Container } from '@mui/material';
import { AISignalProphet } from '../../components/AISignalProphet';
import { useAlerts } from '../../contexts/AlertContext';

const AISignalProphetPage: React.FC = () => {
    const { addAlert } = useAlerts();

    const handleSignalGenerated = (signal: any) => {
        addAlert({
            id: `signal-${Date.now()}`,
            type: signal.type || 'CALL',
            symbol: signal.symbol || 'UNKNOWN',
            confidence: signal.confidence || 0,
            priority: 'HIGH',
            timestamp: new Date(),
            message: `New ${signal.type} signal generated with ${signal.confidence}% confidence!`
        });
    };

    return (
        <Container maxWidth={false} sx={{ py: 3 }}>
            <AISignalProphet onSignalGenerated={handleSignalGenerated} />
        </Container>
    );
};

export default AISignalProphetPage;
