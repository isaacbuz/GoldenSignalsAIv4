import React from 'react';
import { List, ListItem, ListItemText } from '@mui/material';
import { useSignals } from '../../store';
import { Signal } from '../../types/signals';

const SignalStream: React.FC = () => {
    const { signals } = useSignals();

    return (
        <List>
            {signals.slice(0, 5).map((signal: Signal, index: number) => (
                <ListItem key={index}>
                    <ListItemText
                        primary={signal.symbol}
                        secondary={`Type: ${signal.signal_type} | Confidence: ${signal.confidence_score}%`}
                    />
                </ListItem>
            ))}
        </List>
    );
};

export default SignalStream; 