import React from 'react';
import { Chip, Tooltip } from '@mui/material';
import { FiberManualRecord } from '@mui/icons-material';
import { useWebSocketConnection } from '../../services/websocket';

export const WebSocketStatus: React.FC = () => {
    const isConnected = useWebSocketConnection();

    return (
        <Tooltip title={isConnected ? 'Real-time data connected' : 'Real-time data disconnected'}>
            <Chip
                size="small"
                icon={<FiberManualRecord sx={{ fontSize: 12 }} />}
                label={isConnected ? 'Live' : 'Offline'}
                color={isConnected ? 'success' : 'default'}
                sx={{
                    '& .MuiChip-icon': {
                        color: isConnected ? 'success.main' : 'text.disabled',
                    },
                }}
            />
        </Tooltip>
    );
}; 