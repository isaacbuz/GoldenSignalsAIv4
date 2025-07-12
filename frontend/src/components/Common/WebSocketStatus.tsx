import React from 'react';

interface WebSocketStatusProps {
    isConnected?: boolean;
    className?: string;
}

/**
 * WebSocketStatus - Shows WebSocket connection status
 * 
 * This is a placeholder component to fix import errors.
 * In a real implementation, this would show the actual WebSocket connection status.
 */
export const WebSocketStatus: React.FC<WebSocketStatusProps> = ({
    isConnected = false,
    className = ''
}) => {
    return (
        <div className={`websocket-status ${className}`}>
            <div className={`flex items-center gap-2 text-sm ${isConnected ? 'text-green-600' : 'text-gray-400'}`}>
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-600' : 'bg-gray-400'}`} />
                <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
        </div>
    );
};

export default WebSocketStatus; 