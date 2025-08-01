import React, { useState } from 'react';
import { useConsoleMonitor } from '../../services/monitoring/ConsoleMonitor';
import { ENV } from '../../config/environment';

export const ConsoleViewer: React.FC = () => {
    const { logs, stats, clearLogs, getLogs } = useConsoleMonitor();
    const [filter, setFilter] = useState<string>('');

    // Only show in development
    if (!ENV.IS_DEVELOPMENT) {
        return null;
    }

    const filteredLogs = filter
        ? logs.filter(log => log.message.toLowerCase().includes(filter.toLowerCase()))
        : logs;

    return (
        <div className="console-viewer">
            <div className="console-header">
                <h3>Console Monitor</h3>
                <div className="console-stats">
                    <span>Total: {stats.total}</span>
                    <span>Errors: {stats.byLevel.error || 0}</span>
                    <span>Warnings: {stats.byLevel.warn || 0}</span>
                    <span>Recent Errors: {stats.recentErrors}</span>
                </div>
                <div className="console-controls">
                    <input
                        type="text"
                        placeholder="Filter logs..."
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                    />
                    <button onClick={clearLogs}>Clear</button>
                </div>
            </div>
            <div className="console-logs">
                {filteredLogs.map((log, index) => (
                    <div key={index} className={`log-entry log-${log.level}`}>
                        <span className="log-timestamp">
                            {new Date(log.timestamp).toLocaleTimeString()}
                        </span>
                        <span className="log-level">{log.level.toUpperCase()}</span>
                        <span className="log-message">{log.message}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};
