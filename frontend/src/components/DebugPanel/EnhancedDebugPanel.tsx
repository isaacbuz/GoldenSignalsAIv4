/**
 * EnhancedDebugPanel Component
 * 
 * Development tool for viewing console logs, errors, and performance metrics in real-time
 */

import React, { useState, useEffect } from 'react';
import {
    Box,
    Paper,
    Typography,
    Tabs,
    Tab,
    Button,
    Chip,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    TextField,
    Grid,
    Card,
    CardContent,
    IconButton,
    Collapse
} from '@mui/material';
import {
    BugReport,
    Speed,
    Memory,
    NetworkCheck,
    Refresh,
    ExpandMore,
    ExpandLess,
    Clear,
    Download
} from '@mui/icons-material';
import { useConsoleMonitor, LogEntry } from '../../services/monitoring/ConsoleMonitor';
import { ENV } from '../../config/environment';

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;
    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`debug-tabpanel-${index}`}
            aria-labelledby={`debug-tab-${index}`}
            {...other}
        >
            {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
        </div>
    );
}

export const EnhancedDebugPanel: React.FC = () => {
    const { logs, stats, clearLogs } = useConsoleMonitor();
    const [tabValue, setTabValue] = useState(0);
    const [expanded, setExpanded] = useState(false);
    const [filter, setFilter] = useState('');

    // Only show in development
    if (!ENV.IS_DEVELOPMENT) {
        return null;
    }

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
    };

    const filteredLogs = logs.filter(log =>
        log.message.toLowerCase().includes(filter.toLowerCase())
    );

    const downloadLogs = () => {
        const logData = JSON.stringify(logs, null, 2);
        const blob = new Blob([logData], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `debug-logs-${new Date().toISOString()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <Paper
            elevation={3}
            sx={{
                position: 'fixed',
                bottom: 20,
                right: 20,
                width: expanded ? 800 : 300,
                maxHeight: expanded ? 600 : 200,
                zIndex: 9999,
                transition: 'all 0.3s ease'
            }}
        >
            <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <BugReport />
                    Debug Panel
                </Typography>
                <Box>
                    <IconButton onClick={() => setExpanded(!expanded)}>
                        {expanded ? <ExpandLess /> : <ExpandMore />}
                    </IconButton>
                </Box>
            </Box>

            <Collapse in={expanded}>
                <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                    <Tabs value={tabValue} onChange={handleTabChange}>
                        <Tab label="Console" />
                        <Tab label="Performance" />
                        <Tab label="Network" />
                        <Tab label="System" />
                    </Tabs>
                </Box>

                <TabPanel value={tabValue} index={0}>
                    <Box sx={{ mb: 2, display: 'flex', gap: 1, alignItems: 'center' }}>
                        <TextField
                            size="small"
                            placeholder="Filter logs..."
                            value={filter}
                            onChange={(e) => setFilter(e.target.value)}
                            sx={{ flexGrow: 1 }}
                        />
                        <Button onClick={clearLogs} startIcon={<Clear />}>
                            Clear
                        </Button>
                        <Button onClick={downloadLogs} startIcon={<Download />}>
                            Download
                        </Button>
                    </Box>

                    <Grid container spacing={2} sx={{ mb: 2 }}>
                        <Grid item xs={3}>
                            <Card>
                                <CardContent>
                                    <Typography variant="h6">{stats.total}</Typography>
                                    <Typography variant="body2">Total Logs</Typography>
                                </CardContent>
                            </Card>
                        </Grid>
                        <Grid item xs={3}>
                            <Card>
                                <CardContent>
                                    <Typography variant="h6" color="error">
                                        {stats.byLevel.error || 0}
                                    </Typography>
                                    <Typography variant="body2">Errors</Typography>
                                </CardContent>
                            </Card>
                        </Grid>
                        <Grid item xs={3}>
                            <Card>
                                <CardContent>
                                    <Typography variant="h6" color="warning.main">
                                        {stats.byLevel.warn || 0}
                                    </Typography>
                                    <Typography variant="body2">Warnings</Typography>
                                </CardContent>
                            </Card>
                        </Grid>
                        <Grid item xs={3}>
                            <Card>
                                <CardContent>
                                    <Typography variant="h6" color="error">
                                        {stats.recentErrors}
                                    </Typography>
                                    <Typography variant="body2">Recent Errors</Typography>
                                </CardContent>
                            </Card>
                        </Grid>
                    </Grid>

                    <TableContainer sx={{ maxHeight: 300 }}>
                        <Table size="small">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Time</TableCell>
                                    <TableCell>Level</TableCell>
                                    <TableCell>Message</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {filteredLogs.slice(-50).map((log: LogEntry, index: number) => (
                                    <TableRow key={index}>
                                        <TableCell>
                                            {new Date(log.timestamp).toLocaleTimeString()}
                                        </TableCell>
                                        <TableCell>
                                            <Chip
                                                size="small"
                                                label={log.level}
                                                color={
                                                    log.level === 'error' ? 'error' :
                                                        log.level === 'warn' ? 'warning' :
                                                            log.level === 'info' ? 'info' : 'default'
                                                }
                                            />
                                        </TableCell>
                                        <TableCell>{log.message}</TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </TabPanel>

                <TabPanel value={tabValue} index={1}>
                    <Typography variant="h6">Performance Metrics</Typography>
                    <Typography variant="body2">
                        Performance monitoring data will be displayed here.
                    </Typography>
                </TabPanel>

                <TabPanel value={tabValue} index={2}>
                    <Typography variant="h6">Network Activity</Typography>
                    <Typography variant="body2">
                        Network requests and responses will be displayed here.
                    </Typography>
                </TabPanel>

                <TabPanel value={tabValue} index={3}>
                    <Typography variant="h6">System Information</Typography>
                    <Typography variant="body2">
                        Environment: {ENV.IS_DEVELOPMENT ? 'Development' : 'Production'}
                    </Typography>
                </TabPanel>
            </Collapse>
        </Paper>
    );
}; 