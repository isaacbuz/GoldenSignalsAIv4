import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Chip,
  Button,
  LinearProgress,
  IconButton,
  Alert,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Memory,
  Storage,
  Speed,
  Warning,
  CheckCircle,
  Error,
  Refresh,
  Settings,
  CloudQueue,
  Security,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { utilityClasses } from '../../theme/goldenTheme';

const SystemCard = styled(Card)(({ theme }) => ({
  ...utilityClasses.glassmorphism,
  height: '100%',
}));

const StatusChip = styled(Chip)<{ status: 'healthy' | 'warning' | 'error' }>(({ status, theme }) => ({
  backgroundColor:
    status === 'healthy' ? 'rgba(76, 175, 80, 0.1)' :
    status === 'warning' ? 'rgba(255, 165, 0, 0.1)' :
    'rgba(244, 67, 54, 0.1)',
  color:
    status === 'healthy' ? '#4CAF50' :
    status === 'warning' ? '#FFA500' :
    '#F44336',
  border: `1px solid ${
    status === 'healthy' ? 'rgba(76, 175, 80, 0.3)' :
    status === 'warning' ? 'rgba(255, 165, 0, 0.3)' :
    'rgba(244, 67, 54, 0.3)'
  }`,
}));

interface SystemMetric {
  name: string;
  value: number;
  max: number;
  unit: string;
  status: 'healthy' | 'warning' | 'error';
}

interface ServiceStatus {
  name: string;
  status: 'running' | 'stopped' | 'error';
  uptime: string;
  cpu: number;
  memory: number;
  requests: number;
}

const AdminMonitoring: React.FC = () => {
  const [systemMetrics, setSystemMetrics] = useState<SystemMetric[]>([]);
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [alerts, setAlerts] = useState([]);
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    loadSystemData();
    
    if (autoRefresh) {
      const interval = setInterval(loadSystemData, 5000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const loadSystemData = async () => {
    // Mock system metrics
    setSystemMetrics([
      { name: 'CPU Usage', value: 45, max: 100, unit: '%', status: 'healthy' },
      { name: 'Memory Usage', value: 62, max: 100, unit: '%', status: 'warning' },
      { name: 'Disk Usage', value: 71, max: 100, unit: '%', status: 'warning' },
      { name: 'Network I/O', value: 23, max: 100, unit: 'Mbps', status: 'healthy' },
    ]);

    // Mock service status
    setServices([
      { name: 'Signal Engine', status: 'running', uptime: '7d 14h', cpu: 12, memory: 256, requests: 15420 },
      { name: 'WebSocket Server', status: 'running', uptime: '7d 14h', cpu: 8, memory: 128, requests: 45231 },
      { name: 'RAG System', status: 'running', uptime: '3d 22h', cpu: 25, memory: 512, requests: 8745 },
      { name: 'Redis Cache', status: 'running', uptime: '14d 3h', cpu: 5, memory: 1024, requests: 982341 },
      { name: 'Agent Coordinator', status: 'error', uptime: '0h 0m', cpu: 0, memory: 0, requests: 0 },
    ]);

    // Mock alerts
    setAlerts([
      { id: 1, severity: 'warning', message: 'High memory usage detected', time: '5 min ago' },
      { id: 2, severity: 'error', message: 'Agent Coordinator service down', time: '2 min ago' },
    ]);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
      case 'healthy':
        return <CheckCircle sx={{ color: '#4CAF50' }} />;
      case 'stopped':
      case 'warning':
        return <Warning sx={{ color: '#FFA500' }} />;
      case 'error':
        return <Error sx={{ color: '#F44336' }} />;
      default:
        return null;
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700, ...utilityClasses.textGradient }}>
            System Monitoring
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Real-time system health and performance monitoring
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                color="primary"
              />
            }
            label="Auto Refresh"
          />
          <IconButton onClick={loadSystemData}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      {/* Alerts */}
      {alerts.length > 0 && (
        <Box sx={{ mb: 3 }}>
          {alerts.map(alert => (
            <Alert
              key={alert.id}
              severity={alert.severity}
              sx={{ mb: 1 }}
              onClose={() => setAlerts(prev => prev.filter(a => a.id !== alert.id))}
            >
              <Typography variant="body2">
                {alert.message} - {alert.time}
              </Typography>
            </Alert>
          ))}
        </Box>
      )}

      <Grid container spacing={3}>
        {/* System Metrics */}
        <Grid item xs={12}>
          <Typography variant="h6" sx={{ mb: 2 }}>System Resources</Typography>
          <Grid container spacing={2}>
            {systemMetrics.map((metric, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <SystemCard>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        {metric.name}
                      </Typography>
                      <StatusChip
                        label={metric.status}
                        status={metric.status}
                        size="small"
                      />
                    </Box>
                    <Typography variant="h4" sx={{ mb: 1, color: '#FFD700' }}>
                      {metric.value}{metric.unit}
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={(metric.value / metric.max) * 100}
                      sx={{
                        height: 6,
                        borderRadius: 3,
                        backgroundColor: 'rgba(255, 255, 255, 0.1)',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor:
                            metric.status === 'healthy' ? '#4CAF50' :
                            metric.status === 'warning' ? '#FFA500' : '#F44336',
                        },
                      }}
                    />
                  </CardContent>
                </SystemCard>
              </Grid>
            ))}
          </Grid>
        </Grid>

        {/* Service Status */}
        <Grid item xs={12}>
          <SystemCard>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3 }}>Service Status</Typography>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Service</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Uptime</TableCell>
                    <TableCell>CPU %</TableCell>
                    <TableCell>Memory (MB)</TableCell>
                    <TableCell>Requests</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {services.map((service) => (
                    <TableRow key={service.name}>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {getStatusIcon(service.status)}
                          {service.name}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <StatusChip
                          label={service.status}
                          status={service.status === 'running' ? 'healthy' : 'error'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{service.uptime}</TableCell>
                      <TableCell>{service.cpu}%</TableCell>
                      <TableCell>{service.memory}</TableCell>
                      <TableCell>{service.requests.toLocaleString()}</TableCell>
                      <TableCell>
                        <Button
                          size="small"
                          variant="outlined"
                          color={service.status === 'running' ? 'error' : 'success'}
                        >
                          {service.status === 'running' ? 'Stop' : 'Start'}
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </SystemCard>
        </Grid>

        {/* Cluster Status */}
        <Grid item xs={12} md={6}>
          <SystemCard>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Cluster Status</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">Active Nodes</Typography>
                  <Chip label="3/3" size="small" sx={{ backgroundColor: 'rgba(76, 175, 80, 0.1)' }} />
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">Load Distribution</Typography>
                  <Chip label="Balanced" size="small" />
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">Failover Status</Typography>
                  <Chip label="Ready" size="small" />
                </Box>
              </Box>
            </CardContent>
          </SystemCard>
        </Grid>

        {/* Security Status */}
        <Grid item xs={12} md={6}>
          <SystemCard>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Security Status</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">SSL Certificates</Typography>
                  <Chip label="Valid" size="small" sx={{ backgroundColor: 'rgba(76, 175, 80, 0.1)' }} />
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">API Rate Limiting</Typography>
                  <Chip label="Active" size="small" />
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">Last Security Scan</Typography>
                  <Chip label="2 hours ago" size="small" />
                </Box>
              </Box>
            </CardContent>
          </SystemCard>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AdminMonitoring;
