import React, { useState } from 'react';
import {
  Box,
  Tabs,
  Tab,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  TextField,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  InputAdornment,
  Tooltip,
} from '@mui/material';
import {
  Settings,
  Security,
  People,
  Storage,
  MonitorHeart,
  Key,
  Edit,
  Delete,
  Add,
  Save,
  Cancel,
  Visibility,
  VisibilityOff,
  Lock,
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  CloudUpload,
  Refresh,
  AdminPanelSettings,
  Speed,
  Memory,
  Timeline,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { utilityClasses } from '../../theme/goldenTheme';
import { useNavigate } from 'react-router-dom';

const StyledCard = styled(Card)(({ theme }) => ({
  ...utilityClasses.glassmorphism,
  height: '100%',
}));

const TabPanel = ({ children, value, index }: any) => (
  <Box hidden={value !== index} sx={{ pt: 3 }}>
    {value === index && children}
  </Box>
);

const StatusChip = styled(Chip)<{ status: 'active' | 'warning' | 'error' }>(({ status }) => ({
  ...(status === 'active' && {
    backgroundColor: 'rgba(76, 175, 80, 0.1)',
    color: '#4CAF50',
    border: '1px solid rgba(76, 175, 80, 0.3)',
  }),
  ...(status === 'warning' && {
    backgroundColor: 'rgba(255, 165, 0, 0.1)',
    color: '#FFA500',
    border: '1px solid rgba(255, 165, 0, 0.3)',
  }),
  ...(status === 'error' && {
    backgroundColor: 'rgba(244, 67, 54, 0.1)',
    color: '#F44336',
    border: '1px solid rgba(244, 67, 54, 0.3)',
  }),
}));

interface APIKey {
  id: string;
  name: string;
  key: string;
  service: string;
  lastUsed: string;
  status: 'active' | 'expired';
}

interface User {
  id: string;
  email: string;
  role: string;
  status: 'active' | 'suspended';
  lastLogin: string;
  apiUsage: number;
}

const AdminPanel: React.FC = () => {
  const navigate = useNavigate();
  const [tabValue, setTabValue] = useState(0);
  const [showApiKey, setShowApiKey] = useState<string | null>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [selectedItem, setSelectedItem] = useState<any>(null);
  
  // Mock data
  const [systemConfig, setSystemConfig] = useState({
    signalGenerationEnabled: true,
    maxConcurrentAgents: 9,
    consensusThreshold: 0.75,
    cacheEnabled: true,
    rateLimitPerMinute: 100,
    maintenanceMode: false,
  });

  const [apiKeys, setApiKeys] = useState<APIKey[]>([
    { id: '1', name: 'Alpha Vantage', key: 'AV_****************', service: 'market_data', lastUsed: '2 min ago', status: 'active' },
    { id: '2', name: 'OpenAI GPT-4', key: 'sk-****************', service: 'ai_analysis', lastUsed: '5 min ago', status: 'active' },
    { id: '3', name: 'Polygon.io', key: 'PG_****************', service: 'options_flow', lastUsed: '1 hour ago', status: 'active' },
  ]);

  const [users, setUsers] = useState<User[]>([
    { id: '1', email: 'admin@goldensignals.ai', role: 'Admin', status: 'active', lastLogin: '10 min ago', apiUsage: 1247 },
    { id: '2', email: 'analyst@goldensignals.ai', role: 'Analyst', status: 'active', lastLogin: '1 hour ago', apiUsage: 823 },
    { id: '3', email: 'viewer@goldensignals.ai', role: 'Viewer', status: 'active', lastLogin: '2 days ago', apiUsage: 156 },
  ]);

  const [systemHealth, setSystemHealth] = useState({
    apiStatus: 'active',
    databaseStatus: 'active',
    redisStatus: 'active',
    websocketStatus: 'active',
    cpuUsage: 45,
    memoryUsage: 62,
    activeConnections: 127,
    signalsGenerated: 1247,
  });

  // Check if user has admin access
  React.useEffect(() => {
    // In production, verify admin access
    const isAdmin = localStorage.getItem('userRole') === 'admin';
    if (!isAdmin && process.env.NODE_ENV === 'production') {
      navigate('/');
    }
  }, [navigate]);

  const handleSaveConfig = () => {
    // Save configuration
    console.log('Saving config:', systemConfig);
    // Show success message
  };

  const handleAddApiKey = () => {
    setSelectedItem(null);
    setEditDialogOpen(true);
  };

  const handleEditApiKey = (apiKey: APIKey) => {
    setSelectedItem(apiKey);
    setEditDialogOpen(true);
  };

  const handleDeleteApiKey = (id: string) => {
    setApiKeys(apiKeys.filter(key => key.id !== id));
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700, mb: 1, ...utilityClasses.textGradient }}>
            Admin Panel
          </Typography>
          <Typography variant="body2" color="text.secondary">
            System administration and configuration
          </Typography>
        </Box>
        <Chip
          icon={<AdminPanelSettings />}
          label="Admin Access"
          sx={{
            backgroundColor: 'rgba(255, 215, 0, 0.1)',
            color: '#FFD700',
            border: '1px solid rgba(255, 215, 0, 0.3)',
          }}
        />
      </Box>

      {/* Security Alert */}
      <Alert severity="warning" icon={<Lock />} sx={{ mb: 3 }}>
        You are in the admin panel. All actions are logged and monitored for security purposes.
      </Alert>

      {/* Tabs */}
      <StyledCard>
        <CardContent>
          <Tabs
            value={tabValue}
            onChange={(_, newValue) => setTabValue(newValue)}
            sx={{
              borderBottom: 1,
              borderColor: 'divider',
              '& .MuiTab-root': {
                textTransform: 'none',
                fontWeight: 600,
              },
            }}
          >
            <Tab icon={<Settings />} label="System Config" />
            <Tab icon={<Key />} label="API Keys" />
            <Tab icon={<People />} label="Users" />
            <Tab icon={<MonitorHeart />} label="Monitoring" />
            <Tab icon={<Storage />} label="Data" />
          </Tabs>

          {/* System Configuration Tab */}
          <TabPanel value={tabValue} index={0}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" sx={{ mb: 2 }}>Core Settings</Typography>
                
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={systemConfig.signalGenerationEnabled}
                        onChange={(e) => setSystemConfig({ ...systemConfig, signalGenerationEnabled: e.target.checked })}
                      />
                    }
                    label="Signal Generation Enabled"
                  />
                  
                  <TextField
                    label="Max Concurrent Agents"
                    type="number"
                    value={systemConfig.maxConcurrentAgents}
                    onChange={(e) => setSystemConfig({ ...systemConfig, maxConcurrentAgents: parseInt(e.target.value) })}
                    fullWidth
                  />
                  
                  <TextField
                    label="Consensus Threshold"
                    type="number"
                    value={systemConfig.consensusThreshold}
                    onChange={(e) => setSystemConfig({ ...systemConfig, consensusThreshold: parseFloat(e.target.value) })}
                    inputProps={{ min: 0, max: 1, step: 0.05 }}
                    fullWidth
                  />
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={systemConfig.cacheEnabled}
                        onChange={(e) => setSystemConfig({ ...systemConfig, cacheEnabled: e.target.checked })}
                      />
                    }
                    label="Cache Enabled"
                  />
                </Box>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="h6" sx={{ mb: 2 }}>Rate Limiting</Typography>
                
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <TextField
                    label="Rate Limit (per minute)"
                    type="number"
                    value={systemConfig.rateLimitPerMinute}
                    onChange={(e) => setSystemConfig({ ...systemConfig, rateLimitPerMinute: parseInt(e.target.value) })}
                    fullWidth
                  />
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={systemConfig.maintenanceMode}
                        onChange={(e) => setSystemConfig({ ...systemConfig, maintenanceMode: e.target.checked })}
                        color="warning"
                      />
                    }
                    label="Maintenance Mode"
                  />
                  
                  <Button
                    variant="contained"
                    startIcon={<Save />}
                    onClick={handleSaveConfig}
                    sx={{ mt: 2 }}
                  >
                    Save Configuration
                  </Button>
                </Box>
              </Grid>
            </Grid>
          </TabPanel>

          {/* API Keys Tab */}
          <TabPanel value={tabValue} index={1}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
              <Typography variant="h6">API Key Management</Typography>
              <Button
                variant="outlined"
                startIcon={<Add />}
                onClick={handleAddApiKey}
                sx={{ borderColor: 'rgba(255, 215, 0, 0.3)' }}
              >
                Add Key
              </Button>
            </Box>
            
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Service</TableCell>
                    <TableCell>Name</TableCell>
                    <TableCell>API Key</TableCell>
                    <TableCell>Last Used</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {apiKeys.map((apiKey) => (
                    <TableRow key={apiKey.id}>
                      <TableCell>{apiKey.service}</TableCell>
                      <TableCell>{apiKey.name}</TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                            {showApiKey === apiKey.id ? apiKey.key : apiKey.key.substring(0, 10) + '********'}
                          </Typography>
                          <IconButton
                            size="small"
                            onClick={() => setShowApiKey(showApiKey === apiKey.id ? null : apiKey.id)}
                          >
                            {showApiKey === apiKey.id ? <VisibilityOff /> : <Visibility />}
                          </IconButton>
                        </Box>
                      </TableCell>
                      <TableCell>{apiKey.lastUsed}</TableCell>
                      <TableCell>
                        <StatusChip
                          label={apiKey.status}
                          size="small"
                          status={apiKey.status === 'active' ? 'active' : 'error'}
                        />
                      </TableCell>
                      <TableCell align="center">
                        <IconButton size="small" onClick={() => handleEditApiKey(apiKey)}>
                          <Edit />
                        </IconButton>
                        <IconButton size="small" onClick={() => handleDeleteApiKey(apiKey.id)}>
                          <Delete />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>

          {/* Users Tab */}
          <TabPanel value={tabValue} index={2}>
            <Typography variant="h6" sx={{ mb: 3 }}>User Management</Typography>
            
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Email</TableCell>
                    <TableCell>Role</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Last Login</TableCell>
                    <TableCell>API Usage</TableCell>
                    <TableCell align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {users.map((user) => (
                    <TableRow key={user.id}>
                      <TableCell>{user.email}</TableCell>
                      <TableCell>
                        <Chip label={user.role} size="small" />
                      </TableCell>
                      <TableCell>
                        <StatusChip
                          label={user.status}
                          size="small"
                          status={user.status === 'active' ? 'active' : 'warning'}
                        />
                      </TableCell>
                      <TableCell>{user.lastLogin}</TableCell>
                      <TableCell>{user.apiUsage.toLocaleString()}</TableCell>
                      <TableCell align="center">
                        <IconButton size="small">
                          <Edit />
                        </IconButton>
                        <IconButton size="small">
                          <Delete />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>

          {/* Monitoring Tab */}
          <TabPanel value={tabValue} index={3}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Typography variant="h6" sx={{ mb: 3 }}>System Health</Typography>
              </Grid>
              
              {/* Service Status */}
              <Grid item xs={12} md={6}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography>API Service</Typography>
                    <StatusChip
                      icon={<CheckCircle />}
                      label="Active"
                      size="small"
                      status="active"
                    />
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography>Database</Typography>
                    <StatusChip
                      icon={<CheckCircle />}
                      label="Active"
                      size="small"
                      status="active"
                    />
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography>Redis Cache</Typography>
                    <StatusChip
                      icon={<CheckCircle />}
                      label="Active"
                      size="small"
                      status="active"
                    />
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography>WebSocket Server</Typography>
                    <StatusChip
                      icon={<CheckCircle />}
                      label="Active"
                      size="small"
                      status="active"
                    />
                  </Box>
                </Box>
              </Grid>
              
              {/* Resource Usage */}
              <Grid item xs={12} md={6}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography>CPU Usage</Typography>
                      <Typography>{systemHealth.cpuUsage}%</Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={systemHealth.cpuUsage}
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Box>
                  <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography>Memory Usage</Typography>
                      <Typography>{systemHealth.memoryUsage}%</Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={systemHealth.memoryUsage}
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Box>
                </Box>
              </Grid>
              
              {/* Metrics */}
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="h6" sx={{ mb: 3 }}>Real-time Metrics</Typography>
                <Grid container spacing={3}>
                  <Grid item xs={6} md={3}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Speed sx={{ fontSize: 40, color: '#FFD700', mb: 1 }} />
                      <Typography variant="h4">{systemHealth.activeConnections}</Typography>
                      <Typography variant="body2" color="text.secondary">Active Connections</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Timeline sx={{ fontSize: 40, color: '#FFD700', mb: 1 }} />
                      <Typography variant="h4">{systemHealth.signalsGenerated}</Typography>
                      <Typography variant="body2" color="text.secondary">Signals Today</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Memory sx={{ fontSize: 40, color: '#FFD700', mb: 1 }} />
                      <Typography variant="h4">94.2%</Typography>
                      <Typography variant="body2" color="text.secondary">Accuracy Rate</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Box sx={{ textAlign: 'center' }}>
                      <MonitorHeart sx={{ fontSize: 40, color: '#FFD700', mb: 1 }} />
                      <Typography variant="h4">99.9%</Typography>
                      <Typography variant="body2" color="text.secondary">Uptime</Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Data Management Tab */}
          <TabPanel value={tabValue} index={4}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" sx={{ mb: 2 }}>Data Sources</Typography>
                <List>
                  <ListItem>
                    <ListItemText
                      primary="Market Data"
                      secondary="Alpha Vantage, Polygon.io"
                    />
                    <ListItemSecondaryAction>
                      <StatusChip label="Active" size="small" status="active" />
                    </ListItemSecondaryAction>
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="News & Sentiment"
                      secondary="NewsAPI, Reddit API"
                    />
                    <ListItemSecondaryAction>
                      <StatusChip label="Active" size="small" status="active" />
                    </ListItemSecondaryAction>
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Options Flow"
                      secondary="CBOE, Custom Feed"
                    />
                    <ListItemSecondaryAction>
                      <StatusChip label="Active" size="small" status="active" />
                    </ListItemSecondaryAction>
                  </ListItem>
                </List>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="h6" sx={{ mb: 2 }}>Cache Management</Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography>Signal Cache</Typography>
                    <Typography>1,247 entries</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography>Market Data Cache</Typography>
                    <Typography>8,432 entries</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography>Model Cache</Typography>
                    <Typography>156 MB</Typography>
                  </Box>
                  <Button
                    variant="outlined"
                    startIcon={<Refresh />}
                    sx={{ mt: 2, borderColor: 'rgba(255, 215, 0, 0.3)' }}
                  >
                    Clear All Caches
                  </Button>
                </Box>
              </Grid>
              
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="h6" sx={{ mb: 2 }}>Backup & Restore</Typography>
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <Button
                    variant="outlined"
                    startIcon={<CloudUpload />}
                    sx={{ borderColor: 'rgba(255, 215, 0, 0.3)' }}
                  >
                    Backup Database
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<Storage />}
                    sx={{ borderColor: 'rgba(255, 215, 0, 0.3)' }}
                  >
                    Export Signals
                  </Button>
                </Box>
              </Grid>
            </Grid>
          </TabPanel>
        </CardContent>
      </StyledCard>

      {/* Edit Dialog */}
      <Dialog
        open={editDialogOpen}
        onClose={() => setEditDialogOpen(false)}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: { ...utilityClasses.glassmorphism }
        }}
      >
        <DialogTitle>
          {selectedItem ? 'Edit API Key' : 'Add New API Key'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
            <FormControl fullWidth>
              <InputLabel>Service</InputLabel>
              <Select defaultValue={selectedItem?.service || ''}>
                <MenuItem value="market_data">Market Data</MenuItem>
                <MenuItem value="ai_analysis">AI Analysis</MenuItem>
                <MenuItem value="options_flow">Options Flow</MenuItem>
                <MenuItem value="news_sentiment">News & Sentiment</MenuItem>
              </Select>
            </FormControl>
            <TextField
              label="Name"
              fullWidth
              defaultValue={selectedItem?.name || ''}
            />
            <TextField
              label="API Key"
              fullWidth
              type="password"
              defaultValue={selectedItem?.key || ''}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={() => setEditDialogOpen(false)}>
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AdminPanel;
