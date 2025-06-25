import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  LinearProgress,
  Button,
  Chip,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  Stack,
} from '@mui/material';
import {
  ModelTraining,
  PlayArrow,
  Stop,
  Refresh,
  History,
  CloudUpload,
  Assessment,
  Memory,
  Speed,
  Storage,
  Warning,
  CheckCircle,
  Error,
  Schedule,
  TrendingUp,
  Code,
  BugReport,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { utilityClasses } from '../../theme/goldenTheme';

const StyledCard = styled(Card)(({ theme }) => ({
  ...utilityClasses.glassmorphism,
  height: '100%',
}));

const StatusChip = styled(Chip)(({ status }: { status: string }) => ({
  fontWeight: 600,
  ...(status === 'running' && {
    backgroundColor: 'rgba(76, 175, 80, 0.1)',
    color: '#4CAF50',
    border: '1px solid rgba(76, 175, 80, 0.3)',
  }),
  ...(status === 'training' && {
    backgroundColor: 'rgba(255, 165, 0, 0.1)',
    color: '#FFA500',
    border: '1px solid rgba(255, 165, 0, 0.3)',
  }),
  ...(status === 'stopped' && {
    backgroundColor: 'rgba(244, 67, 54, 0.1)',
    color: '#F44336',
    border: '1px solid rgba(244, 67, 54, 0.3)',
  }),
  ...(status === 'scheduled' && {
    backgroundColor: 'rgba(33, 150, 243, 0.1)',
    color: '#2196F3',
    border: '1px solid rgba(33, 150, 243, 0.3)',
  }),
}));

const ResourceMeter = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  padding: theme.spacing(2),
  borderRadius: theme.shape.borderRadius,
  backgroundColor: 'rgba(255, 255, 255, 0.02)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
}));

interface Model {
  id: string;
  name: string;
  type: string;
  version: string;
  status: 'running' | 'training' | 'stopped' | 'scheduled';
  accuracy: number;
  lastTrained: string;
  trainingProgress?: number;
  metrics: {
    precision: number;
    recall: number;
    f1Score: number;
  };
}

const ModelDashboard: React.FC = () => {
  const [models, setModels] = useState<Model[]>([
    {
      id: '1',
      name: 'Sentiment Analyzer',
      type: 'NLP',
      version: '2.1.0',
      status: 'running',
      accuracy: 92.5,
      lastTrained: '2 hours ago',
      metrics: { precision: 0.93, recall: 0.91, f1Score: 0.92 },
    },
    {
      id: '2',
      name: 'Technical Pattern Recognition',
      type: 'CNN',
      version: '3.0.2',
      status: 'training',
      accuracy: 94.8,
      lastTrained: '1 day ago',
      trainingProgress: 67,
      metrics: { precision: 0.95, recall: 0.94, f1Score: 0.945 },
    },
    {
      id: '3',
      name: 'Options Flow Analyzer',
      type: 'Ensemble',
      version: '1.5.0',
      status: 'running',
      accuracy: 96.2,
      lastTrained: '3 hours ago',
      metrics: { precision: 0.97, recall: 0.95, f1Score: 0.96 },
    },
    {
      id: '4',
      name: 'Risk Assessment Model',
      type: 'XGBoost',
      version: '2.0.1',
      status: 'scheduled',
      accuracy: 91.7,
      lastTrained: '12 hours ago',
      metrics: { precision: 0.92, recall: 0.91, f1Score: 0.915 },
    },
    {
      id: '5',
      name: 'Market Regime Classifier',
      type: 'LSTM',
      version: '1.8.3',
      status: 'running',
      accuracy: 89.3,
      lastTrained: '5 hours ago',
      metrics: { precision: 0.90, recall: 0.88, f1Score: 0.89 },
    },
  ]);

  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [resourceUsage, setResourceUsage] = useState({
    cpu: 45,
    gpu: 78,
    memory: 62,
    storage: 41,
  });

  const handleTrainModel = (modelId: string) => {
    setModels(prev =>
      prev.map(model =>
        model.id === modelId
          ? { ...model, status: 'training', trainingProgress: 0 }
          : model
      )
    );

    // Simulate training progress
    const interval = setInterval(() => {
      setModels(prev =>
        prev.map(model => {
          if (model.id === modelId && model.status === 'training') {
            const newProgress = (model.trainingProgress || 0) + 10;
            if (newProgress >= 100) {
              clearInterval(interval);
              return {
                ...model,
                status: 'running',
                trainingProgress: undefined,
                lastTrained: 'Just now',
                accuracy: model.accuracy + (Math.random() * 2 - 1),
              };
            }
            return { ...model, trainingProgress: newProgress };
          }
          return model;
        })
      );
    }, 1000);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <CheckCircle sx={{ fontSize: 18 }} />;
      case 'training':
        return <ModelTraining sx={{ fontSize: 18 }} />;
      case 'stopped':
        return <Error sx={{ fontSize: 18 }} />;
      case 'scheduled':
        return <Schedule sx={{ fontSize: 18 }} />;
      default:
        return null;
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 1, ...utilityClasses.textGradient }}>
          Model Dashboard
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Manage and monitor AI model performance and training
        </Typography>
      </Box>

      {/* System Status Alert */}
      <Alert severity="success" icon={<CheckCircle />} sx={{ mb: 3 }}>
        All models operational. Next scheduled training: Market Regime Classifier in 2 hours.
      </Alert>

      {/* Resource Usage */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <ResourceMeter>
            <Memory sx={{ color: '#FFD700' }} />
            <Box sx={{ flex: 1 }}>
              <Typography variant="caption" color="text.secondary">
                CPU Usage
              </Typography>
              <Typography variant="h6">{resourceUsage.cpu}%</Typography>
            </Box>
            <CircularProgress
              variant="determinate"
              value={resourceUsage.cpu}
              size={40}
              thickness={4}
              sx={{ color: '#FFD700' }}
            />
          </ResourceMeter>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <ResourceMeter>
            <Speed sx={{ color: '#4CAF50' }} />
            <Box sx={{ flex: 1 }}>
              <Typography variant="caption" color="text.secondary">
                GPU Usage
              </Typography>
              <Typography variant="h6">{resourceUsage.gpu}%</Typography>
            </Box>
            <CircularProgress
              variant="determinate"
              value={resourceUsage.gpu}
              size={40}
              thickness={4}
              sx={{ color: '#4CAF50' }}
            />
          </ResourceMeter>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <ResourceMeter>
            <Assessment sx={{ color: '#2196F3' }} />
            <Box sx={{ flex: 1 }}>
              <Typography variant="caption" color="text.secondary">
                Memory
              </Typography>
              <Typography variant="h6">{resourceUsage.memory}%</Typography>
            </Box>
            <CircularProgress
              variant="determinate"
              value={resourceUsage.memory}
              size={40}
              thickness={4}
              sx={{ color: '#2196F3' }}
            />
          </ResourceMeter>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <ResourceMeter>
            <Storage sx={{ color: '#FFA500' }} />
            <Box sx={{ flex: 1 }}>
              <Typography variant="caption" color="text.secondary">
                Storage
              </Typography>
              <Typography variant="h6">{resourceUsage.storage}%</Typography>
            </Box>
            <CircularProgress
              variant="determinate"
              value={resourceUsage.storage}
              size={40}
              thickness={4}
              sx={{ color: '#FFA500' }}
            />
          </ResourceMeter>
        </Grid>
      </Grid>

      {/* Models Table */}
      <StyledCard>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Active Models
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="outlined"
                startIcon={<CloudUpload />}
                size="small"
                sx={{ borderColor: 'rgba(255, 215, 0, 0.3)' }}
              >
                Deploy New
              </Button>
              <Button
                variant="outlined"
                startIcon={<Refresh />}
                size="small"
                sx={{ borderColor: 'rgba(255, 215, 0, 0.3)' }}
              >
                Refresh All
              </Button>
            </Box>
          </Box>

          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Model Name</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Version</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell align="right">Accuracy</TableCell>
                  <TableCell>Last Trained</TableCell>
                  <TableCell align="center">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {models.map((model) => (
                  <TableRow key={model.id}>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {model.name}
                      </Typography>
                    </TableCell>
                    <TableCell>{model.type}</TableCell>
                    <TableCell>
                      <Chip label={model.version} size="small" />
                    </TableCell>
                    <TableCell>
                      <StatusChip
                        label={model.status}
                        icon={getStatusIcon(model.status)}
                        size="small"
                        status={model.status}
                      />
                      {model.status === 'training' && model.trainingProgress && (
                        <LinearProgress
                          variant="determinate"
                          value={model.trainingProgress}
                          sx={{ mt: 1, height: 4, borderRadius: 2 }}
                        />
                      )}
                    </TableCell>
                    <TableCell align="right">
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 1 }}>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {model.accuracy.toFixed(1)}%
                        </Typography>
                        {model.accuracy >= 95 && <TrendingUp sx={{ fontSize: 16, color: '#4CAF50' }} />}
                      </Box>
                    </TableCell>
                    <TableCell>{model.lastTrained}</TableCell>
                    <TableCell align="center">
                      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
                        {model.status === 'running' && (
                          <Tooltip title="Stop Model">
                            <IconButton size="small" onClick={() => {}}>
                              <Stop fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                        {model.status === 'stopped' && (
                          <Tooltip title="Start Model">
                            <IconButton size="small" onClick={() => {}}>
                              <PlayArrow fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                        {model.status !== 'training' && (
                          <Tooltip title="Train Model">
                            <IconButton
                              size="small"
                              onClick={() => handleTrainModel(model.id)}
                            >
                              <ModelTraining fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                        <Tooltip title="View Details">
                          <IconButton
                            size="small"
                            onClick={() => {
                              setSelectedModel(model);
                              setDialogOpen(true);
                            }}
                          >
                            <Assessment fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Version History">
                          <IconButton size="small">
                            <History fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </StyledCard>

      {/* Training Schedule */}
      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12} md={6}>
          <StyledCard>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                Training Schedule
              </Typography>
              <Stack spacing={2}>
                <Box
                  sx={{
                    p: 2,
                    borderRadius: 2,
                    border: '1px solid rgba(255, 215, 0, 0.2)',
                    backgroundColor: 'rgba(255, 215, 0, 0.05)',
                  }}
                >
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      Market Regime Classifier
                    </Typography>
                    <Chip label="In 2 hours" size="small" />
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    Scheduled retraining with latest market data
                  </Typography>
                </Box>
                <Box
                  sx={{
                    p: 2,
                    borderRadius: 2,
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                  }}
                >
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      Sentiment Analyzer
                    </Typography>
                    <Chip label="Tomorrow 3 AM" size="small" />
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    Weekly model update with new training data
                  </Typography>
                </Box>
              </Stack>
            </CardContent>
          </StyledCard>
        </Grid>

        <Grid item xs={12} md={6}>
          <StyledCard>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                Recent Updates
              </Typography>
              <Stack spacing={2}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <CheckCircle sx={{ color: '#4CAF50' }} />
                  <Box>
                    <Typography variant="body2">
                      Options Flow Analyzer v1.5.0 deployed
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      3 hours ago • Accuracy improved by 2.1%
                    </Typography>
                  </Box>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Code sx={{ color: '#2196F3' }} />
                  <Box>
                    <Typography variant="body2">
                      Technical Pattern Recognition updated
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      1 day ago • New candlestick patterns added
                    </Typography>
                  </Box>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <BugReport sx={{ color: '#FFA500' }} />
                  <Box>
                    <Typography variant="body2">
                      Risk Model hotfix applied
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      2 days ago • Fixed edge case in volatility calculation
                    </Typography>
                  </Box>
                </Box>
              </Stack>
            </CardContent>
          </StyledCard>
        </Grid>
      </Grid>

      {/* Model Details Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: { ...utilityClasses.glassmorphism }
        }}
      >
        {selectedModel && (
          <>
            <DialogTitle>{selectedModel.name} - Detailed Metrics</DialogTitle>
            <DialogContent>
              <Grid container spacing={3} sx={{ mt: 1 }}>
                <Grid item xs={4}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" sx={{ fontWeight: 700, color: '#FFD700' }}>
                      {selectedModel.metrics.precision.toFixed(3)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Precision
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={4}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" sx={{ fontWeight: 700, color: '#4CAF50' }}>
                      {selectedModel.metrics.recall.toFixed(3)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Recall
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={4}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" sx={{ fontWeight: 700, color: '#2196F3' }}>
                      {selectedModel.metrics.f1Score.toFixed(3)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      F1 Score
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setDialogOpen(false)}>Close</Button>
              <Button variant="contained" startIcon={<ModelTraining />}>
                Retrain Model
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Box>
  );
};

export default ModelDashboard;
