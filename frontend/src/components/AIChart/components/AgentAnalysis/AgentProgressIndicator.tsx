/**
 * AgentProgressIndicator Component
 *
 * Displays real-time progress of agent workflow analysis.
 * Shows which stage of the analysis is currently running and overall progress.
 *
 * The workflow stages are:
 * 1. Market Regime Detection
 * 2. Collecting Agent Signals
 * 3. Searching Historical Patterns
 * 4. Building Consensus
 * 5. Assessing Risk
 * 6. Making Final Decision
 */

import React from 'react';
import {
  Box,
  LinearProgress,
  Typography,
  Paper,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Chip,
  Fade,
  alpha,
  useTheme,
} from '@mui/material';
import {
  TrendingUp as RegimeIcon,
  Psychology as SignalsIcon,
  History as PatternsIcon,
  Groups as ConsensusIcon,
  Security as RiskIcon,
  CheckCircle as DecisionIcon,
  Timer as TimerIcon,
  CheckCircle,
} from '@mui/icons-material';
import { WorkflowStage } from '../../../../types/agent.types';

interface AgentProgressIndicatorProps {
  /**
   * Current stage of the workflow
   */
  currentStage: WorkflowStage | null;

  /**
   * Overall progress percentage (0-100)
   */
  progress: number;

  /**
   * List of status messages from the workflow
   */
  messages: string[];

  /**
   * Whether to show detailed step information
   */
  showDetails?: boolean;

  /**
   * Compact mode for smaller displays
   */
  compact?: boolean;
}

/**
 * Workflow stage configuration with display properties
 */
const WORKFLOW_STAGES: Array<{
  id: WorkflowStage;
  label: string;
  description: string;
  icon: React.ReactNode;
  duration: string;
}> = [
  {
    id: 'market_regime',
    label: 'Market Regime Detection',
    description: 'Analyzing current market conditions',
    icon: <RegimeIcon />,
    duration: '~2s',
  },
  {
    id: 'collecting_signals',
    label: 'Collecting Agent Signals',
    description: 'Gathering signals from 8 specialized agents',
    icon: <SignalsIcon />,
    duration: '~3s',
  },
  {
    id: 'searching_patterns',
    label: 'Historical Pattern Search',
    description: 'Finding similar patterns in historical data',
    icon: <PatternsIcon />,
    duration: '~2s',
  },
  {
    id: 'building_consensus',
    label: 'Building Consensus',
    description: 'Combining agent signals with weighted voting',
    icon: <ConsensusIcon />,
    duration: '~1s',
  },
  {
    id: 'assessing_risk',
    label: 'Risk Assessment',
    description: 'Calculating position size and risk parameters',
    icon: <RiskIcon />,
    duration: '~1s',
  },
  {
    id: 'making_decision',
    label: 'Final Decision',
    description: 'Validating with Guardrails AI',
    icon: <DecisionIcon />,
    duration: '~1s',
  },
];

/**
 * Get the index of the current stage
 */
const getCurrentStageIndex = (stage: WorkflowStage | null): number => {
  if (!stage) return -1;
  return WORKFLOW_STAGES.findIndex(s => s.id === stage);
};

export const AgentProgressIndicator: React.FC<AgentProgressIndicatorProps> = ({
  currentStage,
  progress,
  messages,
  showDetails = true,
  compact = false,
}) => {
  const theme = useTheme();
  const currentStageIndex = getCurrentStageIndex(currentStage);
  const isComplete = currentStage === 'complete' || progress === 100;

  // Compact mode - just a progress bar with current stage
  if (compact) {
    return (
      <Box sx={{ width: '100%', p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Typography variant="body2" color="text.secondary" sx={{ flex: 1 }}>
            {isComplete ? 'Analysis Complete' :
             currentStage ? WORKFLOW_STAGES[currentStageIndex]?.label : 'Initializing...'}
          </Typography>
          <Typography variant="body2" color="primary" fontWeight="medium">
            {progress}%
          </Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{
            height: 6,
            borderRadius: 3,
            backgroundColor: alpha(theme.palette.primary.main, 0.1),
            '& .MuiLinearProgress-bar': {
              borderRadius: 3,
              background: isComplete
                ? theme.palette.success.main
                : `linear-gradient(90deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
            },
          }}
        />
      </Box>
    );
  }

  // Full mode with stepper
  return (
    <Paper
      elevation={0}
      sx={{
        p: 3,
        backgroundColor: alpha(theme.palette.background.paper, 0.8),
        backdropFilter: 'blur(10px)',
        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
      }}
    >
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TimerIcon color="primary" />
            Agent Workflow Analysis
          </Typography>
          <Chip
            label={`${progress}%`}
            color={isComplete ? 'success' : 'primary'}
            size="small"
            sx={{ fontWeight: 'bold' }}
          />
        </Box>

        {/* Overall progress bar */}
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{
            height: 8,
            borderRadius: 4,
            backgroundColor: alpha(theme.palette.primary.main, 0.1),
            '& .MuiLinearProgress-bar': {
              borderRadius: 4,
              background: isComplete
                ? theme.palette.success.main
                : `linear-gradient(90deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
            },
          }}
        />
      </Box>

      {/* Workflow stages */}
      {showDetails && (
        <Stepper activeStep={currentStageIndex} orientation="vertical">
          {WORKFLOW_STAGES.map((stage, index) => {
            const isActive = index === currentStageIndex;
            const isCompleted = index < currentStageIndex || isComplete;

            return (
              <Step key={stage.id} completed={isCompleted}>
                <StepLabel
                  icon={
                    <Box
                      sx={{
                        color: isCompleted
                          ? theme.palette.success.main
                          : isActive
                          ? theme.palette.primary.main
                          : theme.palette.text.disabled,
                      }}
                    >
                      {stage.icon}
                    </Box>
                  }
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography
                      variant="body2"
                      fontWeight={isActive ? 600 : 400}
                      color={isActive ? 'primary' : 'textPrimary'}
                    >
                      {stage.label}
                    </Typography>
                    {isActive && (
                      <Fade in>
                        <Chip
                          label="Running"
                          size="small"
                          color="primary"
                          sx={{ height: 20, fontSize: '0.7rem' }}
                        />
                      </Fade>
                    )}
                  </Box>
                </StepLabel>
                <StepContent>
                  <Typography variant="caption" color="text.secondary">
                    {stage.description}
                  </Typography>
                  {isActive && messages.length > 0 && (
                    <Typography
                      variant="caption"
                      display="block"
                      sx={{
                        mt: 0.5,
                        fontStyle: 'italic',
                        color: theme.palette.primary.main,
                      }}
                    >
                      {messages[messages.length - 1]}
                    </Typography>
                  )}
                </StepContent>
              </Step>
            );
          })}
        </Stepper>
      )}

      {/* Completion message */}
      {isComplete && (
        <Fade in>
          <Box
            sx={{
              mt: 3,
              p: 2,
              borderRadius: 1,
              backgroundColor: alpha(theme.palette.success.main, 0.1),
              border: `1px solid ${alpha(theme.palette.success.main, 0.3)}`,
            }}
          >
            <Typography
              variant="body2"
              color="success.main"
              sx={{ display: 'flex', alignItems: 'center', gap: 1 }}
            >
              <CheckCircle />
              Analysis completed successfully!
            </Typography>
          </Box>
        </Fade>
      )}
    </Paper>
  );
};
