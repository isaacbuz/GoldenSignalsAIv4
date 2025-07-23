import React, { useState } from 'react';
import { Box, Tooltip, IconButton, Popover, Typography, Chip, Grid } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { Hub, CheckCircle, Warning, Error as ErrorIcon } from '@mui/icons-material';

interface AgentStatusPanelProps {
  status: Record<string, any>;
}

export const AgentStatusPanel: React.FC<AgentStatusPanelProps> = ({ status }) => {
  const theme = useTheme();
  const [anchorEl, setAnchorEl] = useState<HTMLButtonElement | null>(null);

  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const open = Boolean(anchorEl);

  // Count agent statuses
  const statusCounts = Object.values(status).reduce((acc, agent: any) => {
    if (agent.available) acc.available++;
    else acc.unavailable++;
    return acc;
  }, { available: 0, unavailable: 0 });

  const getStatusIcon = (available: boolean) => {
    return available ? (
      <CheckCircle sx={{ color: theme.palette.success.main, fontSize: 16 }} />
    ) : (
      <ErrorIcon sx={{ color: theme.palette.error.main, fontSize: 16 }} />
    );
  };

  const getAgentTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      technical_agent: theme.palette.info.main,
      forecast_agent: theme.palette.primary.main,
      sentiment_agent: theme.palette.secondary.main,
      market_agent: theme.palette.warning.main,
      options_agent: theme.palette.success.main
    };
    return colors[type] || theme.palette.grey[500];
  };

  return (
    <>
      <Tooltip title="Agent Network Status">
        <IconButton
          onClick={handleClick}
          size="small"
          sx={{
            position: 'relative',
            color: statusCounts.unavailable > 0 ? theme.palette.warning.main : theme.palette.success.main
          }}
        >
          <Hub />
          {statusCounts.unavailable > 0 && (
            <Box
              sx={{
                position: 'absolute',
                top: -4,
                right: -4,
                width: 8,
                height: 8,
                borderRadius: '50%',
                bgcolor: theme.palette.error.main,
                animation: 'pulse 2s infinite'
              }}
            />
          )}
        </IconButton>
      </Tooltip>

      <Popover
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
      >
        <Box sx={{ p: 2, minWidth: 300 }}>
          <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
            Agent Network Status
          </Typography>

          <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
            <Chip
              icon={<CheckCircle />}
              label={`${statusCounts.available} Active`}
              size="small"
              color="success"
              variant="outlined"
            />
            {statusCounts.unavailable > 0 && (
              <Chip
                icon={<Warning />}
                label={`${statusCounts.unavailable} Offline`}
                size="small"
                color="error"
                variant="outlined"
              />
            )}
          </Box>

          <Grid container spacing={1}>
            {Object.entries(status).map(([agentName, agentStatus]: [string, any]) => (
              <Grid item xs={12} key={agentName}>
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    p: 1,
                    borderRadius: 1,
                    bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)',
                    border: `1px solid ${theme.palette.divider}`
                  }}
                >
                  {getStatusIcon(agentStatus.available)}
                  <Typography variant="body2" sx={{ flex: 1 }}>
                    {agentName}
                  </Typography>
                  <Box
                    sx={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      bgcolor: getAgentTypeColor(agentStatus.type)
                    }}
                  />
                </Box>
              </Grid>
            ))}
          </Grid>

          <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
            Last updated: {new Date().toLocaleTimeString()}
          </Typography>
        </Box>
      </Popover>
    </>
  );
};
