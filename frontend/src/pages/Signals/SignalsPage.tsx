/**
 * Signals Page
 * 
 * Trading signals management with filtering, sorting, and real-time updates
 */

import React from 'react';
import { Box, Typography, Card, CardContent, Chip, Grid, Paper, Divider } from '@mui/material';
import { DataGrid, GridColDef } from '@mui/x-data-grid';
import { useSignals } from '../../store';
import { Signal } from '../../services/api';
import { TrendingUp, TrendingDown } from '@mui/icons-material';
import { motion } from 'framer-motion';

const columns: GridColDef[] = [
  { 
    field: 'symbol', 
    headerName: 'Symbol', 
    width: 130,
    renderCell: (params) => (
      <Chip label={params.value} variant="outlined" size="small" />
    )
  },
  {
    field: 'signal_type',
    headerName: 'Type',
    width: 120,
    renderCell: (params) => (
      <Chip 
        icon={params.value === 'BUY' ? <TrendingUp /> : <TrendingDown />}
        label={params.value} 
        color={params.value === 'BUY' ? 'success' : 'error'}
        size="small"
        variant="filled"
      />
    )
  },
  { field: 'confidence', headerName: 'Confidence', width: 130,
    renderCell: (params) => `${(params.value * 100).toFixed(1)}%`
  },
  { field: 'current_price', headerName: 'Price', width: 130,
    renderCell: (params) => `$${params.value.toFixed(2)}`
  },
  { field: 'reasoning', headerName: 'Reasoning', width: 400 },
  {
    field: 'created_at',
    headerName: 'Timestamp',
    width: 180,
    renderCell: (params) => new Date(params.value).toLocaleString(),
  },
];

export default function SignalsPage() {
  const { signals } = useSignals();

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold' }}>
          AI Signal Feed
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" sx={{ mb: 4 }}>
          Real-time stream of AI-generated trading signals
        </Typography>

        <Card>
          <CardContent>
            <div style={{ height: 600, width: '100%' }}>
              <DataGrid
                rows={signals}
                columns={columns}
                getRowId={(row: Signal) => row.signal_id || `${row.symbol}-${row.signal_type}-${Date.now()}`}
                initialState={{
                  pagination: {
                    paginationModel: { pageSize: 10 },
                  },
                }}
                pageSizeOptions={[10, 25, 50]}
                disableRowSelectionOnClick
                sx={{
                  border: 'none',
                  '& .MuiDataGrid-cell': {
                    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                  },
                }}
              />
            </div>
          </CardContent>
        </Card>
      </Box>
    </motion.div>
  );
} 