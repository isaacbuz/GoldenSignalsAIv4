import { Box, Typography, Chip } from "@mui/material";

interface VolatilityData {
  rationale: string;
  confidence: number;
  value: string;
}

interface VolatilityBreakdownProps {
  data?: VolatilityData;
}

export function VolatilityBreakdown({ data }: VolatilityBreakdownProps) {
  if (!data) return <Typography>Loading volatility analysis...</Typography>;
  
  return (
    <Box>
      <Typography variant="h6" fontWeight="bold" gutterBottom>
        Volatility Analysis
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
        {data.rationale}
      </Typography>
      <Chip 
        label={`Confidence: ${data.confidence}%`}
        color="warning"
        size="small"
        sx={{ mb: 1 }}
      />
      <Typography variant="body2">
        {data.value}
      </Typography>
    </Box>
  );
} 