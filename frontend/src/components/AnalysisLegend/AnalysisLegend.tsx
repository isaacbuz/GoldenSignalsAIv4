import React from 'react';

export interface AnalysisLegendProps {
  items: any; // aiAnalysis type
}

export const AnalysisLegend: React.FC<AnalysisLegendProps> = ({ items }) => {
  if (!items) return null;
  return (
    <div className="legend">
      <p>Stop Loss: ${items.stopLoss}</p>
      <p>Take Profit: ${items.takeProfit}</p>
      <p>Rationale: {items.rationale}</p>
    </div>
  );
};
