import React from "react";

interface AdminKpiCardsProps {
  stats: {
    totalSignals: number;
    avgConfidence: number;
    buyPercentage: number;
    topAgent: string;
  };
}

const AdminKpiCards: React.FC<AdminKpiCardsProps> = ({ stats }) => {
  const box = (title: string, value: string | number) => (
    <div className="bg-gray-800 p-4 rounded-lg shadow w-full text-center">
      <h3 className="text-sm text-gray-400">{title}</h3>
      <p className="text-2xl font-bold text-green-400">{value}</p>
    </div>
  );

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
      {box("Total Signals", stats.totalSignals)}
      {box("Buy %", `${stats.buyPercentage}%`)}
      {box("Avg. Confidence", `${stats.avgConfidence.toFixed(1)}%`)}
      {box("Top Agent", stats.topAgent)}
    </div>
  );
};

export default AdminKpiCards;
