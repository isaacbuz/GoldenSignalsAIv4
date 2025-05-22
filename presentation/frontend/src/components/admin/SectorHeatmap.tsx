import React from "react";

interface SectorData {
  sector: string;
  buys: number;
  sells: number;
}

interface Props {
  data: SectorData[];
  onSelectSector?: (sector: string) => void;
}

const SectorHeatmap: React.FC<Props> = ({ data, onSelectSector }) => {
  const total = (buys: number, sells: number) => buys + sells;

  const getColor = (buys: number, sells: number) => {
    const net = buys - sells;
    if (net > 3) return "bg-green-600";
    if (net < -3) return "bg-red-600";
    return "bg-gray-700";
  };

  return (
    <div className="mt-8">
      <h2 className="text-lg font-bold mb-4">🔥 Sector Signal Heatmap</h2>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
        {data.map((s, i) => (
          <div
            key={i}
            className={`p-4 rounded-lg cursor-pointer text-white text-sm text-center shadow ${getColor(s.buys, s.sells)} hover:scale-105 transition`}
            onClick={() => onSelectSector?.(s.sector)}
            title={`${s.sector}: ${s.buys} buys, ${s.sells} sells`}
          >
            <div className="font-semibold">{s.sector}</div>
            <div className="opacity-75 text-xs">{total(s.buys, s.sells)} signals</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SectorHeatmap;
