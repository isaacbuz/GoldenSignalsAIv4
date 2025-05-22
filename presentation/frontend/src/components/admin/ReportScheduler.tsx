import React, { useState } from "react";

interface Props {
  onSchedule: (settings: {
    type: string;
    format: "csv" | "json";
    frequency: "daily" | "weekly";
  }) => void;
}

const ReportScheduler: React.FC<Props> = ({ onSchedule }) => {
  const [type, setType] = useState("signals");
  const [format, setFormat] = useState<"csv" | "json">("csv");
  const [frequency, setFrequency] = useState<"daily" | "weekly">("daily");

  return (
    <div className="mt-10">
      <h2 className="text-lg font-bold mb-4">📅 Schedule Automated Reports</h2>
      <div className="grid gap-4 md:grid-cols-3">
        <div>
          <label className="block text-sm font-medium mb-1">Report Type</label>
          <select
            value={type}
            onChange={(e) => setType(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-2 w-full"
          >
            <option value="signals">Signals</option>
            <option value="agents">Agent Performance</option>
            <option value="audit">Audit Logs</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Format</label>
          <select
            value={format}
            onChange={(e) => setFormat(e.target.value as "csv" | "json")}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-2 w-full"
          >
            <option value="csv">CSV</option>
            <option value="json">JSON</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Frequency</label>
          <select
            value={frequency}
            onChange={(e) => setFrequency(e.target.value as "daily" | "weekly")}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-2 w-full"
          >
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
          </select>
        </div>
      </div>
      <button
        className="mt-4 px-4 py-2 bg-green-600 hover:bg-green-700 text-sm font-medium rounded text-black"
        onClick={() => onSchedule({ type, format, frequency })}
      >
        📤 Schedule Report
      </button>
    </div>
  );
};

export default ReportScheduler;
