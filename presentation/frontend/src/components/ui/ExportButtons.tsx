import React from "react";
import { downloadCSV, downloadJSON } from "../../utils/exportUtils";

interface Props {
  data: any[];
  title: string;
}

const ExportButtons: React.FC<Props> = ({ data, title }) => {
  return (
    <div className="flex space-x-2 my-4">
      <button
        onClick={() => downloadCSV(data, `${title}.csv`)}
        className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white text-sm"
      >
        📄 Export CSV
      </button>
      <button
        onClick={() => downloadJSON(data, `${title}.json`)}
        className="px-3 py-1 bg-gray-700 hover:bg-gray-800 rounded text-white text-sm"
      >
        🔐 Export JSON
      </button>
    </div>
  );
};

export default ExportButtons;
