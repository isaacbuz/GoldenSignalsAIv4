export function downloadJSON(data: any, filename = "export.json") {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  triggerDownload(url, filename);
}

export function downloadCSV(data: Record<string, any>[], filename = "export.csv") {
  const keys = Object.keys(data[0]);
  const csv = [keys.join(",")].concat(
    data.map(row => keys.map(k => `"${(row[k] ?? "").toString().replace(/"/g, '""')}"`).join(","))
  ).join("\n");

  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  triggerDownload(url, filename);
}

function triggerDownload(url: string, filename: string) {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
