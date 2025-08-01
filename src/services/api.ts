import axios from 'axios';
const api = axios.create({ baseURL: 'http://localhost:8000' });
// Fetch signals from backend
export const fetchSignals = async (symbol) => {
    const res = await api.get(`/signals?symbol=${symbol}`);
    return res.data;
};
// Similar for predictions, options

export const fetchStockData = async (symbol: string, timeframe: string) => {
    const response = await axios.get(`/api/fetch-data?symbol=${symbol}&timeframe=${timeframe}`);
    return response.data; // { time, open, high, low, close }[]
};

export const runAIAnalysis = async (data: any, symbol: string, timeframe: string) => {
    const response = await axios.post('/api/analyze', { data, symbol, timeframe });
    return response.data; // { entries, profitZones, stopLoss, takeProfit, rationale }
};

export default api;
export { fetchSignals };
