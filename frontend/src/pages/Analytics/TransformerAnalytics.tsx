import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { transformerService, TransformerPredictionData } from '../../services/transformerService';
import TransformerPredictionChart from '../../components/AI/TransformerPredictionChart';

const TransformerAnalytics: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState<string>('EURUSD');
  const [timeframe, setTimeframe] = useState<string>('1h');
  const [predictionData, setPredictionData] = useState<TransformerPredictionData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'];
  const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];

  useEffect(() => {
    fetchPredictionData();
  }, [selectedSymbol, timeframe]);

  const fetchPredictionData = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await transformerService.getPrediction(selectedSymbol, timeframe);
      setPredictionData(data);
    } catch (err) {
      setError('Failed to fetch prediction data. Please try again later.');
      console.error('Error fetching prediction data:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-7xl mx-auto"
      >
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Transformer Analytics</h1>
          <p className="text-gray-400">
            Advanced price predictions using transformer neural networks
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
          <div className="lg:col-span-1">
            <div className="bg-gray-800 rounded-lg p-4">
              <h2 className="text-xl font-semibold text-white mb-4">Settings</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">
                    Symbol
                  </label>
                  <select
                    value={selectedSymbol}
                    onChange={(e) => setSelectedSymbol(e.target.value)}
                    className="w-full bg-gray-700 text-white rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {symbols.map((symbol) => (
                      <option key={symbol} value={symbol}>
                        {symbol}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">
                    Timeframe
                  </label>
                  <select
                    value={timeframe}
                    onChange={(e) => setTimeframe(e.target.value)}
                    className="w-full bg-gray-700 text-white rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {timeframes.map((tf) => (
                      <option key={tf} value={tf}>
                        {tf}
                      </option>
                    ))}
                  </select>
                </div>

                <button
                  onClick={fetchPredictionData}
                  className="w-full bg-blue-600 text-white rounded-lg px-4 py-2 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  Refresh Data
                </button>
              </div>
            </div>
          </div>

          <div className="lg:col-span-3">
            {loading ? (
              <div className="bg-gray-800 rounded-lg p-8 flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
              </div>
            ) : error ? (
              <div className="bg-gray-800 rounded-lg p-8">
                <div className="text-red-500 text-center">{error}</div>
              </div>
            ) : predictionData ? (
              <TransformerPredictionChart data={predictionData} />
            ) : (
              <div className="bg-gray-800 rounded-lg p-8">
                <div className="text-gray-400 text-center">No prediction data available</div>
              </div>
            )}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">About Transformer Model</h2>
          <div className="prose prose-invert max-w-none">
            <p className="text-gray-400">
              This transformer model uses a neural network architecture similar to those used in large language models
              to predict future price movements. The model analyzes historical price data and technical indicators
              to identify patterns and make predictions about future price movements.
            </p>
            <p className="text-gray-400 mt-4">
              Key features of the transformer model:
            </p>
            <ul className="list-disc list-inside text-gray-400 mt-2 space-y-2">
              <li>Attention mechanism to identify important price patterns</li>
              <li>Multiple technical indicators for comprehensive analysis</li>
              <li>Real-time predictions with confidence scores</li>
              <li>Adaptive learning from market conditions</li>
            </ul>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default TransformerAnalytics; 