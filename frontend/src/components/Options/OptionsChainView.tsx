import React, { useState } from 'react';
import { ArrowUp, ArrowDown, Activity, AlertCircle, TrendingUp } from 'lucide-react';

interface OptionContract {
  strike: number;
  expiration: string;
  call: {
    bid: number;
    ask: number;
    last: number;
    volume: number;
    openInterest: number;
    impliedVolatility: number;
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
  };
  put: {
    bid: number;
    ask: number;
    last: number;
    volume: number;
    openInterest: number;
    impliedVolatility: number;
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
  };
}

interface OptionsChainViewProps {
  symbol: string;
  currentPrice: number;
  chain: OptionContract[];
  onSelectOption?: (contract: OptionContract, type: 'call' | 'put') => void;
  highlightUnusualActivity?: boolean;
  showGreeks?: boolean;
}

export const OptionsChainView: React.FC<OptionsChainViewProps> = ({
  symbol,
  currentPrice,
  chain,
  onSelectOption,
  highlightUnusualActivity = true,
  showGreeks = true
}) => {
  const [selectedExpiration, setSelectedExpiration] = useState<string>('all');
  const [viewMode, setViewMode] = useState<'standard' | 'greeks' | 'flow'>('standard');

  // Get unique expirations
  const expirations = Array.from(new Set(chain.map(c => c.expiration))).sort();

  // Filter chain by expiration
  const filteredChain = selectedExpiration === 'all' 
    ? chain 
    : chain.filter(c => c.expiration === selectedExpiration);

  // Identify unusual activity
  const isUnusualActivity = (volume: number, openInterest: number): boolean => {
    return volume > openInterest * 0.5 && volume > 1000;
  };

  // Calculate moneyness
  const getMoneyness = (strike: number): string => {
    const ratio = strike / currentPrice;
    if (ratio < 0.95) return 'ITM';
    if (ratio > 1.05) return 'OTM';
    return 'ATM';
  };

  // Style based on moneyness
  const getStrikeStyle = (strike: number): string => {
    const moneyness = getMoneyness(strike);
    if (moneyness === 'ITM') return 'bg-green-500/10';
    if (moneyness === 'ATM') return 'bg-blue-500/20 border-blue-500';
    return '';
  };

  return (
    <div className="bg-gray-900 rounded-xl shadow-2xl">
      {/* Header */}
      <div className="p-6 border-b border-gray-800">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h2 className="text-2xl font-bold text-white">{symbol} Options Chain</h2>
            <p className="text-gray-400 text-sm mt-1">
              Current Price: <span className="text-white font-medium">${currentPrice.toFixed(2)}</span>
            </p>
          </div>
          
          {/* View Mode Selector */}
          <div className="flex bg-gray-800 rounded-lg p-1">
            {['standard', 'greeks', 'flow'].map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode as any)}
                className={`px-4 py-2 text-sm rounded-md transition-colors capitalize ${
                  viewMode === mode
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>

        {/* Expiration Filter */}
        <div className="flex gap-2 overflow-x-auto pb-2">
          <button
            onClick={() => setSelectedExpiration('all')}
            className={`px-3 py-1 text-sm rounded-md whitespace-nowrap ${
              selectedExpiration === 'all'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-white'
            }`}
          >
            All Expirations
          </button>
          {expirations.map((exp) => (
            <button
              key={exp}
              onClick={() => setSelectedExpiration(exp)}
              className={`px-3 py-1 text-sm rounded-md whitespace-nowrap ${
                selectedExpiration === exp
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              {exp}
            </button>
          ))}
        </div>
      </div>

      {/* Options Chain Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-800/50">
            <tr>
              {/* Calls Header */}
              <th colSpan={viewMode === 'greeks' ? 7 : 6} className="text-center py-3 text-green-400 font-semibold border-r border-gray-700">
                CALLS
              </th>
              
              {/* Strike Header */}
              <th className="text-center py-3 text-white font-semibold px-4">
                STRIKE
              </th>
              
              {/* Puts Header */}
              <th colSpan={viewMode === 'greeks' ? 7 : 6} className="text-center py-3 text-red-400 font-semibold border-l border-gray-700">
                PUTS
              </th>
            </tr>
            <tr className="text-xs text-gray-400">
              {/* Calls columns */}
              {viewMode === 'greeks' && <th className="py-2 px-2 text-right">Δ</th>}
              <th className="py-2 px-2 text-right">Bid</th>
              <th className="py-2 px-2 text-right">Ask</th>
              <th className="py-2 px-2 text-right">Last</th>
              <th className="py-2 px-2 text-right">Vol</th>
              <th className="py-2 px-2 text-right">OI</th>
              <th className="py-2 px-2 text-right border-r border-gray-700">IV</th>
              
              {/* Strike column */}
              <th className="py-2 px-4"></th>
              
              {/* Puts columns */}
              <th className="py-2 px-2 text-left border-l border-gray-700">IV</th>
              <th className="py-2 px-2 text-left">OI</th>
              <th className="py-2 px-2 text-left">Vol</th>
              <th className="py-2 px-2 text-left">Last</th>
              <th className="py-2 px-2 text-left">Ask</th>
              <th className="py-2 px-2 text-left">Bid</th>
              {viewMode === 'greeks' && <th className="py-2 px-2 text-left">Δ</th>}
            </tr>
          </thead>
          <tbody>
            {filteredChain.map((contract) => {
              const callUnusual = isUnusualActivity(contract.call.volume, contract.call.openInterest);
              const putUnusual = isUnusualActivity(contract.put.volume, contract.put.openInterest);
              
              return (
                <tr 
                  key={`${contract.strike}-${contract.expiration}`}
                  className={`border-b border-gray-800 hover:bg-gray-800/30 transition-colors ${getStrikeStyle(contract.strike)}`}
                >
                  {/* Call side */}
                  {viewMode === 'greeks' && (
                    <td className="py-2 px-2 text-right text-sm text-gray-300">
                      {contract.call.delta.toFixed(2)}
                    </td>
                  )}
                  <td className="py-2 px-2 text-right text-sm">
                    <button
                      onClick={() => onSelectOption?.(contract, 'call')}
                      className="text-green-400 hover:text-green-300 font-medium"
                    >
                      {contract.call.bid.toFixed(2)}
                    </button>
                  </td>
                  <td className="py-2 px-2 text-right text-sm">
                    <button
                      onClick={() => onSelectOption?.(contract, 'call')}
                      className="text-green-400 hover:text-green-300 font-medium"
                    >
                      {contract.call.ask.toFixed(2)}
                    </button>
                  </td>
                  <td className="py-2 px-2 text-right text-sm text-gray-300">
                    {contract.call.last.toFixed(2)}
                  </td>
                  <td className={`py-2 px-2 text-right text-sm ${callUnusual ? 'text-yellow-400 font-bold' : 'text-gray-300'}`}>
                    <div className="flex items-center justify-end gap-1">
                      {callUnusual && <Activity className="w-3 h-3" />}
                      {contract.call.volume.toLocaleString()}
                    </div>
                  </td>
                  <td className="py-2 px-2 text-right text-sm text-gray-300">
                    {contract.call.openInterest.toLocaleString()}
                  </td>
                  <td className="py-2 px-2 text-right text-sm text-gray-300 border-r border-gray-700">
                    {(contract.call.impliedVolatility * 100).toFixed(1)}%
                  </td>
                  
                  {/* Strike */}
                  <td className="py-2 px-4 text-center font-bold text-white">
                    ${contract.strike}
                    {getMoneyness(contract.strike) === 'ATM' && (
                      <span className="block text-xs text-blue-400 font-normal">ATM</span>
                    )}
                  </td>
                  
                  {/* Put side */}
                  <td className="py-2 px-2 text-left text-sm text-gray-300 border-l border-gray-700">
                    {(contract.put.impliedVolatility * 100).toFixed(1)}%
                  </td>
                  <td className="py-2 px-2 text-left text-sm text-gray-300">
                    {contract.put.openInterest.toLocaleString()}
                  </td>
                  <td className={`py-2 px-2 text-left text-sm ${putUnusual ? 'text-yellow-400 font-bold' : 'text-gray-300'}`}>
                    <div className="flex items-center gap-1">
                      {contract.put.volume.toLocaleString()}
                      {putUnusual && <Activity className="w-3 h-3" />}
                    </div>
                  </td>
                  <td className="py-2 px-2 text-left text-sm text-gray-300">
                    {contract.put.last.toFixed(2)}
                  </td>
                  <td className="py-2 px-2 text-left text-sm">
                    <button
                      onClick={() => onSelectOption?.(contract, 'put')}
                      className="text-red-400 hover:text-red-300 font-medium"
                    >
                      {contract.put.ask.toFixed(2)}
                    </button>
                  </td>
                  <td className="py-2 px-2 text-left text-sm">
                    <button
                      onClick={() => onSelectOption?.(contract, 'put')}
                      className="text-red-400 hover:text-red-300 font-medium"
                    >
                      {contract.put.bid.toFixed(2)}
                    </button>
                  </td>
                  {viewMode === 'greeks' && (
                    <td className="py-2 px-2 text-left text-sm text-gray-300">
                      {contract.put.delta.toFixed(2)}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div className="p-4 border-t border-gray-800 flex items-center gap-6 text-xs text-gray-400">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-yellow-400" />
          <span>Unusual Activity (Vol {'>'} 50% OI)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-500/20 rounded"></div>
          <span>In The Money (ITM)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-blue-500/20 rounded border border-blue-500"></div>
          <span>At The Money (ATM)</span>
        </div>
        <div className="flex items-center gap-4 ml-auto">
          <span>Click bid/ask to trade</span>
          <TrendingUp className="w-4 h-4 text-green-400" />
        </div>
      </div>
    </div>
  );
}; 