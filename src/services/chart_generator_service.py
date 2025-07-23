"""
Chart Generator Service for AI Trading Analyst
Creates TradingView-style charts with various analysis overlays
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class ChartGeneratorService:
    """
    Service for generating interactive trading charts
    """
    
    def __init__(self):
        self.chart_configs = {}
        self.color_scheme = {
            'background': '#0e1117',
            'grid': '#1f2937',
            'text': '#e5e7eb',
            'bullish': '#10b981',
            'bearish': '#ef4444',
            'neutral': '#6b7280',
            'primary': '#3b82f6',
            'secondary': '#8b5cf6',
            'warning': '#f59e0b'
        }
        
    async def create_technical_chart(self, 
                                   symbol: str,
                                   timeframe: str,
                                   indicators: Dict[str, Any],
                                   patterns: List[Dict[str, Any]] = None,
                                   annotations: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a comprehensive technical analysis chart
        """
        chart_config = {
            'type': 'technical_analysis',
            'symbol': symbol,
            'timeframe': timeframe,
            'layout': {
                'backgroundColor': self.color_scheme['background'],
                'textColor': self.color_scheme['text'],
                'fontSize': 12,
                'fontFamily': 'Inter, system-ui, sans-serif'
            },
            'grid': {
                'vertLines': {'color': self.color_scheme['grid'], 'style': 'dashed'},
                'horzLines': {'color': self.color_scheme['grid'], 'style': 'dashed'}
            },
            'series': []
        }
        
        # Main price series (candlestick)
        chart_config['series'].append({
            'type': 'candlestick',
            'id': 'price',
            'name': f'{symbol} Price',
            'data': [],  # Will be populated with actual data
            'upColor': self.color_scheme['bullish'],
            'downColor': self.color_scheme['bearish'],
            'borderVisible': False,
            'wickUpColor': self.color_scheme['bullish'],
            'wickDownColor': self.color_scheme['bearish']
        })
        
        # Add technical indicators
        if indicators:
            chart_config['indicators'] = self._add_indicators(indicators)
        
        # Add pattern overlays
        if patterns:
            chart_config['patterns'] = self._add_pattern_overlays(patterns)
        
        # Add annotations
        if annotations:
            chart_config['annotations'] = annotations
        
        # Add drawing tools configuration
        chart_config['tools'] = {
            'enabled': True,
            'tools': ['trend_line', 'horizontal_line', 'fibonacci', 'rectangle', 'text']
        }
        
        return chart_config
    
    async def create_multi_timeframe_chart(self,
                                         symbol: str,
                                         timeframes: List[str],
                                         sync_crosshair: bool = True) -> Dict[str, Any]:
        """
        Create synchronized multi-timeframe analysis charts
        """
        chart_config = {
            'type': 'multi_timeframe',
            'symbol': symbol,
            'layout': 'grid',
            'sync_crosshair': sync_crosshair,
            'charts': []
        }
        
        # Create a sub-chart for each timeframe
        for i, tf in enumerate(timeframes):
            sub_chart = {
                'id': f'chart_{tf}',
                'timeframe': tf,
                'height_ratio': 1 / len(timeframes),
                'series': [{
                    'type': 'candlestick',
                    'name': f'{symbol} {tf}',
                    'upColor': self.color_scheme['bullish'],
                    'downColor': self.color_scheme['bearish']
                }],
                'indicators': {
                    'volume': {'position': 'bottom', 'height': 0.2},
                    'rsi': {'position': 'bottom', 'height': 0.15}
                }
            }
            chart_config['charts'].append(sub_chart)
        
        return chart_config
    
    async def create_volume_profile_chart(self,
                                        symbol: str,
                                        timeframe: str,
                                        show_poc: bool = True,
                                        show_value_areas: bool = True) -> Dict[str, Any]:
        """
        Create volume profile chart with market structure analysis
        """
        chart_config = {
            'type': 'volume_profile',
            'symbol': symbol,
            'timeframe': timeframe,
            'layout': {
                'backgroundColor': self.color_scheme['background'],
                'textColor': self.color_scheme['text']
            },
            'series': [{
                'type': 'candlestick',
                'id': 'price',
                'name': f'{symbol} Price'
            }],
            'volume_profile': {
                'enabled': True,
                'position': 'right',
                'width': 0.15,
                'opacity': 0.5,
                'upColor': self.color_scheme['bullish'] + '80',
                'downColor': self.color_scheme['bearish'] + '80',
                'poc': {
                    'visible': show_poc,
                    'color': self.color_scheme['warning'],
                    'width': 2,
                    'style': 'solid'
                },
                'value_areas': {
                    'visible': show_value_areas,
                    'vah_color': self.color_scheme['primary'],
                    'val_color': self.color_scheme['secondary'],
                    'opacity': 0.2
                }
            }
        }
        
        return chart_config
    
    async def create_options_flow_chart(self,
                                      symbol: str,
                                      timeframe: str = '5m') -> Dict[str, Any]:
        """
        Create options flow visualization chart
        """
        chart_config = {
            'type': 'options_flow',
            'symbol': symbol,
            'timeframe': timeframe,
            'layout': {
                'backgroundColor': self.color_scheme['background']
            },
            'series': [
                {
                    'type': 'candlestick',
                    'id': 'price',
                    'name': f'{symbol} Price',
                    'pane': 0
                },
                {
                    'type': 'histogram',
                    'id': 'call_volume',
                    'name': 'Call Volume',
                    'color': self.color_scheme['bullish'],
                    'pane': 1
                },
                {
                    'type': 'histogram',
                    'id': 'put_volume',
                    'name': 'Put Volume',
                    'color': self.color_scheme['bearish'],
                    'pane': 1
                },
                {
                    'type': 'line',
                    'id': 'put_call_ratio',
                    'name': 'Put/Call Ratio',
                    'color': self.color_scheme['primary'],
                    'pane': 2
                }
            ],
            'panes': [
                {'height': 0.6},  # Price
                {'height': 0.25}, # Volume
                {'height': 0.15}  # Ratio
            ]
        }
        
        return chart_config
    
    async def create_correlation_heatmap(self,
                                       symbols: List[str],
                                       timeframe: str = '1d',
                                       period: int = 30) -> Dict[str, Any]:
        """
        Create correlation heatmap for multiple symbols
        """
        chart_config = {
            'type': 'correlation_heatmap',
            'symbols': symbols,
            'timeframe': timeframe,
            'period': period,
            'layout': {
                'backgroundColor': self.color_scheme['background']
            },
            'heatmap': {
                'colorScale': {
                    '-1': self.color_scheme['bearish'],
                    '0': self.color_scheme['neutral'],
                    '1': self.color_scheme['bullish']
                },
                'showValues': True,
                'cellBorder': True
            }
        }
        
        return chart_config
    
    def _add_indicators(self, indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Add technical indicators to chart configuration
        """
        indicator_configs = []
        
        # Moving Averages
        if 'sma' in indicators:
            for period in indicators['sma']:
                indicator_configs.append({
                    'type': 'sma',
                    'period': period,
                    'color': self._get_ma_color(period),
                    'lineWidth': 2,
                    'name': f'SMA {period}'
                })
        
        if 'ema' in indicators:
            for period in indicators['ema']:
                indicator_configs.append({
                    'type': 'ema',
                    'period': period,
                    'color': self._get_ma_color(period),
                    'lineWidth': 2,
                    'lineStyle': 'dashed',
                    'name': f'EMA {period}'
                })
        
        # Bollinger Bands
        if 'bollinger_bands' in indicators:
            bb = indicators['bollinger_bands']
            indicator_configs.append({
                'type': 'bollinger_bands',
                'period': bb.get('period', 20),
                'stdDev': bb.get('std_dev', 2),
                'upperColor': self.color_scheme['primary'] + '40',
                'lowerColor': self.color_scheme['primary'] + '40',
                'middleColor': self.color_scheme['primary'],
                'fillColor': self.color_scheme['primary'] + '10'
            })
        
        # RSI (separate pane)
        if 'rsi' in indicators:
            indicator_configs.append({
                'type': 'rsi',
                'period': indicators['rsi'].get('period', 14),
                'overbought': 70,
                'oversold': 30,
                'color': self.color_scheme['secondary'],
                'pane': 'rsi_pane'
            })
        
        # MACD (separate pane)
        if 'macd' in indicators:
            macd = indicators['macd']
            indicator_configs.append({
                'type': 'macd',
                'fastPeriod': macd.get('fast', 12),
                'slowPeriod': macd.get('slow', 26),
                'signalPeriod': macd.get('signal', 9),
                'macdColor': self.color_scheme['primary'],
                'signalColor': self.color_scheme['secondary'],
                'histogramColor': self.color_scheme['neutral'],
                'pane': 'macd_pane'
            })
        
        return indicator_configs
    
    def _add_pattern_overlays(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add pattern overlays to chart
        """
        pattern_overlays = []
        
        for pattern in patterns:
            overlay = {
                'type': pattern['type'],
                'name': pattern['name'],
                'points': pattern['points'],
                'style': {
                    'color': self.color_scheme['primary'] if pattern.get('bullish') else self.color_scheme['bearish'],
                    'lineWidth': 2,
                    'lineStyle': 'solid',
                    'fillOpacity': 0.1
                },
                'label': {
                    'text': f"{pattern['name']} ({pattern['confidence']:.0%})",
                    'position': 'top',
                    'backgroundColor': self.color_scheme['background'],
                    'borderColor': pattern.get('color', self.color_scheme['primary'])
                }
            }
            pattern_overlays.append(overlay)
        
        return pattern_overlays
    
    def _get_ma_color(self, period: int) -> str:
        """
        Get color for moving average based on period
        """
        if period <= 20:
            return self.color_scheme['bullish']
        elif period <= 50:
            return self.color_scheme['primary']
        elif period <= 100:
            return self.color_scheme['secondary']
        else:
            return self.color_scheme['warning']
    
    async def create_market_structure_chart(self,
                                          symbol: str,
                                          timeframe: str) -> Dict[str, Any]:
        """
        Create market structure analysis chart
        """
        chart_config = {
            'type': 'market_structure',
            'symbol': symbol,
            'timeframe': timeframe,
            'series': [{
                'type': 'candlestick',
                'id': 'price',
                'name': f'{symbol} Price'
            }],
            'overlays': [
                {
                    'type': 'swing_highs_lows',
                    'showLabels': True,
                    'highColor': self.color_scheme['bullish'],
                    'lowColor': self.color_scheme['bearish']
                },
                {
                    'type': 'trend_lines',
                    'autoDetect': True,
                    'color': self.color_scheme['primary']
                },
                {
                    'type': 'support_resistance',
                    'levels': [],  # Will be populated
                    'supportColor': self.color_scheme['bullish'],
                    'resistanceColor': self.color_scheme['bearish']
                }
            ]
        }
        
        return chart_config
    
    def generate_chart_annotations(self,
                                 analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate chart annotations from analysis results
        """
        annotations = []
        
        # Add entry/exit annotations
        if 'entry_points' in analysis_results:
            for entry in analysis_results['entry_points']:
                annotations.append({
                    'type': 'marker',
                    'time': entry['time'],
                    'position': 'belowBar',
                    'color': self.color_scheme['bullish'],
                    'shape': 'arrowUp',
                    'text': f"Entry: {entry['price']}"
                })
        
        if 'exit_points' in analysis_results:
            for exit in analysis_results['exit_points']:
                annotations.append({
                    'type': 'marker',
                    'time': exit['time'],
                    'position': 'aboveBar',
                    'color': self.color_scheme['bearish'],
                    'shape': 'arrowDown',
                    'text': f"Exit: {exit['price']}"
                })
        
        # Add key level annotations
        if 'key_levels' in analysis_results:
            for level in analysis_results['key_levels']:
                annotations.append({
                    'type': 'horizontal_line',
                    'price': level['price'],
                    'color': level.get('color', self.color_scheme['primary']),
                    'lineStyle': 'dashed',
                    'lineWidth': 1,
                    'text': level['label']
                })
        
        return annotations


# Example usage
async def test_chart_generator():
    generator = ChartGeneratorService()
    
    # Create technical analysis chart
    tech_chart = await generator.create_technical_chart(
        symbol='AAPL',
        timeframe='1h',
        indicators={
            'sma': [20, 50],
            'ema': [9, 21],
            'rsi': {'period': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger_bands': {'period': 20, 'std_dev': 2}
        },
        patterns=[{
            'type': 'triangle',
            'name': 'Ascending Triangle',
            'confidence': 0.85,
            'bullish': True,
            'points': []  # Would contain actual coordinates
        }]
    )
    
    print("Technical Chart Config:", json.dumps(tech_chart, indent=2))
    
    # Create multi-timeframe chart
    mtf_chart = await generator.create_multi_timeframe_chart(
        symbol='SPY',
        timeframes=['5m', '1h', '1d'],
        sync_crosshair=True
    )
    
    print("\nMulti-Timeframe Chart Config:", json.dumps(mtf_chart, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_chart_generator()) 