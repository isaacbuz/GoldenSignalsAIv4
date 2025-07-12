#!/usr/bin/env python3
"""
Fix all agents to implement required abstract methods.
"""

import os
import re
from pathlib import Path

# Template for the methods to add
METHODS_TEMPLATE = '''
    async def analyze(self, market_data: MarketData) -> Signal:
        """
        Analyze market data and generate trading signal.
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            Signal: Trading signal with analysis
        """
        # Extract close prices from market data
        if hasattr(market_data, 'data') and not market_data.data.empty:
            close_prices = market_data.data['close'].tolist()
        else:
            # Fallback to mock data if no real data
            close_prices = [market_data.current_price] * 100
        
        # Process the data
        result = self.process({"close_prices": close_prices})
        
        # Map action to signal type
        signal_type_map = {
            "buy": SignalType.BUY,
            "sell": SignalType.SELL,
            "hold": SignalType.HOLD
        }
        
        # Determine signal strength based on confidence
        if result["confidence"] >= 0.8:
            strength = SignalStrength.STRONG
        elif result["confidence"] >= 0.6:
            strength = SignalStrength.MODERATE
        elif result["confidence"] >= 0.4:
            strength = SignalStrength.MEDIUM
        else:
            strength = SignalStrength.WEAK
        
        # Create and return signal
        signal = Signal(
            symbol=market_data.symbol,
            signal_type=signal_type_map.get(result["action"], SignalType.HOLD),
            confidence=result.get("confidence", 0.0),
            strength=strength,
            source=SignalSource.TECHNICAL_ANALYSIS,
            current_price=market_data.current_price,
            reasoning=f"{self.name} analysis: {result.get('action', 'hold')} signal with {result.get('confidence', 0.0):.2%} confidence",
            indicators=result.get("metadata", {})
        )
        
        return signal
    
    def get_required_data_types(self) -> List[str]:
        """
        Returns list of required data types for analysis.
        
        Returns:
            List of data type strings
        """
        return ["close_prices", "ohlcv"]'''

# Imports to add
IMPORTS_TO_ADD = [
    "from typing import List",
    "from src.ml.models.market_data import MarketData",
    "from src.ml.models.signals import Signal, SignalType, SignalStrength, SignalSource"
]

def needs_fixing(file_path):
    """Check if a file needs the abstract methods."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if it's an agent that inherits from BaseAgent
    if 'BaseAgent' not in content:
        return False
    
    # Check if it already has the methods
    if 'async def analyze' in content and 'def get_required_data_types' in content:
        return False
    
    # Skip test files
    if 'test_' in str(file_path) or '/tests/' in str(file_path):
        return False
    
    return True

def fix_imports(content):
    """Add necessary imports if they're missing."""
    lines = content.split('\n')
    import_section_end = 0
    
    # Find where imports end
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith(('import ', 'from ', '#', '"""')):
            import_section_end = i
            break
    
    # Check which imports are missing
    imports_to_insert = []
    for imp in IMPORTS_TO_ADD:
        if imp not in content:
            # Special handling for List import
            if "from typing import List" in imp and "from typing import" in content:
                # Find the typing import line and add List to it
                for i, line in enumerate(lines):
                    if line.startswith("from typing import"):
                        if "List" not in line:
                            lines[i] = line.rstrip() + ", List"
                        break
            else:
                imports_to_insert.append(imp)
    
    # Insert missing imports
    if imports_to_insert:
        for imp in reversed(imports_to_insert):
            lines.insert(import_section_end, imp)
    
    return '\n'.join(lines)

def fix_agent_file(file_path):
    """Fix a single agent file."""
    print(f"Fixing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add imports
    content = fix_imports(content)
    
    # Find the last method in the class
    lines = content.split('\n')
    last_method_line = -1
    indent_level = ""
    
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if line.strip().startswith('def ') and not line.strip().startswith('def __'):
            last_method_line = i
            # Get the indentation level
            indent_level = line[:len(line) - len(line.lstrip())]
            break
    
    if last_method_line == -1:
        print(f"  Could not find where to insert methods in {file_path}")
        return
    
    # Find the end of the last method
    method_end = last_method_line
    for i in range(last_method_line + 1, len(lines)):
        if lines[i].strip() and not lines[i].startswith((' ', '\t')):
            method_end = i - 1
            break
    else:
        method_end = len(lines) - 1
    
    # Insert the new methods
    methods_lines = METHODS_TEMPLATE.split('\n')
    # Adjust indentation
    adjusted_methods = []
    for line in methods_lines:
        if line.strip():
            adjusted_methods.append(indent_level + line[4:])  # Remove 4 spaces and add proper indent
        else:
            adjusted_methods.append('')
    
    # Insert methods after the last method
    for i, method_line in enumerate(reversed(adjusted_methods)):
        lines.insert(method_end + 1, method_line)
    
    # Write back
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✅ Fixed {file_path}")

def main():
    """Fix all agent files."""
    # List of agent files to fix
    agent_files = [
        # Technical agents
        "agents/core/technical/momentum/momentum_divergence_agent.py",
        "agents/core/technical/momentum/rsi_macd_agent.py",
        "agents/core/technical/bollinger_bands_agent.py",
        "agents/core/technical/breakout_agent.py",
        "agents/core/technical/ema_agent.py",
        "agents/core/technical/fibonacci_agent.py",
        "agents/core/technical/ichimoku_agent.py",
        "agents/core/technical/ma_crossover_agent.py",
        "agents/core/technical/macd_agent.py",
        "agents/core/technical/mean_reversion_agent.py",
        "agents/core/technical/parabolic_sar_agent.py",
        "agents/core/technical/pattern_agent.py",
        "agents/core/technical/stochastic_agent.py",
        "agents/core/technical/vwap_agent.py",
        # Sentiment agents
        "agents/core/sentiment/news_agent.py",
        "agents/core/sentiment/simple_sentiment_agent.py",
        # Options agents
        "agents/core/options/gamma_exposure_agent.py",
        "agents/core/options/iv_rank_agent.py",
        "agents/core/options/simple_options_flow_agent.py",
        "agents/core/options/skew_agent.py",
        "agents/core/options/volatility_agent.py",
        # Volume agents
        "agents/core/volume/volume_profile_agent.py",
        "agents/core/volume/volume_spike_agent.py",
    ]
    
    fixed_count = 0
    skipped_count = 0
    
    for agent_file in agent_files:
        if os.path.exists(agent_file):
            if needs_fixing(agent_file):
                fix_agent_file(agent_file)
                fixed_count += 1
            else:
                print(f"Skipping {agent_file} - already has required methods")
                skipped_count += 1
        else:
            print(f"File not found: {agent_file}")
    
    print(f"\n✅ Fixed {fixed_count} files")
    print(f"⏭️  Skipped {skipped_count} files (already fixed)")

if __name__ == "__main__":
    main() 