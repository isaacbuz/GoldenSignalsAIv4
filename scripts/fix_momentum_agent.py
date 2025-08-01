#!/usr/bin/env python3
"""
Fix momentum agent to work with BaseAgent
"""

import re

def fix_momentum_agent():
    """Fix momentum agent to inherit from BaseAgent properly"""

    filepath = 'agents/technical/momentum_agent.py'

    with open(filepath, 'r') as f:
        content = f.read()

    # Remove the _register_capabilities method
    content = re.sub(
        r'    def _register_capabilities\(self\):.*?(?=\n    def|\n    async def|\Z)',
        '',
        content,
        flags=re.DOTALL
    )

    # Remove the _register_message_handlers method
    content = re.sub(
        r'    def _register_message_handlers\(self\):.*?(?=\n    def|\n    async def|\Z)',
        '',
        content,
        flags=re.DOTALL
    )

    # Replace AgentMessage with Dict[str, Any] in all async methods
    content = re.sub(r'message: AgentMessage', 'message: Dict[str, Any]', content)

    # Fix message.payload references
    content = content.replace('message.payload.get(', 'message.get(')

    # Add missing abstract methods from BaseAgent
    if 'def analyze(' not in content:
        # Find a good place to insert the analyze method
        insert_pos = content.find('    def _calculate_momentum_indicators')
        if insert_pos > 0:
            analyze_method = '''    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum for the given data"""
        symbol = data.get('symbol', 'UNKNOWN')
        market_data = data.get('data')

        if not isinstance(market_data, pd.DataFrame):
            return {'error': 'Invalid data format'}

        try:
            indicators = self._calculate_momentum_indicators(market_data)
            strength = self._calculate_momentum_strength(indicators)
            signals = self._generate_momentum_signals(symbol, market_data, indicators, strength)

            return {
                'symbol': symbol,
                'indicators': indicators,
                'strength': strength,
                'signals': signals,
                'interpretation': self._interpret_momentum(indicators, strength)
            }
        except Exception as e:
            return {'error': str(e)}

    def get_required_data_types(self) -> List[str]:
        """Return required data types for momentum analysis"""
        return ['price', 'volume']

    '''
            content = content[:insert_pos] + analyze_method + content[insert_pos:]

    # Write back the fixed content
    with open(filepath, 'w') as f:
        f.write(content)

    print(f"Fixed {filepath}")

if __name__ == "__main__":
    fix_momentum_agent()
