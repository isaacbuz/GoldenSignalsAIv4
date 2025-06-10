"""
SignalSummarizerAgent

Uses LLM (OpenAI GPT-4 or Grok) to summarize agent outputs and meta signals in plain English for user guidance.
"""
import os
from openai import OpenAI
from agents.grok_agents import GrokStrategyAgent

class SignalSummarizerAgent:
    def __init__(self, api_key=None, model="gpt-4", use_grok=False, grok_api_key=None):
        self.model = model
        self.use_grok = use_grok
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY")) if not use_grok else None
        self.grok_agent = GrokStrategyAgent(grok_api_key or os.getenv("GROK_API_KEY")) if use_grok else None

    def summarize(self, symbol, agent_outputs, meta_signal):
        prompt = f"""
You are a trading assistant. Summarize the reasoning behind a meta signal generated for the stock symbol {symbol}.
The signal is: {meta_signal.get('signal', '').upper()} with confidence {meta_signal.get('confidence', 0):.2f}.

Here are the individual agent outputs:
{agent_outputs}

Respond with a clear, plain-English explanation of what caused this signal, and what traders should understand.
"""
        if self.use_grok and self.grok_agent:
            # Use Grok for summarization
            return self.grok_agent.generate_logic(symbol, timeframe="1h")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial AI summarizer."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM Error] {str(e)}"
