"""
FeedbackAnalystAgent

Uses LLM (OpenAI GPT-4 or Grok) to analyze recent win/loss feedback and explain patterns or lessons learned for the user.
"""
import os
from openai import OpenAI
from agents.grok_agents import GrokStrategyAgent

class FeedbackAnalystAgent:
    def __init__(self, api_key=None, model="gpt-4", use_grok=False, grok_api_key=None):
        self.model = model
        self.use_grok = use_grok
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY")) if not use_grok else None
        self.grok_agent = GrokStrategyAgent(grok_api_key or os.getenv("GROK_API_KEY")) if use_grok else None

    def analyze(self, feedback_logs, agent_name=None):
        prompt = f"""
You are a trading feedback analyst. Review the following recent win/loss logs for {agent_name or 'the agent'}:
{feedback_logs}
Summarize any patterns, lessons learned, and actionable insights for the user in plain English.
"""
        if self.use_grok and self.grok_agent:
            # Use Grok for summarization
            return self.grok_agent.generate_logic(agent_name or "Unknown", timeframe="1h")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial feedback analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM Error] {str(e)}"
