"""
Sentiment analysis agent using NLTK for basic sentiment analysis.
"""
from typing import Dict, Any, List
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from agents.base.base_agent import BaseAgent

class SentimentAgent(BaseAgent):
    def __init__(self, name: str = "Sentiment"):
        super().__init__(name=name, agent_type="sentiment")
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        self.analyzer = SentimentIntensityAnalyzer()
        
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text"""
        return self.analyzer.polarity_scores(text)
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process news/social media data and generate sentiment signals"""
        if "texts" not in data:
            raise ValueError("No text data found in input")
            
        texts = data["texts"]
        if not texts:
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": "No texts to analyze"}
            }
            
        # Analyze each text
        sentiments = [self.analyze_text(text) for text in texts]
        
        # Calculate aggregate sentiment
        compound_scores = [s['compound'] for s in sentiments]
        avg_sentiment = sum(compound_scores) / len(compound_scores)
        
        # Convert sentiment to trading signal
        if avg_sentiment > 0.2:
            action = "buy"
            confidence = min(avg_sentiment, 1.0)
        elif avg_sentiment < -0.2:
            action = "sell"
            confidence = min(abs(avg_sentiment), 1.0)
        else:
            action = "hold"
            confidence = 0.0
            
        return {
            "action": action,
            "confidence": confidence,
            "metadata": {
                "average_sentiment": avg_sentiment,
                "sentiment_distribution": {
                    "positive": len([s for s in compound_scores if s > 0.2]),
                    "negative": len([s for s in compound_scores if s < -0.2]),
                    "neutral": len([s for s in compound_scores if -0.2 <= s <= 0.2])
                },
                "analyzed_texts": len(texts)
            }
        } 