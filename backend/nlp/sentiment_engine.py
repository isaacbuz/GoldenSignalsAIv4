import random

def analyze_sentiment(headlines: list) -> tuple:
    """Mock: Returns a summary and a random sentiment score between -1 and 1."""
    summary = " ".join(headlines[:2])
    score = round(random.uniform(-1, 1), 2)
    return summary, score

