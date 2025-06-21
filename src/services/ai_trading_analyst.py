"""
AI Trading Analyst Service for GoldenSignalsAI
A sophisticated AI analyst that provides comprehensive market analysis
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass
from enum import Enum

from src.agents.core.agent_factory import get_agent_factory
from src.agents.orchestration.meta_orchestrator import MetaOrchestrator, MetaStrategy
from src.services.chart_generator_service import ChartGeneratorService
from src.services.nlp_service import NLPService


class AnalysisIntent(Enum):
    """Types of analysis intents"""
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    PRICE_PREDICTION = "price_prediction"
    PATTERN_RECOGNITION = "pattern_recognition"
    RISK_ASSESSMENT = "risk_assessment"
    COMPARISON = "comparison"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    MARKET_OVERVIEW = "market_overview"
    OPTIONS_ANALYSIS = "options_analysis"


@dataclass
class AnalystResponse:
    """Structured response from the AI analyst"""
    text_analysis: str
    charts: List[Dict[str, Any]]
    data_tables: List[Dict[str, Any]]
    key_insights: List[str]
    confidence_score: float
    recommendations: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    follow_up_questions: List[str]


class AITradingAnalyst:
    """
    AI Trading Analyst that acts as a professional market analyst
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agent_factory = get_agent_factory()
        self.chart_generator = ChartGeneratorService()
        self.nlp_service = NLPService()
        
        # Initialize specialized components
        self.technical_analyzer = TechnicalAnalyzer(self.agent_factory)
        self.sentiment_analyzer = SentimentAnalyzer(self.agent_factory)
        self.pattern_recognizer = PatternRecognizer(self.agent_factory)
        self.risk_analyzer = RiskAnalyzer(self.agent_factory)
        self.prediction_engine = PredictionEngine(self.agent_factory)
        
        # Analysis context
        self.conversation_context: Dict[str, Any] = {}
        self.analysis_cache: Dict[str, Any] = {}
        
    async def analyze_query(self, query: str, context: Dict[str, Any] = None) -> AnalystResponse:
        """
        Main entry point for AI analyst queries
        """
        # Update context
        if context:
            self.conversation_context.update(context)
        
        # Parse query intent and entities
        intent, entities = await self.nlp_service.parse_query(query)
        
        # Route to appropriate analysis method
        if intent == AnalysisIntent.TECHNICAL_ANALYSIS:
            return await self._perform_technical_analysis(entities)
        elif intent == AnalysisIntent.SENTIMENT_ANALYSIS:
            return await self._perform_sentiment_analysis(entities)
        elif intent == AnalysisIntent.PATTERN_RECOGNITION:
            return await self._perform_pattern_recognition(entities)
        elif intent == AnalysisIntent.PRICE_PREDICTION:
            return await self._perform_price_prediction(entities)
        elif intent == AnalysisIntent.RISK_ASSESSMENT:
            return await self._perform_risk_assessment(entities)
        elif intent == AnalysisIntent.COMPARISON:
            return await self._perform_comparison_analysis(entities)
        else:
            return await self._perform_comprehensive_analysis(entities)
    
    async def _perform_comprehensive_analysis(self, entities: Dict[str, Any]) -> AnalystResponse:
        """
        Perform comprehensive analysis covering all aspects
        """
        symbol = entities.get('symbol', 'SPY')
        timeframe = entities.get('timeframe', '1d')
        
        # Run parallel analysis across all domains
        results = await asyncio.gather(
            self.technical_analyzer.analyze(symbol, timeframe),
            self.sentiment_analyzer.analyze(symbol),
            self.pattern_recognizer.detect_patterns(symbol, timeframe),
            self.risk_analyzer.assess_risk(symbol),
            self.prediction_engine.predict(symbol, timeframe),
            return_exceptions=True
        )
        
        # Unpack results
        technical_result = results[0] if not isinstance(results[0], Exception) else {}
        sentiment_result = results[1] if not isinstance(results[1], Exception) else {}
        pattern_result = results[2] if not isinstance(results[2], Exception) else {}
        risk_result = results[3] if not isinstance(results[3], Exception) else {}
        prediction_result = results[4] if not isinstance(results[4], Exception) else {}
        
        # Generate comprehensive analysis text
        analysis_text = self._generate_comprehensive_analysis_text(
            symbol, technical_result, sentiment_result, 
            pattern_result, risk_result, prediction_result
        )
        
        # Generate charts
        charts = await self._generate_comprehensive_charts(
            symbol, timeframe, technical_result, pattern_result
        )
        
        # Extract key insights
        key_insights = self._extract_key_insights(
            technical_result, sentiment_result, pattern_result, 
            risk_result, prediction_result
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            technical_result, sentiment_result, risk_result, prediction_result
        )
        
        # Calculate confidence score
        confidence = self._calculate_analysis_confidence(results)
        
        return AnalystResponse(
            text_analysis=analysis_text,
            charts=charts,
            data_tables=self._create_data_tables(results),
            key_insights=key_insights,
            confidence_score=confidence,
            recommendations=recommendations,
            alerts=self._check_for_alerts(results),
            follow_up_questions=self._suggest_follow_up_questions(symbol)
        )
    
    def _generate_comprehensive_analysis_text(self, symbol: str, 
                                            technical: Dict, sentiment: Dict,
                                            patterns: Dict, risk: Dict, 
                                            prediction: Dict) -> str:
        """
        Generate professional analyst commentary
        """
        analysis = f"""
## Comprehensive Analysis for {symbol}

### Executive Summary
Based on my analysis, {symbol} is currently exhibiting {self._describe_market_condition(technical)}. 
The convergence of technical indicators, market sentiment, and pattern recognition suggests 
{self._describe_outlook(technical, sentiment, prediction)}.

### Technical Analysis
{self._format_technical_analysis(technical)}

### Market Sentiment
{self._format_sentiment_analysis(sentiment)}

### Pattern Recognition
{self._format_pattern_analysis(patterns)}

### Risk Assessment
{self._format_risk_analysis(risk)}

### Price Projections
{self._format_predictions(prediction)}

### Trading Strategy
{self._suggest_trading_strategy(technical, sentiment, risk)}

*Analysis generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with {self._describe_confidence()} confidence*
"""
        return analysis
    
    async def _generate_comprehensive_charts(self, symbol: str, timeframe: str,
                                           technical: Dict, patterns: Dict) -> List[Dict]:
        """
        Generate multiple charts with different perspectives
        """
        charts = []
        
        # Main price chart with technical indicators
        main_chart = await self.chart_generator.create_technical_chart(
            symbol=symbol,
            timeframe=timeframe,
            indicators=technical.get('indicators', {}),
            patterns=patterns.get('patterns', []),
            annotations=self._create_chart_annotations(technical, patterns)
        )
        charts.append({
            'type': 'main_analysis',
            'title': f'{symbol} Technical Analysis',
            'config': main_chart
        })
        
        # Multi-timeframe analysis chart
        mtf_chart = await self.chart_generator.create_multi_timeframe_chart(
            symbol=symbol,
            timeframes=['5m', '1h', '1d'],
            sync_crosshair=True
        )
        charts.append({
            'type': 'multi_timeframe',
            'title': 'Multi-Timeframe Analysis',
            'config': mtf_chart
        })
        
        # Volume profile chart
        volume_chart = await self.chart_generator.create_volume_profile_chart(
            symbol=symbol,
            timeframe=timeframe,
            show_poc=True,  # Point of Control
            show_value_areas=True
        )
        charts.append({
            'type': 'volume_profile',
            'title': 'Volume Profile Analysis',
            'config': volume_chart
        })
        
        return charts
    
    def _extract_key_insights(self, *results) -> List[str]:
        """
        Extract the most important insights from all analyses
        """
        insights = []
        
        # Technical insights
        technical = results[0]
        if technical.get('momentum_strength', 0) > 0.7:
            insights.append("🚀 Strong bullish momentum detected with multiple indicator confirmation")
        elif technical.get('momentum_strength', 0) < -0.7:
            insights.append("⚠️ Strong bearish momentum with increasing selling pressure")
        
        # Pattern insights
        patterns = results[2]
        if patterns.get('patterns'):
            for pattern in patterns['patterns'][:2]:  # Top 2 patterns
                insights.append(f"📊 {pattern['name']} pattern identified with {pattern['confidence']:.0%} confidence")
        
        # Risk insights
        risk = results[3]
        if risk.get('risk_score', 0) > 0.7:
            insights.append("🔴 High risk environment - consider reducing position sizes")
        
        # Prediction insights
        prediction = results[4]
        if prediction.get('prediction'):
            direction = "📈" if prediction['direction'] == 'up' else "📉"
            insights.append(f"{direction} AI models predict {prediction['magnitude']:.1%} move in next {prediction['timeframe']}")
        
        return insights
    
    def _generate_recommendations(self, technical: Dict, sentiment: Dict, 
                                risk: Dict, prediction: Dict) -> List[Dict]:
        """
        Generate actionable trading recommendations
        """
        recommendations = []
        
        # Determine overall bias
        technical_bias = technical.get('overall_bias', 0)
        sentiment_bias = sentiment.get('overall_sentiment', 0)
        risk_score = risk.get('risk_score', 0.5)
        
        # Entry recommendations
        if technical_bias > 0.5 and sentiment_bias > 0.3 and risk_score < 0.6:
            recommendations.append({
                'type': 'entry',
                'action': 'BUY',
                'confidence': 0.8,
                'rationale': 'Technical and sentiment alignment with acceptable risk',
                'entry_zone': technical.get('support_levels', [])[-1] if technical.get('support_levels') else None,
                'stop_loss': technical.get('stop_loss_level'),
                'take_profit': technical.get('resistance_levels', [])[0] if technical.get('resistance_levels') else None
            })
        
        # Risk management recommendations
        if risk_score > 0.7:
            recommendations.append({
                'type': 'risk_management',
                'action': 'REDUCE_POSITION',
                'confidence': 0.9,
                'rationale': 'Elevated risk levels detected',
                'suggested_reduction': 0.5
            })
        
        # Options strategy recommendations
        if technical.get('iv_rank', 0) > 0.7:
            recommendations.append({
                'type': 'options_strategy',
                'action': 'SELL_PREMIUM',
                'confidence': 0.7,
                'rationale': 'High implied volatility presents premium selling opportunity',
                'suggested_strategy': 'Iron Condor or Credit Spreads'
            })
        
        return recommendations
    
    def _calculate_analysis_confidence(self, results: List[Any]) -> float:
        """
        Calculate overall confidence in the analysis
        """
        confidence_factors = []
        
        # Data quality confidence
        for result in results:
            if isinstance(result, dict) and 'data_quality' in result:
                confidence_factors.append(result['data_quality'])
        
        # Indicator agreement confidence
        technical = results[0] if isinstance(results[0], dict) else {}
        if technical.get('indicator_agreement'):
            confidence_factors.append(technical['indicator_agreement'])
        
        # Model confidence
        prediction = results[4] if isinstance(results[4], dict) else {}
        if prediction.get('model_confidence'):
            confidence_factors.append(prediction['model_confidence'])
        
        # Calculate weighted average
        if confidence_factors:
            return np.mean(confidence_factors)
        return 0.5  # Default medium confidence


class TechnicalAnalyzer:
    """Specialized technical analysis component"""
    
    def __init__(self, agent_factory):
        self.agent_factory = agent_factory
        self.indicators = {}
        
    async def analyze(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Perform comprehensive technical analysis"""
        # Create technical analysis agents
        agents = {
            'momentum': self.agent_factory.create_agent('momentum_agent'),
            'mean_reversion': self.agent_factory.create_agent('mean_reversion_agent'),
            'pattern': self.agent_factory.create_agent('pattern_recognition_agent')
        }
        
        # Get market data
        data = await self._fetch_market_data(symbol, timeframe)
        
        # Run parallel analysis
        results = await asyncio.gather(
            agents['momentum'].process_request({
                'type': 'analyze',
                'symbol': symbol,
                'data': data
            }),
            agents['mean_reversion'].process_request({
                'type': 'analyze',
                'symbol': symbol,
                'data': data
            }),
            agents['pattern'].process_request({
                'type': 'detect_patterns',
                'symbol': symbol,
                'data': data
            })
        )
        
        # Aggregate results
        return self._aggregate_technical_results(results)
    
    async def _fetch_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch market data for analysis"""
        # Implementation would fetch real market data
        # For now, return mock data structure
        return pd.DataFrame()
    
    def _aggregate_technical_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from multiple technical agents"""
        aggregated = {
            'momentum_strength': results[0].get('strength', 0),
            'trend_direction': results[0].get('trend', 'neutral'),
            'support_levels': results[1].get('support_levels', []),
            'resistance_levels': results[1].get('resistance_levels', []),
            'patterns': results[2].get('patterns', []),
            'indicators': {
                'rsi': results[0].get('indicators', {}).get('rsi'),
                'macd': results[0].get('indicators', {}).get('macd'),
                'bollinger_bands': results[1].get('indicators', {}).get('bb'),
            },
            'overall_bias': self._calculate_technical_bias(results)
        }
        return aggregated
    
    def _calculate_technical_bias(self, results: List[Dict]) -> float:
        """Calculate overall technical bias (-1 to 1)"""
        biases = []
        
        # Momentum bias
        momentum = results[0].get('strength', 0)
        biases.append(momentum)
        
        # Mean reversion bias
        mr_signal = results[1].get('signal', 0)
        biases.append(mr_signal)
        
        # Pattern bias
        pattern_bias = 0
        for pattern in results[2].get('patterns', []):
            if pattern.get('bullish'):
                pattern_bias += 0.5
            else:
                pattern_bias -= 0.5
        biases.append(np.clip(pattern_bias, -1, 1))
        
        return np.mean(biases)


class SentimentAnalyzer:
    """Specialized sentiment analysis component"""
    
    def __init__(self, agent_factory):
        self.agent_factory = agent_factory
        
    async def analyze(self, symbol: str) -> Dict[str, Any]:
        """Perform comprehensive sentiment analysis"""
        agents = {
            'news': self.agent_factory.create_agent('news_sentiment_agent'),
            'social': self.agent_factory.create_agent('social_sentiment_agent')
        }
        
        # Run parallel sentiment analysis
        results = await asyncio.gather(
            agents['news'].process_request({
                'type': 'analyze_sentiment',
                'symbol': symbol
            }),
            agents['social'].process_request({
                'type': 'analyze_sentiment',
                'symbol': symbol
            })
        )
        
        return self._aggregate_sentiment_results(results)
    
    def _aggregate_sentiment_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate sentiment from multiple sources"""
        return {
            'news_sentiment': results[0].get('sentiment_score', 0),
            'social_sentiment': results[1].get('sentiment_score', 0),
            'overall_sentiment': np.mean([
                results[0].get('sentiment_score', 0),
                results[1].get('sentiment_score', 0)
            ]),
            'sentiment_momentum': results[1].get('momentum', 0),
            'key_topics': results[0].get('topics', []) + results[1].get('topics', [])
        }


class PatternRecognizer:
    """Specialized pattern recognition component"""
    
    def __init__(self, agent_factory):
        self.agent_factory = agent_factory
        
    async def detect_patterns(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Detect chart patterns and formations"""
        agent = self.agent_factory.create_agent('pattern_recognition_agent')
        
        result = await agent.process_request({
            'type': 'detect_all_patterns',
            'symbol': symbol,
            'timeframe': timeframe
        })
        
        return {
            'patterns': result.get('patterns', []),
            'formations': result.get('formations', []),
            'key_levels': result.get('key_levels', {})
        }


class RiskAnalyzer:
    """Specialized risk analysis component"""
    
    def __init__(self, agent_factory):
        self.agent_factory = agent_factory
        
    async def assess_risk(self, symbol: str) -> Dict[str, Any]:
        """Assess various risk factors"""
        agent = self.agent_factory.create_agent('portfolio_risk_agent')
        
        result = await agent.process_request({
            'type': 'assess_risk',
            'symbol': symbol
        })
        
        return {
            'risk_score': result.get('overall_risk', 0.5),
            'volatility': result.get('volatility', 0),
            'var_95': result.get('var_95', 0),
            'max_drawdown': result.get('max_drawdown', 0),
            'correlation_risk': result.get('correlation_risk', 0)
        }


class PredictionEngine:
    """ML-based prediction component"""
    
    def __init__(self, agent_factory):
        self.agent_factory = agent_factory
        
    async def predict(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Generate price predictions using ML models"""
        agent = self.agent_factory.create_agent('ml_predictor_agent')
        
        result = await agent.process_request({
            'type': 'predict',
            'symbol': symbol,
            'timeframe': timeframe
        })
        
        return {
            'prediction': result.get('prediction'),
            'direction': result.get('direction'),
            'magnitude': result.get('magnitude'),
            'timeframe': result.get('timeframe'),
            'model_confidence': result.get('confidence', 0.5),
            'probability_distribution': result.get('distribution', {})
        }


# Example usage in API endpoint
async def ai_analyst_endpoint(query: str, context: Dict[str, Any] = None):
    """API endpoint for AI analyst queries"""
    analyst = AITradingAnalyst()
    response = await analyst.analyze_query(query, context)
    
    return {
        'analysis': response.text_analysis,
        'charts': response.charts,
        'insights': response.key_insights,
        'recommendations': response.recommendations,
        'confidence': response.confidence_score,
        'data': response.data_tables,
        'alerts': response.alerts,
        'follow_up': response.follow_up_questions
    } 