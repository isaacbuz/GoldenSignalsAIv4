#!/usr/bin/env python3
"""
🎯 GoldenSignalsAI - ML Model Blending System
Intelligently combines outputs from multiple models into actionable signals
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelOutputs:
    """Container for all model outputs"""
    signal_probabilities: Dict[str, float]  # Bear, Neutral, Bull
    direction_probability: float  # 0-1 for upward movement
    risk_score: float  # Volatility estimate
    confidence_scores: Dict[str, float]  # Per-model confidence

class MLSignalBlender:
    """Blends multiple ML model outputs into trading signals"""
    
    def __init__(self):
        self.weights = {
            'signal_classifier': 0.4,
            'direction_predictor': 0.3,
            'risk_assessor': 0.2,
            'technical_confirmation': 0.1
        }
        
        # Confidence thresholds
        self.thresholds = {
            'high_confidence': 0.7,
            'medium_confidence': 0.5,
            'signal_trigger': 0.6
        }
    
    def blend_models(self, model_outputs: ModelOutputs, technical_signals: Dict) -> Dict:
        """
        Blend all model outputs into a final signal
        
        Returns:
            Dict with signal, confidence, and reasoning
        """
        
        # Step 1: Calculate base signal from classifier
        base_signal = self._calculate_base_signal(model_outputs.signal_probabilities)
        
        # Step 2: Adjust with direction predictor
        direction_adjusted = self._apply_direction_adjustment(
            base_signal, 
            model_outputs.direction_probability
        )
        
        # Step 3: Apply risk-based position sizing
        risk_adjusted = self._apply_risk_adjustment(
            direction_adjusted,
            model_outputs.risk_score
        )
        
        # Step 4: Technical confirmation
        final_signal = self._apply_technical_confirmation(
            risk_adjusted,
            technical_signals
        )
        
        # Step 5: Calculate final confidence
        final_confidence = self._calculate_blended_confidence(
            model_outputs,
            technical_signals
        )
        
        # Step 6: Generate reasoning
        reasoning = self._generate_reasoning(
            model_outputs,
            technical_signals,
            final_signal
        )
        
        return {
            'signal': final_signal['type'],
            'confidence': final_confidence,
            'strength': final_signal['strength'],
            'risk_level': self._categorize_risk(model_outputs.risk_score),
            'model_consensus': self._check_consensus(model_outputs),
            'reasoning': reasoning,
            'detailed_scores': {
                'ml_signal': model_outputs.signal_probabilities,
                'direction': model_outputs.direction_probability,
                'risk': model_outputs.risk_score,
                'technical': technical_signals
            }
        }
    
    def _calculate_base_signal(self, probabilities: Dict[str, float]) -> Dict:
        """Calculate base signal from classifier probabilities"""
        
        # Find dominant signal
        max_prob = max(probabilities.values())
        signal_type = max(probabilities, key=probabilities.get)
        
        # Calculate signal strength
        if signal_type == 'bull':
            if probabilities['bull'] > 0.7:
                return {'type': 'BUY_CALL', 'strength': 'strong', 'base_conf': probabilities['bull']}
            elif probabilities['bull'] > 0.5:
                return {'type': 'BUY_CALL', 'strength': 'moderate', 'base_conf': probabilities['bull']}
        
        elif signal_type == 'bear':
            if probabilities['bear'] > 0.7:
                return {'type': 'BUY_PUT', 'strength': 'strong', 'base_conf': probabilities['bear']}
            elif probabilities['bear'] > 0.5:
                return {'type': 'BUY_PUT', 'strength': 'moderate', 'base_conf': probabilities['bear']}
        
        # Default to HOLD for neutral or weak signals
        return {'type': 'HOLD', 'strength': 'weak', 'base_conf': probabilities.get('neutral', 0.5)}
    
    def _apply_direction_adjustment(self, base_signal: Dict, direction_prob: float) -> Dict:
        """Adjust signal based on direction predictor"""
        
        signal = base_signal.copy()
        
        # For CALL signals, we want high direction probability
        if base_signal['type'] == 'BUY_CALL':
            if direction_prob < 0.4:  # Contradiction - model says down but signal says up
                signal['type'] = 'HOLD'
                signal['strength'] = 'weak'
                signal['contradiction'] = True
            elif direction_prob > 0.7:  # Strong confirmation
                signal['strength'] = 'strong'
                signal['direction_conf'] = direction_prob
        
        # For PUT signals, we want low direction probability
        elif base_signal['type'] == 'BUY_PUT':
            if direction_prob > 0.6:  # Contradiction - model says up but signal says down
                signal['type'] = 'HOLD'
                signal['strength'] = 'weak'
                signal['contradiction'] = True
            elif direction_prob < 0.3:  # Strong confirmation
                signal['strength'] = 'strong'
                signal['direction_conf'] = 1 - direction_prob
        
        return signal
    
    def _apply_risk_adjustment(self, signal: Dict, risk_score: float) -> Dict:
        """Adjust signal based on risk assessment"""
        
        # High risk might reduce confidence or change signal
        if risk_score > 0.8:  # Very high risk
            if signal['strength'] == 'moderate':
                signal['strength'] = 'weak'
            signal['high_risk_warning'] = True
        
        elif risk_score < 0.2:  # Very low risk - might be opportunity
            if signal['strength'] == 'moderate':
                signal['strength'] = 'strong'
            signal['low_risk_opportunity'] = True
        
        signal['risk_score'] = risk_score
        return signal
    
    def _apply_technical_confirmation(self, signal: Dict, technical: Dict) -> Dict:
        """Apply technical indicator confirmation"""
        
        confirmations = 0
        
        # Check RSI
        if 'rsi' in technical:
            if signal['type'] == 'BUY_CALL' and technical['rsi'] < 30:
                confirmations += 1
            elif signal['type'] == 'BUY_PUT' and technical['rsi'] > 70:
                confirmations += 1
        
        # Check MACD
        if 'macd_signal' in technical:
            if signal['type'] == 'BUY_CALL' and technical['macd_signal'] == 'bullish':
                confirmations += 1
            elif signal['type'] == 'BUY_PUT' and technical['macd_signal'] == 'bearish':
                confirmations += 1
        
        # Check Volume
        if 'volume_spike' in technical and technical['volume_spike']:
            confirmations += 1
        
        # Upgrade signal if strong technical confirmation
        if confirmations >= 2 and signal['strength'] == 'moderate':
            signal['strength'] = 'strong'
            signal['technical_confirmation'] = True
        
        signal['technical_confirmations'] = confirmations
        return signal
    
    def _calculate_blended_confidence(self, outputs: ModelOutputs, technical: Dict) -> float:
        """Calculate final blended confidence score"""
        
        # Start with signal classifier confidence
        base_confidence = max(outputs.signal_probabilities.values())
        
        # Weight by direction agreement
        direction_weight = outputs.direction_probability if outputs.direction_probability > 0.5 else (1 - outputs.direction_probability)
        
        # Adjust for risk
        risk_multiplier = 1.0
        if outputs.risk_score > 0.7:
            risk_multiplier = 0.8  # Reduce confidence for high risk
        elif outputs.risk_score < 0.3:
            risk_multiplier = 1.1  # Increase confidence for low risk
        
        # Technical boost
        technical_boost = 0.0
        if technical.get('confirmations', 0) >= 2:
            technical_boost = 0.1
        
        # Blend all components
        blended = (
            base_confidence * self.weights['signal_classifier'] +
            direction_weight * self.weights['direction_predictor'] +
            (1 - outputs.risk_score) * self.weights['risk_assessor'] +
            technical_boost
        )
        
        return min(blended, 0.95)  # Cap at 95%
    
    def _check_consensus(self, outputs: ModelOutputs) -> str:
        """Check if models agree or disagree"""
        
        # Get signal direction
        signal_direction = 'bull' if max(outputs.signal_probabilities, key=outputs.signal_probabilities.get) == 'bull' else 'bear'
        
        # Check direction predictor agreement
        direction_agrees = (
            (signal_direction == 'bull' and outputs.direction_probability > 0.6) or
            (signal_direction == 'bear' and outputs.direction_probability < 0.4)
        )
        
        # Check risk appropriateness
        risk_appropriate = outputs.risk_score < 0.7
        
        if direction_agrees and risk_appropriate:
            return "strong_consensus"
        elif direction_agrees or risk_appropriate:
            return "partial_consensus"
        else:
            return "no_consensus"
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk level"""
        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.7:
            return "medium"
        else:
            return "high"
    
    def _generate_reasoning(self, outputs: ModelOutputs, technical: Dict, signal: Dict) -> str:
        """Generate human-readable reasoning"""
        
        reasons = []
        
        # ML model reasoning
        dominant = max(outputs.signal_probabilities, key=outputs.signal_probabilities.get)
        reasons.append(f"ML models indicate {dominant} market ({outputs.signal_probabilities[dominant]:.0%})")
        
        # Direction reasoning
        if outputs.direction_probability > 0.6:
            reasons.append(f"Direction model predicts upward movement ({outputs.direction_probability:.0%})")
        elif outputs.direction_probability < 0.4:
            reasons.append(f"Direction model predicts downward movement ({(1-outputs.direction_probability):.0%})")
        
        # Risk reasoning
        risk_level = self._categorize_risk(outputs.risk_score)
        reasons.append(f"Risk level: {risk_level}")
        
        # Technical reasoning
        if technical.get('rsi'):
            if technical['rsi'] < 30:
                reasons.append("RSI oversold")
            elif technical['rsi'] > 70:
                reasons.append("RSI overbought")
        
        # Consensus
        consensus = self._check_consensus(outputs)
        if consensus == "strong_consensus":
            reasons.append("Strong model consensus")
        elif consensus == "no_consensus":
            reasons.append("Models show mixed signals")
        
        return " + ".join(reasons)

def demonstrate_blending():
    """Demonstrate the ML blending system"""
    
    blender = MLSignalBlender()
    
    # Example 1: Strong Bull Signal
    print("="*60)
    print("Example 1: Strong Bullish Signal")
    print("="*60)
    
    bull_outputs = ModelOutputs(
        signal_probabilities={'bear': 0.15, 'neutral': 0.25, 'bull': 0.60},
        direction_probability=0.72,
        risk_score=0.35,
        confidence_scores={'signal': 0.8, 'direction': 0.75, 'risk': 0.7}
    )
    
    bull_technical = {
        'rsi': 28,
        'macd_signal': 'bullish',
        'volume_spike': True,
        'confirmations': 3
    }
    
    bull_signal = blender.blend_models(bull_outputs, bull_technical)
    print_signal(bull_signal)
    
    # Example 2: Conflicting Signals
    print("\nExample 2: Conflicting Signals")
    print("="*60)
    
    conflict_outputs = ModelOutputs(
        signal_probabilities={'bear': 0.20, 'neutral': 0.45, 'bull': 0.35},
        direction_probability=0.65,  # Says up
        risk_score=0.75,  # High risk
        confidence_scores={'signal': 0.5, 'direction': 0.6, 'risk': 0.8}
    )
    
    conflict_technical = {
        'rsi': 72,  # Overbought
        'macd_signal': 'bearish',
        'volume_spike': False,
        'confirmations': 0
    }
    
    conflict_signal = blender.blend_models(conflict_outputs, conflict_technical)
    print_signal(conflict_signal)
    
    # Example 3: High Risk Put Signal
    print("\nExample 3: High Risk Put Signal")
    print("="*60)
    
    put_outputs = ModelOutputs(
        signal_probabilities={'bear': 0.55, 'neutral': 0.30, 'bull': 0.15},
        direction_probability=0.25,  # Predicts down
        risk_score=0.82,  # Very high risk
        confidence_scores={'signal': 0.7, 'direction': 0.8, 'risk': 0.6}
    )
    
    put_technical = {
        'rsi': 75,
        'macd_signal': 'bearish',
        'volume_spike': True,
        'confirmations': 2
    }
    
    put_signal = blender.blend_models(put_outputs, put_technical)
    print_signal(put_signal)

def print_signal(signal: Dict):
    """Pretty print the blended signal"""
    
    # Emoji based on signal
    emoji = "🟢" if signal['signal'] == "BUY_CALL" else "🔴" if signal['signal'] == "BUY_PUT" else "⚪"
    
    print(f"\n{emoji} Signal: {signal['signal']}")
    print(f"🎯 Confidence: {signal['confidence']:.1%}")
    print(f"💪 Strength: {signal['strength']}")
    print(f"⚠️  Risk Level: {signal['risk_level']}")
    print(f"🤝 Model Consensus: {signal['model_consensus']}")
    print(f"\n💡 Reasoning: {signal['reasoning']}")
    
    # Detailed scores
    print(f"\n📊 Detailed Scores:")
    scores = signal['detailed_scores']
    print(f"   ML Signal: Bull={scores['ml_signal']['bull']:.0%}, "
          f"Bear={scores['ml_signal']['bear']:.0%}, "
          f"Neutral={scores['ml_signal']['neutral']:.0%}")
    print(f"   Direction: {scores['direction']:.0%} probability of upward movement")
    print(f"   Risk Score: {scores['risk']:.2f}")
    print(f"   Technical Confirmations: {scores['technical'].get('confirmations', 0)}")

if __name__ == "__main__":
    print("🎯 GoldenSignalsAI - ML Model Blending System")
    print("="*60)
    demonstrate_blending() 