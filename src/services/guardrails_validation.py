"""
Guardrails AI Validation Service
Ensures AI outputs are safe, consistent, and within defined boundaries
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from guardrails import Guard
from guardrails.validators import FailResult, PassResult, ValidationResult, ValidChoices, ValidRange
from pydantic import BaseModel, Field, root_validator, validator

logger = logging.getLogger(__name__)


class TradingAction(str, Enum):
    """Valid trading actions"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RiskLevel(str, Enum):
    """Valid risk levels"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class TradingDecisionModel(BaseModel):
    """Validated trading decision model"""

    action: TradingAction
    confidence: float = Field(ge=0.0, le=1.0)
    position_size: float = Field(ge=0.0, le=1.0)
    stop_loss: Optional[float] = Field(default=None, ge=0.0)
    take_profit: Optional[float] = Field(default=None, ge=0.0)
    reasoning: str = Field(min_length=10, max_length=1000)
    risk_level: RiskLevel

    @validator("stop_loss")
    def validate_stop_loss(cls, v, values):
        """Ensure stop loss is reasonable"""
        if v is not None and "action" in values:
            if values["action"] == TradingAction.BUY and v <= 0:
                raise ValueError("Stop loss must be positive for BUY orders")
        return v

    @validator("take_profit")
    def validate_take_profit(cls, v, values):
        """Ensure take profit is reasonable"""
        if v is not None and "action" in values:
            if values["action"] == TradingAction.BUY and v <= 0:
                raise ValueError("Take profit must be positive for BUY orders")
        return v

    @root_validator
    def validate_risk_alignment(cls, values):
        """Ensure risk level aligns with position size"""
        risk_level = values.get("risk_level")
        position_size = values.get("position_size", 0)

        if risk_level == RiskLevel.EXTREME and position_size > 0.1:
            raise ValueError("Position size too large for EXTREME risk")
        elif risk_level == RiskLevel.HIGH and position_size > 0.3:
            raise ValueError("Position size too large for HIGH risk")

        return values


class MarketAnalysisModel(BaseModel):
    """Validated market analysis model"""

    sentiment: str = Field(pattern="^(bullish|bearish|neutral)$")
    confidence: float = Field(ge=0.0, le=1.0)
    key_levels: Dict[str, float]
    risk_factors: List[str] = Field(max_items=10)
    opportunities: List[str] = Field(max_items=10)

    @validator("key_levels")
    def validate_key_levels(cls, v):
        """Ensure key levels are reasonable"""
        required_keys = ["support", "resistance"]
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required key level: {key}")
            if v[key] <= 0:
                raise ValueError(f"Key level {key} must be positive")
        return v


class AgentSignalModel(BaseModel):
    """Validated agent signal model"""

    agent_name: str = Field(min_length=3, max_length=50)
    signal: TradingAction
    confidence: float = Field(ge=0.0, le=1.0)
    indicator_value: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("metadata")
    def validate_metadata_size(cls, v):
        """Ensure metadata isn't too large"""
        if len(json.dumps(v)) > 10000:
            raise ValueError("Metadata too large")
        return v


# Custom Validators


class PositionSizeValidator:
    """Validates position size based on account and risk rules"""

    def __init__(self, max_position_pct: float = 0.1, max_risk_pct: float = 0.02):
        self.max_position_pct = max_position_pct
        self.max_risk_pct = max_risk_pct

    def validate(self, value: float, metadata: Dict[str, Any]) -> ValidationResult:
        """Validate position size"""
        account_value = metadata.get("account_value", 100000)
        position_value = value * metadata.get("current_price", 100)
        position_pct = position_value / account_value

        if position_pct > self.max_position_pct:
            return FailResult(
                error_message=f"Position size {position_pct:.1%} exceeds max {self.max_position_pct:.1%}",
                fix_value=self.max_position_pct
                * account_value
                / metadata.get("current_price", 100),
            )

        # Check risk
        stop_loss = metadata.get("stop_loss", 0)
        if stop_loss > 0:
            risk_amount = abs(metadata.get("current_price", 100) - stop_loss) * value
            risk_pct = risk_amount / account_value

            if risk_pct > self.max_risk_pct:
                return FailResult(
                    error_message=f"Risk {risk_pct:.1%} exceeds max {self.max_risk_pct:.1%}",
                    fix_value=self.max_risk_pct
                    * account_value
                    / abs(metadata.get("current_price", 100) - stop_loss),
                )

        return PassResult()


class ConsistencyValidator:
    """Validates consistency between different parts of output"""

    def validate(self, value: Dict[str, Any], metadata: Dict[str, Any]) -> ValidationResult:
        """Check for logical consistency"""
        action = value.get("action")
        confidence = value.get("confidence", 0)
        reasoning = value.get("reasoning", "").lower()

        # Check action matches reasoning
        if action == "BUY" and any(word in reasoning for word in ["bearish", "sell", "decline"]):
            return FailResult(
                error_message="Action BUY contradicts bearish reasoning",
                fix_value={**value, "action": "HOLD"},
            )

        if action == "SELL" and any(word in reasoning for word in ["bullish", "buy", "rally"]):
            return FailResult(
                error_message="Action SELL contradicts bullish reasoning",
                fix_value={**value, "action": "HOLD"},
            )

        # Check confidence alignment
        if confidence > 0.8 and action == "HOLD":
            return FailResult(error_message="High confidence but HOLD action is inconsistent")

        if confidence < 0.5 and action != "HOLD":
            return FailResult(
                error_message="Low confidence should result in HOLD action",
                fix_value={**value, "action": "HOLD"},
            )

        return PassResult()


class SafetyValidator:
    """Validates output doesn't contain harmful instructions"""

    def __init__(self):
        self.dangerous_patterns = [
            "guaranteed profit",
            "risk-free",
            "100% success",
            "insider information",
            "market manipulation",
        ]

    def validate(self, value: str, metadata: Dict[str, Any]) -> ValidationResult:
        """Check for dangerous claims"""
        value_lower = value.lower()

        for pattern in self.dangerous_patterns:
            if pattern in value_lower:
                return FailResult(
                    error_message=f"Contains dangerous claim: '{pattern}'",
                    fix_value=value.replace(pattern, "[removed]"),
                )

        return PassResult()


class GuardrailsService:
    """Main service for validating AI outputs"""

    def __init__(self):
        # Initialize validators
        self.position_validator = PositionSizeValidator()
        self.consistency_validator = ConsistencyValidator()
        self.safety_validator = SafetyValidator()

        # Create guards for different output types
        self.trading_guard = self._create_trading_guard()
        self.analysis_guard = self._create_analysis_guard()
        self.signal_guard = self._create_signal_guard()

        # Metrics
        self.validation_metrics = {"total_validations": 0, "passed": 0, "failed": 0, "fixed": 0}

    def _create_trading_guard(self) -> Guard:
        """Create guard for trading decisions"""
        from guardrails.rail import Rail

        rail_spec = """
<rail version="0.1">
<output>
    <string name="action" validators="valid-choices: {choices: ['BUY', 'SELL', 'HOLD']}" />
    <float name="confidence" validators="valid-range: {min: 0.0, max: 1.0}" />
    <float name="position_size" validators="valid-range: {min: 0.0, max: 1.0}" />
    <string name="reasoning" validators="length: {min: 10, max: 1000}" />
</output>
</rail>
"""

        guard = Guard.from_rail_string(rail_spec)
        return guard

    def _create_analysis_guard(self) -> Guard:
        """Create guard for market analysis"""
        from guardrails.rail import Rail

        rail_spec = """
<rail version="0.1">
<output>
    <string name="sentiment" validators="valid-choices: {choices: ['bullish', 'bearish', 'neutral']}" />
    <float name="confidence" validators="valid-range: {min: 0.0, max: 1.0}" />
    <list name="risk_factors">
        <string validators="length: {max: 200}" />
    </list>
</output>
</rail>
"""

        guard = Guard.from_rail_string(rail_spec)
        return guard

    def _create_signal_guard(self) -> Guard:
        """Create guard for agent signals"""
        from guardrails.rail import Rail

        rail_spec = """
<rail version="0.1">
<output>
    <string name="agent_name" validators="length: {min: 3, max: 50}" />
    <string name="signal" validators="valid-choices: {choices: ['BUY', 'SELL', 'HOLD']}" />
    <float name="confidence" validators="valid-range: {min: 0.0, max: 1.0}" />
</output>
</rail>
"""

        guard = Guard.from_rail_string(rail_spec)
        return guard

    async def validate_trading_decision(
        self, decision: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate a trading decision"""
        try:
            self.validation_metrics["total_validations"] += 1

            # First, validate with Pydantic model
            try:
                validated_model = TradingDecisionModel(**decision)
                decision = validated_model.dict()
            except Exception as e:
                logger.error(f"Pydantic validation failed: {e}")
                self.validation_metrics["failed"] += 1
                return {"valid": False, "errors": str(e), "original": decision}

            # Apply custom validators
            position_result = self.position_validator.validate(
                decision.get("position_size", 0), context or {}
            )

            if isinstance(position_result, FailResult):
                logger.warning(f"Position size validation failed: {position_result.error_message}")
                if position_result.fix_value:
                    decision["position_size"] = position_result.fix_value
                    self.validation_metrics["fixed"] += 1

            consistency_result = self.consistency_validator.validate(decision, context or {})
            if isinstance(consistency_result, FailResult):
                logger.warning(f"Consistency validation failed: {consistency_result.error_message}")
                if consistency_result.fix_value:
                    decision.update(consistency_result.fix_value)
                    self.validation_metrics["fixed"] += 1

            # Safety check on reasoning
            safety_result = self.safety_validator.validate(
                decision.get("reasoning", ""), context or {}
            )

            if isinstance(safety_result, FailResult):
                logger.warning(f"Safety validation failed: {safety_result.error_message}")
                decision["reasoning"] = safety_result.fix_value or decision["reasoning"]
                self.validation_metrics["fixed"] += 1

            self.validation_metrics["passed"] += 1

            return {
                "valid": True,
                "decision": decision,
                "modifications": position_result != PassResult()
                or consistency_result != PassResult()
                or safety_result != PassResult(),
            }

        except Exception as e:
            logger.error(f"Validation error: {e}")
            self.validation_metrics["failed"] += 1
            return {"valid": False, "errors": str(e), "original": decision}

    async def validate_market_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate market analysis output"""
        try:
            # Validate with Pydantic
            validated_model = MarketAnalysisModel(**analysis)

            # Additional safety checks
            for risk in validated_model.risk_factors:
                safety_result = self.safety_validator.validate(risk, {})
                if isinstance(safety_result, FailResult):
                    logger.warning(f"Unsafe risk factor: {risk}")
                    validated_model.risk_factors.remove(risk)

            return {"valid": True, "analysis": validated_model.dict()}

        except Exception as e:
            logger.error(f"Market analysis validation failed: {e}")
            return {"valid": False, "errors": str(e), "original": analysis}

    async def validate_agent_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Validate agent signal"""
        try:
            # Validate with Pydantic
            validated_model = AgentSignalModel(**signal)

            return {"valid": True, "signal": validated_model.dict()}

        except Exception as e:
            logger.error(f"Agent signal validation failed: {e}")
            return {"valid": False, "errors": str(e), "original": signal}

    def get_metrics(self) -> Dict[str, Any]:
        """Get validation metrics"""
        total = self.validation_metrics["total_validations"]

        return {
            **self.validation_metrics,
            "pass_rate": self.validation_metrics["passed"] / total if total > 0 else 0,
            "fail_rate": self.validation_metrics["failed"] / total if total > 0 else 0,
            "fix_rate": self.validation_metrics["fixed"] / total if total > 0 else 0,
        }

    async def create_validation_report(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """Create a validation report"""
        return {
            "timeframe_hours": timeframe_hours,
            "metrics": self.get_metrics(),
            "common_failures": self._get_common_failures(),
            "recommendations": self._get_recommendations(),
            "timestamp": datetime.now().isoformat(),
        }

    def _get_common_failures(self) -> List[str]:
        """Identify common validation failures"""
        # In production, this would analyze actual failure patterns
        return [
            "Position size exceeds risk limits",
            "Inconsistent action and reasoning",
            "Confidence below threshold for execution",
            "Missing required fields",
        ]

    def _get_recommendations(self) -> List[str]:
        """Get recommendations for improving validation"""
        metrics = self.get_metrics()
        recommendations = []

        if metrics.get("fail_rate", 0) > 0.1:
            recommendations.append("High failure rate - review AI prompt engineering")

        if metrics.get("fix_rate", 0) > 0.2:
            recommendations.append("Many auto-fixes applied - consider retraining models")

        return recommendations


# Singleton instance
guardrails_service = GuardrailsService()


# Convenience functions
async def validate_trading_decision(
    decision: Dict[str, Any], context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Validate a trading decision"""
    return await guardrails_service.validate_trading_decision(decision, context)


async def validate_market_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Validate market analysis"""
    return await guardrails_service.validate_market_analysis(analysis)


async def validate_agent_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    """Validate agent signal"""
    return await guardrails_service.validate_agent_signal(signal)


def get_validation_metrics() -> Dict[str, Any]:
    """Get validation metrics"""
    return guardrails_service.get_metrics()
