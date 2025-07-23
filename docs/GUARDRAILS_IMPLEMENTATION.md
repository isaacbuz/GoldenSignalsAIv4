# Guardrails AI Implementation

## Overview

We've implemented Guardrails AI to ensure all AI outputs are safe, consistent, and within defined boundaries. This provides the final layer of protection before trading decisions are executed.

## Key Components

### 1. Validation Models

Using Pydantic for strong typing:

```python
class TradingDecisionModel(BaseModel):
    action: TradingAction  # BUY, SELL, HOLD
    confidence: float = Field(ge=0.0, le=1.0)
    position_size: float = Field(ge=0.0, le=1.0)
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str = Field(min_length=10, max_length=1000)
    risk_level: RiskLevel  # LOW, MEDIUM, HIGH, EXTREME
```

### 2. Custom Validators

#### Position Size Validator
- Ensures position size doesn't exceed account limits (default 10%)
- Validates risk per trade doesn't exceed limits (default 2%)
- Auto-fixes oversized positions

#### Consistency Validator
- Checks action matches reasoning (no "BUY" with bearish reasoning)
- Ensures confidence aligns with action
- Validates risk level matches position size

#### Safety Validator
- Removes dangerous claims ("guaranteed profit", "risk-free")
- Prevents market manipulation language
- Ensures compliance with regulations

### 3. Integration Points

#### LangGraph Workflow
- Validates final trading decisions before execution
- Auto-fixes minor issues when possible
- Defaults to HOLD on validation failure

#### API Endpoints
- Direct validation endpoints for testing
- Metrics and reporting endpoints
- Batch validation support

## API Endpoints

### Validate Trading Decision
```bash
POST /api/v1/guardrails/validate/decision
{
    "action": "BUY",
    "confidence": 0.85,
    "position_size": 0.15,
    "reasoning": "Strong bullish signals detected",
    "risk_level": "MEDIUM"
}
```

### Validate Market Analysis
```bash
POST /api/v1/guardrails/validate/analysis
{
    "sentiment": "bullish",
    "confidence": 0.75,
    "key_levels": {"support": 150, "resistance": 160},
    "risk_factors": ["Fed meeting", "Earnings report"]
}
```

### Get Validation Metrics
```bash
GET /api/v1/guardrails/metrics
```

### Get Validation Report
```bash
GET /api/v1/guardrails/report?hours=24
```

## Validation Rules

### 1. Trading Decision Rules
- Action must be BUY, SELL, or HOLD
- Confidence between 0 and 1
- Position size between 0 and 1 (as fraction of capital)
- Reasoning must be 10-1000 characters
- Risk level must align with position size

### 2. Position Sizing Rules
- Maximum 10% of account per position
- Maximum 2% risk per trade
- Extreme risk → max 10% position
- High risk → max 30% position

### 3. Consistency Rules
- BUY action cannot have bearish reasoning
- SELL action cannot have bullish reasoning
- High confidence (>80%) requires non-HOLD action
- Low confidence (<50%) requires HOLD action

### 4. Safety Rules
Prohibited phrases:
- "guaranteed profit"
- "risk-free"
- "100% success"
- "insider information"
- "market manipulation"

## Example Validation Flow

```python
# Original decision from AI
decision = {
    "action": "BUY",
    "confidence": 0.85,
    "position_size": 0.25,  # Too large!
    "reasoning": "Strong bullish momentum detected",
    "risk_level": "MEDIUM"
}

# Validation context
context = {
    "account_value": 100000,
    "current_price": 150.50,
    "stop_loss": 145.00
}

# Validate
result = await validate_trading_decision(decision, context)

# Result with auto-fix
{
    "valid": True,
    "modifications": True,
    "decision": {
        "action": "BUY",
        "confidence": 0.85,
        "position_size": 0.10,  # Fixed to max allowed
        "reasoning": "Strong bullish momentum detected",
        "risk_level": "MEDIUM"
    }
}
```

## Metrics and Monitoring

The system tracks:
- **Total Validations**: Number of validation requests
- **Pass Rate**: Percentage passing without modification
- **Fail Rate**: Percentage failing validation
- **Fix Rate**: Percentage auto-fixed

## Benefits

1. **Safety First**: Prevents dangerous trades and claims
2. **Consistency**: Ensures outputs are logically consistent
3. **Risk Management**: Enforces position and risk limits
4. **Compliance**: Helps meet regulatory requirements
5. **Auto-Correction**: Fixes minor issues automatically

## Configuration

Customize validation rules:

```python
# Custom position limits
position_validator = PositionSizeValidator(
    max_position_pct=0.05,  # 5% max position
    max_risk_pct=0.01       # 1% max risk
)

# Custom safety patterns
safety_validator.dangerous_patterns.extend([
    "pump and dump",
    "market cornering"
])
```

## Integration with Other Systems

- **LangSmith**: Tracks validation failures for analysis
- **Vector Memory**: Stores validation patterns
- **MCP Tools**: Validates tool outputs
- **Observability**: Monitors validation performance

## Best Practices

1. **Fail Safe**: Always default to HOLD on validation failure
2. **Log Everything**: Track all validations for compliance
3. **Regular Updates**: Review and update rules based on failures
4. **Test Thoroughly**: Test edge cases and unusual inputs
5. **Monitor Metrics**: Watch for increasing failure rates

The Guardrails implementation provides the final safety net, ensuring all AI-generated trading decisions meet strict safety and consistency standards before execution.
