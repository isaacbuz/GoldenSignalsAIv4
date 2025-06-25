from typing import Dict, Any

from src.models.signals import Signal, SignalType, SignalStrength


def legacy_output_to_signal(legacy_out: Dict[str, Any], symbol: str, current_price: float, source: str) -> Signal:
    """Convert a legacy agent output dict into a V3 ``Signal`` object.

    Expected legacy format::
        {
            "action": "buy" | "sell" | "hold",
            "confidence": float,                 # 0 – 1
            "metadata": {...}                    # optional
        }
    """
    action_map = {
        "buy": SignalType.BUY,
        "sell": SignalType.SELL,
        "hold": SignalType.HOLD,
    }

    action_str = str(legacy_out.get("action", "hold")).lower()
    signal_type = action_map.get(action_str, SignalType.HOLD)

    confidence = float(legacy_out.get("confidence", 0.0))
    confidence = max(0.0, min(confidence, 1.0))  # clamp

    # Derive strength based on confidence for legacy outputs.
    if confidence >= 0.8:
        strength = SignalStrength.STRONG
    elif confidence >= 0.6:
        strength = SignalStrength.MODERATE
    else:
        strength = SignalStrength.WEAK

    return Signal(
        symbol=symbol,
        signal_type=signal_type,
        confidence=confidence,
        strength=strength,
        source=source,
        current_price=current_price,
        # For now we leave target, stop, risk_score as None / 0; orchestrator will fill.
        target_price=None,
        stop_loss=None,
        risk_score=1.0 - confidence,  # simple placeholder
        reasoning=legacy_out.get("metadata", {}).get("explanation") if legacy_out.get("metadata") else None,
        indicators=legacy_out.get("metadata", {}),
    ) 