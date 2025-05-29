from backend.db.connect import SessionLocal
from backend.db.models import SignalRecord
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

# Save a signal bundle (from signal_engine) to PostgreSQL
def save_signals(symbol, signals, raw_data=None):
    """
    Save a list of normalized signals to PostgreSQL.
    Each signal should have: name, source, confidence, explanation, signal
    'source' is stored in signal_type for clarity.
    """
    session = SessionLocal()
    try:
        for sig in signals:
            record = SignalRecord(
                symbol=symbol,
                signal_name=sig.get("name"),
                signal_type=sig.get("source", sig.get("type", "unknown")),
                signal=sig.get("signal"),
                confidence=sig.get("confidence", 0),
                explanation=sig.get("explanation", ""),
                raw_data=raw_data if raw_data else sig,
                timestamp=datetime.utcnow(),
            )
            session.add(record)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        print(f"[DB] Error saving signals: {e}")
    finally:
        session.close()
