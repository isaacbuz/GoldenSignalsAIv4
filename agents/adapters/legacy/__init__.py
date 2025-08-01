"""Namespace for all adapters that wrap code from the pre-V3 GoldenSignalsAI codebase."""

# Re-export commonly used adapter classes for convenience
from .options import OptionsFlowLegacyAdapter  # noqa: F401
from .technical import MACDLegacyAdapter, RSILegacyAdapter  # noqa: F401
from .volume import OBVLegacyAdapter  # noqa: F401
