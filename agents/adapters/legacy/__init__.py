"""Namespace for all adapters that wrap code from the pre-V3 GoldenSignalsAI codebase."""

# Re-export commonly used adapter classes for convenience
from .technical import RSILegacyAdapter, MACDLegacyAdapter  # noqa: F401
from .volume import OBVLegacyAdapter  # noqa: F401
from .options import OptionsFlowLegacyAdapter  # noqa: F401 