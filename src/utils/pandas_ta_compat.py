"""
Compatibility layer for pandas_ta with newer numpy versions
Fixes the numpy.NaN -> numpy.nan issue
"""

import sys

import numpy as np

# Monkey patch numpy to support old NaN reference
if not hasattr(np, "NaN"):
    np.NaN = np.nan

# Import pandas_ta after patching
try:
    import pandas_ta as ta

    print("pandas_ta imported successfully with compatibility patch")
except ImportError as e:
    print(f"Failed to import pandas_ta even with patch: {e}")
    ta = None

__all__ = ["ta"]
