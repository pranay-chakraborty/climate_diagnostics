"""
Time Series Analysis Module

This module provides time series analysis capabilities for climate data,
including basic plotting, decomposition, and trend analysis.
"""

__all__ = []

try:
    from .TimeSeries import TimeSeriesAccessor
    __all__.append("TimeSeriesAccessor")
except ImportError:
    pass

try:
    from .Trends import TrendsAccessor
    __all__.append("TrendsAccessor")
except ImportError:
    pass
