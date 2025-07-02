"""
Plotting Module

This module provides climate-specific plotting capabilities with cartographic
projections and visualization tools.
"""

__all__ = []

try:
    from .plot import PlotsAccessor
    __all__.append("PlotsAccessor")
except ImportError:
    pass
