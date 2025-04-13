"""
Climate Diagnostics Toolkit

A Python package for analyzing and visualizing climate data.
"""

__version__ = "0.1.0"

# Import and register accessors
def accessors():
    """
    Register all custom accessors for xarray objects.
    
    This ensures that extensions like `.climate_plots`, `.climate_TimeSeries`, 
    and `.climate_trends` are available on xarray Dataset objects.
    """
    
    from climate_diagnostics.TimeSeries.TimeSeries import TimeSeriesAccessor
    from climate_diagnostics.plots.plot import PlotsAccessor
    from climate_diagnostics.TimeSeries.Trends import TrendsAccessor

accessors()