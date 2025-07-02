"""
Climate Diagnostics Toolkit
===========================

A comprehensive Python package for analyzing and visualizing climate data from
various sources including model output, reanalysis products, and observations.

This toolkit provides specialized tools for:
- Temporal analysis (trends, variability, decomposition)
- Spatial pattern visualization
- Statistical climate diagnostics
- Multi-model comparison and evaluation

The package extends xarray functionality through custom accessors that seamlessly
integrate with xarray Dataset objects.

Main Components
--------------
- climate_plots: Geographical visualizations with customized projections
- climate_timeseries: Time series analysis and decomposition
- climate_trends: Linear trend analysis and significance testing

Examples
--------
>>> import xarray as xr
>>> import climate_diagnostics
>>> 
>>> # Open a NetCDF climate dataset
>>> ds = xr.open_dataset("era5_monthly_temperature.nc")
>>> 
>>> # Create a spatial mean plot of temperature
>>> ds.climate_plots.plot_mean(variable="t2m", season="djf")
>>> 
>>> # Decompose a temperature time series
>>> ds.climate_timeseries.decompose_time_series(variable="t2m", latitude=slice(60, 30))
>>> 
>>> # Calculate and visualize temperature trends
>>> ds.climate_trends.calculate_spatial_trends(variable="t2m", num_years=10)
"""

__version__ = "1.1.1"

# Conditional import of main modules to handle optional dependencies gracefully
# This allows the package to function even if some optional components are unavailable
try:
    from . import utils
    _utils_available = True
except ImportError:
    _utils_available = False

try:
    from . import models  
    _models_available = True
except ImportError:
    _models_available = False

try:
    from . import TimeSeries
    _timeseries_available = True
except ImportError:
    _timeseries_available = False
    
try:
    from . import plots
    _plots_available = True
except ImportError:
    _plots_available = False

# Import and register accessors
def register_accessors():
    """
    Register all custom accessors for xarray objects.
    
    This function imports and registers the custom xarray accessors that extend
    xarray.Dataset objects with climate-specific analysis capabilities. After
    registration, the following accessors become available:
    
    - .climate_plots: Geographic visualization methods for climate data
    - .climate_timeseries: Time series analysis tools
    - .climate_trends: Statistical trend calculation and visualization
    
    The registration happens automatically when importing the package. It only
    needs to be called manually if working with custom import patterns.
    
    Examples
    --------
    >>> import xarray as xr
    >>> import climate_diagnostics
    >>> 
    >>> # Accessors are already registered
    >>> ds = xr.open_dataset("climate_data.nc")
    >>> ds.climate_plots.plot_mean(variable="temperature")
    >>> 
    >>> # If using custom import patterns:
    >>> from climate_diagnostics import register_accessors
    >>> register_accessors()
    """
    
    # Import and register all accessor classes with comprehensive error handling
    try:
        from climate_diagnostics.TimeSeries.TimeSeries import TimeSeriesAccessor
        from climate_diagnostics.plots.plot import PlotsAccessor
        from climate_diagnostics.TimeSeries.Trends import TrendsAccessor
        print("✓ All xarray accessors registered successfully")
    except ImportError as e:
        print(f"Warning: Could not register some accessors due to missing dependencies: {e}")
        print("Install all dependencies to use the full functionality.")

# Automatically register accessors when the module is imported
# This provides immediate access to .climate_* methods on xarray objects
register_accessors()

# Define what gets imported with "from climate_diagnostics import *"
__all__ = ["__version__", "register_accessors"]

# Conditionally add available modules to __all__ based on successful imports
if _utils_available:
    __all__.append("utils")
if _models_available:
    __all__.append("models") 
if _timeseries_available:
    __all__.append("TimeSeries")
if _plots_available:
    __all__.append("plots")