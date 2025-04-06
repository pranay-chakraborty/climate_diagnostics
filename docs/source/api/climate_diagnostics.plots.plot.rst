climate\_diagnostics.plots.plot module
======================================

.. automodule:: climate_diagnostics.plots.plot
   :members:
   :undoc-members:
   :show-inheritance:

Module Description
-----------------

This module provides comprehensive visualization capabilities for climate data through the ``Plots`` class.
It enables meteorologists and climate scientists to create publication-quality maps and figures from
NetCDF files using proper geographical projections via Cartopy. The module handles spatial averaging,
seasonal filtering, and statistical computations with built-in support for large datasets through dask.

Key Features
-----------

* Automatic handling of large datasets with dask chunking for memory efficiency
* Filtering by meteorological seasons (Annual, DJF, MAM, JJA, JJAS, SON)
* Spatial visualization with proper map projections and coastlines
* Flexible selection by latitude, longitude, pressure level, and time range
* Statistical visualizations including means and standard deviations across both spatial and temporal dimensions
* Progress bars for long-running computations

Examples
--------

Basic usage to plot the mean of a variable:

.. code-block:: python

   from climate_diagnostics import Plots
   
   # Initialize with a netCDF file
   plots = Plots("path/to/climate_data.nc")
   
   # Plot mean temperature
   plots.plot_mean(
       variable="air",
       level=850,
       season="djf"  # December-January-February
   )

Plot spatial mean with regional focus:

.. code-block:: python

   plots.plot_mean(
       latitude=slice(0, 40),     # 0° to 40°N
       longitude=slice(60, 100),  # 60°E to 100°E
       variable="precip",         # Precipitation
       season="jjas"              # June-July-August-September
   )

Plot standard deviation over time:

.. code-block:: python

   plots.plot_std_time(
       latitude=slice(-30, 30),   # 30°S to 30°N
       longitude=slice(0, 180),   # 0° to 180°E
       variable="air",
       season="jjas"              # June-July-August-September
   )

Plot interannual variability of precipitation:

.. code-block:: python

   plots.plot_std_time(
       variable="precip",        # Precipitation
       time_range=slice("1980-01-01", "2020-01-01"),  # Study period
       season="jjas"             # June-July-August-September
   )

See Also
--------

* :mod:`climate_diagnostics.TimeSeries.TimeSeries` - For time series analysis and decomposition
* :mod:`climate_diagnostics.TimeSeries.Trends` - For trend analysis and statistical significance testing
* :mod:`xarray` - For underlying data structures and operations
* :mod:`cartopy` - For map projections and geographical features