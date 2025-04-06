climate\_diagnostics.plots.plot module
======================================

.. automodule:: climate_diagnostics.plots.plot
   :members:
   :undoc-members:
   :show-inheritance:

Module Description
-----------------

This module provides visualization capabilities for climate data through the ``Plots`` class.
It includes functions for plotting spatial means, standard deviations, and other climate metrics
with proper geographical projections using Cartopy.

Key Features
-----------

* Automatic handling of datasets with dask chunking
* Filtering by meteorological seasons (Annual, DJF, MAM, JJAS)
* Spatial visualization with proper map projections
* Selection by latitude, longitude, level, and time range

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

Plot standard deviation over time:

.. code-block:: python

   plots.plot_std_time(
       latitude=slice(-30, 30),
       longitude=slice(0, 180),
       variable="air",
       season="jjas"  # June-July-August-September
   )

See Also
--------

* :mod:`climate_diagnostics.TimeSeries.TimeSeries` - For time series analysis
* :mod:`climate_diagnostics.TimeSeries.Trends` - For trend analysis