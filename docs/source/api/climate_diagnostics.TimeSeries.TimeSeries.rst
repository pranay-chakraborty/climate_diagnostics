climate\_diagnostics.TimeSeries.TimeSeries module
================================================

.. automodule:: climate_diagnostics.TimeSeries.TimeSeries
   :members:
   :undoc-members:
   :show-inheritance:

Module Description
-----------------

This module provides the core time series analysis functionality for climate data through the ``TimeSeries`` class.
It includes methods for extracting, processing, and visualizing time series from climate datasets,
with support for seasonal filtering and area-weighted averaging.

Key Features
-----------

* Automatic handling of datasets with dask chunking for efficient memory usage
* Filtering by meteorological seasons (Annual, DJF, MAM, JJAS)
* Spatial and temporal selection by latitude, longitude, level, and time range
* Area-weighted averaging that accounts for grid cell size differences
* Time series decomposition using Seasonal-Trend decomposition using LOESS (STL)

Examples
--------

Basic usage to plot a time series:

.. code-block:: python

   from climate_diagnostics import TimeSeries
   
   # Initialize with a netCDF file
   ts = TimeSeries("path/to/climate_data.nc")
   
   # Plot time series for a specific region
   ts.plot_time_series(
       latitude=slice(-30, 30),
       longitude=slice(0, 180),
       level=850,
       variable="air",
       season="djf"  # December-January-February
   )

Decompose a time series into trend, seasonal, and residual components:

.. code-block:: python

   # Decompose time series with STL
   results, fig = ts.decompose_time_series(
       variable="air",
       level=850,
       latitude=slice(-30, 30),
       longitude=slice(0, 180),
       season="annual",
       stl_period=12,  # For monthly data
       area_weighted=True,
       plot_results=True
   )
   
   # Access the decomposition components
   trend = results['trend']
   seasonal = results['seasonal']
   residual = results['residual']

See Also
--------

* :mod:`climate_diagnostics.TimeSeries.Trends` - For trend analysis and statistical testing
* :mod:`climate_diagnostics.plots.plot` - For spatial visualization of climate data