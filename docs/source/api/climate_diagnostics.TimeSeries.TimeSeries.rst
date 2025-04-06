climate\_diagnostics.TimeSeries.TimeSeries module
================================================

.. automodule:: climate_diagnostics.TimeSeries.TimeSeries
   :members:
   :undoc-members:
   :show-inheritance:

Module Description
-----------------

This module provides comprehensive time series analysis functionality for climate data through the ``TimeSeries`` class.
It enables researchers to extract, process, analyze, and visualize time series from NetCDF climate datasets,
with intelligent handling of spatial dimensions and proper area-weighted averaging.

Key Features
-----------

* Efficient data handling with automatic dask chunking for large climate datasets
* Meteorological season filtering (Annual, DJF, MAM, JJAS) with flexible selection
* Spatial and temporal subsetting by latitude, longitude, pressure level, and time range
* Proper area-weighted averaging that accounts for decreasing grid cell size toward the poles
* Time series visualization with customizable parameters and automatic unit handling 
* Advanced time series decomposition using Seasonal-Trend decomposition using LOESS (STL)
* Spatial variability analysis with standard deviation calculations across dimensions

Class Methods
------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Method
     - Description
   * - ``__init__(filepath)``
     - Initialize with NetCDF file path for climate data analysis
   * - ``plot_time_series()``
     - Generate area-weighted time series plots with flexible spatial-temporal selection
   * - ``plot_std_space()``
     - Plot spatial standard deviation over time to analyze spatial variability
   * - ``decompose_time_series()``
     - Decompose time series into trend, seasonal cycle, and residual components

Examples
--------

Basic usage to plot a global or regional time series:

.. code-block:: python

   from climate_diagnostics import TimeSeries
   
   # Initialize with a netCDF file
   ts = TimeSeries("path/to/climate_data.nc")
   
   # Plot global mean time series
   ts.plot_time_series(
       variable="air",
       level=850,  # 850 hPa pressure level
       season="annual"
   )
   
   # Plot time series for a specific tropical region
   ts.plot_time_series(
       latitude=slice(-30, 30),    # 30°S to 30°N
       longitude=slice(0, 180),    # 0° to 180°E
       level=850,
       variable="air",
       season="djf",               # December-January-February
       time_range=slice("1980", "2020")
   )

Analyze spatial variability using standard deviation:

.. code-block:: python

   # Plot how spatial variability changes over time
   ts.plot_std_space(
       latitude=slice(0, 45),      # 0° to 45°N
       longitude=slice(60, 120),   # 60°E to 120°E
       variable="precip",          # Precipitation
       season="jjas",              # June-July-August-September (monsoon season)
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
       stl_period=12,              # For monthly data
       area_weighted=True,
       plot_results=True           # Creates visualization of components
   )
   
   # Access the decomposition components
   trend = results['trend']        # Long-term trend component
   seasonal = results['seasonal']  # Seasonal cycle
   residual = results['residual']  # Residual variability/anomalies
   
   # Further analysis with trend component
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 6))
   plt.plot(trend.index, trend.values)
   plt.title("Long-term Temperature Trend")
   plt.xlabel("Year")
   plt.ylabel("Temperature (K)")
   plt.grid(True, linestyle='--', alpha=0.7)

See Also
--------

* :class:`climate_diagnostics.TimeSeries.Trends` - For trend analysis and statistical testing
* :class:`climate_diagnostics.Plots` - For spatial visualization and mapping of climate data