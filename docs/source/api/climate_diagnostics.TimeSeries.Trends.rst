climate\_diagnostics.TimeSeries.Trends module
=============================================

.. automodule:: climate_diagnostics.TimeSeries.Trends
   :members:
   :undoc-members:
   :show-inheritance:

Module Description
-----------------

This module provides advanced trend analysis capabilities for climate data through the ``Trends`` class.
It enables users to calculate and visualize trends from climate time series using STL (Seasonal-Trend decomposition
using LOESS) and linear regression, with proper statistical testing for significance.

Key Features
-----------

* Automatic data handling with dask chunking for efficient processing of large datasets
* Filtering by meteorological seasons (Annual, DJF, MAM, JJAS)
* Area-weighted averaging that properly accounts for decreasing grid cell area toward the poles
* STL decomposition to extract the trend component from noisy climate data
* Linear regression with proper statistical testing (p-values, confidence intervals)
* Scaling of trend values to meaningful climate metrics (e.g., °C per decade)

Examples
--------

Basic usage to calculate and visualize a global temperature trend:

.. code-block:: python

   from climate_diagnostics import Trends
   
   # Initialize with a netCDF file
   trends = Trends("path/to/climate_data.nc")
   
   # Calculate global temperature trend
   results = trends.calculate_trend(
       variable="air",
       season="annual",
       area_weighted=True,
       plot=True,
       return_results=True
   )
   
   # Access trend statistics
   slope = results['trend_statistics'].loc['slope', 'value']
   p_value = results['trend_statistics'].loc['p_value', 'value']
   print(f"Trend: {slope:.2f} units/time, p-value: {p_value:.3f}")

Calculate regional trend with specific latitude/longitude bounds:

.. code-block:: python

   # Calculate regional trend for the tropics
   tropical_results = trends.calculate_trend(
       variable="air",
       latitude=slice(-30, 30),  # 30°S to 30°N
       level=850,                # 850 hPa pressure level
       season="djf",             # December-January-February
       area_weighted=True,
       frequency='M',            # Monthly data
       plot=True,
       return_results=True
   )

See Also
--------

* :mod:`climate_diagnostics.TimeSeries.TimeSeries` - For time series extraction and decomposition
* :mod:`climate_diagnostics.plots.plot` - For spatial visualization of climate data