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

The ``Trends`` class is designed to handle common climate data formats with automatic detection of coordinate names
(lat, lon, time, level) for maximum compatibility across different datasets and conventions.

Key Features
-----------

* Automatic data handling with dask chunking for efficient processing of large datasets
* Filtering by meteorological seasons (Annual, DJF, MAM, JJA, JJAS, SON)
* Area-weighted averaging that properly accounts for decreasing grid cell area toward the poles
* STL decomposition to extract the trend component from noisy climate data
* Linear regression with proper statistical testing (p-values, confidence intervals)
* Scaling of trend values to meaningful climate metrics (e.g., °C per decade)
* Support for global, regional, and point-based trend analysis
* Automatic coordinate name detection for compatibility with different dataset conventions
* Publication-quality visualization of trend components and regression results

Classes
-------

.. autoclass:: Trends
   :members:

   The core class for trend analysis in climate datasets. It provides methods to load,
   filter, and analyze climate data trends from NetCDF files.

   .. automethod:: __init__

      Initialize the Trends class for analyzing climate data trends.

      :param filepath: Path to the NetCDF or other compatible climate data file.
                      If None, the dataset must be loaded manually later.

   .. automethod:: calculate_trend

      Calculate trends from time series using STL decomposition and linear regression.
      
      :param variable: Variable name to analyze in the dataset (default: 'air')
      :param latitude: Latitude selection as point value (float) or region (slice)
      :param longitude: Longitude selection as point value (float) or region (slice)
      :param level: Pressure level selection (if applicable)
      :param frequency: Time frequency of data: 'M' (monthly), 'D' (daily), or 'Y' (yearly)
      :param season: Season to analyze: 'annual', 'jjas' (Jun-Sep), 'djf' (Dec-Feb), 'mam' (Mar-May), 'jja' (Jun-Aug), 'son' (Sep-Nov)
      :param area_weighted: Apply cosine latitude weighting for area-representative averaging
      :param period: Period for STL decomposition (12 for monthly data, 365 for daily data)
      :param plot: Generate visualization of the trend analysis
      :param return_results: Return dictionary with calculation results
      :return: Dictionary with results if return_results=True, otherwise None

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
       level=850,  # 850 hPa level
       season="annual",
       area_weighted=True,
       plot=True,
       return_results=True
   )
   
   # Check if trend is statistically significant (p < 0.05)
   p_value = results['trend_statistics'].loc[
       results['trend_statistics']['statistic'] == 'p_value', 'value'
   ].values[0]
   print(f"Trend is {'significant' if p_value < 0.05 else 'not significant'}")
   
   # Access trend value in units per decade
   slope = results['trend_statistics'].loc[
       results['trend_statistics']['statistic'] == 'slope', 'value'
   ].values[0]
   print(f"Trend magnitude: {slope:.2f} units/time")

Calculate regional trend with specific latitude/longitude bounds:

.. code-block:: python

   # Calculate regional trend for the tropics
   tropical_results = trends.calculate_trend(
       variable="precip",
       latitude=slice(-15, 15),   # Tropical band
       longitude=slice(60, 180),  # Maritime continent to Pacific
       level=850,                 # 850 hPa pressure level
       season="djf",              # Northern winter/Southern summer
       area_weighted=True,
       frequency='M',             # Monthly data
       period=12,                 # For monthly data
       plot=True,
       return_results=True
   )
   
   # Access the extracted trend component (pandas.Series)
   trend = tropical_results['trend_component']
   
   # Access the linear regression model
   model = tropical_results['regression_model']

Analyzing point-specific trends:

.. code-block:: python

   # Analyze trend at a specific location
   point_results = trends.calculate_trend(
       variable="air",
       latitude=0,            # Equator
       longitude=180,         # International Date Line
       level=850,             # 850 hPa pressure level
       season="jjas",         # Extended Northern summer
       frequency='M',         # Monthly data
       return_results=True
   )

See Also
--------

* :mod:`climate_diagnostics.TimeSeries.TimeSeries` - For time series extraction and decomposition
* :mod:`climate_diagnostics.plots.plot` - For spatial visualization of climate data