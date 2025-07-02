===============================
Trend Analysis API Reference
===============================

The ``climate_trends`` accessor provides trend analysis capabilities for climate data.

Overview
========

The Trends module extends xarray Datasets with a ``.climate_trends`` accessor that provides:

- Time series trend calculation
- Spatial trend analysis across geographic domains
- STL decomposition with linear regression
- **Performance optimization for large-scale trend analysis**
- **Memory-efficient parallel processing**

Quick Example
=============

.. code-block:: python

   import xarray as xr
   import climate_diagnostics
   
   ds = xr.open_dataset("temperature_data.nc")
   
   # Calculate spatial trends
   trends = ds.climate_trends.calculate_spatial_trends(
       variable="air",
       num_years=10  # Trend per decade
   )

Accessor Class
==============

.. autoclass:: climate_diagnostics.TimeSeries.Trends.TrendsAccessor
   :members:
   :undoc-members:
   :show-inheritance:

Available Methods
=================

Trend Calculation
-----------------

.. automethod:: climate_diagnostics.TimeSeries.Trends.TrendsAccessor.calculate_trend
   :no-index:

Spatial Trend Analysis
----------------------

.. automethod:: climate_diagnostics.TimeSeries.Trends.TrendsAccessor.calculate_spatial_trends
   :no-index:

Optimization Methods
--------------------

.. automethod:: climate_diagnostics.TimeSeries.Trends.TrendsAccessor.optimize_for_trends
   :no-index:

Basic Examples
==============

Simple Trend Analysis
---------------------

.. code-block:: python

   # Calculate trend for a time series
   result = ds.climate_trends.calculate_trend(
       variable="air",
       latitude=slice(60, 90),  # Arctic region
       plot=True
   )

Spatial Trends
--------------

.. code-block:: python

   # Calculate spatial trends (creates plot automatically)
   trends = ds.climate_trends.calculate_spatial_trends(
       variable="air",
       num_years=1,  # Trend per year
       plot_map=True
   )

Regional Trend Comparison
-------------------------

.. code-block:: python

   # Compare trends for different regions
   regions = {
       "Arctic": {"latitude": slice(90, 60)},
       "Tropics": {"latitude": slice(23.5, -23.5)}
   }
   
   regional_trends = {}
   for name, bounds in regions.items():
       regional_data = ds.sel(**bounds)
       
       # Calculate trend for the region using spatial trends
       trend_result = regional_data.climate_trends.calculate_spatial_trends(
           variable="air",
           num_years=30,
           plot_map=False  # Don't plot individual regions
       )
       
       # Get mean trend for the region
       regional_trends[name] = trend_result.mean()
   
   # Plot comparison
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 6))
   trend_values = list(regional_trends.values())
   region_names = list(regional_trends.keys())
   plt.bar(region_names, trend_values)
   plt.ylabel("Temperature Trend (K/30 years)")
   plt.title("Regional Trend Comparison")
   plt.show()

Working with Trend Results
==========================

.. code-block:: python

   # Calculate trends and work with results
   trends = ds.climate_trends.calculate_spatial_trends(
       variable="air",
       num_years=10,
       plot_map=False  # Don't auto-plot
   )
   
   # Print statistics
   print(f"Mean global trend: {trends.mean().values:.4f} K/decade")
   print(f"Max trend: {trends.max().values:.4f} K/decade")
   print(f"Min trend: {trends.min().values:.4f} K/decade")
   
   # Create custom plot
   import matplotlib.pyplot as plt
   fig, ax = plt.subplots(figsize=(12, 8))
   trends.plot.contourf(ax=ax, levels=20, cmap="RdBu_r", center=0)
   ax.set_title("Temperature Trends (K/decade)")
   plt.show()

Performance Optimization for Trends
===================================

Optimizing Large Datasets
-------------------------

.. code-block:: python

   # Optimize dataset for trend analysis
   ds_optimized = ds.climate_trends.optimize_for_trends(
       variable="air",
       use_case='trend_analysis'
   )
   
   # Calculate trends on optimized dataset
   trends = ds_optimized.climate_trends.calculate_spatial_trends(
       variable="air",
       num_years=30,
       optimize_chunks=True
   )

Memory-Efficient Trend Workflows
--------------------------------

.. code-block:: python

   # Complete workflow for large datasets
   import xarray as xr
   import climate_diagnostics
   
   # Load large climate dataset
   ds = xr.open_dataset("large_climate_model.nc")
   
   # Optimize for trend calculations
   ds_opt = ds.climate_trends.optimize_for_trends(
       variable="temperature"
   )
   
   # Calculate long-term trends efficiently
   long_term_trends = ds_opt.climate_trends.calculate_spatial_trends(
       variable="temperature",
       num_years=50,  # 50-year trends
       n_workers=8,   # Use 8 parallel workers
       optimize_chunks=True,
       chunk_target_mb=150
   )
   
   # Calculate regional trend time series
   regional_trend = ds_opt.climate_trends.calculate_trend(
       variable="temperature",
       latitude=slice(30, 60),
       longitude=slice(-120, -80),
       plot=True
   )

See Also
========

* :doc:`./timeseries` - Time series analysis methods
* :doc:`./plots` - Plotting functions
