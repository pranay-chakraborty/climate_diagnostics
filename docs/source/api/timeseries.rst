====================================
Time Series Analysis API Reference
====================================

The ``climate_timeseries`` accessor provides time series analysis capabilities for climate data.

Overview
========

The TimeSeries module extends xarray Datasets with a ``.climate_timeseries`` accessor that provides:

- Time series plotting and visualization
- Spatial standard deviation analysis
- STL decomposition for trend and seasonal analysis
- **Advanced chunking optimization for large datasets**
- **Memory-efficient processing strategies**
- **Performance tuning and diagnostics**

Quick Example
=============

.. code-block:: python

   import xarray as xr
   import climate_diagnostics
   
   ds = xr.open_dataset("temperature_data.nc")
   
   # Plot a time series
   fig = ds.climate_timeseries.plot_time_series(
       variable="air",
       latitude=slice(30, 60)
   )

Accessor Class
==============

.. autoclass:: climate_diagnostics.TimeSeries.TimeSeries.TimeSeriesAccessor
   :members:
   :undoc-members:
   :show-inheritance:

Available Methods
=================

Time Series Plotting
--------------------

.. automethod:: climate_diagnostics.TimeSeries.TimeSeries.TimeSeriesAccessor.plot_time_series
   :no-index:

Statistical Analysis
--------------------

.. automethod:: climate_diagnostics.TimeSeries.TimeSeries.TimeSeriesAccessor.plot_std_space
   :no-index:

Decomposition Methods
---------------------

.. automethod:: climate_diagnostics.TimeSeries.TimeSeries.TimeSeriesAccessor.decompose_time_series
   :no-index:

Chunking and Optimization
-------------------------

.. automethod:: climate_diagnostics.TimeSeries.TimeSeries.TimeSeriesAccessor.optimize_chunks
   :no-index:

.. automethod:: climate_diagnostics.TimeSeries.TimeSeries.TimeSeriesAccessor.optimize_chunks_advanced
   :no-index:

.. automethod:: climate_diagnostics.TimeSeries.TimeSeries.TimeSeriesAccessor.print_chunking_info
   :no-index:

.. automethod:: climate_diagnostics.TimeSeries.TimeSeries.TimeSeriesAccessor.analyze_chunking_strategy
   :no-index:

.. automethod:: climate_diagnostics.TimeSeries.TimeSeries.TimeSeriesAccessor.optimize_for_decomposition
   :no-index:

Basic Examples
==============

Simple Time Series Plot
-----------------------

.. code-block:: python

   # Plot global mean time series
   fig = ds.climate_timeseries.plot_time_series(
       variable="air",
       area_weighted=True
   )

Regional Time Series
--------------------

.. code-block:: python

   # Plot Arctic time series
   fig = ds.climate_timeseries.plot_time_series(
       variable="air",
       latitude=slice(60, 90)
   )

Time Series Decomposition
-------------------------

.. code-block:: python

   # Decompose time series
   decomp = ds.climate_timeseries.decompose_time_series(
       variable="air",
       period=12  # Annual cycle
   )

Spatial Statistics
------------------

.. code-block:: python

   # Plot spatial standard deviation
   fig = ds.climate_timeseries.plot_std_space(
       variable="air"
   )

Performance Optimization
========================

Chunking for Large Datasets
---------------------------

.. code-block:: python

   # Basic chunking optimization
   ds_optimized = ds.climate_timeseries.optimize_chunks(
       target_mb=100,
       variable="air"
   )
   
   # Advanced chunking with custom strategies
   ds_advanced = ds.climate_timeseries.optimize_chunks_advanced(
       operation_type='timeseries',
       performance_priority='memory',
       variable="air"
   )

Chunking Analysis and Diagnostics
---------------------------------

.. code-block:: python

   # Print current chunking information
   ds.climate_timeseries.print_chunking_info(detailed=True)
   
   # Analyze chunking strategies
   ds.climate_timeseries.analyze_chunking_strategy(variable="air")
   
   # Optimize specifically for decomposition
   ds_decomp = ds.climate_timeseries.optimize_for_decomposition(
       variable="air"
   )

Memory-Efficient Workflows
--------------------------

.. code-block:: python

   # Complete workflow with optimization
   import xarray as xr
   import climate_diagnostics
   
   # Load large dataset
   ds = xr.open_dataset("large_climate_data.nc")
   
   # Optimize chunking for time series analysis
   ds_opt = ds.climate_timeseries.optimize_chunks_advanced(
       operation_type='timeseries',
       performance_priority='balanced',
       memory_limit_gb=8.0
   )
   
   # Perform analysis on optimized dataset
   fig = ds_opt.climate_timeseries.plot_time_series(
       variable="temperature",
       latitude=slice(60, 90)
   )
   
   # Decompose with optimized chunking
   decomp = ds_opt.climate_timeseries.decompose_time_series(
       variable="temperature",
       optimize_chunks=True
   )

Working with Regional Data
==========================

.. code-block:: python

   # Calculate regional mean using utilities
   from climate_diagnostics.utils import get_spatial_mean
   
   # Select region
   arctic_data = ds.sel(latitude=slice(60, 90))
   
   # Get mean time series
   arctic_ts = get_spatial_mean(arctic_data.air, area_weighted=True)
   
   # Plot using matplotlib
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 6))
   arctic_ts.plot()
   plt.title("Arctic Temperature")
   plt.show()

See Also
========

* :doc:`./trends` - Trend analysis methods
* :doc:`./plots` - Plotting functions
