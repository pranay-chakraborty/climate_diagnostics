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

Basic Examples
==============

Comprehensive Analysis Workflow
----------------------------------

This example demonstrates a complete workflow, from optimizing data chunks to decomposition and visualization.

.. code-block:: python

   import xarray as xr
   import matplotlib.pyplot as plt
   import climate_diagnostics

   # Load a sample dataset
   ds = xr.tutorial.load_dataset("air_temperature")

   # 1. Create a time series plot
   fig = ds.climate_timeseries.plot_time_series(
       variable="air",
       latitude=slice(30, 60),
       longitude=slice(-120, -60)
   )
   # 2. Decompose the time series for a specific region
   decomposition = ds.climate_timeseries.decompose_time_series(
       variable="air",
       latitude=slice(30, 40),
       longitude=slice(-100, -90)
   )

   # 3. Plot the original and decomposed time series components
   fig, ax = plt.subplots(figsize=(12, 8))
   decomposition['original'].plot(ax=ax, label="Original")
   decomposition['trend'].plot(ax=ax, label="Trend")
   decomposition['seasonal'].plot(ax=ax, label="Seasonal")
   ax.legend()
   ax.set_title("Time Series Decomposition")
   plt.show()

      # 4. Analyze spatial standard deviation of the original data
   fig_std = ds.climate_timeseries.plot_std_space(
       variable="air",
       title="Spatial Standard Deviation of Air Temperature"
   )
   plt.show()

Working with Regional Data
==========================
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
