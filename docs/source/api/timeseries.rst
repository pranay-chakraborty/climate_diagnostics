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
================

Time Series Plotting
-------------------

.. automethod:: climate_diagnostics.TimeSeries.TimeSeries.TimeSeriesAccessor.plot_time_series

Statistical Analysis
-------------------

.. automethod:: climate_diagnostics.TimeSeries.TimeSeries.TimeSeriesAccessor.plot_std_space

Decomposition Methods
--------------------

.. automethod:: climate_diagnostics.TimeSeries.TimeSeries.TimeSeriesAccessor.decompose_time_series

Basic Examples
==============

Simple Time Series Plot
-----------------------

.. code-block:: python

   # Plot global mean time series
   fig = ds.climate_timeseries.plot_time_series(
       variable="air",
       title="Global Mean Temperature"
   )

Regional Time Series
-------------------

.. code-block:: python

   # Plot Arctic time series
   fig = ds.climate_timeseries.plot_time_series(
       variable="air",
       latitude=slice(60, 90),
       title="Arctic Temperature"
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
-----------------

.. code-block:: python

   # Plot spatial standard deviation
   fig = ds.climate_timeseries.plot_std_space(
       variable="air"
   )

Working with Regional Data
=========================

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

* :doc:`trends` - Trend analysis methods
* :doc:`plots` - Plotting functions
