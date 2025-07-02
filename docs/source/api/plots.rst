=============================
Plotting API Reference
=============================

The ``climate_plots`` accessor provides plotting capabilities for climate data visualization.

Overview
========

The Plots module extends xarray Datasets with a ``.climate_plots`` accessor that provides:

- Geographic visualization with automatic projections
- Statistical plots (mean, standard deviation, percentiles)
- Precipitation indices and extreme event analysis

Quick Example
=============

.. code-block:: python

   import xarray as xr
   import climate_diagnostics
   
   ds = xr.open_dataset("temperature_data.nc")
   
   # Plot mean temperature
   fig = ds.climate_plots.plot_mean(
       variable="air",
       title="Mean Temperature"
   )

Accessor Class
==============

.. autoclass:: climate_diagnostics.plots.plot.PlotsAccessor
   :members:
   :undoc-members:
   :show-inheritance:

Available Plotting Methods
==========================

Basic Statistical Plots
-----------------------

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_mean
   :no-index:

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_std_time
   :no-index:

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_percentile_spatial
   :no-index:

Precipitation Indices
---------------------

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_rx1day
   :no-index:

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_rx5day
   :no-index:

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_wet_spell_duration
   :no-index:

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_dry_spell_duration
   :no-index:

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_consecutive_wet_days
   :no-index:

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_consecutive_dry_days
   :no-index:

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_precipitation_above_threshold
   :no-index:

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_annual_sum_mean
   :no-index:

Basic Examples
==============

Temperature Plots
-----------------

.. code-block:: python

   # Plot mean temperature
   fig = ds.climate_plots.plot_mean(
       variable="air",
       title="Mean Air Temperature"
   )
   
   # Plot temperature variability
   fig = ds.climate_plots.plot_std_time(
       variable="air",
       title="Temperature Variability"
   )

Precipitation Analysis
----------------------

.. code-block:: python

   # Plot annual precipitation
   fig = ds.climate_plots.plot_annual_sum_mean(
       variable="prate",
       title="Annual Mean Precipitation"
   )
   
   # Plot extreme precipitation (RX1DAY)
   fig = ds.climate_plots.plot_rx1day(
       variable="prate",
       title="Maximum 1-Day Precipitation"
   )

Regional Focus
--------------

.. code-block:: python

   # Focus on a specific region
   fig = ds.climate_plots.plot_mean(
       variable="air",
       latitude=slice(20, 80),
       longitude=slice(-140, 40),
       title="North Atlantic Temperature"
   )

Customization Options
=====================

.. code-block:: python

   # Customize plot appearance
   fig = ds.climate_plots.plot_mean(
       variable="air",
       cmap="viridis",
       figsize=(12, 8),
       levels=20,
       land_only=True,
       save_plot_path="temperature_map.png"
   )

Working with Different Variables
================================

.. code-block:: python

   # Temperature data
   temp_fig = ds.climate_plots.plot_mean(variable="air")
   
   # Precipitation data (if available)
   if "prate" in ds.data_vars:
       precip_fig = ds.climate_plots.plot_annual_sum_mean(variable="prate")
       
   # Check available variables
   print("Available variables:", list(ds.data_vars))

See Also
========

* :doc:`./timeseries` - Time series analysis methods
* :doc:`./trends` - Trend analysis methods
