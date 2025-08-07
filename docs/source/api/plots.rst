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
   
   # Plot mean temperature with a specific projection
   fig = ds.climate_plots.plot_mean(
       variable="air",
       title="Mean Temperature",
       projection="Robinson"
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

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_std
   :no-index:

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_percentile
   :no-index:

Precipitation Indices
---------------------

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_prcptot
   :no-index:

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_rx1day
   :no-index:

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_sdii
   :no-index:

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_cdd
   :no-index:

Temperature and Extreme Indices
------------------------------

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_days_above_threshold
   :no-index:

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_wsdi
   :no-index:

.. automethod:: climate_diagnostics.plots.plot.PlotsAccessor.plot_csdi
   :no-index:

Basic Examples
==============

Comprehensive Plotting Workflow
--------------------------------

This example demonstrates a complete workflow for creating various climate data visualizations, from basic statistical plots to specialized precipitation indices.

.. code-block:: python

   import xarray as xr
   import matplotlib.pyplot as plt
   import climate_diagnostics
   import numpy as np

   # Load a sample dataset
   ds = xr.tutorial.load_dataset("air_temperature")

   # Create a dummy precipitation variable for demonstration purposes
   ds['pr'] = np.abs(ds.air.data - 273.15) / 10
   ds['pr'].attrs['units'] = 'mm/day'


   # 1. Plot mean temperature with a custom projection and colormap
   fig1 = ds.climate_plots.plot_mean(
       variable="air",
       title="Mean Air Temperature",
       projection="Robinson",
       cmap="viridis"
   )
   plt.show()

   # 2. Plot the maximum 1-day precipitation (a common climate index)
   fig2 = ds.climate_plots.plot_rx1day(
       variable="pr",
       title="Maximum 1-Day Precipitation (Rx1day)",
       projection="PlateCarree"
   )
   plt.show()

   # 3. Plot the 95th percentile of spatial temperature data
   fig3 = ds.climate_plots.plot_percentile(
       variable="air",
       percentile=95,
       title="95th Percentile of Air Temperature"
   )
   plt.show()

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
       precip_fig = ds.climate_plots.plot_prcptot(variable="prate")
       
   # Check available variables
   print("Available variables:", list(ds.data_vars))

See Also
========

* :doc:`./timeseries` - Time series analysis methods
* :doc:`./trends` - Trend analysis methods
