==========
Quickstart
==========

This guide will get you up and running with the Climate Diagnostics Toolkit in minutes.

Basic Setup
===========

After installation, import the toolkit and load some sample data:

.. code-block:: python

   import xarray as xr
   import climate_diagnostics
   import matplotlib.pyplot as plt
   
   # Load sample data (ERA5 temperature)
   ds = xr.tutorial.open_dataset("air_temperature")
   print(ds)

The toolkit extends xarray with three main accessors:

- ``.climate_plots`` - Geographic visualizations
- ``.climate_timeseries`` - Time series analysis  
- ``.climate_trends`` - Trend analysis

Performance Optimization
========================

For large datasets, optimize chunking for better performance:

.. code-block:: python

   # Check current chunking status
   ds.climate_timeseries.print_chunking_info()
   
   # Get chunking recommendations
   recommendations = ds.climate_timeseries.analyze_chunking_strategy()
   
   # Optimize chunks for time series operations
   ds_optimized = ds.climate_timeseries.optimize_chunks_advanced(
       operation_type='timeseries',
       performance_priority='memory'  # 'speed', 'memory', or 'balanced'
   )
   
   # For spatial operations
   ds_spatial = ds.climate_trends.optimize_for_trends(
       performance_priority='speed'
   )

First Visualization
===================

Create your first climate visualization:

.. code-block:: python

   # Plot the temporal mean
   fig = ds.climate_plots.plot_mean(
       variable="air",
       title="Mean Air Temperature"
   )
   plt.show()

This creates a global map with:

- Automatic projection and coordinate detection
- Coastlines and geographic features
- Colorbar with proper units
- Clean styling

Time Series Analysis
====================

Extract and analyze time series data:

.. code-block:: python

   # Create a time series plot for spatial mean
   fig = ds.climate_timeseries.plot_time_series(
       variable="air",
       latitude=slice(30, 60),  # Northern mid-latitudes
       longitude=slice(-180, 180),  # Global
       title="Northern Mid-Latitude Mean Air Temperature"
   )
   plt.show()

Seasonal Analysis
=================

Analyze seasonal patterns:

.. code-block:: python

   # Calculate seasonal means
   seasonal = ds.groupby("time.season").mean("time")
   
   # Plot all seasons
   import cartopy.crs as ccrs
   fig, axes = plt.subplots(2, 2, figsize=(15, 10), 
                           subplot_kw={'projection': ccrs.PlateCarree()})
   
   seasons = ["DJF", "MAM", "JJA", "SON"]
   for i, season in enumerate(seasons):
       ax = axes.flat[i]
       seasonal.sel(season=season).climate_plots.plot_mean(
           variable="air",
           ax=ax,
           title=f"{season} Mean Temperature"
       )

Trend Analysis
==============

Calculate and visualize trends:

.. code-block:: python

   # Calculate linear trends over the full period
   # This method plots the trends automatically when plot_map=True (default)
   trends = ds.climate_trends.calculate_spatial_trends(
       variable="air",
       num_years=1,  # Trend per year
       plot_map=True  # Shows the trend map
   )
   
   # The trends variable contains the computed trend values
   print(f"Trend data shape: {trends.shape}")
   print(f"Mean global trend: {trends.mean().values:.4f} K/year")

Time Series Decomposition
=========================

Decompose time series into components:

.. code-block:: python

   # Perform STL decomposition on a spatial average
   decomp = ds.climate_timeseries.decompose_time_series(
       variable="air",
       latitude=slice(90, 60),  # Arctic region
       longitude=slice(-180, 180),
       period=12  # Annual cycle
   )
   
   # The decomposition returns a figure
   plt.show()

Advanced Features
=================

Regional Statistics
-------------------

Calculate statistics for predefined regions:

.. code-block:: python

   # Define custom regions
   regions = {
       "Arctic": {"latitude": slice(90, 60)},
       "Tropics": {"latitude": slice(23.5, -23.5)},
       "Antarctic": {"latitude": slice(-60, -90)}
   }
   
   # Calculate regional means using xarray operations
   regional_stats = {}
   for name, bounds in regions.items():
       regional_data = ds.sel(**bounds)
       # Calculate spatial mean for the region
       regional_stats[name] = regional_data.mean(["lat", "lon"])
       
   # Plot regional time series
   plt.figure(figsize=(12, 6))
   for name, data in regional_stats.items():
       data.air.plot(label=name, alpha=0.8)
   plt.legend()
   plt.title("Regional Temperature Time Series")
   plt.ylabel("Temperature (K)")
   plt.grid(True, alpha=0.3)

Multi-Model Comparison
----------------------

Compare multiple datasets:

.. code-block:: python

   # Load multiple datasets (example with different models)
   models = {
       "ERA5": xr.tutorial.open_dataset("air_temperature"),
       "Model1": xr.tutorial.open_dataset("air_temperature"),  # Replace with actual data
   }
   
   # Calculate global means for each model
   model_ts = {}
   for name, data in models.items():
       # Calculate global spatial mean
       model_ts[name] = data.air.mean(["lat", "lon"])
   
   # Plot comparison
   plt.figure(figsize=(12, 6))
   for name, ts in model_ts.items():
       ts.plot(label=name, alpha=0.8)
   plt.legend()
   plt.title("Multi-Model Temperature Comparison")
   plt.ylabel("Temperature (K)")
   plt.grid(True, alpha=0.3)

Best Practices
==============

Memory Management
-----------------

For large datasets, use chunking:

.. code-block:: python

   # Open with chunks for better memory management
   ds_chunked = xr.open_dataset(
       "large_file.nc",
       chunks={"time": 100, "lat": 50, "lon": 50}
   )

Data Preprocessing
------------------

Standardize your data:

.. code-block:: python

   # Convert units if needed
   if ds.air.attrs.get("units") == "K":
       ds["air_celsius"] = ds.air - 273.15
       ds.air_celsius.attrs["units"] = "°C"
   
   # Set time coordinate
   if "time" in ds.coords:
       ds = ds.sel(time=slice("1980", "2020"))

Performance Tips
================

1. **Use chunking** for large datasets
2. **Subset data** before analysis when possible
3. **Use Dask** for parallel computation
4. **Cache results** for repeated analysis

.. code-block:: python

   # Enable Dask for parallel processing
   import dask
   with dask.config.set(scheduler='threads'):
       result = ds.climate_trends.calculate_spatial_trends(variable="air")

Next Steps
==========

Now that you've seen the basics:

1. **Explore the API**: Check out the :doc:`api/index` for detailed function documentation
2. **Try tutorials**: Work through :doc:`tutorials/index` for in-depth examples  
3. **Read the user guide**: Learn advanced techniques in :doc:`user_guide/index`
4. **Join the community**: Get help and share your work

Common Patterns
===============

Here are some common analysis patterns:

**Climate Anomalies:**

.. code-block:: python

   # Calculate anomalies relative to climatology
   climatology = ds.groupby("time.month").mean("time")
   anomalies = ds.groupby("time.month") - climatology

**Seasonal Cycles:**

.. code-block:: python

   # Analyze seasonal cycle
   seasonal_cycle = ds.groupby("time.month").mean("time")
   seasonal_cycle.climate_plots.plot_cycle(variable="air")

**Extreme Events:**

.. code-block:: python

   # Identify extreme values
   percentiles = ds.quantile([0.05, 0.95], dim="time")
   extremes = ds.where((ds < percentiles.sel(quantile=0.05)) | 
                       (ds > percentiles.sel(quantile=0.95)))

Need Help?
==========

- 📖 **Documentation**: You're reading it!
- 🐛 **Issues**: `GitHub Issues <https://github.com/pranay-chakraborty/climate_diagnostics/issues>`_
- 💬 **Discussions**: `GitHub Discussions <https://github.com/pranay-chakraborty/climate_diagnostics/discussions>`_
