===============
Plotting Guide
===============

Learn to create climate visualizations with the Climate Diagnostics Toolkit.

🎨 **Overview**
================

The ``climate_plots`` accessor provides visualization capabilities designed for climate data. All plotting functions integrate with Cartopy for geographic projections and matplotlib for customization.

🗺️ **Geographic Plots**
=======================

**Basic Map Plotting**

.. code-block:: python

   import xarray as xr
   import climate_diagnostics

   ds = xr.open_dataset("temperature.nc")
   
   # Simple mean plot
   fig = ds.climate_plots.plot_mean(
       variable="temperature",
       title="Annual Mean Temperature"
   )

**Custom Projections**

.. code-block:: python

   # Different projections
   projections = ["PlateCarree", "Robinson", "Mollweide", "Orthographic"]
   
   for proj in projections:
       fig = ds.climate_plots.plot_mean(
           variable="temperature",
           projection=proj,
           title=f"Temperature - {proj} Projection"
       )

**Seasonal Analysis**

.. code-block:: python

   # Plot seasonal means
   seasons = ["DJF", "MAM", "JJA", "SON"]
   
   fig, axes = plt.subplots(2, 2, figsize=(15, 10))
   
   for i, season in enumerate(seasons):
       ax = axes.flat[i]
       ds.climate_plots.plot_mean(
           variable="temperature",
           season=season,
           ax=ax,
           title=f"{season} Mean Temperature"
       )

🎨 **Styling and Customization**
=================================

**Color Schemes**

.. code-block:: python

   # Built-in climate colormaps
   fig = ds.climate_plots.plot_mean(
       variable="air",
       colormap="RdBu_r",  # Red-Blue reversed
       levels=20,
       extend="both"
   )
   
   # Custom color levels
   levels = np.arange(-30, 31, 5)
   fig = ds.climate_plots.plot_mean(
       variable="air",
       levels=levels,
       colormap="coolwarm"
   )

**Geographic Features**

.. code-block:: python

   # Add geographic features
   fig = ds.climate_plots.plot_mean(
       variable="air",
       coastlines=True,
       borders=True,
       gridlines=True,
       ocean_color="lightblue",
       land_color="lightgray"
   )

**Annotations and Labels**

.. code-block:: python

   fig = ds.climate_plots.plot_mean(
       variable="air",
       title="Global Surface Temperature",
       colorbar_label="Temperature (°C)",
       units="°C",
       source="ERA5 Reanalysis"
   )

📊 **Statistical Overlays**
============================

**Significance Testing**

.. code-block:: python

   # Plot with significance stippling
   fig = ds.climate_plots.plot_mean(
       variable="air",
       significance_data=p_values,
       significance_level=0.05,
       stipple=True
   )

**Confidence Intervals**

.. code-block:: python

   # Show uncertainty
   fig = ds.climate_plots.plot_mean(
       variable="air",
       uncertainty=temperature_std,
       show_confidence=True
   )

📈 **Multiple Variables**
==========================

**Side-by-Side Comparison**

.. code-block:: python

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
   
   ds.climate_plots.plot_mean(
       variable="air",
       ax=ax1,
       title="Temperature"
   )
   
   ds.climate_plots.plot_mean(
       variable="prate", 
       ax=ax2,
       title="Precipitation"
   )

**Difference Plots**

.. code-block:: python

   # Calculate and plot differences
   diff = future_ds - historical_ds
   
   fig = diff.climate_plots.plot_mean(
       variable="air",
       colormap="RdBu_r",
       title="Temperature Change",
       center=0  # Center colormap at zero
   )

🎯 **Best Practices**
======================

.. tip::
   **Choose Appropriate Projections**
   
   - **Global data**: Robinson, Mollweide
   - **Regional data**: PlateCarree, Lambert Conformal
   - **Polar regions**: Orthographic, Stereographic

.. note::
   **Color Scheme Guidelines**
   
   - **Temperature**: Use diverging colormaps (RdBu_r, coolwarm)
   - **Precipitation**: Use sequential colormaps (Blues, viridis)
   - **Anomalies**: Center at zero with diverging colors

.. warning::
   **Performance Tips**
   
   - Use ``dask`` for large datasets
   - Consider downsampling for quick previews
   - Cache processed data when possible

🔧 **Advanced Techniques**
===========================

**Custom Colormaps**

.. code-block:: python

   from matplotlib.colors import LinearSegmentedColormap
   
   # Create custom colormap
   colors = ['blue', 'white', 'red']
   custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
   
   fig = ds.climate_plots.plot_mean(
       variable="air",
       colormap=custom_cmap
   )

**Subplot Layouts**

.. code-block:: python

   # Complex subplot arrangements
   fig = plt.figure(figsize=(20, 12))
   
   # Main plot
   ax_main = plt.subplot(2, 3, (1, 4))
   ds.climate_plots.plot_mean(variable="air", ax=ax_main)
   
   # Time series
   ax_ts = plt.subplot(2, 3, (2, 3))
   global_ts.plot(ax=ax_ts)
   
   # Regional plots
   for i, region in enumerate(regions):
       ax = plt.subplot(2, 3, 5+i)
       region_data.climate_plots.plot_mean(ax=ax)

🖼️ **Output Options**
=====================

**High-Resolution Output**

.. code-block:: python

   fig = ds.climate_plots.plot_mean(
       variable="air",
       figsize=(12, 8)
   )
   
   plt.savefig("temperature_map.png", dpi=300, bbox_inches="tight")

**Custom Styling**

.. code-block:: python

   # Custom matplotlib styling
   plt.rcParams.update({
       'font.size': 12,
       'font.family': 'sans-serif',
       'axes.linewidth': 1,
       'axes.spines.top': False,
       'axes.spines.right': False
   })
   
   fig = ds.climate_plots.plot_mean(variable="air")
