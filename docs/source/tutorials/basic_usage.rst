===============
Basic Usage
===============

This tutorial will get you started with the Climate Diagnostics Toolkit. You'll learn how to load data, access the toolkit's features, and create your first visualization.

🚀 **Getting Started**
======================

First, let's import the necessary libraries:

.. code-block:: python

   import xarray as xr
   import numpy as np
   import matplotlib.pyplot as plt
   
   # Import the climate diagnostics toolkit
   import climate_diagnostics

📊 **Loading Climate Data**
===========================

The toolkit works with xarray Datasets. Let's start with some sample data:

.. code-block:: python

   # Load a climate dataset (ERA5 example)
   ds = xr.open_dataset("era5_temperature.nc")
   
   # Or create sample data for this tutorial
   import numpy as np
   
   # Create sample temperature data
   lon = np.arange(0, 360, 2.5)
   lat = np.arange(-90, 91, 2.5)
   time = pd.date_range("2000-01-01", "2020-12-31", freq="MS")
   
   # Generate realistic temperature data
   temp_data = 15 + 20 * np.cos(np.radians(lat[None, :, None])) + \
               5 * np.random.randn(len(time), len(lat), len(lon))
   
   ds = xr.Dataset({
       "air": (["time", "lat", "lon"], temp_data)
   }, coords={
       "time": time,
       "lat": lat, 
       "lon": lon
   })

🔧 **Accessing Toolkit Features**
=================================

Once you have a Dataset, the toolkit provides three main accessors:

1. **climate_plots** - For visualizations
2. **climate_timeseries** - For temporal analysis  
3. **climate_trends** - For trend calculations

.. code-block:: python

   # Check available methods
   print(dir(ds.climate_plots))
   print(dir(ds.climate_timeseries))
   print(dir(ds.climate_trends))

� **Your First Plot**
======================

Let's create a simple temperature map:

.. code-block:: python

   # Plot the mean temperature
   fig = ds.climate_plots.plot_mean(
       variable="air",
       title="Global Mean Temperature"
   )
   plt.show()

📈 **Time Series Analysis**
===========================

Extract and plot a time series:

.. code-block:: python

   # Plot regional time series
   ts = ds.climate_timeseries.plot_time_series(
       variable="air",
       latitude=slice(30, 60),
       longitude=slice(-120, -80)
   )

📊 **Trend Analysis**
======================

Calculate and visualize trends:

.. code-block:: python

   # Calculate trend for a region
   trend = ds.climate_trends.calculate_trend(
       variable="air",
       latitude=slice(40, 50),
       longitude=slice(-100, -90)
   )
   
   print(f"Temperature trend: {trend.values:.3f} units/year")

🎯 **Key Concepts**
====================

.. note::
   **xarray Integration**: All toolkit features are accessed through xarray accessor methods (.climate_plots, .climate_timeseries, .climate_trends)

.. tip::
   **Data Requirements**: Your data should have coordinate dimensions named 'lat'/'latitude', 'lon'/'longitude', and 'time' for optimal compatibility.

.. warning::
   **Memory Management**: For large datasets, consider using Dask arrays or chunking your data.

🔄 **Next Steps**
==================

Now that you've learned the basics, you're ready to:

- :doc:`../user_guide/plotting` - Learn advanced plotting techniques
- :doc:`../api/index` - Explore the complete API reference

💡 **Common Patterns**
=======================

Here are some common usage patterns you'll use frequently:

.. code-block:: python

   # Seasonal analysis
   winter_mean = ds.sel(time=ds.time.dt.season == "DJF").mean("time")
   
   # Regional subset
   arctic = ds.sel(lat=slice(60, 90))
   
   # Multi-variable analysis
   for var in ["air", "prate"]:
       if var in ds.data_vars:
           fig = ds.climate_plots.plot_mean(variable=var)
           plt.show()
