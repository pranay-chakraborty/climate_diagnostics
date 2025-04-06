Quickstart Guide
===============

This quickstart guide will help you get started with the Climate Diagnostics Toolkit, showing you how to perform common climate data analysis tasks with minimal code.

Installation
-----------

From Source
~~~~~~~~~~

To install the latest development version from source:

.. code-block:: bash

   git clone https://github.com/pranay-chakraborty/climate_diagnostics.git
   pip install -e .

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~

For development purposes, install with additional development dependencies:

.. code-block:: bash

   git clone https://github.com/pranay-chakraborty/climate_diagnostics.git
   cd climate-diagnostics
   pip install -e ".[dev]"

Basic Usage
----------

The toolkit provides three main components:

* ``Plots`` - For spatial visualization of climate data
* ``TimeSeries`` - For time series analysis and decomposition
* ``Trends`` - For trend analysis and significance testing

Loading Data
-----------

All components can read NetCDF files directly:

.. code-block:: python

   from climate_diagnostics import Plots, TimeSeries, Trends

   # Initialize with a NetCDF file
   plots = Plots("path/to/data.nc")
   ts = TimeSeries("path/to/data.nc")
   trends = Trends("path/to/data.nc")

Spatial Visualization
--------------------

Create maps of climate variables with proper projections:

.. code-block:: python

   # Plot mean temperature at 850 hPa for winter (DJF)
   plots.plot_mean(
       variable="air",
       level=850,
       season="djf"
   )

   # Focus on a specific region (South Asia)
   plots.plot_mean(
       variable="precip",
       latitude=slice(35, 5),     # 5°N to 35°N
       longitude=slice(65, 95),   # 65°E to 95°E
       season="jjas"              # Summer monsoon season
   )

Time Series Analysis
-------------------

Analyze and visualize time series data:

.. code-block:: python

   # Extract and plot a time series for a region
   ts.plot_time_series(
       variable="air",
       latitude=slice(-10, 10),   # 10°S to 10°N
       longitude=slice(160, 220), # 160°E to 220°E (Central Pacific)
       level=850,
       season="annual"
   )

   # Decompose a time series into trend, seasonal, and residual components
   ts.decompose_time_series(
       variable="air",
       latitude=slice(-5, 5),     # 5°S to 5°N
       longitude=slice(190, 240), # 190°E to 240°E (Niño 3.4 region)
       season="annual",
       stl_period=12,
       area_weighted=True
   )

Trend Analysis
-------------

Calculate and visualize trends in climate data:

.. code-block:: python

   # Calculate linear trend and statistical significance
   results = trends.calculate_trend(
       variable="air",
       level=850,
       time_range=slice("1980", "2020"),
       season="annual",
       frequency="M",
       period=12,
       area_weighted=True,
       plot=True,
       return_results=True
   )
   
   # Access trend statistics
   print(f"Trend slope: {results['trend_statistics'].loc['slope', 'value']}")
   print(f"P-value: {results['trend_statistics'].loc['p_value', 'value']}")
   
   # Check if trend is significant (p < 0.05)
   p_value = results['trend_statistics'].loc['p_value', 'value']
   print(f"Trend is {'significant' if p_value < 0.05 else 'not significant'}")

Analyzing Spatial Variability
----------------------------

Visualize standard deviation over time to identify regions of high variability:

.. code-block:: python

   # Plot standard deviation of temperature over time
   plots.plot_std_time(
       variable="air",
       latitude=slice(-60, 60),  # 60°S to 60°N
       level=850,
       season="djf"
   )
   
   # Plot spatial standard deviation time series
   ts.plot_std_space(
       variable="precip",
       latitude=slice(0, 45),     # 0° to 45°N
       longitude=slice(60, 120),  # 60°E to 120°E
       season="jjas"
   )

Working with Dask for Big Data
------------------------------

The Climate Diagnostics Toolkit automatically handles large datasets using Dask:

.. code-block:: python

   plots = Plots(
       "path/to/large_dataset.nc"
   )

   # Operations will now use Dask's parallel computing capabilities , the dataset is auto-chunked by default
   plots.plot_mean(variable="air", level=850)

Customizing Plots
----------------

All plotting functions return matplotlib Axes objects that you can further customize:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Create a plot and customize it
   fig = plt.figure(figsize=(12, 8))
   ax = plots.plot_mean(
       variable="air",
       level=850,
       season="djf"
   )

   # Add title, customize colorbar, etc.
   ax.set_title("Winter (DJF) Mean Temperature at 850 hPa", fontsize=16)
   plt.tight_layout()
   plt.savefig("temperature_map.png", dpi=300)

Next Steps
---------

Check the :ref:`user-guide` for more detailed explanations and advanced usage examples.