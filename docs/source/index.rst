.. Climate Diagnostics Toolkit documentation master file, created by
   sphinx-quickstart on Sun Apr 13 22:09:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Climate Diagnostics Toolkit Documentation
=========================================

Welcome to the Climate Diagnostics Toolkit documentation!

Overview
--------

The Climate Diagnostics Toolkit is a comprehensive Python library designed for analyzing, processing, and visualizing climate data from various sources including model simulations, reanalysis products, and observational datasets. Built on top of xarray, this toolkit extends its functionality through specialized accessors that seamlessly integrate with your existing data workflows.

Key Features
------------

* **Temporal Analysis**: Robust tools for trend detection, time series decomposition, and variability analysis
* **Spatial Visualization**: Publication-quality maps with customized projections and spatial averaging capabilities
* **Statistical Diagnostics**: Advanced statistical methods tailored for climate science applications
* **Multi-model Analysis**: Tools for comparing and evaluating climate model outputs
* **Performance Optimization**: Support for Dask-powered parallel processing of large datasets

Installation
-------------

Via pip (recommended)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install climate-diagnostics

From Source
~~~~~~~~~~~

.. code-block:: bash

   git https://github.com/pranay-chakraborty/climate_diagnostics.git
   cd climate_diagnostics
   pip install -e .

Core Components
---------------

The toolkit provides three main xarray accessors:

* **climate_plots**: Geographic visualization with support for seasonal filtering, level selection, and area-weighted statistics
* **climate_timeseries**: Time series analysis including STL decomposition, spatial averaging, and temporal filtering
* **climate_trends**: Trend calculation and significance testing with robust visualization options

Getting Started
---------------

.. code-block:: python

   import xarray as xr
   from climate_diagnostics import accessors
   # Open a dataset
   ds = xr.open_dataset("/home/user/Downloads/air.mon.mean.nc")

   # Create a visualization
   ds.climate_plots.plot_mean(variable="air", season="djf")

   # Analyze trends
   ds.climate_trends.calculate_spatial_trends(
      variable="air", 
      num_years=10,
      latitude = slice(40,6),
      longitude = slice(60,110)
   
   )


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   API Reference <modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`