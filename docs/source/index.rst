.. Climate Diagnostics Toolkit documentation master file

.. image:: https://img.shields.io/badge/Python-3.11+-blue.svg
   :alt: Python 3.11+

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :alt: MIT License

Climate Diagnostics Toolkit
==========================

*A Python package for analyzing and visualizing climate data with a focus on time series analysis and spatial pattern visualization.*

Overview
--------

Climate Diagnostics Toolkit provides powerful tools to process, analyze, and visualize climate data from NetCDF files. 
The package offers robust functionality for seasonal filtering, spatial averaging with proper area weighting, 
trend analysis, and time series decomposition to help climate scientists extract meaningful insights from large climate datasets.

Key Features
-----------

**Data Processing**

* Load and process NetCDF climate data files with automatic chunking using dask
* Filter data by meteorological seasons (Annual, DJF, MAM, JJAS)
* Select data by latitude, longitude, level, and time range

**Analysis**

* Calculate area-weighted spatial averages that account for grid cell sizes
* Compute time series statistics with proper spatial weighting
* Decompose time series into trend, seasonal, and residual components
* Extract and analyze trends with statistical significance testing

**Visualization**

* Generate publication-quality spatial maps with Cartopy projections
* Create time series plots with proper metadata and formatting
* Visualize trends and statistical distributions

Quick Start
----------

Installation
^^^^^^^^^^^

.. code-block:: bash

   pip install climate-diagnostics-toolkit

Basic Usage
^^^^^^^^^^

.. code-block:: python

   from climate_diagnostics import TimeSeries, Plots, Trends

   # Load data
   ts = TimeSeries("path/to/climate_data.nc")
   
   # Plot time series for a specific region
   ts.plot_time_series(
       latitude=slice(-30, 30),
       longitude=slice(0, 180),
       level=850,
       season="jjas"  # June-July-August-September
   )
   
   # Calculate and visualize trends
   trends = Trends("path/to/climate_data.nc")
   results = trends.calculate_trend(
       variable="air",
       latitude=slice(-30, 30),
       season="annual"
   )

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   installation
   quickstart
   user_guide/index
   examples/index
   
.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/modules
   api/climate_diagnostics
   
.. toctree::
   :maxdepth: 1
   :caption: Development
   
   contributing
   changelog
   license

Indices and Tables
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. note::
   This package is designed for climate scientists and researchers working with large meteorological datasets.
   
.. warning::
   Handling large datasets may require significant memory. Use the dask-backed functions for efficient processing.
   
.. admonition:: Citation
   :class: important
   
   If you use Climate Diagnostics Toolkit in your research, please cite:
   
   Chakraborty, P. (2025). Climate Diagnostics Toolkit: Tools for analyzing and visualizing climate data.