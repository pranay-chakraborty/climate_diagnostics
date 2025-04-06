.. Climate Diagnostics Toolkit documentation master file

.. image:: https://img.shields.io/badge/Python-3.11+-blue.svg
   :alt: Python 3.11+
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :alt: MIT License
   :target: license.html

.. image:: https://img.shields.io/badge/Documentation-Latest-brightgreen.svg
   :alt: Documentation Status
   :target: https://climate-diagnostics-toolkit.readthedocs.io/

=====================
Climate Diagnostics Toolkit
=====================

*A comprehensive Python package for analyzing and visualizing climate data with powerful capabilities for time series analysis, trend detection, and spatial pattern visualization.*

Overview
--------

Climate Diagnostics Toolkit provides robust tools to process, analyze, and visualize climate data from NetCDF files.
Designed specifically for climate scientists and researchers, this package offers sophisticated functionality for 
seasonal filtering, spatial averaging with proper area weighting, trend analysis, and time series decomposition
to extract meaningful insights from large meteorological datasets.

Key Features
-----------

**Data Processing**

* Load and process NetCDF climate data files with automatic chunking using dask for memory efficiency
* Filter data by meteorological seasons (Annual, DJF, MAM, JJA, JJAS, SON)
* Flexible selection by latitude, longitude, pressure level, and time range
* Automatic handling of coordinate naming conventions for maximum compatibility

**Analysis**

* Calculate area-weighted spatial averages that properly account for decreasing grid cell area toward the poles
* Compute time series statistics with sophisticated spatial weighting algorithms
* Decompose time series into trend, seasonal, and residual components using STL (Seasonal-Trend decomposition using LOESS)
* Extract and analyze trends with proper statistical significance testing (p-values, confidence intervals)

**Visualization**

* Generate publication-quality spatial maps with Cartopy projections and geographical features
* Create informative time series plots with automatic metadata handling and formatting
* Visualize trends, statistical distributions, and spatial variability patterns
* Support for customizing all visualization aspects using familiar matplotlib interfaces

Core Components
--------------

The toolkit provides three primary modules:

* ``TimeSeries`` - For extracting, analyzing and decomposing time series from climate data
* ``Plots`` - For spatial visualization and mapping of climate variables
* ``Trends`` - For calculating trends and assessing their statistical significance

Quick Start
----------

Installation
^^^^^^^^^^^

.. code-block:: bash

   pip install climate-diagnostics-toolkit

   # For development installation with additional dependencies:
   git clone https://github.com/pranay-chakraborty/climate_diagnostics.git
   cd climate-diagnostics
   pip install -e ".[dev]"

Basic Usage
^^^^^^^^^^

.. code-block:: python

   from climate_diagnostics import TimeSeries, Plots, Trends

   # Load data
   ts = TimeSeries("path/to/climate_data.nc")
   plots = Plots("path/to/climate_data.nc")
   trends = Trends("path/to/climate_data.nc")
   
   # Visualize spatial patterns
   plots.plot_mean(
       variable="air",
       level=850,
       season="djf"  # December-January-February
   )
   
   # Plot time series for a specific region
   ts.plot_time_series(
       latitude=slice(-30, 30),     # 30°S to 30°N
       longitude=slice(0, 180),     # 0° to 180°E
       level=850,
       variable="air",
       season="jjas"                # June-July-August-September
   )
   
   # Calculate and visualize trends
   trends.calculate_trend(
       variable="air",
       latitude=slice(-30, 30),
       season="annual",
       plot=True                    # Generate visualization
   )

Documentation
------------

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
