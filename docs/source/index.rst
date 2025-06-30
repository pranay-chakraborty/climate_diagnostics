=======================================
Climate Diagnostics Toolkit
=======================================

.. image:: _static/WeCLiMb_LOGO_1.png
   :align: center
   :alt: WeCLiMb Logo
   :width: 200px

.. image:: https://img.shields.io/badge/python-3.11+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://img.shields.io/badge/version-1.1-brightgreen.svg
   :alt: Version

.. image:: https://img.shields.io/badge/status-alpha-orange.svg
   :alt: Status

A Python toolkit for analyzing and visualizing climate data from model output, reanalysis, and observations. Built on xarray, it provides specialized accessors for time series analysis, trend calculation, and spatial plotting.

🌍 **Key Features**
==================

✨ **xarray Integration**
   Access features via ``.climate_plots``, ``.climate_timeseries``, and ``.climate_trends`` accessors on xarray Datasets.

📊 **Time Series Analysis** 
   Extract and analyze time series with spatial averaging, seasonal filtering, and STL decomposition.

🗺️ **Spatial Visualization**
   Create climate maps with Cartopy integration and automatic coordinate detection.

🔬 **Climate Indices**
   Calculate ETCCDI precipitation indices like Rx1day, Rx5day, wet/dry spell durations.

⚡ **Dask Support**
   Process large datasets efficiently with built-in Dask integration.

🚀 **Quick Start**
=================

.. code-block:: python

   import xarray as xr
   import climate_diagnostics

   # Open a climate dataset
   ds = xr.open_dataset("temperature_data.nc")

   # Create basic visualizations
   ds.climate_plots.plot_mean(variable="air")

   # Analyze time series
   ts = ds.climate_timeseries.plot_time_series(
       variable="air", 
       latitude=slice(30, 60)
   )

   # Calculate trends
   trend = ds.climate_trends.calculate_trend(
       variable="air"
   )

📚 **Documentation Contents**
============================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/plotting
   user_guide/time_series
   user_guide/trends
   user_guide/advanced

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/plots
   api/timeseries
   api/trends
   api/models
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic_plotting
   examples/trend_analysis
   examples/time_series_analysis
   examples/multi_model_evaluation

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

🔧 **Installation**
==================

**With pip:**

.. code-block:: bash

   pip install climate-diagnostics

**With conda (recommended):**

.. code-block:: bash

   conda env create -f environment.yml
   conda activate climate-diagnostics
   pip install -e .

📖 **Core Modules**
==================

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🗺️ Climate Plots
      :text-align: center

      Geographic visualizations with Cartopy integration.
      
      +++
      
      .. button-ref:: api/plots
         :expand:
         :color: primary
         :click-parent:
         
         Explore Plotting API

   .. grid-item-card:: 📈 Time Series Analysis
      :text-align: center

      Temporal analysis including decomposition and trend detection.
      
      +++
      
      .. button-ref:: api/timeseries
         :expand:
         :color: primary
         :click-parent:
         
         Explore TimeSeries API

   .. grid-item-card:: 📊 Trend Analysis
      :text-align: center

      Statistical trend calculation with visualization.
      
      +++
      
      .. button-ref:: api/trends
         :expand:
         :color: primary
         :click-parent:
         
         Explore Trends API

   .. grid-item-card:: 🔧 Utilities
      :text-align: center

      Helper functions for data processing and coordinates.
      
      +++
      
      .. button-ref:: api/utils
         :expand:
         :color: primary
         :click-parent:
         
         Explore Utils API

💡 **Quick Examples**
====================

**Create a Mean Temperature Map:**

.. code-block:: python

   # Load your data
   ds = xr.open_dataset("temperature_data.nc")
   
   # Plot mean with basic styling
   fig = ds.climate_plots.plot_mean(
       variable="air",
       title="Mean Temperature"
   )

**Analyze Temperature Trends:**

.. code-block:: python

   # Calculate spatial trends
   trends = ds.climate_trends.calculate_spatial_trends(
       variable="air",
       num_years=30
   )

**Time Series Analysis:**

.. code-block:: python

   # Extract regional time series
   regional_ts = ds.climate_timeseries.plot_time_series(
       variable="air",
       latitude=slice(30, 60)
   )
   
   # Perform decomposition
   decomp = ds.climate_timeseries.decompose_time_series(
       variable="air"
   )

🤝 **Contributing**
==================

We welcome contributions! Please see our `Contributing Guide <contributing.html>`_ for details on:

- Setting up a development environment
- Code style guidelines  
- Testing procedures
- Submitting pull requests

📧 **Support & Community**
=========================

- **Documentation**: You're reading it! 📚
- **Issues**: `GitHub Issues <https://github.com/yourusername/climate_diagnostics/issues>`_
- **Discussions**: `GitHub Discussions <https://github.com/yourusername/climate_diagnostics/discussions>`_

📄 **Citation**
==============

If you use this toolkit in your research, please cite:

.. code-block:: bibtex

   @software{climate_diagnostics_2025,
     title = {Climate Diagnostics Toolkit},
     author = {Chakraborty, Pranay and Muhammed, Adil I. K.},
     year = {2025},
     version = {1.1},
     url = {https://github.com/yourusername/climate_diagnostics}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

