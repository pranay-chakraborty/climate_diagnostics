============
Installation
============

This guide covers the installation of the Climate Diagnostics Toolkit and its dependencies.

Prerequisites
=============

The Climate Diagnostics Toolkit requires:

- Python 3.11 or later
- NumPy, SciPy, pandas, xarray
- Matplotlib and Cartopy for plotting
- Dask for parallel processing and chunking
- netCDF4 and bottleneck for data handling
- statsmodels for statistical analysis

Environment Setup
=================

We strongly recommend using conda to manage dependencies, especially for geospatial packages like Cartopy.

Using Conda (Recommended)
-------------------------

1. **Clone the repository:**

.. code-block:: bash

   git clone https://github.com/pranay-chakraborty/climate_diagnostics.git
   cd climate_diagnostics

2. **Create the conda environment:**

.. code-block:: bash

   conda env create -f environment.yml

3. **Activate the environment:**

.. code-block:: bash

   conda activate climate-diagnostics

4. **Install the package in development mode:**

.. code-block:: bash

   pip install -e .

Using pip
---------

If you prefer pip, ensure you have the required system dependencies for Cartopy:

**On Ubuntu/Debian:**

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install libproj-dev proj-data proj-bin libgeos-dev

**On macOS (with Homebrew):**

.. code-block:: bash

   brew install proj geos

**Install the package:**

.. code-block:: bash

   pip install climate_diagnostics

Development Installation
========================

For development work:

1. **Fork and clone the repository:**

.. code-block:: bash

   git clone https://github.com/pranay-chakraborty/climate_diagnostics.git
   cd climate_diagnostics

2. **Create a development environment:**

.. code-block:: bash

   conda env create -f environment.yml
   conda activate climate-diagnostics

3. **Install in editable mode with development dependencies:**

.. code-block:: bash

   pip install -e ".[dev]"

4. **Install pre-commit hooks:**

.. code-block:: bash

   pre-commit install

Verification
============

Test your installation:

.. code-block:: python

   import climate_diagnostics
   import xarray as xr
   
   # Check version
   print(climate_diagnostics.__version__)
   
   # Test basic functionality
   ds = xr.tutorial.open_dataset("air_temperature")
   print("✅ Installation successful!")

Optional Dependencies
=====================

Additional packages for enhanced functionality:

**For Jupyter notebooks:**

.. code-block:: bash

   conda install jupyter ipywidgets

**For faster computations:**

.. code-block:: bash

   conda install dask distributed

**For advanced statistical analysis:**

.. code-block:: bash

   conda install scikit-learn statsmodels

Troubleshooting
===============

Common Issues
-------------

**Cartopy installation fails:**
   Use conda instead of pip for Cartopy and its dependencies.

**Import errors with GEOS/PROJ:**
   Ensure system libraries are installed (see pip section above).

**Memory issues with large datasets:**
   Install Dask: ``conda install dask``

**Plotting issues on headless systems:**
   Set the matplotlib backend: ``export MPLBACKEND=Agg``

Getting Help
------------

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/pranay-chakraborty/climate_diagnostics/issues>`_
2. Search `Stack Overflow <https://stackoverflow.com/questions/tagged/climate-diagnostics>`_
3. Open a new issue with:
   - Your OS and Python version
   - Full error traceback
   - Minimal example to reproduce the issue

System Requirements
===================

**Minimum:**
- Python 3.11+
- 4 GB RAM
- 1 GB free disk space

**Recommended:**
- Python 3.11+
- 16 GB RAM (for large datasets)
- SSD storage
- Multi-core CPU for parallel processing
