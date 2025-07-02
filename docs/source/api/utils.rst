==========================
Utilities API Reference
==========================

The utilities module provides helper functions for climate data processing.

Overview
========

The utilities include:

- Coordinate name detection and validation
- Data selection and processing
- Spatial averaging functions
- Dask client management
- **Advanced chunking strategies and optimization**

Available Functions
===================

Coordinate Utilities
====================

.. automodule:: climate_diagnostics.utils.coord_utils
   :members:

Data Utilities
==============

.. automodule:: climate_diagnostics.utils.data_utils
   :members:

Dask Utilities
==============

.. automodule:: climate_diagnostics.utils.dask_utils
   :members:

Chunking Utilities
==================

.. automodule:: climate_diagnostics.utils.chunking_utils
   :members:

Basic Examples
==============

Working with Coordinates
------------------------

.. code-block:: python

   from climate_diagnostics.utils import get_coord_name
   
   # Find coordinate names automatically
   time_coord = get_coord_name(ds, ['time', 't'])
   lat_coord = get_coord_name(ds, ['lat', 'latitude'])
   
   print(f"Time coordinate: {time_coord}")
   print(f"Latitude coordinate: {lat_coord}")

Spatial Averaging
-----------------

.. code-block:: python

   from climate_diagnostics.utils import get_spatial_mean
   
   # Calculate area-weighted spatial mean
   global_mean = get_spatial_mean(ds.air, area_weighted=True)
   
   # Simple spatial mean (no area weighting)
   simple_mean = get_spatial_mean(ds.air, area_weighted=False)

Data Selection
--------------

.. code-block:: python

   from climate_diagnostics.utils import select_process_data
   
   # Select and process data with automatic coordinate handling
   processed_data = select_process_data(
       ds,
       variable="air",
       latitude=slice(30, 60),
       longitude=slice(-10, 40),
       season="jja"  # Summer season
   )

Advanced Chunking
-----------------

.. code-block:: python

   from climate_diagnostics.utils import (
       dynamic_chunk_calculator, 
       suggest_chunking_strategy,
       print_chunking_info
   )
   
   # Analyze current chunking
   print_chunking_info(ds, detailed=True)
   
   # Get chunking recommendations
   strategy = suggest_chunking_strategy(
       ds, 
       operation_type='timeseries',
       target_mb=100
   )
   
   # Calculate optimal chunks dynamically
   optimal_chunks = dynamic_chunk_calculator(
       ds,
       operation_type='spatial',
       performance_priority='speed'
   )
   
   # Select and process data with automatic coordinate handling
   processed_data = select_process_data(
       ds,
       variable="air",
       latitude=slice(30, 60),
       longitude=slice(-10, 40),
       season="jja"  # Summer season
   )

Seasonal Filtering
------------------

.. code-block:: python

   from climate_diagnostics.utils import filter_by_season
   
   # Filter data by season
   summer_data = filter_by_season(ds, season="jja")
   winter_data = filter_by_season(ds, season="djf")

Practical Usage
===============

Complete Analysis Workflow
--------------------------

.. code-block:: python

   import xarray as xr
   from climate_diagnostics.utils import (
       get_coord_name, 
       select_process_data, 
       get_spatial_mean
   )
   
   # Load data
   ds = xr.open_dataset("temperature_data.nc")
   
   # Check coordinates
   time_coord = get_coord_name(ds, ['time', 't'])
   print(f"Time coordinate found: {time_coord}")
   
   # Select and process regional data
   arctic_data = select_process_data(
       ds,
       variable="air", 
       latitude=slice(60, 90),
       season="annual"
   )
   
   # Calculate regional mean
   arctic_mean = get_spatial_mean(arctic_data, area_weighted=True)
   
   # Plot results
   import matplotlib.pyplot as plt
   arctic_mean.plot()
   plt.title("Arctic Mean Temperature")
   plt.show()

Memory-Efficient Processing
---------------------------

.. code-block:: python

   from climate_diagnostics.utils import get_or_create_dask_client
   
   # Ensure Dask client is available for large datasets
   client = get_or_create_dask_client()
   
   # Process large dataset
   large_ds = xr.open_dataset("large_file.nc", chunks={'time': 100})
   result = get_spatial_mean(large_ds.air, area_weighted=True)
   
   # Compute result
   computed_result = result.compute()

See Also
========

* :doc:`./timeseries` - Time series analysis methods
* :doc:`./trends` - Trend analysis methods
* :doc:`./plots` - Plotting functions
