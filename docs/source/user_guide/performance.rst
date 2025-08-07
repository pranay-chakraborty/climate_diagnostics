============================
Performance Optimization
============================

This guide covers performance optimization techniques for the Climate Diagnostics Toolkit, focusing on manual chunking strategies for large climate datasets.

Overview
========

For large climate datasets, proper chunking is essential for good performance:

- **Manual chunking** based on your data structure and analysis needs
- **Memory management** by choosing appropriate chunk sizes
- **Dask integration** for out-of-core processing

Quick Performance Tips
=======================

1. **Use appropriate chunk sizes** for your data and available memory
2. **Chunk time dimensions** to balance memory usage and computation efficiency  
3. **Consider your analysis type** when choosing spatial chunk sizes
4. **Monitor memory usage** during large computations

Manual Chunking Strategies
===========================

Basic Manual Chunking
----------------------

For most climate analyses, start with manual chunking:

.. code-block:: python

   import xarray as xr
   import climate_diagnostics
   
   # Load your dataset
   ds = xr.open_dataset("large_climate_data.nc")
   
   # Apply basic manual chunking
   ds_chunked = ds.chunk({
       'time': 120,    # ~10 years of monthly data
       'lat': 50,      # 50 latitude points per chunk
       'lon': 50       # 50 longitude points per chunk
   })

Time Series Analysis Chunking
-----------------------------

For time series analysis, optimize temporal chunks:

.. code-block:: python

   # Optimize for time series operations
   ds_timeseries = ds.chunk({
       'time': 240,    # Larger time chunks for time series
       'lat': -1,      # Keep full spatial extent
       'lon': -1
   })

Spatial Analysis Chunking  
-------------------------

For spatial analysis, optimize spatial chunks:

.. code-block:: python

   # Optimize for spatial operations
   ds_spatial = ds.chunk({
       'time': 12,     # Smaller time chunks
       'lat': 90,      # Larger spatial chunks
       'lon': 180
   })

Analysis-Specific Recommendations
=================================

Time Series Decomposition
-------------------------

.. code-block:: python

   # Good chunking for decomposition
   ds_decomp = ds.chunk({'time': 360, 'lat': 20, 'lon': 20})
   
   # Perform decomposition
   result = ds_decomp.climate_timeseries.decompose_time_series(
       variable="temperature",
       latitude=slice(30, 60)
   )

Trend Analysis
--------------

.. code-block:: python

   # Good chunking for trend calculation
   ds_trend = ds.chunk({'time': -1, 'lat': 30, 'lon': 30})
   
   # Calculate trends
   trends = ds_trend.climate_trends.calculate_spatial_trends(
       variable="temperature",
       frequency="Y"
   )

Plotting and Visualization
--------------------------

.. code-block:: python

   # Balanced chunking for plotting
   ds_plot = ds.chunk({'time': 60, 'lat': 40, 'lon': 40})
   
   # Create plots
   ds_plot.climate_plots.plot_mean(variable="temperature")

Memory Management
=================

Monitor Memory Usage
--------------------

.. code-block:: python

   # Check chunk sizes
   print(f"Chunk sizes: {ds_chunked.chunks}")
   
   # Estimate memory usage
   chunk_size_mb = ds_chunked.temperature.nbytes / (1024**2)
   print(f"Estimated chunk size: {chunk_size_mb:.1f} MB")

Adaptive Chunking
-----------------

Adjust chunks based on your system:

.. code-block:: python

   import psutil
   
   # Get available memory
   available_memory_gb = psutil.virtual_memory().available / (1024**3)
   print(f"Available memory: {available_memory_gb:.1f} GB")
   
   # Adjust chunk size accordingly
   if available_memory_gb > 16:
       time_chunk = 240
       spatial_chunk = 60
   elif available_memory_gb > 8:
       time_chunk = 120
       spatial_chunk = 40
   else:
       time_chunk = 60
       spatial_chunk = 20
   
   ds_adaptive = ds.chunk({
       'time': time_chunk,
       'lat': spatial_chunk, 
       'lon': spatial_chunk
   })

Using Chunking Utilities
------------------------

The library provides utilities for chunking analysis:

.. code-block:: python

   from climate_diagnostics.utils import print_chunking_info
   
   # Check chunk information (utility function)
   print_chunking_info(ds_chunked, detailed=True)

Troubleshooting Performance Issues
==================================

Common Issues
-------------

1. **Memory errors**: Reduce chunk sizes
2. **Slow computation**: Increase chunk sizes (within memory limits)
3. **Network/IO bottlenecks**: Balance chunk sizes with data access patterns

Example Solutions
-----------------

.. code-block:: python

   # If getting memory errors
   ds_small_chunks = ds.chunk({'time': 30, 'lat': 20, 'lon': 20})
   
   # If computation is too slow
   ds_larger_chunks = ds.chunk({'time': 480, 'lat': 80, 'lon': 80})
   
   # For network datasets, smaller chunks may be better
   ds_network = ds.chunk({'time': 60, 'lat': 30, 'lon': 30})

Best Practices Summary
======================

1. **Start with conservative chunk sizes** and adjust based on performance
2. **Monitor memory usage** during development
3. **Test different chunking strategies** for your specific use case
4. **Document successful chunking patterns** for your datasets
5. **Use Dask dashboard** to monitor computation progress

.. note::
   The optimal chunking strategy depends on your specific dataset characteristics,
   available system resources, and analysis requirements. Experiment with different
   chunk sizes to find what works best for your use case.
