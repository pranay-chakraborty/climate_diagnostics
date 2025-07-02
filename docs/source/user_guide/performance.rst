============================
Performance Optimization
============================

This guide covers performance optimization techniques for the Climate Diagnostics Toolkit, with a focus on sophisticated chunking strategies and memory management for large climate datasets.

🚀 **Overview**
================

The Climate Diagnostics Toolkit includes advanced performance optimization features:

- **Disk-aware chunking** that adapts to file structure
- **Operation-specific optimization** for different analysis types
- **Memory-conscious processing** with automatic scaling
- **Dynamic chunk calculation** based on system resources

⚡ **Quick Performance Tips**
==============================

1. **Always optimize chunking** for your specific analysis type
2. **Use operation-specific methods** like ``optimize_for_trends()``
3. **Monitor memory usage** with built-in analysis tools
4. **Choose appropriate chunk sizes** based on your data frequency

🔧 **Chunking Strategies**
===========================

Basic Chunking Optimization
----------------------------

Start with basic chunking optimization for any analysis:

.. code-block:: python

   import xarray as xr
   import climate_diagnostics
   
   # Load your dataset
   ds = xr.open_dataset("large_climate_data.nc")
   
   # Apply basic optimization
   ds_optimized = ds.climate_timeseries.optimize_chunks(
       target_mb=50,  # Target 50 MB chunks
       variable='temperature'
   )

Advanced Chunking Strategies
-----------------------------

For more sophisticated optimization:

.. code-block:: python

   # Advanced optimization with operation-specific tuning
   ds_advanced = ds.climate_timeseries.optimize_chunks_advanced(
       operation_type='timeseries',        # 'timeseries', 'spatial', 'statistical'
       performance_priority='balanced',    # 'memory', 'speed', 'balanced'
       memory_limit_gb=8.0,               # Set memory limit
       use_disk_chunks=True               # Preserve spatial disk chunks
   )

Operation-Specific Optimization
--------------------------------

Different analysis types benefit from different chunking strategies:

**Time Series Analysis:**

.. code-block:: python

   # Optimize for time series operations
   ds_ts = ds.climate_timeseries.optimize_chunks_advanced(
       operation_type='timeseries',
       performance_priority='memory'
   )
   
   # Or use the dedicated method
   ds_ts = ds.climate_timeseries.optimize_for_decomposition()

**Trend Analysis:**

.. code-block:: python

   # Optimize for trend calculations
   ds_trends = ds.climate_trends.optimize_for_trends(
       variable='temperature',
       use_case='spatial_trends'
   )

**Spatial Analysis:**

.. code-block:: python

   # Optimize for spatial operations and plotting
   ds_spatial = ds.climate_timeseries.optimize_chunks_advanced(
       operation_type='spatial',
       performance_priority='speed'
   )

📊 **Performance Analysis**
===========================

Chunking Analysis Tools
------------------------

Analyze your current chunking strategy:

.. code-block:: python

   # Print detailed chunking information
   ds.climate_timeseries.print_chunking_info(detailed=True)
   
   # Get chunking recommendations for different use cases
   ds.climate_timeseries.analyze_chunking_strategy()

Example output:

.. code-block:: text

   Climate Data Chunking Analysis
   ================================================
   
   Recommended chunking strategies:
   
   Time Series:
     Target: 25 MB chunks
     Max: 100 MB chunks  
     Chunks: {'time': 48, 'lat': 73, 'lon': 144}
     Use: Optimized for time series analysis with smaller chunks
   
   Spatial Analysis:
     Target: 100 MB chunks
     Max: 500 MB chunks
     Chunks: {'time': 12, 'lat': 145, 'lon': 288}
     Use: Larger chunks for spatial operations and mapping

Memory Management
-----------------

Monitor and control memory usage:

.. code-block:: python

   # Check system memory
   from climate_diagnostics.utils.chunking_utils import get_system_memory_info
   
   memory_info = get_system_memory_info()
   print(f"Available memory: {memory_info['available']:.1f} GB")
   
   # Optimize for memory-constrained systems
   ds_memory = ds.climate_timeseries.optimize_chunks_advanced(
       operation_type='general',
       performance_priority='memory',
       memory_limit_gb=4.0  # Limit to 4 GB
   )

🎯 **Best Practices by Data Type**
===================================

Daily Data (High Frequency)
----------------------------

.. code-block:: python

   # For daily data (365+ time steps per year)
   ds_daily = ds.climate_timeseries.optimize_chunks(
       target_mb=75,
       time_freq='daily'
   )

Monthly Data (Standard Climate)
-------------------------------

.. code-block:: python

   # For monthly data (12 time steps per year)
   ds_monthly = ds.climate_timeseries.optimize_chunks(
       target_mb=50,
       time_freq='monthly'
   )

High-Resolution Spatial Data
----------------------------

.. code-block:: python

   # For high-resolution grids (>1000x1000)
   ds_hires = ds.climate_timeseries.optimize_chunks_advanced(
       operation_type='spatial',
       performance_priority='memory',
       memory_limit_gb=8.0
   )

🔍 **Troubleshooting Performance Issues**
==========================================

Common Issues and Solutions
----------------------------

**Memory Errors:**

.. code-block:: python

   # Reduce chunk sizes
   ds_safe = ds.climate_timeseries.optimize_chunks_advanced(
       performance_priority='memory',
       memory_limit_gb=2.0  # Conservative limit
   )

**Slow Processing:**

.. code-block:: python

   # Increase chunk sizes for speed
   ds_fast = ds.climate_timeseries.optimize_chunks_advanced(
       performance_priority='speed',
       operation_type='spatial'
   )

**Poor Parallelization:**

.. code-block:: python

   # Ensure sufficient chunks for parallel processing
   ds_parallel = ds.climate_timeseries.optimize_chunks_advanced(
       operation_type='general',
       memory_limit_gb=16.0  # Allow larger memory for more chunks
   )

📈 **Performance Monitoring**
==============================

Track Performance Improvements
-------------------------------

.. code-block:: python

   import time
   from dask.diagnostics import ProgressBar
   
   # Time operations with different chunking strategies
   def time_operation(dataset, operation_name):
       start = time.time()
       with ProgressBar():
           result = dataset.air.mean(['lat', 'lon']).compute()
       end = time.time()
       print(f"{operation_name}: {end - start:.2f} seconds")
       return result
   
   # Compare performance
   time_operation(ds_original, "Original chunking")
   time_operation(ds_optimized, "Optimized chunking")

Real-World Examples
-------------------

**Large Climate Model Output:**

.. code-block:: python

   # For CMIP6-style data (>10 GB files)
   ds_cmip = xr.open_dataset("cmip6_tas_daily.nc")
   ds_cmip_opt = ds_cmip.climate_timeseries.optimize_chunks_advanced(
       operation_type='timeseries',
       performance_priority='balanced',
       memory_limit_gb=12.0,
       variable='tas'
   )

**Observational Gridded Data:**

.. code-block:: python

   # For observational products (ERA5, etc.)
   ds_obs = xr.open_dataset("era5_temperature.nc")
   ds_obs_opt = ds_obs.climate_timeseries.optimize_chunks(
       target_mb=100,
       time_freq='hourly',
       variable='t2m'
   )

🎛️ **Advanced Configuration**
===============================

Custom Chunking Strategies
---------------------------

For specialized use cases, you can create custom chunking:

.. code-block:: python

   from climate_diagnostics.utils.chunking_utils import (
       calculate_optimal_chunks_from_disk,
       dynamic_chunk_calculator
   )
   
   # Custom disk-aware chunking
   custom_chunks = calculate_optimal_chunks_from_disk(
       ds, 
       target_mb=150,
       variable='precipitation'
   )
   ds_custom = ds.chunk(custom_chunks)
   
   # Dynamic chunking with custom parameters
   adaptive_chunks = dynamic_chunk_calculator(
       ds,
       operation_type='statistical',
       memory_limit_gb=6.0,
       performance_priority='speed'
   )

See Also
========

* :doc:`../quickstart` - Getting started with basic optimization
* :doc:`../api/utils` - Full API reference for chunking utilities
* :doc:`plotting` - Visualization performance tips
