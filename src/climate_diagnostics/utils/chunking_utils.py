"""
Sophisticated chunking utilities for climate data processing.

This module provides dynamic chunking strategies that adapt to data characteristics,
memory constraints, and computational patterns for optimal performance when 
processing large climate datasets with xarray and Dask.

Based on advanced chunking strategies from climate data processing workflows.
"""

import numpy as np
import xarray as xr
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from typing import Dict, Union, Tuple, Optional, Any, List
import warnings
import math


def get_system_memory_info() -> Dict[str, float]:
    """
    Get system memory information.
    
    Returns
    -------
    Dict[str, float]
        Dictionary with 'total', 'available', and 'percent_used' memory in GB.
    """
    if PSUTIL_AVAILABLE:
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024**3),  # GB
            'available': memory.available / (1024**3),  # GB
            'percent_used': memory.percent
        }
    else:
        # Fallback if psutil not available
        warnings.warn("psutil not available, using default memory estimates")
        return {
            'total': 16.0,  # Assume 16GB default
            'available': 8.0,  # Assume 8GB available
            'percent_used': 50.0
        }


def inspect_disk_chunking(dataset: xr.Dataset, variable: Optional[str] = None) -> Dict[str, Any]:
    """
    Inspect on-disk chunking from file encoding.
    
    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to inspect.
    variable : str, optional
        Specific variable to inspect. If None, inspects first available.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with disk chunking information.
    """
    result: Dict[str, Any] = {'disk_chunks': None, 'bytes_per_timestep': 0}
    # Determine which variable to inspect for chunking info
    if variable is not None:
        if variable not in dataset.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset")
        var_name = variable
    else:
        if not dataset.data_vars:
            raise ValueError("Dataset contains no data variables")
        var_name = list(dataset.data_vars.keys())[0]
    var = dataset[var_name]

    # Try to get chunking from encoding (NetCDF/HDF5 chunking info)
    if hasattr(var, 'encoding'):
        chunksizes = var.encoding.get('chunksizes')
        if chunksizes:
            # Map chunk sizes to dimension names for clarity
            result['disk_chunks'] = dict(zip(var.dims, [int(size) for size in chunksizes]))
            print(f"   on-disk chunks : {chunksizes}")

            # Calculate bytes per time step if time dimension exists
            time_dim = _find_time_dimension(dataset)
            if time_dim and time_dim in var.dims:
                time_idx = var.dims.index(time_dim)
                # Only multiply spatial chunk sizes (exclude time)
                spatial_chunks = [chunksizes[i] for i in range(len(chunksizes)) if i != time_idx]
                bytes_per_element = np.dtype(var.dtype).itemsize
                bytes_per_timestep = np.prod(spatial_chunks) * bytes_per_element
                result['bytes_per_timestep'] = bytes_per_timestep
                print(f"   bytes per time step : {bytes_per_timestep/1e6:6.1f} MB")
    # If no chunking info, result will indicate so
    return result


def choose_time_chunk(bytes_per_tstep: float,
                     target_mb: float = 50, 
                     max_mb: float = 200, 
                     min_chunks: int = 32,
                     time_steps_per_year: int = 365*4) -> int:
    """
    Return a time-chunk length that optimizes memory usage and performance.
    
    This implementation follows the sophisticated chunking strategy that:
    • Stays close to *target_mb* (in MiB) per chunk
    • Never exceeds *max_mb*
    • Produces at least *min_chunks* chunks over the dataset
    • Ensures reasonable parallelization opportunities
    
    Parameters
    ----------
    bytes_per_tstep : float
        Bytes required per time step for all spatial dimensions.
    target_mb : float, optional
        Target chunk size in megabytes. Defaults to 50 MB.
    max_mb : float, optional
        Maximum allowed chunk size in megabytes. Defaults to 200 MB.
    min_chunks : int, optional
        Minimum number of chunks to create. Defaults to 32.
    time_steps_per_year : int, optional
        Total time steps to consider for chunking (e.g., 4 years of 6-hourly = 365*4*4). 
        Defaults to 365*4 (4 years of daily data).
    
    Returns
    -------
    int
        Optimal time chunk size in number of time steps.
    
    Examples
    --------
    >>> # For 6-hourly data with 10 MB per time step
    >>> bytes_per_step = 10 * 1024**2  # 10 MB in bytes
    >>> chunk_size = choose_time_chunk(bytes_per_step, target_mb=50)
    >>> print(f"Optimal chunk: {chunk_size} steps ≈ {chunk_size * bytes_per_step / 1e6:5.1f} MB per variable")
    """
    # Input validation: ensure positive bytes per timestep
    if bytes_per_tstep <= 0:
        raise ValueError(f"bytes_per_tstep must be positive, got {bytes_per_tstep}")

    # Convert megabyte limits to bytes for precise calculations
    target_bytes = target_mb * 1024**2  # Target size in bytes
    max_bytes = max_mb * 1024**2        # Maximum allowed size in bytes

    # Calculate initial chunk size based on target memory usage
    # Use ceiling to ensure we don't underestimate memory requirements
    target = int(np.ceil(target_bytes / bytes_per_tstep))
    if target < 1:
        target = 1  # Minimum of 1 timestep per chunk

    # Parallelization optimization: ensure sufficient chunks for good performance
    # Reduce chunk size if we'd have too few total chunks for effective parallelization
    if time_steps_per_year > 0 and time_steps_per_year // target < min_chunks:
        target = max(1, time_steps_per_year // min_chunks)
        print(f"   Reduced chunk size to ensure {min_chunks} parallel chunks")

    # Memory safety: iteratively reduce chunk size if it exceeds memory limits
    # Use binary reduction for efficient convergence to acceptable size
    while target * bytes_per_tstep > max_bytes and target > 1:
        target //= 2
        print(f"   Reduced chunk to {target} steps due to memory constraints")

    return target


def calculate_optimal_chunks_from_disk(dataset: xr.Dataset, 
                                     variable: Optional[str] = None,
                                     target_mb: float = 50,
                                     max_mb: float = 200,
                                     min_chunks: int = 32) -> Dict[str, int]:
    """
    Calculate optimal chunks using disk chunking information and the sophisticated
    chunking strategy.
    
    This function leverages on-disk chunk information from NetCDF/HDF5 files to make
    intelligent chunking decisions that respect the original data layout while 
    optimizing for the specified memory and performance constraints.
    
    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to analyze for optimal chunking.
    variable : str, optional
        Variable to optimize chunking for. If None, uses first available variable.
        Focusing on a specific variable yields better optimization.
    target_mb : float, optional
        Target chunk size in megabytes. Defaults to 50 MB, which balances
        memory usage with computational efficiency.
    max_mb : float, optional
        Maximum allowable chunk size in megabytes. Defaults to 200 MB.
        Prevents memory exhaustion on constrained systems.
    min_chunks : int, optional
        Minimum number of chunks to ensure good parallelization. Defaults to 32.
        Higher values improve parallel efficiency but may reduce chunk sizes.
    
    Returns
    -------
    Dict[str, int]
        Dictionary mapping dimension names to optimal chunk sizes.
        Only includes dimensions that require custom chunking.
    
    Notes
    -----
    This function implements a sophisticated strategy that:
    - Preserves beneficial spatial chunking from the source file
    - Optimizes time dimension chunking for the specified constraints
    - Balances memory usage with computational performance
    - Ensures sufficient parallelization opportunities
    """
    print("Calculating optimal chunks using disk-aware strategy...")
    
    # Step 1: Analyze on-disk chunking patterns from source files
    # This leverages chunking information embedded in NetCDF/HDF5 metadata
    disk_info = inspect_disk_chunking(dataset, variable)
    
    # Step 2: Extract memory requirements per time step for chunk sizing
    bytes_per_tstep = disk_info['bytes_per_timestep']
    if bytes_per_tstep == 0:
        # Fallback: calculate memory requirements when disk info unavailable
        bytes_per_tstep = estimate_bytes_per_timestep(dataset, variable)
    
    # 3. Choose optimal time chunk using sophisticated algorithm
    time_dim = _find_time_dimension(dataset)
    time_chunk = None
    if time_dim and bytes_per_tstep > 0:
        # Estimate total time steps (assume dataset represents 4 years if not found)
        total_time_steps = dataset.sizes.get(time_dim, 365*4)
        time_chunk = choose_time_chunk(
            bytes_per_tstep, 
            target_mb=target_mb, 
            max_mb=max_mb, 
            min_chunks=min_chunks,
            time_steps_per_year=total_time_steps  # Note: This is actually total time steps in dataset, not per year
        )
        print(f"   chosen time chunk     : {time_chunk} steps "
              f"≈ {time_chunk * bytes_per_tstep / 1e6:5.1f} MB per variable")
    
    # 4. Build the final chunk dictionary
    chunks = {}
    
    # Use disk chunks for spatial dimensions if available
    if disk_info['disk_chunks']:
        for dim, chunk_size in disk_info['disk_chunks'].items():
            if dim != time_dim:  # Keep disk chunks for spatial dims
                chunks[dim] = chunk_size
    
    # Set time chunk
    if time_dim and time_chunk:
        chunks[time_dim] = time_chunk
    
    # Fill in any missing dimensions with reasonable defaults
    for dim, size in dataset.sizes.items():
        if dim not in chunks:
            # Heuristics for common climate dimensions
            if 'lat' in dim.lower():
                chunks[dim] = min(100, max(10, size // 4))
            elif 'lon' in dim.lower():
                chunks[dim] = min(200, max(20, size // 4))
            elif any(lev in dim.lower() for lev in ['lev', 'pressure']):
                chunks[dim] = size if size <= 50 else 25
            else:
                chunks[dim] = min(100, max(1, size // 2))
    
    return chunks


def dynamic_chunk_calculator(dataset: xr.Dataset,
                           operation_type: str = 'general',
                           memory_limit_gb: Optional[float] = None,
                           performance_priority: str = 'balanced') -> Dict[str, int]:
    """
    Dynamically calculate optimal chunk sizes based on data characteristics,
    operation type, and system constraints.
    
    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to analyze for optimal chunking.
    operation_type : str, optional
        Type of operation that will be performed. Options:
        - 'general': General purpose chunking
        - 'timeseries': Time series analysis (trends, decomposition)
        - 'spatial': Spatial analysis and plotting
        - 'statistical': Statistical computations
        - 'io': Input/output operations
        Default is 'general'.
    memory_limit_gb : float, optional
        Memory limit in GB. If None, uses 25% of available system memory.
    performance_priority : str, optional
        Performance optimization priority. Options:
        - 'memory': Minimize memory usage
        - 'speed': Maximize computational speed
        - 'balanced': Balance memory and speed
        Default is 'balanced'.
    
    Returns
    -------
    Dict[str, int]
        Dictionary mapping dimension names to optimal chunk sizes.
    """
    # Get system constraints
    memory_info = get_system_memory_info()
    if memory_limit_gb is None:
        memory_limit_gb = memory_info['available'] * 0.25  # Use 25% of available memory
    
    # Determine operation-specific parameters
    operation_params = _get_operation_parameters(operation_type, performance_priority)
    
    # Start with disk-aware chunking
    chunks = calculate_optimal_chunks_from_disk(
        dataset, 
        target_mb=operation_params['target_chunk_mb'],
        max_mb=operation_params['max_chunk_mb'],
        min_chunks=operation_params.get('parallelization_factor', 32)
    )
    
    # Adjust based on operation type (e.g., larger time chunks for timeseries)
    chunks = _adjust_chunks_for_operation(dataset, chunks, operation_params)
    
    # Validate and adjust for memory constraints
    chunks = _validate_and_adjust_chunks(dataset, chunks, memory_limit_gb, operation_params)
    
    return chunks


def _get_operation_parameters(operation_type: str, performance_priority: str) -> Dict[str, Any]:
    """Get operation-specific parameters for chunking optimization."""
    
    base_params = {
        'target_chunk_mb': 64,
        'max_chunk_mb': 256,
        'parallelization_factor': 32,
        'time_preference': 'medium',
        'spatial_preference': 'medium'
    }
    
    # Adjust based on operation type
    operation_adjustments = {
        'timeseries': {
            'target_chunk_mb': 32,
            'time_preference': 'large',
            'spatial_preference': 'small',
            'parallelization_factor': 64
        },
        'spatial': {
            'target_chunk_mb': 128,
            'time_preference': 'small',
            'spatial_preference': 'large',
            'parallelization_factor': 16
        },
        'statistical': {
            'target_chunk_mb': 96,
            'time_preference': 'medium',
            'spatial_preference': 'medium',
            'parallelization_factor': 32
        },
        'io': {
            'target_chunk_mb': 256,
            'max_chunk_mb': 512,
            'time_preference': 'large',
            'spatial_preference': 'large',
            'parallelization_factor': 16
        }
    }
    
    # Apply operation adjustments
    if operation_type in operation_adjustments:
        base_params.update(operation_adjustments[operation_type])
    
    # Adjust based on performance priority
    if performance_priority == 'memory':
        base_params['target_chunk_mb'] = int(float(base_params['target_chunk_mb']) * 0.5)
        base_params['max_chunk_mb'] = int(float(base_params['max_chunk_mb']) * 0.5)
        base_params['parallelization_factor'] = int(base_params['parallelization_factor']) * 2
    elif performance_priority == 'speed':
        base_params['target_chunk_mb'] = int(float(base_params['target_chunk_mb']) * 1.5)
        base_params['max_chunk_mb'] = int(float(base_params['max_chunk_mb']) * 2)
        base_params['parallelization_factor'] = max(16, int(base_params['parallelization_factor']) // 2)
    
    return base_params


def _adjust_chunks_for_operation(dataset: xr.Dataset, chunks: Dict[str, int], 
                                operation_params: Dict[str, Any]) -> Dict[str, int]:
    """Adjust chunk sizes based on operation preferences."""
    
    time_dim = _find_time_dimension(dataset)
    time_pref = operation_params.get('time_preference', 'medium')
    spatial_pref = operation_params.get('spatial_preference', 'medium')
    
    # Adjust time chunks
    if time_dim and time_dim in chunks:
        current_time_chunk = chunks[time_dim]
        if time_pref == 'large':
            chunks[time_dim] = min(current_time_chunk * 2, dataset.sizes[time_dim])
        elif time_pref == 'small':
            chunks[time_dim] = max(current_time_chunk // 2, 1)
    
    # Adjust spatial chunks
    for dim in chunks:
        if dim != time_dim and any(spatial_key in dim.lower() for spatial_key in ['lat', 'lon', 'x', 'y']):
            current_chunk = chunks[dim]
            if spatial_pref == 'large':
                chunks[dim] = min(current_chunk * 2, dataset.sizes[dim])
            elif spatial_pref == 'small':
                chunks[dim] = max(current_chunk // 2, 1)
    
    return chunks


def _validate_and_adjust_chunks(dataset: xr.Dataset, chunks: Dict[str, int],
                               memory_limit_gb: float, operation_params: Dict[str, Any]) -> Dict[str, int]:
    """
    Validate and adjust chunk sizes to ensure they meet memory constraints.
    
    This function performs the final validation step in the chunking pipeline:
    1. Calculates actual memory usage per chunk (not total dataset)
    2. Scales down chunks proportionally if they exceed memory limits
    3. Ensures chunks don't exceed dimension sizes
    
    The memory calculation is per-chunk, not total dataset, because Dask
    loads one chunk at a time into memory.
    
    Parameters
    ----------
    dataset : xr.Dataset
        The dataset being chunked.
    chunks : Dict[str, int]
        Current chunk sizes per dimension.
    memory_limit_gb : float
        Memory limit in GB per chunk.
    operation_params : Dict[str, Any]
        Operation-specific parameters (currently unused but kept for extensibility).
    
    Returns
    -------
    Dict[str, int]
        Validated and adjusted chunk sizes.
        
    Notes
    -----
    The scaling uses sqrt() to preserve aspect ratios while reducing memory.
    This is more effective than linear scaling for multi-dimensional data.
    """
    
    # Calculate memory usage per chunk (not total dataset memory)
    # This is the critical distinction - we care about peak memory per operation
    max_chunk_memory_mb = 0
    for var_name, var in dataset.data_vars.items():
        var_chunk_memory = np.dtype(var.dtype).itemsize  # Start with bytes per element
        
        # Calculate chunk memory by multiplying element size by chunk dimensions
        for dim_name in var.dims:
            if dim_name in chunks:
                var_chunk_memory *= chunks[dim_name]
            else:
                # For dimensions not in chunks, use representative size
                # Cap at 100 to avoid massive memory calculations for unchunked dims
                var_chunk_memory *= min(var.sizes[dim_name], 100)
        
        chunk_memory_mb = var_chunk_memory / (1024**2)
        max_chunk_memory_mb = max(max_chunk_memory_mb, chunk_memory_mb)
    
    # If chunk memory exceeds limit, scale down chunks proportionally
    # Use sqrt() scaling to preserve dimensional relationships
    memory_limit_mb = memory_limit_gb * 1024
    if max_chunk_memory_mb > memory_limit_mb:
        scale_factor = math.sqrt(memory_limit_mb / max_chunk_memory_mb)
        for dim_name in chunks:
            chunks[dim_name] = max(1, int(chunks[dim_name] * scale_factor))
    
    # Final safety check: ensure chunks don't exceed dimension sizes
    # This can happen with very small datasets or after scaling
    for dim_name, chunk_size in chunks.items():
        if dim_name in dataset.sizes:
            chunks[dim_name] = min(chunk_size, dataset.sizes[dim_name])
    
    return chunks


def estimate_bytes_per_timestep(dataset: xr.Dataset, 
                               variable: Optional[str] = None,
                               exclude_time_dims: bool = True) -> float:
    """
    Estimate bytes required per time step for a dataset or variable.
    
    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to analyze.
    variable : str, optional
        Specific variable to analyze. If None, analyzes all data variables.
    exclude_time_dims : bool, optional
        Whether to exclude time dimensions from size calculation. Defaults to True.
    
    Returns
    -------
    float
        Estimated bytes per time step.
    """
    total_bytes = 0
    
    if variable is not None:
        if variable not in dataset.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset")
        data_vars = [variable]
    else:
        data_vars = list(dataset.data_vars.keys())
    
    for var_name in data_vars:
        var = dataset[var_name]
        
        # Get data type size
        bytes_per_element = np.dtype(var.dtype).itemsize
        
        # Calculate size excluding time dimension
        if exclude_time_dims:
            spatial_dims = [dim for dim in var.dims if not dim.lower().startswith(('time', 't'))]
            spatial_size = np.prod([var.sizes[dim] for dim in spatial_dims])
        else:
            spatial_size = var.size
        
        total_bytes += spatial_size * bytes_per_element
    
    return total_bytes


def _find_time_dimension(dataset: xr.Dataset) -> Optional[str]:
    """Find the time dimension in a dataset."""
    time_candidates = ['time', 't', 'time_counter', 'TIME']
    
    for dim in dataset.dims:
        if dim in time_candidates or dim.lower().startswith('time'):
            return dim
    
    return None


def get_optimal_chunks(dataset: xr.Dataset,
                      variable: Optional[str] = None,
                      target_mb: float = 50,
                      max_mb: float = 200,
                      min_chunks: int = 32,
                      preserve_disk_chunks: bool = True,
                      time_freq: Optional[str] = None) -> Dict[str, int]:
    """
    Generate optimal chunk sizes for a climate dataset using the sophisticated
    chunking strategy.
    
    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to analyze for optimal chunking.
    variable : str, optional
        Focus chunking optimization on a specific variable. If None,
        considers all variables.
    target_mb : float, optional
        Target chunk size in megabytes. Defaults to 50 MB.
    max_mb : float, optional
        Maximum chunk size in megabytes. Defaults to 200 MB.
    min_chunks : int, optional
        Minimum number of chunks for parallelization. Defaults to 32.
    preserve_disk_chunks : bool, optional
        Whether to preserve spatial chunking from disk when possible. 
        Defaults to True.
    time_freq : str, optional
        Time frequency hint ('daily', 'monthly', 'hourly', '6hourly').
        Used to estimate time steps per year.
    
    Returns
    -------
    Dict[str, int]
        Dictionary mapping dimension names to optimal chunk sizes.
    
    Examples
    --------
    >>> import xarray as xr
    >>> ds = xr.tutorial.open_dataset('air_temperature')
    >>> chunks = get_optimal_chunks(ds, target_mb=100)
    >>> ds_chunked = ds.chunk(chunks)
    """
    if preserve_disk_chunks:
        # Use the sophisticated disk-aware chunking strategy
        return calculate_optimal_chunks_from_disk(
            dataset, variable, target_mb, max_mb, min_chunks
        )
    else:
        # Use dynamic chunking without disk awareness
        return dynamic_chunk_calculator(dataset, 'general')


def rechunk_dataset(dataset: xr.Dataset,
                   target_mb: float = 50,
                   max_mb: float = 200,
                   variable: Optional[str] = None,
                   time_freq: Optional[str] = None,
                   preserve_spatial_chunks: bool = True) -> xr.Dataset:
    """
    Rechunk a dataset with optimal chunk sizes for performance.
    
    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to rechunk.
    target_mb : float, optional
        Target chunk size in megabytes. Defaults to 50 MB.
    max_mb : float, optional
        Maximum chunk size in megabytes. Defaults to 200 MB.
    variable : str, optional
        Variable to optimize chunking for. If None, optimizes for all variables.
    time_freq : str, optional
        Time frequency hint for better optimization.
    preserve_spatial_chunks : bool, optional
        Whether to preserve spatial chunking from disk. Defaults to True.
    
    Returns
    -------
    xr.Dataset
        Rechunked dataset.
    
    Examples
    --------
    >>> ds_chunked = rechunk_dataset(ds, target_mb=100, time_freq='daily')
    """
    # Get optimal chunks
    chunks = get_optimal_chunks(
        dataset, variable, target_mb, max_mb, 
        preserve_disk_chunks=preserve_spatial_chunks, 
        time_freq=time_freq
    )
    
    # Apply chunking
    print(f"Rechunking dataset with strategy: {chunks}")
    print(f"Estimated memory per chunk: ~{target_mb} MB (max: {max_mb} MB)")
    
    return dataset.chunk(chunks)


def print_chunking_info(dataset: xr.Dataset, detailed: bool = False) -> None:
    """
    Print detailed information about dataset chunking.
    
    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to analyze.
    detailed : bool, optional
        Whether to print detailed per-variable information. Defaults to False.
    """
    print("Dataset Chunking Information")
    print("=" * 50)
    
    # Overall dataset info
    total_size_mb = dataset.nbytes / (1024**2)
    print(f"Total dataset size: {total_size_mb:.1f} MB")
    
    # Memory info
    memory_info = get_system_memory_info()
    print(f"System memory: {memory_info['total']:.1f} GB total, "
          f"{memory_info['available']:.1f} GB available")
    
    # Check if chunked
    is_chunked = any(hasattr(var.data, 'chunks') for var in dataset.data_vars.values())
    print(f"Is chunked: {is_chunked}")
    
    if is_chunked:
        # Estimate chunk sizes
        chunk_sizes = []
        for var_name, var in dataset.data_vars.items():
            if hasattr(var.data, 'chunksize'):
                chunk_size_mb = var.data.chunksize / (1024**2)
                chunk_sizes.append(chunk_size_mb)
        
        if chunk_sizes:
            avg_chunk_mb = np.mean(chunk_sizes)
            max_chunk_mb = np.max(chunk_sizes)
            print(f"Average chunk size: {avg_chunk_mb:.1f} MB")
            print(f"Maximum chunk size: {max_chunk_mb:.1f} MB")
    
    # Dimension information
    print("\nDimensions:")
    for dim, size in dataset.sizes.items():
        print(f"  {dim}: {size}")
    
    # Inspect disk chunking
    try:
        disk_info = inspect_disk_chunking(dataset)
        if disk_info['disk_chunks']:
            print(f"\nDisk chunks: {disk_info['disk_chunks']}")
            if disk_info['bytes_per_timestep'] > 0:
                print(f"Bytes per timestep: {disk_info['bytes_per_timestep']/1e6:.1f} MB")
    except Exception:
        pass
    
    if detailed:
        print("\nPer-variable chunking:")
        for var_name, var in dataset.data_vars.items():
            print(f"\n  {var_name}:")
            print(f"    Shape: {var.shape}")
            print(f"    Size: {var.nbytes / (1024**2):.1f} MB")
            if hasattr(var.data, 'chunks'):
                print(f"    Chunks: {var.data.chunks}")


def suggest_chunking_strategy(dataset: xr.Dataset,
                            target_use_case: str = 'general') -> Dict[str, Any]:
    """
    Suggest chunking strategies for different use cases using the sophisticated
    chunking algorithm.
    
    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to analyze.
    target_use_case : str, optional
        Target use case: 'general', 'time_series', 'spatial_analysis', 
        'trend_analysis', 'memory_limited'. Defaults to 'general'.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with chunking recommendations and rationale.
    """
    strategies = {
        'general': {
            'target_mb': 50.0,
            'max_mb': 200.0,
            'description': 'Balanced approach for general climate analysis'
        },
        'time_series': {
            'target_mb': 25.0,
            'max_mb': 100.0,
            'description': 'Optimized for time series analysis with smaller chunks'
        },
        'spatial_analysis': {
            'target_mb': 100.0,
            'max_mb': 500.0,
            'description': 'Larger chunks for spatial operations and mapping'
        },
        'trend_analysis': {
            'target_mb': 75.0,
            'max_mb': 300.0,
            'description': 'Medium chunks suitable for STL decomposition and trends'
        },
        'memory_limited': {
            'target_mb': 20.0,
            'max_mb': 50.0,
            'description': 'Conservative chunking for memory-constrained environments'
        }
    }
    
    if target_use_case not in strategies:
        raise ValueError(f"Unknown use case '{target_use_case}'. "
                        f"Available: {list(strategies.keys())}")
    
    strategy = strategies[target_use_case]
    
    # Get optimal chunks for this strategy using sophisticated algorithm
    chunks = calculate_optimal_chunks_from_disk(
        dataset, 
        target_mb=float(strategy['target_mb']),
        max_mb=float(strategy['max_mb'])
    )
    
    return {
        'chunks': chunks,
        'strategy': strategy,
        'bytes_per_timestep': estimate_bytes_per_timestep(dataset),
        'recommendation': f"Use {strategy['description']} with chunks: {chunks}"
    }


__all__ = [
    'get_system_memory_info',
    'inspect_disk_chunking',
    'choose_time_chunk',
    'calculate_optimal_chunks_from_disk',
    'dynamic_chunk_calculator',
    'estimate_bytes_per_timestep', 
    'get_optimal_chunks',
    'rechunk_dataset',
    'print_chunking_info',
    'suggest_chunking_strategy'
]
