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
            print(f"   on-disk chunks : {result['disk_chunks']}")

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
                     total_time_steps: int,
                     target_mb: float = 50,
                     max_mb: float = 200,
                     min_chunks: int = 32) -> int:
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
    total_time_steps : int
        #[MODIFICATION] Renamed from `time_steps_per_year` for clarity, as per review.
        # This parameter represents the total number of time steps in the dataset.
        Total time steps in the dataset.
    target_mb : float, optional
        Target chunk size in megabytes. Defaults to 50 MB.
    max_mb : float, optional
        Maximum allowed chunk size in megabytes. Defaults to 200 MB.
    min_chunks : int, optional
        Minimum number of chunks to create. Defaults to 32.
    
    Returns
    -------
    int
        Optimal time chunk size in number of time steps.
    
    Examples
    --------
    >>> # For 6-hourly data with 10 MB per time step
    >>> bytes_per_step = 10 * 1024**2  # 10 MB in bytes
    >>> total_steps = 365 * 24 * 10  # 10 years of hourly data
    >>> chunk_size = choose_time_chunk(bytes_per_step, total_steps, target_mb=50)
    >>> print(f"Optimal chunk: {chunk_size} steps ≈ {chunk_size * bytes_per_step / 1e6:5.1f} MB per variable")
    """
    #[MODIFICATION] Added validation for input parameters as per review suggestions.
    if not all(p > 0 for p in [bytes_per_tstep, total_time_steps, target_mb, max_mb, min_chunks]):
        raise ValueError("All numerical arguments to choose_time_chunk must be positive.")
    if target_mb > max_mb:
        warnings.warn(f"target_mb ({target_mb}) is greater than max_mb ({max_mb}). Using max_mb as the target.")
        target_mb = max_mb

    # Convert megabyte limits to bytes for precise calculations
    target_bytes = target_mb * 1024**2
    max_bytes = max_mb * 1024**2

    # Calculate initial chunk size based on target memory usage
    target = int(np.ceil(target_bytes / bytes_per_tstep))
    if target < 1:
        target = 1

    # Parallelization optimization: ensure sufficient chunks for good performance
    if total_time_steps // target < min_chunks:
        target = max(1, total_time_steps // min_chunks)
        print(f"   Reduced chunk size to ensure at least {min_chunks} parallel chunks")

    # Memory safety: iteratively reduce chunk size if it exceeds memory limits
    while target * bytes_per_tstep > max_bytes and target > 1:
        target = max(1, target // 2)
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
    target_mb : float, optional
        Target chunk size in megabytes. Defaults to 50 MB.
    max_mb : float, optional
        Maximum allowable chunk size in megabytes. Defaults to 200 MB.
    min_chunks : int, optional
        Minimum number of chunks to ensure good parallelization. Defaults to 32.
    
    Returns
    -------
    Dict[str, int]
        Dictionary mapping dimension names to optimal chunk sizes.
    """
    print("Calculating optimal chunks using disk-aware strategy...")
    disk_info = inspect_disk_chunking(dataset, variable)
    bytes_per_tstep = disk_info['bytes_per_timestep']
    
    if bytes_per_tstep == 0:
        bytes_per_tstep = estimate_bytes_per_timestep(dataset, variable)
    
    time_dim = _find_time_dimension(dataset)
    time_chunk = None
    if time_dim and bytes_per_tstep > 0 and time_dim in dataset.sizes:
        total_time_steps = dataset.sizes[time_dim]
        time_chunk = choose_time_chunk(
            bytes_per_tstep,
            total_time_steps=total_time_steps,
            target_mb=target_mb,
            max_mb=max_mb,
            min_chunks=min_chunks
        )

        #[MODIFICATION] Added chunk alignment verification as per review.
        # This aligns the calculated Dask chunk with the on-disk chunk for efficient I/O.
        disk_chunks = disk_info.get('disk_chunks')
        if disk_chunks and time_dim in disk_chunks and disk_chunks[time_dim] > 1:
            disk_time_chunk = disk_chunks[time_dim]
            # Find the nearest multiple of the disk chunk that fits memory constraints
            multiplier = round(time_chunk / disk_time_chunk)
            aligned_chunk = max(1, int(multiplier)) * disk_time_chunk

            # If the aligned chunk is too large, try one multiplier smaller
            if aligned_chunk * bytes_per_tstep > (max_mb * 1024**2) and multiplier > 1:
                aligned_chunk = (int(multiplier) - 1) * disk_time_chunk
            
            # Fallback to the base disk chunk if it's still too large
            if aligned_chunk == 0 or aligned_chunk * bytes_per_tstep > (max_mb * 1024**2):
                 aligned_chunk = disk_time_chunk

            # Only apply if the alignment is valid and improves the chunking
            if 0 < aligned_chunk * bytes_per_tstep <= (max_mb * 1024**2):
                if time_chunk != aligned_chunk:
                    print(f"   Aligning time chunk to on-disk chunking ({disk_time_chunk} steps). Was {time_chunk}, is now {aligned_chunk}.")
                    time_chunk = aligned_chunk
            else:
                warnings.warn(f"Could not align time chunk with on-disk chunk size of {disk_time_chunk} due to memory constraints.")
        
        print(f"   chosen time chunk     : {time_chunk} steps "
              f"≈ {time_chunk * bytes_per_tstep / 1e6:5.1f} MB per variable")
    
    chunks = {}
    if disk_info['disk_chunks']:
        for dim, chunk_size in disk_info['disk_chunks'].items():
            if dim != time_dim:
                chunks[dim] = chunk_size
    
    if time_dim and time_chunk:
        chunks[time_dim] = time_chunk
    
    for dim, size in dataset.sizes.items():
        if dim not in chunks:
            if 'lat' in dim.lower(): chunks[dim] = min(100, max(10, size // 4))
            elif 'lon' in dim.lower(): chunks[dim] = min(200, max(20, size // 4))
            elif any(lev in dim.lower() for lev in ['lev', 'pressure']): chunks[dim] = size if size <= 50 else 25
            else: chunks[dim] = min(100, max(1, size // 2))
    
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
        Type of operation. Options: 'general', 'timeseries', 'spatial', 'statistical', 'io'.
    memory_limit_gb : float, optional
        Memory limit in GB. If None, uses 25% of available system memory.
    performance_priority : str, optional
        Optimization priority. Options: 'memory', 'speed', 'balanced'.
    
    Returns
    -------
    Dict[str, int]
        Dictionary mapping dimension names to optimal chunk sizes.
    """
    memory_info = get_system_memory_info()
    if memory_limit_gb is None:
        memory_limit_gb = memory_info['available'] * 0.25
    
    operation_params = _get_operation_parameters(operation_type, performance_priority)
    
    chunks = calculate_optimal_chunks_from_disk(
        dataset, 
        target_mb=operation_params['target_chunk_mb'],
        max_mb=operation_params['max_chunk_mb'],
        min_chunks=operation_params.get('parallelization_factor', 32)
    )
    
    chunks = _adjust_chunks_for_operation(dataset, chunks, operation_params)
    chunks = _validate_and_adjust_chunks(dataset, chunks, memory_limit_gb, operation_params)
    
    return chunks


def _get_operation_parameters(operation_type: str, performance_priority: str) -> Dict[str, Any]:
    """Get operation-specific parameters for chunking optimization."""
    #[MODIFICATION] Added validation for input parameters to improve robustness, as per review.
    valid_ops = ['general', 'timeseries', 'spatial', 'statistical', 'io']
    if operation_type not in valid_ops:
        raise ValueError(f"Invalid operation_type '{operation_type}'. Must be one of {valid_ops}")

    valid_prios = ['balanced', 'memory', 'speed']
    if performance_priority not in valid_prios:
        raise ValueError(f"Invalid performance_priority '{performance_priority}'. Must be one of {valid_prios}")
    
    base_params = {
        'target_chunk_mb': 64, 'max_chunk_mb': 256, 'parallelization_factor': 32,
        'time_preference': 'medium', 'spatial_preference': 'medium'
    }
    
    operation_adjustments = {
        'timeseries': {'target_chunk_mb': 32, 'time_preference': 'large', 'spatial_preference': 'small', 'parallelization_factor': 64},
        'spatial': {'target_chunk_mb': 128, 'time_preference': 'small', 'spatial_preference': 'large', 'parallelization_factor': 16},
        'statistical': {'target_chunk_mb': 96, 'parallelization_factor': 32},
        'io': {'target_chunk_mb': 256, 'max_chunk_mb': 512, 'time_preference': 'large', 'spatial_preference': 'large', 'parallelization_factor': 16}
    }
    
    if operation_type in operation_adjustments:
        base_params.update(operation_adjustments[operation_type])
    
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
    
    if time_dim and time_dim in chunks:
        current_time_chunk = chunks[time_dim]
        if time_pref == 'large': chunks[time_dim] = min(current_time_chunk * 2, dataset.sizes[time_dim])
        elif time_pref == 'small': chunks[time_dim] = max(current_time_chunk // 2, 1)
    
    for dim in chunks:
        if dim != time_dim and any(key in dim.lower() for key in ['lat', 'lon', 'x', 'y']):
            current_chunk = chunks[dim]
            if spatial_pref == 'large': chunks[dim] = min(current_chunk * 2, dataset.sizes[dim])
            elif spatial_pref == 'small': chunks[dim] = max(current_chunk // 2, 1)
    
    return chunks


def _validate_and_adjust_chunks(dataset: xr.Dataset, chunks: Dict[str, int],
                               memory_limit_gb: float, operation_params: Dict[str, Any]) -> Dict[str, int]:
    """Validate and adjust chunk sizes to meet memory constraints."""
    max_chunk_memory_mb = 0
    for var_name, var in dataset.data_vars.items():
        var_chunk_memory = np.dtype(var.dtype).itemsize
        for dim_name in var.dims:
            var_chunk_memory *= chunks.get(dim_name, min(var.sizes[dim_name], 100))
        max_chunk_memory_mb = max(max_chunk_memory_mb, var_chunk_memory / (1024**2))
    
    memory_limit_mb = memory_limit_gb * 1024
    if max_chunk_memory_mb > memory_limit_mb:
        scale_factor = math.sqrt(memory_limit_mb / max_chunk_memory_mb)
        for dim_name in chunks:
            chunks[dim_name] = max(1, int(chunks[dim_name] * scale_factor))
    
    for dim_name, chunk_size in chunks.items():
        if dim_name in dataset.sizes:
            chunks[dim_name] = min(chunk_size, dataset.sizes[dim_name])
    
    return chunks


def estimate_bytes_per_timestep(dataset: xr.Dataset, 
                               variable: Optional[str] = None) -> float:
    """Estimate bytes required per time step for a dataset or variable."""
    total_bytes = 0
    data_vars = [variable] if variable and variable in dataset.data_vars else list(dataset.data_vars.keys())
    if not data_vars: return 0.0

    time_dim = _find_time_dimension(dataset)
    
    for var_name in data_vars:
        var = dataset[var_name]
        bytes_per_element = np.dtype(var.dtype).itemsize
        
        # Calculate size excluding time dimension
        size_per_timestep = 1
        for dim in var.dims:
            if dim != time_dim:
                size_per_timestep *= var.sizes[dim]
        
        total_bytes += size_per_timestep * bytes_per_element
    
    return total_bytes


def _find_time_dimension(dataset: xr.Dataset) -> Optional[str]:
    """Find the time dimension in a dataset."""
    time_candidates = ['time', 't', 'time_counter', 'TIME']
    for dim in dataset.dims:
        if dim in time_candidates or dim.lower().startswith('time'): return dim
    return None


def get_optimal_chunks(dataset: xr.Dataset, **kwargs) -> Dict[str, int]:
    """
    Generate optimal chunk sizes for a climate dataset. This is a convenience
    wrapper around the core chunking functions.
    
    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to analyze.
    **kwargs
        Arguments passed to `calculate_optimal_chunks_from_disk` or 
        `dynamic_chunk_calculator`. E.g., `target_mb`, `max_mb`, `min_chunks`,
        `preserve_disk_chunks`, `operation_type`.

    Returns
    -------
    Dict[str, int]
        Dictionary mapping dimension names to optimal chunk sizes.
    """
    preserve_disk = kwargs.pop('preserve_disk_chunks', True)
    if preserve_disk:
        return calculate_optimal_chunks_from_disk(dataset, **kwargs)
    else:
        return dynamic_chunk_calculator(dataset, **kwargs)


def rechunk_dataset(dataset: xr.Dataset, **kwargs) -> xr.Dataset:
    """
    Rechunk a dataset with optimal chunk sizes for performance.
    
    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to rechunk.
    **kwargs
        Arguments passed to `get_optimal_chunks`, e.g., `target_mb`, `max_mb`.

    Returns
    -------
    xr.Dataset
        Rechunked dataset.
    """
    chunks = get_optimal_chunks(dataset, **kwargs)
    target_mb = kwargs.get('target_mb', 50)
    max_mb = kwargs.get('max_mb', 200)

    print(f"Rechunking dataset with strategy: {chunks}")
    print(f"Estimated memory per chunk: ~{target_mb} MB (max: {max_mb} MB)")
    
    return dataset.chunk(chunks)


def print_chunking_info(dataset: xr.Dataset, detailed: bool = False) -> None:
    """Print detailed information about dataset chunking."""
    print("Dataset Chunking Information\n" + "=" * 50)
    
    total_size_mb = dataset.nbytes / (1024**2)
    print(f"Total dataset size: {total_size_mb:.1f} MB")
    
    memory_info = get_system_memory_info()
    print(f"System memory: {memory_info['total']:.1f} GB total, {memory_info['available']:.1f} GB available")
    
    is_chunked = any(hasattr(var.data, 'chunks') for var in dataset.data_vars.values())
    print(f"Is chunked: {is_chunked}")
    
    if is_chunked:
        chunk_sizes = [var.data.chunksize / (1024**2) for var in dataset.data_vars.values() if hasattr(var.data, 'chunksize')]
        if chunk_sizes:
            print(f"Average chunk size: {np.mean(chunk_sizes):.1f} MB")
            print(f"Maximum chunk size: {np.max(chunk_sizes):.1f} MB")
    
    print("\nDimensions:")
    for dim, size in dataset.sizes.items(): print(f"  {dim}: {size}")
    
    try:
        disk_info = inspect_disk_chunking(dataset)
        if disk_info['disk_chunks']:
            print(f"\nDisk chunks: {disk_info['disk_chunks']}")
            if disk_info['bytes_per_timestep'] > 0:
                print(f"Bytes per timestep: {disk_info['bytes_per_timestep']/1e6:.1f} MB")
    except ValueError as e:
        print(f"\nCould not inspect disk chunking: {e}")
    
    if detailed:
        print("\nPer-variable chunking:")
        for var_name, var in dataset.data_vars.items():
            print(f"\n  {var_name}:\n    Shape: {var.shape}\n    Size: {var.nbytes / (1024**2):.1f} MB")
            if hasattr(var.data, 'chunks'):
                print(f"    Chunks: {var.data.chunks}")


def suggest_chunking_strategy(dataset: xr.Dataset,
                            target_use_case: str = 'general') -> Dict[str, Any]:
    """
    Suggest chunking strategies for different use cases.
    
    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to analyze.
    target_use_case : str, optional
        Options: 'general', 'time_series', 'spatial_analysis', 'memory_limited'.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with chunking recommendations and rationale.
    """
    strategies = {
        'general': {'target_mb': 50.0, 'max_mb': 200.0, 'description': 'Balanced approach'},
        'time_series': {'target_mb': 25.0, 'max_mb': 100.0, 'description': 'Optimized for time series analysis'},
        'spatial_analysis': {'target_mb': 100.0, 'max_mb': 500.0, 'description': 'Optimized for spatial operations'},
        'memory_limited': {'target_mb': 20.0, 'max_mb': 50.0, 'description': 'Conservative for low-memory environments'}
    }
    
    if target_use_case not in strategies:
        raise ValueError(f"Unknown use case '{target_use_case}'. Available: {list(strategies.keys())}")
    
    strategy = strategies[target_use_case]
    chunks = calculate_optimal_chunks_from_disk(
        dataset, target_mb=float(strategy['target_mb']), max_mb=float(strategy['max_mb'])
    )
    
    return {
        'use_case': target_use_case,
        'chunks': chunks,
        'rationale': strategy['description'],
        'estimated_bytes_per_timestep': estimate_bytes_per_timestep(dataset),
        'recommendation': f"For '{strategy['description']}', use chunks: {chunks}"
    }


__all__ = [
    'get_system_memory_info', 'inspect_disk_chunking', 'choose_time_chunk',
    'calculate_optimal_chunks_from_disk', 'dynamic_chunk_calculator',
    'estimate_bytes_per_timestep', 'get_optimal_chunks', 'rechunk_dataset',
    'print_chunking_info', 'suggest_chunking_strategy'
]