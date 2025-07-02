from typing import Optional
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from statsmodels.tsa.seasonal import STL
from ..utils import get_coord_name, filter_by_season, get_or_create_dask_client, select_process_data, get_spatial_mean

# Try to import chunking utilities
try:
    from ..utils.chunking_utils import (
        get_optimal_chunks, rechunk_dataset, print_chunking_info, 
        dynamic_chunk_calculator, calculate_optimal_chunks_from_disk,
        inspect_disk_chunking, suggest_chunking_strategy
    )
    CHUNKING_AVAILABLE = True
except ImportError:
    CHUNKING_AVAILABLE = False

@xr.register_dataset_accessor("climate_timeseries")
class TimeSeriesAccessor:
    """
    Accessor for analyzing and visualizing climate time series from xarray datasets.
    Provides methods for extracting, processing, and visualizing time series
    with support for weighted spatial averaging, seasonal filtering, and time series decomposition.
    """

    # --------------------------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------------------------
    def __init__(self, xarray_obj):
        """Initialize the accessor with a Dataset object."""
        self._obj = xarray_obj

    # --------------------------------------------------------------------------
    # CHUNKING AND OPTIMIZATION METHODS
    # --------------------------------------------------------------------------
    def optimize_chunks(self, target_mb: float = 50, max_mb: float = 200,
                       variable: Optional[str] = None, time_freq: Optional[str] = None,
                       inplace: bool = False):
        """
        Optimize dataset chunking for better time series analysis performance.
        
        This method applies intelligent chunking strategies specifically tuned for
        time series operations. It analyzes the dataset structure and applies
        memory-efficient chunking that balances:
        - Memory usage (target chunk sizes)
        - Computational efficiency (parallel processing)
        - I/O performance (disk access patterns)
        
        The chunking strategy preserves spatial chunks from disk when beneficial
        and optimizes time dimension chunking for typical time series workflows.
        
        Parameters
        ----------
        target_mb : float, optional
            Target chunk size in megabytes. Defaults to 50 MB.
            This is the "sweet spot" for most time series operations.
        max_mb : float, optional
            Maximum chunk size in megabytes. Defaults to 200 MB.
            Hard limit to prevent memory exhaustion.
        variable : str, optional
            Variable to optimize chunking for. If None, optimizes for all variables.
            Focusing on a specific variable can yield better optimization.
        time_freq : str, optional
            Time frequency hint ('daily', 'monthly', 'hourly', '6hourly').
            Helps the algorithm make better chunking decisions.
        inplace : bool, optional
            DEPRECATED: In-place modification is not supported for xarray datasets.
            This parameter is kept for API compatibility but will raise an error if True.
            Always use False and reassign the result.
        
        Returns
        -------
        xr.Dataset or None
            Optimally chunked dataset if successful, None if chunking unavailable.
        
        Raises
        ------
        ValueError
            If inplace=True is requested (not supported).
        
        Examples
        --------
        >>> # Basic optimization for time series analysis
        >>> ds_optimized = ds.climate_timeseries.optimize_chunks(target_mb=100)
        >>> 
        >>> # Focus on specific variable with time frequency hint
        >>> ds_opt = ds.climate_timeseries.optimize_chunks(
        ...     variable='temperature', time_freq='daily', target_mb=75
        ... )
        
        Notes
        -----
        This method is a simplified interface to the sophisticated chunking
        system. For advanced control, use optimize_chunks_advanced().
        """
        # Parameter validation with detailed error messages
        if inplace:
            raise ValueError("In-place modification is not supported for xarray Dataset chunking. "
                           "Use inplace=False and reassign the result: ds = ds.climate_timeseries.optimize_chunks()")
        
        # Check if advanced chunking utilities are available
        if not CHUNKING_AVAILABLE:
            print("Warning: Chunking utilities not available. Skipping optimization.")
            return None
        
        try:
            # Apply sophisticated chunking strategy via the rechunk_dataset utility
            # This leverages disk-aware chunking and memory optimization strategies
            optimized_ds = rechunk_dataset(
                self._obj, target_mb=target_mb, max_mb=max_mb,
                variable=variable, time_freq=time_freq
            )
            return optimized_ds
                
        except Exception as e:
            print(f"Warning: Could not optimize chunks: {e}")
            return None

    def print_chunking_info(self, detailed: bool = False):
        """
        Print information about current dataset chunking.
        
        Provides a clear overview of current chunking configuration, which is
        essential for understanding memory usage and performance characteristics.
        
        Parameters
        ----------
        detailed : bool, optional
            Whether to print detailed per-variable information. Defaults to False.
            When True, shows chunking info for each variable individually.
        """
        if CHUNKING_AVAILABLE:
            # Use the advanced chunking info utility if available
            print_chunking_info(self._obj, detailed=detailed)
        else:
            # Fallback to basic information when chunking utilities unavailable
            print("Chunking information utilities not available.")
            print(f"Dataset shape: {dict(self._obj.sizes)}")
            total_size_mb = self._obj.nbytes / (1024**2)
            print(f"Total size: {total_size_mb:.1f} MB")

    def optimize_chunks_advanced(self, operation_type: str = 'timeseries',
                                 memory_limit_gb: Optional[float] = None,
                                 performance_priority: str = 'balanced',
                                 variable: Optional[str] = None,
                                 use_disk_chunks: bool = True,
                                 inplace: bool = False):
        """
        Advanced chunking optimization using sophisticated strategies.
        
        This method implements the advanced chunking strategy that:
        • Inspects on-disk chunking from file encoding
        • Calculates bytes per time step for optimal memory management
        • Chooses time chunks based on target memory usage and parallelization
        • Preserves spatial chunking when beneficial
        • Adapts to different operation types and performance priorities
        
        Parameters
        ----------
        operation_type : str, optional
            Type of operation to optimize for. Options:
            - 'timeseries': Time series analysis (trends, decomposition)
            - 'spatial': Spatial analysis and plotting  
            - 'statistical': Statistical computations
            - 'general': General purpose chunking
            - 'io': Input/output operations
            Default is 'timeseries'.
        memory_limit_gb : float, optional
            Memory limit in GB. If None, uses 25% of available system memory.
        performance_priority : str, optional
            Performance optimization priority. Options:
            - 'memory': Minimize memory usage
            - 'speed': Maximize computational speed  
            - 'balanced': Balance memory and speed
            Default is 'balanced'.
        variable : str, optional
            Variable to optimize chunking for. If None, optimizes for all variables.
        use_disk_chunks : bool, optional
            Whether to use disk-aware chunking strategy. Defaults to True.
        inplace : bool, optional
            DEPRECATED: In-place modification is not supported for xarray datasets.
            This parameter is kept for API compatibility but will raise an error if True.
        
        Returns
        -------
        xr.Dataset or None
            Optimally chunked dataset if inplace=False, None otherwise.
        
        Examples
        --------
        >>> # Optimize for time series analysis with memory priority
        >>> ds_opt = ds.climate_timeseries.optimize_chunks_advanced(
        ...     operation_type='timeseries', 
        ...     performance_priority='memory'
        ... )
        
        >>> # Optimize for spatial analysis with speed priority  
        >>> ds = ds.climate_timeseries.optimize_chunks_advanced(
        ...     operation_type='spatial',
        ...     performance_priority='speed'
        ... )
        """
        # Validate parameters first
        if inplace:
            raise ValueError("In-place modification is not supported for xarray Dataset chunking. "
                           "Use inplace=False and reassign the result.")
        
        if not CHUNKING_AVAILABLE:
            print("Warning: Advanced chunking utilities not available. Skipping optimization.")
            return None
        
        try:
            print(f"Applying advanced chunking strategy for '{operation_type}' operations...")
            
            if use_disk_chunks:
                # Use sophisticated disk-aware chunking
                chunks = calculate_optimal_chunks_from_disk(
                    self._obj, 
                    variable=variable
                )
                print("✓ Using disk-aware chunking strategy")
            else:
                # Use dynamic chunking calculator
                chunks = dynamic_chunk_calculator(
                    self._obj,
                    operation_type=operation_type,
                    memory_limit_gb=memory_limit_gb,
                    performance_priority=performance_priority
                )
                print("✓ Using dynamic chunking strategy")
            
            # Apply chunking
            optimized_ds = self._obj.chunk(chunks)
            print(f"✓ Applied chunking strategy: {chunks}")
            
            # Calculate and display memory information
            total_memory_mb = optimized_ds.nbytes / (1024**2)
            chunk_memory_estimate = 0
            for var_name, var in optimized_ds.data_vars.items():
                if hasattr(var.data, 'chunksize'):
                    chunk_memory_estimate = max(chunk_memory_estimate, var.data.chunksize / (1024**2))
            
            print(f"✓ Dataset size: {total_memory_mb:.1f} MB")
            if chunk_memory_estimate > 0:
                print(f"✓ Estimated chunk size: ~{chunk_memory_estimate:.1f} MB")
            
            if inplace:
                # Cannot truly modify in place due to xarray's immutable structure
                # This would require reassigning the entire dataset object which 
                # is not possible from within an accessor method
                raise ValueError("In-place modification is not supported for xarray Dataset chunking. "
                               "Use inplace=False and reassign the result: ds = ds.climate_timeseries.optimize_chunks_advanced()")
            else:
                return optimized_ds
                
        except Exception as e:
            print(f"Warning: Could not apply advanced chunking: {e}")
            return None if not inplace else None

    def analyze_chunking_strategy(self, variable: Optional[str] = None):
        """
        Analyze and suggest optimal chunking strategies for different use cases.
        
        This method inspects the dataset and provides recommendations for
        different types of climate analysis operations.
        
        Parameters
        ----------
        variable : str, optional
            Variable to analyze. If None, analyzes all variables.
        
        Examples
        --------
        >>> ds.climate_timeseries.analyze_chunking_strategy()
        """
        if not CHUNKING_AVAILABLE:
            print("Chunking analysis utilities not available.")
            return
        
        try:
            print("Climate Data Chunking Analysis")
            print("=" * 50)
            
            # Inspect disk chunking
            disk_info = inspect_disk_chunking(self._obj, variable)
            
            # Suggest strategies for different use cases
            use_cases = ['time_series', 'spatial_analysis', 'trend_analysis', 'memory_limited']
            
            print("\nRecommended chunking strategies:")
            for use_case in use_cases:
                strategy = suggest_chunking_strategy(self._obj, use_case)
                print(f"\n{use_case.replace('_', ' ').title()}:")
                print(f"  Target: {strategy['strategy']['target_mb']:.0f} MB chunks")
                print(f"  Max: {strategy['strategy']['max_mb']:.0f} MB chunks") 
                print(f"  Chunks: {strategy['chunks']}")
                print(f"  Use: {strategy['strategy']['description']}")
            
        except Exception as e:
            print(f"Warning: Could not analyze chunking strategy: {e}")

    def optimize_for_decomposition(self, variable: Optional[str] = None):
        """
        Optimize chunking specifically for STL time series decomposition.
        
        STL decomposition benefits from larger time chunks and smaller spatial chunks
        to minimize memory usage while maintaining good performance.
        
        Parameters
        ---------- 
        variable : str, optional
            Variable to optimize for. If None, optimizes for all variables.
        
        Returns
        -------
        xr.Dataset or None
            Optimally chunked dataset for STL decomposition.
        """
        return self.optimize_chunks_advanced(
            operation_type='timeseries',
            performance_priority='memory',
            variable=variable,
            inplace=False
        )

    # ==============================================================================
    # PUBLIC PLOTTING METHODS
    # ==============================================================================

    # --------------------------------------------------------------------------
    # A. Basic Time Series Plots
    # --------------------------------------------------------------------------
    def plot_time_series(self, variable='air', latitude=None, longitude=None, level=None,
                         time_range=None, season='annual', year=None,
                         area_weighted=True, figsize=(16, 10), save_plot_path=None,
                         optimize_chunks=True, chunk_target_mb=50):
        """
        Plot a time series of a spatially averaged variable.

        This function selects data for a given variable, performs spatial averaging
        over the specified domain, and plots the resulting time series.

        Parameters
        ----------
        variable : str, optional
            Name of the variable to plot. Defaults to 'air'.
        latitude : float, slice, or list, optional
            Latitude range for spatial averaging.
        longitude : float, slice, or list, optional
            Longitude range for spatial averaging.
        level : float, slice, or list, optional
            Vertical level selection.
        time_range : slice, optional
            Time range for the series.
        season : str, optional
            Seasonal filter. Defaults to 'annual'.
        year : int, optional
            Filter for a specific year.
        area_weighted : bool, optional
            If True, use latitude-based area weighting for the spatial mean. Defaults to True.
        figsize : tuple, optional
            Figure size. Defaults to (16, 10).
        save_plot_path : str or None, optional
            If provided, the path to save the plot figure.
        optimize_chunks : bool, optional
            Whether to automatically optimize chunking for performance. Defaults to True.
        chunk_target_mb : float, optional
            Target chunk size in MB for optimization. Defaults to 50 MB.

        Returns
        -------
        matplotlib.axes.Axes or None
            The Axes object of the plot, or None if no data could be plotted.
        """
        # Parameter validation
        if not isinstance(variable, str):
            raise TypeError("Variable must be a string")
        if variable not in self._obj.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset")
        if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
            raise ValueError("figsize must be a tuple or list of two numbers")
        
        # Optimize chunking if requested and available
        if optimize_chunks and CHUNKING_AVAILABLE:
            try:
                # Use advanced chunking strategy optimized for time series plotting
                dataset = self.optimize_chunks_advanced(
                    operation_type='timeseries',
                    performance_priority='balanced',
                    variable=variable,
                    inplace=False
                )
                if dataset is not None:
                    print(f"✓ Dataset optimized for time series plotting using advanced chunking strategy")
                else:
                    # Fallback to legacy chunking
                    dataset = rechunk_dataset(self._obj, target_mb=chunk_target_mb, 
                                            variable=variable, time_freq='daily')
                    print(f"Dataset rechunked for optimal performance (target: {chunk_target_mb} MB per chunk)")
            except Exception as e:
                print(f"Warning: Could not optimize chunks: {e}. Using original dataset.")
                dataset = self._obj
        else:
            dataset = self._obj
        
        get_or_create_dask_client()
        # --- Step 1: Select and process the data ---
        data_selected = select_process_data(
            dataset, variable, latitude, longitude, level, time_range, season, year
        )
        time_name = get_coord_name(data_selected, ['time', 't'])
        if not time_name or time_name not in data_selected.dims:
            raise ValueError("Time dimension not found for time series plot.")
        if data_selected.size == 0:
            print("Warning: No data to plot after selections.")
            return None

        # --- Step 2: Calculate the spatial mean time series ---
        ts_data = get_spatial_mean(data_selected, area_weighted)

        # --- Step 3: Ensure data is in memory for plotting ---
        if hasattr(ts_data, 'chunks') and ts_data.chunks:
            print("Computing time series...")
            with ProgressBar(): 
                ts_data = ts_data.compute()
        
        if ts_data.size == 0:
            print("Warning: Time series is empty after spatial averaging.")
            return None

        # Check for all NaN values
        if hasattr(ts_data, 'values') and np.isnan(ts_data.values).all():
            print("Warning: Time series contains only NaN values.")
            return None

        # --- Step 4: Create the plot ---
        plt.figure(figsize=figsize)
        ts_data.plot(marker='.')
        ax = plt.gca()

        # --- Step 5: Customize plot labels and title ---
        units = data_selected.attrs.get("units", "")
        long_name = data_selected.attrs.get("long_name", variable.replace('_', ' ').capitalize())
        ax.set_ylabel(f"{long_name} ({units})")
        ax.set_xlabel('Time')

        season_display = season.upper() if season.lower() != 'annual' else 'Annual'
        year_display = f" for {year}" if year is not None else ""
        weight_display = "Area-Weighted " if area_weighted and get_coord_name(data_selected, ['lat', 'latitude']) in data_selected.dims else ""
        ax.set_title(f"{season_display}{year_display}: {weight_display}Spatial Mean Time Series of {long_name} ({units})")

        # --- Step 6: Finalize and save the plot ---
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save_plot_path:
            plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved to: {save_plot_path}")
        return ax

    def plot_std_space(self, variable='air', latitude=None, longitude=None, level=None,
                       time_range=None, season='annual', year=None,
                       area_weighted=True, figsize=(16, 10), save_plot_path=None, title=None):
        """
        Plot a time series of the spatial standard deviation of a variable.

        This function calculates the standard deviation across the spatial domain
        for each time step and plots the resulting time series. This can be used
        to analyze the spatial variability of a field over time.

        Parameters
        ----------
        variable : str, optional
            Name of the variable to plot. Defaults to 'air'.
        latitude : float, slice, or list, optional
            Latitude range for the calculation.
        longitude : float, slice, or list, optional
            Longitude range for the calculation.
        level : float, slice, or list, optional
            Vertical level selection.
        time_range : slice, optional
            Time range for the series.
        season : str, optional
            Seasonal filter. Defaults to 'annual'.
        year : int, optional
            Filter for a specific year.
        area_weighted : bool, optional
            If True, use latitude-based area weighting for the standard deviation. Defaults to True.
        figsize : tuple, optional
            Figure size. Defaults to (16, 10).
        save_plot_path : str or None, optional
            If provided, the path to save the plot figure.
        title : str or None, optional
            Custom plot title. A default title is generated if not provided.

        Returns
        -------
        matplotlib.axes.Axes or None
            The Axes object of the plot, or None if no data could be plotted.
        """
        # Parameter validation
        if not isinstance(variable, str):
            raise TypeError("Variable must be a string")
        if variable not in self._obj.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset")
        if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
            raise ValueError("figsize must be a tuple or list of two numbers")
        
        get_or_create_dask_client()
        # --- Step 1: Select and process the data ---
        data_selected = select_process_data(
             self._obj, variable, latitude, longitude, level, time_range, season, year
        )
        time_name = get_coord_name(data_selected, ['time', 't'])
        if not time_name or time_name not in data_selected.dims:
            raise ValueError("Time dimension not found for spatial standard deviation plot.")
        if data_selected.size == 0:
            print("Warning: No data to plot after selections.")
            return None

        # --- Step 2: Calculate the time series of spatial standard deviation ---
        lat_name = get_coord_name(data_selected, ['lat', 'latitude'])
        lon_name = get_coord_name(data_selected, ['lon', 'longitude'])
        spatial_dims = [d for d in [lat_name, lon_name] if d and d in data_selected.dims]
        if not spatial_dims:
            raise ValueError("No spatial dimensions found for standard deviation calculation.")

        std_ts_data = None
        if area_weighted and lat_name in spatial_dims:
            # Weighted standard deviation
            weights = np.cos(np.deg2rad(data_selected[lat_name]))
            weights.name = "weights"
            std_ts_data = data_selected.weighted(weights).std(dim=spatial_dims, skipna=True)
            print("Calculating area-weighted spatial standard deviation time series.")
        else:
            # Unweighted standard deviation
            std_ts_data = data_selected.std(dim=spatial_dims, skipna=True)
            weight_msg = "(unweighted)" if lat_name in spatial_dims else ""
            print(f"Calculating simple spatial standard deviation {weight_msg} time series.")

        # --- Step 3: Ensure data is in memory for plotting ---
        if hasattr(std_ts_data, 'chunks') and std_ts_data.chunks:
            print("Computing spatial standard deviation time series...")
            with ProgressBar(): 
                std_ts_data = std_ts_data.compute()
        
        if std_ts_data.size == 0:
            print("Warning: Spatial standard deviation time series is empty.")
            return None

        # Check for all NaN values
        if hasattr(std_ts_data, 'values') and np.isnan(std_ts_data.values).all():
            print("Warning: Spatial standard deviation time series contains only NaN values.")
            return None

        # --- Step 4: Create the plot ---
        plt.figure(figsize=figsize)
        std_ts_data.plot(marker='.')
        ax = plt.gca()

        # --- Step 5: Customize plot labels and title ---
        units = data_selected.attrs.get("units", "")
        long_name = data_selected.attrs.get("long_name", variable.replace('_', ' ').capitalize())
        ax.set_ylabel(f"Spatial Std. Dev. ({units})" if units else "Spatial Std. Dev.")
        ax.set_xlabel('Time')

        if title is None:
            season_display = season.upper() if season.lower() != 'annual' else 'Annual'
            year_display = f" for {year}" if year is not None else ""
            weight_display = "Area-Weighted " if area_weighted and lat_name in spatial_dims else ""
            title = f"{season_display}{year_display}: Time Series of {weight_display}Spatial Std Dev of {long_name} ({units})"
        ax.set_title(title)

        # --- Step 6: Finalize and save the plot ---
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save_plot_path:
            plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved to: {save_plot_path}")
        return ax

    # --------------------------------------------------------------------------
    # C. Time Series Decomposition
    # --------------------------------------------------------------------------
    def decompose_time_series(self, variable='air', level=None, latitude=None, longitude=None,
                              time_range=None, season='annual', year=None,
                              stl_seasonal=13, stl_period=12, area_weighted=True,
                              plot_results=True, figsize=(16, 10), save_plot_path=None,
                              optimize_chunks=True, chunk_target_mb=75):
        """
        Decompose a time series into trend, seasonal, and residual components using STL.

        Seasonal-Trend decomposition using LOESS (STL) is a robust method for
        decomposing a time series. This function first creates a spatially-averaged
        time series and then applies the STL algorithm.

        Parameters
        ----------
        variable : str, optional
            Name of the variable to decompose. Defaults to 'air'.
        level : float, slice, or list, optional
            Vertical level selection.
        latitude : float, slice, or list, optional
            Latitude range for spatial averaging.
        longitude : float, slice, or list, optional
            Longitude range for spatial averaging.
        time_range : slice, optional
            Time range for the series.
        season : str, optional
            Seasonal filter. Defaults to 'annual'.
        year : int, optional
            Filter for a specific year.
        stl_seasonal : int, optional
            Length of the seasonal smoother for STL. Must be an odd integer. Defaults to 13.
        stl_period : int, optional
            The period of the seasonal component. For monthly data, this is typically 12. Defaults to 12.
        area_weighted : bool, optional
            If True, use area weighting for the spatial mean. Defaults to True.
        plot_results : bool, optional
            If True, plot the original series and its decomposed components. Defaults to True.
        figsize : tuple, optional
            Figure size for the plot. Defaults to (16, 10).
        save_plot_path : str or None, optional
            Path to save the decomposition plot.
        optimize_chunks : bool, optional
            Whether to automatically optimize chunking for STL performance. Defaults to True.
        chunk_target_mb : float, optional
            Target chunk size in MB for optimization. Defaults to 75 MB.

        Returns
        -------
        dict or (dict, matplotlib.figure.Figure) or None or (None, None)
            If `plot_results` is False, returns a dictionary containing the
            'original', 'trend', 'seasonal', and 'residual' components as pandas Series.
            If `plot_results` is True, returns a tuple of (dictionary, figure object).
            In error cases: returns None if `plot_results` is False, or (None, None) if `plot_results` is True.
        """
        # Parameter validation
        if not isinstance(variable, str):
            raise TypeError("Variable must be a string")
        if variable not in self._obj.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset")
        if not isinstance(stl_seasonal, int) or stl_seasonal <= 0:
            raise ValueError("stl_seasonal must be a positive integer")
        if not isinstance(stl_period, int) or stl_period <= 0:
            raise ValueError("stl_period must be a positive integer")
        if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
            raise ValueError("figsize must be a tuple or list of two numbers")
        
        # Optimize chunking for STL decomposition if requested
        if optimize_chunks and CHUNKING_AVAILABLE:
            try:
                # Use advanced chunking strategy optimized for time series decomposition
                dataset = self.optimize_for_decomposition(variable=variable, inplace=False)
                if dataset is not None:
                    print(f"✓ Dataset optimized for STL decomposition using advanced chunking strategy")
                else:
                    # Fallback to legacy chunking
                    dataset = rechunk_dataset(self._obj, target_mb=chunk_target_mb, 
                                            variable=variable, time_freq='monthly')
                    print(f"Dataset rechunked for STL decomposition (target: {chunk_target_mb} MB per chunk)")
            except Exception as e:
                print(f"Warning: Could not optimize chunks: {e}. Using original dataset.")
                dataset = self._obj
        else:
            dataset = self._obj
        
        get_or_create_dask_client()
        # --- Step 1: Select and process data for the time series ---
        data_selected = select_process_data(
             dataset, variable, latitude, longitude, level, time_range, season, year
        )
        time_name = get_coord_name(data_selected, ['time', 't'])
        if not time_name or time_name not in data_selected.dims:
            raise ValueError("Time dimension required for decomposition.")
        if data_selected.size == 0: 
            print("Warning: No data after selections.")
            return (None, None) if plot_results else None

        # --- Step 2: Compute the spatially-averaged time series ---
        ts_spatial_mean = get_spatial_mean(data_selected, area_weighted)
        if hasattr(ts_spatial_mean, 'chunks') and ts_spatial_mean.chunks:
            print("Computing mean time series for decomposition...")
            with ProgressBar(): 
                ts_spatial_mean = ts_spatial_mean.compute()
        if ts_spatial_mean.size == 0: 
            print("Warning: Time series empty after spatial mean.")
            return (None, None) if plot_results else None
        
        # --- Step 3: Convert the xarray DataArray to a pandas Series for STL ---
        ts_spatial_mean = ts_spatial_mean.squeeze(drop=True)
        if ts_spatial_mean.ndim > 1:
             raise ValueError(f"Spatially averaged data for STL still has >1 dimension: {ts_spatial_mean.dims}")

        try:
            ts_pd = ts_spatial_mean.to_series()
        except Exception as e_pd:
            raise ValueError(f"Could not convert to pandas Series for STL: {e_pd}")

        # --- Step 4: Prepare the time series for STL (drop NaNs, check length) ---
        ts_pd = ts_pd.dropna()
        if ts_pd.empty:
            print("Warning: Time series is empty or all NaN after processing for STL.")
            return (None, None) if plot_results else None
        if len(ts_pd) <= 2 * stl_period: 
            print(f"Warning: Time series length ({len(ts_pd)}) must be > 2 * stl_period ({2*stl_period}) for STL.")
            return (None, None) if plot_results else None
        
        # --- Step 5: Perform STL decomposition ---
        if stl_seasonal % 2 == 0:
            stl_seasonal += 1
            print(f"Adjusted stl_seasonal to be odd: {stl_seasonal}")

        print(f"Performing STL decomposition (period={stl_period}, seasonal_smooth={stl_seasonal})...")
        try:
            stl_result = STL(ts_pd, seasonal=stl_seasonal, period=stl_period, robust=True).fit()
        except Exception as e:
             print(f"Error: STL decomposition failed: {e}. Check time series properties (length, NaNs, period).")
             return (None, None) if plot_results else None

        results_dict = {
            'original': stl_result.observed,
            'trend': stl_result.trend,
            'seasonal': stl_result.seasonal,
            'residual': stl_result.resid
        }

        # --- Step 6: Plot the results if requested ---
        if plot_results:
            print("Plotting decomposition results...")
            fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
            units = data_selected.attrs.get("units", "")
            long_name = data_selected.attrs.get("long_name", variable.replace('_', ' ').capitalize())
            
            # Plot each component
            axes[0].plot(results_dict['original'].index, results_dict['original'].values, label='Observed')
            axes[0].set_ylabel(f"Observed ({units})")
            title_prefix = f'{season.upper() if season.lower() != "annual" else "Annual"}'
            year_info = f" for {year}" if year else ""
            axes[0].set_title(f'{title_prefix}{year_info} Time Series Decomposition: {long_name}')

            axes[1].plot(results_dict['trend'].index, results_dict['trend'].values, label='Trend')
            axes[1].set_ylabel(f"Trend ({units})")
            axes[2].plot(results_dict['seasonal'].index, results_dict['seasonal'].values, label='Seasonal')
            axes[2].set_ylabel(f"Seasonal ({units})")
            axes[3].plot(results_dict['residual'].index, results_dict['residual'].values, label='Residual', marker='.', linestyle='None', markersize=3, alpha=0.7)
            axes[3].axhline(0, color='grey', linestyle='--', alpha=0.5)
            axes[3].set_ylabel(f"Residual ({units})")
            axes[3].set_xlabel("Time")

            # Finalize and save the plot
            for ax_i in axes:
                ax_i.grid(True, linestyle='--', alpha=0.6)
                ax_i.legend(loc='upper left', fontsize='small')
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            
            if save_plot_path:
                try:
                    plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
                    print(f"Plot saved to: {save_plot_path}")
                except Exception as e:
                    print(f"Warning: Could not save plot to {save_plot_path}: {e}")
            return results_dict, fig
        else:
            return results_dict

__all__ = ['TimeSeriesAccessor']
