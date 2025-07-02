from typing import Optional
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from dask.diagnostics import ProgressBar
import pandas as pd
from statsmodels.tsa.seasonal import STL
import warnings
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from dask.distributed import Client, LocalCluster
import logging
from ..utils import get_coord_name, select_process_data, get_spatial_mean, get_or_create_dask_client

# Try to import chunking utilities
try:
    from ..utils.chunking_utils import get_optimal_chunks, rechunk_dataset, suggest_chunking_strategy
    CHUNKING_AVAILABLE = True
except ImportError:
    CHUNKING_AVAILABLE = False

@xr.register_dataset_accessor("climate_trends")
class TrendsAccessor:
    """
    Accessor for analyzing and visualizing trend patterns in climate datasets.
    
    This accessor provides methods to analyze climate data trends from xarray Datasets
    using statistical decomposition techniques. It supports trend analysis using STL 
    decomposition and linear regression, with proper spatial (area-weighted) averaging,
    seasonal filtering, and robust visualization options.
    
    The accessor handles common climate data formats with automatic detection of 
    coordinate names (lat, lon, time, level) for maximum compatibility across 
    different datasets and model output conventions.
    """
    
    # --------------------------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------------------------
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    # --------------------------------------------------------------------------
    # CHUNKING AND OPTIMIZATION METHODS
    # --------------------------------------------------------------------------
    def optimize_for_trends(self, variable: Optional[str] = None, 
                           use_case: str = 'trend_analysis') -> Optional[xr.Dataset]:
        """
        Optimize dataset chunking specifically for trend analysis workflows.
        
        Trend analysis has different computational patterns than regular time series:
        - Requires larger time chunks for STL decomposition stability
        - Benefits from smaller spatial chunks for parallel trend calculations
        - Needs balanced memory usage for statistical operations
        
        This method applies pre-configured chunking strategies optimized for
        various trend analysis scenarios.
        
        Parameters
        ----------
        variable : str, optional
            Variable to optimize for. If None, optimizes for all variables.
            Focusing on a specific variable provides better optimization.
        use_case : str, optional
            Specific trend analysis use case. Options:
            - 'trend_analysis': General trend calculation (default)
            - 'spatial_trends': Grid-point trend analysis
            - 'memory_limited': Conservative chunking for low-memory systems
        
        Returns
        -------
        xr.Dataset or None
            Optimized dataset with trend-analysis-friendly chunking,
            or None if optimization fails or chunking utilities unavailable.
        
        Examples
        --------
        >>> # Optimize for general trend analysis
        >>> ds_opt = ds.climate_trends.optimize_for_trends()
        >>> 
        >>> # Optimize for spatial trend calculations
        >>> ds_spatial = ds.climate_trends.optimize_for_trends(
        ...     variable='temperature', use_case='spatial_trends'
        ... )
        
        Notes
        -----
        This optimization is particularly beneficial before calling:
        - calculate_trend()
        - plot_spatial_trends()
        - Any STL decomposition operations
        """
        if not CHUNKING_AVAILABLE:
            print("Warning: Chunking optimization not available.")
            return None
        
        try:
            # Get pre-configured strategy for the specified use case
            strategy = suggest_chunking_strategy(self._obj, target_use_case=use_case)
            print(f"Applying chunking strategy: {strategy['strategy']['description']}")
            print(f"Recommended chunks: {strategy['chunks']}")
            
            # Apply the chunking strategy
            optimized_ds = self._obj.chunk(strategy['chunks'])
            return optimized_ds
            
        except Exception as e:
            print(f"Warning: Could not optimize for trends: {e}")
            return None

    # --------------------------------------------------------------------------
    # INTERNAL HELPER METHODS
    # --------------------------------------------------------------------------


    # ==============================================================================
    # PUBLIC TREND ANALYSIS METHODS
    # ==============================================================================

    # --------------------------------------------------------------------------
    # A. Time Series Trend (Point or Regional Average)
    # --------------------------------------------------------------------------
    def calculate_trend(self,
                        variable='air',
                        latitude=None,
                        longitude=None,
                        level=None,
                        frequency='M',
                        season='annual',
                        area_weighted=True,
                        period=12,
                        plot=True,
                        return_results=False,
                        save_plot_path=None
                        ):
        """
        Calculate and visualize the trend of a time series for a specified variable and region.

        This method performs the following steps:
        1. Selects the data for the given variable and spatial/level domain.
        2. Applies a seasonal filter.
        3. Computes a spatial average (area-weighted or simple) to get a 1D time series.
        4. Applies Seasonal-Trend decomposition using LOESS (STL) to isolate the trend component.
        5. Fits a linear regression to the trend component to calculate the slope, p-value, etc.
        6. Optionally plots the STL trend and the linear fit.

        Parameters
        ----------
        variable : str, optional
            Name of the variable to analyze. Defaults to 'air'.
        latitude : float, slice, list, or None, optional
            Latitude selection for the analysis domain. Can be a single point, a slice,
            or a list of values. If None, the full latitude range is used.
        longitude : float, slice, list, or None, optional
            Longitude selection for the analysis domain.
        level : float, slice, or None, optional
            Vertical level selection. If a slice is provided, data is averaged over the levels.
            If None and multiple levels exist, the first level is used by default.
        frequency : {'Y', 'M', 'D'}, optional
            The time frequency used to report the slope of the trend line.
            'Y' for per year, 'M' for per month, 'D' for per day. Defaults to 'M'.
        season : str, optional
            Seasonal filter to apply before analysis. Supported: 'annual', 'jjas',
            'djf', 'mam', 'jja', 'son'. Defaults to 'annual'.
        area_weighted : bool, optional
            If True, performs area-weighted spatial averaging using latitude weights.
            Defaults to True. Ignored for point selections.
        period : int, optional
            The periodicity of the seasonal component for STL decomposition.
            For monthly data, this is typically 12. Defaults to 12.
        plot : bool, optional
            If True, a plot of the trend component and its linear fit is generated.
            Defaults to True.
        return_results : bool, optional
            If True, a dictionary containing the detailed results of the analysis is returned.
            Defaults to False.
        save_plot_path : str or None, optional
            If provided, the path where the plot will be saved.

        Returns
        -------
        dict or None
            If `return_results` is True, returns a dictionary containing the analysis results,
            including the trend component (pandas Series), the predicted trend line,
            region details, and a DataFrame with trend statistics (slope, p-value, etc.).
            Otherwise, returns None.

        Raises
        ------
        ValueError
            If the variable is not found, no time coordinate is present, or if the
            data selection and processing result in an empty time series.
        """
        # Parameter validation
        if not isinstance(variable, str):
            raise TypeError("Variable must be a string")
        if frequency.upper() not in ['Y', 'M', 'D']:
            raise ValueError("frequency must be one of 'Y', 'M', 'D', 'y', 'm', or 'd'")
        if not isinstance(period, int) or period <= 0:
            raise ValueError("period must be a positive integer")
        
        # --- Step 1: Initialize Dask and get coordinates ---
        get_or_create_dask_client()
        time_coord_name = get_coord_name(self._obj, ['time', 't'])
        if not time_coord_name:
            raise ValueError("Dataset must contain a recognizable time coordinate.")
        if variable not in self._obj.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset")
        
        # --- Step 2: Select and process data using the centralized utility ---
        data_selected = select_process_data(
            self._obj, variable, latitude, longitude, level, season=season
        )

        is_point = (isinstance(latitude, (int, float)) and isinstance(longitude, (int, float)))

        # --- Step 3: Compute spatial mean to get a 1D time series ---
        processed_ts_da = get_spatial_mean(data_selected, area_weighted and not is_point)
        
        if time_coord_name not in processed_ts_da.dims or processed_ts_da.sizes[time_coord_name] == 0:
            raise ValueError(f"Selection resulted in zero time points for variable '{variable}' and season '{season}'.")

        if hasattr(processed_ts_da, 'chunks') and processed_ts_da.chunks:
            with ProgressBar():
                processed_ts_da = processed_ts_da.compute()

        # --- Step 4: Convert to pandas Series for STL ---
        try:
            ts_pd = processed_ts_da.to_series()
        except Exception:
            try:
                ts_pd = processed_ts_da.to_pandas()
            except Exception as e:
                raise ValueError(f"Could not convert DataArray to pandas Series: {e}")

        if not isinstance(ts_pd, pd.Series):
            if isinstance(ts_pd, pd.DataFrame) and len(ts_pd.columns) == 1:
                    ts_pd = ts_pd.iloc[:, 0]
            else:
                raise TypeError(f"Could not convert DataArray to a pandas Series. Got type: {type(ts_pd)}")

        # --- Step 5: Apply STL decomposition to isolate the trend ---
        if ts_pd.isnull().all():
            raise ValueError("Time series is all NaNs after selection/averaging.")

        original_index = ts_pd.index
        ts_pd_clean = ts_pd.dropna()

        if ts_pd_clean.empty:
            raise ValueError("Time series is all NaNs after dropping NaN values.")
            
        min_stl_len = 2 * period
        if len(ts_pd_clean) < min_stl_len:
            raise ValueError(f"Time series length ({len(ts_pd_clean)}) for STL is less than required minimum ({min_stl_len}). Need at least 2*period.")

        print("Applying STL decomposition...")
        try:
            stl_result = STL(ts_pd_clean, period=period, robust=True).fit()
        except Exception as e:
            raise ValueError(f"STL decomposition failed: {e}")
        
        trend_component = stl_result.trend.reindex(original_index)

        # --- Step 6: Perform linear regression on the trend component ---
        print("Performing linear regression...")
        trend_component_clean = trend_component.dropna()
        if trend_component_clean.empty:
            raise ValueError("Trend component is all NaNs after STL and dropna.")

        if pd.api.types.is_datetime64_any_dtype(trend_component_clean.index):
            # Calculate time in numeric units for regression
            first_date = trend_component_clean.index.min()
            frequency_upper = frequency.upper()
            if frequency_upper == 'M':
                scale_seconds = 24 * 3600 * (365.25 / 12)
                time_unit_for_slope = "month"
            elif frequency_upper == 'D':
                scale_seconds = 24 * 3600
                time_unit_for_slope = "day"
            elif frequency_upper == 'Y':
                scale_seconds = 24 * 3600 * 365.25
                time_unit_for_slope = "year"
            else:
                print(f"Warning: Unknown frequency '{frequency}', defaulting to years for slope calculation.")
                scale_seconds = 24 * 3600 * 365.25
                time_unit_for_slope = "year"
            
            x_numeric_for_regression = ((trend_component_clean.index - first_date).total_seconds() / scale_seconds).values
        
        elif pd.api.types.is_numeric_dtype(trend_component_clean.index):
            x_numeric_for_regression = trend_component_clean.index.values
            time_unit_for_slope = "index_unit"
        else:
            raise TypeError(f"Trend index type ({trend_component_clean.index.dtype}) not recognized for regression.")

        if len(x_numeric_for_regression) < 2:
             raise ValueError("Not enough data points in the cleaned trend component for linear regression.")

        y_values_for_regression = trend_component_clean.values
        
        slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x_numeric_for_regression, y_values_for_regression)
        
        y_pred_values_on_clean_index = intercept + slope * x_numeric_for_regression
        predicted_trend_series = pd.Series(y_pred_values_on_clean_index, index=trend_component_clean.index).reindex(original_index)
        
        trend_stats_df = pd.DataFrame({
            'statistic': ['slope', 'intercept', 'p_value', 'r_value', 'r_squared', 'standard_error_slope'],
            'value': [slope, intercept, p_value, r_value, r_value**2, slope_std_error]
        })
         
        # --- Step 7: Generate plot if requested ---
        if plot:
            print("Generating plot...")
            plt.figure(figsize=(16, 10), dpi=100)
            
            # Plot the raw STL trend and the linear fit
            plt.scatter(trend_component.index, trend_component.values, color='blue', alpha=0.5, s=10, 
                       label='STL Trend Component')
            
            units_label = processed_ts_da.attrs.get('units', '')
            slope_label = f'Linear Fit (Slope: {slope:.3e} {units_label}/{time_unit_for_slope})'
            plt.plot(predicted_trend_series.index, predicted_trend_series.values, color='red', linewidth=2, 
                    label=slope_label)

            # Create a descriptive title
            title_parts = [f"Trend: {variable.capitalize()}"]
            ylabel = f'{variable.capitalize()} Trend' + (f' ({units_label})' if units_label else '')
            
            region_str = "Global"
            if latitude is not None or longitude is not None:
                region_str = "Regional" if isinstance(latitude, (slice, list)) or isinstance(longitude, (slice, list)) else "Point"
            title_parts.append(f"({region_str} Analysis)")
            
            if season.lower() != 'annual':
                title_parts.append(f"Season={season.upper()}")
            
            full_title = " ".join(title_parts)
            full_title += "\n(STL Trend + Linear Regression)"

            plt.title(full_title, fontsize=14)
            plt.xlabel('Time', fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            ax = plt.gca()
            try:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both'))
            except TypeError:
                print("Warning: Could not set major locator for x-axis due to index type.")

            plt.tight_layout()
            if save_plot_path is not None:
                try:
                    plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
                    print(f"Plot saved to: {save_plot_path}")
                except Exception as e:
                    print(f"Warning: Could not save plot to {save_plot_path}: {e}")
            plt.show()

        # --- Step 8: Return detailed results if requested ---
        if return_results:
            results = {
                'calculation_type': region_str,
                'trend_component': trend_component,
                'predicted_trend_line': predicted_trend_series,
                'area_weighted': area_weighted,
                'region_details': {'variable': variable, 'season': season},
                'stl_period': period,
                'trend_statistics': trend_stats_df,
                'time_unit_of_slope': time_unit_for_slope
            }
            return results
        return None
        
    # --------------------------------------------------------------------------
    # B. Spatial Trend Analysis (Pixel-by-pixel)
    # --------------------------------------------------------------------------
    def calculate_spatial_trends(self,
                           variable='air',
                           latitude=None,
                           longitude=None,
                           time_range=None,
                           level=None,
                           season='annual',
                           num_years=1, 
                           n_workers=4,
                           robust_stl=True,
                           period=12,
                           plot_map=True,
                           land_only=False,
                           save_plot_path=None,
                           cmap='coolwarm',
                           optimize_chunks=True,
                           chunk_target_mb=100):
        """
        Calculate and visualize spatial trends across a geographic domain.

        This method computes the trend at each grid point over a specified time period
        and spatial domain. It leverages Dask for parallel processing to efficiently
        handle large datasets. The trend is calculated by applying STL decomposition
        and linear regression to the time series of each grid cell.
        
        The trend is calculated robustly by performing a linear regression against
        time (converted to fractional years), making the calculation independent
        of the data's native time frequency.

        Parameters
        ----------
        variable : str, optional
            Name of the variable for which to calculate trends. Defaults to 'air'.
        latitude : slice, optional
            A slice defining the latitude range for the analysis. Defaults to the full range.
        longitude : slice, optional
            A slice defining the longitude range for the analysis. Defaults to the full range.
        time_range : slice, optional
            A slice defining the time period for the trend analysis. Defaults to the full range.
        level : float or None, optional
            A single vertical level to select for the analysis. If None and multiple levels
            exist, the first level is used by default.
        season : str, optional
            Seasonal filter to apply before analysis. Defaults to 'annual'.
        num_years : int, optional
            The number of years over which the trend should be reported (e.g., 1 for
            trend per year, 10 for trend per decade). Defaults to 1.
        n_workers : int, optional
            The number of Dask workers to use for parallel computation. Defaults to 4.
        robust_stl : bool, optional
            If True, use a robust version of the STL algorithm, which is less sensitive
            to outliers. Defaults to True.
        period : int, optional
            The periodicity of the seasonal component for STL. Defaults to 12.
        plot_map : bool, optional
            If True, plots the resulting spatial trend map. Defaults to True.
        land_only : bool, optional
            If True, the output map will mask ocean areas. Defaults to False.
        save_plot_path : str or None, optional
            Path to save the output trend map plot.
        cmap : str, optional
            The colormap to use for the trend map plot. Defaults to 'coolwarm'.
        optimize_chunks : bool, optional
            Whether to optimize chunking for spatial trends calculation. Defaults to True.
        chunk_target_mb : float, optional
            Target chunk size in MB for spatial operations. Defaults to 100 MB.

        Returns
        -------
        xr.DataArray
            A DataArray containing the calculated trend values for each grid point,
            typically in units of [variable_units / num_years].

        Raises
        ------
        ValueError
            If essential coordinates (time, lat, lon) are not found, or if the
            data selection results in insufficient data for trend calculation.
        """
        
        # Parameter validation
        if not isinstance(variable, str):
            raise TypeError("Variable must be a string")
        if not isinstance(num_years, int) or num_years <= 0:
            raise ValueError("num_years must be a positive integer")
        if not isinstance(n_workers, int) or n_workers <= 0:
            raise ValueError("n_workers must be a positive integer")
        if not isinstance(period, int) or period <= 0:
            raise ValueError("period must be a positive integer")
        
        # Optimize chunking for spatial trends if requested
        if optimize_chunks and CHUNKING_AVAILABLE:
            try:
                dataset = rechunk_dataset(self._obj, target_mb=chunk_target_mb, 
                                        variable=variable, time_freq='monthly')
                print(f"Dataset optimized for spatial trends (target: {chunk_target_mb} MB per chunk)")
            except Exception as e:
                print(f"Warning: Could not optimize chunks: {e}. Using original dataset.")
                dataset = self._obj
        else:
            dataset = self._obj
        
        # --- Step 1: Set up labels and coordinates ---
        if num_years == 1: 
            period_str_label = "year"
        elif num_years == 10: 
            period_str_label = "decade"
        else: 
            period_str_label = f"{num_years} years"

        time_coord_name = get_coord_name(dataset, ['time', 't'])
        lat_coord_name = get_coord_name(dataset, ['lat', 'latitude'])
        lon_coord_name = get_coord_name(dataset, ['lon', 'longitude'])
        
        if not all([time_coord_name, lat_coord_name, lon_coord_name]):
            raise ValueError("Dataset must contain time, latitude, and longitude for spatial trends.")
        if variable not in dataset.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset")

        # --- Step 2: Use existing Dask client or create one for this operation ---
        existing_client = None
        cluster = None
        client = None
        
        try:
            # Try to get existing Dask client
            existing_client = Client.current()
            print(f"Using existing Dask client: {existing_client.dashboard_link}")
        except (ValueError, AttributeError):
            # No existing client, create a new one
            try:
                cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, silence_logs=logging.WARNING)
                client = Client(cluster)
                print(f"Created new Dask client for spatial trends: {client.dashboard_link}")
            except Exception as e:
                print(f"Warning: Could not create Dask cluster: {e}. Will use synchronous computation.")
        
        try:
            # --- Step 3: Select, filter, and prepare the data using the utility ---
            data_var = select_process_data(
                dataset, variable, latitude, longitude, level, time_range, season
            )
            
            if data_var[time_coord_name].size < 2 * period:
                raise ValueError(f"Insufficient time points ({data_var[time_coord_name].size}) after filtering. Need at least {2 * period}.")
            
            print(f"Data selected for spatial trends: {data_var.sizes}")
            level_selection_info_title = ""
            level_coord_name = get_coord_name(data_var, ['level', 'lev', 'plev'])
            if level_coord_name and level_coord_name in data_var.coords:
                level_selection_info_title = f"Level={data_var[level_coord_name].item()}"
            
            # --- Step 4: Define the function to calculate trend for a single grid cell ---
            def apply_stl_slope_spatial(da_1d_time_series, time_coord_array):
                """Apply STL decomposition and linear regression to a single grid cell time series."""
                try:
                    values = np.asarray(da_1d_time_series).squeeze()
                    time_coords = pd.to_datetime(np.asarray(time_coord_array).squeeze())

                    # a. Check for sufficient valid data
                    min_pts_for_stl = 2 * period
                    if values.ndim == 0 or len(values) < min_pts_for_stl or np.isnan(values).all():
                        return np.nan
                    
                    valid_mask = ~np.isnan(values)
                    num_valid_pts = np.sum(valid_mask)
                    if num_valid_pts < min_pts_for_stl:
                        return np.nan
                    
                    ts_for_stl = pd.Series(values[valid_mask], index=time_coords[valid_mask])

                    # b. Apply STL decomposition
                    stl_result = STL(ts_for_stl, period=period, robust=robust_stl,
                                     low_pass_jump=period//2,
                                     trend_jump=period//2,
                                     seasonal_jump=period//2
                                    ).fit(iter=2)
                    trend = stl_result.trend

                    if trend.isnull().all(): 
                        return np.nan

                    # c. Perform linear regression on the trend component
                    trend_clean = trend.dropna()
                    if len(trend_clean) < 2: 
                        return np.nan
                    
                    first_date = trend_clean.index.min()
                    scale_seconds_in_year = 24 * 3600 * 365.25
                    
                    x_numeric_for_regression = ((trend_clean.index - first_date).total_seconds() / scale_seconds_in_year).values
                    y_values_for_regression = trend_clean.values
                    
                    slope_per_year, _, _, _, _ = stats.linregress(x_numeric_for_regression, y_values_for_regression)

                    if np.isnan(slope_per_year): 
                        return np.nan

                    # d. Return slope scaled by the desired number of years
                    return slope_per_year * num_years
                except Exception:
                    return np.nan

            # --- Step 5: Chunk data and apply the trend function in parallel with Dask ---
            data_var = data_var.chunk({time_coord_name: -1, 
                                       lat_coord_name: 'auto',
                                       lon_coord_name: 'auto'})

            print("Computing spatial trends in parallel with xarray.apply_ufunc...")
            trend_da = xr.apply_ufunc(
                apply_stl_slope_spatial,
                data_var,
                data_var[time_coord_name],
                input_core_dims=[[time_coord_name], [time_coord_name]],
                output_core_dims=[[]],
                exclude_dims=set((time_coord_name,)),
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'allow_rechunk': True} 
            ).rename(f"{variable}_trend_per_{period_str_label}")

            # --- Step 6: Trigger computation and get the results ---
            with ProgressBar(dt=2.0):
                trend_computed_map = trend_da.compute()
            print("Spatial trend computation complete.")

            # --- Step 7: Plot the resulting trend map if requested ---
            if plot_map:
                print("Generating spatial trend map...")
                try:
                    start_time_str = pd.to_datetime(data_var[time_coord_name].min().item()).strftime('%Y-%m')
                    end_time_str = pd.to_datetime(data_var[time_coord_name].max().item()).strftime('%Y-%m')
                    time_period_title_str = f"{start_time_str} to {end_time_str}"
                except Exception: time_period_title_str = "Selected Time Period"

                data_units = data_var.attrs.get('units', '')
                var_long_name = data_var.attrs.get('long_name', variable)
                cbar_label_str = f"Trend ({data_units} / {period_str_label})" if data_units else f"Trend ({period_str_label})"

                fig = plt.figure(figsize=(14, 8))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

                # Create the plot using contourf for filled contours
                contour_plot = trend_computed_map.plot.contourf(
                    ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                    levels=30,
                    robust=True,
                    extend='both',
                    cbar_kwargs={'label': cbar_label_str, 'orientation': 'vertical', 'shrink': 0.8, 'pad':0.05}
                )
                if contour_plot.colorbar:
                    contour_plot.colorbar.set_label(cbar_label_str, size=12)
                    contour_plot.colorbar.ax.tick_params(labelsize=10)

                # Add geographic features
                if land_only:
                    ax.add_feature(cfeature.OCEAN, zorder=2, facecolor='lightgrey')
                    ax.add_feature(cfeature.LAND, zorder=1, facecolor='white')
                    ax.coastlines(zorder=3, linewidth=0.8)
                    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3, linewidth=0.6)
                else:
                    ax.coastlines(linewidth=0.8)
                    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6)
                
                # Customize gridlines and labels
                gl = ax.gridlines(draw_labels=True, linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = False; gl.right_labels = False
                gl.xlabel_style = {'size': 10}; gl.ylabel_style = {'size': 10}
                
                # Set a descriptive title
                season_title_str = season.upper() if season.lower() != 'annual' else 'Annual'
                plot_title = (f"{season_title_str} {var_long_name.capitalize()} Trend ({period_str_label})\n"
                              f"{time_period_title_str}")
                if level_selection_info_title: plot_title += f" at {level_selection_info_title}"
                ax.set_title(plot_title, fontsize=14)

                plt.tight_layout(pad=1.5)
                if save_plot_path:
                    try:
                        plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
                        print(f"Plot saved to {save_plot_path}")
                    except Exception as e:
                        print(f"Warning: Could not save plot to {save_plot_path}: {e}")
                plt.show()

            # --- Step 8: Return the computed trend data ---
            return trend_computed_map

        except Exception as e:
            print(f"An error occurred during spatial trend processing: {e}")
            raise
        finally:
            # --- Step 9: Ensure Dask client and cluster are closed (only if we created them) ---
            if client: 
                client.close()
            if cluster: 
                cluster.close()
            if client or cluster:
                print("Dask client and cluster for spatial trends closed.")

__all__ = ['TrendsAccessor']
