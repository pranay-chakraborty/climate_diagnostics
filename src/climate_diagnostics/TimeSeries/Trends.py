from typing import Optional
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from dask.diagnostics import ProgressBar
import pandas as pd
import warnings
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from dask.distributed import Client, LocalCluster
import logging
from ..utils import get_coord_name, select_process_data, get_spatial_mean, get_or_create_dask_client
from ..utils.plot_utils import get_projection

# Try to import statsmodels for trend analysis
try:
    from statsmodels.tsa.seasonal import STL
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

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
    # INITIALIZATION AND HELPERS
    # --------------------------------------------------------------------------
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _warn_if_not_chunked(self, operation_name: str) -> None:
        """
        Warn users if data is not chunked when it might benefit from chunking.
        
        Parameters
        ----------
        operation_name : str
            Name of the operation being performed.
        """
        # Check if any data variables use Dask arrays
        has_dask_arrays = any(
            hasattr(var.data, 'chunks') for var in self._obj.data_vars.values()
        )
        
        if not has_dask_arrays:
            warnings.warn(
                f"Data is not chunked for '{operation_name}'. For large datasets, "
                f"consider chunking your data before calling this method to improve "
                f"memory efficiency and enable parallel processing. "
                f"Example: data = data.chunk({'time': 120, 'lat': 50, 'lon': 50})",
                UserWarning,
                stacklevel=3
            )


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
                        time_range=None,
                        season='annual',
                        year=None,
                        frequency='M',
                        area_weighted=True,
                        period=12,
                        plot=True,
                        return_results=False,
                        save_plot_path=None,
                        title=None
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
        time_range : slice, optional
            A slice defining the time period for the analysis. Defaults to the full range.
        season : str, optional
            Seasonal filter to apply before analysis. Supported: 'annual', 'jjas',
            'djf', 'mam', 'jja', 'son'. Defaults to 'annual'.
        year : int, optional
            Filter data to a specific year. If provided, only data from this year
            will be used in the analysis. Defaults to None (use all years).
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
        title : str, optional
            The title for the plot. If not provided, a descriptive title will be
            generated automatically.

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
        # Check for statsmodels availability
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels is required for trend analysis. "
                "Install it with: pip install statsmodels"
            )
        
        # Warn if data is not chunked
        self._warn_if_not_chunked("trend calculation")
        
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
            self._obj, variable, latitude, longitude, level, time_range, season, year
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
        except (ValueError, TypeError, AttributeError):
            try:
                ts_pd = processed_ts_da.to_pandas()
            except (ValueError, TypeError, AttributeError) as e:
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

        warnings.warn("Applying STL decomposition...", UserWarning)
        try:
            stl_result = STL(ts_pd_clean, period=period, robust=True).fit()
        except (ValueError, ImportError, RuntimeError) as e:
            raise ValueError(f"STL decomposition failed: {e}")
        
        trend_component = stl_result.trend.reindex(original_index)

        # --- Step 6: Perform linear regression on the trend component ---
        warnings.warn("Performing linear regression...", UserWarning)
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
                warnings.warn(f"Unknown frequency '{frequency}', defaulting to years for slope calculation.", UserWarning)
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
            warnings.warn("Generating plot...", UserWarning)
            plt.figure(figsize=(16, 10), dpi=100)
            
            # Plot the raw STL trend and the linear fit
            plt.scatter(trend_component.index, trend_component.values, color='blue', alpha=0.5, s=10, 
                       label='STL Trend Component')
            
            units_label = processed_ts_da.attrs.get('units', '')
            slope_label = f'Linear Fit (Slope: {slope:.3e} {units_label}/{time_unit_for_slope})'
            plt.plot(predicted_trend_series.index, predicted_trend_series.values, color='red', linewidth=2, 
                    label=slope_label)

            # Create a descriptive title
            ylabel = f'{variable.capitalize()} Trend' + (f' ({units_label})' if units_label else '')
            if title is None:
                title_parts = [f"Trend: {variable.capitalize()}"]
                
                region_str = "Global"
                if latitude is not None or longitude is not None:
                    region_str = "Regional" if isinstance(latitude, (slice, list)) or isinstance(longitude, (slice, list)) else "Point"
                title_parts.append(f"({region_str} Analysis)")
                
                if season.lower() != 'annual':
                    title_parts.append(f"Season={season.upper()}")
                
                full_title = " ".join(title_parts)
                full_title += "\n(STL Trend + Linear Regression)"
            else:
                full_title = title

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
                warnings.warn("Could not set major locator for x-axis due to index type.", UserWarning)

            plt.tight_layout()
            if save_plot_path is not None:
                try:
                    plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
                    warnings.warn(f"Plot saved to: {save_plot_path}", UserWarning)
                except (OSError, IOError, ValueError) as e:
                    warnings.warn(f"Could not save plot to {save_plot_path}: {e}", UserWarning)
            # Note: plt.show() removed - user should call it explicitly if needed

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
                           year=None,
                           frequency='Y',
                           robust_stl=True,
                           period=12,
                           plot_map=True,
                           figsize=(14, 8),
                           cmap='coolwarm',
                           levels=30,
                           land_only=False,
                           projection='PlateCarree',
                           title=None,
                           save_plot_path=None):
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
        year : int, optional
            Filter data to a specific year. If provided, only data from this year
            will be used in the analysis. Defaults to None (use all years).
        frequency : {'Y', 'M', 'D'}, optional
            The time frequency used to report the slope of the trend line.
            'Y' for per year, 'M' for per month, 'D' for per day. Defaults to 'Y'.
        robust_stl : bool, optional
            If True, use a robust version of the STL algorithm, which is less sensitive
            to outliers. Defaults to True.
        period : int, optional
            The periodicity of the seasonal component for STL. Defaults to 12.
        plot_map : bool, optional
            If True, plots the resulting spatial trend map. Defaults to True.
        figsize : tuple, optional
            Figure size for the spatial trend map. Defaults to (14, 8).
        cmap : str, optional
            The colormap to use for the trend map plot. Defaults to 'coolwarm'.
        levels : int, optional
            Number of contour levels for the spatial trend map. Defaults to 30.
        land_only : bool, optional
            If True, the output map will mask ocean areas. Defaults to False.
        projection : str, optional
            The name of the cartopy projection to use. Defaults to 'PlateCarree'.
        title : str, optional
            The title for the plot. If not provided, a descriptive title will be
            generated automatically.
        save_plot_path : str or None, optional
            Path to save the output trend map plot.

        Returns
        -------
        xr.DataArray
            A DataArray containing the calculated trend values for each grid point,
            in units of [variable_units / frequency], where frequency is the specified
            time unit ('Y', 'M', or 'D').

        Raises
        ------
        ValueError
            If essential coordinates (time, lat, lon) are not found, or if the
            data selection results in insufficient data for trend calculation.

        Notes
        -----
        For large datasets, consider chunking your data before calling this method
        to improve memory efficiency and enable parallel processing.
        Example: data = data.chunk({'time': 120, 'lat': 50, 'lon': 50})
        """
        
        # Check for statsmodels availability
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels is required for trend analysis. "
                "Install it with: pip install statsmodels"
            )
        
        # Warn if data is not chunked
        self._warn_if_not_chunked("spatial trends")
        
        # Parameter validation
        if not isinstance(variable, str):
            raise TypeError("Variable must be a string")
        if frequency.upper() not in ['Y', 'M', 'D']:
            raise ValueError("frequency must be one of 'Y', 'M', 'D', 'y', 'm', or 'd'")
        if not isinstance(period, int) or period <= 0:
            raise ValueError("period must be a positive integer")
        
        # Use the original dataset
        dataset = self._obj
        
        # --- Step 1: Set up labels and coordinates ---
        frequency_upper = frequency.upper()
        if frequency_upper == 'Y':
            time_unit_label = "year"
        elif frequency_upper == 'M':
            time_unit_label = "month"
        elif frequency_upper == 'D':
            time_unit_label = "day"
        else:
            time_unit_label = "time_unit"

        time_coord_name = get_coord_name(dataset, ['time', 't'])
        lat_coord_name = get_coord_name(dataset, ['lat', 'latitude'])
        lon_coord_name = get_coord_name(dataset, ['lon', 'longitude'])
        
        if not all([time_coord_name, lat_coord_name, lon_coord_name]):
            raise ValueError("Dataset must contain time, latitude, and longitude for spatial trends.")
        if variable not in dataset.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset")

        # --- Step 2: Initialize Dask client ---
        get_or_create_dask_client()
        
        try:
            # --- Step 3: Select, filter, and prepare the data using the utility ---
            data_var = select_process_data(
                dataset, variable, latitude, longitude, level, time_range, season, year
            )
            
            if data_var[time_coord_name].size < 2 * period:
                raise ValueError(f"Insufficient time points ({data_var[time_coord_name].size}) after filtering. Need at least {2 * period}.")
            
            warnings.warn(f"Data selected for spatial trends: {data_var.sizes}", UserWarning)
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
                    
                    # Use the same frequency logic as calculate_trend
                    if frequency_upper == 'M':
                        scale_seconds = 24 * 3600 * (365.25 / 12)
                    elif frequency_upper == 'D':
                        scale_seconds = 24 * 3600
                    elif frequency_upper == 'Y':
                        scale_seconds = 24 * 3600 * 365.25
                    else:
                        scale_seconds = 24 * 3600 * 365.25  # Default to years
                    
                    x_numeric_for_regression = ((trend_clean.index - first_date).total_seconds() / scale_seconds).values
                    y_values_for_regression = trend_clean.values
                    
                    slope, _, _, _, _ = stats.linregress(x_numeric_for_regression, y_values_for_regression)

                    if np.isnan(slope): 
                        return np.nan

                    # d. Return slope (now correctly scaled by frequency)
                    return slope
                except (ValueError, TypeError, AttributeError):
                    return np.nan

            # --- Step 5: Chunk data and apply the trend function in parallel with Dask ---
            data_var = data_var.chunk({time_coord_name: -1, 
                                       lat_coord_name: 'auto',
                                       lon_coord_name: 'auto'})

            warnings.warn("Computing spatial trends in parallel with xarray.apply_ufunc...", UserWarning)
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
            ).rename(f"{variable}_trend_per_{time_unit_label}")

            # --- Step 6: Trigger computation and get the results ---
            with ProgressBar(dt=2.0):
                trend_computed_map = trend_da.compute()
            warnings.warn("Spatial trend computation complete.", UserWarning)

            # --- Step 7: Plot the resulting trend map if requested ---
            if plot_map:
                warnings.warn("Generating spatial trend map...", UserWarning)
                try:
                    start_time_str = pd.to_datetime(data_var[time_coord_name].min().item()).strftime('%Y-%m')
                    end_time_str = pd.to_datetime(data_var[time_coord_name].max().item()).strftime('%Y-%m')
                    time_period_title_str = f"{start_time_str} to {end_time_str}"
                except (ValueError, TypeError, AttributeError): time_period_title_str = "Selected Time Period"

                data_units = data_var.attrs.get('units', '')
                var_long_name = data_var.attrs.get('long_name', variable)
                cbar_label_str = f"Trend ({data_units} / {time_unit_label})" if data_units else f"Trend ({time_unit_label})"

                fig = plt.figure(figsize=figsize)
                proj = get_projection(projection)
                ax = fig.add_subplot(1, 1, 1, projection=proj)

                # Create the plot using contourf for filled contours
                contour_plot = trend_computed_map.plot.contourf(
                    ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                    levels=levels,
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
                if title is None:
                    season_title_str = season.upper() if season.lower() != 'annual' else 'Annual'
                    plot_title = (f"{season_title_str} {var_long_name.capitalize()} Trend ({time_unit_label})\n"
                                  f"{time_period_title_str}")
                    if level_selection_info_title: plot_title += f" at {level_selection_info_title}"
                else:
                    plot_title = title
                ax.set_title(plot_title, fontsize=14)

                plt.tight_layout(pad=1.5)
                if save_plot_path:
                    try:
                        plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
                        warnings.warn(f"Plot saved to {save_plot_path}", UserWarning)
                    except (OSError, IOError, ValueError) as e:
                        warnings.warn(f"Could not save plot to {save_plot_path}: {e}", UserWarning)
                # Note: plt.show() removed - user should call it explicitly if needed

            # --- Step 8: Return the computed trend data ---
            return trend_computed_map

        except (ValueError, RuntimeError, MemoryError) as e:
            warnings.warn(f"An error occurred during spatial trend processing: {e}", UserWarning)
            raise

__all__ = ['TrendsAccessor']
