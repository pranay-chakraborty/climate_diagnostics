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
    # INTERNAL HELPER METHODS
    # --------------------------------------------------------------------------
    def _get_coord_name(self, possible_names):
        """
        Find the actual coordinate name in the dataset from a list of common alternatives.

        This function checks for coordinate names in a case-sensitive manner first,
        then falls back to a case-insensitive check on the lower-cased names.

        Parameters
        ----------
        possible_names : list of str
            A list of potential coordinate names to search for (e.g., ['lat', 'latitude']).

        Returns
        -------
        str or None
            The matching coordinate name found in the dataset, or None if no match is found.
        """
        if self._obj is None: return None
        for name in possible_names:
            if name in self._obj.coords:
                return name
        coord_names_lower = {name.lower(): name for name in self._obj.coords}
        for name in possible_names:
            if name.lower() in coord_names_lower:
                return coord_names_lower[name.lower()]
        return None

    def _filter_by_season(self, data_array, season='annual', time_coord_name='time'):
        """
        Filter an xarray DataArray for a specific meteorological season.

        Supported seasons are 'annual', 'jjas', 'djf', 'mam', 'jja', 'son'.
        It identifies the month from the time coordinate and filters the data
        to include only the months corresponding to the specified season.

        Parameters
        ----------
        data_array : xr.DataArray
            The data to be filtered, which must have a time dimension.
        season : str, optional
            The season to filter by. Defaults to 'annual'.
        time_coord_name : str, optional
            The name of the time coordinate. Defaults to 'time'.

        Returns
        -------
        xr.DataArray
            The data filtered for the specified season.

        Raises
        ------
        ValueError
            If the time coordinate cannot be processed for month extraction.
        """
        # --- Step 1: Handle annual case and prepare for filtering ---
        season_input = season
        season = season.lower()
        if season == 'annual':
            return data_array

        if time_coord_name not in data_array.dims:
            print(f"Warning: Cannot filter by season - no '{time_coord_name}' dimension found.")
            return data_array

        # --- Step 2: Extract month information from the time coordinate ---
        time_da = data_array[time_coord_name]
        if 'month' in data_array.coords and time_coord_name in data_array['month'].coords:
             month_coord = data_array['month']
        elif hasattr(time_da.dt, 'month'):
            month_coord = time_da.dt.month
        elif time_da.size > 0 and hasattr(time_da.data[0], 'month'):
            try:
                # Fallback for cftime objects that don't directly support .dt accessor
                time_values = time_da.compute().data if hasattr(time_da, 'chunks') and time_da.chunks else time_da.data
                months_list = [t.month for t in time_values]
                month_coord = xr.DataArray(months_list, coords={time_coord_name: time_da}, dims=[time_coord_name])
                print("Warning: Extracted months from cftime objects manually for seasonal filter.")
            except Exception as e:
                 raise ValueError(f"Time coordinate '{time_coord_name}' (type: {time_da.dtype}) "
                                  f"could not be processed for month extraction for seasonal filter. Error: {e}")
        else:
            print(f"Warning: Cannot determine month for seasonal filtering from '{time_coord_name}' (dtype: {time_da.dtype}). Returning unfiltered data.")
            return data_array
        
        # --- Step 3: Define seasons and apply the filter ---
        season_months_map = {
            'jjas': [6, 7, 8, 9], 
            'djf': [12, 1, 2], 
            'mam': [3, 4, 5], 
            'jja': [6, 7, 8], 
            'son': [9, 10, 11]
        }
        
        selected_months = season_months_map.get(season)

        if selected_months:
            mask = month_coord.isin(selected_months)
            filtered_data = data_array.where(mask, drop=True)
            if time_coord_name in filtered_data.dims and filtered_data[time_coord_name].size == 0:
                warnings.warn(f"Filtering by season '{season_input.upper()}' resulted in no data points.", UserWarning)
            return filtered_data
        else:
            print(f"Warning: Unknown season '{season_input}'. Using annual data.")
            return data_array


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
        # --- Step 1: Initial validation and coordinate name fetching ---
        if self._obj is None: raise ValueError("Dataset not loaded.")
        if variable not in self._obj.variables: raise ValueError(f"Variable '{variable}' not found.")

        lat_coord_name = self._get_coord_name(['lat', 'latitude'])
        lon_coord_name = self._get_coord_name(['lon', 'longitude'])
        level_coord_name = self._get_coord_name(['level', 'lev', 'plev', 'zlev', 'height', 'altitude'])
        time_coord_name = self._get_coord_name(['time', 't'])
        
        if not time_coord_name:
            raise ValueError("Dataset must contain a recognizable time coordinate.")
        
        data_var = self._obj[variable]
        if time_coord_name not in data_var.dims:
            raise ValueError(f"Variable '{variable}' has no '{time_coord_name}' dimension.")

        # --- Step 2: Determine calculation type and apply seasonal filter ---
        is_global = latitude is None and longitude is None
        is_point_lat = isinstance(latitude, (int, float))
        is_point_lon = isinstance(longitude, (int, float))
        is_point = is_point_lat and is_point_lon and lat_coord_name and lon_coord_name
        
        calculation_type = 'global' if is_global else ('point' if is_point else 'region')

        if calculation_type == 'point':
            area_weighted = False

        print(f"Starting trend calculation: type='{calculation_type}', variable='{variable}', season='{season}', area_weighted={area_weighted}")

        data_var = self._filter_by_season(data_var, season=season, time_coord_name=time_coord_name)
        if data_var[time_coord_name].size == 0:
            raise ValueError(f"No data remains for variable '{variable}' after filtering for season '{season}'.")

        # --- Step 3: Validate selection ranges against data bounds ---
        if latitude is not None and lat_coord_name and lat_coord_name in data_var.coords:
            lat_min, lat_max = data_var[lat_coord_name].min().item(), data_var[lat_coord_name].max().item()
            if isinstance(latitude, slice):
                if latitude.start is not None and latitude.start > lat_max: raise ValueError(f"Lat min {latitude.start} > data max {lat_max}")
                if latitude.stop is not None and latitude.stop < lat_min: raise ValueError(f"Lat max {latitude.stop} < data min {lat_min}")
            elif isinstance(latitude, (list, np.ndarray)):
                if min(latitude) > lat_max or max(latitude) < lat_min: raise ValueError(f"Lat list out of bounds")
            else:
                if latitude < lat_min or latitude > lat_max: raise ValueError(f"Lat scalar {latitude} out of bounds [{lat_min}, {lat_max}]")
        
        if longitude is not None and lon_coord_name and lon_coord_name in data_var.coords:
            lon_min, lon_max = data_var[lon_coord_name].min().item(), data_var[lon_coord_name].max().item()
            if isinstance(longitude, slice):
                if longitude.start is not None and longitude.start > lon_max: raise ValueError(f"Lon min {longitude.start} > data max {lon_max}")
                if longitude.stop is not None and longitude.stop < lon_min: raise ValueError(f"Lon max {longitude.stop} < data min {lon_min}")
            elif isinstance(longitude, (list, np.ndarray)):
                if min(longitude) > lon_max or max(longitude) < lon_min: raise ValueError(f"Lon list out of bounds")
            else:
                if longitude < lon_min or longitude > lon_max: raise ValueError(f"Lon scalar {longitude} out of bounds [{lon_min}, {lon_max}]")
        
        level_dim_exists_in_var = level_coord_name and level_coord_name in data_var.dims
        if level is not None and level_dim_exists_in_var:
            level_min, level_max = data_var[level_coord_name].min().item(), data_var[level_coord_name].max().item()
            if isinstance(level, (slice, list, np.ndarray)):
                pass
            else:
                if level < level_min * 0.5 or level > level_max * 1.5:
                    print(f"Warning: Requested level {level} may be far from available range [{level_min}, {level_max}]")

        # --- Step 4: Prepare and apply selections for lat, lon, and level ---
        sel_slices = {}
        sel_points = {}
        level_selection_info = ""

        if latitude is not None and lat_coord_name:
            if isinstance(latitude, slice): sel_slices[lat_coord_name] = latitude
            else: sel_points[lat_coord_name] = latitude
        if longitude is not None and lon_coord_name:
            if isinstance(longitude, slice): sel_slices[lon_coord_name] = longitude
            else: sel_points[lon_coord_name] = longitude

        if level is not None:
            if level_dim_exists_in_var:
                level_selection_info = f"level(s)={level}"
                if isinstance(level, slice): sel_slices[level_coord_name] = level
                else: sel_points[level_coord_name] = level
            else:
                print(f"Warning: Level coordinate '{level_coord_name}' not found in var '{variable}'. Level selection ignored.")
        elif level_dim_exists_in_var and data_var.sizes[level_coord_name] > 1:
            # Default to first level if multiple exist and none are specified
            default_level_val = data_var[level_coord_name][0].item()
            sel_points[level_coord_name] = default_level_val
            level_selection_info = f"level={default_level_val} (defaulted)"
            print(f"Warning: Defaulting to first level: {default_level_val}")

        try:
            if sel_slices:
                print(f"Applying slice selection: {sel_slices}")
                data_var = data_var.sel(sel_slices)
            if sel_points:
                print(f"Applying point selection with method='nearest': {sel_points}")
                data_var = data_var.sel(sel_points, method='nearest')
        except Exception as e:
            raise ValueError(f"Error during .sel() operation: {e}")

        # --- Step 5: Perform spatial averaging if necessary ---
        selected_sizes = data_var.sizes
        print(f"Dimensions after selection: {selected_sizes}")
        if time_coord_name not in selected_sizes or selected_sizes[time_coord_name] == 0:
            raise ValueError(f"Selection resulted in zero time points ({selected_sizes}).")

        processed_ts_da = data_var
        dims_to_average_over = []
        if lat_coord_name and lat_coord_name in selected_sizes and selected_sizes[lat_coord_name] > 1:
            dims_to_average_over.append(lat_coord_name)
        if lon_coord_name and lon_coord_name in selected_sizes and selected_sizes[lon_coord_name] > 1:
            dims_to_average_over.append(lon_coord_name)
        
        # Also average over level if a slice was provided
        if level_coord_name and level_coord_name in selected_sizes and selected_sizes[level_coord_name] > 1 and \
           isinstance(level, slice):
            dims_to_average_over.append(level_coord_name)
            print(f"Including level dimension '{level_coord_name}' in averaging due to slice input.")

        region_coords = {d: data_var[d].values for d in data_var.coords if d != time_coord_name and d in data_var.coords}

        if dims_to_average_over:
            print(f"Averaging over dimensions: {dims_to_average_over}")
            if area_weighted and lat_coord_name and lat_coord_name in dims_to_average_over:
                # Weighted average
                if lat_coord_name not in data_var.coords:
                    raise ValueError(f"Latitude coordinate '{lat_coord_name}' needed for area weighting but not found.")
                weights = np.cos(np.deg2rad(data_var[lat_coord_name]))
                weights.name = "weights"
                with ProgressBar(dt=1.0):
                    processed_ts_da = data_var.weighted(weights).mean(dim=dims_to_average_over, skipna=True).compute()
            else:
                # Unweighted average
                if area_weighted and not (lat_coord_name and lat_coord_name in dims_to_average_over):
                    print("Warning: Area weighting requested but latitude dimension not available for averaging or not present.")
                with ProgressBar(dt=1.0):
                    processed_ts_da = data_var.mean(dim=dims_to_average_over, skipna=True).compute()
        else:
            # No averaging needed for point selections
            print("Point selection or no spatial dimensions to average. Computing...")
            with ProgressBar(dt=1.0):
                processed_ts_da = data_var.compute()
            if calculation_type != 'point' and not dims_to_average_over:
                 calculation_type = 'point'
            region_coords = {d: processed_ts_da[d].values for d in processed_ts_da.coords 
                            if d != time_coord_name and d in processed_ts_da.coords}

        # --- Step 6: Convert to pandas Series for STL ---
        if processed_ts_da is None: raise RuntimeError("Time series data not processed.")
        if time_coord_name not in processed_ts_da.dims:
            if not processed_ts_da.dims: raise ValueError("Processed data became a scalar value.")
            raise ValueError(f"Processed data lost time dimension. Final dims: {processed_ts_da.dims}")

        try:
            ts_pd = processed_ts_da.to_pandas()
        except Exception as e:
            # Handle cases where conversion to pandas is not straightforward
            if processed_ts_da.ndim > 1:
                squeezable_dims = [d for d in processed_ts_da.dims if d != time_coord_name and processed_ts_da.sizes[d] == 1]
                if squeezable_dims:
                    processed_ts_da_squeezed = processed_ts_da.squeeze(dim=squeezable_dims, drop=True)
                    if processed_ts_da_squeezed.ndim == 1:
                        ts_pd = processed_ts_da_squeezed.to_pandas()
                    else:
                        raise ValueError(f"Data for pandas conversion has >1 non-squeezable dimension after spatial processing: {processed_ts_da.dims}. Error: {e}")
                else:
                    raise ValueError(f"Data for pandas conversion has >1 dimension: {processed_ts_da.dims}. Error: {e}")
            elif processed_ts_da.ndim == 1:
                 try: # Fallback for cftime index
                    ts_pd = pd.Series(processed_ts_da.data, index=pd.to_datetime(processed_ts_da[time_coord_name].data))
                 except Exception as e_pd_series:
                     raise TypeError(f"Could not convert 1D DataArray to pandas Series. Error: {e_pd_series}")
            else:
                 raise TypeError(f"Data for pandas conversion is not a 1D time series. Dims: {processed_ts_da.dims}. Error: {e}")

        if not isinstance(ts_pd, pd.Series):
            if isinstance(ts_pd, pd.DataFrame):
                if len(ts_pd.columns) == 1:
                    ts_pd = ts_pd.iloc[:, 0]
                else:
                    warnings.warn(f"Pandas conversion resulted in DataFrame with {len(ts_pd.columns)} columns; using mean across columns.")
                    ts_pd = ts_pd.mean(axis=1)
            elif np.isscalar(ts_pd):
                raise TypeError(f"Expected pandas Series, but got a scalar ({ts_pd}). Data might be too aggregated.")
            else:
                raise TypeError(f"Expected pandas Series, but got {type(ts_pd)}")

        # --- Step 7: Apply STL decomposition to isolate the trend ---
        if ts_pd.isnull().all():
            raise ValueError(f"Time series is all NaNs after selection/averaging.")

        original_index = ts_pd.index
        ts_pd_clean = ts_pd.dropna()

        if ts_pd_clean.empty:
            raise ValueError(f"Time series is all NaNs after dropping NaN values.")
            
        min_stl_len = 2 * period
        if len(ts_pd_clean) < min_stl_len:
            raise ValueError(f"Time series length ({len(ts_pd_clean)}) for STL is less than required minimum ({min_stl_len}). Need at least 2*period.")

        print("Applying STL decomposition...")
        stl_result = STL(ts_pd_clean, period=period, robust=True).fit()
        trend_component = stl_result.trend.reindex(original_index)

        # --- Step 8: Perform linear regression on the trend component ---
        print("Performing linear regression...")
        trend_component_clean = trend_component.dropna()
        if trend_component_clean.empty:
            raise ValueError("Trend component is all NaNs after STL and dropna.")

        if pd.api.types.is_datetime64_any_dtype(trend_component_clean.index):
            # Calculate time in numeric units for regression
            first_date = trend_component_clean.index.min()
            if frequency.upper() == 'M':
                scale_seconds = 24 * 3600 * (365.25 / 12)
                time_unit_for_slope = "month"
            elif frequency.upper() == 'D':
                scale_seconds = 24 * 3600
                time_unit_for_slope = "day"
            elif frequency.upper() == 'Y':
                scale_seconds = 24 * 3600 * 365.25
                time_unit_for_slope = "year"
            else:
                scale_seconds = 24 * 3600 * 365.25
                time_unit_for_slope = "year"
                print(f"Warning: Unknown frequency '{frequency}', defaulting to years for slope calculation.")
            
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
         
        # --- Step 9: Generate plot if requested ---
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
            
            coord_strs = []
            def format_coord_for_title(coord_val_name, actual_coord_name, coords_dict_from_data):
                if actual_coord_name and actual_coord_name in coords_dict_from_data:
                    vals = np.atleast_1d(coords_dict_from_data[actual_coord_name])
                    if len(vals) == 0: return None
                    
                    name_map = {'lat': 'Lat', 'latitude': 'Lat', 'lon': 'Lon', 'longitude': 'Lon', 
                                'level': 'Level', 'lev': 'Level', 'plev': 'Level', 'height':'Height', 'altitude':'Alt'}
                    prefix = name_map.get(actual_coord_name, actual_coord_name.capitalize())
                    
                    original_request = None
                    if coord_val_name == 'latitude': original_request = latitude
                    elif coord_val_name == 'longitude': original_request = longitude
                    elif coord_val_name == 'level': original_request = level

                    if isinstance(original_request, (int,float,np.number)):
                         return f"{prefix}={original_request:.2f} (nearest: {vals.item():.2f})"
                    elif len(vals) > 1 and not np.all(np.isnan(vals)):
                        return f"{prefix}=[{np.nanmin(vals):.2f} to {np.nanmax(vals):.2f}]"
                    elif len(vals) == 1 and not np.isnan(vals.item()):
                        return f"{prefix}={vals.item():.2f}"
                return None

            lat_str = format_coord_for_title('latitude', lat_coord_name, region_coords)
            lon_str = format_coord_for_title('longitude', lon_coord_name, region_coords)
            level_str_from_data = format_coord_for_title('level', level_coord_name, region_coords)

            if calculation_type == 'point':
                title_parts.append("(Point Analysis)")
                if lat_str: coord_strs.append(lat_str)
                if lon_str: coord_strs.append(lon_str)
                if level_str_from_data: coord_strs.append(level_str_from_data)
            elif calculation_type == 'region' or calculation_type == 'global':
                avg_str_suffix = "Mean" if dims_to_average_over else "Selection"
                avg_str_prefix = "Weighted " if area_weighted and lat_coord_name in dims_to_average_over else ""
                title_parts.append(f"({'Global' if is_global else 'Regional'} {avg_str_prefix}{avg_str_suffix})")
                if lat_str: coord_strs.append(lat_str)
                if lon_str: coord_strs.append(lon_str)
                if level_str_from_data: coord_strs.append(level_str_from_data)
            
            if season.lower() != 'annual': coord_strs.append(f"Season={season.upper()}")
            
            full_title = title_parts[0]
            if len(title_parts) > 1: full_title += f" {title_parts[1]}"
            if coord_strs: full_title += "\n" + ", ".join(filter(None, coord_strs))
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
                plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
                print(f"Plot saved to: {save_plot_path}")
            plt.show()

        # --- Step 10: Return detailed results if requested ---
        if return_results:
            results = {
                'calculation_type': calculation_type,
                'trend_component': trend_component,
                'predicted_trend_line': predicted_trend_series,
                'area_weighted': area_weighted,
                'region_details': {'variable': variable, 'season': season, 
                                   'level_info': level_selection_info,
                                   'actual_level_from_data': region_coords.get(level_coord_name, "N/A")},
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
                           latitude=slice(None, None),
                           longitude=slice(None, None),
                           time_range=slice(None, None),
                           level=None,
                           season='annual',
                           num_years=1, 
                           n_workers=4,
                           robust_stl=True,
                           period=12,
                           plot_map=True,
                           land_only = False,
                           save_plot_path=None,
                           cmap = 'coolwarm'):
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
        
        # --- Step 1: Initial validation and coordinate name fetching ---
        if self._obj is None:
            raise ValueError("Dataset not loaded.")
        if variable not in self._obj.variables:
            raise ValueError(f"Variable '{variable}' not found.")

        if num_years == 1: period_str_label = "year"
        elif num_years == 10: period_str_label = "decade"
        else: period_str_label = f"{num_years} years"

        lat_coord_name = self._get_coord_name(['lat', 'latitude'])
        lon_coord_name = self._get_coord_name(['lon', 'longitude'])
        level_coord_name = self._get_coord_name(['level', 'lev', 'plev', 'zlev', 'height', 'altitude'])
        time_coord_name = self._get_coord_name(['time', 't'])
        
        if not all([time_coord_name, lat_coord_name, lon_coord_name]):
            raise ValueError("Dataset must contain recognizable time, latitude, and longitude coordinates for spatial trends.")

        # --- Step 2: Validate input selection ranges ---
        data_var_pre_dask = self._obj[variable]
        
        if lat_coord_name in data_var_pre_dask.coords and not isinstance(latitude, slice) or \
           (isinstance(latitude, slice) and (latitude.start is not None or latitude.stop is not None)):
            lat_min, lat_max = data_var_pre_dask[lat_coord_name].min().item(), data_var_pre_dask[lat_coord_name].max().item()
            if isinstance(latitude, slice):
                if latitude.start is not None and latitude.start > lat_max: raise ValueError(f"Lat min {latitude.start} > data max {lat_max}")
                if latitude.stop is not None and latitude.stop < lat_min: raise ValueError(f"Lat max {latitude.stop} < data min {lat_min}")
        
        if lon_coord_name in data_var_pre_dask.coords and not isinstance(longitude, slice) or \
           (isinstance(longitude, slice) and (longitude.start is not None or longitude.stop is not None)):
            lon_min, lon_max = data_var_pre_dask[lon_coord_name].min().item(), data_var_pre_dask[lon_coord_name].max().item()
            if isinstance(longitude, slice):
                if longitude.start is not None and longitude.start > lon_max: raise ValueError(f"Lon min {longitude.start} > data max {lon_max}")
                if longitude.stop is not None and longitude.stop < lon_min: raise ValueError(f"Lon max {longitude.stop} < data min {lon_min}")

        if time_range is not None and time_coord_name in data_var_pre_dask.dims:
            if not isinstance(time_range, slice): raise TypeError(f"time_range must be a slice, got {type(time_range)}")
            if time_range.start is not None or time_range.stop is not None:
                try:
                    time_min_val, time_max_val = np.datetime64(data_var_pre_dask[time_coord_name].min().values), np.datetime64(data_var_pre_dask[time_coord_name].max().values)
                    if time_range.start is not None and np.datetime64(time_range.start) > time_max_val: raise ValueError(f"Start time > data max")
                    if time_range.stop is not None and np.datetime64(time_range.stop) < time_min_val: raise ValueError(f"End time < data min")
                except Exception as e: print(f"Warning: Could not fully validate time range: {e}")
        
        level_dim_exists_in_var = level_coord_name and level_coord_name in data_var_pre_dask.dims
        if level is not None and level_dim_exists_in_var:
            level_min_val, level_max_val = data_var_pre_dask[level_coord_name].min().item(), data_var_pre_dask[level_coord_name].max().item()
            if isinstance(level, (int, float)):
                if level < level_min_val * 0.5 or level > level_max_val * 1.5: print(f"Warning: Level {level} far from range [{level_min_val}-{level_max_val}]")
            else: raise TypeError(f"Level for spatial trends should be scalar, got {type(level)}")

        # --- Step 3: Set up and start Dask client for parallel processing ---
        client = None
        cluster = None
        try:
            cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, silence_logs=logging.WARNING)
            client = Client(cluster)
            print(f"Dask client started for spatial trends: {client.dashboard_link}")
            
            # --- Step 4: Select, filter, and prepare the data ---
            data_var = self._obj[variable]
            
            if time_range is not None: data_var = data_var.sel({time_coord_name: time_range})
            
            level_selection_info_title = ""
            if level_dim_exists_in_var:
                if level is None and data_var.sizes.get(level_coord_name, 1) > 1:
                    level = data_var[level_coord_name].values[0]
                    print(f"Defaulting to first level for spatial trends: {level}")
                    level_selection_info_title = f"Level={level} (defaulted)"
                
                if level is not None:
                    data_var = data_var.sel({level_coord_name: level}, method='nearest')
                    actual_level_val = data_var[level_coord_name].item()
                    level_selection_info_title = f"Level={actual_level_val}"
                    print(f"Selected level for spatial trends: {actual_level_val}")
            
            data_var = self._filter_by_season(data_var, season=season, time_coord_name=time_coord_name)
            if data_var[time_coord_name].size < 2 * period:
                raise ValueError(f"Insufficient time points ({data_var[time_coord_name].size}) after filtering for season '{season}'. Need at least {2 * period}.")
            
            data_var = data_var.sel({lat_coord_name: latitude, lon_coord_name: longitude})
            
            print(f"Data selected for spatial trends: {data_var.sizes}")
            
            # --- Step 5: Define the function to calculate trend for a single grid cell ---
            def apply_stl_slope_spatial(da_1d_time_series, time_coord_array):
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

                try:
                    # b. Apply STL decomposition
                    stl_result = STL(ts_for_stl, period=period, robust=robust_stl,
                                     low_pass_jump=period//2,
                                     trend_jump=period//2,
                                     seasonal_jump=period//2
                                    ).fit(iter=2)
                    trend = stl_result.trend

                    if trend.isnull().all(): return np.nan

                    # c. Perform linear regression on the trend component
                    trend_clean = trend.dropna()
                    if len(trend_clean) < 2: return np.nan
                    
                    first_date = trend_clean.index.min()
                    scale_seconds_in_year = 24 * 3600 * 365.25
                    
                    x_numeric_for_regression = ((trend_clean.index - first_date).total_seconds() / scale_seconds_in_year).values
                    y_values_for_regression = trend_clean.values
                    
                    slope_per_year, _, _, _, _ = stats.linregress(x_numeric_for_regression, y_values_for_regression)

                    if np.isnan(slope_per_year): return np.nan

                    # d. Return slope scaled by the desired number of years
                    return slope_per_year * num_years
                except Exception:
                    return np.nan

            # --- Step 6: Chunk data and apply the trend function in parallel with Dask ---
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

            # --- Step 7: Trigger computation and get the results ---
            with ProgressBar(dt=2.0):
                trend_computed_map = trend_da.compute()
            print("Spatial trend computation complete.")

            # --- Step 8: Plot the resulting trend map if requested ---
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
                    plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
                    print(f"Plot saved to {save_plot_path}")
                plt.show()

            # --- Step 9: Return the computed trend data ---
            return trend_computed_map

        except Exception as e:
            print(f"An error occurred during spatial trend processing: {e}")
            raise
        finally:
            # --- Step 10: Ensure Dask client and cluster are closed ---
            if client: client.close()
            if cluster: cluster.close()
            print("Dask client and cluster for spatial trends closed.")

__all__ = ['TrendsAccessor']
