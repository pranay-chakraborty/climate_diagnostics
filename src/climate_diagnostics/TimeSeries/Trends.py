import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from dask.diagnostics import ProgressBar
from sklearn.linear_model import LinearRegression
import pandas as pd
from statsmodels.tsa.seasonal import STL
import warnings
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from dask.distributed import Client, LocalCluster 
import os

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
    
    Parameters
    ----------
    xarray_obj : xarray.Dataset
        The xarray Dataset this accessor is attached to.
        
    Attributes
    ----------
    _obj : xarray.Dataset
        Reference to the attached dataset with climate variables.
        
    Examples
    --------
    >>> import xarray as xr
    >>> from climate_diagnostics import register_accessors
    >>> ds = xr.open_dataset("climate_data.nc")
    >>> # Calculate global temperature trend
    >>> results = ds.climate_trends.calculate_trend(
    ...     variable="air",
    ...     level=850,  # 850 hPa level
    ...     season="annual",
    ...     area_weighted=True,
    ...     return_results=True
    ... )
    >>> # Calculate spatial trends for precipitation
    >>> trend_map = ds.climate_trends.calculate_spatial_trends(
    ...     variable="precip",
    ...     season="jjas",
    ...     num_years=10  # Trends per decade
    ... )
    
    Notes
    -----
    This class uses dask for efficient handling of large climate datasets and
    supports proper area-weighted averaging to account for the decreasing grid
    cell area toward the poles.
    """
    
    def __init__(self, xarray_obj):
        """
        Initialize the climate trends accessor with the provided xarray Dataset.
        
        Parameters
        ----------
        xarray_obj : xarray.Dataset
            The Dataset object this accessor is attached to. Should contain climate
            data with time, latitude, and longitude dimensions.
            
        Notes
        -----
        The accessor automatically detects common coordinate naming conventions
        (e.g., 'lat'/'latitude', 'lon'/'longitude', etc.) to ensure compatibility 
        with datasets from different sources.
        """
        self._obj = xarray_obj

    

    def _get_coord_name(self, possible_names):
        """
        Find the actual coordinate name in the dataset from a list of common alternatives.
        
        This helper method searches for coordinate names in the dataset using a list
        of possible naming conventions. For example, latitude might be named 'lat' or
        'latitude' depending on the dataset conventions.
        
        Parameters
        ----------
        possible_names : list of str
            List of possible coordinate names to search for (e.g., ['lat', 'latitude'])
            
        Returns
        -------
        str or None
            The first matching coordinate name found in the dataset, or None if no match
        """
        if self._obj is None: return None
        for name in possible_names:
            if name in self._obj.coords:
                return name
        return None

    def _filter_by_season(self, data_array, season='annual', time_coord_name='time'):
        """
        Filter an xarray DataArray by meteorological season.
        
        Creates a subset of the data containing only values from the specified
        meteorological season. If the season is 'annual', no filtering is applied.
        
        Parameters
        ----------
        data_array : xarray.DataArray
            Input data to be filtered
        season : str, default='annual'
            Season selection: 
            - 'annual': No filtering, returns all data
            - 'djf': December, January, February (Northern Hemisphere winter)
            - 'mam': March, April, May (Northern Hemisphere spring)
            - 'jja': June, July, August (Northern Hemisphere summer)
            - 'jjas': June, July, August, September (Northern Hemisphere extended summer)
            - 'son': September, October, November (Northern Hemisphere fall/autumn)
        time_coord_name : str, default='time'
            Name of the time coordinate in the data array
            
        Returns
        -------
        xarray.DataArray
            Filtered data for the selected season
            
        Raises
        ------
        UserWarning
            If filtering by the specified season results in no data points
        """
        if season.lower() == 'annual':
            return data_array

        if time_coord_name not in data_array.dims:
            print(f"Warning: Cannot filter by season - no '{time_coord_name}' dimension found.")
            return data_array

        if 'month' not in data_array.coords:
            if np.issubdtype(data_array[time_coord_name].dtype, np.datetime64):
                data_array = data_array.assign_coords(month=(data_array[time_coord_name].dt.month))
            else:
                print(f"Warning: Cannot create 'month' coordinate - '{time_coord_name}' is not datetime type.")
                return data_array

        month_coord = data_array['month']
        if season.lower() == 'jjas':
            mask = month_coord.isin([6, 7, 8, 9])
        elif season.lower() == 'djf':
            mask = month_coord.isin([12, 1, 2])
        elif season.lower() == 'mam':
            mask = month_coord.isin([3, 4, 5])
        elif season.lower() == 'jja':
            mask = month_coord.isin([6, 7, 8])
        elif season.lower() == 'son':
            mask = month_coord.isin([9, 10, 11])
        else:
            print(f"Warning: Unknown season '{season}'. Using annual data.")
            return data_array

        filtered_data = data_array.where(mask, drop=True)
        if time_coord_name in filtered_data.dims and len(filtered_data[time_coord_name]) == 0:
            warnings.warn(f"Filtering by season '{season}' resulted in no data points.", UserWarning)

        return filtered_data

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
                        save_plot_path = None
                        ):
        """
        Calculate trends from time series using STL decomposition and linear regression.
        
        This method extracts a variable from the dataset, applies spatial and temporal
        filtering, performs area-weighted averaging if requested, decomposes the time
        series using STL to extract the trend component, and fits a linear regression 
        to quantify the trend.
        
        Parameters
        ----------
        variable : str, default='air'
            Variable name to analyze in the dataset
        latitude : float, slice, or None
            Latitude selection as point value (float) or region (slice).
            If None, uses all latitudes.
        longitude : float, slice, or None
            Longitude selection as point value (float) or region (slice).
            If None, uses all longitudes.
        level : float, slice, or None
            Pressure level selection (if applicable).
            If None and level dimension exists, defaults to first level.
        frequency : str, default='M'
            Time frequency of data: 'M' (monthly), 'D' (daily), or 'Y' (yearly).
            Used for proper scaling of trend rates.
        season : str, default='annual'
            Season to analyze: 'annual', 'jjas' (Jun-Sep), 'djf' (Dec-Feb), 'mam' (Mar-May)
        area_weighted : bool, default=True
            Apply cosine latitude weighting for area-representative averaging.
            Only applicable for regional or global calculations.
        period : int, default=12
            Period for STL decomposition (12 for monthly data, 365 for daily data).
            Should match the seasonal cycle in the data.
        plot : bool, default=True
            Generate visualization of the trend analysis
        return_results : bool, default=False
            Return dictionary with calculation results
        save_plot_path : str, optional
            Path where the plot should be saved. If None (default), the plot is not saved.
            
        Returns
        -------
        dict or None
            If return_results=True, returns a dictionary containing:
            - calculation_type: 'global', 'point', or 'region'
            - trend_component: pandas.Series of the extracted trend
            - regression_model: fitted LinearRegression object
            - predicted_trend: pandas.Series of fitted trend values
            - area_weighted: whether area weighting was applied
            - region_details: metadata about variable, season, and level
            - stl_period: period used for STL decomposition
            - trend_statistics: pandas.DataFrame with regression statistics including:
            - slope, intercept, p_value, r_value, r_squared, standard_error
            
        Raises
        ------
        ValueError
            If dataset not loaded, variable not found, or selection/filtering issues
        TypeError
            For data type conversion issues or incompatible index types
        RuntimeError
            For processing failures
            
        Notes
        -----
        - For global calculations (latitude=None, longitude=None), area-weighted
          averaging is applied by default to account for decreasing grid cell area
          towards the poles.
        - Trend significance is assessed through the p_value in the trend_statistics.
        - The plot includes the extracted trend component and the linear fit.
        """
        if variable not in self._obj.variables: raise ValueError(f"Variable '{variable}' not found.")

        # Get coordinate names
        lat_coord = self._get_coord_name(['lat', 'latitude'])
        lon_coord = self._get_coord_name(['lon', 'longitude'])
        level_coord = self._get_coord_name(['level', 'lev', 'plev', 'zlev'])
        time_coord = self._get_coord_name(['time'])
        
        if not all([lat_coord, lon_coord, time_coord]):
            raise ValueError("Dataset must contain recognizable time, latitude, and longitude coordinates.")

        # Initial data selection
        data_var = self._obj[variable]
        if time_coord not in data_var.dims:
            raise ValueError(f"Variable '{variable}' has no '{time_coord}' dimension.")

        # Determine calculation type
        is_global = latitude is None and longitude is None
        is_point_lat = isinstance(latitude, (int, float))
        is_point_lon = isinstance(longitude, (int, float))
        is_point = is_point_lat and is_point_lon
        
        calculation_type = 'global' if is_global else ('point' if is_point else 'region')

        # Set default area weighting
        if area_weighted is None:
            area_weighted = calculation_type == 'global'
        if calculation_type == 'point':
            area_weighted = False

        print(f"Starting trend calculation: type='{calculation_type}', variable='{variable}', season='{season}', area_weighted={area_weighted}")

        # Filter by season
        data_var = self._filter_by_season(data_var, season=season, time_coord_name=time_coord)
        if variable not in self._obj.variables: 
            raise ValueError(f"Variable '{variable}' not found.")

        # Get coordinate names
        lat_coord = self._get_coord_name(['lat', 'latitude'])
        lon_coord = self._get_coord_name(['lon', 'longitude'])
        level_coord = self._get_coord_name(['level', 'lev', 'plev', 'zlev'])
        time_coord = self._get_coord_name(['time'])
        
        if not all([lat_coord, lon_coord, time_coord]):
            raise ValueError("Dataset must contain recognizable time, latitude, and longitude coordinates.")

        # Initial data selection
        data_var = self._obj[variable]
        if time_coord not in data_var.dims:
            raise ValueError(f"Variable '{variable}' has no '{time_coord}' dimension.")

        # Determine calculation type
        is_global = latitude is None and longitude is None
        is_point_lat = isinstance(latitude, (int, float))
        is_point_lon = isinstance(longitude, (int, float))
        is_point = is_point_lat and is_point_lon
        
        calculation_type = 'global' if is_global else ('point' if is_point else 'region')

        # Set default area weighting
        if area_weighted is None:
            area_weighted = calculation_type == 'global'
        if calculation_type == 'point':
            area_weighted = False

        print(f"Starting trend calculation: type='{calculation_type}', variable='{variable}', season='{season}', area_weighted={area_weighted}")

        # Filter by season
        data_var = self._filter_by_season(data_var, season=season, time_coord_name=time_coord)
        if len(data_var[time_coord]) == 0:
            raise ValueError(f"No data remains for variable '{variable}' after filtering for season '{season}'.")

        # Validate coordinates before selection
        
        # Validate latitude coordinates
        if latitude is not None and lat_coord in data_var.coords:
            lat_min, lat_max = data_var[lat_coord].min().item(), data_var[lat_coord].max().item()
            
            # Check if latitude selection is completely outside the available range
            if isinstance(latitude, slice):
                if latitude.start is not None and latitude.start > lat_max:
                    raise ValueError(f"Requested latitude minimum {latitude.start} is greater than available maximum {lat_max}")
                if latitude.stop is not None and latitude.stop < lat_min:
                    raise ValueError(f"Requested latitude maximum {latitude.stop} is less than available minimum {lat_min}")
            elif isinstance(latitude, (list, np.ndarray)):
                if min(latitude) > lat_max or max(latitude) < lat_min:
                    raise ValueError(f"Requested latitudes [{min(latitude)}, {max(latitude)}] are outside available range [{lat_min}, {lat_max}]")
            else:  # scalar
                if latitude < lat_min or latitude > lat_max:
                    raise ValueError(f"Requested latitude {latitude} is outside available range [{lat_min}, {lat_max}]")
        
        # Validate longitude coordinates
        if longitude is not None and lon_coord in data_var.coords:
            lon_min, lon_max = data_var[lon_coord].min().item(), data_var[lon_coord].max().item()
            
            # Check if longitude selection is completely outside the available range
            if isinstance(longitude, slice):
                if longitude.start is not None and longitude.start > lon_max:
                    raise ValueError(f"Requested longitude minimum {longitude.start} is greater than available maximum {lon_max}")
                if longitude.stop is not None and longitude.stop < lon_min:
                    raise ValueError(f"Requested longitude maximum {longitude.stop} is less than available minimum {lon_min}")
            elif isinstance(longitude, (list, np.ndarray)):
                if min(longitude) > lon_max or max(longitude) < lon_min:
                    raise ValueError(f"Requested longitudes [{min(longitude)}, {max(longitude)}] are outside available range [{lon_min}, {lon_max}]")
            else:  # scalar
                if longitude < lon_min or longitude > lon_max:
                    raise ValueError(f"Requested longitude {longitude} is outside available range [{lon_min}, {lon_max}]")
        
        # Validate level coordinates
        level_dim_exists = level_coord and level_coord in data_var.dims
        if level is not None and level_dim_exists:
            level_min, level_max = data_var[level_coord].min().item(), data_var[level_coord].max().item()
            
            if isinstance(level, (slice, list, np.ndarray)):
                # Check if completely outside range
                if isinstance(level, slice):
                    if level.start is not None and level.start > level_max:
                        raise ValueError(f"Requested level minimum {level.start} is greater than available maximum {level_max}")
                    if level.stop is not None and level.stop < level_min:
                        raise ValueError(f"Requested level maximum {level.stop} is less than available minimum {level_min}")
                elif min(level) > level_max or max(level) < level_min:
                    raise ValueError(f"Requested levels [{min(level)}, {max(level)}] are outside available range [{level_min}, {level_max}]")
            else:  # scalar
                if level < level_min * 0.5 or level > level_max * 1.5:
                    print(f"Warning: Requested level {level} is far from available range [{level_min}, {level_max}]")

        # Separate slice and point selectors for proper application
        sel_slices = {}
        sel_points = {}
        level_selection_info = ""
        level_dim_exists = level_coord and level_coord in data_var.dims

        # Set up latitude/longitude selectors
        if latitude is not None:
            if isinstance(latitude, slice): sel_slices[lat_coord] = latitude
            else: sel_points[lat_coord] = latitude
        if longitude is not None:
            if isinstance(longitude, slice): sel_slices[lon_coord] = longitude
            else: sel_points[lon_coord] = longitude

        # Handle level selection
        if level is not None:
            if level_dim_exists:
                level_selection_info = f"level(s)={level}"
                if isinstance(level, slice): sel_slices[level_coord] = level
                else: sel_points[level_coord] = level
            else:
                print(f"Warning: Level coordinate '{level_coord}' not found. Level selection ignored.")
        elif level_dim_exists and calculation_type != 'point':
            default_level_val = data_var[level_coord][0].item()
            sel_points[level_coord] = default_level_val
            level_selection_info = f"level={default_level_val} (defaulted)"
            print(f"Warning: Defaulting to first level: {default_level_val}")

        # Apply spatial selections in correct order
        try:
            if sel_slices:
                print(f"Applying slice selection: {sel_slices}")
                data_var = data_var.sel(sel_slices)

            if sel_points:
                print(f"Applying point selection with method='nearest': {sel_points}")
                data_var = data_var.sel(sel_points, method='nearest')
        except Exception as e:
            current_selectors = {**sel_slices, **sel_points}
            raise ValueError(f"Error during .sel() operation with selectors {current_selectors}: {e}")

        # Validate selection results
        selected_sizes = data_var.sizes
        print(f"Dimensions after selection: {selected_sizes}")
        spatial_dims_selected = [d for d in [lat_coord, lon_coord, level_coord] if d in selected_sizes]
        if any(selected_sizes[d] == 0 for d in spatial_dims_selected) or \
           time_coord not in selected_sizes or selected_sizes[time_coord] == 0:
            raise ValueError(f"Selection resulted in zero data points ({selected_sizes}). Check slice ranges.")

        # Spatial averaging if needed
        dims_to_average = [d for d in [lat_coord, lon_coord] if d in selected_sizes and selected_sizes[d] > 1]
        processed_ts_da = data_var
        region_coords = {d: data_var[d].values for d in data_var.coords if d != time_coord and d in data_var.coords}

        if dims_to_average:
            print(f"Averaging over dimensions: {dims_to_average}")
            if area_weighted:
                if lat_coord not in data_var.coords:
                    raise ValueError("Latitude coordinate needed for area weighting.")
                weights = np.cos(np.deg2rad(data_var[lat_coord]))
                weights.name = "weights"
                weights = weights.broadcast_like(data_var)
                with ProgressBar(dt=1.0):
                    processed_ts_da = data_var.weighted(weights).mean(dim=dims_to_average).compute()
            else:
                with ProgressBar(dt=1.0):
                    processed_ts_da = data_var.mean(dim=dims_to_average).compute()
        else:
            if calculation_type != 'point' and not dims_to_average:
                print("Selection resulted in a single spatial point. No averaging needed.")
                calculation_type = 'point'
            else:
                print("Point selection complete. No averaging needed.")
            
            with ProgressBar(dt=1.0):
                processed_ts_da = data_var.compute()
            region_coords = {d: processed_ts_da[d].values for d in processed_ts_da.coords 
                            if d != time_coord and d in processed_ts_da.coords}

        # Convert to pandas series
        if processed_ts_da is None:
            raise RuntimeError("Time series data not processed.")
        if time_coord not in processed_ts_da.dims:
            if not processed_ts_da.dims:
                raise ValueError("Processed data became a scalar value, cannot create time series.")
            raise ValueError(f"Processed data lost time dimension. Final dims: {processed_ts_da.dims}")

        ts_pd = processed_ts_da.to_pandas()

         # Handle different pandas output types
        if not isinstance(ts_pd, pd.Series):
            if isinstance(ts_pd, pd.DataFrame):
                if len(ts_pd.columns) == 1:
                    warnings.warn(f"Conversion resulted in DataFrame; extracting single column '{ts_pd.columns[0]}'.")
                    ts_pd = ts_pd.iloc[:, 0]
                else:
                    warnings.warn(f"Conversion resulted in DataFrame with multiple columns ({len(ts_pd.columns)}); using mean across columns.")
                    ts_pd = ts_pd.mean(axis=1)
            elif np.isscalar(ts_pd):
                raise TypeError(f"Expected pandas Series, but got a scalar ({ts_pd}).")
            else:
                raise TypeError(f"Expected pandas Series, but got {type(ts_pd)}")

        if ts_pd.isnull().all():
            raise ValueError(f"Time series is all NaNs after selection/averaging.")

        # Clean time series and check length for STL
        original_index = ts_pd.index
        ts_pd_clean = ts_pd.dropna()

        if ts_pd_clean.empty:
            raise ValueError(f"Time series is all NaNs after dropping NaN values.")
            
        min_stl_len = 2 * period
        if len(ts_pd_clean) < min_stl_len:
            raise ValueError(f"Time series length ({len(ts_pd_clean)}) is less than required minimum ({min_stl_len}).")

        # Apply STL decomposition
        print("Applying STL decomposition...")
        try:
            stl_result = STL(ts_pd_clean, period=period, robust=True).fit()
            trend_component = stl_result.trend.reindex(original_index)
        except Exception as e:
            print(f"Error during STL decomposition: {e}")
            raise

        # Linear regression on trend component
        print("Performing linear regression...")
        trend_component_clean = trend_component.dropna()
        if trend_component_clean.empty:
            raise ValueError("Trend component is all NaNs after STL.")

        try:
            # Convert index to numerical format for regression
            if pd.api.types.is_datetime64_any_dtype(trend_component_clean.index):
                first_date = trend_component_clean.index.min()
                
                
                # Set appropriate time unit based on frequency parameter
                if frequency == 'M':
                    scale = 24*3600*30  # seconds in ~month
                    time_unit = "months"
                    to_decade = 120      # months in a decade
                elif frequency == 'D':
                    scale = 24*3600      # seconds in day
                    time_unit = "days"
                    to_decade = 3652.5   # days in decade
                elif frequency == 'Y':
                    scale = 24*3600*365.25  # seconds in year
                    time_unit = "years"
                    to_decade = 10       # years in decade
                else:
                    # Default to years for climate data
                    scale = 24*3600*365.25
                    time_unit = "years"
                    to_decade = 10
                    print(f"Warning: Unknown frequency '{frequency}', defaulting to years")
                
                # Convert to the specified time units
                dates_numeric = ((trend_component_clean.index - first_date).total_seconds() / scale).values.reshape(-1, 1)
                
                x_vals = dates_numeric.flatten()
                y_vals = trend_component_clean.values
                slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x_vals, y_vals)
                
      
                
                
                
            elif pd.api.types.is_numeric_dtype(trend_component_clean.index):
                dates_numeric = trend_component_clean.index.values.reshape(-1, 1)
                time_unit = "units"
                to_decade = None
                x_vals = dates_numeric.flatten()
                y_vals = trend_component_clean.values
                slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x_vals, y_vals)
                
                
            else:
                raise TypeError(f"Trend index type ({trend_component_clean.index.dtype}) not recognized.")

            y_train = trend_component_clean.values
            reg = LinearRegression()
            reg.fit(dates_numeric, y_train)
            y_pred_values = reg.predict(dates_numeric)
            y_pred_series = pd.Series(y_pred_values, index=trend_component_clean.index).reindex(original_index)
            
        except Exception as e:
            print(f"Error during linear regression: {e}")
            raise
        
        
        
        trend_stats_df = pd.DataFrame({
        'statistic': ['slope', 'intercept', 'p_value', 'r_value', 'r_squared','standard_error'],
        'value': [slope, intercept, p_value, r_value, r_value**2,slope_std_error]
         })
         
        # Plotting
        if plot:
            print("Generating plot...")
            plt.figure(figsize=(16, 10), dpi=100)
            plt.scatter(trend_component.index, trend_component.values, color='blue', alpha=0.5, s=10, 
                       label='STL Trend Component')
            plt.plot(y_pred_series.index, y_pred_series.values, color='red', linewidth=2, 
                    label='Linear Trend Fit')

            # Dynamic title generation
            title = f"Trend: {variable.capitalize()}"
            units = processed_ts_da.attrs.get('units', '')
            ylabel = f'{variable.capitalize()} Trend' + (f' ({units})' if units else '')

            coord_strs = []
            
            def format_coord(coord_name, coords_dict):
                if coord_name in coords_dict:
                    vals = np.atleast_1d(coords_dict[coord_name])
                    if len(vals) == 0: return None
                    name_map = {'lat': 'Lat', 'latitude': 'Lat', 'lon': 'Lon', 'longitude': 'Lon', 
                               'level': 'Level', 'lev': 'Level', 'plev': 'Level'}
                    prefix = name_map.get(coord_name, coord_name.capitalize())
                    if len(vals) > 1:
                        return f"{prefix}=[{np.nanmin(vals):.2f}:{np.nanmax(vals):.2f}]"
                    else:
                        scalar_val = vals.item() if vals.ndim == 0 or vals.size == 1 else vals[0]
                        return f"{prefix}={scalar_val:.2f}"
                return None

            lat_str = format_coord(lat_coord, region_coords)
            lon_str = format_coord(lon_coord, region_coords)
            level_str = format_coord(level_coord, region_coords)

            # Title components based on calculation type
            if calculation_type == 'point':
                title += f" (Point Analysis)"
                if lat_str: coord_strs.append(lat_str)
                if lon_str: coord_strs.append(lon_str)
                if level_str: coord_strs.append(level_str)
            elif calculation_type == 'region' or calculation_type == 'global':
                avg_str = f"{'Weighted' if area_weighted else 'Unweighted'} Mean" if dims_to_average else "Selection"
                title += f" ({'Global' if is_global else 'Regional'} {avg_str})"
                if lat_str: coord_strs.append(lat_str)
                if lon_str: coord_strs.append(lon_str)
                if level_str: coord_strs.append(level_str)
            else:
                title += " (Unknown Type)"

            if season.lower() != 'annual': coord_strs.append(f"Season={season.upper()}")
            title += "\n" + ", ".join(filter(None, coord_strs))

            plt.title(title + "\n(STL Trend + Linear Regression)", fontsize=14)
            plt.xlabel('Time', fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(10))
            plt.tight_layout()
            if save_plot_path is not None:
                plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
                print(f"Plot saved to: {save_plot_path}")
            plt.show()

        # Return results
        if return_results:
            results = {
                'calculation_type': calculation_type,
                'trend_component': trend_component,
                'regression_model': reg,
                'predicted_trend': y_pred_series,
                'area_weighted': area_weighted,
                'region_details': {'variable': variable, 'season': season, 'level_info': level_selection_info},
                'stl_period': period,
                'trend_statistics' : trend_stats_df
            }
            return results
        else:
            return None
        
    def calculate_spatial_trends(self,
                           variable='air',
                           latitude=slice(None, None),
                           longitude=slice(None, None),
                           time_range=slice(None, None),
                           level=None,
                           season='annual',
                           frequency='M',  # Frequency parameter
                           num_years=1,    # New parameter for time period
                           n_workers=4,
                           robust_stl=True,
                           period=12,
                           plot_map=True,
                           land_only = False,
                           save_plot_path=None,
                           cmap = 'coolwarm'):
        """
        Calculate and visualize spatial trends using STL decomposition for each grid point.

        This method computes trends at each grid point within the specified region using 
        Seasonal-Trend decomposition by LOESS (STL) and parallelizes the computation with Dask.
        The result is a spatial map of trends (e.g., °C per year, decade, etc.).

        Parameters
        ----------
        variable : str, default='air'
            Variable name to analyze trends for
        latitude : slice, optional
            Latitude slice for region selection. Default is all latitudes.
        longitude : slice, optional
            Longitude slice for region selection. Default is all longitudes.
        time_range : slice, optional
            Time slice for analysis period. Default is all times.
        level : float, int, or None, optional
            Vertical level selection. If None and multiple levels exist, defaults to first level.
        season : str, default='annual'
            Season to analyze: 'annual', 'jjas' (Jun-Sep), 'djf' (Dec-Feb), 'mam' (Mar-May), etc.
        frequency : str, default='M'
            Time frequency of data: 'M' (monthly), 'D' (daily), or 'Y' (yearly).
            Used for proper scaling of trend rates.
        num_years : int, default=1
            Number of years over which to express the trend (e.g., 1=per year, 10=per decade).
        n_workers : int, default=4
            Number of Dask workers for parallel computation
        robust_stl : bool, default=True
            Use robust STL fitting to reduce impact of outliers
        period : int, default=12
            Period for STL decomposition (12 for monthly data, 365 for daily data)
        plot_map : bool, default=True
            Whether to generate and show the trend map plot
        land_only : bool, default=False
            If True, mask out ocean areas to show land-only data
        save_plot_path : str, optional
            Path to save the plot image. If None, plot is shown but not saved.
        cmap : str, optional
            cmap for the contour plots, defaults to 'coolwarm'

        Returns
        -------
        xarray.DataArray
            DataArray of computed trends (in units per specified time period).
            Return value includes lat/lon coordinates and appropriate metadata.

        Raises
        ------
        ValueError
            If dataset is not loaded, variable not found, insufficient data points, 
            or other validation errors.
            
        Notes
        -----
        - This method is computationally intensive but efficiently parallelized using Dask.
        - For each grid point, STL decomposition extracts the trend component before 
          computing the linear trend slope.
        - Trends are automatically scaled according to the frequency parameter and num_years
          value for consistency (e.g., calculating decadal trends from monthly data).
        - The resulting visualization includes statistical significance masking, appropriate
          map projections, and comprehensive metadata.
        """
        
        if self._obj is None:
            raise ValueError("Dataset not loaded.")
        if variable not in self._obj.variables:
            raise ValueError(f"Variable '{variable}' not found.")

        # Format the time period string for labels
        if num_years == 1:
            period_str = "year"
        elif num_years == 10:
            period_str = "decade"
        else:
            period_str = f"{num_years} years"

        # Get coordinate names
        lat_coord = self._get_coord_name(['lat', 'latitude'])
        lon_coord = self._get_coord_name(['lon', 'longitude'])
        level_coord = self._get_coord_name(['level', 'lev', 'plev', 'zlev'])
        time_coord = self._get_coord_name(['time'])
        
        if not all([lat_coord, lon_coord, time_coord]):
            raise ValueError("Dataset must contain recognizable time, latitude, and longitude coordinates.")

        # Validate coordinates before Dask setup
        data_var = self._obj[variable]
        
        # Validate latitude coordinates
        if lat_coord in data_var.coords and not isinstance(latitude, slice) or (
                isinstance(latitude, slice) and (latitude.start is not None or latitude.stop is not None)):
            lat_min, lat_max = data_var[lat_coord].min().item(), data_var[lat_coord].max().item()
            
            # Check if latitude selection is completely outside the available range
            if isinstance(latitude, slice):
                if latitude.start is not None and latitude.start > lat_max:
                    raise ValueError(f"Requested latitude minimum {latitude.start} is greater than available maximum {lat_max}")
                if latitude.stop is not None and latitude.stop < lat_min:
                    raise ValueError(f"Requested latitude maximum {latitude.stop} is less than available minimum {lat_min}")
            elif isinstance(latitude, (list, np.ndarray)):
                if min(latitude) > lat_max or max(latitude) < lat_min:
                    raise ValueError(f"Requested latitudes [{min(latitude)}, {max(latitude)}] are outside available range [{lat_min}, {lat_max}]")
            else:  # scalar
                if latitude < lat_min or latitude > lat_max:
                    raise ValueError(f"Requested latitude {latitude} is outside available range [{lat_min}, {lat_max}]")
        
        # Validate longitude coordinates
        if lon_coord in data_var.coords and not isinstance(longitude, slice) or (
                isinstance(longitude, slice) and (longitude.start is not None or longitude.stop is not None)):
            lon_min, lon_max = data_var[lon_coord].min().item(), data_var[lon_coord].max().item()
            
            # Check if longitude selection is completely outside the available range
            if isinstance(longitude, slice):
                if longitude.start is not None and longitude.start > lon_max:
                    raise ValueError(f"Requested longitude minimum {longitude.start} is greater than available maximum {lon_max}")
                if longitude.stop is not None and longitude.stop < lon_min:
                    raise ValueError(f"Requested longitude maximum {longitude.stop} is less than available minimum {lon_min}")
            elif isinstance(longitude, (list, np.ndarray)):
                if min(longitude) > lon_max or max(longitude) < lon_min:
                    raise ValueError(f"Requested longitudes [{min(longitude)}, {max(longitude)}] are outside available range [{lon_min}, {lon_max}]")
            else:  # scalar
                if longitude < lon_min or longitude > lon_max:
                    raise ValueError(f"Requested longitude {longitude} is outside available range [{lon_min}, {lon_max}]")
        
        # Validate time range
        if time_range is not None and time_coord in data_var.dims:
            if time_range != slice(None, None) and not isinstance(time_range, slice):
                raise TypeError(f"time_range must be a slice, got {type(time_range)}")
                
            if isinstance(time_range, slice) and (time_range.start is not None or time_range.stop is not None):
                try:
                    time_min, time_max = data_var[time_coord].min().values, data_var[time_coord].max().values
                    
                    # Check if time selection is completely outside the available range
                    if time_range.start is not None:
                        if np.datetime64(time_range.start) > time_max:
                            raise ValueError(f"Requested start time {time_range.start} is after available maximum {time_max}")
                    if time_range.stop is not None:
                        if np.datetime64(time_range.stop) < time_min:
                            raise ValueError(f"Requested end time {time_range.stop} is before available minimum {time_min}")
                except (TypeError, ValueError) as e:
                    # If we can't compare the times, proceed and let xarray handle it
                    print(f"Warning: Could not validate time range: {e}")
        
        # Validate level if provided
        if level is not None and level_coord in data_var.dims:
            level_min, level_max = data_var[level_coord].min().item(), data_var[level_coord].max().item()
            
            if isinstance(level, (int, float)):
                if level < level_min * 0.5 or level > level_max * 1.5:
                    print(f"Warning: Requested level {level} is far from available range [{level_min}, {level_max}]")
                elif level < level_min or level > level_max:
                    raise ValueError(f"Requested level {level} is outside available range [{level_min}, {level_max}]")

        # --- Dask Client Setup ---
        try:
            cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
            client = Client(cluster)
            print(f"Dask client started: {client.dashboard_link}")
            
            # --- Data Selection and Seasonal Filtering ---
            data_var = self._obj[variable]
            
            # Apply time range selection if provided
            if time_range is not None:
                data_var = data_var.sel(time=time_range)
            
            # Handle level selection
            level_selection_info = ""
            if level_coord in data_var.dims:
                if level is None:
                    # Default to first level
                    level = data_var[level_coord].values[0]
                    print(f"Defaulting to first level: {level}")
                    level_selection_info = f"Level={level} (defaulted)"
                
                data_var = data_var.sel({level_coord: level}, method='nearest')
                level_selection_info = f"Level={data_var[level_coord].values.item()}"
                print(f"Selected level: {data_var[level_coord].values.item()}")
            
            # Apply seasonal filtering
            data_var = self._filter_by_season(data_var, season=season, time_coord_name=time_coord)
            if len(data_var[time_coord]) < 2 * period:
                raise ValueError(f"Insufficient time points ({len(data_var[time_coord])}) after filtering for season '{season}'. Need at least {2 * period}.")
            
            # Apply spatial selection
            data_var = data_var.sel({lat_coord: latitude, lon_coord: longitude})
            
            print(f"Data selected: {data_var.sizes}")
            
            # --- Trend Calculation Function ---
            def apply_stl_slope(da):
                if hasattr(da, 'values'):
                    values = da.values
                else:
                    values = da # Assume numpy array if not xarray

                values = np.asarray(values).squeeze()

                # Check for sufficient data and NaNs before STL
                min_periods = 2 * period
                if values.ndim == 0 or len(values) < min_periods or np.isnan(values).all():
                    return np.nan
                if np.isnan(values).any():
                    if np.sum(~np.isnan(values)) < min_periods:
                        return np.nan

                try:
                    # Apply STL
                    stl_result = STL(values, period=period, robust=robust_stl).fit()
                    trend = stl_result.trend

                    # Check trend for NaNs
                    if np.isnan(trend).all():
                        return np.nan

                    # Fit linear trend only to non-NaN trend values
                    valid_indices = ~np.isnan(trend)
                    if np.sum(valid_indices) < 2:
                        return np.nan

                    x = np.arange(len(trend))
                    slope, _ = np.polyfit(x[valid_indices], trend[valid_indices], 1)

                    # Use the provided frequency parameter instead of detection
                    if frequency.upper() == 'M':
                        # Monthly data - multiply by 12 (months per year)
                        slope_per_year = slope * 12
                    elif frequency.upper() == 'D':
                        # Daily data - multiply by 365.25 (days per year)
                        slope_per_year = slope * 365.25
                    elif frequency.upper() == 'Y' or frequency.upper() == 'A':
                        # Yearly data - keep as is
                        slope_per_year = slope
                    else:
                        # Default for other frequencies (assume monthly)
                        slope_per_year = slope * 12
                        print(f"Using default conversion for frequency '{frequency}'. Assuming monthly data.")
                    
                    # Multiply by num_years to get trend over specified period
                    slope_per_period = slope_per_year * num_years
                    
                    return slope_per_period
                except Exception:
                    return np.nan

            # --- Parallel Computation using Dask ---
            # Ensure the time dimension is a single chunk for apply_ufunc
            data_var = data_var.chunk({time_coord: -1, lat_coord: 'auto', lon_coord: 'auto'})

            print("Computing trends in parallel with xarray...")
            trend_result = xr.apply_ufunc(
                apply_stl_slope,
                data_var,
                input_core_dims=[[time_coord]],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'allow_rechunk': True}
            ).rename(f"{variable}_trend_per_{period_str}")

            # Compute the result with a progress bar
            with ProgressBar():
                trend_computed = trend_result.compute()
            print("Trend computation complete.")

            # --- Plotting ---
            if plot_map:
                print("Generating plot...")
                # Get time range for the title
                try:
                    start_time = data_var[time_coord].min().dt.strftime('%Y-%m').item()
                    end_time = data_var[time_coord].max().dt.strftime('%Y-%m').item()
                    time_period_str = f"{start_time} to {end_time}"
                except Exception:
                    time_period_str = "Selected Time Period"

                units = data_var.attrs.get('units', '')
                var_name = data_var.attrs.get('long_name', variable)
                cbar_label = f"{var_name} trend per {period_str} ({units})" if units else f"{var_name} trend per {period_str}"

                try:
                    # --- Cartopy Plot ---
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

                    contour = trend_computed.plot.contourf(
                        ax=ax,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        robust=True,
                        levels=40,
                        extend='both',
                        cbar_kwargs={'label': cbar_label}
                    )
                    contour.colorbar.set_label(cbar_label, size=14)
                    contour.colorbar.ax.tick_params(labelsize=14)

                    # Add geographic features
                    if land_only:
                        # Add OCEAN feature to mask out oceans when land_only=True
                        ax.add_feature(cfeature.OCEAN, zorder=2, facecolor='white')
                        ax.coastlines(zorder=3)  # Ensure coastlines are drawn on top
                        ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)  # Borders on top too
                    else:
                        # Standard display without ocean masking
                        ax.coastlines()
                        ax.add_feature(cfeature.BORDERS, linestyle=':')
                    
                    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlabel_style = {'size': 14}
                    gl.ylabel_style = {'size': 14}
                    
                    # Create title
                    season_str = season.upper() if season.lower() != 'annual' else 'Annual'
                    title = f"{season_str} {var_name.capitalize()} Trends ({units} per {period_str})\n{time_period_str}"
                    if level_selection_info:
                        title += f"\n{level_selection_info}"
                    ax.set_title(title, fontsize=14)

                    plt.tight_layout()

                    if save_plot_path:
                        plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
                        print(f"Plot saved to {save_plot_path}")
                    plt.show()

                except Exception as plot_err:
                    print(f"An error occurred during plotting: {plot_err}")

            return trend_computed

        except Exception as e:
            print(f"An error occurred during processing: {e}")
            return None
        finally:
            # --- Dask Client Shutdown ---
            if 'client' in locals() and client is not None:
                print("Shutting down Dask client...")
                client.close()
                if 'cluster' in locals() and cluster is not None:
                    cluster.close()
                print("Dask client closed.")
__all__ = ['TrendsAccessor']