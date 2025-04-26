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
    
    def _select_process_data(self, variable, latitude=None, longitude=None, level=None, time_range=None, season='annual'):
        """
        Select and process data based on specified parameters.
        
        Handles spatial, vertical and temporal subsetting of data.
        """
        # Initial data selection
        if variable not in self._obj.variables:
            raise ValueError(f"Variable '{variable}' not found.")
            
        # Get coordinate names
        lat_coord = self._get_coord_name(['lat', 'latitude'])
        lon_coord = self._get_coord_name(['lon', 'longitude'])
        level_coord = self._get_coord_name(['level', 'lev', 'plev', 'zlev'])
        time_coord = self._get_coord_name(['time'])
        
        if not all([lat_coord, lon_coord, time_coord]):
            raise ValueError("Dataset must contain recognizable time, latitude, and longitude coordinates.")
            
        data_var = self._obj[variable]
        if time_coord not in data_var.dims:
            raise ValueError(f"Variable '{variable}' has no '{time_coord}' dimension.")
        
        # Filter by season
        data_var = self._filter_by_season(data_var, season=season, time_coord_name=time_coord)
        if len(data_var[time_coord]) == 0:
            raise ValueError(f"No data remains for variable '{variable}' after filtering for season '{season}'.")
            
        selection_dict = {}
        needs_nearest = False
        
        # Validate latitude coordinates
        if latitude is not None and lat_coord in data_var.coords:
            lat_min, lat_max = data_var[lat_coord].min().item(), data_var[lat_coord].max().item()
            
            if isinstance(latitude, slice):
                if latitude.start is not None and latitude.start > lat_max:
                    raise ValueError(f"Requested latitude minimum {latitude.start} is greater than available maximum {lat_max}")
                if latitude.stop is not None and latitude.stop < lat_min:
                    raise ValueError(f"Requested latitude maximum {latitude.stop} is less than available minimum {lat_min}")
            elif isinstance(latitude, (list, np.ndarray)):
                if min(latitude) > lat_max or max(latitude) < lat_min:
                    raise ValueError(f"Requested latitudes [{min(latitude)}, {max(latitude)}] are outside available range [{lat_min}, {lat_max}]")
            else:  
                if latitude < lat_min or latitude > lat_max:
                    raise ValueError(f"Requested latitude {latitude} is outside available range [{lat_min}, {lat_max}]")
            
            selection_dict[lat_coord] = latitude
            if isinstance(latitude, (int, float)):
                needs_nearest = True
                
        # Validate longitude coordinates
        if longitude is not None and lon_coord in data_var.coords:
            lon_min, lon_max = data_var[lon_coord].min().item(), data_var[lon_coord].max().item()
            
            if isinstance(longitude, slice):
                if longitude.start is not None and longitude.start > lon_max:
                    raise ValueError(f"Requested longitude minimum {longitude.start} is greater than available maximum {lon_max}")
                if longitude.stop is not None and longitude.stop < lon_min:
                    raise ValueError(f"Requested longitude maximum {longitude.stop} is less than available minimum {lon_min}")
            elif isinstance(longitude, (list, np.ndarray)):
                if min(longitude) > lon_max or max(longitude) < lon_min:
                    raise ValueError(f"Requested longitudes [{min(longitude)}, {max(longitude)}] are outside available range [{lon_min}, {lon_max}]")
            else:  
                if longitude < lon_min or longitude > lon_max:
                    raise ValueError(f"Requested longitude {longitude} is outside available range [{lon_min}, {lon_max}]")
            
            selection_dict[lon_coord] = longitude
            if isinstance(longitude, (int, float)):
                needs_nearest = True
                
        # Validate time range
        if time_range is not None and time_coord in data_var.dims:
            try:
                time_min, time_max = data_var[time_coord].min().values, data_var[time_coord].max().values
                
                if isinstance(time_range, slice):
                    if time_range.start is not None:
                        if np.datetime64(time_range.start) > time_max:
                            raise ValueError(f"Requested start time {time_range.start} is after available maximum {time_max}")
                    if time_range.stop is not None:
                        if np.datetime64(time_range.stop) < time_min:
                            raise ValueError(f"Requested end time {time_range.stop} is before available minimum {time_min}")
                elif isinstance(time_range, (list, np.ndarray)):
                    t_min, t_max = np.min(time_range), np.max(time_range)
                    if np.datetime64(t_min) > time_max or np.datetime64(t_max) < time_min:
                        raise ValueError(f"Requested time range is outside available range [{time_min}, {time_max}]")
            except (TypeError, ValueError) as e:
                print(f"Warning: Could not validate time range: {e}")
            
            selection_dict[time_coord] = time_range
        
        # Handle level selection
        if level_coord and level_coord in data_var.dims:
            if level is not None:
                level_min, level_max = data_var[level_coord].min().item(), data_var[level_coord].max().item()
                
                if isinstance(level, (slice, list, np.ndarray)):
                    if isinstance(level, slice):
                        if level.start is not None and level.start > level_max:
                            raise ValueError(f"Requested level minimum {level.start} is greater than available maximum {level_max}")
                        if level.stop is not None and level.stop < level_min:
                            raise ValueError(f"Requested level maximum {level.stop} is less than available minimum {level_min}")
                    elif min(level) > level_max or max(level) < level_min:
                        raise ValueError(f"Requested levels [{min(level)}, {max(level)}] are outside available range [{level_min}, {level_max}]")
                    
                    selection_dict[level_coord] = level
                elif isinstance(level, (int, float)):
                    if level < level_min or level > level_max:
                        if level < level_min * 0.5 or level > level_max * 1.5:
                            raise ValueError(f"Requested level {level} is far outside available range [{level_min}, {level_max}]")
                        else:
                            print(f"Warning: Requested level {level} is slightly outside available range [{level_min}, {level_max}]")
                    
                    selection_dict[level_coord] = level
                    needs_nearest = True
            elif len(data_var[level_coord]) > 1:
                level_val = data_var[level_coord].values[0]
                selection_dict[level_coord] = level_val
                print(f"Warning: Multiple levels found. Using first level: {level_val}")
        elif level is not None:
            print(f"Warning: Level dimension not found. Ignoring 'level' parameter.")
        
        if selection_dict:
            has_slice = any(isinstance(v, slice) for v in selection_dict.values())
            method_to_use = 'nearest' if needs_nearest and not has_slice else None

            if method_to_use is None and needs_nearest and has_slice:
                print("Warning: Using slice selection and nearest neighbor selection together is not supported. Using slice selection only.")

            print(f"Applying final selection: {selection_dict} with method='{method_to_use}'")
            try:
                data_var = data_var.sel(**selection_dict, method=method_to_use)
            except Exception as e:
                print(f"Error during final .sel() operation: {e}")
                print(f"Selection dictionary: {selection_dict}")
                print(f"Method used: {method_to_use}")
                raise 

        if data_var.size == 0:
            print("Warning: Selection resulted in an empty DataArray.")

        return data_var
    
#======================HELPER METHODS===========================#
    def _apply_spatial_averaging(self, data_var, area_weighted=True):
        """
        Apply spatial averaging with optional area-weighting.
        
        Parameters
        ----------
        data_var : xarray.DataArray
            Data to average
        area_weighted : bool, default=True
            Whether to apply area weighting
            
        Returns
        -------
        xarray.DataArray
            Spatially averaged data
        tuple
            (calculation_type, dims_to_average, region_coords)
        """
        # Get coordinate names
        lat_coord = self._get_coord_name(['lat', 'latitude'])
        lon_coord = self._get_coord_name(['lon', 'longitude'])
        time_coord = self._get_coord_name(['time'])
        
        # Determine calculation type based on dimensions
        selected_sizes = data_var.sizes
        is_point = all(selected_sizes.get(d, 0) <= 1 for d in [lat_coord, lon_coord])
        is_global = all(d in selected_sizes for d in [lat_coord, lon_coord])
        calculation_type = 'point' if is_point else ('global' if is_global else 'region')
        
        # Identify dimensions to average
        dims_to_average = [d for d in [lat_coord, lon_coord] if d in selected_sizes and selected_sizes[d] > 1]
        region_coords = {d: data_var[d].values for d in data_var.coords if d != time_coord and d in data_var.coords}
        
        # Skip averaging for point calculations
        if not dims_to_average:
            print("No spatial averaging needed.")
            with ProgressBar(dt=1.0):
                return data_var.compute(), calculation_type, dims_to_average, region_coords
        
        print(f"Averaging over dimensions: {dims_to_average}")
        if area_weighted and lat_coord in dims_to_average:
            if lat_coord not in data_var.coords:
                raise ValueError("Latitude coordinate needed for area weighting.")
            weights = np.cos(np.deg2rad(data_var[lat_coord]))
            weights.name = "weights"
            weights = weights.broadcast_like(data_var)
            with ProgressBar(dt=1.0):
                return data_var.weighted(weights).mean(dim=dims_to_average).compute(), calculation_type, dims_to_average, region_coords
        else:
            with ProgressBar(dt=1.0):
                return data_var.mean(dim=dims_to_average).compute(), calculation_type, dims_to_average, region_coords

    def _convert_to_pandas_series(self, data_array):
        """
        Convert xarray DataArray to pandas Series with validation.
        
        Parameters
        ----------
        data_array : xarray.DataArray
            Data to convert
            
        Returns
        -------
        pandas.Series
            The converted time series
        """
        time_coord = self._get_coord_name(['time'])
        
        if data_array is None:
            raise RuntimeError("Time series data not processed.")
        if time_coord not in data_array.dims:
            if not data_array.dims:
                raise ValueError("Processed data became a scalar value, cannot create time series.")
            raise ValueError(f"Processed data lost time dimension. Final dims: {data_array.dims}")
        
        ts_pd = data_array.to_pandas()
        
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
            raise ValueError("Time series is all NaNs after selection/averaging.")
        
        return ts_pd

    def _perform_stl_decomposition(self, time_series, period=12):
        """
        Perform STL decomposition on a time series.
        
        Parameters
        ----------
        time_series : pandas.Series
            Time series to decompose
        period : int, default=12
            Period for STL decomposition
            
        Returns
        -------
        pandas.Series
            Trend component
        """
        print("Applying STL decomposition...")
        
        # Clean time series and check length for STL
        original_index = time_series.index
        ts_pd_clean = time_series.dropna()
        
        if ts_pd_clean.empty:
            raise ValueError("Time series is all NaNs after dropping NaN values.")
            
        min_stl_len = 2 * period
        if len(ts_pd_clean) < min_stl_len:
            raise ValueError(f"Time series length ({len(ts_pd_clean)}) is less than required minimum ({min_stl_len}).")
        
        try:
            stl_result = STL(ts_pd_clean, period=period, robust=True).fit()
            trend_component = stl_result.trend.reindex(original_index)
            return trend_component
        except Exception as e:
            print(f"Error during STL decomposition: {e}")
            raise

    def _calculate_linear_regression(self, trend_component, frequency='M'):
        """
        Calculate linear regression on trend component.
        
        Parameters
        ----------
        trend_component : pandas.Series
            Trend component from STL decomposition
        frequency : str, default='M'
            Time frequency of data
            
        Returns
        -------
        dict
            Regression results including model, predicted values, and statistics
        """
        print("Performing linear regression...")
        trend_component_clean = trend_component.dropna()
        if trend_component_clean.empty:
            raise ValueError("Trend component is all NaNs after STL.")
        
        try:
            # Get time units and convert index
            time_info = self._get_time_units(trend_component_clean.index, frequency)
            dates_numeric = time_info['dates_numeric']
            time_unit = time_info['time_unit']
            
            # Calculate regression statistics
            x_vals = dates_numeric.flatten()
            y_vals = trend_component_clean.values
            slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x_vals, y_vals)
            
            # Fit sklearn model for prediction
            reg = LinearRegression()
            reg.fit(dates_numeric, y_vals)
            y_pred_values = reg.predict(dates_numeric)
            y_pred_series = pd.Series(y_pred_values, index=trend_component_clean.index)
            y_pred_series = y_pred_series.reindex(trend_component.index)
            
            # Create statistics DataFrame
            trend_stats_df = pd.DataFrame({
                'statistic': ['slope', 'intercept', 'p_value', 'r_value', 'r_squared', 'standard_error'],
                'value': [slope, intercept, p_value, r_value, r_value**2, slope_std_error]
            })
            
            return {
                'model': reg,
                'predicted': y_pred_series,
                'statistics': trend_stats_df,
                'time_unit': time_unit
            }
        except Exception as e:
            print(f"Error during linear regression: {e}")
            raise

    def _get_time_units(self, time_index, frequency='M'):
        """
        Determine time units and convert time index to numeric values.
        
        Parameters
        ----------
        time_index : pandas.DatetimeIndex or pandas.Index
            Time index from the pandas Series
        frequency : str, default='M'
            Time frequency of data
            
        Returns
        -------
        dict
            Time units information including numeric dates and unit name
        """
        if pd.api.types.is_datetime64_any_dtype(time_index):
            first_date = time_index.min()
            
            # Set appropriate time unit based on frequency
            if frequency == 'M':
                scale = 24*3600*30  # seconds in ~month
                time_unit = "months"
                to_decade = 120      # months in decade
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
            dates_numeric = ((time_index - first_date).total_seconds() / scale).values.reshape(-1, 1)
        elif pd.api.types.is_numeric_dtype(time_index):
            dates_numeric = time_index.values.reshape(-1, 1)
            time_unit = "units"
            to_decade = None
        else:
            raise TypeError(f"Time index type ({time_index.dtype}) not recognized.")
        
        return {
            'dates_numeric': dates_numeric,
            'time_unit': time_unit,
            'to_decade': to_decade
        }

    def _plot_trend_results(self, trend_component, y_pred_series, processed_ts_da, 
                        variable, calculation_type, is_global, area_weighted, 
                        dims_to_average, region_coords, season, level_selection_info=None,
                        save_plot_path=None):
        """
        Plot trend components and regression line.
        
        Parameters
        ----------
        trend_component : pandas.Series
            STL trend component
        y_pred_series : pandas.Series
            Predicted trend values from regression
        processed_ts_da : xarray.DataArray
            Processed data array with metadata
        variable, calculation_type, season : str
            Analysis parameters
        is_global, area_weighted : bool
            Analysis flags
        dims_to_average : list
            Dimensions that were averaged
        region_coords : dict
            Dictionary of coordinates from the analysis
        level_selection_info : str, optional
            Information about level selection
        save_plot_path : str, optional
            Path to save the plot
        """
        print("Generating plot...")
        
        # Get coordinate names
        lat_coord = self._get_coord_name(['lat', 'latitude'])
        lon_coord = self._get_coord_name(['lon', 'longitude'])
        level_coord = self._get_coord_name(['level', 'lev', 'plev', 'zlev'])
        
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

        if level_selection_info:
            coord_strs.append(level_selection_info)
        if season.lower() != 'annual': 
            coord_strs.append(f"Season={season.upper()}")
        
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
        
        
    def _plot_spatial_trends(self, trend_data, variable, units, period_str, 
                       time_period_str, level_info, season,
                       land_only=False, save_plot_path=None, cmap='coolwarm'):
        """
        Plot spatial trend maps using cartopy.
        
        Parameters
        ----------
        trend_data : xarray.DataArray
            DataArray containing trend values
        variable : str
            Variable name for plot title
        units : str
            Units of the variable
        period_str : str
            String describing the time period (e.g., "year", "decade")
        time_period_str : str
            String describing the time range analyzed
        level_info : str
            Information about vertical level
        season : str
            Season analyzed
        land_only : bool, default=False
            Whether to mask ocean areas
        save_plot_path : str, optional
            Path to save the plot
        cmap : str, default='coolwarm'
            Colormap for the plot
        """
        print("Generating plot...")
        cbar_label = f"{variable} trend per {period_str} ({units})" if units else f"{variable} trend per {period_str}"

        try:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

            contour = trend_data.plot.contourf(
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
                ax.add_feature(cfeature.OCEAN, zorder=2, facecolor='white')
                ax.coastlines(zorder=3)
                ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
            else:
                ax.coastlines()
                ax.add_feature(cfeature.BORDERS, linestyle=':')
            
            gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 14}
            gl.ylabel_style = {'size': 14}
            
            # Create title
            season_str = season.upper() if season.lower() != 'annual' else 'Annual'
            title = f"{season_str} {variable.capitalize()} Trends ({units} per {period_str})\n{time_period_str}"
            if level_info:
                title += f"\n{level_info}"
            ax.set_title(title, fontsize=14)

            plt.tight_layout()

            if save_plot_path:
                plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_plot_path}")
            plt.show()

        except Exception as plot_err:
            print(f"An error occurred during plotting: {plot_err}")
    def _apply_stl_slope_function(self, da, period=12, robust_stl=True, frequency='M', num_years=1):
        """
        Calculate trend slope using STL decomposition for a single time series.
        
        Parameters
        ----------
        da : xarray.DataArray or numpy.ndarray
            Time series data to analyze
        period : int, default=12
            Period for STL decomposition (12 for monthly data)
        robust_stl : bool, default=True
            Whether to use robust STL decomposition
        frequency : str, default='M'
            Time frequency: 'M' (monthly), 'D' (daily), 'Y' (yearly)
        num_years : int, default=1
            Number of years to express trend over (e.g., 1=per year, 10=per decade)
            
        Returns
        -------
        float
            Calculated trend slope (in units per specified period) or np.nan if calculation fails
        """
        if hasattr(da, 'values'):
            values = da.values
        else:
            values = da  # Assume numpy array if not xarray

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

            if np.isnan(trend).all():
                return np.nan

            # Fit linear trend only to non-NaN trend values
            valid_indices = ~np.isnan(trend)
            if np.sum(valid_indices) < 2:
                return np.nan

            x = np.arange(len(trend))
            slope, _ = np.polyfit(x[valid_indices], trend[valid_indices], 1)

            # Convert slope to appropriate time units
            if frequency.upper() == 'M':
                slope_per_year = slope * 12
            elif frequency.upper() == 'D':
                slope_per_year = slope * 365.25
            elif frequency.upper() == 'Y' or frequency.upper() == 'A':
                slope_per_year = slope
            elif frequency.upper() == 'W':
                slope_per_year = slope * 52 # weekly
            else:
                slope_per_year = slope * 12
                print(f"Using default conversion for frequency '{frequency}'. Assuming monthly data.")
            
            slope_per_period = slope_per_year * num_years
            
            return slope_per_period
        except Exception:
            return np.nan

#======================HELPER METHODS END=======================#

    def calculate_trends(self, variable='air', latitude=None, longitude=None, level=None,
                    frequency='M', season='annual', area_weighted=True, period=12,
                    plot=True, return_results=False, save_plot_path=None):
        """
 
        Calculate climate time series trends using STL decomposition and linear regression.
        
        This method computes trends for a climate variable using Seasonal-Trend decomposition 
        by LOESS (STL) to extract the trend component, followed by linear regression to 
        quantify the trend rate. Data can be spatially averaged with proper area weighting 
        for regional or global analyses.
        
        Parameters
        ----------
        variable : str, default='air'
            Variable name to analyze (must exist in the dataset)
        latitude : float, slice, or None, optional
            Latitude selection: specific value for point analysis, slice for regional selection,
            or None for all available latitudes (global)
        longitude : float, slice, or None, optional
            Longitude selection: specific value for point analysis, slice for regional selection,
            or None for all available longitudes (global)
        level : float, int, or None, optional
            Vertical level selection (e.g., pressure level in hPa)
            If None and multiple levels exist, defaults to first level
        frequency : str, default='M'
            Time frequency of data: 'M' (monthly), 'D' (daily), 'Y' (yearly)
            Used for proper scaling of trend rates
        season : str, default='annual'
            Season to analyze: 'annual', 'jjas' (Jun-Sep), 'djf' (Dec-Feb), etc.
        area_weighted : bool or None, default=True
            Whether to apply area weighting for spatial averaging
            If None, automatically set to True for global analysis
            Ignored for point analyses
        period : int, default=12
            Period for STL decomposition (e.g., 12 for monthly data)
        plot : bool, default=True
            Whether to generate and display a trend plot
        return_results : bool, default=False
            Whether to return a dictionary with detailed trend results
        save_plot_path : str, optional
            Path to save the trend plot (if None, plot is shown but not saved)
        
        Returns
        -------
        dict or None
            If return_results=True, returns a dictionary containing:
            - 'calculation_type': Type of analysis performed ('point', 'region', 'global')
            - 'trend_component': Pandas Series of STL trend component
            - 'regression_model': Fitted linear regression model
            - 'predicted_trend': Pandas Series of predicted trend values
            - 'area_weighted': Whether area weighting was applied
            - 'region_details': Dict with region metadata
            - 'stl_period': Period used for STL decomposition
            - 'trend_statistics': DataFrame with regression statistics
            If return_results=False, returns None
        
        Notes
        -----
        This method handles three types of analyses:
        1. Point analysis: Analyzes a single grid point time series
        2. Regional analysis: Spatially averages data over a region before trend analysis
        3. Global analysis: Spatially averages data over the entire globe
        
        For valid results, the time series should have at least 2*period time points
        after applying seasonal filtering.
        
        Examples
        --------
        >>> # Calculate global mean temperature trend
        >>> ds.climate_trends.calculate_trends(
        ...     variable='air',
        ...     season='annual',
        ...     area_weighted=True,
        ...     return_results=True
        ... )
        
        >>> # Calculate regional precipitation trend for summer monsoon season
        >>> results = ds.climate_trends.calculate_trends(
        ...     variable='air',
        ...     latitude=slice(30, 10),
        ...     longitude=slice(70, 100),
        ...     season='jjas',
        ...     save_plot_path='monsoon_trend.png',
        ...     return_results=True
        ... )
    """""        
        data_var = self._select_process_data(
            variable=variable, 
            latitude=latitude, 
            longitude=longitude, 
            level=level,
            season=season
        )
        
        # Determine calculation type
        is_global = latitude is None and longitude is None
        is_point = isinstance(latitude, (int, float)) and isinstance(longitude, (int, float))
        calculation_type = 'global' if is_global else ('point' if is_point else 'region')
        
        if area_weighted is None:
            area_weighted = calculation_type == 'global'
        if calculation_type == 'point':
            area_weighted = False
        
        print(f"Starting trend calculation: type='{calculation_type}', variable='{variable}', season='{season}', area_weighted={area_weighted}")
        
        # Get level information for metadata
        level_coord = self._get_coord_name(['level', 'lev', 'plev', 'zlev'])
        level_selection_info = ""
        if level_coord in data_var.coords:
            level_selection_info = f"level={data_var[level_coord].values.item()}"
        
        # Apply spatial averaging
        processed_ts_da, calc_type, dims_to_average, region_coords = self._apply_spatial_averaging(data_var, area_weighted)
        calculation_type = calc_type  # Update calculation type based on dimensions
        
        # Convert to pandas series
        ts_pd = self._convert_to_pandas_series(processed_ts_da)
        
        # Apply STL decomposition
        trend_component = self._perform_stl_decomposition(ts_pd, period)
        
        # Calculate linear regression
        regression_results = self._calculate_linear_regression(trend_component, frequency)
        reg = regression_results['model']
        y_pred_series = regression_results['predicted']
        trend_stats_df = regression_results['statistics']
        
        if plot:
            self._plot_trend_results(
                trend_component, y_pred_series, processed_ts_da, 
                variable, calculation_type, is_global, area_weighted,
                dims_to_average, region_coords, season, level_selection_info,
                save_plot_path
            )
        
        if return_results:
            results = {
                'calculation_type': calculation_type,
                'trend_component': trend_component,
                'regression_model': reg,
                'predicted_trend': y_pred_series,
                'area_weighted': area_weighted,
                'region_details': {'variable': variable, 'season': season, 'level_info': level_selection_info},
                'stl_period': period,
                'trend_statistics': trend_stats_df
            }
            return results
        
        return None
        
    def calculate_spatial_trends(self,
                       variable='air',
                       latitude=slice(None, None),
                       longitude=slice(None, None),
                       time_range=slice(None, None),
                       level=None,
                       season='annual',
                       frequency='M',  
                       num_years=1,    
                       robust_stl=True,
                       period=12,
                       plot_map=True,
                       land_only=False,
                       save_plot_path=None,
                       cmap='coolwarm',
                       return_results = False):
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
        return_results : bool, default=False
            Whether to display the output dataset, or store it in a variable


        Returns
        -------
        xarray.DataArray
            DataArray of computed trends (in units per specified time period).
            Return value includes lat/lon coordinates and appropriate metadata.
        """
        if self._obj is None:
            raise ValueError("Dataset not loaded.")
        
        
        try:
            from dask.distributed import get_client
            client = get_client()
            print(f"Using existing Dask client: {client}")
        except (ImportError, ValueError):
            print("No Dask client detected. For better performance with large datasets,")
            print("consider creating a Dask client before calling this function:")
            print("    from dask.distributed import Client")
            print("    client = Client(n_workers=4, threads_per_worker=1)")
        
        
        period_str = "year" if num_years == 1 else ("decade" if num_years == 10 else f"{num_years} years")
        
        # Get coordinate names
        time_coord = self._get_coord_name(['time'])
        level_coord = self._get_coord_name(['level', 'lev', 'plev', 'zlev'])
        
        # Select and process data
        data_var = self._select_process_data(
            variable=variable,
            latitude=latitude,
            longitude=longitude,
            level=level,
            time_range=time_range,
            season=season
        )
        
        level_info = ""
        if level_coord in data_var.coords:
            level_info = f"Level={data_var[level_coord].values.item()}"
            print(f"Selected level: {data_var[level_coord].values.item()}")
        
        # Ensure minimum required time points for STL
        if len(data_var[time_coord]) < 2 * period:
            raise ValueError(f"Insufficient time points ({len(data_var[time_coord])}) after filtering for season '{season}'. Need at least {2 * period}.")
        
        print(f"Data selected: {data_var.sizes}")
        
        # Ensure the time dimension is a single chunk for apply_ufunc
        data_var = data_var.chunk({time_coord: -1})
        
        print("Computing trends in parallel with xarray...")
        trend_result = xr.apply_ufunc(
            lambda x: self._apply_stl_slope_function(
                x, period=period, robust_stl=robust_stl, frequency=frequency, num_years=num_years
            ),
            data_var,
            input_core_dims=[[time_coord]],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            dask_gufunc_kwargs={'allow_rechunk': True}
        ).rename(f"{variable}_trend_per_{period_str}")
        
        with ProgressBar():
            trend_computed = trend_result.compute()
        print("Trend computation complete.")
        
        if plot_map:
            try:
                start_time = data_var[time_coord].min().dt.strftime('%Y-%m').item()
                end_time = data_var[time_coord].max().dt.strftime('%Y-%m').item()
                time_period_str = f"{start_time} to {end_time}"
            except Exception:
                time_period_str = "Selected Time Period"
            
            units = data_var.attrs.get('units', '')
            variable = data_var.attrs.get('long_name', variable)
            
            self._plot_spatial_trends(
                trend_data=trend_computed,
                variable=variable,
                units=units,
                period_str=period_str,
                time_period_str=time_period_str,
                level_info=level_info,
                season=season,
                land_only=land_only,
                save_plot_path=save_plot_path,
                cmap=cmap
            )
        
        if return_results == True:
            return trend_computed

__all__ = ['TrendsAccessor']
