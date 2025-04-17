import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from dask.diagnostics import ProgressBar
from statsmodels.tsa.seasonal import STL

@xr.register_dataset_accessor("climate_timeseries")
class TimeSeriesAccessor:
    """
    Accessor for analyzing and visualizing climate time series from xarray datasets.
    
    Provides methods for extracting, processing, and visualizing time series
    with support for weighted spatial averaging, seasonal filtering, and time series decomposition.
    """

    def __init__(self, xarray_obj):
        """Initialize the accessor with the provided xarray Dataset."""
        self._obj = xarray_obj

    def _get_coord_name(self, possible_names):
        """Find the actual coordinate name in the dataset from a list of common alternatives."""
        if self._obj is None: return None
        for name in possible_names:
            if name in self._obj.coords:
                return name
        coord_names = {name.lower(): name for name in self._obj.coords}
        for name in possible_names:
            if name.lower() in coord_names:
                return coord_names[name.lower()]
           
        return None

    def _filter_by_season(self, data_subset, season='annual', time_coord_name=None):
        """
        Filter data by meteorological season.
        
        Parameters:
        - data_subset: Data to filter
        - season: Options: 'annual', 'jjas', 'djf', 'mam', 'son'
        - time_coord_name: Name of time coordinate
        """
        
        if time_coord_name is None:
            time_coord_name = self._get_coord_name(['time', 't'])
            if time_coord_name is None:
                raise ValueError("Cannot find time coordinate for seasonal filtering.")
        
        if season.lower() == 'annual':
            return data_subset
            
        if time_coord_name not in data_subset.dims:  
            raise ValueError(f"Cannot filter by season - '{time_coord_name}' dimension not found.")

        if 'month' in data_subset.coords:
            month_coord = data_subset['month']
        elif np.issubdtype(data_subset[time_coord_name].dtype, np.datetime64):  
            month_coord = data_subset[time_coord_name].dt.month  
        else:
            raise ValueError("Cannot determine month for seasonal filtering.")

        season_months = {'jjas': [6, 7, 8, 9], 'djf': [12, 1, 2], 'mam': [3, 4, 5], 'son': [9, 10, 11]}
        selected_months = season_months.get(season.lower())

        if selected_months:
            filtered_data = data_subset.where(month_coord.isin(selected_months), drop=True)
            if filtered_data[time_coord_name].size == 0: 
                print(f"Warning: No data found for season '{season.upper()}' within the selected time range.")
            return filtered_data
        else:
            print(f"Warning: Unknown season '{season}'. Returning unfiltered data.")
            return data_subset

    def _select_process_data(self, variable, latitude=None, longitude=None, level=None, time_range=None, season='annual', year=None):
        """
        Select and process data based on specified parameters.
        
        Handles spatial, vertical and temporal subsetting of data.
        """
        data_filtered = self._obj
        
        # Get coordinate names
        time_name = self._get_coord_name(['time', 't'])
        
        # Filter by season and year if time dimension exists
        if time_name and time_name in data_filtered.dims:
            data_filtered = self._filter_by_season(data_filtered, season, time_name)
            if data_filtered[time_name].size == 0:
                raise ValueError(f"No data available for season '{season}'.")
            
            if year is not None:
                try:
                    year_match = data_filtered[time_name].dt.year == year
                except TypeError:
                    year_match = xr.DataArray([t.year == year for t in data_filtered[time_name].values],
                                            coords={time_name: data_filtered[time_name]}, dims=[time_name])
                data_filtered = data_filtered.sel({time_name: year_match})

                if data_filtered[time_name].size == 0:
                    raise ValueError(f"No data available for year {year} within season '{season}'.")
        elif season.lower() != 'annual' or year is not None:
            print(f"Warning: Cannot filter by season or year - '{time_name}' dimension not found.")

        if variable not in data_filtered.data_vars:
            raise ValueError(f"Variable '{variable}' not found. Available: {list(data_filtered.data_vars)}")
        
        data_var = data_filtered[variable]
        selection_dict = {}
        needs_nearest = False

        # Get coordinate names for spatial dimensions
        lat_name = self._get_coord_name(['lat', 'latitude', 'LAT', 'LATITUDE', 'y', 'rlat', 'nav_lat'])
        lon_name = self._get_coord_name(['lon', 'longitude', 'LON', 'LONGITUDE', 'x', 'rlon', 'nav_lon'])
        level_name = self._get_coord_name(['level', 'lev', 'plev', 'height', 'altitude', 'depth', 'z'])

        # Validate latitude coordinates
        if latitude is not None and lat_name and lat_name in data_var.coords:
            lat_min, lat_max = data_var[lat_name].min().item(), data_var[lat_name].max().item()
            
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
            
            selection_dict[lat_name] = latitude
            if isinstance(latitude, (int, float)):
                needs_nearest = True

        # Validate longitude coordinates
        if longitude is not None and lon_name and lon_name in data_var.coords:
            lon_min, lon_max = data_var[lon_name].min().item(), data_var[lon_name].max().item()
            
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
            
            selection_dict[lon_name] = longitude
            if isinstance(longitude, (int, float)):
                needs_nearest = True

        # Validate time range
        if time_range is not None and time_name and time_name in data_var.dims:
            try:
                time_min, time_max = data_var[time_name].min().values, data_var[time_name].max().values
                
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
            
            selection_dict[time_name] = time_range

        # Handle level selection
        if level_name and level_name in data_var.dims:
            if level is not None:
                level_min, level_max = data_var[level_name].min().item(), data_var[level_name].max().item()
                
                if isinstance(level, (slice, list, np.ndarray)):
                    if isinstance(level, slice):
                        if level.start is not None and level.start > level_max:
                            raise ValueError(f"Requested level minimum {level.start} is greater than available maximum {level_max}")
                        if level.stop is not None and level.stop < level_min:
                            raise ValueError(f"Requested level maximum {level.stop} is less than available minimum {level_min}")
                    elif min(level) > level_max or max(level) < level_min:
                        raise ValueError(f"Requested levels [{min(level)}, {max(level)}] are outside available range [{level_min}, {level_max}]")
                    
                    print(f"Averaging over levels: {level}")
                    with xr.set_options(keep_attrs=True):
                        data_var = data_var.sel({level_name: level}).mean(dim=level_name)
                elif isinstance(level, (int, float)):
                    if level < level_min * 0.5 or level > level_max * 1.5:
                        print(f"Warning: Requested level {level} is far from available range [{level_min}, {level_max}]")
                    
                    selection_dict[level_name] = level
                    needs_nearest = True
                else:
                    selection_dict[level_name] = level
            elif len(data_var[level_name]) > 1:
                level_val = data_var[level_name].values[0]
                selection_dict[level_name] = level_val
                print(f"Warning: Multiple levels found. Using first level: {level_val}")
        elif level is not None:
            print(f"Warning: Level dimension not found. Ignoring 'level' parameter.")
        
        # Apply selections
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

    def plot_time_series(self, 
                         latitude=None, 
                         longitude=None, 
                         level=None,
                         time_range=None, 
                         variable='air', 
                         figsize=(16, 10),
                         season='annual', 
                         year=None, 
                         area_weighted=True,
                         save_plot_path=None):
        """
        Plot time series with spatial averaging.
        
        Creates a time series plot with optional area-weighting and filtering.
        
        Parameters:
        - latitude/longitude/level: Spatial and vertical subsetting
        - time_range: Time period to include
        - variable: Name of variable to plot
        - season: One of 'annual', 'jjas', 'djf', 'mam', 'son'
        - area_weighted: Whether to use latitude weighting
        - save_plot_path: Path to save the figure
        
        Returns:
        - matplotlib.axes.Axes object
        """

        data_var = self._select_process_data(
            variable, latitude, longitude, level, time_range, season, year
        )
        time_name = self._get_coord_name(['time', 't'])

        if not time_name or time_name not in data_var.dims:
            raise ValueError("Time dimension not found in processed data for time series plot.")

        # Get coordinate names
        lat_name = self._get_coord_name(['lat', 'latitude', 'LAT', 'LATITUDE', 'y', 'rlat', 'nav_lat'])
        lon_name = self._get_coord_name(['lon', 'longitude', 'LON', 'LONGITUDE', 'x', 'rlon', 'nav_lon'])
        spatial_dims = [d for d in [lat_name, lon_name] if d and d in data_var.dims]
        plot_data = None

        if spatial_dims:
            if area_weighted and lat_name in spatial_dims:
                weights = np.cos(np.deg2rad(data_var[lat_name]))
                weights.name = "weights"
                plot_data = data_var.weighted(weights).mean(dim=spatial_dims)
                print("Calculating area-weighted spatial mean.")
            else:
                plot_data = data_var.mean(dim=spatial_dims)
                weight_msg = "(unweighted)" if lat_name in spatial_dims else ""
                print(f"Calculating simple spatial mean {weight_msg}.")
        else:
            plot_data = data_var
            print("Plotting time series for single point (no spatial average).")

        if hasattr(plot_data, 'compute'):
            print("Computing time series...")
            with ProgressBar():
                plot_data = plot_data.compute()

        plt.figure(figsize=figsize)
        plot_data.plot()

        ax = plt.gca()
        units = data_var.attrs.get("units", "")
        long_name = data_var.attrs.get("long_name", variable)
        ax.set_ylabel(f"{long_name} ({units})")
        ax.set_xlabel('Time')

        season_display = season.upper() if season.lower() != 'annual' else 'Annual'
        year_display = f"for {year}" if year is not None else ""
        weight_display = "Area-Weighted " if area_weighted and lat_name in spatial_dims else ""
        ax.set_title(f'{season_display} {weight_display}Spatial Mean Time Series {year_display}\nVariable: {variable}')

        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_plot_path is not None:
            plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved to: {save_plot_path}")
        
        return ax


    def plot_std_space(self, 
                       latitude=None, 
                       longitude=None, 
                       level=None,
                       time_range=None, 
                       variable='air', 
                       figsize=(16, 10),
                       season='annual', 
                       area_weighted=True,
                       save_plot_path=None):
        """
        Plot time series of spatial standard deviation.
        
        Shows how spatial variability changes over time with optional area-weighting.
        
        Parameters:
        - latitude/longitude/level: Spatial and vertical subsetting
        - time_range: Time period to include
        - variable: Name of variable to plot
        - season: One of 'annual', 'jjas', 'djf', 'mam', 'son'
        - area_weighted: Whether to use latitude weighting
        - save_plot_path: Path to save the figure
        
        Returns:
        - matplotlib.axes.Axes object
        """

        data_var = self._select_process_data(
             variable, latitude, longitude, level, time_range, season
        )
        time_name = self._get_coord_name(['time', 't'])

        if not time_name or time_name not in data_var.dims:
            raise ValueError("Time dimension not found for spatial standard deviation plot.")


        lat_name = self._get_coord_name(['lat', 'latitude', 'LAT', 'LATITUDE', 'y', 'rlat', 'nav_lat'])
        lon_name = self._get_coord_name(['lon', 'longitude', 'LON', 'LONGITUDE', 'x', 'rlon', 'nav_lon'])
        spatial_dims = [d for d in [lat_name, lon_name] if d and d in data_var.dims]
        plot_data = None

        if not spatial_dims:
            raise ValueError(f"No spatial dimensions ('{lat_name}', '{lon_name}') found for calculating spatial standard deviation.")

        if area_weighted and lat_name in spatial_dims:
            weights = np.cos(np.deg2rad(data_var[lat_name]))
            weights.name = "weights"
            plot_data = data_var.weighted(weights).std(dim=spatial_dims)
            print("Calculating area-weighted spatial standard deviation.")
        else:
            plot_data = data_var.std(dim=spatial_dims)
            weight_msg = "(unweighted)" if lat_name in spatial_dims else ""
            print(f"Calculating simple spatial standard deviation {weight_msg}.")

        if hasattr(plot_data, 'compute'):
            print("Computing spatial standard deviation time series...")
            with ProgressBar():
                plot_data = plot_data.compute()

        plt.figure(figsize=figsize)
        plot_data.plot()

        ax = plt.gca()
        units = data_var.attrs.get("units", "")
        long_name = data_var.attrs.get("long_name", variable)
        ax.set_ylabel(f"Spatial Std. Dev. ({units})")
        ax.set_xlabel('Time')

        season_display = season.upper() if season.lower() != 'annual' else 'Annual'
        weight_display = "Area-Weighted " if area_weighted and lat_name in spatial_dims else ""
        ax.set_title(f'{season_display} Time Series of {weight_display}Spatial Standard Deviation\nVariable: {variable}')

        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_plot_path is not None:
            plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved to: {save_plot_path}")
        return ax


    def decompose_time_series(
        self,
        variable='air',
        level=None,
        latitude=None,
        longitude=None,
        time_range=None,
        season='annual',
        stl_seasonal=13,
        stl_period=12,
        area_weighted=True,
        plot_results=True,
        figsize=(16, 10),
        save_plot_path=None
    ):
        """
        Decompose time series into trend, seasonal, and residual components using STL.
        
        Parameters:
        - variable: Name of the variable to decompose
        - latitude/longitude/level: Spatial and vertical subsetting
        - time_range: Time period to include
        - season: One of 'annual', 'jjas', 'djf', 'mam', 'son'
        - stl_seasonal: Length of the seasonal smoother (must be odd)
        - stl_period: Period of the seasonal component (e.g., 12 for monthly)
        - area_weighted: Whether to use latitude weighting for spatial averaging
        - plot_results: Whether to display decomposition plots
        - save_plot_path: Path to save the figure
        
        Returns:
        - Dictionary with components (original, trend, seasonal, residual)
        - If plot_results is True, also returns the figure object
        """

        data_var = self._select_process_data(
             variable, latitude, longitude, level, time_range, season
        )
        time_name = self._get_coord_name(['time', 't'])


        if not time_name or time_name not in data_var.dims:
            raise ValueError("Time dimension required for decomposition.")

        units = data_var.attrs.get("units", "")
        long_name = data_var.attrs.get("long_name", variable)

        # Get coordinate names
        lat_name = self._get_coord_name(['lat', 'latitude', 'LAT', 'LATITUDE', 'y', 'rlat', 'nav_lat'])
        lon_name = self._get_coord_name(['lon', 'longitude', 'LON', 'LONGITUDE', 'x', 'rlon', 'nav_lon'])
        spatial_dims = [d for d in [lat_name, lon_name] if d and d in data_var.dims]
        ts_mean = None

        if spatial_dims:
            if area_weighted and lat_name in spatial_dims:
                weights = np.cos(np.deg2rad(data_var[lat_name]))
                weights.name = "weights"
                ts_mean = data_var.weighted(weights).mean(dim=spatial_dims)
                print("Calculating area-weighted spatial mean for decomposition.")
            else:
                ts_mean = data_var.mean(dim=spatial_dims)
                print("Calculating simple spatial mean for decomposition.")
        else:
            ts_mean = data_var
            print("Decomposing time series for single point.")

        if hasattr(ts_mean, 'compute'):
            print("Computing mean time series for decomposition...")
            with ProgressBar():
                ts_mean = ts_mean.compute()

        ts_pd = ts_mean.to_pandas().dropna()
        if ts_pd.empty:
            raise ValueError("Time series is empty or all NaN after processing and spatial averaging.")
        if len(ts_pd) <= 2 * stl_period:
            raise ValueError(f"Time series length ({len(ts_pd)}) is too short for the specified STL period ({stl_period}). Needs > 2*period.")

        print(f"Performing STL decomposition (period={stl_period}, seasonal_smooth={stl_seasonal})...")
        try:
            if stl_seasonal % 2 == 0:
                 stl_seasonal += 1
                 print(f"Adjusted stl_seasonal to be odd: {stl_seasonal}")
            stl_result = STL(ts_pd, seasonal=stl_seasonal, period=stl_period).fit()
        except Exception as e:
             print(f"STL decomposition failed: {e}")
             print("Check time series length, NaNs, and stl_period.")
             raise

        results = {
            'original': stl_result.observed,
            'trend': stl_result.trend,
            'seasonal': stl_result.seasonal,
            'residual': stl_result.resid
        }

        if plot_results:
            print("Plotting decomposition results...")
            fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

            axes[0].plot(results['original'].index, results['original'].values, label='Observed')
            axes[0].set_ylabel(f"Observed ({units})")
            title_prefix = f'{season.upper() if season.lower() != "annual" else "Annual"}'
            level_info = f" at {level}" if level is not None else ""
            axes[0].set_title(f'{title_prefix} Time Series Decomposition: {long_name}{level_info}')

            axes[1].plot(results['trend'].index, results['trend'].values, label='Trend')
            axes[1].set_ylabel(f"Trend ({units})")

            axes[2].plot(results['seasonal'].index, results['seasonal'].values, label='Seasonal')
            axes[2].set_ylabel(f"Seasonal ({units})")

            axes[3].plot(results['residual'].index, results['residual'].values, label='Residual', linestyle='-', marker='.', markersize=2, alpha=0.7)
            axes[3].axhline(0, color='grey', linestyle='--', alpha=0.5)
            axes[3].set_ylabel(f"Residual ({units})")
            axes[3].set_xlabel("Time")

            for ax in axes:
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend(loc='upper left', fontsize='small')

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            
            if save_plot_path is not None:
                plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
                print(f"Plot saved to: {save_plot_path}")
            return results, fig
        else:
            return results
__all__ = ['TimeSeriesAccessor']
