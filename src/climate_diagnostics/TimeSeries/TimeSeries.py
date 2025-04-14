import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from dask.diagnostics import ProgressBar
from statsmodels.tsa.seasonal import STL

@xr.register_dataset_accessor("climate_timeseries")
class TimeSeriesAccessor:
    """
    Accessor for analyzing and visualizing climate time series from xarray datasets.
    
    This accessor provides methods for extracting, processing, and visualizing time series
    from climate datasets with support for:
    - Weighted spatial averaging across regions
    - Seasonal filtering and temporal subsetting
    - Vertical level selection
    - Time series decomposition using STL
    - Spatial variability analysis
    
    Examples
    --------
   >>> import xarray as xr
    >>> from climate_diagnostics import register_accessors
    >>> ds = xr.open_dataset("climate_data.nc")
    >>> 
    >>> # Plot time series of temperature averaged over a given slice
    >>> ds.climate_timeseries.plot_time_series(
    ...     variable="air",
    ...     latitude=slice(60, 6),
    ...     longitude=slice(60, 110),
    ...     season="annual"
    ... )
    >>> 
    >>> # Plot monsoon precipitation time series with area weighting
    >>> ds.climate_timeseries.plot_time_series(
    ...     variable="prate", 
    ...     latitude=slice(60, 4),
    ...     longitude=slice(60, 100), 
    ...     season="jjas",
    ...     time_range=slice("2000-01-01", "2020-12-31"),
    ...     area_weighted=True
    ... )
    >>> 
    >>> # Decompose precipitation time series into trend and seasonal components
    >>> decomp = ds.climate_timeseries.decompose_time_series(
    ...     variable="prate", 
    ...     level=850,
    ...     stl_period=12,  # Monthly data
    ...     plot_results=True
    ... )
    """

    def __init__(self, xarray_obj):
        """
        Initialize the accessor with the provided xarray Dataset.
        
        Parameters
        ----------
        xarray_obj : xarray.Dataset
            The Dataset object this accessor is attached to
        """
        self._obj = xarray_obj

    

    def _filter_by_season(self, data_subset, season='annual'):
        """
        Filter data subset by meteorological season.
    
        Parameters
        ----------
        data_subset : xarray.Dataset or xarray.DataArray
            Data to be filtered by season
        season : str, default 'annual'
            Meteorological season to filter by. Options:
            - 'annual': All seasons (no filtering)
            - 'jjas': June, July, August, September (summer monsoon)
            - 'djf': December, January, February (winter)
            - 'mam': March, April, May (spring)
            - 'son': September, October, November (fall)
        
        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Filtered data containing only the specified season
            
        Raises
        ------
        ValueError
            If time dimension is missing or month information cannot be determined
        """
        if season.lower() == 'annual':
            return data_subset
        if 'time' not in data_subset.dims:
            raise ValueError("Cannot filter by season - 'time' dimension not found.")

        if 'month' in data_subset.coords:
            month_coord = data_subset['month']
        elif np.issubdtype(data_subset['time'].dtype, np.datetime64):
             month_coord = data_subset.time.dt.month
        else:
             raise ValueError("Cannot determine month for seasonal filtering.")

        season_months = {'jjas': [6, 7, 8, 9], 'djf': [12, 1, 2], 'mam': [3, 4, 5], 'son': [9, 10, 11]}
        selected_months = season_months.get(season.lower())

        if selected_months:
            filtered_data = data_subset.where(month_coord.isin(selected_months), drop=True)
            if filtered_data.time.size == 0:
                 print(f"Warning: No data found for season '{season.upper()}' within the selected time range.")
            return filtered_data
        else:
            print(f"Warning: Unknown season '{season}'. Returning unfiltered data.")
            return data_subset

    def _select_process_data(self, variable, latitude=None, longitude=None, level=None, time_range=None, season='annual', year=None):
        """
        Select and process data based on specified parameters.
    
        This helper method handles data selection and filtering based on:
        - Variable selection
        - Spatial subsetting (latitude/longitude)
        - Vertical level selection
        - Temporal filtering (time range, season, year)
        
        Parameters
        ----------
        variable : str
            Name of the variable to select from the dataset
        latitude : float, list, slice, or None
            Latitude selection criteria
        longitude : float, list, slice, or None
            Longitude selection criteria
        level : float, int, list, slice, or None
            Vertical level selection criteria. If a slice or list is provided,
            the levels will be averaged.
        time_range : slice or None
            Time range selection criteria
        season : str, default 'annual'
            Season to filter by ('annual', 'jjas', 'djf', 'mam', 'son')
        year : int or None
            Specific year to filter by
            
        Returns
        -------
        xarray.DataArray
            Processed data subset based on the specified selection criteria
            
        Raises
        ------
        ValueError
            If the dataset is not loaded, variable doesn't exist, or if specified selections 
            result in empty data
            
        Notes
        -----
        This method handles both exact coordinate matching and nearest-neighbor selection
        when appropriate. For level selection, it will automatically average multiple levels
        if a slice or list is provided.
        """
        data_filtered = self._obj

        if 'time' in data_filtered.dims:
            data_filtered = self._filter_by_season(data_filtered, season)
            if data_filtered.time.size == 0:
                raise ValueError(f"No data available for season '{season}'.")
            if year is not None:
                try:
                    year_match = data_filtered.time.dt.year == year
                except TypeError:
                    year_match = xr.DataArray([t.year == year for t in data_filtered.time.values],
                                            coords={'time': data_filtered.time}, dims=['time'])
                data_filtered = data_filtered.sel(time=year_match)

                if data_filtered.time.size == 0:
                    raise ValueError(f"No data available for year {year} within season '{season}'.")
        elif season.lower() != 'annual' or year is not None:
            print("Warning: Cannot filter by season or year - 'time' dimension not found.")

        if variable not in data_filtered.data_vars:
            raise ValueError(f"Variable '{variable}' not found. Available: {list(data_filtered.data_vars)}")
        data_var = data_filtered[variable]

        selection_dict = {}
        needs_nearest = False

        # Validate latitude coordinates
        if latitude is not None and 'lat' in data_var.coords:
            lat_min, lat_max = data_var.lat.min().item(), data_var.lat.max().item()
            
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
            
            selection_dict['lat'] = latitude
            if isinstance(latitude, (int, float)):
                needs_nearest = True

        # Validate longitude coordinates
        if longitude is not None and 'lon' in data_var.coords:
            lon_min, lon_max = data_var.lon.min().item(), data_var.lon.max().item()
            
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
            
            selection_dict['lon'] = longitude
            if isinstance(longitude, (int, float)):
                needs_nearest = True

        # Validate time range
        if time_range is not None and 'time' in data_var.dims:
            try:
                time_min, time_max = data_var.time.min().values, data_var.time.max().values
                
                # Check if time selection is completely outside the available range
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
                # If we can't compare the times, proceed and let xarray handle it
                print(f"Warning: Could not validate time range: {e}")
            
            selection_dict['time'] = time_range

        level_dim_name = next((dim for dim in ['level', 'lev'] if dim in data_var.dims), None)
        if level_dim_name:
            if level is not None:
                # Validate level coordinates
                level_min, level_max = data_var[level_dim_name].min().item(), data_var[level_dim_name].max().item()
                
                if isinstance(level, (slice, list, np.ndarray)):
                    # Check if completely outside range
                    if isinstance(level, slice):
                        if level.start is not None and level.start > level_max:
                            raise ValueError(f"Requested level minimum {level.start} is greater than available maximum {level_max}")
                        if level.stop is not None and level.stop < level_min:
                            raise ValueError(f"Requested level maximum {level.stop} is less than available minimum {level_min}")
                    elif min(level) > level_max or max(level) < level_min:
                        raise ValueError(f"Requested levels [{min(level)}, {max(level)}] are outside available range [{level_min}, {level_max}]")
                    
                    print(f"Averaging over levels: {level}")
                    with xr.set_options(keep_attrs=True):
                        data_var = data_var.sel({level_dim_name: level}).mean(dim=level_dim_name)
                elif isinstance(level, (int, float)):
                    # For single values with 'nearest' method, warn if far outside range
                    if level < level_min * 0.5 or level > level_max * 1.5:
                        print(f"Warning: Requested level {level} is far from available range [{level_min}, {level_max}]")
                    
                    selection_dict[level_dim_name] = level
                    needs_nearest = True
                else:
                    selection_dict[level_dim_name] = level
            elif len(data_var[level_dim_name]) > 1:
                level_val = data_var[level_dim_name].values[0]
                selection_dict[level_dim_name] = level_val
                print(f"Warning: Multiple levels found. Using first level: {level_val}")
        elif level is not None:
            print(f"Warning: Level dimension not found. Ignoring 'level' parameter.")
        
        # Continue with existing code for selection
        if selection_dict:
            has_slice = any(isinstance(v, slice) for v in selection_dict.values())
            method_to_use = 'nearest' if needs_nearest and not has_slice else None

            if method_to_use is None and needs_nearest and has_slice:
                print("Warning: Using slice selection for one dimension and nearest neighbor "
                    "for another in the same .sel() call is not supported by xarray. "
                    "Nearest neighbor selection will be ignored for non-slice dimensions in this step.")

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
                         save_plot_path = None):
        """
        Plot time series of spatial standard deviation.
    
        Creates a time series plot showing how the spatial variability (standard deviation)
        of the selected variable evolves over time, with optional area-weighting
        and seasonal/temporal filtering.
        
        Parameters
        ----------
        latitude : float, list, slice, or None
            Latitude selection criteria for spatial subsetting
        longitude : float, list, slice, or None
            Longitude selection criteria for spatial subsetting
        level : float, int, list, slice, or None
            Vertical level selection criteria. Multiple levels will be averaged.
        time_range : slice or None
            Time range to include in the plot
        variable : str, default 'air'
            Name of the variable to analyze
        figsize : tuple, default (16, 10)
            Figure size in inches (width, height)
        season : str, default 'annual'
            Season to filter by ('annual', 'jjas', 'djf', 'mam', 'son')
        area_weighted : bool, default True
            Whether to use cosine(latitude) weighting for spatial standard deviation
        save_plot_path : str, optional
            Path where the plot should be saved. If None (default), the plot is not saved.
            
        Returns
        -------
        matplotlib.axes.Axes
            The plot's axes object for further customization
            
        Raises
        ------
        ValueError
            If time dimension is not found or if no spatial dimensions exist
            
        Notes
        -----
        - Area weighting is applied using cosine of latitude when enabled
        - For Dask arrays, computation is performed with a progress bar
        - The plot displays appropriate units and metadata from the variable attributes
        - Grid lines are added for better readability
            
        Examples
        --------
        >>> # Basic time series of temperature over a region
        >>> ds.climate_timeseries.plot_time_series(
        ...     variable='air',
        ...     latitude=slice(40, 6),
        ...     longitude=slice(60, 100)
        ... )
        >>> 
        >>> # Compare winter temperature from specific pressure level
        >>> ds.climate_timeseries.plot_time_series(
        ...     variable='air',
        ...     level=500,  # 500 hPa
        ...     season='djf',
        ...     time_range=slice('2000-01-01', '2020-12-31'),
        ...     save_plot_path='/home/user/Downloads/winter_temp_500hPa.png'
        ... )
        >>> 
        >>> # Time series of precipitation without area weighting
        >>> ax = ds.climate_timeseries.plot_time_series(
        ...     variable='prate',
        ...     latitude=slice(40, 6),
        ...     longitude=slice(60,110),
        ...     area_weighted=False
        ... )
        >>> # Customize the plot further
        >>> ax.set_ylim(0, 10)
        >>> ax.set_title('Tropical Precipitation')
        """

        data_var = self._select_process_data(
            variable, latitude, longitude, level, time_range, season, year
        )

        if 'time' not in data_var.dims:
            raise ValueError("Time dimension not found in processed data for time series plot.")

        spatial_dims = [d for d in ['lat', 'lon'] if d in data_var.dims]
        plot_data = None

        if spatial_dims:
            if area_weighted and 'lat' in spatial_dims:
                weights = np.cos(np.deg2rad(data_var['lat']))
                weights.name = "weights"
                plot_data = data_var.weighted(weights).mean(dim=spatial_dims)
                print("Calculating area-weighted spatial mean.")
            else:
                plot_data = data_var.mean(dim=spatial_dims)
                weight_msg = "(unweighted)" if 'lat' in spatial_dims else ""
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
        weight_display = "Area-Weighted " if area_weighted and 'lat' in spatial_dims else ""
        ax.set_title(f'{season_display} {weight_display}Spatial Mean Time Series {year_display}\nVariable: {variable}')

        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        # Save plot if path is provided
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
                       save_plot_path = None):
        """
        Plot time series of spatial standard deviation.
    
        Creates a time series plot showing how the spatial variability (standard deviation)
        of the selected variable evolves over time, with optional area-weighting
        and seasonal/temporal filtering.
        
        Parameters
        ----------
        latitude : float, list, slice, or None
            Latitude selection criteria for spatial subsetting
        longitude : float, list, slice, or None
            Longitude selection criteria for spatial subsetting
        level : float, int, list, slice, or None
            Vertical level selection criteria. Multiple levels will be averaged.
        time_range : slice or None
            Time range to include in the plot
        variable : str, default 'air'
            Name of the variable to analyze
        figsize : tuple, default (16, 10)
            Figure size in inches (width, height)
        season : str, default 'annual'
            Season to filter by ('annual', 'jjas', 'djf', 'mam', 'son')
        area_weighted : bool, default True
            Whether to use cosine(latitude) weighting for spatial standard deviation
        save_plot_path : str, optional
            Path where the plot should be saved. If None (default), the plot is not saved.
            
        Returns
        -------
        matplotlib.axes.Axes
            The plot's axes object for further customization
            
        Raises
        ------
        ValueError
            If time dimension is not found or if no spatial dimensions exist
        Notes
        -----
        - This method is particularly useful for analyzing the variability across regions
        - Higher standard deviation values indicate greater spatial heterogeneity
        - Area weighting accounts for the decreasing grid cell size toward the poles
        - The computation uses Dask's progress bar
        
        Examples
        --------
        >>> # Basic spatial variability of temperature over time
        >>> ds.climate_timeseries.plot_std_space(
        ...     variable='air',
        ...     longitude = slice(40,6)
        ...     latitude=slice(60,110), 
        ...     area_weighted=True
        ... )
        >>> 
        >>> # Regional monsoon precipitation variability
        >>> ax = ds.climate_timeseries.plot_std_space(
        ...     variable='prate',
        ...     latitude=slice(40, 6),
        ...     longitude=slice(60, 110),
        ...     season='jjas',  # Summer monsoon
        ...     time_range=slice('1980-01-01', '2020-12-31')
        ... )
        >>> 
        >>> # spatial variability at different atmospheric levels
        >>> ds.climate_timeseries.plot_std_space(
        ...     variable='geopotential_height',
        ...     level=850,
        ... )
        >>> ds.climate_timeseries.plot_std_space(
        ...     variable='geopotential_height',
        ...     level=500,
        ... )
        
        """

        data_var = self._select_process_data(
             variable, latitude, longitude, level, time_range, season
        )

        if 'time' not in data_var.dims:
            raise ValueError("Time dimension not found for spatial standard deviation plot.")

        spatial_dims = [d for d in ['lat', 'lon'] if d in data_var.dims]
        plot_data = None

        if not spatial_dims:
             raise ValueError("No spatial dimensions ('lat', 'lon') found for calculating spatial standard deviation.")

        if area_weighted and 'lat' in spatial_dims:
            weights = np.cos(np.deg2rad(data_var['lat']))
            weights.name = "weights"
            plot_data = data_var.weighted(weights).std(dim=spatial_dims)
            print("Calculating area-weighted spatial standard deviation.")
        else:
            plot_data = data_var.std(dim=spatial_dims)
            weight_msg = "(unweighted)" if 'lat' in spatial_dims else ""
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
        weight_display = "Area-Weighted " if area_weighted and 'lat' in spatial_dims else ""
        ax.set_title(f'{season_display} Time Series of {weight_display}Spatial Standard Deviation\nVariable: {variable}')

        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        # Save plot if path is provided
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
        save_plot_path = None
    ):
        """
        Decompose a time series into trend, seasonal, and residual components.
        
        Applies the Seasonal-Trend decomposition using LOESS (STL) to the spatially 
        averaged time series. The decomposition separates the time series into:
        - Trend component (long-term change)
        - Seasonal component (recurring patterns)
        - Residual component (remaining variation)
        
        Parameters
        ----------
        variable : str, default 'air'
            Name of the variable to decompose
        level : float, int, list, slice, or None
            Vertical level selection criteria. Multiple levels will be averaged.
        latitude : float, list, slice, or None
            Latitude selection criteria for spatial subsetting
        longitude : float, list, slice, or None
            Longitude selection criteria for spatial subsetting
        time_range : slice or None
            Time range to include in the decomposition
        season : str, default 'annual'
            Season to filter by ('annual', 'jjas', 'djf', 'mam', 'son')
        stl_seasonal : int, default 13
            STL decomposition parameter: Length of the seasonal smoother.
            Must be odd; if even, will be incremented by 1.
        stl_period : int, default 12
            STL decomposition parameter: Period of the seasonal component 
            (e.g., 12 for monthly data, 4 for quarterly data)
        area_weighted : bool, default True
            Whether to use cosine(latitude) weighting for spatial averaging
        plot_results : bool, default True
            Whether to create and display decomposition plots
        figsize : tuple, default (16, 10)
            Figure size in inches (width, height) when plotting
        save_plot_path : str, optional
            Path where the plot should be saved if plot_results is True. 
            If None (default), the plot is not saved.
            
        Returns
        -------
        dict or tuple
            If plot_results is False, returns a dictionary with the decomposition components
            (original, trend, seasonal, residual).
            If plot_results is True, returns a tuple (dictionary, figure) with both the
            decomposition components and the matplotlib figure object.
            
        Raises
        ------
        ValueError
            If the time series is too short for decomposition or contains invalid values
            
        Notes
        -----
        - The STL decomposition requires at least 2*period+1 data points
        - For monthly data (period=12), this means at least 25 time points
        - The stl_seasonal parameter controls smoothness of the seasonal component
        - Higher stl_seasonal values result in smoother seasonal components
        - Time series is first spatially averaged (area-weighted) before decomposition
        - The trend component can be used to analyze long-term climate change signals
        - The seasonal component reveals recurring annual patterns
        - The residual component can highlight anomalous events
        
        Examples
        ---------
         
        >>> # Analyze regional precipitation trends without plotting
        >>> precip_components = ds.climate_timeseries.decompose_time_series(
        ...     variable='prate',
        ...     latitude=slice(40, 6),
        ...     longitude=slice(60, 100),
        ...     plot_results=False,
        ...     stl_seasonal=13,  # Smoother seasonal component
        ...     stl_period=12
        ... )
        
    """

        data_var = self._select_process_data(
             variable, latitude, longitude, level, time_range, season
        )

        if 'time' not in data_var.dims:
            raise ValueError("Time dimension required for decomposition.")

        units = data_var.attrs.get("units", "")
        long_name = data_var.attrs.get("long_name", variable)

        spatial_dims = [d for d in ['lat', 'lon'] if d in data_var.dims]
        ts_mean = None

        if spatial_dims:
            if area_weighted and 'lat' in spatial_dims:
                weights = np.cos(np.deg2rad(data_var['lat']))
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
            # Save plot if path is provided
            if save_plot_path is not None:
                plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
                print(f"Plot saved to: {save_plot_path}")
            return results, fig
        else:
            return results
__all__ = ['TimeSeriesAccessor']