import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from dask.diagnostics import ProgressBar
from statsmodels.tsa.seasonal import STL

class TimeSeries:
    """
    A class for analyzing and visualizing time series from climate data.
    
    This class provides methods to load, filter, and analyze climate data
    from NetCDF files. It supports spatial averaging with proper area weighting,
    seasonal filtering, time series visualization, and decomposition into trend,
    seasonal, and residual components using STL.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the NetCDF or other compatible climate data file.
        
    Attributes
    ----------
    filepath : str
        Path to the input data file.
    dataset : xarray.Dataset
        Loaded dataset with climate variables.
        
    Examples
    --------
    >>> from climate_diagnostics import TimeSeries
    >>> ts = TimeSeries("/path/to/climate_data.nc")
    >>> ts.plot_time_series(variable="air", level=850, season="djf")
    
    Notes
    -----
    This class uses dask for efficient handling of large climate datasets.
    """
    
    def __init__(self, filepath=None):
        """
        Initialize the TimeSeries class with climate data.
        
        Parameters
        ----------
        filepath : str, optional
            Path to the NetCDF or other compatible climate data file.
            If None, no data will be loaded automatically.
        """
        self.filepath = filepath
        self.dataset = None
        self._load_data()
        
    def _load_data(self):
        """
        Load dataset from the provided filepath with automatic chunking.
        
        Uses dask for efficient memory management when handling large
        climate datasets. Data is loaded with automatic chunking to 
        optimize performance.
        
        Raises
        ------
        Exception
            If the file cannot be loaded or is in an incompatible format.
        """
        try:
            if self.filepath:
                self.dataset = xr.open_dataset(self.filepath, chunks='auto')
                print(f"Dataset loaded from {self.filepath} with auto-chunking")
            else:
                print("Invalid filepath provided. Please specify a valid filepath.")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def _filter_by_season(self, season='annual'):
        """
        Filter the dataset by meteorological season.
        
        Creates a subset of the dataset containing only data from the 
        specified meteorological season.
        
        Parameters
        ----------
        season : str, optional
            Season to filter by. Options are:
            - 'annual': All months (default)
            - 'jjas': June, July, August, September (Northern Hemisphere summer)
            - 'djf': December, January, February (Northern Hemisphere winter)
            - 'mam': March, April, May (Northern Hemisphere spring)
            
        Returns
        -------
        xarray.Dataset
            Filtered dataset containing only data from the specified season
            
        Raises
        ------
        ValueError
            If no dataset is available for filtering
            
        Notes
        -----
        This function requires a time dimension in the dataset with datetime
        values that include month information.
        """
        if self.dataset is None:
            raise ValueError("No dataset available for filtering. Please load data first.")
            
        if season.lower() == 'annual':
            return self.dataset
            
        if 'time' not in self.dataset.dims:
            print("Warning: Cannot filter by season - no time dimension found.")
            return self.dataset
            
        filtered_data = self.dataset.copy()
            
        if 'month' not in filtered_data.coords:
            filtered_data = filtered_data.assign_coords(month=filtered_data.time.dt.month)
            
        if season.lower() == 'jjas':
            return filtered_data.sel(time=filtered_data.time.dt.month.isin([6, 7, 8, 9]))
        elif season.lower() == 'djf':
            return filtered_data.sel(time=filtered_data.time.dt.month.isin([12, 1, 2]))
        elif season.lower() == 'mam':
            return filtered_data.sel(time=filtered_data.time.dt.month.isin([3, 4, 5]))
        else:
            print(f"Warning: Unknown season '{season}'. Using annual data.")
            return filtered_data
            
    def plot_time_series(self, latitude=None, longitude=None, level=None,
                         time_range=None, variable='air', figsize=(20, 10),
                         season='annual', year=None):
        """
        Plot the time series for the selected variable with proper area weighting.
        
        Creates a time series visualization of the specified climate variable,
        spatially averaged over the selected region with proper area weighting
        that accounts for the decreasing grid cell area toward the poles.
        
        Parameters
        ----------
        latitude : float, slice, or array-like, optional
            Latitude selection. Can be a specific value, a slice (e.g., 
            slice(-30, 30) for 30°S to 30°N), or an array of values.
        longitude : float, slice, or array-like, optional
            Longitude selection. Can be a specific value, a slice (e.g.,
            slice(0, 90) for 0° to 90°E), or an array of values.
        level : float or int, optional
            Pressure level selection in hPa or the unit used in the dataset.
            If None, the first level is used if available.
        time_range : slice or str, optional
            Time range selection. Can be a slice (e.g., slice('2010', '2020'))
            or a string format understood by xarray.
        variable : str, optional
            Variable name to plot (default: 'air' for air temperature)
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (20, 10).
        season : str, optional
            Season to plot. Options are:
            - 'annual': All months (default)
            - 'jjas': June, July, August, September (Northern Hemisphere summer)
            - 'djf': December, January, February (Northern Hemisphere winter)
            - 'mam': March, April, May (Northern Hemisphere spring)
        year : int, optional
            If provided, plots data only for the specified year.
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot, which can be further customized.
            
        Raises
        ------
        ValueError
            If the dataset is not loaded, the season has no data, or the 
            variable is not found in the dataset.
            
        Examples
        --------
        >>> ts = TimeSeries("climate_data.nc")
        >>> # Plot global mean temperature at 850 hPa for winter season
        >>> ts.plot_time_series(
        ...     level=850,
        ...     variable="air",
        ...     season="djf"
        ... )
        
        >>> # Plot temperature for a specific region and time period
        >>> ts.plot_time_series(
        ...     latitude=slice(-15, 15),   # 15°S to 15°N
        ...     longitude=slice(40, 100),  # 40°E to 100°E
        ...     level=500,                 # 500 hPa
        ...     time_range=slice('2000', '2020'),
        ...     variable="air"
        ... )
            
        Notes
        -----
        This method applies cosine latitude weighting to properly account
        for the decreasing grid cell area toward the poles when calculating
        spatial averages.
        """
        if self.dataset is None:
            raise ValueError("No dataset available for plotting. Please load data first.")
        
        data = self._filter_by_season(season)
        
        if year is not None:
            data = data.sel(time=data.time.dt.year == year)
            
        if 'time' in data.dims and len(data.time) == 0:
            raise ValueError(f"No data available for season '{season}' in the dataset.")
        
        if variable not in list(data.data_vars):
            raise ValueError(f"Variable '{variable}' not found in dataset. Available variables: {list(data.data_vars)}")

        # Set default level if None and level dimension exists
        if level is None and 'level' in data.dims and len(data.level) > 0:
            level = data.level.values[0]

        if latitude is not None:
            data = data.sel(lat=latitude, method='nearest' if isinstance(latitude, (int, float)) else None)

        if longitude is not None:
            data = data.sel(lon=longitude, method='nearest' if isinstance(longitude, (int, float)) else None)

        if 'level' in data.dims:
            if level is not None:
                if isinstance(level, (slice, list, np.ndarray)):
                    data = data.sel(level=level)
                    data = data.mean(dim='level')
                else:
                    data = data.sel(level=level)
        elif 'lev' in data.dims:  
            if level is not None:
                if isinstance(level, (slice, list, np.ndarray)):
                    data = data.sel(lev=level)
                    data = data.mean(dim='lev')
                else:
                    data = data.sel(lev=level)
        else:
            print("Warning: Level dimension not found in dataset.")

        if 'time' not in data.dims:
            raise ValueError("Time dimension not found in dataset. Please load data with time dimension.")
            
        if time_range is not None:
            data = data.sel(time=time_range)
        
        spatial_dims = []
        if 'lat' in data.dims:
            spatial_dims.append('lat')
        if 'lon' in data.dims:
            spatial_dims.append('lon')
        
        if spatial_dims:
            weights = np.cos(np.deg2rad(data.lat))
            plot_data = (data[variable] * weights).sum(dim=spatial_dims) / weights.sum()
        else:
            plot_data = data[variable]
            
        if hasattr(data[variable], 'compute'):
            with ProgressBar():
                plot_data = plot_data.compute()

        plt.figure(figsize=figsize)
        ax = plot_data.plot()
        
        season_display = season.upper() if season.lower() != 'annual' else 'Annual'
        plt.title(f'{season_display} Time series of {variable}')
        plt.xlabel('Time')
        plt.ylabel(f'{variable} {data[variable].attrs.get("units", "")}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        return ax
    
    def plot_std_space(self, latitude=None, longitude=None, level=None,
                       time_range=None, variable='air', figsize=(20, 10),
                       season='annual'):
        """
        Plot the spatial standard deviation time series of the selected variable.
        
        Creates a time series visualization showing how the spatial variability
        (standard deviation across latitudes and longitudes) of the selected
        variable changes over time.
        
        Parameters
        ----------
        latitude : float, slice, or array-like, optional
            Latitude selection. Can be a specific value, a slice (e.g., 
            slice(-30, 30) for 30°S to 30°N), or an array of values.
        longitude : float, slice, or array-like, optional
            Longitude selection. Can be a specific value, a slice (e.g.,
            slice(0, 90) for 0° to 90°E), or an array of values.
        level : float or int, optional
            Pressure level selection in hPa or the unit used in the dataset.
            If None, the first level is used if available.
        time_range : slice or str, optional
            Time range selection. Can be a slice (e.g., slice('2010', '2020'))
            or a string format understood by xarray.
        variable : str, optional
            Variable name to plot (default: 'air' for air temperature)
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (20, 10).
        season : str, optional
            Season to plot. Options are:
            - 'annual': All months (default)
            - 'jjas': June, July, August, September (Northern Hemisphere summer)
            - 'djf': December, January, February (Northern Hemisphere winter)
            - 'mam': March, April, May (Northern Hemisphere spring)
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot, which can be further customized.
            
        Raises
        ------
        ValueError
            If the dataset is not loaded, the season has no data, the variable
            is not found, or there's no time dimension.
            
        Examples
        --------
        >>> ts = TimeSeries("climate_data.nc")
        >>> # Plot spatial standard deviation of temperature
        >>> ts.plot_std_space(
        ...     level=850,
        ...     variable="air",
        ...     season="djf"
        ... )
        
        >>> # Plot spatial variability for a specific region 
        >>> ts.plot_std_space(
        ...     latitude=slice(0, 45),     # 0° to 45°N
        ...     longitude=slice(60, 120),  # 60°E to 120°E
        ...     level=500,                 # 500 hPa
        ...     variable="air",
        ...     season="mam"               # March-April-May
        ... )
            
        Notes
        -----
        This method calculates the standard deviation across spatial dimensions
        (latitude and longitude) for each time point, providing insights into
        how spatial variability changes over time.
        """
        if self.dataset is None:
            raise ValueError("No dataset available for plotting. Please load data first.")
        
        data = self._filter_by_season(season)
        
        if 'time' in data.dims and len(data.time) == 0:
            raise ValueError(f"No data available for season '{season}' in the dataset.")
        
        if variable not in list(data.data_vars):
            raise ValueError(f"Variable '{variable}' not found in dataset. Available variables: {list(data.data_vars)}")

        # Set default level if None and level dimension exists
        if level is None and 'level' in data.dims and len(data.level) > 0:
            level = data.level.values[0]

        if latitude is not None:
            data = data.sel(lat=latitude, method='nearest' if isinstance(latitude, (int, float)) else None)

        if longitude is not None:
            data = data.sel(lon=longitude, method='nearest' if isinstance(longitude, (int, float)) else None)

        if 'level' in data.dims:
            if level is not None:
                if isinstance(level, (slice, list, np.ndarray)):
                    data = data.sel(level=level)
                    data = data.mean(dim='level')
                else:
                    data = data.sel(level=level)
        elif 'lev' in data.dims:  
            if level is not None:
                if isinstance(level, (slice, list, np.ndarray)):
                    data = data.sel(lev=level)
                    data = data.mean(dim='lev')
                else:
                    data = data.sel(lev=level)
        else:
            print("Warning: Level dimension not found in dataset.")

        if 'time' not in data.dims:
            raise ValueError("Time dimension not found in dataset. Please load data with time dimension.")
            
        if time_range is not None:
            data = data.sel(time=time_range)
            
        # Calculate the standard deviation over latitude and longitude
        data = data.std(dim=['lat', 'lon'])
        
        if hasattr(data[variable], 'compute'):
            with ProgressBar():
                data = data.compute()

        plt.figure(figsize=figsize)
        ax = data[variable].plot()
        
        season_display = season.upper() if season.lower() != 'annual' else 'Annual'
        plt.title(f'{season_display} Normalized Standard Deviation of {variable} over Latitude and Longitude')
        plt.xlabel('Time')
        plt.ylabel(f'Normalized {variable} Standard Deviation')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
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
        figsize=(14, 12)
    ):
        """
        Decompose a climate time series into trend, seasonal, and residual components.
        
        Uses the Seasonal-Trend decomposition using LOESS (STL) method to separate
        a climate time series into its constituent components. The method computes
        a spatial average for the selected region with proper area weighting, then
        performs the decomposition, extracting trend, seasonal cycle, and residual
        components.
        
        Parameters
        ----------
        variable : str, optional
            Variable name to analyze (default: 'air' for air temperature)
        level : float or int, optional
            Pressure level to analyze in hPa. If None, uses first available level.
        latitude : float, slice, or array-like, optional
            Latitude selection. Can be a specific value, a slice (e.g., 
            slice(-30, 30) for 30°S to 30°N), or an array of values.
        longitude : float, slice, or array-like, optional
            Longitude selection. Can be a specific value, a slice (e.g.,
            slice(0, 90) for 0° to 90°E), or an array of values.
        time_range : slice or str, optional
            Time range selection. Can be a slice (e.g., slice('2010', '2020'))
            or a string format understood by xarray.
        season : str, optional
            Season to analyze. Options are:
            - 'annual': All months (default)
            - 'jjas': June, July, August, September (Northern Hemisphere summer)
            - 'djf': December, January, February (Northern Hemisphere winter)
            - 'mam': March, April, May (Northern Hemisphere spring)
        stl_seasonal : int, optional
            Length of the seasonal smoother in the STL decomposition (default: 13).
            Higher values produce a smoother seasonal component.
        stl_period : int, optional
            Period of the seasonal component, default is 12 (annual cycle for monthly data).
            Set according to the frequency of your data (e.g., 4 for quarterly, 12 for monthly).
        area_weighted : bool, optional
            Whether to apply area weighting when computing spatial means to account
            for the decreasing grid cell area toward the poles (default: True).
        plot_results : bool, optional
            Whether to plot the decomposition results (default: True).
        figsize : tuple, optional
            Figure size if plotting results, default is (14, 12).
        
        Returns
        -------
        dict
            Dictionary containing the decomposition components:
            - 'original': Original time series
            - 'trend': Trend component
            - 'seasonal': Seasonal component
            - 'residual': Residual component
        matplotlib.figure.Figure, optional
            Figure object if plot_results is True, containing four subplots
            showing the original time series and its decomposed components.
            
        Raises
        ------
        ValueError
            If the dataset is not loaded, the season has no data, or the
            variable is not found in the dataset.
            
        Examples
        --------
        >>> ts = TimeSeries("climate_data.nc")
        >>> # Decompose global mean temperature time series
        >>> results, fig = ts.decompose_time_series(
        ...     variable="air", 
        ...     level=850,
        ...     stl_period=12,  # For monthly data
        ...     plot_results=True
        ... )
        >>> 
        >>> # Access trend component
        >>> trend = results['trend']
        >>> 
        >>> # Decompose regional time series without plotting
        >>> results = ts.decompose_time_series(
        ...     variable="air",
        ...     latitude=slice(-30, 30),  # 30°S to 30°N
        ...     longitude=slice(0, 360),  # Global longitude
        ...     level=500,                # 500 hPa
        ...     time_range=slice('1980', '2020'),
        ...     season="annual",
        ...     area_weighted=True,
        ...     plot_results=False
        ... )
            
        Notes
        -----
        The STL decomposition is particularly useful for climate data as it can
        handle non-linear trends and seasonality that may change over time.
        The 'residual' component often contains climate modes of variability
        and extreme events after the trend and seasonal cycle are removed.
        
        
        """
        if self.dataset is None:
            raise ValueError("No dataset available for analysis. Please load data first.")
       
        data = self._filter_by_season(season)
            
        if 'time' in data.dims and len(data.time) == 0:
            raise ValueError(f"No data available for season '{season}' in the dataset.")
        
        if variable not in list(data.data_vars):
            raise ValueError(f"Variable '{variable}' not found in dataset. Available variables: {list(data.data_vars)}")
        
        if latitude is not None:
            data = data.sel(lat=latitude, method='nearest' if isinstance(latitude, (int, float)) else None)

        if longitude is not None:
            data = data.sel(lon=longitude, method='nearest' if isinstance(longitude, (int, float)) else None)
        
        # Handle level selection
        if 'level' in data.dims:
            if level is None:
                level = data.level.values[0]
                print(f"No level specified, using first available level: {level}")
            
            data = data.sel(level=level)
        elif 'lev' in data.dims:
            if level is None:
                level = data.lev.values[0]
                print(f"No level specified, using first available level: {level}")
            
            data = data.sel(lev=level)
        
        if time_range is not None:
            data = data.sel(time=time_range)
        
        if area_weighted and 'lat' in data.dims:
            weights = np.cos(np.deg2rad(data.lat))
            global_mean = (data[variable] * weights).sum(dim=['lat', 'lon']) / weights.sum()
        else:
            dims_to_avg = [d for d in ['lat', 'lon'] if d in data.dims]
            global_mean = data[variable].mean(dim=dims_to_avg)
        
        if hasattr(global_mean, 'compute'):
            with ProgressBar():
                global_mean = global_mean.compute()
                
        ts_global = global_mean.to_pandas()
        
        stl_result = STL(ts_global, seasonal=stl_seasonal, period=stl_period).fit()
        
        trend = stl_result.trend
        seasonal = stl_result.seasonal
        residual = stl_result.resid
        
        # Storing results in dictionary
        results = {
            'original': ts_global,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }
        
        if plot_results:
            fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
            
            axes[0].plot(ts_global.index, ts_global.values)
            axes[0].set_ylabel(f'{variable} ({data[variable].attrs.get("units", "")})')
            level_info = f" at {level} hPa" if level is not None else ""
            season_display = season.upper() if season.lower() != 'annual' else 'Annual'
            axes[0].set_title(f'{season_display} Time Series - Global Mean {variable}{level_info}')
            
            # Trend component
            axes[1].plot(ts_global.index, trend)
            axes[1].set_ylabel('Trend')
            axes[1].set_title('Long-term Trend Component')
            
            # Seasonal component
            axes[2].plot(ts_global.index, seasonal)
            axes[2].set_ylabel('Seasonal')
            axes[2].set_title(f'Seasonal Component (Period={stl_period})')
            
            # Residual component
            axes[3].plot(ts_global.index, residual)
            axes[3].set_ylabel('Residual')
            axes[3].set_title('Residual Component (Anomalies)')
            
            for ax in axes:
                ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            return results, fig
        
        return results