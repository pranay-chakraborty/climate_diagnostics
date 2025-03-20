import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from dask.diagnostics import ProgressBar
from statsmodels.tsa.seasonal import STL

class TimeSeries:
    def __init__(self, filepath=None):
        """
        Initialize the TimeSeries class.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to the netCDF or other compatible data file.
        """
        self.filepath = filepath
        self.dataset = None
        self._load_data()
        
    def _load_data(self):
        """Load dataset from the provided filepath with automatic chunking."""
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
        
        Parameters:
        -----------
        season : str, optional
            Season to filter by. Options are:
            - 'annual': All months (default)
            - 'jjas': June, July, August, September
            - 'djf': December, January, February
            - 'mam': March, April, May
            
        Returns:
        --------
        xarray.Dataset
            Filtered dataset
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
                    season='annual'):
        """
        Plot the time series for the selected variable (averaged across latitude and longitude).
        
        Parameters:
        -----------
        latitude : float, slice, or array-like, optional
            Latitude selection
        longitude : float, slice, or array-like, optional
            Longitude selection
        level : float or int, optional
            Pressure level selection. If None, first level is used if available.
        time_range : slice or str, optional
            Time range selection
        variable : str, optional
            Variable name to plot (default: 'air')
        figsize : tuple, optional
            Figure size (width, height) in inches
        season : str, optional
            Season to plot. Options are:
            - 'annual': All months (default)
            - 'jjas': June, July, August, September
            - 'djf': December, January, February
            - 'mam': March, April, May
            
        Returns:
        --------
        matplotlib.axes.Axes
            The axes object containing the plot
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
        
        spatial_dims = []
        if 'lat' in data.dims:
            spatial_dims.append('lat')
        if 'lon' in data.dims:
            spatial_dims.append('lon')
        
        if spatial_dims:
            data = data.mean(dim=spatial_dims)
            
        if hasattr(data[variable], 'compute'):
            with ProgressBar():
                data = data.compute()

        plt.figure(figsize=figsize)
        ax = data[variable].plot()
        
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
        Plot the standard deviation over time for the selected variable (averaged across latitude and longitude). Normalised.
        
        Parameters:
        -----------
        latitude : float, slice, or array-like, optional
            Latitude selection
        longitude : float, slice, or array-like, optional
            Longitude selection
        level : float or int, optional
            Pressure level selection. If None, first level is used if available.
        time_range : slice or str, optional
            Time range selection
        variable : str, optional
            Variable name to plot (default: 'air')
        figsize : tuple, optional
            Figure size (width, height) in inches
        season : str, optional
            Season to plot. Options are:
            - 'annual': All months (default)
            - 'jjas': June, July, August, September
            - 'djf': December, January, February
            - 'mam': March, April, May
            
        Returns:
        --------
        matplotlib.axes.Axes
            The axes object containing the plot
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
        area_weighted=False,
        plot_results=True,
        figsize=(14, 12)
    ):
        """
        Decompose a climate time series into trend, seasonal, and residual components using STL.
        
        Parameters:
        -----------
        variable : str, optional
            Variable name to analyze, default is 'air'
        level : float or int, optional
            Pressure level to analyze. If None, uses first available level
        latitude : float, slice, or array-like, optional
            Latitude selection
        longitude : float, slice, or array-like, optional
            Longitude selection
        time_range : slice or str, optional
            Time range selection
        season : str, optional
            Season to analyze. Options are:
            - 'annual': All months (default)
            - 'jjas': June, July, August, September
            - 'djf': December, January, February
            - 'mam': March, April, May
        stl_seasonal : int, optional
            Length of the seasonal smoother, default is 13
        stl_period : int, optional
            Period of the seasonal component, default is 12 (annual for monthly data)
        area_weighted : bool, optional
            Whether to apply area weighting when computing spatial means, default is True
        plot_results : bool, optional
            Whether to plot the decomposition results, default is True
        figsize : tuple, optional
            Figure size if plotting results, default is (14, 12)
        
        Returns:
        --------
        dict
            Dictionary containing the decomposition components:
            - 'original': Original time series
            - 'trend': Trend component
            - 'seasonal': Seasonal component
            - 'residual': Residual component
        matplotlib.figure.Figure, optional
            Figure object if plot_results is True
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
        