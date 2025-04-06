import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from dask.diagnostics import ProgressBar

class Plots:
    """
    A class for creating geographical visualizations of climate data.
    
    This class provides methods to load, filter, and visualize climate data
    from NetCDF files. It supports spatial visualization with proper map
    projections, filtering by meteorological seasons, and computation of
    statistical properties like means and standard deviations.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the netCDF or other compatible data file.
        
    Attributes
    ----------
    filepath : str
        Path to the input data file.
    dataset : xarray.Dataset
        Loaded dataset with climate variables.
        
    Examples
    --------
    >>> from climate_diagnostics import Plots
    >>> plots = Plots("/path/to/climate_data.nc")
    >>> plots.plot_mean(variable="air", level=850, season="djf")
    """
    
    def __init__(self, filepath=None):
        """
        Initialize the Plots class.
        
        Parameters
        ----------
        filepath : str, optional
            Path to the netCDF or other compatible data file.
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

    def plot_mean(self, latitude=None, longitude=None, level=None,
                  time_range=None, variable='air', figsize=(20, 10),
                  season='annual'):
        """
        Plot the mean of the selected variable across specified dimensions.
        
        Creates a map visualization showing the spatial pattern of the mean
        value for the selected variable, filtered by the given parameters.
        
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
            The axes object containing the plot, which can be further customized
            
        Raises
        ------
        ValueError
            If the dataset is not loaded, the season has no data, or the 
            variable is not found in the dataset
            
        Examples
        --------
        >>> plots = Plots("climate_data.nc")
        >>> # Plot mean temperature at 850 hPa for winter season
        >>> plots.plot_mean(
        ...     latitude=slice(-60, 60),  # 60°S to 60°N
        ...     level=850,                # 850 hPa
        ...     variable="air",           # Air temperature
        ...     season="djf"              # December-January-February
        ... )
        
        >>> # Plot mean precipitation for summer in a specific region
        >>> plots.plot_mean(
        ...     latitude=slice(0, 40),    # 0° to 40°N
        ...     longitude=slice(60, 100), # 60°E to 100°E
        ...     variable="precip",        # Precipitation
        ...     season="jjas"             # June-July-August-September
        ... )
        """
        if self.dataset is None:
            raise ValueError("No dataset available for plotting. Please load data first.")

        data = self._filter_by_season(season)
        
        if len(data.time) == 0:
            raise ValueError(f"No data available for season '{season}' in the dataset.")

        if variable not in data:
            raise ValueError(f"Variable '{variable}' not found in dataset. Available variables: {list(data.data_vars)}")

        # Set default level if None and level dimension exists
        if level is None and 'level' in data.dims and len(data.level) > 0:
            level = data.level.values[0]

        if latitude is not None:
            data = data.sel(lat=latitude, method='nearest' if isinstance(latitude, (int, float)) else None)

        if longitude is not None:
            data = data.sel(lon=longitude, method='nearest' if isinstance(longitude, (int, float)) else None)

        # Handle level selection
        if 'level' in data.dims:
            if level is not None:
                if isinstance(level, (slice, list, np.ndarray)):
                    data = data.sel(level=level)
                    data = data.mean(dim='level')
                elif isinstance(level, (int, float)):
                    data = data.sel(level = level , method = 'nearest')
                else:
                    data = data.sel(level=level)
        elif 'lev' in data.dims:  # Handle alternative naming
            if level is not None:
                if isinstance(level, (slice, list, np.ndarray)):
                    data = data.sel(lev=level)
                    data = data.mean(dim='lev')
                elif isinstance(level, (int, float)):
                    data = data.sel(lev = level , method = 'nearest')
                else:
                    data = data.sel(lev=level)
        else:
            print("Warning: Level dimension not found in dataset.")

        if time_range is not None and 'time' in data.dims:
            data = data.sel(time=time_range)

        if 'time' in data.dims:
            data = data.mean(dim='time')
            if hasattr(data[variable], 'compute'):
                with ProgressBar():
                    data = data.compute()

        plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(draw_labels=True)

        im = data[variable].plot(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm')
        unit_label = data[variable].attrs.get('units', '')
        
        season_display = season.upper() if season.lower() != 'annual' else 'Annual'
        plt.title(f'{season_display} Mean {variable} data')

        return ax

    def plot_std_time(self, latitude=None, longitude=None, level=None,
                      time_range=None, variable='air', figsize=(20, 10),
                      season='annual'):
        """
        Plot the standard deviation over time for the selected variable.
        
        Creates a map visualization showing the spatial pattern of temporal
        variability (standard deviation) for the selected variable.
        
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
            The axes object containing the plot, which can be further customized
            
        Raises
        ------
        ValueError
            If the dataset is not loaded, the season has no data, the 
            variable is not found, or there's no time dimension
            
        Examples
        --------
        >>> plots = Plots("climate_data.nc")
        >>> # Plot standard deviation of temperature at 850 hPa for winter
        >>> plots.plot_std_time(
        ...     latitude=slice(-60, 60),  # 60°S to 60°N
        ...     level=850,                # 850 hPa
        ...     variable="air",           # Air temperature
        ...     season="djf"              # December-January-February
        ... )
        
        >>> # Plot interannual variability of precipitation
        >>> plots.plot_std_time(
        ...     variable="precip",        # Precipitation
        ...     time_range=slice("1980", "2020"),  # Study period
        ...     season="jjas"             # June-July-August-September
        ... )
            
        Notes
        -----
        This function calculates the standard deviation across the time
        dimension for each grid point, providing insights into regions
        with high temporal variability.
        """
        if self.dataset is None:
            raise ValueError("No dataset available for plotting. Please load data first.")
        
        data = self._filter_by_season(season)
        
        if len(data.time) == 0:
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
            
        data = data.std(dim='time')
        if hasattr(data[variable], 'compute'):
            with ProgressBar():
                data = data.compute()

        plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(draw_labels=True)

        im = data[variable].plot(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm')
        unit_label = data[variable].attrs.get('units', '')
        
        season_display = season.upper() if season.lower() != 'annual' else 'Annual'
        plt.title(f'{season_display} Standard Deviation of {variable} data')
        
        return ax