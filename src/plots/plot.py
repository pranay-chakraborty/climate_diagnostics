import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np


class Plots:
    def __init__(self, filepath=None):
        """
        Initialize the Plots class.
        
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

    def plot_mean(self, latitude=None, longitude=None, level=None,
                  time_range=None, variable='air', figsize=(20, 10)):
        """
        Plot the mean of the selected variable across specified dimensions.
        
        Parameters:
        -----------
        latitude : float, slice, or array-like, optional
            Latitude selection
        longitude : float, slice, or array-like, optional
            Longitude selection
        level : float or int, optional
            Pressure level selection
        time_range : slice or str, optional
            Time range selection
        variable : str, optional
            Variable name to plot (default: 'air')
        figsize : tuple, optional
            Figure size (width, height) in inches
            
        Returns:
        --------
        matplotlib.axes.Axes
            The axes object containing the plot
        """
        if self.dataset is None:
            raise ValueError("No dataset available for plotting. Please load data first.")

        data = self.dataset

        if variable not in data:
            raise ValueError(f"Variable '{variable}' not found in dataset. Available variables: {list(data.data_vars)}")

        if latitude is not None:
            data = data.sel(lat=latitude, method='nearest' if isinstance(latitude, (int, float)) else None)

        if longitude is not None:
            data = data.sel(lon=longitude, method='nearest' if isinstance(longitude, (int, float)) else None)

        if level is not None and 'level' in data.dims:
            data = data.sel(level=level)

        if time_range is not None and 'time' in data.dims:
            data = data.sel(time=time_range)

        if 'time' in data.dims:
            data = data.mean(dim='time')
            if hasattr(data[variable], 'compute'):
                data = data.compute()

        if 'level' in data.dims:
            data = data.mean(dim='level')
            if hasattr(data[variable], 'compute'):
                data = data.compute()

        plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(draw_labels=True)

        im = data[variable].plot(ax=ax, transform=ccrs.PlateCarree())
        unit_label = data[variable].attrs.get('units', '')
        plt.colorbar(im, ax=ax, shrink=0.7, label=f'{variable} ({unit_label})')
        plt.title(f'Mean {variable} data')

        return ax

    def plot_std_time(self, latitude=None, longitude=None, level=None,
                      time_range=None, variable='air', figsize=(20, 10)):
        """
        Plot the standard deviation over time for the selected variable.
        
        Parameters:
        -----------
        latitude : float, slice, or array-like, optional
            Latitude selection
        longitude : float, slice, or array-like, optional
            Longitude selection
        level : float or int, optional
            Pressure level selection
        time_range : slice or str, optional
            Time range selection
        variable : str, optional
            Variable name to plot (default: 'air')
        figsize : tuple, optional
            Figure size (width, height) in inches
            
        Returns:
        --------
        matplotlib.axes.Axes
            The axes object containing the plot
        """
        if self.dataset is None:
            raise ValueError("No dataset available for plotting. Please load data first.")
        
        data = self.dataset
        
        if variable not in list(data.data_vars):
            raise ValueError(f"Variable '{variable}' not found in dataset. Available variables: {list(data.data_vars)}")

        if latitude is not None:
            data = data.sel(lat=latitude, method='nearest' if isinstance(latitude, (int, float)) else None)

        if longitude is not None:
            data = data.sel(lon=longitude, method='nearest' if isinstance(longitude, (int, float)) else None)

        if 'level' in data.dims or 'lev' in data.dims:
            level_dim = 'level' if 'level' in data.dims else 'lev'
            if level is not None:
                data = data.sel({level_dim: level})
        else:
            print("Warning: Level dimension not found in dataset.")

        if 'time' in data.dims:
            if time_range is not None:
                data = data.sel(time=time_range)
            data = data.std(dim='time')
            if hasattr(data[variable], 'compute'):
                data = data.compute()
        else:
            raise ValueError("Time dimension not found in dataset. Please load data with time dimension.")

        plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(draw_labels=True)

        im = data[variable].plot(ax=ax, transform=ccrs.PlateCarree())
        unit_label = data[variable].attrs.get('units', '')
        plt.colorbar(im, ax=ax, shrink=0.7, label=f'{variable} ({unit_label})')
        plt.title(f'Standard Deviation of {variable} data')
        
        return ax
        