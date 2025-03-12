import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np


class Plots:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.dataset = None
        self._load_data()

    def _load_data(self):
        try:
            if self.filepath:
                self.dataset = xr.open_dataset(self.filepath, chunks='auto')
                print(f"Dataset loaded from {self.filepath} with auto-chunking")
            else:
                print("Invalid filepath provided. Please specify a valid filepath.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def plot_mean(self, latitude=None, longitude=None, level=None,
                       time_range=None, variable='air', cmap='coolwarm'):
        if self.dataset is None:
            raise ValueError("No dataset available for plotting. Please load data first.")

        data = self.dataset

        if variable not in data:
            raise ValueError(f"Variable '{variable}' not found in dataset. Available variables: {list(data.data_vars)}")

        if latitude is not None:
            data = data.sel(lat=latitude, method='nearest' if isinstance(latitude, (int, float)) else None)

        if longitude is not None:
            data = data.sel(lon=longitude, method='nearest' if isinstance(longitude, (int, float)) else None)

        if level is not None:
            data = data.sel(level=level)

        if time_range is not None:
            data = data.sel(time=time_range)

        if 'time' in data.dims:
            data = data.mean(dim='time')
            if hasattr(data, 'compute'):
                data = data.compute()

        if 'level' in data.dims:
            data = data.mean(dim='level')
            if hasattr(data, 'compute'):
                data = data.compute()

        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(draw_labels=True)

        im = data[variable].plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap)
        unit_label = data[variable].attrs.get('units', '')
        plt.colorbar(im, ax=ax, shrink=0.7, label=f'{variable} ({unit_label})')
        plt.title(f'Mean {variable} data')

        return ax
