#------------- importing necessary libraries ----------------#

from plots.urls import uwnd_mon_mean, vwnd_mon_mean, air_mon_mean
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


class plots:
    def __init__(self, uwnd_url=None, vwnd_url=None, air_temp_url=None):
        # Use parameters if provided, otherwise use imported constants
        self.uwnd_url = uwnd_url if uwnd_url is not None else uwnd_mon_mean
        self.vwnd_url = vwnd_url if vwnd_url is not None else vwnd_mon_mean
        self.air_temp_url = air_temp_url if air_temp_url is not None else air_mon_mean
        self.uwnd_data = None
        self.vwnd_data = None
        self.air_temp_data = None

    def load_air_temperature(self):
        """Load data from the URLs"""
        try:
            self.air_temp_data = xr.open_dataset(self.air_temp_url)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def plot_air_temperature(self, file_path=None, ax=None, **kwargs):
        """
        Plot air temperature from a NetCDF file.

        Args:
            file_path (str): Path to the NetCDF file. If None, uses self.air_temp_data
            ax (matplotlib.axes.Axes): Existing axes to plot on (optional)
            **kwargs: Additional plotting arguments for xarray.DataArray.plot()

        Returns:
            matplotlib.axes.Axes: Axes object with the plot
        """
        if file_path:
            ds = xr.open_dataset(file_path)
        elif self.air_temp_data is not None:
            ds = self.air_temp_data
        else:
            raise ValueError("No data source specified. Provide file_path or call load_data() first")

        if 'air' not in ds.variables:
            raise ValueError("Dataset does not contain 'air' variable")

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 10),
                                   subplot_kw={'projection': ccrs.PlateCarree()})

        air_ds = ds['air']
        plot = air_ds.plot(ax=ax,**kwargs)
        ax.coastlines()
        ax.gridlines(draw_labels=True)

        return ax
