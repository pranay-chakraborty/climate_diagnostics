import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from dask.diagnostics import ProgressBar
import scipy.ndimage as ndimage  # For gaussian filter

class Plots:
    """
    Geographical visualizations of climate data using contour plots.
    Supports seasonal filtering, mean/std dev calculation, and Gaussian smoothing.
    """

    def __init__(self, filepath=None):
        """Initialize and load data."""
        self.filepath = filepath
        self.dataset = None
        self._load_data()

    def _load_data(self):
        """Load dataset using xarray with dask chunks."""
        try:
            if self.filepath:
                # Attempt to load suppressing some warnings, use netcdf4 engine
                with xr.set_options(keep_attrs=True):
                     self.dataset = xr.open_dataset(self.filepath, chunks='auto', engine='netcdf4')
                print(f"Dataset loaded from {self.filepath}")
                # Attempt common renames
                try:
                    self.dataset = self.dataset.rename({'latitude': 'lat', 'longitude': 'lon'})
                except ValueError:
                    pass 
                if 'lat' not in self.dataset.coords or 'lon' not in self.dataset.coords:
                    print("Warning: Standard 'lat'/'lon' coordinates not found.")
            else:
                print("No filepath provided during initialization.")
        except FileNotFoundError:
             print(f"Error: File not found at {self.filepath}")
             self.dataset = None
        except Exception as e:
            print(f"Error loading data from {self.filepath}: {e}")
            self.dataset = None

    def _filter_by_season(self, data_subset, season='annual'):
        """Filter data by meteorological season."""
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
            filtered_data = data_subset.where(month_coord.isin(selected_months), drop=True) # Use where for safety
            if filtered_data.time.size == 0:
                 print(f"Warning: No data found for season '{season.upper()}' within the selected time range.")
            return filtered_data
        else:
            print(f"Warning: Unknown season '{season}'. Returning unfiltered data.")
            return data_subset

    def _apply_gaussian_filter(self, data_array, gaussian_sigma):
        """Apply Gaussian filter to spatial dimensions."""
        if gaussian_sigma is None or gaussian_sigma <= 0:
            return data_array, False # No filtering

        if data_array.ndim < 2:
            print("Warning: Gaussian filtering skipped (data < 2D).")
            return data_array, False

        # Assume spatial dimensions are the last two
        sigma_array = [0] * (data_array.ndim - 2) + [gaussian_sigma] * 2

        try:
            # Ensure data is computed if dask array
            computed_data = data_array.compute() if hasattr(data_array, 'compute') else data_array
            smoothed_values = ndimage.gaussian_filter(computed_data.values, sigma=sigma_array, mode='nearest') # Use mode='nearest' for boundaries

            smoothed_da = xr.DataArray(
                smoothed_values, coords=computed_data.coords, dims=computed_data.dims,
                name=computed_data.name, attrs=computed_data.attrs
            )
            smoothed_da.attrs['filter'] = f'Gaussian smoothed (sigma={gaussian_sigma})'
            return smoothed_da, True
        except Exception as e:
            print(f"Warning: Could not apply Gaussian filter: {e}")
            return data_array, False

    def _select_data(self, variable, latitude=None, longitude=None, level=None, time_range=None):
        """Selects data based on provided dimensions."""
        if self.dataset is None:
            raise ValueError("No dataset available. Load data first.")
        if variable not in self.dataset.data_vars:
            raise ValueError(f"Variable '{variable}' not found.")

        data_var = self.dataset[variable]
        selection_dict = {}
        method_dict = {} # For 'nearest'

        if latitude is not None: selection_dict['lat'] = latitude
        if longitude is not None: selection_dict['lon'] = longitude
        if time_range is not None and 'time' in data_var.dims: selection_dict['time'] = time_range

        level_dim_name = next((dim for dim in ['level', 'lev'] if dim in data_var.dims), None)
        level_op = None # To track if level mean/selection occurred

        if level_dim_name:
            if level is not None:
                if isinstance(level, (slice, list, np.ndarray)):
                    # Select range/list of levels for potential later averaging
                    selection_dict[level_dim_name] = level
                    level_op = 'range_selected'
                elif isinstance(level, (int, float)):
                    # Select single level with nearest neighbor
                    selection_dict[level_dim_name] = level
                    method_dict[level_dim_name] = 'nearest'
                    level_op = 'single_selected'
                else: # e.g. single value already in coords
                    selection_dict[level_dim_name] = level
                    level_op = 'single_selected'
            elif len(data_var[level_dim_name]) > 1:
                # Default to first level if multiple exist and none specified
                level_val = data_var[level_dim_name].values[0]
                selection_dict[level_dim_name] = level_val
                level_op = 'single_selected'
                print(f"Warning: Multiple levels found. Using first level: {level_val}")
        elif level is not None:
            print("Warning: Level dimension not found. Ignoring 'level' parameter.")

        # Perform selection
        selected_data = data_var.sel(**selection_dict, method=method_dict if method_dict else None)

        return selected_data, level_dim_name, level_op


    def plot_mean(self, variable='air', latitude=None, longitude=None, level=None,
                  time_range=None, season='annual', gaussian_sigma=None,
                  figsize=(16, 10), cmap='coolwarm', levels=15): # Added levels arg
        """Plot spatial mean using contourf, optionally smoothed."""
        selected_data, level_dim_name, level_op = self._select_data(
            variable, latitude, longitude, level, time_range
        )

        # Average over levels if a range was selected
        if level_op == 'range_selected' and level_dim_name in selected_data.dims:
             selected_data = selected_data.mean(dim=level_dim_name)
             print(f"Averaging over selected levels.")

        # Filter by season
        data_season = self._filter_by_season(selected_data, season)
        if data_season.size == 0:
            raise ValueError(f"No data after selections and season filter ('{season}').")

        # Compute time mean
        if 'time' in data_season.dims:
            if data_season.chunks:
                print("Computing time mean...")
                with ProgressBar(): mean_data = data_season.mean(dim='time').compute()
            else:
                 mean_data = data_season.mean(dim='time')
        else:
            mean_data = data_season # No time dim to average
            print("Warning: No time dimension found for averaging.")

        # Apply smoothing
        smoothed_data, was_smoothed = self._apply_gaussian_filter(mean_data, gaussian_sigma)

        # --- Plotting with contourf because it gives better results---
        plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl.top_labels = False; gl.right_labels = False # Tidy labels

        # Prepare for contourf
        lon_coords = smoothed_data['lon'].values
        lat_coords = smoothed_data['lat'].values
        plot_data_np = smoothed_data.values

        # Handle potential all-nan slices after operations
        if np.all(np.isnan(plot_data_np)):
             print("Warning: Data is all NaN after calculations. Cannot plot contours.")
             ax.set_title(f'{variable.capitalize()} - Data is all NaN')
             return ax

        im = ax.contourf(lon_coords, lat_coords, plot_data_np,
                         levels=levels, cmap=cmap, # Use specified levels
                         transform=ccrs.PlateCarree(), extend='both')

        # Add Colorbar
        unit_label = smoothed_data.attrs.get('units', '')
        cbar_label = f"{smoothed_data.attrs.get('long_name', variable)} ({unit_label})"
        plt.colorbar(im, label=cbar_label, orientation='vertical', pad=0.05, shrink=0.8)

        # Add Title
        title = f'{season.upper() if season.lower() != "annual" else "Annual"} Mean {variable.capitalize()}'
        if level_op == 'single_selected' and level is not None:
            level_unit = smoothed_data.coords[level_dim_name].attrs.get('units', '') if level_dim_name else ''
            title += f' at {level} {level_unit}'
        elif level_op == 'range_selected':
             title += ' (Level Mean)'
        if was_smoothed:
            title += f' (Smoothed σ={gaussian_sigma})'
        ax.set_title(title, fontsize=12) # Slightly smaller title

        # Set extent from data - adjust if needed
        try:
            ax.set_extent([lon_coords.min(), lon_coords.max(), lat_coords.min(), lat_coords.max()], crs=ccrs.PlateCarree())
        except ValueError as e:
             print(f"Warning: Could not automatically set extent: {e}") # e.g., if coords are non-monotonic after sel

        return ax


    def plot_std_time(self, variable='air', latitude=None, longitude=None, level=None,
                      time_range=None, season='annual', gaussian_sigma=None,
                      figsize=(16,10), cmap='viridis', levels=15): # Added levels arg
        """Plot temporal standard deviation using contourf, optionally smoothed."""
        selected_data, level_dim_name, level_op = self._select_data(
            variable, latitude, longitude, level, time_range
        )

        # Filter by season
        if 'time' not in selected_data.dims:
             raise ValueError("Standard deviation requires 'time' dimension.")
        data_season = self._filter_by_season(selected_data, season)

        if data_season.size == 0:
            raise ValueError(f"No data after selections and season filter ('{season}').")
        if data_season.dims['time'] < 2:
             raise ValueError(f"Std dev requires > 1 time point (found {data_season.dims['time']}).")

        # Compute standard deviation over time
        if data_season.chunks:
            print("Computing standard deviation over time...")
            with ProgressBar(): std_data = data_season.std(dim='time').compute()
        else:
            std_data = data_season.std(dim='time')

        # Average std dev map across levels if a range was selected originally
        if level_op == 'range_selected' and level_dim_name in std_data.dims:
             std_data = std_data.mean(dim=level_dim_name)
             print(f"Averaging standard deviation map across selected levels.")

        # Apply smoothing
        smoothed_data, was_smoothed = self._apply_gaussian_filter(std_data, gaussian_sigma)

        # --- Plotting with contourf ---
        plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl.top_labels = False; gl.right_labels = False

        lon_coords = smoothed_data['lon'].values
        lat_coords = smoothed_data['lat'].values
        plot_data_np = smoothed_data.values

        if np.all(np.isnan(plot_data_np)):
             print("Warning: Data is all NaN after calculations. Cannot plot contours.")
             ax.set_title(f'{variable.capitalize()} Std Dev - Data is all NaN')
             return ax

        im = ax.contourf(lon_coords, lat_coords, plot_data_np,
                         levels=levels, cmap=cmap, # Use specified levels
                         transform=ccrs.PlateCarree(), extend='both')

        # Colorbar
        unit_label = smoothed_data.attrs.get('units', '') # Std dev has same units
        cbar_label = f"Std. Dev. ({unit_label})"
        plt.colorbar(im, label=cbar_label, orientation='vertical', pad=0.05, shrink=0.8)

        # Title
        title = f'{season.upper() if season.lower() != "annual" else "Annual"} Std. Dev. of {variable.capitalize()}'
        if level_op == 'single_selected' and level is not None:
            level_unit = smoothed_data.coords[level_dim_name].attrs.get('units', '') if level_dim_name else ''
            title += f' at {level} {level_unit}'
        elif level_op == 'range_selected':
            title += ' (Level Mean)'
        if was_smoothed:
            title += f' (Smoothed σ={gaussian_sigma})'
        ax.set_title(title, fontsize=12)

        try:
            ax.set_extent([lon_coords.min(), lon_coords.max(), lat_coords.min(), lat_coords.max()], crs=ccrs.PlateCarree())
        except ValueError as e:
             print(f"Warning: Could not automatically set extent: {e}")

        return ax