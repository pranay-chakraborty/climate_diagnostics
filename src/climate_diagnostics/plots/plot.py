import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from dask.diagnostics import ProgressBar
import scipy.ndimage as ndimage  # For gaussian filter

@xr.register_dataset_accessor("climate_plots")
class PlotsAccessor:
    """
    Geographical visualizations of climate data using contour plots.
    
    This accessor provides methods for visualizing climate data with support for:
    - Seasonal filtering (annual, DJF, MAM, JJA, JJAS, SON)
    - Spatial mean calculations
    - Temporal standard deviation calculations
    - Gaussian smoothing for cleaner visualizations
    - Level selection and averaging
    - Land-only visualization options
    
    Access via the .climate_plots attribute on xarray Datasets.
    """

    def __init__(self, xarray_obj):
        """
        Initialize the climate plots accessor.
        
        Parameters
        ----------
        xarray_obj : xarray.Dataset
            The xarray Dataset containing climate data variables with spatial
            coordinates (latitude/longitude) and optionally time and level dimensions.
        """
        self._obj = xarray_obj


    def _filter_by_season(self, data_subset, season='annual'):
        """
         Filter data by meteorological season.
        
        Parameters
        ----------
        data_subset : xarray.Dataset or xarray.DataArray
            Input data containing a time dimension to filter by season.
        season : str, default 'annual'
            Meteorological season to filter by. Options:
            - 'annual': No filtering, returns all data
            - 'djf': December, January, February (Winter)
            - 'mam': March, April, May (Spring)
            - 'jjas': June, July, August, September (Summer Monsoon)
            - 'son': September, October, November (Autumn)
            
        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Data filtered to include only the specified season.
            
        Raises
        ------
        ValueError
            If time dimension is not found or month information cannot be determined.
        
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
            filtered_data = data_subset.where(month_coord.isin(selected_months), drop=True) # Use where for safety
            if filtered_data.time.size == 0:
                 print(f"Warning: No data found for season '{season.upper()}' within the selected time range.")
            return filtered_data
        else:
            print(f"Warning: Unknown season '{season}'. Returning unfiltered data.")
            return data_subset

    def _apply_gaussian_filter(self, data_array, gaussian_sigma):
        """
        Apply Gaussian smoothing filter to spatial dimensions of data.
        
        Parameters
        ----------
        data_array : xarray.DataArray
            Input data array to smooth. Should have at least 2 spatial dimensions,
            typically latitude and longitude as the last two dimensions.
        gaussian_sigma : float or None
            Standard deviation for Gaussian kernel. If None or <= 0,
            no smoothing is applied.
            
        Returns
        -------
        xarray.DataArray
            Smoothed data array with the same dimensions and coordinates.
        bool
            Flag indicating whether smoothing was successfully applied.
            
        Notes
        -----
        Gaussian filter is applied only to the last two dimensions (assumed to be 
        spatial). All other dimensions are preserved as-is. The function handles
        Dask arrays by computing them before filtering. Mode 'nearest' is used
        for boundary handling to minimize edge artifacts.
        """
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
        """
        Select data subset based on variable name and dimension constraints.
        
        Parameters
        ----------
        variable : str
            Name of the variable to select from the dataset.
        latitude : slice, array-like, or scalar, optional
            Latitude range or points to select.
        longitude : slice, array-like, or scalar, optional
            Longitude range or points to select.
        level : int, float, slice, list, or array-like, optional
            Vertical level(s) to select. If a single value, the nearest level is used.
            If multiple values, they're selected for potential averaging.
        time_range : slice or array-like, optional
            Time range to select.
            
        Returns
        -------
        xarray.DataArray
            Selected data subset.
        str or None
            Name of the level dimension if found, otherwise None.
        str or None
            Level operation type performed:
            - 'range_selected': Multiple levels selected
            - 'single_selected': Single level selected
            - None: No level dimension or selection
            
        Raises
        ------
        ValueError
            If the dataset is None or the variable is not found.
            
        Notes
        -----
        For level selection, this method will try to identify the level dimension as
        either 'level' or 'lev'. Nearest neighbor interpolation is used when selecting 
        a single level value that doesn't exactly match coordinates.
        """
        
        if self._obj is None:
            raise ValueError("No dataset available. Load data first.")
        if variable not in self._obj.data_vars:
            raise ValueError(f"Variable '{variable}' not found.")

        data_var = self._obj[variable]
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
        selected_data = data_var
        for dim, method in method_dict.items():
            if dim in selection_dict:
                selected_data = selected_data.sel({dim: selection_dict[dim]}, method=method)
                del selection_dict[dim]  
        
        
        if selection_dict:
            selected_data = selected_data.sel(selection_dict)
            
        return selected_data, level_dim_name, level_op


    def plot_mean(self, 
                  variable='air', 
                  latitude=None, 
                  longitude=None, 
                  level=None,
                  time_range=None, 
                  season='annual', 
                  gaussian_sigma=None,
                  figsize=(16, 10), 
                  cmap='coolwarm',
                  land_only = False,
                  levels=30): 
        """
        Plot spatial mean of a climate variable with optional filtering and smoothing.
        
        Creates a filled contour plot showing the temporal mean of the selected variable,
        with support for seasonal filtering, level selection, and spatial smoothing.
        
        Parameters
        ----------
        variable : str, default 'air'
            Name of the climate variable to plot from the dataset.
        latitude : slice, array-like, or scalar, optional
            Latitude range or points to select.
        longitude : slice, array-like, or scalar, optional
            Longitude range or points to select.
        level : int, float, slice, list, or array-like, optional
            Vertical level(s) to select. If a single value, the nearest level is used.
            If multiple values, they're averaged.
        time_range : slice or array-like, optional
            Time range to select for temporal averaging.
        season : str, default 'annual'
            Season to filter by: 'annual', 'djf', 'mam', 'jja', 'jjas', or 'son'.
        gaussian_sigma : float or None, default None
            Standard deviation for Gaussian smoothing. If None or <= 0, no smoothing.
        figsize : tuple, default (16, 10)
            Figure size (width, height) in inches.
        cmap : str or matplotlib colormap, default 'coolwarm'
            Colormap for the contour plot.
        land_only : bool, default False
            If True, mask out ocean areas to show land-only data.
        levels : int or array-like, default 30
            Number of contour levels or explicit level boundaries for contourf.
            
        Returns
        -------
        matplotlib.axes.Axes
            The plot axes object for further customization.
            
        Raises
        ------
        ValueError
            If no data remains after selections and filtering, or if the dataset is None.
            
        Notes
        -----
        This method supports Dask arrays through progress bar integration. The plot
        includes automatic title generation with time period, level, and smoothing details.
        """
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
        
        # Add geographic features
        if land_only:
            # Add borders first with higher zorder
            ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1, zorder=3)
            # Add coastlines with higher zorder
            ax.coastlines(zorder=3)
        else:
            # Standard display without ocean masking
            ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1)
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
                         
        # Add ocean mask if land_only is True
        if land_only:
            # Mask out oceans with white color
            ax.add_feature(cfeature.OCEAN, zorder=2, facecolor='white')

        # Add Colorbar
        unit_label = smoothed_data.attrs.get('units', '')
        cbar_label = f"{smoothed_data.attrs.get('long_name', variable)} ({unit_label})"
        plt.colorbar(im, label=cbar_label, orientation='vertical', pad=0.05, shrink=0.8)

        # Add Title
        season_map = {
        'annual': "Annual",
        'djf': "Winter (DJF)",
        'mam': "Spring (MAM)",
        'jja': "Summer (JJA)", 
        'jjas': "Summer Monsoon (JJAS)",
        'son': "Autumn (SON)"
        }
        season_str = season_map.get(season.lower(), season.upper())

        # Format variable name nicely
        var_name = variable.replace('_', ' ').capitalize()

        # Base title
        title = f"{season_str} Mean of {var_name}"

        # Add level information
        if level_op == 'single_selected' and level_dim_name:
            # Get the actual selected level value after selection with 'nearest' method
            actual_level = smoothed_data[level_dim_name].values.item()
            level_unit = smoothed_data[level_dim_name].attrs.get('units', '')
            title += f"\nLevel={actual_level} {level_unit}"
        elif level_op == 'range_selected':
            title += " (Level Mean)"

        
        try:
            # Extract directly from the data
            start_time = data_season['time'].min().dt.strftime('%Y').item()
            end_time = data_season['time'].max().dt.strftime('%Y').item()
            time_str = f"\n({start_time}-{end_time})"
            title += time_str
        except Exception as e:
            # Fallback to parameter-based approach if possible
            if time_range is not None and hasattr(time_range, 'start') and hasattr(time_range, 'stop'):
                try:
                    time_str = f"\n({time_range.start.strftime('%Y')}-{time_range.stop.strftime('%Y')})"
                    title += time_str
                except (AttributeError, TypeError):
                    pass  # Skip if this approach fails too

        # Add smoothing info
        if was_smoothed:
            title += f"\nGaussian Smoothed (σ={gaussian_sigma})"

        ax.set_title(title, fontsize=12)

        # Set extent from data - adjust if needed
        try:
            ax.set_extent([lon_coords.min(), lon_coords.max(), lat_coords.min(), lat_coords.max()], crs=ccrs.PlateCarree())
        except ValueError as e:
             print(f"Warning: Could not automatically set extent: {e}") # e.g., if coords are non-monotonic after sel

        return ax


    def plot_std_time(self, 
                      variable='air', 
                      latitude=None, 
                      longitude=None, 
                      level=None,
                      time_range=None, 
                      season='annual', 
                      gaussian_sigma=None,
                      figsize=(16,10), 
                      cmap='viridis', 
                      land_only = False,
                      levels=30): # Added levels arg
        """
        Plot temporal standard deviation of a climate variable.
        
        Creates a filled contour plot showing the standard deviation over time for the
        selected variable, with support for seasonal filtering, level selection, and
        spatial smoothing.
        
        Parameters
        ----------
        variable : str, default 'air'
            Name of the climate variable to plot from the dataset.
        latitude : slice, array-like, or scalar, optional
            Latitude range or points to select.
        longitude : slice, array-like, or scalar, optional
            Longitude range or points to select.
        level : int, float, slice, list, or array-like, optional
            Vertical level(s) to select. If a single value, the nearest level is used.
            If multiple values, standard deviations are averaged across levels.
        time_range : slice or array-like, optional
            Time range to select for calculating temporal standard deviation.
        season : str, default 'annual'
            Season to filter by: 'annual', 'djf', 'mam', 'jja', 'jjas', or 'son'.
        gaussian_sigma : float or None, default None
            Standard deviation for Gaussian smoothing. If None or <= 0, no smoothing.
        figsize : tuple, default (16, 10)
            Figure size (width, height) in inches.
        cmap : str or matplotlib colormap, default 'viridis'
            Colormap for the contour plot.
        land_only : bool, default False
            If True, mask out ocean areas to show land-only data.
        levels : int or array-like, default 30
            Number of contour levels or explicit level boundaries for contourf.
            
        Returns
        -------
        matplotlib.axes.Axes
            The plot axes object for further customization.
            
        Raises
        ------
        ValueError
            If no data remains after selections and filtering, if the dataset is None,
            or if fewer than 2 time points are available for standard deviation calculation.
            
        Notes
        -----
        This method shows the geographic pattern of temporal variability, highlighting
        regions with high or low variability over the selected time period. The plot
        includes automatic title generation with time period, level, and smoothing details.
        """
        selected_data, level_dim_name, level_op = self._select_data(
            variable, latitude, longitude, level, time_range
        )

       # Filter by season
        if 'time' not in selected_data.dims:
             raise ValueError("Standard deviation requires 'time' dimension.")
        data_season = self._filter_by_season(selected_data, season)

        if data_season.size == 0:
            raise ValueError(f"No data after selections and season filter ('{season}').")
        if data_season.sizes['time'] < 2:
             raise ValueError(f"Std dev requires > 1 time point (found {data_season.sizes['time']}).")


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
               # --- Plotting with contourf ---
        plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add geographic features
        if land_only:
            # Add borders first with higher zorder
            ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1, zorder=3)
            # Add coastlines with higher zorder
            ax.coastlines(zorder=3)
        else:
            # Standard display without ocean masking
            ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1)
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
                         
        # Add ocean mask if land_only is True
        if land_only:
            # Mask out oceans with white color
            ax.add_feature(cfeature.OCEAN, zorder=2, facecolor='white')

        # Colorbar
        unit_label = smoothed_data.attrs.get('units', '') # Std dev has same units
        cbar_label = f"Std. Dev. ({unit_label})"
        plt.colorbar(im, label=cbar_label, orientation='vertical', pad=0.05, shrink=0.8)

        # Title
        # Create more descriptive title
        # Format season name
        season_map = {
            'annual': "Annual",
            'djf': "Winter (DJF)",
            'mam': "Spring (MAM)",
            'jja': "Summer (JJA)", 
            'jjas': "Summer Monsoon (JJAS)",
            'son': "Autumn (SON)"
        }
        season_str = season_map.get(season.lower(), season.upper())

        # Format variable name nicely
        var_name = variable.replace('_', ' ').capitalize()

        # Base title
        title = f"{season_str} Standard Deviation of {var_name}"

        # Add level information
        # Add level information
        if level_op == 'single_selected' and level_dim_name:
            # Get the actual selected level value after selection with 'nearest' method
            actual_level = smoothed_data[level_dim_name].values.item()
            level_unit = smoothed_data[level_dim_name].attrs.get('units', '')
            title += f"\nLevel={actual_level} {level_unit}"
        elif level_op == 'range_selected':
            title += " (Level Mean)"

        # Add time range if available
        # Add time range information
        try:
            # Extract directly from the data
            start_time = data_season['time'].min().dt.strftime('%Y').item()
            end_time = data_season['time'].max().dt.strftime('%Y').item()
            time_str = f"\n({start_time}-{end_time})"
            title += time_str
        except Exception as e:
            # Fallback to parameter-based approach if possible
            if time_range is not None and hasattr(time_range, 'start') and hasattr(time_range, 'stop'):
                try:
                    time_str = f"\n({time_range.start.strftime('%Y')}-{time_range.stop.strftime('%Y')})"
                    title += time_str
                except (AttributeError, TypeError):
                    pass  # Skip if this approach fails too

        # Add smoothing info
        if was_smoothed:
            title += f"\nGaussian Smoothed (σ={gaussian_sigma})"

        ax.set_title(title, fontsize=12)

        try:
            ax.set_extent([lon_coords.min(), lon_coords.max(), lat_coords.min(), lat_coords.max()], crs=ccrs.PlateCarree())
        except ValueError as e:
             print(f"Warning: Could not automatically set extent: {e}")

        return ax
    
__all__ = ['PlotsAccessor']