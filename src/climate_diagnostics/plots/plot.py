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
    Examples
    --------
    >>> import xarray as xr
    >>> # Open a climate dataset with lat, lon, time dimensions
    >>> ds = xr.open_dataset('climate_data.nc')
    >>> 
    >>> # Create a basic plot using the accessor
    >>> ds.climate_plots.plot_mean(variable='air')
    >>> 
    >>> # Filter for winter season and specific region
    >>> ds.climate_plots.plot_mean(
    ...     variable='air', 
    ...     season='djf',
    ...     latitude=slice(40, 6),
    ...     longitude=slice(65,110)
    ... )
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
            If the dataset is None, the variable is not found, or if 
            requested coordinates are outside available range.
            
        Notes
        -----
        For level selection, this method will try to identify the level dimension as
        either 'level' or 'lev'. Nearest neighbor interpolation is used when selecting 
        a single level value that doesn't exactly match coordinates.
        """
        
        
        if variable not in self._obj.data_vars:
            raise ValueError(f"Variable '{variable}' not found.")

        data_var = self._obj[variable]
        selection_dict = {}
        method_dict = {} # For 'nearest'

        # Validate latitude coordinates
        if latitude is not None:
            if 'lat' not in data_var.coords:
                raise ValueError("Latitude coordinate 'lat' not found in dataset.")
            
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

        # Validate longitude coordinates
        if longitude is not None:
            if 'lon' not in data_var.coords:
                raise ValueError("Longitude coordinate 'lon' not found in dataset.")
            
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

        # Validate time range
        if time_range is not None and 'time' in data_var.dims:
            if 'time' not in data_var.coords:
                raise ValueError("Time coordinate 'time' not found in dataset.")
            
            # For time, we need to be careful with datetime comparisons
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
        level_op = None # To track if level mean/selection occurred

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
                    
                    selection_dict[level_dim_name] = level
                    level_op = 'range_selected'
                elif isinstance(level, (int, float)):
                    # For single values with 'nearest' method, warn if far outside range
                    if level < level_min * 0.5 or level > level_max * 1.5:
                        print(f"Warning: Requested level {level} is far from available range [{level_min}, {level_max}]")
                    
                    selection_dict[level_dim_name] = level
                    method_dict[level_dim_name] = 'nearest'
                    level_op = 'single_selected'
                else:
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
                  levels=30,
                  save_plot_path = None
                  ): 
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
        save_plot_path : str, optional
             Path where the plot should be saved. If None (default), the plot is not saved.
            
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
        
        Examples
        --------
        >>> # Basic temperature plot
        >>> ds.climate_plots.plot_mean(variable='temperature')
        >>> 
        >>> # Plot with seasonal filtering and region selection
        >>> ds.climate_plots.plot_mean(
        ...     variable='air',
        ...     season='jjas',  # summer monsoon
        ...     latitude=slice(40, 6),
        ...     longitude=slice(65, 100),
        ...     levels=150,
        ...     cmap='coolwarm'
        ... )
        >>> 
        >>> # Plot at specific pressure level with smoothing and saving
        >>> ds.climate_plots.plot_mean(
        ...     variable='air',
        ...     level=500,  # 500 hPa
        ...     gaussian_sigma=1,
        ...     land_only=True,
        ...     save_plot_path='/home/user/Downloads/gph_500hPa.png'
        ... )
        >>> 
        >>> # Plot with time range selection
        >>> ds.climate_plots.plot_mean(
        ...     variable='air',
        ...     time_range=slice('2000-01-01','2010-12-31')
        ... )
        
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

        if save_plot_path is not None:
            plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved to: {save_plot_path}")
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
                      levels=30,
                      save_plot_path = None): # Added levels arg
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
        save_plot_path : str, optional
            Path where the plot should be saved. If None (default), the plot is not saved.
            
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
        
        Examples
        --------
        >>> # Basic standard deviation plot
        >>> ds.climate_plots.plot_std_time(variable='temperature')
        >>> 
        >>> # Plot standard deviation of monsoon rainfall
        >>> ds.climate_plots.plot_std_time(
        ...     variable='prate',
        ...     season='jjas',
        ...     latitude=slice(40, 6),
        ...     longitude=slice(60, 100),
        ...     cmap='YlOrRd',
        ...     levels=150
        ... )
        >>> 
        >>> # Plot interannual variability with smoothing
        >>> ds.climate_plots.plot_std_time(
        ...     variable='sea_surface_temperature',
        ...     gaussian_sigma=1.0,
        ...     time_range=slice('1950-01-01', '2020-12-28')
        ... )
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

        # Save plot if path is provided
        if save_plot_path is not None:
            plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved to: {save_plot_path}")
        return ax
    
__all__ = ['PlotsAccessor']